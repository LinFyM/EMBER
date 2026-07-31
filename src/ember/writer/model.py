"""Canonical Language-Axial Core + Causal Procedure PI05 Writer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

import torch

from ember.writer.temporal import (
    CausalProcedureEncoder,
    LanguageSemanticCore,
    SlotNormalizedCoreProcedureCompiler,
)
from ember.writer.video_program import Pi05LanguageAxialEncoder


class WriterModelError(RuntimeError):
    """Raised when the Writer input or sealed LoRA contract is inconsistent."""


LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS = frozenset(
    {
        "image_width",
        "expert_width",
        "program_width",
        "text_meta_lora_rank",
        "vl_meta_lora_rank",
        "action_meta_lora_rank",
        "patch_grounding_heads",
        "max_frames_per_encoder_call",
        "action_horizon",
        "padded_action_dim",
        "semantic_core_heads",
        "semantic_core_blocks",
        "frame_attention_initial_lambda",
        "procedure_heads",
        "procedure_blocks",
        "fusion_heads",
        "factor_hidden_width",
        "initialization_seed",
        "activation_checkpointing",
    }
)


@dataclass(frozen=True)
class LoraTensorSpec:
    """One row-oriented output tensor in a PEFT LoRA state."""

    name: str
    module: str
    module_index: int
    factor_index: int
    rank: int
    width: int
    transpose_output: bool


def build_lora_tensor_specs(
    state: Mapping[str, torch.Tensor],
) -> tuple[LoraTensorSpec, ...]:
    """Build paired A/B output specifications from a real PEFT state."""

    marker_a = ".lora_A.default.weight"
    marker_b = ".lora_B.default.weight"
    modules: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    for name, value in state.items():
        if name.endswith(marker_a):
            marker, factor = marker_a, "A"
        elif name.endswith(marker_b):
            marker, factor = marker_b, "B"
        else:
            raise WriterModelError(f"non-LoRA tensor in template: {name}")
        if value.ndim != 2:
            raise WriterModelError(f"LoRA tensor is not a matrix: {name}")
        module = name[: -len(marker)]
        modules.setdefault(module, {})[factor] = (name, value)

    if not modules or any(set(pair) != {"A", "B"} for pair in modules.values()):
        raise WriterModelError("every target module must contain one LoRA A/B pair")

    result: list[LoraTensorSpec] = []
    for module_index, module in enumerate(sorted(modules)):
        name_a, value_a = modules[module]["A"]
        name_b, value_b = modules[module]["B"]
        rank, input_width = value_a.shape
        output_width, rank_b = value_b.shape
        if rank <= 0 or rank_b != rank:
            raise WriterModelError(f"LoRA rank differs for {module}")
        result.extend(
            (
                LoraTensorSpec(
                    name=name_a,
                    module=module,
                    module_index=module_index,
                    factor_index=0,
                    rank=rank,
                    width=input_width,
                    transpose_output=False,
                ),
                LoraTensorSpec(
                    name=name_b,
                    module=module,
                    module_index=module_index,
                    factor_index=1,
                    rank=rank,
                    width=output_width,
                    transpose_output=True,
                ),
            )
        )
    return tuple(result)


class FactorHead(torch.nn.Module):
    """Decode one video-conditioned rank-slot state into one LoRA row."""

    def __init__(self, input_width: int, hidden_width: int, output_width: int) -> None:
        super().__init__()
        if min(input_width, hidden_width, output_width) <= 0:
            raise WriterModelError("invalid LoRA factor-head dimensions")
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_width, hidden_width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, output_width, bias=False),
        )
        torch.nn.init.zeros_(self.network[-1].weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim < 3:
            raise WriterModelError("factor head lost its rank-slot dimension")
        return self.network(value)


class CompleteLoRAWriter(torch.nn.Module):
    """Map task language and one raw video to the sealed rank-16 task LoRA."""

    EXPERT_LAYERS = 18
    PUBLIC_LORA_RANK = 16
    _EXPERT_MODULE = re.compile(
        r".*gemma_expert\.model\.layers\.([0-9]+)\.self_attn\.(q_proj|v_proj)$"
    )
    FACTOR_WIDTHS = {
        "q_a": 1024,
        "q_b": 2048,
        "v_a": 1024,
        "v_b": 256,
        "action_in_a": 32,
        "action_in_b": 1024,
        "action_out_a": 1024,
        "action_out_b": 32,
    }

    def __init__(
        self,
        tensor_specs: tuple[LoraTensorSpec, ...],
        *,
        template_state: Mapping[str, torch.Tensor],
        paligemma_model: torch.nn.Module,
        expert_model: torch.nn.Module,
        image_width: int,
        expert_width: int,
        program_width: int,
        text_meta_lora_rank: int,
        vl_meta_lora_rank: int,
        action_meta_lora_rank: int,
        patch_grounding_heads: int,
        max_frames_per_encoder_call: int,
        action_horizon: int,
        padded_action_dim: int,
        semantic_core_heads: int,
        semantic_core_blocks: int,
        frame_attention_initial_lambda: float,
        procedure_heads: int,
        procedure_blocks: int,
        fusion_heads: int,
        factor_hidden_width: int,
        initialization_seed: int,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        if (
            not tensor_specs
            or set(template_state) != {item.name for item in tensor_specs}
            or len(paligemma_model.layers) != self.EXPERT_LAYERS
            or len(expert_model.layers) != self.EXPERT_LAYERS
            or {item.rank for item in tensor_specs} != {self.PUBLIC_LORA_RANK}
            or image_width != 2048
            or expert_width != 1024
            or program_width != 256
            or text_meta_lora_rank != 4
            or vl_meta_lora_rank != 4
            or action_meta_lora_rank != 4
            or patch_grounding_heads != 8
            or action_horizon != 50
            or padded_action_dim != 32
            or semantic_core_heads != 8
            or semantic_core_blocks != 2
            or abs(float(frame_attention_initial_lambda) - 0.05) > 1e-12
            or procedure_heads != 8
            or procedure_blocks != 2
            or fusion_heads != 8
            or factor_hidden_width != 216
        ):
            raise WriterModelError("invalid Language-Axial Writer topology")
        self.tensor_specs = tensor_specs
        self.program_width = int(program_width)
        self.semantic_encoder = Pi05LanguageAxialEncoder(
            paligemma_model=paligemma_model,
            expert_model=expert_model,
            image_width=image_width,
            expert_width=expert_width,
            program_width=program_width,
            text_meta_lora_rank=text_meta_lora_rank,
            vl_meta_lora_rank=vl_meta_lora_rank,
            action_meta_lora_rank=action_meta_lora_rank,
            patch_grounding_heads=patch_grounding_heads,
            max_frames_per_encoder_call=max_frames_per_encoder_call,
            action_horizon=action_horizon,
            padded_action_dim=padded_action_dim,
            initialization_seed=initialization_seed,
            activation_checkpointing=activation_checkpointing,
        )
        self.semantic_core = LanguageSemanticCore(
            width=program_width,
            heads=semantic_core_heads,
            blocks=semantic_core_blocks,
            frame_attention_initial_lambda=frame_attention_initial_lambda,
        )
        self.procedure = CausalProcedureEncoder(
            width=program_width,
            heads=procedure_heads,
            blocks=procedure_blocks,
        )
        self.compiler = SlotNormalizedCoreProcedureCompiler(
            width=program_width,
            heads=fusion_heads,
            initialization_seed=initialization_seed + 1,
        )
        self.factor_heads = torch.nn.ModuleDict(
            {
                name: FactorHead(
                    program_width,
                    factor_hidden_width,
                    width,
                )
                for name, width in self.FACTOR_WIDTHS.items()
            }
        )
        self._register_template_state(tensor_specs, template_state)

    def _register_template_state(
        self,
        tensor_specs: tuple[LoraTensorSpec, ...],
        template_state: Mapping[str, torch.Tensor],
    ) -> None:
        self._template_buffers: dict[str, str] = {}
        self._decoding: dict[str, tuple[str, int | None]] = {}
        observed_heads: dict[str, int] = {}
        observed_layers: set[int] = set()
        for index, item in enumerate(tensor_specs):
            key, layer = self._decode_owner(item)
            observed_heads[key] = item.width
            if layer is not None:
                observed_layers.add(layer)
            value = template_state[item.name].detach().contiguous()
            if item.factor_index == 1 and torch.count_nonzero(value):
                raise WriterModelError("LoRA-B template must begin at physical zero")
            buffer_name = f"template_{index:03d}"
            self.register_buffer(buffer_name, value, persistent=True)
            self._template_buffers[item.name] = buffer_name
            self._decoding[item.name] = (key, layer)
        if (
            observed_heads != self.FACTOR_WIDTHS
            or observed_layers != set(range(self.EXPERT_LAYERS))
        ):
            raise WriterModelError("sealed PI05 LoRA modules changed topology")

    @staticmethod
    def _validated_offsets(
        offsets: torch.Tensor,
        total: int,
    ) -> tuple[int, ...]:
        if offsets.ndim != 1 or offsets.numel() < 2:
            raise WriterModelError("Writer video offsets are invalid")
        values = tuple(
            int(value)
            for value in offsets.detach().to(device="cpu", dtype=torch.long).tolist()
        )
        if (
            values[0] != 0
            or values[-1] != total
            or any(right <= left for left, right in zip(values, values[1:]))
        ):
            raise WriterModelError("Writer video offsets are invalid")
        return values

    def _decode_owner(self, item: LoraTensorSpec) -> tuple[str, int | None]:
        factor = "a" if item.factor_index == 0 else "b"
        if item.module.endswith("action_in_proj"):
            return f"action_in_{factor}", None
        if item.module.endswith("action_out_proj"):
            return f"action_out_{factor}", None
        match = self._EXPERT_MODULE.fullmatch(item.module)
        if match is None:
            raise WriterModelError(
                f"unsupported PI05 task-LoRA module: {item.module}"
            )
        layer = int(match.group(1))
        if not 0 <= layer < self.EXPERT_LAYERS:
            raise WriterModelError("PI05 task-LoRA layer is outside Action Expert")
        return f"{match.group(2)[0]}_{factor}", layer

    def _pack_video_program(
        self,
        frame_evidence: torch.Tensor,
        interactions: torch.Tensor,
        frame_indices: torch.Tensor,
        offsets: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = len(offsets) - 1
        lengths = tuple(right - left for left, right in zip(offsets, offsets[1:]))
        maximum = max(lengths)
        packed_evidence = frame_evidence.new_zeros(
            batch,
            maximum,
            frame_evidence.shape[1],
            self.program_width,
        )
        packed_interactions = interactions.new_zeros(
            batch,
            maximum,
            self.program_width,
        )
        positions = torch.zeros(
            batch,
            maximum,
            dtype=torch.long,
            device=interactions.device,
        )
        valid_frames = torch.zeros(
            batch,
            maximum,
            dtype=torch.bool,
            device=interactions.device,
        )
        for row, (left, right) in enumerate(zip(offsets, offsets[1:])):
            length = right - left
            active_positions = frame_indices[left:right]
            if (
                int(active_positions[0]) != 0
                or bool((active_positions[1:] <= active_positions[:-1]).any())
            ):
                raise WriterModelError(
                    "sampled frame ordinals must start at zero and increase"
                )
            packed_evidence[row, :length] = frame_evidence[left:right]
            packed_interactions[row, :length] = interactions[left:right]
            positions[row, :length] = active_positions
            valid_frames[row, :length] = True
        return packed_evidence, packed_interactions, positions, valid_frames

    def encode_task(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        offsets = self._validated_offsets(video_offsets, frames.shape[0])
        conditions = len(offsets) - 1
        if (
            frames.ndim != 4
            or frame_indices.ndim != 1
            or frame_indices.shape[0] != frames.shape[0]
            or frame_indices.dtype != torch.long
            or language_tokens.ndim != 2
            or language_tokens.shape[0] != conditions
            or language_mask.shape != language_tokens.shape
            or language_mask.dtype != torch.bool
            or task_span_mask.shape != language_tokens.shape
            or task_span_mask.dtype != torch.bool
        ):
            raise WriterModelError("Writer frame-language condition batch changed")
        lengths = torch.tensor(
            [right - left for left, right in zip(offsets, offsets[1:])],
            dtype=torch.long,
            device=frames.device,
        )
        condition_ids = torch.repeat_interleave(
            torch.arange(conditions, device=frames.device),
            lengths,
        )
        (
            text_queries,
            frame_evidence,
            interactions,
            valid_task_tokens,
        ) = self.semantic_encoder(
            policy,
            frames,
            condition_ids,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        (
            packed_evidence,
            packed_interactions,
            positions,
            valid_frames,
        ) = self._pack_video_program(
            frame_evidence,
            interactions,
            frame_indices,
            offsets,
        )
        core_memory, frame_attention = self.semantic_core(
            text_queries,
            packed_evidence,
            valid_frames,
            valid_task_tokens,
        )
        procedure_memory = self.procedure(
            packed_interactions,
            positions,
            valid_frames,
        )
        return (
            core_memory,
            valid_task_tokens,
            procedure_memory,
            positions,
            valid_frames,
            frame_attention,
        )

    def forward(
        self,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        *,
        policy: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        (
            core_memory,
            valid_core,
            procedure_memory,
            positions,
            valid_frames,
            _,
        ) = self.encode_task(
            policy,
            frames,
            frame_indices,
            video_offsets,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        expert, action_in, action_out = self.compiler(
            core_memory,
            valid_core,
            procedure_memory,
            positions,
            valid_frames,
        )
        result: dict[str, torch.Tensor] = {}
        for item in self.tensor_specs:
            key, layer = self._decoding[item.name]
            if key.startswith("action_in_"):
                source = action_in
            elif key.startswith("action_out_"):
                source = action_out
            else:
                if layer is None:
                    raise WriterModelError("expert LoRA output lost its layer")
                source = expert[:, layer]
            rows = self.factor_heads[key](source)
            generated = rows.transpose(-1, -2) if item.transpose_output else rows
            template = getattr(self, self._template_buffers[item.name])
            value = generated.to(dtype=template.dtype) + template[None]
            result[item.name] = value[0] if core_memory.shape[0] == 1 else value
        return result
