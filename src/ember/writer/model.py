"""Canonical target-bound role-preserving PI05 Writer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from ember.pi05_lora import pi05_target_names
from ember.writer.program_compiler import (
    FactorHead,
    TargetBoundRoleCompiler,
)
from ember.writer.semantic_core import MeanBackedSemanticCore
from ember.writer.semantic_program import TargetBoundRoleProgram
from ember.writer.video_program import Pi05SemanticEvidenceEncoder


class WriterModelError(RuntimeError):
    """Raised when the Writer input or sealed LoRA contract is inconsistent."""


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


class CompleteLoRAWriter(torch.nn.Module):
    """Map task language and one raw video to the sealed rank-16 task LoRA."""

    EXPERT_LAYERS = 18
    PUBLIC_LORA_RANK = 16
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
        program_heads: int,
        program_blocks: int,
        compiler_heads: int,
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
            or program_heads != 8
            or program_blocks != 2
            or compiler_heads != 8
            or factor_hidden_width != 256
        ):
            raise WriterModelError("invalid target-bound role Writer topology")
        self.tensor_specs = tensor_specs
        self.program_width = int(program_width)
        self.semantic_encoder = Pi05SemanticEvidenceEncoder(
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
        self.semantic_core = MeanBackedSemanticCore(
            width=program_width,
            heads=semantic_core_heads,
            blocks=semantic_core_blocks,
        )
        self.compiler = TargetBoundRoleCompiler(
            width=program_width,
            heads=compiler_heads,
            target_count=len(pi05_target_names()),
            rank=self.PUBLIC_LORA_RANK,
            initialization_seed=initialization_seed + 3,
        )
        self.semantic_program = TargetBoundRoleProgram(
            width=program_width,
            heads=program_heads,
            blocks=program_blocks,
            initialization_seed=initialization_seed + 1,
        )
        self.factor_heads = torch.nn.ModuleDict(
            {
                name: FactorHead(
                    4 * program_width,
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
        self._decoding: dict[str, tuple[str, int]] = {}
        observed_heads: dict[str, int] = {}
        observed_targets: set[int] = set()
        for index, item in enumerate(tensor_specs):
            key, target_index = self._decode_owner(item)
            observed_heads[key] = item.width
            observed_targets.add(target_index)
            value = template_state[item.name].detach().contiguous()
            if item.factor_index == 1 and torch.count_nonzero(value):
                raise WriterModelError("LoRA-B template must begin at physical zero")
            buffer_name = f"template_{index:03d}"
            self.register_buffer(buffer_name, value, persistent=True)
            self._template_buffers[item.name] = buffer_name
            self._decoding[item.name] = (key, target_index)
        if (
            observed_heads != self.FACTOR_WIDTHS
            or observed_targets != set(range(len(pi05_target_names())))
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

    def _decode_owner(self, item: LoraTensorSpec) -> tuple[str, int]:
        target_indices = {
            name: index for index, name in enumerate(pi05_target_names())
        }
        if item.module not in target_indices:
            raise WriterModelError(
                f"unsupported PI05 task-LoRA module: {item.module}"
            )
        factor = "a" if item.factor_index == 0 else "b"
        if item.module.endswith("action_in_proj"):
            return f"action_in_{factor}", target_indices[item.module]
        if item.module.endswith("action_out_proj"):
            return f"action_out_{factor}", target_indices[item.module]
        projection = item.module.rsplit(".", 1)[-1]
        if projection not in {"q_proj", "v_proj"}:
            raise WriterModelError("PI05 task-LoRA projection changed")
        return f"{projection[0]}_{factor}", target_indices[item.module]

    def _pack_video_program(
        self,
        frame_evidence: torch.Tensor,
        grounded_evidence: torch.Tensor,
        action_probes: torch.Tensor,
        frame_indices: torch.Tensor,
        offsets: tuple[int, ...],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch = len(offsets) - 1
        lengths = tuple(right - left for left, right in zip(offsets, offsets[1:]))
        maximum = max(lengths)
        packed_evidence = frame_evidence.new_zeros(
            batch,
            maximum,
            frame_evidence.shape[1],
            self.program_width,
        )
        packed_grounded = grounded_evidence.new_zeros(
            batch,
            maximum,
            grounded_evidence.shape[1],
            self.program_width,
        )
        packed_actions = action_probes.new_zeros(
            batch,
            maximum,
            self.program_width,
        )
        positions = torch.zeros(
            batch,
            maximum,
            dtype=torch.long,
            device=action_probes.device,
        )
        valid_frames = torch.zeros(
            batch,
            maximum,
            dtype=torch.bool,
            device=action_probes.device,
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
            packed_grounded[row, :length] = grounded_evidence[left:right]
            packed_actions[row, :length] = action_probes[left:right]
            positions[row, :length] = active_positions
            valid_frames[row, :length] = True
        return (
            packed_evidence,
            packed_grounded,
            packed_actions,
            positions,
            valid_frames,
        )

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
            grounded_evidence,
            action_probes,
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
            packed_grounded,
            packed_actions,
            positions,
            valid_frames,
        ) = self._pack_video_program(
            frame_evidence,
            grounded_evidence,
            action_probes,
            frame_indices,
            offsets,
        )
        core_memory, _frame_attention = self.semantic_core(
            text_queries,
            packed_evidence,
            valid_frames,
            valid_task_tokens,
        )
        target_query, target_core = self.compiler.read_target_core(
            core_memory,
            valid_task_tokens,
        )
        (
            program,
            endpoint_positions,
            valid_intervals,
        ) = self.semantic_program(
            packed_grounded,
            packed_actions,
            positions,
            valid_frames,
            valid_task_tokens,
            target_query,
            target_core,
        )
        return (
            target_query,
            target_core,
            program,
            endpoint_positions,
            valid_intervals,
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
            target_query,
            target_core,
            program,
            endpoint_positions,
            valid_intervals,
        ) = self.encode_task(
            policy,
            frames,
            frame_indices,
            video_offsets,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        coordinates = self.compiler(
            target_query,
            target_core,
            program,
            endpoint_positions,
            valid_intervals,
        )
        result: dict[str, torch.Tensor] = {}
        for item in self.tensor_specs:
            key, target_index = self._decoding[item.name]
            source = coordinates[:, target_index]
            rows = self.factor_heads[key](source)
            generated = rows.transpose(-1, -2) if item.transpose_output else rows
            template = getattr(self, self._template_buffers[item.name])
            value = generated.to(dtype=template.dtype) + template[None]
            result[item.name] = value[0] if target_core.shape[0] == 1 else value
        return result
