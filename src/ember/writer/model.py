"""K4 phase-aligned Language-Axial Semantic-Procedure PI05 Writer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from ember.writer.architecture import (
    LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS,
    validate_writer_dimensions,
)
from ember.writer.temporal import (
    CausalProcedureEncoder,
    LanguageSemanticCore,
    SlotNormalizedCoreProcedureCompiler,
    TaskGroundedVisualTransitionFusion,
)
from ember.writer.video_program import Pi05LanguageAxialEncoder


class WriterModelError(RuntimeError):
    """Raised when the Writer input or sealed LoRA contract is inconsistent."""


@dataclass(frozen=True)
class LoraTensorSpec:
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
        modules.setdefault(name[: -len(marker)], {})[factor] = (name, value)
    if not modules or any(set(pair) != {"A", "B"} for pair in modules.values()):
        raise WriterModelError("every target module must contain one LoRA A/B pair")
    result = []
    for module_index, module in enumerate(sorted(modules)):
        name_a, value_a = modules[module]["A"]
        name_b, value_b = modules[module]["B"]
        rank, input_width = value_a.shape
        output_width, rank_b = value_b.shape
        if rank <= 0 or rank_b != rank:
            raise WriterModelError(f"LoRA rank differs for {module}")
        result.extend(
            (
                LoraTensorSpec(name_a, module, module_index, 0, rank, input_width, False),
                LoraTensorSpec(name_b, module, module_index, 1, rank, output_width, True),
            )
        )
    return tuple(result)


class FactorHead(torch.nn.Module):
    def __init__(self, input_width: int, hidden_width: int, output_width: int) -> None:
        super().__init__()
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
    """Map exact task language and four videos to one complete rank-16 LoRA."""

    EXPERT_LAYERS = 18
    PUBLIC_LORA_RANK = 16
    PROGRAM_SLOTS = EXPERT_LAYERS * PUBLIC_LORA_RANK + 2 * PUBLIC_LORA_RANK
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
        procedure_heads: int,
        procedure_blocks: int,
        visual_transition_heads: int,
        fusion_heads: int,
        factor_hidden_width: int,
        videos_per_condition: int,
        phase_slots: int,
        initialization_seed: int,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        constructor_values = locals()
        dimensions = {
            name: constructor_values[name]
            for name in LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS
            if name not in {"max_frames_per_encoder_call", "initialization_seed", "activation_checkpointing"}
        }
        try:
            validate_writer_dimensions(dimensions)
        except ValueError as error:
            raise WriterModelError("invalid K4 phase-aligned Writer topology") from error
        if not tensor_specs or set(template_state) != {item.name for item in tensor_specs}:
            raise WriterModelError("Writer LoRA template changed")
        if (
            len(paligemma_model.layers) != self.EXPERT_LAYERS
            or len(expert_model.layers) != self.EXPERT_LAYERS
            or {item.rank for item in tensor_specs} != {self.PUBLIC_LORA_RANK}
        ):
            raise WriterModelError("frozen PI05 public topology changed")
        self.tensor_specs = tensor_specs
        self.program_width = int(program_width)
        self.videos_per_condition = int(videos_per_condition)
        self.phase_slots = int(phase_slots)
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
            width=program_width, heads=semantic_core_heads, blocks=semantic_core_blocks
        )
        self.visual_transition = TaskGroundedVisualTransitionFusion(
            width=program_width, heads=visual_transition_heads
        )
        self.procedure = CausalProcedureEncoder(
            width=program_width, heads=procedure_heads, blocks=procedure_blocks
        )
        self.compiler = SlotNormalizedCoreProcedureCompiler(
            width=program_width, heads=fusion_heads, initialization_seed=initialization_seed + 1
        )
        self.factor_heads = torch.nn.ModuleDict(
            {
                name: FactorHead(program_width, factor_hidden_width, width)
                for name, width in self.FACTOR_WIDTHS.items()
            }
        )
        self._register_template_state(template_state)

    def _register_template_state(self, template_state: Mapping[str, torch.Tensor]) -> None:
        self._template_buffers: dict[str, str] = {}
        self._decoding: dict[str, tuple[str, int | None]] = {}
        observed_heads: dict[str, int] = {}
        observed_layers: set[int] = set()
        for index, item in enumerate(self.tensor_specs):
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
        if observed_heads != self.FACTOR_WIDTHS or observed_layers != set(range(self.EXPERT_LAYERS)):
            raise WriterModelError("sealed PI05 LoRA modules changed topology")

    def _decode_owner(self, item: LoraTensorSpec) -> tuple[str, int | None]:
        factor = "a" if item.factor_index == 0 else "b"
        if item.module.endswith("action_in_proj"):
            return f"action_in_{factor}", None
        if item.module.endswith("action_out_proj"):
            return f"action_out_{factor}", None
        match = self._EXPERT_MODULE.fullmatch(item.module)
        if match is None:
            raise WriterModelError(f"unsupported PI05 task-LoRA module: {item.module}")
        return f"{match.group(2)[0]}_{factor}", int(match.group(1))

    @staticmethod
    def _validated_offsets(offsets: torch.Tensor, total: int, label: str) -> tuple[int, ...]:
        if offsets.ndim != 1 or offsets.dtype != torch.long or offsets.numel() < 2:
            raise WriterModelError(f"Writer {label} offsets are invalid")
        values = tuple(int(value) for value in offsets.detach().cpu().tolist())
        if values[0] != 0 or values[-1] != total or any(
            right <= left for left, right in zip(values, values[1:])
        ):
            raise WriterModelError(f"Writer {label} offsets are invalid")
        return values

    def _phase_resample(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim < 2 or value.shape[0] < 2:
            raise WriterModelError("video needs at least two sampled frames")
        source_dtype = value.dtype
        flat = value.to(torch.float32).reshape(value.shape[0], -1).transpose(0, 1)[None]
        aligned = F.interpolate(flat, size=self.phase_slots, mode="linear", align_corners=True)
        return aligned[0].transpose(0, 1).reshape(self.phase_slots, *value.shape[1:]).to(source_dtype)

    def encode_memories(
        self,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        video_offsets: torch.Tensor,
        condition_video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        *,
        policy: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        video_bounds = self._validated_offsets(video_offsets, frames.shape[0], "frame-video")
        video_count = len(video_bounds) - 1
        condition_bounds = self._validated_offsets(
            condition_video_offsets, video_count, "video-condition"
        )
        conditions = len(condition_bounds) - 1
        if (
            frames.ndim != 4
            or frame_indices.shape != (frames.shape[0],)
            or language_tokens.shape[0] != conditions
            or language_mask.shape != language_tokens.shape
            or task_span_mask.shape != language_tokens.shape
            or any(right - left != self.videos_per_condition for left, right in zip(condition_bounds, condition_bounds[1:]))
        ):
            raise WriterModelError("Writer K4 frame-language batch changed")
        for left, right in zip(video_bounds, video_bounds[1:]):
            positions = frame_indices[left:right]
            if int(positions[0]) != 0 or bool((positions[1:] <= positions[:-1]).any()):
                raise WriterModelError("each sampled video must start at frame zero and increase")
        video_condition_ids = torch.repeat_interleave(
            torch.arange(conditions, device=frames.device),
            torch.tensor(
                [right - left for left, right in zip(condition_bounds, condition_bounds[1:])],
                dtype=torch.long,
                device=frames.device,
            ),
        )
        frame_video_ids = torch.repeat_interleave(
            torch.arange(video_count, device=frames.device),
            torch.tensor(
                [right - left for left, right in zip(video_bounds, video_bounds[1:])],
                dtype=torch.long,
                device=frames.device,
            ),
        )
        frame_condition_ids = video_condition_ids.index_select(0, frame_video_ids)
        text, evidence, grounded, interactions, valid_tokens = self.semantic_encoder(
            policy,
            frames,
            frame_condition_ids,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        aligned_evidence = torch.stack(
            [self._phase_resample(evidence[left:right]) for left, right in zip(video_bounds, video_bounds[1:])]
        )
        aligned_grounded = torch.stack(
            [self._phase_resample(grounded[left:right]) for left, right in zip(video_bounds, video_bounds[1:])]
        )
        aligned_interactions = torch.stack(
            [self._phase_resample(interactions[left:right]) for left, right in zip(video_bounds, video_bounds[1:])]
        )
        semantic_evidence = aligned_evidence.reshape(
            conditions, self.videos_per_condition * self.phase_slots, aligned_evidence.shape[2], self.program_width
        )
        semantic_valid = torch.ones(
            semantic_evidence.shape[:2], dtype=torch.bool, device=frames.device
        )
        core, _ = self.semantic_core(text, semantic_evidence, semantic_valid, valid_tokens)
        per_video_tokens = valid_tokens.index_select(0, video_condition_ids)
        procedure_input, _ = self.visual_transition(
            aligned_interactions,
            aligned_grounded,
            torch.ones(video_count, self.phase_slots, dtype=torch.bool, device=frames.device),
            per_video_tokens,
        )
        phase_positions = torch.arange(
            self.phase_slots, dtype=torch.long, device=frames.device
        )[None].expand(video_count, -1)
        procedure = self.procedure(
            procedure_input,
            phase_positions,
            torch.ones_like(phase_positions, dtype=torch.bool),
        ).reshape(conditions, self.videos_per_condition, self.phase_slots, self.program_width).mean(dim=1)
        condition_positions = phase_positions[:conditions]
        condition_valid = torch.ones_like(condition_positions, dtype=torch.bool)
        return core, valid_tokens, procedure, condition_positions, condition_valid

    def encode_program(self, *args: torch.Tensor, policy: torch.nn.Module) -> torch.Tensor:
        core, valid_core, procedure, positions, valid_procedure = self.encode_memories(
            *args, policy=policy
        )
        program, _ = self.compiler.fused_slots(
            core, valid_core, procedure, positions, valid_procedure
        )
        return program

    def decode_program(self, program: torch.Tensor) -> dict[str, torch.Tensor]:
        if program.ndim != 3 or program.shape[1:] != (self.PROGRAM_SLOTS, self.program_width):
            raise WriterModelError("invalid Writer fused program")
        stop = self.EXPERT_LAYERS * self.PUBLIC_LORA_RANK
        expert = program[:, :stop].reshape(
            program.shape[0], self.EXPERT_LAYERS, self.PUBLIC_LORA_RANK, self.program_width
        )
        action_in = program[:, stop : stop + self.PUBLIC_LORA_RANK]
        action_out = program[:, -self.PUBLIC_LORA_RANK :]
        result = {}
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
            value = generated.to(getattr(self, self._template_buffers[item.name]).dtype)
            value = value + getattr(self, self._template_buffers[item.name])[None]
            result[item.name] = value[0] if program.shape[0] == 1 else value
        return result

    def forward(self, *args: torch.Tensor, policy: torch.nn.Module) -> dict[str, torch.Tensor]:
        return self.decode_program(self.encode_program(*args, policy=policy))
