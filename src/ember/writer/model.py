"""Canonical evidence-factorized K4 policy-layer trace M2P PI05 Writer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

import torch

from ember.writer.architecture import validate_writer_dimensions
from ember.writer.fewshot_m2p import (
    PolicyLayerTraceM2P,
    build_policy_layer_groups,
    build_policy_target_specs,
)
from ember.writer.video_program import Pi05FrozenConditionDescriptor


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
    """Jointly map task language and K action-hidden videos to one complete LoRA."""

    EXPERT_LAYERS = 18
    POLICY_TARGETS = 38
    PUBLIC_LORA_RANK = 16
    POLICY_GROUPS = EXPERT_LAYERS + 2
    _EXPERT_MODULE = re.compile(
        r".*gemma_expert\.model\.layers\.([0-9]+)\.self_attn\.(q_proj|v_proj)$"
    )

    def __init__(
        self,
        tensor_specs: tuple[LoraTensorSpec, ...],
        *,
        template_state: Mapping[str, torch.Tensor],
        paligemma_model: torch.nn.Module,
        expert_model: torch.nn.Module,
        image_width: int,
        expert_width: int,
        policy_groups: int,
        trace_temporal_terms: int,
        memory_slots: int,
        m2p_width: int,
        m2p_heads: int,
        m2p_blocks: int,
        m2p_ffn_expansion: int,
        max_frames_per_encoder_call: int,
        action_horizon: int,
        padded_action_dim: int,
        videos_per_condition: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        observed = {
            name: value
            for name, value in locals().items()
            if name
            in {
                "image_width",
                "expert_width",
                "policy_groups",
                "trace_temporal_terms",
                "memory_slots",
                "m2p_width",
                "m2p_heads",
                "m2p_blocks",
                "m2p_ffn_expansion",
                "action_horizon",
                "padded_action_dim",
                "videos_per_condition",
            }
        }
        try:
            validate_writer_dimensions(observed)
        except ValueError as error:
            raise WriterModelError("invalid K4 layer-trace M2P Writer topology") from error
        if not tensor_specs or max_frames_per_encoder_call <= 0:
            raise WriterModelError("invalid K4 layer-trace M2P Writer topology")
        if set(template_state) != {item.name for item in tensor_specs}:
            raise WriterModelError("Writer LoRA template names changed")
        if (
            len(paligemma_model.layers) != self.EXPERT_LAYERS
            or len(expert_model.layers) != self.EXPERT_LAYERS
            or {item.rank for item in tensor_specs} != {self.PUBLIC_LORA_RANK}
        ):
            raise WriterModelError("frozen PI05 public topology changed")
        self._validate_policy_targets(tensor_specs)
        self.tensor_specs = tensor_specs
        self.program_width = int(m2p_width)
        self.program_groups = int(policy_groups)
        self.program_slots = int(memory_slots)
        self.videos_per_condition = int(videos_per_condition)
        self.condition_descriptor = Pi05FrozenConditionDescriptor(
            image_width=image_width,
            expert_width=expert_width,
            max_frames_per_encoder_call=max_frames_per_encoder_call,
            action_horizon=action_horizon,
            padded_action_dim=padded_action_dim,
            initialization_seed=initialization_seed,
        )
        targets = build_policy_target_specs(tensor_specs)
        if len(targets) != self.POLICY_TARGETS:
            raise WriterModelError("PI05 public policy target count changed")
        groups = build_policy_layer_groups(
            targets,
            expert_layers=self.EXPERT_LAYERS,
        )
        if len(groups) != self.POLICY_GROUPS:
            raise WriterModelError("PI05 policy-layer group count changed")
        self.layer_m2p = PolicyLayerTraceM2P(
            groups,
            template_state=template_state,
            width=m2p_width,
            memory_slots=memory_slots,
            temporal_terms=trace_temporal_terms,
            heads=m2p_heads,
            blocks=m2p_blocks,
            ffn_expansion=m2p_ffn_expansion,
            initialization_seed=initialization_seed + 1,
        )

    def _validate_policy_targets(
        self, tensor_specs: tuple[LoraTensorSpec, ...]
    ) -> None:
        observed_layers: dict[str, set[int]] = {"q": set(), "v": set()}
        action_modules: set[str] = set()
        for item in tensor_specs:
            if item.module.endswith("action_in_proj"):
                action_modules.add("action_in")
                continue
            if item.module.endswith("action_out_proj"):
                action_modules.add("action_out")
                continue
            match = self._EXPERT_MODULE.fullmatch(item.module)
            if match is None:
                raise WriterModelError(
                    f"unsupported PI05 task-LoRA module: {item.module}"
                )
            observed_layers[match.group(2)[0]].add(int(match.group(1)))
        expected = set(range(self.EXPERT_LAYERS))
        if observed_layers != {"q": expected, "v": expected} or action_modules != {
            "action_in",
            "action_out",
        }:
            raise WriterModelError("sealed PI05 policy targets changed")

    @staticmethod
    def _validated_offsets(
        offsets: torch.Tensor,
        total: int,
        *,
        label: str,
    ) -> tuple[int, ...]:
        if offsets.ndim != 1 or offsets.dtype != torch.long or offsets.numel() < 2:
            raise WriterModelError(f"Writer {label} offsets are invalid")
        values = tuple(int(value) for value in offsets.detach().cpu().tolist())
        if (
            values[0] != 0
            or values[-1] != total
            or any(right <= left for left, right in zip(values, values[1:]))
        ):
            raise WriterModelError(f"Writer {label} offsets are invalid")
        return values

    def encode_program(
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
    ) -> torch.Tensor:
        video_bounds = self._validated_offsets(
            video_offsets, frames.shape[0], label="frame-video"
        )
        video_count = len(video_bounds) - 1
        condition_bounds = self._validated_offsets(
            condition_video_offsets, video_count, label="video-condition"
        )
        conditions = len(condition_bounds) - 1
        if (
            frames.ndim != 4
            or frame_indices.shape != (frames.shape[0],)
            or frame_indices.dtype != torch.long
            or language_tokens.ndim != 2
            or language_tokens.shape[0] != conditions
            or language_mask.shape != language_tokens.shape
            or language_mask.dtype != torch.bool
            or task_span_mask.shape != language_tokens.shape
            or task_span_mask.dtype != torch.bool
            or any(
                right - left != self.videos_per_condition
                for left, right in zip(condition_bounds, condition_bounds[1:])
            )
        ):
            raise WriterModelError("Writer K4 frame-language batch changed")
        for left, right in zip(video_bounds, video_bounds[1:]):
            positions = frame_indices[left:right]
            if (
                int(positions[0]) != 0
                or bool((positions[1:] <= positions[:-1]).any())
            ):
                raise WriterModelError(
                    "each sampled video must start at frame zero and increase"
                )
        video_lengths = torch.tensor(
            [right - left for left, right in zip(video_bounds, video_bounds[1:])],
            dtype=torch.long,
            device=frames.device,
        )
        frame_video_ids = torch.repeat_interleave(
            torch.arange(video_count, device=frames.device), video_lengths
        )
        condition_lengths = torch.tensor(
            [right - left for left, right in zip(condition_bounds, condition_bounds[1:])],
            dtype=torch.long,
            device=frames.device,
        )
        video_condition_ids = torch.repeat_interleave(
            torch.arange(conditions, device=frames.device), condition_lengths
        )
        video_traces = self.condition_descriptor(
            policy,
            frames,
            frame_video_ids,
            video_offsets,
            language_tokens.index_select(0, video_condition_ids),
            language_mask.index_select(0, video_condition_ids),
            task_span_mask.index_select(0, video_condition_ids),
        )
        program = self.layer_m2p.encode(
            video_traces,
            condition_video_offsets,
        )
        if program.shape != (
            conditions,
            self.program_groups,
            self.program_slots,
            self.program_width,
        ) or not bool(torch.isfinite(program).all()):
            raise WriterModelError("Writer policy-layer memory changed shape")
        return program

    def decode_program(self, program: torch.Tensor) -> dict[str, torch.Tensor]:
        if program.ndim != 4 or program.shape[1:] != (
            self.program_groups,
            self.program_slots,
            self.program_width,
        ) or not bool(torch.isfinite(program).all()):
            raise WriterModelError("invalid Writer policy-layer memory")
        result = self.layer_m2p.decode(program)
        if program.shape[0] == 1:
            return {name: value[0] for name, value in result.items()}
        return result

    def forward(
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
    ) -> dict[str, torch.Tensor]:
        return self.decode_program(
            self.encode_program(
                frames,
                frame_indices,
                video_offsets,
                condition_video_offsets,
                language_tokens,
                language_mask,
                task_span_mask,
                policy=policy,
            )
        )
