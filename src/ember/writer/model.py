"""Canonical fixed-condition-kernel Program Memory PI05 Writer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

import torch

from ember.writer.architecture import validate_writer_dimensions
from ember.writer.condition_kernel import (
    FactorizedConditionFeature,
    ProgramValueMemory,
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
    PROGRAM_SLOTS = (EXPERT_LAYERS + 2) * PUBLIC_LORA_RANK
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
        condition_authority: Mapping[str, torch.Tensor],
        image_width: int,
        expert_width: int,
        program_width: int,
        max_frames_per_encoder_call: int,
        action_horizon: int,
        padded_action_dim: int,
        factor_hidden_width: int,
        condition_task_rff_frequencies: int,
        condition_video_rff_frequencies: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        try:
            validate_writer_dimensions(
                {
                    "image_width": image_width,
                    "expert_width": expert_width,
                    "program_width": program_width,
                    "action_horizon": action_horizon,
                    "padded_action_dim": padded_action_dim,
                    "factor_hidden_width": factor_hidden_width,
                    "condition_task_rff_frequencies": condition_task_rff_frequencies,
                    "condition_video_rff_frequencies": condition_video_rff_frequencies,
                }
            )
        except ValueError as error:
            raise WriterModelError("invalid condition-kernel Writer topology") from error
        if (
            not tensor_specs
            or max_frames_per_encoder_call <= 0
            or set(condition_authority)
            != {"task_center", "task_frequencies", "video_frequencies"}
        ):
            raise WriterModelError("invalid condition-kernel Writer topology")
        if set(template_state) != {item.name for item in tensor_specs}:
            raise WriterModelError("Writer LoRA template names changed")
        if (
            len(paligemma_model.layers) != self.EXPERT_LAYERS
            or len(expert_model.layers) != self.EXPERT_LAYERS
        ):
            raise WriterModelError("frozen PI05 expert depth changed")
        if {item.rank for item in tensor_specs} != {self.PUBLIC_LORA_RANK}:
            raise WriterModelError("public Writer LoRA rank changed")
        self.tensor_specs = tensor_specs
        self.program_width = int(program_width)
        self.condition_descriptor = Pi05FrozenConditionDescriptor(
            image_width=image_width,
            expert_width=expert_width,
            max_frames_per_encoder_call=max_frames_per_encoder_call,
            action_horizon=action_horizon,
            padded_action_dim=padded_action_dim,
            initialization_seed=initialization_seed,
        )
        self.condition_feature = FactorizedConditionFeature(
            task_center=condition_authority["task_center"],
            task_frequencies=condition_authority["task_frequencies"],
            video_frequencies=condition_authority["video_frequencies"],
        )
        if (
            self.condition_feature.task_frequencies.shape[0]
            != condition_task_rff_frequencies
            or self.condition_feature.video_frequencies.shape[0]
            != condition_video_rff_frequencies
            or self.condition_descriptor.task_descriptor_width
            != self.condition_feature.task_center.numel()
            or self.condition_descriptor.video_descriptor_width
            != self.condition_feature.video_frequencies.shape[1]
        ):
            raise WriterModelError("condition address authority changed dimensions")
        self.program_memory = ProgramValueMemory(
            feature_width=self.condition_feature.feature_width,
            program_slots=self.PROGRAM_SLOTS,
            program_width=program_width,
            initialization_seed=initialization_seed + 1,
        )
        # Program Memory has an explicit full-task kernel update.  Keeping it
        # outside autograd/Adam is the central ownership boundary of this
        # architecture; AS and reward training both differentiate a detached
        # Program leaf and then write the resulting cotangent through Phi.
        self.program_memory.value.requires_grad_(False)
        self.factor_heads = torch.nn.ModuleDict(
            {
                name: FactorHead(program_width, factor_hidden_width, width)
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

    def encode_condition(
        self,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        *,
        policy: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        for left, right in zip(offsets, offsets[1:]):
            active_positions = frame_indices[left:right]
            if (
                int(active_positions[0]) != 0
                or bool((active_positions[1:] <= active_positions[:-1]).any())
            ):
                raise WriterModelError(
                    "sampled frame ordinals must start at zero and increase"
                )
        lengths = torch.tensor(
            [right - left for left, right in zip(offsets, offsets[1:])],
            dtype=torch.long,
            device=frames.device,
        )
        condition_ids = torch.repeat_interleave(
            torch.arange(conditions, device=frames.device),
            lengths,
        )
        task_descriptor, video_descriptor = self.condition_descriptor(
            policy,
            frames,
            condition_ids,
            video_offsets,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        feature = self.condition_feature(task_descriptor, video_descriptor)
        program = self.program_memory(feature)
        if program.shape != (
            conditions,
            self.PROGRAM_SLOTS,
            self.program_width,
        ):
            raise WriterModelError("Writer policy program changed shape")
        return feature, program

    def encode_program(
        self,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        *,
        policy: torch.nn.Module,
    ) -> torch.Tensor:
        _, program = self.encode_condition(
            frames,
            frame_indices,
            video_offsets,
            language_tokens,
            language_mask,
            task_span_mask,
            policy=policy,
        )
        return program

    def decode_program(self, program: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode one differentiable policy program into the complete public LoRA."""

        if program.ndim != 3 or program.shape[1:] != (
            self.PROGRAM_SLOTS,
            self.program_width,
        ) or not bool(torch.isfinite(program).all()):
            raise WriterModelError("invalid Writer policy program")
        expert_stop = self.EXPERT_LAYERS * self.PUBLIC_LORA_RANK
        action_in_stop = expert_stop + self.PUBLIC_LORA_RANK
        result: dict[str, torch.Tensor] = {}
        for item in self.tensor_specs:
            key, layer = self._decoding[item.name]
            if key.startswith("action_in_"):
                source = program[:, expert_stop:action_in_stop]
            elif key.startswith("action_out_"):
                source = program[:, action_in_stop:]
            else:
                if layer is None:
                    raise WriterModelError("expert LoRA output lost its layer")
                left = layer * self.PUBLIC_LORA_RANK
                source = program[:, left : left + self.PUBLIC_LORA_RANK]
            rows = self.factor_heads[key](source)
            generated = rows.transpose(-1, -2) if item.transpose_output else rows
            template = getattr(self, self._template_buffers[item.name])
            if generated.shape[1:] != template.shape:
                raise WriterModelError("factor output changed public LoRA shape")
            value = generated.to(dtype=template.dtype) + template[None]
            result[item.name] = value[0] if program.shape[0] == 1 else value
        return result

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
        program = self.encode_program(
            frames,
            frame_indices,
            video_offsets,
            language_tokens,
            language_mask,
            task_span_mask,
            policy=policy,
        )
        return self.decode_program(program)
