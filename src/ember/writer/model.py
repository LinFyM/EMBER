"""Canonical Dynamic-K Backbone-Memory PI05 Writer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ember.writer.backbone_memory import (
    BackboneMemoryOutput,
    Pi05BackboneMemoryEncoder,
)
from ember.writer.errors import WriterModelError
from ember.writer.lora_mapper import (
    CompleteLoRAMapper,
    LoraTensorSpec,
    build_lora_tensor_specs,
)
from ember.writer.memory_program import (
    DynamicKMemoryProgram,
    MemoryProgramDiagnostics,
)


@dataclass(frozen=True)
class WriterProgramOutput:
    """Representation chain retained for mechanism diagnostics and training."""

    backbone: BackboneMemoryOutput
    program: torch.Tensor
    diagnostics: MemoryProgramDiagnostics


class CompleteLoRAWriter(torch.nn.Module):
    """Generate one complete rank-8 LoRA from language and one-to-four videos."""

    PUBLIC_LORA_RANK = 8
    EXPERT_LAYERS = 18
    PROGRAM_WIDTH = 256
    MAPPER_WIDTH = 1024

    def __init__(
        self,
        tensor_specs: tuple[LoraTensorSpec, ...],
        *,
        template_state: Mapping[str, torch.Tensor],
        bridge: torch.nn.Module,
        image_width: int = 2048,
        expert_width: int = 1024,
        program_width: int = PROGRAM_WIDTH,
        mapper_width: int = MAPPER_WIDTH,
        action_meta_lora_rank: int = 4,
        temporal_heads: int = 8,
        max_frames_per_encoder_call: int = 32,
        action_horizon: int = 50,
        padded_action_dim: int = 32,
        initialization_seed: int = 7,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if (
            not tensor_specs
            or {item.rank for item in tensor_specs} != {self.PUBLIC_LORA_RANK}
            or program_width != self.PROGRAM_WIDTH
            or mapper_width != self.MAPPER_WIDTH
            or image_width != 2048
            or expert_width != 1024
            or temporal_heads != 8
        ):
            raise WriterModelError("invalid Dynamic-K Writer topology")
        self.tensor_specs = tensor_specs
        self.backbone_memory = Pi05BackboneMemoryEncoder(
            bridge=bridge,
            image_width=image_width,
            expert_width=expert_width,
            max_frames_per_encoder_call=max_frames_per_encoder_call,
            action_horizon=action_horizon,
            padded_action_dim=padded_action_dim,
            initialization_seed=initialization_seed,
            action_meta_lora_rank=action_meta_lora_rank,
            activation_checkpointing=activation_checkpointing,
        )
        self.memory_program = DynamicKMemoryProgram(heads=temporal_heads)
        self.lora_mapper = CompleteLoRAMapper(
            tensor_specs,
            template_state=template_state,
            program_width=program_width,
            mapper_width=mapper_width,
            dynamic_a=False,
        )

    @classmethod
    def from_policy(
        cls,
        *,
        policy: torch.nn.Module,
        template_state: Mapping[str, torch.Tensor],
        writer_config: Mapping[str, Any],
    ) -> CompleteLoRAWriter:
        """Construct from the frozen policy that owns the real joint backbone."""

        bridge = getattr(
            getattr(policy, "model", None), "paligemma_with_expert", None
        )
        if bridge is None:
            raise WriterModelError("PI05 policy lost its joint backbone")
        allowed = {
            "image_width",
            "expert_width",
            "program_width",
            "mapper_width",
            "action_meta_lora_rank",
            "temporal_heads",
            "max_frames_per_encoder_call",
            "action_horizon",
            "padded_action_dim",
            "initialization_seed",
            "activation_checkpointing",
        }
        arguments = {
            name: writer_config[name]
            for name in allowed
            if name in writer_config
        }
        return cls(
            build_lora_tensor_specs(template_state),
            template_state=template_state,
            bridge=bridge,
            **arguments,
        )

    @staticmethod
    def _offsets(
        value: torch.Tensor,
        *,
        final: int,
        name: str,
    ) -> tuple[int, ...]:
        if (
            value.device.type != "cpu"
            or value.ndim != 1
            or value.dtype != torch.long
        ):
            raise WriterModelError(f"{name} must be a CPU long tensor")
        offsets = tuple(int(item) for item in value.tolist())
        if (
            len(offsets) < 2
            or offsets[0] != 0
            or offsets[-1] != final
            or any(right <= left for left, right in zip(offsets, offsets[1:]))
        ):
            raise WriterModelError(f"invalid {name}")
        return offsets

    def template_state(self) -> dict[str, torch.Tensor]:
        return self.lora_mapper.template_state()

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
        singleton_video_index: int = 0,
    ) -> WriterProgramOutput:
        """Run the one physical frame pass and compile its directed set program."""

        video_bounds = self._offsets(
            video_offsets, final=frames.shape[0], name="video offsets"
        )
        condition_bounds = self._offsets(
            condition_video_offsets,
            final=len(video_bounds) - 1,
            name="condition video offsets",
        )
        if (
            frame_indices.shape != (frames.shape[0],)
            or frame_indices.dtype != torch.long
            or frame_indices.device != frames.device
            or language_tokens.ndim != 2
            or len(condition_bounds) - 1 != language_tokens.shape[0]
            or language_mask.shape != language_tokens.shape
            or task_span_mask.shape != language_tokens.shape
        ):
            raise WriterModelError("invalid Dynamic-K Writer batch")
        video_counts = torch.tensor(
            [right - left for left, right in zip(video_bounds, video_bounds[1:])],
            dtype=torch.long,
            device=frames.device,
        )
        condition_video_counts = torch.tensor(
            [
                right - left
                for left, right in zip(condition_bounds, condition_bounds[1:])
            ],
            dtype=torch.long,
            device=frames.device,
        )
        video_condition_ids = torch.repeat_interleave(
            torch.arange(len(condition_bounds) - 1, device=frames.device),
            condition_video_counts,
        )
        frame_condition_ids = torch.repeat_interleave(
            video_condition_ids,
            video_counts,
        )
        backbone = self.backbone_memory(
            policy,
            frames,
            frame_condition_ids,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        program, diagnostics = self.memory_program(
            backbone.layer_memory,
            frame_indices,
            video_offsets,
            condition_video_offsets,
            singleton_video_index=singleton_video_index,
        )
        return WriterProgramOutput(backbone, program, diagnostics)

    def decode_program(self, program: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.lora_mapper(program)

    def forward_training(
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
        singleton_video_index: int = 0,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        encoded = self.encode_program(
            frames,
            frame_indices,
            video_offsets,
            condition_video_offsets,
            language_tokens,
            language_mask,
            task_span_mask,
            policy=policy,
            singleton_video_index=singleton_video_index,
        )
        return (
            self.decode_program(encoded.program),
            encoded.diagnostics.consistency_loss,
        )

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
        generated, _ = self.forward_training(
            frames,
            frame_indices,
            video_offsets,
            condition_video_offsets,
            language_tokens,
            language_mask,
            task_span_mask,
            policy=policy,
        )
        return generated
