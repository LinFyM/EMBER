"""Canonical dynamic-K bridge over the frozen native v6 Writer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ember.expert_manifold.legacy_v6_architecture import (
    LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS,
)
from ember.expert_manifold.legacy_v6_model import (
    CompleteLoRAWriter as NativeV6Writer,
)
from ember.expert_manifold.legacy_v6_model import build_lora_tensor_specs
from ember.expert_manifold.v6_prior import load_v6_prior_warm_start_
from ember.writer.errors import WriterModelError
from ember.writer.slot_set import PolicySlotSetFusion, SlotSetDiagnostics


@dataclass(frozen=True)
class WriterProgramOutput:
    """Frozen per-video v6 slots and their trainable set aggregation."""

    program: torch.Tensor
    diagnostics: SlotSetDiagnostics


class CompleteLoRAWriter(torch.nn.Module):
    """Generate one rank-16 LoRA from one-to-four ordered teaching videos."""

    PUBLIC_LORA_RANK = 16
    PROGRAM_WIDTH = 256
    POLICY_SLOTS = 320

    def __init__(self, base_writer: NativeV6Writer) -> None:
        super().__init__()
        if (
            int(base_writer.program_width) != self.PROGRAM_WIDTH
            or base_writer.PUBLIC_LORA_RANK != self.PUBLIC_LORA_RANK
        ):
            raise WriterModelError("native v6 Writer topology changed")
        self.base_writer = base_writer.requires_grad_(False).eval()
        self.slot_set = PolicySlotSetFusion(width=self.PROGRAM_WIDTH)

    @classmethod
    def from_policy(
        cls,
        *,
        policy: torch.nn.Module,
        template_state: Mapping[str, torch.Tensor],
        writer_config: Mapping[str, Any],
        warm_start_checkpoint: Path,
    ) -> CompleteLoRAWriter:
        """Construct and strictly load the frozen v6-fast performance base."""

        bridge = getattr(
            getattr(policy, "model", None), "paligemma_with_expert", None
        )
        if bridge is None:
            raise WriterModelError("PI05 policy lost its joint backbone")
        arguments = {
            name: writer_config[name]
            for name in LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS
            if name in writer_config
        }
        base = NativeV6Writer(
            build_lora_tensor_specs(template_state),
            template_state=template_state,
            paligemma_model=bridge.paligemma.model.language_model,
            expert_model=bridge.gemma_expert.model,
            **arguments,
        )
        load_v6_prior_warm_start_(base, warm_start_checkpoint)
        return cls(base)

    def train(self, mode: bool = True) -> CompleteLoRAWriter:
        """Train only Slot-Set while the loaded v6 base remains in eval mode."""

        super().train(mode)
        self.base_writer.eval()
        return self

    @staticmethod
    def _offsets(
        value: torch.Tensor,
        *,
        final: int,
        name: str,
    ) -> tuple[int, ...]:
        if (
            value.device.type != "cpu"
            or value.dtype != torch.long
            or value.ndim != 1
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
        return self.base_writer.template_state()

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
        """Compile each video independently, then aggregate aligned v6 slots."""

        del singleton_video_index
        video_bounds = self._offsets(
            video_offsets, final=frames.shape[0], name="video offsets"
        )
        condition_bounds = self._offsets(
            condition_video_offsets,
            final=len(video_bounds) - 1,
            name="condition video offsets",
        )
        cardinalities = tuple(
            right - left
            for left, right in zip(condition_bounds, condition_bounds[1:])
        )
        condition_counts = torch.tensor(
            cardinalities,
            dtype=torch.long,
            device=language_tokens.device,
        )
        if (
            frame_indices.shape != (frames.shape[0],)
            or frame_indices.dtype != torch.long
            or frame_indices.device != frames.device
            or language_tokens.ndim != 2
            or language_tokens.shape[0] != len(condition_bounds) - 1
            or language_mask.shape != language_tokens.shape
            or task_span_mask.shape != language_tokens.shape
            or any(count not in range(1, 5) for count in cardinalities)
        ):
            raise WriterModelError("invalid v6 Dynamic Slot-Set batch")
        video_condition_ids = torch.repeat_interleave(
            torch.arange(
                len(condition_bounds) - 1,
                dtype=torch.long,
                device=language_tokens.device,
            ),
            condition_counts,
        )
        with torch.no_grad():
            evidence = self.base_writer.encode_video_evidence(
                policy,
                frames,
                video_offsets,
                language_tokens.index_select(0, video_condition_ids),
                language_mask.index_select(0, video_condition_ids),
                task_span_mask.index_select(0, video_condition_ids),
            )
            memories = self.base_writer.build_memories(evidence, frame_indices)
            per_video_slots = self.base_writer.compile_slots(memories)
        if per_video_slots.shape != (
            len(video_bounds) - 1,
            self.POLICY_SLOTS,
            self.PROGRAM_WIDTH,
        ):
            raise WriterModelError("native v6 per-video slot topology changed")
        shared, diagnostics = self.slot_set(
            per_video_slots.detach(), condition_video_offsets
        )
        return WriterProgramOutput(shared, diagnostics)

    def decode_program(self, program: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.base_writer.decode_slots(program)

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
        return self.decode_program(encoded.program), encoded.diagnostics.auxiliary_loss

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
