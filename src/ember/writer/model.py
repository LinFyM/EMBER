"""Canonical dynamic-K shared-Core Procedure-set bridge over native v6."""

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
from ember.writer.slot_set import PolicyProcedureSetFusion


@dataclass(frozen=True)
class WriterProgramDiagnostics:
    """Memory-level evidence retained for mechanism analysis."""

    shared_core_slots: torch.Tensor
    per_video_procedure_slots: torch.Tensor
    shared_procedure_slots: torch.Tensor
    attention: tuple[torch.Tensor, ...]
    auxiliary_loss: torch.Tensor


@dataclass(frozen=True)
class WriterProgramOutput:
    """Frozen v6 memory readouts and their trainable Procedure set."""

    program: torch.Tensor
    diagnostics: WriterProgramDiagnostics


class CompleteLoRAWriter(torch.nn.Module):
    """Compile one shared v6 Core/Procedure program from one-to-four videos."""

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
        self.procedure_set = PolicyProcedureSetFusion(width=self.PROGRAM_WIDTH)

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
        """Train only Procedure-Set while the loaded v6 base remains in eval mode."""

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
        """Build per-video memories, then share Core and aggregate Procedure."""

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
            raise WriterModelError("invalid v6 shared-Core Procedure-set batch")
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
            compiler = self.base_writer.compiler
            shared_core_slots: list[torch.Tensor | None] = [
                None
            ] * len(cardinalities)
            for video_count in range(1, 5):
                condition_ids = [
                    condition_id
                    for condition_id, count in enumerate(cardinalities)
                    if count == video_count
                ]
                if not condition_ids:
                    continue
                video_ids = torch.tensor(
                    [
                        video_id
                        for condition_id in condition_ids
                        for video_id in range(
                            condition_bounds[condition_id],
                            condition_bounds[condition_id + 1],
                        )
                    ],
                    dtype=torch.long,
                    device=memories.core.device,
                )
                core = memories.core.index_select(0, video_ids).reshape(
                    len(condition_ids), -1, self.PROGRAM_WIDTH
                )
                valid_core = memories.valid_core.index_select(
                    0, video_ids
                ).reshape(len(condition_ids), -1)
                routing = compiler.routing(len(condition_ids))
                core_slots = compiler.read_core_slots(routing, core, valid_core)
                normalized_core = compiler.normalize_core_slots(core_slots)
                for row, condition_id in enumerate(condition_ids):
                    shared_core_slots[condition_id] = normalized_core[row]
            if any(value is None for value in shared_core_slots):
                raise WriterModelError("missing shared Core condition")
            shared_core = torch.stack(
                [value for value in shared_core_slots if value is not None]
            )
            procedure_routing = compiler.routing(len(video_bounds) - 1)
            per_video_procedure, _ = compiler.read_procedure_slots(
                procedure_routing,
                shared_core.index_select(0, video_condition_ids),
                memories.procedure,
                memories.positions,
                memories.valid_procedure,
            )
        if (
            shared_core.shape
            != (len(condition_bounds) - 1, self.POLICY_SLOTS, self.PROGRAM_WIDTH)
            or per_video_procedure.shape
            != (len(video_bounds) - 1, self.POLICY_SLOTS, self.PROGRAM_WIDTH)
        ):
            raise WriterModelError("native v6 memory readout topology changed")
        shared_procedure, set_diagnostics = self.procedure_set(
            per_video_procedure.detach(), condition_video_offsets
        )
        routing = self.base_writer.compiler.routing(shared_core.shape[0])
        program, _, _ = self.base_writer.compiler.fuse_readouts(
            routing,
            shared_core,
            shared_procedure,
        )
        return WriterProgramOutput(
            program,
            WriterProgramDiagnostics(
                shared_core_slots=shared_core,
                per_video_procedure_slots=per_video_procedure,
                shared_procedure_slots=shared_procedure,
                attention=set_diagnostics.attention,
                auxiliary_loss=set_diagnostics.auxiliary_loss,
            ),
        )

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
