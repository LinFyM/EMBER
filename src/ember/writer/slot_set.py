"""Permutation-invariant aggregation of aligned v6 policy-program slots."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ember.writer.errors import WriterModelError
from ember.writer.temporal import RMSNorm


@dataclass(frozen=True)
class SlotSetDiagnostics:
    """Small representation record used by mechanism tests and analysis."""

    per_video_slots: torch.Tensor
    shared_slots: torch.Tensor
    attention: tuple[torch.Tensor, ...]
    auxiliary_loss: torch.Tensor


class PolicySlotSetFusion(torch.nn.Module):
    """Select centered video evidence around a stable per-slot mean backbone."""

    def __init__(self, *, width: int = 256) -> None:
        super().__init__()
        if width <= 0:
            raise WriterModelError("invalid v6 Slot-Set width")
        self.width = int(width)
        self.query_norm = RMSNorm(width)
        self.evidence_norm = RMSNorm(width)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)
        torch.nn.init.zeros_(self.output.weight)

    @staticmethod
    def _offsets(value: torch.Tensor, videos: int) -> tuple[int, ...]:
        if (
            value.device.type != "cpu"
            or value.dtype != torch.long
            or value.ndim != 1
        ):
            raise WriterModelError("Slot-Set condition offsets must be CPU long")
        offsets = tuple(int(item) for item in value.tolist())
        if (
            len(offsets) < 2
            or offsets[0] != 0
            or offsets[-1] != videos
            or any(right <= left for left, right in zip(offsets, offsets[1:]))
            or any(right - left > 4 for left, right in zip(offsets, offsets[1:]))
        ):
            raise WriterModelError("Slot-Set condition cardinality left K=1..4")
        return offsets

    def _one_condition(
        self, slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = slots.mean(dim=0)
        centered = slots - mean[None]
        query = self.query(self.query_norm(mean))
        key = self.key(self.evidence_norm(slots))
        logits = (key * query[None]).sum(dim=-1) * (self.width**-0.5)
        attention = torch.softmax(logits.to(torch.float32), dim=0).to(logits.dtype)
        residual = (attention[..., None] * centered).sum(dim=0)
        return mean + self.output(residual), attention

    def forward(
        self,
        per_video_slots: torch.Tensor,
        condition_video_offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, SlotSetDiagnostics]:
        if (
            per_video_slots.ndim != 3
            or per_video_slots.shape[-1] != self.width
            or per_video_slots.shape[0] <= 0
        ):
            raise WriterModelError("v6 per-video slots changed shape")
        offsets = self._offsets(condition_video_offsets, per_video_slots.shape[0])
        outputs = []
        attentions = []
        for left, right in zip(offsets, offsets[1:]):
            output, attention = self._one_condition(per_video_slots[left:right])
            outputs.append(output)
            attentions.append(attention)
        shared = torch.stack(outputs)
        auxiliary = shared.new_zeros(())
        return shared, SlotSetDiagnostics(
            per_video_slots=per_video_slots,
            shared_slots=shared,
            attention=tuple(attentions),
            auxiliary_loss=auxiliary,
        )
