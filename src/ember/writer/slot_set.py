"""Permutation-invariant fusion of language-aligned per-video Semantic Core."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ember.writer.errors import WriterModelError
from ember.writer.temporal import RMSNorm


@dataclass(frozen=True)
class SemanticCoreSetDiagnostics:
    """Shared Core corrections and video-set attention for mechanism analysis."""

    per_video_core: torch.Tensor
    corrected_per_video_core: torch.Tensor
    shared_corrections: torch.Tensor
    attention: tuple[torch.Tensor, ...]
    auxiliary_loss: torch.Tensor


class SemanticCoreSetFusion(torch.nn.Module):
    """Write one selected centered video-set residual into every aligned Core."""

    def __init__(self, *, width: int = 256) -> None:
        super().__init__()
        if width <= 0:
            raise WriterModelError("invalid v6 Semantic-Core Set width")
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
            raise WriterModelError("Semantic-Core Set offsets must be CPU long")
        offsets = tuple(int(item) for item in value.tolist())
        if (
            len(offsets) < 2
            or offsets[0] != 0
            or offsets[-1] != videos
            or any(right <= left for left, right in zip(offsets, offsets[1:]))
            or any(right - left > 4 for left, right in zip(offsets, offsets[1:]))
        ):
            raise WriterModelError("Semantic-Core Set cardinality left K=1..4")
        return offsets

    def _one_condition(
        self,
        core: torch.Tensor,
        valid_core: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = core.mean(dim=0)
        centered = core - mean[None]
        query = self.query(self.query_norm(mean))
        key = self.key(self.evidence_norm(core))
        logits = (key * query[None]).sum(dim=-1) * (self.width**-0.5)
        attention = torch.softmax(logits.to(torch.float32), dim=0).to(logits.dtype)
        residual = (attention[..., None] * centered).sum(dim=0)
        valid = valid_core[0]
        correction = self.output(residual).masked_fill(~valid[..., None], 0.0)
        attention = attention.masked_fill(~valid[None], 0.0)
        return core + correction[None], correction, attention

    def forward(
        self,
        per_video_core: torch.Tensor,
        valid_core: torch.Tensor,
        condition_video_offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, SemanticCoreSetDiagnostics]:
        if (
            per_video_core.ndim != 3
            or per_video_core.shape[-1] != self.width
            or per_video_core.shape[0] <= 0
            or valid_core.shape != per_video_core.shape[:2]
            or valid_core.dtype != torch.bool
        ):
            raise WriterModelError("v6 per-video Semantic Core changed shape")
        offsets = self._offsets(condition_video_offsets, per_video_core.shape[0])
        corrected = []
        corrections = []
        attentions = []
        for left, right in zip(offsets, offsets[1:]):
            output, correction, attention = self._one_condition(
                per_video_core[left:right], valid_core[left:right]
            )
            corrected.append(output)
            corrections.append(correction)
            attentions.append(attention)
        corrected_core = torch.cat(corrected, dim=0)
        shared_corrections = torch.stack(corrections)
        auxiliary = corrected_core.new_zeros(())
        return corrected_core, SemanticCoreSetDiagnostics(
            per_video_core=per_video_core,
            corrected_per_video_core=corrected_core,
            shared_corrections=shared_corrections,
            attention=tuple(attentions),
            auxiliary_loss=auxiliary,
        )
