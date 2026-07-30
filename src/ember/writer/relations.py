"""Task-semantic and visible-motion relations for EMBER Loom."""

from __future__ import annotations

import math

import torch

from ember.writer.temporal import (
    RMSNorm,
    VariableEpisodeInputError,
    _merge_heads,
    _split_heads,
)


def _rms_energy(value: torch.Tensor) -> torch.Tensor:
    """Return exact-zero-preserving RMS energy over the content axis."""

    return value.to(torch.float32).square().mean(dim=-1).sqrt().to(value.dtype)


def _fourier_2d(value: torch.Tensor, *, zero_at_origin: bool) -> torch.Tensor:
    """Encode normalized 2-D coordinates into 32 deterministic features."""

    if value.shape[-1] != 2:
        raise VariableEpisodeInputError("2-D Fourier input changed shape")
    frequencies = (
        torch.arange(8, device=value.device, dtype=torch.float32).exp2()
        * math.pi
    )
    angles = value.to(torch.float32)[..., :, None] * frequencies
    sine = torch.sin(angles)
    cosine = torch.cos(angles)
    if zero_at_origin:
        cosine = cosine - 1.0
    return torch.cat((sine, cosine), dim=-1).flatten(-2).to(value.dtype)


class TaskSemanticRelation(torch.nn.Module):
    """Express task-token state changes without an absolute-content bypass."""

    def __init__(self, *, width: int) -> None:
        super().__init__()
        if width <= 0:
            raise VariableEpisodeInputError("invalid semantic relation width")
        self.delta = torch.nn.Linear(width, width, bias=False)
        self.gate = torch.nn.Linear(width, width, bias=False)
        self.context = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)

    def forward(
        self,
        task_evidence: torch.Tensor,
        valid_frames: torch.Tensor,
        valid_task_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if (
            task_evidence.ndim != 4
            or task_evidence.shape[1] < 2
            or valid_frames.shape != task_evidence.shape[:2]
            or valid_frames.dtype != torch.bool
            or valid_task_tokens.shape
            != (task_evidence.shape[0], task_evidence.shape[2])
            or valid_task_tokens.dtype != torch.bool
        ):
            raise VariableEpisodeInputError("invalid task-semantic trajectory")
        before = task_evidence[:, :-1]
        after = task_evidence[:, 1:]
        difference = after - before
        midpoint = (before + after) * 0.5
        relation = self.output(
            self.delta(difference)
            + torch.tanh(self.gate(difference)) * self.context(midpoint)
        )
        active = (
            (valid_frames[:, :-1] & valid_frames[:, 1:])[:, :, None]
            & valid_task_tokens[:, None, :]
        )
        return relation.masked_fill(~active[..., None], 0.0)


class VisualChangeRelation(torch.nn.Module):
    """Map same-grid, matched, and displacement changes into one relation."""

    DISPLACEMENT_WIDTH = 32

    def __init__(self, *, width: int) -> None:
        super().__init__()
        relation_width = 2 * width + self.DISPLACEMENT_WIDTH
        self.delta = torch.nn.Linear(relation_width, width, bias=False)
        self.gate = torch.nn.Linear(relation_width, width, bias=False)
        self.context = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        matched_target: torch.Tensor,
        displacement: torch.Tensor,
        *,
        active: torch.Tensor,
        identical_interval: torch.Tensor,
    ) -> torch.Tensor:
        if (
            source.shape != target.shape
            or matched_target.shape != source.shape
            or displacement.shape != (*source.shape[:-1], 2)
            or active.shape != source.shape[:2]
            or identical_interval.shape != source.shape[:2]
        ):
            raise VariableEpisodeInputError("invalid visual relation batch")
        change = torch.cat(
            (
                target - source,
                matched_target - source,
                _fourier_2d(displacement, zero_at_origin=True),
            ),
            dim=-1,
        )
        midpoint = (source + matched_target) * 0.5
        relation = self.output(
            self.delta(change)
            + torch.tanh(self.gate(change)) * self.context(midpoint)
        )
        keep = active & ~identical_interval
        return relation.masked_fill(~keep[..., None], 0.0)


class BidirectionalVisualRelation(torch.nn.Module):
    """Find cross-frame correspondence, then retain forward/backward changes."""

    PATCH_TOKENS = 256

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        if (
            min(width, heads) <= 0
            or width % heads
            or self.PATCH_TOKENS != 16 * 16
        ):
            raise VariableEpisodeInputError("invalid visual matcher dimensions")
        self.heads = int(heads)
        self.head_width = width // heads
        self.content_norm = RMSNorm(width)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.coordinate_query = torch.nn.Linear(32, width, bias=False)
        self.coordinate_key = torch.nn.Linear(32, width, bias=False)
        self.relation = VisualChangeRelation(width=width)

    @staticmethod
    def _coordinates(
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        axis = torch.linspace(-1.0, 1.0, 16, device=device, dtype=torch.float32)
        rows, columns = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack((columns, rows), dim=-1).reshape(-1, 2).to(dtype)

    def _match(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, intervals, patches, width = source.shape
        flat_source = source.reshape(batch * intervals, patches, width)
        flat_target = target.reshape(batch * intervals, patches, width)
        coordinate_features = _fourier_2d(
            coordinates,
            zero_at_origin=False,
        )
        query_identity = self.coordinate_query(coordinate_features)[None]
        key_identity = self.coordinate_key(coordinate_features)[None]
        query = _split_heads(
            self.query(self.content_norm(flat_source)) + query_identity,
            self.heads,
        )
        key = _split_heads(
            self.key(self.content_norm(flat_target)) + key_identity,
            self.heads,
        )
        logits = torch.matmul(query, key.transpose(-1, -2))
        logits = logits / math.sqrt(self.head_width)
        weights = torch.softmax(logits.to(torch.float32), dim=-1).to(logits.dtype)

        target_heads = _split_heads(flat_target, self.heads)
        matched = torch.matmul(weights, target_heads)
        matched = _merge_heads(matched).reshape(
            batch,
            intervals,
            patches,
            width,
        )
        offsets = (
            coordinates[None, :] - coordinates[:, None]
        ).to(weights.dtype)
        displacement = torch.einsum(
            "bhij,ijd->bhid",
            weights,
            offsets,
        ).mean(dim=1)
        displacement = displacement.reshape(batch, intervals, patches, 2)
        entropy = -(
            weights.to(torch.float32)
            * weights.to(torch.float32).clamp_min(1e-12).log()
        ).sum(dim=-1)
        entropy = entropy.mean(dim=(1, 2)).reshape(batch, intervals)
        return matched, displacement, entropy

    def forward(
        self,
        visual_evidence: torch.Tensor,
        valid_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if (
            visual_evidence.ndim != 4
            or visual_evidence.shape[1] < 2
            or visual_evidence.shape[2] != self.PATCH_TOKENS
            or valid_frames.shape != visual_evidence.shape[:2]
            or valid_frames.dtype != torch.bool
        ):
            raise VariableEpisodeInputError("invalid visual evidence trajectory")
        before = visual_evidence[:, :-1]
        after = visual_evidence[:, 1:]
        batch, intervals, patches, _ = before.shape
        coordinates = self._coordinates(
            device=before.device,
            dtype=before.dtype,
        )
        matched_after, forward_displacement, forward_entropy = self._match(
            before,
            after,
            coordinates,
        )
        matched_before, backward_displacement, backward_entropy = self._match(
            after,
            before,
            coordinates,
        )
        active_interval = valid_frames[:, :-1] & valid_frames[:, 1:]
        active = active_interval[:, :, None].expand(-1, -1, patches)
        identical = before.eq(after).all(dim=(-1, -2))
        identical = identical[:, :, None].expand(-1, -1, patches)
        forward = self.relation(
            before,
            after,
            matched_after,
            forward_displacement,
            active=active,
            identical_interval=identical,
        )
        backward = self.relation(
            after,
            before,
            matched_before,
            backward_displacement,
            active=active,
            identical_interval=identical,
        )
        return forward, backward, {
            "forward_matcher_entropy": forward_entropy.masked_fill(
                ~active_interval,
                0.0,
            ),
            "backward_matcher_entropy": backward_entropy.masked_fill(
                ~active_interval,
                0.0,
            ),
            "forward_displacement": forward_displacement.masked_fill(
                ~active[..., None],
                0.0,
            ),
            "backward_displacement": backward_displacement.masked_fill(
                ~active[..., None],
                0.0,
            ),
        }


class EvidencePreservingEventTokenizer(torch.nn.Module):
    """Keep three deterministic evidence floors plus five learned events."""

    BACKBONE_EVENTS = 3
    LEARNED_EVENTS = 5
    EVENT_COUNT = BACKBONE_EVENTS + LEARNED_EVENTS

    def __init__(self, *, width: int, heads: int, initialization_seed: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise VariableEpisodeInputError("invalid Event tokenizer dimensions")
        self.heads = int(heads)
        self.head_width = width // heads
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        queries = torch.empty(self.LEARNED_EVENTS, width)
        queries.normal_(mean=0.0, std=0.02, generator=generator)
        types = torch.empty(self.BACKBONE_EVENTS, width)
        types.normal_(mean=0.0, std=0.02, generator=generator)
        self.event_identity = torch.nn.Parameter(queries)
        self.type_identity = torch.nn.Parameter(types)
        self.relation_norm = RMSNorm(width)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.value = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)

    @staticmethod
    def _backbone(
        relation: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        energy = _rms_energy(relation).masked_fill(~valid_tokens, 0.0)
        numerator = (energy[..., None] * relation).sum(dim=-2)
        denominator = energy.sum(dim=-1, keepdim=True) + 1e-6
        return numerator / denominator

    def forward(
        self,
        semantic: torch.Tensor,
        visual_forward: torch.Tensor,
        visual_backward: torch.Tensor,
        valid_intervals: torch.Tensor,
        valid_task_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if (
            semantic.ndim != 4
            or visual_forward.ndim != 4
            or visual_backward.shape != visual_forward.shape
            or semantic.shape[:2] != visual_forward.shape[:2]
            or semantic.shape[-1] != visual_forward.shape[-1]
            or valid_intervals.shape != semantic.shape[:2]
            or valid_intervals.dtype != torch.bool
            or valid_task_tokens.shape
            != (semantic.shape[0], semantic.shape[2])
            or valid_task_tokens.dtype != torch.bool
        ):
            raise VariableEpisodeInputError("invalid relation Event batch")
        batch, intervals, _, width = semantic.shape
        patches = visual_forward.shape[2]
        valid_semantic = (
            valid_intervals[:, :, None] & valid_task_tokens[:, None, :]
        )
        valid_visual = valid_intervals[:, :, None].expand(-1, -1, patches)
        backbone = torch.stack(
            (
                self._backbone(semantic, valid_semantic),
                self._backbone(visual_forward, valid_visual),
                self._backbone(visual_backward, valid_visual),
            ),
            dim=2,
        )

        relation = torch.cat((semantic, visual_forward, visual_backward), dim=2)
        valid_relation = torch.cat(
            (valid_semantic, valid_visual, valid_visual),
            dim=2,
        )
        type_ids = torch.cat(
            (
                torch.zeros(semantic.shape[2], dtype=torch.long, device=relation.device),
                torch.ones(patches, dtype=torch.long, device=relation.device),
                torch.full(
                    (patches,),
                    2,
                    dtype=torch.long,
                    device=relation.device,
                ),
            )
        )
        token_count = relation.shape[2]
        flat_relation = relation.reshape(batch * intervals, token_count, width)
        flat_valid = valid_relation.reshape(batch * intervals, token_count)
        safe_valid = flat_valid.clone()
        invalid_interval = ~valid_intervals.reshape(-1)
        safe_valid[invalid_interval, 0] = True

        query_identity = self.event_identity[None].expand(
            batch * intervals,
            -1,
            -1,
        )
        query = _split_heads(self.query(query_identity), self.heads)
        key_input = self.relation_norm(flat_relation)
        key_input = key_input + self.type_identity[type_ids][None]
        key = _split_heads(self.key(key_input), self.heads)
        value = _split_heads(self.value(flat_relation), self.heads)
        logits = torch.matmul(query, key.transpose(-1, -2))
        logits = logits / math.sqrt(self.head_width)
        energy = _rms_energy(flat_relation)
        logits = logits + energy[:, None, None, :].clamp_min(1e-12).log()
        logits = logits.masked_fill(
            ~safe_valid[:, None, None, :],
            torch.finfo(logits.dtype).min,
        )
        weights = torch.softmax(logits.to(torch.float32), dim=-1).to(logits.dtype)
        learned = self.output(_merge_heads(torch.matmul(weights, value)))
        learned = learned.reshape(
            batch,
            intervals,
            self.LEARNED_EVENTS,
            width,
        ).masked_fill(~valid_intervals[:, :, None, None], 0.0)
        events = torch.cat((backbone, learned), dim=2)

        entropy = -(
            weights.to(torch.float32)
            * weights.to(torch.float32).clamp_min(1e-12).log()
        ).sum(dim=-1).mean(dim=1)
        entropy = entropy.reshape(
            batch,
            intervals,
            self.LEARNED_EVENTS,
        ).masked_fill(~valid_intervals[:, :, None], 0.0)
        return events, {
            "backbone_events": backbone,
            "learned_events": learned,
            "learned_event_entropy": entropy,
            "semantic_relation_energy": _rms_energy(semantic),
            "visual_forward_energy": _rms_energy(visual_forward),
            "visual_backward_energy": _rms_energy(visual_backward),
        }
