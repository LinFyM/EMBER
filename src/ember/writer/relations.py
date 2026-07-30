"""Teacher-owned semantic and visible-motion Event construction for Loom."""

from __future__ import annotations

import math

import torch
from torch.utils.checkpoint import checkpoint

from ember.writer.temporal import RMSNorm, VariableEpisodeInputError
from ember.writer.temporal import _merge_heads, _split_heads


_EPSILON = 1e-6
_PATCH_SIDE = 16
_PATCH_TOKENS = _PATCH_SIDE * _PATCH_SIDE


def _rms(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.float32).square().mean(dim=-1).sqrt()


def _normalized_content(value: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
    normalized = value.to(torch.float32) / (energy[..., None] + _EPSILON)
    return normalized.to(value.dtype)


def _bounded_presence(energy: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return energy / (energy + tau.to(device=energy.device, dtype=energy.dtype))


def _fourier_2d(value: torch.Tensor, *, zero_at_origin: bool) -> torch.Tensor:
    if value.shape[-1] != 2:
        raise VariableEpisodeInputError("2-D Fourier input changed shape")
    frequencies = torch.arange(
        8, device=value.device, dtype=torch.float32
    ).exp2() * math.pi
    angles = value.to(torch.float32)[..., :, None] * frequencies
    sine = torch.sin(angles)
    cosine = torch.cos(angles)
    if zero_at_origin:
        cosine = cosine - 1.0
    return torch.cat((sine, cosine), dim=-1).flatten(-2).to(value.dtype)


def _grid_coordinates(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    axis = torch.linspace(
        -1.0, 1.0, _PATCH_SIDE, device=device, dtype=torch.float32
    )
    rows, columns = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((columns, rows), dim=-1).reshape(-1, 2).to(dtype)


class TaskSemanticRelation(torch.nn.Module):
    """Encode task-token midpoint and difference without a static bypass."""

    def __init__(self, *, width: int) -> None:
        super().__init__()
        if width <= 0:
            raise VariableEpisodeInputError("invalid semantic relation width")
        self.width = int(width)
        self.delta = torch.nn.Linear(width, width, bias=False)
        self.gate = torch.nn.Linear(width, width, bias=False)
        self.midpoint = torch.nn.Linear(width, width, bias=False)
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
            or task_evidence.shape[-1] != self.width
            or valid_frames.shape != task_evidence.shape[:2]
            or valid_frames.dtype != torch.bool
            or valid_task_tokens.shape != task_evidence.shape[::2]
            or valid_task_tokens.dtype != torch.bool
        ):
            raise VariableEpisodeInputError("invalid task-semantic trajectory")
        before = task_evidence[:, :-1]
        after = task_evidence[:, 1:]
        difference = after - before
        midpoint = (before + after) * 0.5
        relation = self.output(
            self.delta(difference)
            + torch.tanh(self.gate(difference)) * self.midpoint(midpoint)
        )
        active = (valid_frames[:, :-1] & valid_frames[:, 1:])[:, :, None]
        active = active & valid_task_tokens[:, None, :]
        return relation.masked_fill(~active[..., None], 0.0)


class VisualChangeRelation(torch.nn.Module):
    """Combine same-grid, matched-patch, and displacement changes."""

    DISPLACEMENT_WIDTH = 32

    def __init__(self, *, width: int) -> None:
        super().__init__()
        relation_width = 2 * width + self.DISPLACEMENT_WIDTH
        self.width = int(width)
        self.delta = torch.nn.Linear(relation_width, width, bias=False)
        self.gate = torch.nn.Linear(relation_width, width, bias=False)
        self.midpoint = torch.nn.Linear(width, width, bias=False)
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
            source.ndim != 4
            or source.shape != target.shape
            or source.shape != matched_target.shape
            or source.shape[-1] != self.width
            or displacement.shape != (*source.shape[:-1], 2)
            or active.shape != source.shape[:-1]
            or active.dtype != torch.bool
            or identical_interval.shape != source.shape[:2]
            or identical_interval.dtype != torch.bool
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
            + torch.tanh(self.gate(change)) * self.midpoint(midpoint)
        )
        keep = active & ~identical_interval[..., None]
        return relation.masked_fill(~keep[..., None], 0.0)


class BidirectionalPatchRelations(torch.nn.Module):
    """Match adjacent patch grids and retain both visible directions."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise VariableEpisodeInputError("invalid visual matcher dimensions")
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = width // heads
        self.content_norm = RMSNorm(width)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.coordinate_query = torch.nn.Linear(32, width, bias=False)
        self.coordinate_key = torch.nn.Linear(32, width, bias=False)
        self.relation = VisualChangeRelation(width=width)

    def _match(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, intervals, patches, width = source.shape
        flat_source = source.reshape(batch * intervals, patches, width)
        flat_target = target.reshape(batch * intervals, patches, width)
        coordinate_features = _fourier_2d(coordinates, zero_at_origin=False)
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
        weights = torch.softmax(logits.to(torch.float32), dim=-1)
        value_weights = weights.to(flat_target.dtype)

        matched = torch.matmul(
            value_weights, _split_heads(flat_target, self.heads)
        )
        matched = _merge_heads(matched).reshape(batch, intervals, patches, width)
        offsets = (coordinates[None, :] - coordinates[:, None]).to(
            value_weights.dtype
        )
        displacement = torch.einsum(
            "bhij,ijd->bhid", value_weights, offsets
        ).mean(dim=1)
        displacement = displacement.reshape(batch, intervals, patches, 2)
        normalized_entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
        normalized_entropy = normalized_entropy.mean(dim=1) / math.log(patches)
        normalized_entropy = normalized_entropy.reshape(batch, intervals, patches)
        return (
            matched,
            displacement,
            normalized_entropy,
            value_weights.reshape(batch, intervals, self.heads, patches, patches),
        )

    @staticmethod
    def _mutual_consistency(
        forward: torch.Tensor,
        backward: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forward_consistency = (forward * backward.transpose(-1, -2)).sum(-1)
        backward_consistency = (backward * forward.transpose(-1, -2)).sum(-1)
        return (
            forward_consistency.mean(dim=2),
            backward_consistency.mean(dim=2),
        )

    def _match_pair(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        forward = self._match(before, after, coordinates)
        backward = self._match(after, before, coordinates)
        mutual = self._mutual_consistency(forward[3], backward[3])
        return *forward[:3], *backward[:3], *mutual

    def forward(
        self,
        visual_evidence: torch.Tensor,
        valid_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if (
            visual_evidence.ndim != 4
            or visual_evidence.shape[1] < 2
            or visual_evidence.shape[2] != _PATCH_TOKENS
            or visual_evidence.shape[-1] != self.width
            or valid_frames.shape != visual_evidence.shape[:2]
            or valid_frames.dtype != torch.bool
        ):
            raise VariableEpisodeInputError("invalid visual evidence trajectory")
        before = visual_evidence[:, :-1]
        after = visual_evidence[:, 1:]
        patches = before.shape[2]
        coordinates = _grid_coordinates(device=before.device, dtype=before.dtype)
        matching = (
            checkpoint(
                self._match_pair,
                before,
                after,
                coordinates,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            if self.training and torch.is_grad_enabled()
            else self._match_pair(before, after, coordinates)
        )
        (matched_after, forward_displacement, forward_entropy,
         matched_before, backward_displacement, backward_entropy,
         forward_mutual, backward_mutual) = matching

        valid_intervals = valid_frames[:, :-1] & valid_frames[:, 1:]
        active = valid_intervals[:, :, None].expand(-1, -1, patches)
        identical = before.eq(after).all(dim=(-1, -2))
        relation_kwargs = {"active": active, "identical_interval": identical}
        forward_relation = self.relation(
            before, after, matched_after, forward_displacement, **relation_kwargs
        )
        backward_relation = self.relation(
            after, before, matched_before, backward_displacement, **relation_kwargs
        )
        return forward_relation, backward_relation, {
            "forward_matcher_entropy_normalized": forward_entropy.masked_fill(
                ~active, 0.0
            ),
            "backward_matcher_entropy_normalized": backward_entropy.masked_fill(
                ~active, 0.0
            ),
            "forward_mutual_consistency": forward_mutual.masked_fill(~active, 0.0),
            "backward_mutual_consistency": backward_mutual.masked_fill(~active, 0.0),
            "forward_displacement": forward_displacement.masked_fill(
                ~active[..., None], 0.0
            ),
            "backward_displacement": backward_displacement.masked_fill(
                ~active[..., None], 0.0
            ),
        }


class VisualTaskRelevance(torch.nn.Module):
    """Score normalized visual relations with stable task queries in Q/K only."""

    def __init__(self, *, width: int, heads: int, tau: float) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads or tau <= 0:
            raise VariableEpisodeInputError("invalid visual relevance dimensions")
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = width // heads
        self.task_norm = RMSNorm(width)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.register_buffer(
            "tau", torch.tensor(float(tau), dtype=torch.float32), persistent=True
        )

    def forward(
        self,
        task_queries: torch.Tensor,
        normalized_relation: torch.Tensor,
        valid_task_tokens: torch.Tensor,
        valid_relation: torch.Tensor,
    ) -> torch.Tensor:
        if (
            task_queries.ndim != 3
            or normalized_relation.ndim != 4
            or task_queries.shape[0] != normalized_relation.shape[0]
            or task_queries.shape[-1] != self.width
            or normalized_relation.shape[-1] != self.width
            or valid_task_tokens.shape != task_queries.shape[:2]
            or valid_task_tokens.dtype != torch.bool
            or valid_relation.shape != normalized_relation.shape[:-1]
            or valid_relation.dtype != torch.bool
            or not bool(valid_task_tokens.any(dim=1).all())
        ):
            raise VariableEpisodeInputError("invalid visual task relevance batch")
        batch, intervals, relations, _ = normalized_relation.shape
        query = _split_heads(self.query(self.task_norm(task_queries)), self.heads)
        flat_relation = normalized_relation.reshape(
            batch, intervals * relations, self.width
        )
        key = _split_heads(
            self.key(flat_relation), self.heads
        )
        logits = torch.einsum("bhnd,bhld->bhnl", key, query)
        logits = logits.to(torch.float32) / math.sqrt(self.head_width)
        logits = logits.masked_fill(
            ~valid_task_tokens[:, None, None, :],
            torch.finfo(logits.dtype).min,
        )
        task_counts = valid_task_tokens.sum(dim=1).to(torch.float32)
        neutral_centered = torch.logsumexp(logits, dim=-1)
        neutral_centered = neutral_centered - task_counts.log()[:, None, None]
        score = neutral_centered.mean(dim=1).reshape(batch, intervals, relations)
        relevance = torch.tanh(torch.relu(
            score / self.tau.to(device=score.device)
        ))
        return relevance.masked_fill(~valid_relation, 0.0)


class RelationalEventTokenizer(torch.nn.Module):
    """Build three deterministic evidence floors and five learned Events."""

    BACKBONE_EVENTS = 3
    LEARNED_EVENTS = 5
    EVENT_COUNT = BACKBONE_EVENTS + LEARNED_EVENTS

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        initialization_seed: int,
        type_tau: float,
    ) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads or type_tau <= 0:
            raise VariableEpisodeInputError("invalid Event tokenizer dimensions")
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = width // heads
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        event_identity = torch.empty(self.LEARNED_EVENTS, width)
        event_identity.normal_(mean=0.0, std=0.02, generator=generator)
        type_identity = torch.empty(self.BACKBONE_EVENTS, width)
        type_identity.normal_(mean=0.0, std=0.02, generator=generator)
        self.event_identity = torch.nn.Parameter(event_identity)
        self.type_identity = torch.nn.Parameter(type_identity)
        self.relation_norm = RMSNorm(width)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.position_key = torch.nn.Linear(32, width, bias=False)
        self.value = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)
        self.register_buffer(
            "type_tau",
            torch.tensor(type_tau, dtype=torch.float32),
            persistent=True,
        )

    def _backbone(
        self,
        normalized_relation: torch.Tensor,
        confidence: torch.Tensor,
        valid_relation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        confidence = confidence.masked_fill(~valid_relation, 0.0)
        counts = valid_relation.sum(dim=-1).clamp_min(1).to(torch.float32)
        mean_confidence = confidence.sum(dim=-1) / counts
        type_confidence = mean_confidence / (
            mean_confidence
            + self.type_tau.to(
                device=mean_confidence.device,
                dtype=mean_confidence.dtype,
            )
        )
        weights = confidence.to(normalized_relation.dtype)[..., None]
        numerator = (weights * normalized_relation).sum(dim=-2)
        denominator = confidence.sum(dim=-1, keepdim=True) + _EPSILON
        aggregate = numerator / denominator.to(normalized_relation.dtype)
        event = aggregate * type_confidence.to(aggregate.dtype)[..., None]
        return event, type_confidence

    @staticmethod
    def _semantic_coordinates(
        tokens: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if tokens == 1:
            axis = torch.zeros(1, device=device, dtype=torch.float32)
        else:
            axis = torch.linspace(
                -1.0, 1.0, tokens, device=device, dtype=torch.float32
            )
        return torch.stack((axis, torch.zeros_like(axis)), dim=-1).to(dtype)

    def _relation_key_identity(
        self,
        semantic_tokens: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        semantic_coordinates = self._semantic_coordinates(
            semantic_tokens, device=device, dtype=dtype
        )
        visual_coordinates = _grid_coordinates(device=device, dtype=dtype)
        coordinates = torch.cat(
            (semantic_coordinates, visual_coordinates, visual_coordinates), dim=0
        )
        type_ids = torch.cat(
            (
                torch.zeros(semantic_tokens, dtype=torch.long, device=device),
                torch.ones(_PATCH_TOKENS, dtype=torch.long, device=device),
                torch.full((_PATCH_TOKENS,), 2, dtype=torch.long, device=device),
            )
        )
        position = self.position_key(_fourier_2d(
            coordinates, zero_at_origin=False
        ))
        return position + self.type_identity[type_ids]

    def _learned_events(
        self,
        normalized_relation: torch.Tensor,
        confidence: torch.Tensor,
        valid_relation: torch.Tensor,
        valid_intervals: torch.Tensor,
        semantic_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, intervals, relations, width = normalized_relation.shape
        flat_relation = normalized_relation.reshape(
            batch * intervals, relations, width
        )
        flat_confidence = confidence.reshape(batch * intervals, relations)
        flat_valid = valid_relation.reshape(batch * intervals, relations)
        safe_valid = flat_valid.clone()
        invalid_intervals = ~valid_intervals.reshape(-1)
        safe_valid[invalid_intervals, 0] = True

        query_identity = self.event_identity[None].expand(
            batch * intervals, -1, -1
        )
        query = _split_heads(self.query(query_identity), self.heads)
        key_identity = self._relation_key_identity(
            semantic_tokens,
            device=normalized_relation.device,
            dtype=normalized_relation.dtype,
        )
        key = _split_heads(
            self.key(self.relation_norm(flat_relation)) + key_identity[None],
            self.heads,
        )
        value = _split_heads(self.value(flat_relation), self.heads)
        logits = torch.matmul(query, key.transpose(-1, -2))
        logits = logits.to(torch.float32) / math.sqrt(self.head_width)
        confidence_prior = flat_confidence.clamp_min(_EPSILON).log()
        logits = logits + confidence_prior[:, None, None, :]
        logits = logits.masked_fill(
            ~safe_valid[:, None, None, :],
            torch.finfo(logits.dtype).min,
        )
        weights = torch.softmax(logits, dim=-1)
        attended = torch.matmul(weights.to(value.dtype), value)
        learned = self.output(_merge_heads(attended))
        confidence_heads = (weights * flat_confidence[:, None, None, :]).sum(-1)
        learned_confidence = confidence_heads.mean(dim=1)
        learned = learned * learned_confidence.to(learned.dtype)[..., None]

        valid_counts = flat_valid.sum(dim=-1).clamp_min(2).to(torch.float32)
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
        entropy = entropy / valid_counts.log()[:, None, None]
        entropy = entropy.mean(dim=1)

        event_shape = (batch, intervals, self.LEARNED_EVENTS)
        learned = learned.reshape(*event_shape, width)
        learned_confidence = learned_confidence.reshape(event_shape)
        entropy = entropy.reshape(event_shape)
        active = valid_intervals[:, :, None]
        return (
            learned.masked_fill(~active[..., None], 0.0),
            learned_confidence.masked_fill(~active, 0.0),
            entropy.masked_fill(~active, 0.0),
        )

    def forward(
        self,
        semantic: torch.Tensor,
        visual_forward: torch.Tensor,
        visual_backward: torch.Tensor,
        semantic_confidence: torch.Tensor,
        visual_forward_confidence: torch.Tensor,
        visual_backward_confidence: torch.Tensor,
        valid_intervals: torch.Tensor,
        valid_task_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if (
            semantic.ndim != 4
            or visual_forward.ndim != 4
            or visual_backward.shape != visual_forward.shape
            or semantic.shape[:2] != visual_forward.shape[:2]
            or semantic.shape[-1] != self.width
            or visual_forward.shape[2:] != (_PATCH_TOKENS, self.width)
            or semantic_confidence.shape != semantic.shape[:-1]
            or visual_forward_confidence.shape != visual_forward.shape[:-1]
            or visual_backward_confidence.shape != visual_backward.shape[:-1]
            or valid_intervals.shape != semantic.shape[:2]
            or valid_intervals.dtype != torch.bool
            or valid_task_tokens.shape != semantic.shape[::2]
            or valid_task_tokens.dtype != torch.bool
        ):
            raise VariableEpisodeInputError("invalid relation Event batch")
        semantic_valid = valid_intervals[:, :, None]
        semantic_valid = semantic_valid & valid_task_tokens[:, None, :]
        visual_valid = valid_intervals[:, :, None].expand(
            -1, -1, _PATCH_TOKENS
        )
        backbone_rows = (
            self._backbone(semantic, semantic_confidence, semantic_valid),
            self._backbone(visual_forward, visual_forward_confidence, visual_valid),
            self._backbone(visual_backward, visual_backward_confidence, visual_valid),
        )
        backbone_events = torch.stack(
            tuple(row[0] for row in backbone_rows), dim=2
        )
        backbone_confidence = torch.stack(
            tuple(row[1] for row in backbone_rows), dim=2
        )

        relation = torch.cat((semantic, visual_forward, visual_backward), dim=2)
        confidence = torch.cat(
            (
                semantic_confidence,
                visual_forward_confidence,
                visual_backward_confidence,
            ),
            dim=2,
        )
        valid_relation = torch.cat(
            (semantic_valid, visual_valid, visual_valid), dim=2
        )
        learned_events, learned_confidence, learned_entropy = self._learned_events(
            relation,
            confidence,
            valid_relation,
            valid_intervals,
            semantic.shape[2],
        )
        events = torch.cat((backbone_events, learned_events), dim=2)
        initial_confidence = torch.cat(
            (backbone_confidence, learned_confidence), dim=2
        )
        return events, initial_confidence, {
            "backbone_events": backbone_events,
            "learned_events": learned_events,
            "backbone_confidence": backbone_confidence,
            "learned_event_confidence": learned_confidence,
            "learned_event_entropy_normalized": learned_entropy,
        }


class TeacherEventBuilder(torch.nn.Module):
    """Convert task-grounded frame evidence into eight teacher-owned Events."""

    EVENT_COUNT = RelationalEventTokenizer.EVENT_COUNT

    def __init__(
        self,
        width: int,
        heads: int,
        initialization_seed: int,
        change_tau: float = 0.1,
        relevance_tau: float = 1.0,
        type_tau: float = 0.1,
    ) -> None:
        super().__init__()
        if (
            min(width, heads) <= 0
            or width % heads
            or min(change_tau, relevance_tau, type_tau) <= 0
        ):
            raise VariableEpisodeInputError("invalid Teacher Event dimensions")
        self.width = int(width)
        self.semantic_relation = TaskSemanticRelation(width=width)
        self.visual_relation = BidirectionalPatchRelations(width=width, heads=heads)
        self.visual_relevance = VisualTaskRelevance(
            width=width, heads=heads, tau=relevance_tau
        )
        self.event_tokenizer = RelationalEventTokenizer(
            width=width,
            heads=heads,
            initialization_seed=initialization_seed,
            type_tau=type_tau,
        )
        self.register_buffer(
            "change_tau",
            torch.tensor(change_tau, dtype=torch.float32),
            persistent=True,
        )

    def _validate_inputs(
        self,
        task_queries: torch.Tensor,
        task_evidence: torch.Tensor,
        visual_evidence: torch.Tensor,
        valid_frames: torch.Tensor,
        valid_task_tokens: torch.Tensor,
    ) -> None:
        if (
            task_queries.ndim != 3
            or task_evidence.ndim != 4
            or visual_evidence.ndim != 4
            or task_queries.shape[0] != task_evidence.shape[0]
            or task_queries.shape[1:] != task_evidence.shape[2:]
            or task_queries.shape[0] != visual_evidence.shape[0]
            or task_evidence.shape[:2] != visual_evidence.shape[:2]
            or task_queries.shape[-1] != self.width
            or visual_evidence.shape[2:] != (_PATCH_TOKENS, self.width)
            or task_evidence.shape[1] < 2
            or valid_frames.shape != task_evidence.shape[:2]
            or valid_frames.dtype != torch.bool
            or valid_task_tokens.shape != task_queries.shape[:2]
            or valid_task_tokens.dtype != torch.bool
            or not bool(valid_frames.any(dim=1).all())
            or not bool(valid_task_tokens.any(dim=1).all())
        ):
            raise VariableEpisodeInputError("invalid Teacher Event input batch")

    @staticmethod
    def _masked_confidence(
        presence: torch.Tensor,
        valid: torch.Tensor,
        *factors: torch.Tensor,
    ) -> torch.Tensor:
        confidence = presence
        for factor in factors:
            confidence = confidence * factor.to(confidence.dtype)
        return confidence.clamp(0.0, 1.0).masked_fill(~valid, 0.0)

    def _content_and_confidence(
        self,
        task_queries: torch.Tensor,
        semantic: torch.Tensor,
        visual_forward: torch.Tensor,
        visual_backward: torch.Tensor,
        matching: dict[str, torch.Tensor],
        semantic_valid: torch.Tensor,
        visual_valid: torch.Tensor,
        valid_task_tokens: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        relations = (semantic, visual_forward, visual_backward)
        energies = tuple(_rms(value) for value in relations)
        content = tuple(
            _normalized_content(value, energy)
            for value, energy in zip(relations, energies, strict=True)
        )
        presence = tuple(
            _bounded_presence(energy, self.change_tau) for energy in energies
        )
        relevance = (
            self.visual_relevance(
                task_queries, content[1], valid_task_tokens, visual_valid
            ),
            self.visual_relevance(
                task_queries, content[2], valid_task_tokens, visual_valid
            ),
        )
        nonuniform = (
            (1.0 - matching["forward_matcher_entropy_normalized"]).clamp(0.0, 1.0),
            (1.0 - matching["backward_matcher_entropy_normalized"]).clamp(0.0, 1.0),
        )
        confidence = (
            self._masked_confidence(presence[0], semantic_valid),
            self._masked_confidence(
                presence[1],
                visual_valid,
                relevance[0],
                matching["forward_mutual_consistency"],
                nonuniform[0],
            ),
            self._masked_confidence(
                presence[2],
                visual_valid,
                relevance[1],
                matching["backward_mutual_consistency"],
                nonuniform[1],
            ),
        )
        return content, confidence, {
            "semantic_relation_rms": energies[0],
            "visual_forward_relation_rms": energies[1],
            "visual_backward_relation_rms": energies[2],
            "semantic_presence": presence[0].masked_fill(~semantic_valid, 0.0),
            "visual_forward_presence": presence[1].masked_fill(~visual_valid, 0.0),
            "visual_backward_presence": presence[2].masked_fill(~visual_valid, 0.0),
            "visual_forward_task_relevance": relevance[0],
            "visual_backward_task_relevance": relevance[1],
            "semantic_confidence": confidence[0],
            "visual_forward_confidence": confidence[1],
            "visual_backward_confidence": confidence[2],
        }

    def forward(
        self,
        task_queries: torch.Tensor,
        task_evidence: torch.Tensor,
        visual_evidence: torch.Tensor,
        valid_frames: torch.Tensor,
        valid_task_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_inputs(
            task_queries, task_evidence, visual_evidence, valid_frames,
            valid_task_tokens,
        )
        valid_intervals = valid_frames[:, :-1] & valid_frames[:, 1:]
        semantic_valid = valid_intervals[:, :, None]
        semantic_valid = semantic_valid & valid_task_tokens[:, None, :]
        visual_valid = valid_intervals[:, :, None].expand(
            -1, -1, _PATCH_TOKENS
        )

        semantic = self.semantic_relation(
            task_evidence, valid_frames, valid_task_tokens
        )
        visual_forward, visual_backward, matching = self.visual_relation(
            visual_evidence, valid_frames
        )
        content, confidence, relation_diagnostics = self._content_and_confidence(
            task_queries,
            semantic,
            visual_forward,
            visual_backward,
            matching,
            semantic_valid,
            visual_valid,
            valid_task_tokens,
        )
        events, initial_confidence, event_diagnostics = self.event_tokenizer(
            *content,
            *confidence,
            valid_intervals,
            valid_task_tokens,
        )
        diagnostics = {
            **matching,
            **event_diagnostics,
            **relation_diagnostics,
            "valid_intervals": valid_intervals,
        }
        return events, initial_confidence, diagnostics
