"""Differentiable B0 anchor construction and native replay-query solve."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.checkpoint import checkpoint

from ember.ecp.bank_conditioning.anchor import (
    AnchorProgramState,
    ProgramNativeAnchorScorer,
)
from ember.ecp.bank_conditioning.functional_polar import (
    StreamingCenteredAnchor,
    normalize_replay_queries,
)
from ember.ecp.bank_conditioning.operator import (
    BankStatistics,
    SpectralBankQuery,
    batched_spectral_bank_query,
)
from ember.ecp.bank_conditioning.whitening import FeatureWhitener
from ember.ecp.contracts import ACTION_HORIZON, TargetOwner
from ember.ecp.native_factors import (
    G1_PROBE_COUNT,
    G1_RESIDUAL_RANK,
    OUTPUT_BANK_TYPES,
    NativeFactorError,
    NativeOutputBankState,
    native_output_group_count,
)


@dataclass(frozen=True)
class ReplayBankPlan:
    """Native B1 queries and diagnostics produced by the B0 solve."""

    input_queries: tuple[torch.Tensor, ...]
    output_queries: tuple[tuple[torch.Tensor, ...], ...]
    frame_measure: torch.Tensor
    group_gains: torch.Tensor
    solve_metrics: torch.Tensor
    conditioning_metrics: torch.Tensor


@dataclass
class _AnchorStream:
    input_accumulators: list[StreamingCenteredAnchor]
    output_accumulators: list[tuple[StreamingCenteredAnchor, ...]]
    boundaries: list[NativeOutputBankState]


def candidate_mass(frame_mass: torch.Tensor, *, output: bool) -> torch.Tensor:
    """Lift a frame measure onto native probe/horizon/type candidate axes."""

    shape = (*frame_mass.shape, G1_PROBE_COUNT, ACTION_HORIZON)
    if output:
        shape = (*shape, len(OUTPUT_BANK_TYPES))
    leading = (*frame_mass.shape, *((1,) * (len(shape) - frame_mass.ndim)))
    return frame_mass.reshape(leading).expand(shape)


def input_event_keys(
    scorer: ProgramNativeAnchorScorer,
    value: torch.Tensor,
    metadata: torch.Tensor,
    *,
    target: int,
    whitener: FeatureWhitener,
) -> torch.Tensor:
    key = scorer.input_keys(value, metadata, target=target)
    return scorer.input_projected_keys(whitener.whiten(key), target=target)


def output_event_keys(
    scorer: ProgramNativeAnchorScorer,
    value: torch.Tensor,
    metadata: torch.Tensor,
    *,
    target: int,
    whiteners: tuple[FeatureWhitener, ...],
) -> torch.Tensor:
    key = scorer.output_keys(value, metadata, target=target)
    if len(whiteners) != key.shape[0]:
        raise NativeFactorError("compiler output feature groups changed")
    whitened = torch.stack(
        tuple(
            whitener.whiten(key[group])
            for group, whitener in enumerate(whiteners)
        ),
        dim=1,
    )
    return scorer.output_projected_keys(whitened, target=target)


def _input_anchor_compatibility(
    scorer: ProgramNativeAnchorScorer,
    value: torch.Tensor,
    metadata: torch.Tensor,
    query: torch.Tensor,
    weights: torch.Tensor,
    ratio: torch.Tensor,
    target: int,
    whitener: FeatureWhitener,
) -> torch.Tensor:
    keys = input_event_keys(
        scorer, value, metadata, target=target, whitener=whitener
    )
    event = scorer.input_compatibility(query, keys, target=target)
    return torch.einsum("re,et,rebtph->rbtph", weights, ratio, event)


def _output_anchor_compatibility(
    scorer: ProgramNativeAnchorScorer,
    value: torch.Tensor,
    metadata: torch.Tensor,
    query: torch.Tensor,
    weights: torch.Tensor,
    ratio: torch.Tensor,
    target: int,
    whiteners: tuple[FeatureWhitener, ...],
) -> torch.Tensor:
    keys = output_event_keys(
        scorer, value, metadata, target=target, whiteners=whiteners
    )
    event = scorer.output_compatibility(query, keys, target=target)
    return torch.einsum("re,et,grebtphu->grbtphu", weights, ratio, event)


def _solve_statistics(
    entries: Sequence[tuple[tuple[Any, ...], BankStatistics]],
    *,
    relative_floor: float,
) -> dict[tuple[Any, ...], SpectralBankQuery]:
    grouped: dict[
        tuple[int, tuple[int, ...]],
        list[tuple[tuple[Any, ...], BankStatistics]],
    ] = defaultdict(list)
    for key, statistics in entries:
        grouped[
            (statistics.mean.numel(), tuple(statistics.anchor.shape[:-1]))
        ].append((key, statistics))
    solved: dict[tuple[Any, ...], SpectralBankQuery] = {}
    for rows in grouped.values():
        queries = batched_spectral_bank_query(
            tuple(row[1] for row in rows),
            relative_eigenvalue_floor=relative_floor,
        )
        solved.update(
            (row[0], query) for row, query in zip(rows, queries, strict=True)
        )
    return solved


def _new_anchor_stream(video: Any, functional: Any) -> _AnchorStream:
    return _AnchorStream(
        input_accumulators=[
            StreamingCenteredAnchor(row, query_shape=(G1_RESIDUAL_RANK, 2))
            for row in functional.input_statistics
        ],
        output_accumulators=[
            tuple(
                StreamingCenteredAnchor(row, query_shape=(G1_RESIDUAL_RANK, 2))
                for row in groups
            )
            for groups in functional.output_statistics
        ],
        boundaries=[
            NativeOutputBankState(final=value.detach())
            for value in video.native.final_outputs
        ],
    )


def _accumulate_anchor_stream(
    *,
    video: Any,
    functional: Any,
    state: AnchorProgramState,
    owners: Sequence[TargetOwner],
    scorer: ProgramNativeAnchorScorer,
    stream: _AnchorStream,
) -> None:
    next_frame = 0
    for chunk in video.native.chunks():
        stop = next_frame + chunk.frame_count
        if (
            chunk.start_frame != next_frame
            or stop > video.native.frame_count
            or len(chunk.inputs) != len(owners)
            or len(chunk.outputs) != len(owners)
        ):
            raise NativeFactorError("compiler B0b native stream changed")
        assignment = video.canonical_assignment[next_frame:stop].float()
        frame_metadata = scorer.frame_metadata(
            assignment, video.frame_positions[next_frame:stop]
        )
        x_metadata = scorer.candidate_metadata(frame_metadata, output=False)
        y_metadata = scorer.candidate_metadata(frame_metadata, output=True)
        ratio = functional.event_frame[:, next_frame:stop] / functional.base_frame[
            next_frame:stop
        ][None].clamp_min(1e-12)
        x_mass = candidate_mass(
            functional.base_frame[next_frame:stop], output=False
        )
        y_mass = candidate_mass(
            functional.base_frame[next_frame:stop], output=True
        )
        for target, (owner, x, y) in enumerate(
            zip(owners, chunk.inputs, chunk.outputs, strict=True)
        ):
            compatibility = checkpoint(
                _input_anchor_compatibility,
                scorer,
                x,
                x_metadata,
                functional.input_queries[target],
                state.event_weights[target],
                ratio,
                target,
                functional.feature_plan.input_whiteners[target],
                use_reentrant=False,
                preserve_rng_state=False,
            )
            stream.input_accumulators[target].add(x, x_mass, compatibility)
            bank = stream.boundaries[target].build(y, start_frame=next_frame)
            groups = native_output_group_count(owner)
            grouped = bank.reshape(
                *bank.shape[:-1], groups, owner.out_features // groups
            ).movedim(-2, 0)
            compatibility = checkpoint(
                _output_anchor_compatibility,
                scorer,
                grouped,
                y_metadata[None],
                functional.output_queries[target],
                state.event_weights[target],
                ratio,
                target,
                functional.feature_plan.output_whiteners[target],
                use_reentrant=False,
                preserve_rng_state=False,
            )
            for group, accumulator in enumerate(stream.output_accumulators[target]):
                accumulator.add(grouped[group], y_mass, compatibility[group])
        next_frame = stop
    if next_frame != video.native.frame_count or any(
        boundary.next_frame != next_frame for boundary in stream.boundaries
    ):
        raise NativeFactorError("compiler B0b native stream ended early")


def build_replay_bank_plan(
    *,
    video: Any,
    functional: Any,
    state: AnchorProgramState,
    owners: Sequence[TargetOwner],
    scorer: ProgramNativeAnchorScorer,
    relative_floor: float,
    replay_score_rms: float,
) -> ReplayBankPlan:
    """Reread B0, solve native anchors, and normalize the exact B1 queries."""

    stream = _new_anchor_stream(video, functional)
    _accumulate_anchor_stream(
        video=video,
        functional=functional,
        state=state,
        owners=owners,
        scorer=scorer,
        stream=stream,
    )
    entries = [
        ((target, "input", 0), value.finalize())
        for target, value in enumerate(stream.input_accumulators)
    ]
    entries.extend(
        ((target, "output", group), value.finalize())
        for target, groups in enumerate(stream.output_accumulators)
        for group, value in enumerate(groups)
    )
    solved = _solve_statistics(entries, relative_floor=relative_floor)
    input_queries = []
    output_queries = []
    for target in range(len(owners)):
        query = solved[(target, "input", 0)].query.float()
        normalized, _ = normalize_replay_queries(
            (query,),
            (functional.input_statistics[target],),
            score_rms=replay_score_rms,
        )
        input_queries.append(normalized[0])
        groups = tuple(
            solved[(target, "output", group)].query.float()
            for group in range(len(functional.output_statistics[target]))
        )
        normalized, _ = normalize_replay_queries(
            groups,
            functional.output_statistics[target],
            score_rms=replay_score_rms,
        )
        output_queries.append(normalized)
    diagnostics = tuple(solved.values())
    solve_metrics = state.rank.new_tensor(
        (
            max(value.relative_residual_maximum for value in diagnostics),
            min(value.retained_trace_fraction for value in diagnostics),
            min(value.anchor_projection_minimum for value in diagnostics),
            min(value.retained_rank for value in diagnostics),
        )
    )
    return ReplayBankPlan(
        input_queries=tuple(input_queries),
        output_queries=tuple(output_queries),
        frame_measure=functional.frame_measure,
        group_gains=functional.group_gains,
        solve_metrics=solve_metrics,
        conditioning_metrics=functional.conditioning_metrics,
    )
