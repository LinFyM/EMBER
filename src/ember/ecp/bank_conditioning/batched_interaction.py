"""Shape-batched EBSRI correction heads over frozen candidate descriptors."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Sequence

import torch

from ember.ecp.bank_conditioning.candidate_descriptors import CandidateDescriptor
from ember.ecp.native_factors import G1_RESIDUAL_RANK, OUTPUT_BANK_TYPES


@contextmanager
def _candidate_mlp_matmul(device: torch.device):
    """Allow qualified TF32 only inside learned correction MLPs."""

    if device.type != "cuda":
        yield
        return
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _input_features(
    descriptors: Sequence[CandidateDescriptor],
    summaries: Sequence[Any],
    metadata: torch.Tensor,
) -> torch.Tensor:
    coordinates = torch.stack(tuple(value.coordinates for value in descriptors))
    base_score = torch.stack(tuple(value.base_score for value in descriptors))
    log_norm = torch.stack(tuple(value.log_norm for value in descriptors))
    means = torch.stack(tuple(value.mean for value in summaries))
    deviations = torch.stack(tuple(value.log_variance for value in summaries))
    groups, frames, probes, horizons, width = coordinates.shape
    events = means.shape[1]
    standardized = (coordinates[:, None] - means[:, :, None, None, None]) / (
        0.5 * deviations[:, :, None, None, None]
    ).exp()
    shape = (groups, G1_RESIDUAL_RANK, events, frames, probes, horizons)
    return torch.cat(
        (
            standardized[:, None].expand(-1, G1_RESIDUAL_RANK, -1, -1, -1, -1, -1),
            coordinates[:, None, None].expand(-1, G1_RESIDUAL_RANK, events, -1, -1, -1, -1),
            base_score[:, :, None, ..., None].expand(-1, -1, events, -1, -1, -1, -1),
            log_norm[:, None, None, ..., None].expand(-1, G1_RESIDUAL_RANK, events, -1, -1, -1, -1),
            metadata.reshape(1, 1, 1, frames, probes, horizons, 3).expand(
                groups, G1_RESIDUAL_RANK, events, -1, -1, -1, -1
            ),
        ),
        dim=-1,
    ).reshape(*shape, 2 * width + 5)


def batched_input_corrections(
    interaction: Any,
    *,
    family: str,
    descriptors: Sequence[CandidateDescriptor],
    summaries: Sequence[Any],
    structural_gate: torch.Tensor,
    event_weights: torch.Tensor,
    assignment: torch.Tensor,
    metadata: torch.Tensor,
) -> torch.Tensor:
    features = _input_features(descriptors, summaries, metadata)
    summary_condition = torch.stack(tuple(value.condition for value in summaries))
    with _candidate_mlp_matmul(features.device):
        hidden = interaction.input_candidate[family](features)
        event_delta = interaction.input_event_delta(
            family=family,
            hidden=hidden,
            condition=summary_condition,
            structural_gate=structural_gate,
        )
    correction = (
        event_delta
        * event_weights[:, :, :, None, None, None]
        * assignment.T[None, None, :, :, None, None]
    ).sum(2)
    return torch.stack((correction, -correction), dim=2)


def _output_features(
    descriptors: Sequence[CandidateDescriptor],
    summaries: Sequence[Any],
    metadata: torch.Tensor,
) -> torch.Tensor:
    coordinates = torch.stack(tuple(value.coordinates for value in descriptors))
    base_score = torch.stack(tuple(value.base_score for value in descriptors))
    log_norm = torch.stack(tuple(value.log_norm for value in descriptors))
    means = torch.stack(tuple(value.all_types.mean for value in summaries))
    deviations = torch.stack(
        tuple(value.all_types.log_variance for value in summaries)
    )
    groups, frames, probes, horizons, types, width = coordinates.shape
    events = means.shape[1]
    standardized = (coordinates[:, None] - means[:, :, None, None, None, None]) / (
        0.5 * deviations[:, :, None, None, None, None]
    ).exp()
    shape = (
        groups,
        G1_RESIDUAL_RANK,
        events,
        frames,
        probes,
        horizons,
        types,
    )
    return torch.cat(
        (
            standardized[:, None].expand(-1, G1_RESIDUAL_RANK, -1, -1, -1, -1, -1, -1),
            coordinates[:, None, None].expand(-1, G1_RESIDUAL_RANK, events, -1, -1, -1, -1, -1),
            base_score[:, :, None, ..., None].expand(-1, -1, events, -1, -1, -1, -1, -1),
            log_norm[:, None, None, ..., None].expand(-1, G1_RESIDUAL_RANK, events, -1, -1, -1, -1, -1),
            metadata.reshape(1, 1, 1, frames, probes, horizons, types, 7).expand(
                groups, G1_RESIDUAL_RANK, events, -1, -1, -1, -1, -1
            ),
        ),
        dim=-1,
    ).reshape(*shape, 2 * width + 9)


def batched_output_corrections(
    interaction: Any,
    *,
    family: str,
    descriptors: Sequence[CandidateDescriptor],
    summaries: Sequence[Any],
    structural_gate: torch.Tensor,
    event_weights: torch.Tensor,
    assignment: torch.Tensor,
    metadata: torch.Tensor,
) -> torch.Tensor:
    features = _output_features(descriptors, summaries, metadata)
    all_condition = torch.stack(
        tuple(value.all_types.condition for value in summaries)
    )
    own_condition = torch.stack(
        tuple(
            torch.stack(tuple(scope.condition for scope in value.by_type), dim=2)
            for value in summaries
        )
    )
    condition = torch.cat(
        (
            all_condition[:, :, :, None].expand(
                -1, -1, -1, len(OUTPUT_BANK_TYPES), -1
            ),
            own_condition,
        ),
        dim=-1,
    )
    with _candidate_mlp_matmul(features.device):
        hidden = interaction.output_candidate[family](features)
        event_delta = interaction.output_event_delta(
            family=family,
            hidden=hidden,
            condition=condition,
            structural_gate=structural_gate[:, :, :, None].expand(
                -1, -1, -1, len(OUTPUT_BANK_TYPES), -1
            ),
        )
    correction = (
        event_delta
        * event_weights[:, :, :, None, None, None, None]
        * assignment.T[None, None, :, :, None, None, None]
    ).sum(2)
    return torch.stack((correction, -correction), dim=2)
