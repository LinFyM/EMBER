"""Streaming per-event normalization for shared native candidate features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from ember.ecp.contracts import ACTION_HORIZON, TargetOwner
from ember.ecp.native_factors import (
    G1_PROBE_COUNT,
    OUTPUT_BANK_TYPES,
    NativeOutputBankState,
    native_output_group_count,
)

from ember.ecp.bank_conditioning.operator import BankConditioningError


@dataclass(frozen=True)
class FeatureStatistics:
    mean: torch.Tensor
    covariance: torch.Tensor
    total_mass: torch.Tensor


@dataclass(frozen=True)
class FeatureWhitener:
    """A detached symmetric inverse-square-root in fixed canonical events."""

    mean: torch.Tensor
    inverse_sqrt: torch.Tensor
    retained_ranks: tuple[int, ...]
    retained_trace_minimum: float

    def whiten(self, keys: torch.Tensor) -> torch.Tensor:
        if (
            keys.ndim < 2
            or keys.shape[-1] != self.mean.shape[-1]
            or self.mean.ndim != 2
            or self.inverse_sqrt.shape
            != (self.mean.shape[0], self.mean.shape[1], self.mean.shape[1])
        ):
            raise BankConditioningError("feature-whitening candidate axes changed")
        leading = (self.mean.shape[0], *((1,) * (keys.ndim - 1)))
        centered = keys.float()[None] - self.mean.reshape(
            *leading, self.mean.shape[-1]
        )
        return torch.einsum(
            "e...d,edh->e...h", centered, self.inverse_sqrt
        )


@dataclass(frozen=True)
class FeatureWhiteningPlan:
    input_whiteners: tuple[FeatureWhitener, ...]
    output_whiteners: tuple[tuple[FeatureWhitener, ...], ...]
    metrics: torch.Tensor


@dataclass
class _FeatureStream:
    input_accumulators: list[StreamingFeatureStatistics]
    output_accumulators: list[tuple[StreamingFeatureStatistics, ...]]
    boundaries: list[NativeOutputBankState]


class StreamingFeatureStatistics:
    """Accumulate event-normalized candidate-key moments without retaining keys."""

    def __init__(
        self,
        *,
        events: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if events <= 0 or width <= 0:
            raise BankConditioningError("invalid feature-statistics topology")
        self.events = int(events)
        self.width = int(width)
        self.dtype = dtype
        self.total_mass = torch.zeros(events, device=device, dtype=dtype)
        self.first_moment = torch.zeros(events, width, device=device, dtype=dtype)
        self.second_moment = torch.zeros(
            events, width, width, device=device, dtype=dtype
        )
        self.candidate_count = 0

    def add(self, keys: torch.Tensor, event_mass: torch.Tensor) -> None:
        if (
            keys.ndim < 2
            or keys.shape[-1] != self.width
            or event_mass.shape != (self.events, *keys.shape[:-1])
            or keys.numel() <= 0
        ):
            raise BankConditioningError("feature-statistics candidate axes changed")
        flat_keys = keys.detach().to(dtype=self.dtype).reshape(-1, self.width)
        flat_mass = event_mass.detach().to(dtype=self.dtype).reshape(
            self.events, -1
        )
        if (
            torch.any(flat_mass < 0)
            or not bool(torch.isfinite(flat_keys).all())
            or not bool(torch.isfinite(flat_mass).all())
        ):
            raise BankConditioningError("feature statistics are invalid")
        self.total_mass = self.total_mass + flat_mass.sum(-1)
        self.first_moment = self.first_moment + torch.einsum(
            "en,nd->ed", flat_mass, flat_keys
        )
        self.second_moment = self.second_moment + torch.einsum(
            "en,nd,nf->edf", flat_mass, flat_keys, flat_keys
        )
        self.candidate_count += int(flat_keys.shape[0])

    def finalize(self) -> FeatureStatistics:
        if (
            self.candidate_count <= 1
            or torch.any(self.total_mass <= 0)
            or not bool(torch.isfinite(self.total_mass).all())
        ):
            raise BankConditioningError("feature-statistics stream is empty")
        mean = self.first_moment / self.total_mass[:, None]
        covariance = self.second_moment / self.total_mass[:, None, None]
        covariance = covariance - torch.einsum("ed,ef->edf", mean, mean)
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        if not bool(torch.isfinite(covariance).all()):
            raise BankConditioningError("feature covariance is non-finite")
        return FeatureStatistics(
            mean=mean,
            covariance=covariance,
            total_mass=self.total_mass,
        )


def batched_feature_whiteners(
    statistics: Sequence[FeatureStatistics],
    *,
    relative_eigenvalue_floor: float = 1e-6,
) -> tuple[FeatureWhitener, ...]:
    """Build symmetric whiteners in one batched eigensystem."""

    rows = tuple(statistics)
    if not rows or not 0.0 < relative_eigenvalue_floor < 1.0:
        raise BankConditioningError("invalid feature-whitening contract")
    shape = rows[0].covariance.shape
    if len(shape) != 3 or shape[-1] != shape[-2] or any(
        row.covariance.shape != shape or row.mean.shape != shape[:1] + shape[1:2]
        for row in rows
    ):
        raise BankConditioningError("feature-whitening statistics changed shape")
    covariance = torch.stack(
        tuple(row.covariance.double().detach() for row in rows)
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    maximum = eigenvalues[..., -1].clamp_min(torch.finfo(eigenvalues.dtype).tiny)
    keep = eigenvalues > maximum[..., None] * float(relative_eigenvalue_floor)
    if torch.any(keep.sum(-1) <= 0):
        raise BankConditioningError("feature whitening retained no direction")
    safe = torch.where(keep, eigenvalues, torch.ones_like(eigenvalues))
    coefficients = safe.rsqrt() * keep
    inverse_sqrt = torch.einsum(
        "...di,...i,...hi->...dh", eigenvectors, coefficients, eigenvectors
    )
    positive = eigenvalues.clamp_min(0)
    trace = (positive * keep).sum(-1) / positive.sum(-1).clamp_min(1e-30)
    if not bool(torch.isfinite(inverse_sqrt).all()) or not bool(
        torch.isfinite(trace).all()
    ):
        raise BankConditioningError("feature whitening is non-finite")
    return tuple(
        FeatureWhitener(
            mean=row.mean.float().detach(),
            inverse_sqrt=inverse_sqrt[index].float().detach(),
            retained_ranks=tuple(int(value) for value in keep[index].sum(-1)),
            retained_trace_minimum=float(trace[index].min()),
        )
        for index, row in enumerate(rows)
    )


def _event_candidate_mass(
    event_frame: torch.Tensor,
    *,
    output: bool,
) -> torch.Tensor:
    shape = (
        event_frame.shape[0],
        event_frame.shape[1],
        G1_PROBE_COUNT,
        ACTION_HORIZON,
    )
    if output:
        shape = (*shape, len(OUTPUT_BANK_TYPES))
    leading = (*event_frame.shape, *((1,) * (len(shape) - 2)))
    return event_frame.reshape(leading).expand(shape)


def identity_feature_whitening_plan(
    owners: Sequence[TargetOwner],
    *,
    events: int,
    width: int,
    reference: torch.Tensor,
) -> FeatureWhiteningPlan:
    mean = reference.new_zeros(events, width)
    inverse = torch.eye(
        width, device=reference.device, dtype=reference.dtype
    )[None].expand(events, -1, -1)
    identity = FeatureWhitener(
        mean=mean,
        inverse_sqrt=inverse,
        retained_ranks=(width,) * events,
        retained_trace_minimum=1.0,
    )
    return FeatureWhiteningPlan(
        input_whiteners=(identity,) * len(owners),
        output_whiteners=tuple(
            (identity,) * native_output_group_count(owner) for owner in owners
        ),
        metrics=reference.new_tensor((width, 1.0)),
    )


def _new_feature_stream(
    *,
    video: Any,
    event_frame: torch.Tensor,
    owners: Sequence[TargetOwner],
    events: int,
    width: int,
) -> _FeatureStream:
    output_groups = tuple(native_output_group_count(owner) for owner in owners)
    return _FeatureStream(
        input_accumulators=[
            StreamingFeatureStatistics(
                events=events,
                width=width,
                device=event_frame.device,
            )
            for _ in owners
        ],
        output_accumulators=[
            tuple(
                StreamingFeatureStatistics(
                    events=events,
                    width=width,
                    device=event_frame.device,
                )
                for _ in range(groups)
            )
            for groups in output_groups
        ],
        boundaries=[
            NativeOutputBankState(final=value.detach())
            for value in video.native.final_outputs
        ],
    )


def _accumulate_feature_stream(
    *,
    video: Any,
    event_frame: torch.Tensor,
    scorer: Any,
    owners: Sequence[TargetOwner],
    stream: _FeatureStream,
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
            raise BankConditioningError("feature-whitening stream changed")
        assignment = video.canonical_assignment[next_frame:stop].float()
        frame_metadata = scorer.frame_metadata(
            assignment, video.frame_positions[next_frame:stop]
        )
        input_metadata = scorer.candidate_metadata(frame_metadata, output=False)
        output_metadata = scorer.candidate_metadata(frame_metadata, output=True)
        input_mass = _event_candidate_mass(
            event_frame[:, next_frame:stop], output=False
        )
        output_mass = _event_candidate_mass(
            event_frame[:, next_frame:stop], output=True
        )
        with torch.no_grad():
            for target, (owner, x, y) in enumerate(
                zip(owners, chunk.inputs, chunk.outputs, strict=True)
            ):
                input_key = scorer.input_keys(x, input_metadata, target=target)
                stream.input_accumulators[target].add(input_key, input_mass)
                bank = stream.boundaries[target].build(
                    y, start_frame=next_frame
                )
                groups = native_output_group_count(owner)
                grouped = bank.reshape(
                    *bank.shape[:-1], groups, owner.out_features // groups
                ).movedim(-2, 0)
                output_key = scorer.output_keys(
                    grouped, output_metadata[None], target=target
                )
                for group, accumulator in enumerate(
                    stream.output_accumulators[target]
                ):
                    accumulator.add(output_key[group], output_mass)
        next_frame = stop
    if next_frame != video.native.frame_count or any(
        boundary.next_frame != next_frame for boundary in stream.boundaries
    ):
        raise BankConditioningError("feature-whitening stream ended early")


def _finalize_feature_plan(
    *,
    stream: _FeatureStream,
    owners: Sequence[TargetOwner],
    event_frame: torch.Tensor,
    relative_eigenvalue_floor: float,
) -> FeatureWhiteningPlan:
    entries = [
        ((target, "input", 0), accumulator.finalize())
        for target, accumulator in enumerate(stream.input_accumulators)
    ]
    entries.extend(
        ((target, "output", group), accumulator.finalize())
        for target, accumulators in enumerate(stream.output_accumulators)
        for group, accumulator in enumerate(accumulators)
    )
    whiteners = batched_feature_whiteners(
        tuple(row[1] for row in entries),
        relative_eigenvalue_floor=relative_eigenvalue_floor,
    )
    resolved = {
        key: whitener
        for (key, _), whitener in zip(entries, whiteners, strict=True)
    }
    return FeatureWhiteningPlan(
        input_whiteners=tuple(
            resolved[(target, "input", 0)] for target in range(len(owners))
        ),
        output_whiteners=tuple(
            tuple(
                resolved[(target, "output", group)]
                for group in range(native_output_group_count(owner))
            )
            for target, owner in enumerate(owners)
        ),
        metrics=event_frame.new_tensor(
            (
                min(min(value.retained_ranks) for value in whiteners),
                min(value.retained_trace_minimum for value in whiteners),
            )
        ),
    )


def build_feature_whitening_plan(
    *,
    video: Any,
    event_frame: torch.Tensor,
    scorer: Any,
    owners: Sequence[TargetOwner],
    events: int,
    width: int,
    relative_eigenvalue_floor: float,
) -> FeatureWhiteningPlan:
    """Stream one video's candidate chart into detached per-event gauges."""

    stream = _new_feature_stream(
        video=video,
        event_frame=event_frame,
        owners=owners,
        events=events,
        width=width,
    )
    _accumulate_feature_stream(
        video=video,
        event_frame=event_frame,
        scorer=scorer,
        owners=owners,
        stream=stream,
    )
    return _finalize_feature_plan(
        stream=stream,
        owners=owners,
        event_frame=event_frame,
        relative_eigenvalue_floor=relative_eigenvalue_floor,
    )
