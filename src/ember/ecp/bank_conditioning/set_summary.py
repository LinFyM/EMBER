"""Program-relative, event-conditioned summaries of native candidate banks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from ember.ecp.bank_conditioning.operator import BankConditioningError
from ember.ecp.contracts import ACTION_HORIZON
from ember.ecp.native_factors import (
    G1_PROBE_COUNT,
    G1_RESIDUAL_RANK,
    OUTPUT_BANK_TYPES,
)


@dataclass(frozen=True)
class EventBankSetSummary:
    """Finalized summary for one input bank or one output-summary scope."""

    mean: torch.Tensor
    log_variance: torch.Tensor
    induced_positive: torch.Tensor
    induced_negative: torch.Tensor
    log_partition: torch.Tensor
    event_mass: torch.Tensor
    condition_override: torch.Tensor | None = None

    @property
    def condition(self) -> torch.Tensor:
        if (
            self.induced_positive.ndim != 3
            or self.induced_positive.shape != self.induced_negative.shape
            or self.induced_positive.shape[:2]
            != (G1_RESIDUAL_RANK, self.mean.shape[0])
            or self.log_variance.shape != self.mean.shape
            or self.log_partition.shape
            != (G1_RESIDUAL_RANK, self.mean.shape[0], 2)
        ):
            raise BankConditioningError("rank-specific bank-set summary changed")
        mean = self.mean[None].expand(G1_RESIDUAL_RANK, -1, -1)
        log_variance = self.log_variance[None].expand(
            G1_RESIDUAL_RANK, -1, -1
        )
        value = torch.cat(
            (
                mean,
                log_variance,
                self.induced_positive,
                self.induced_negative,
                self.log_partition,
            ),
            dim=-1,
        )
        if self.condition_override is None:
            return value
        if self.condition_override.shape != value.shape:
            raise BankConditioningError("bank-set condition override shape changed")
        return self.condition_override

    def with_condition(self, condition: torch.Tensor) -> EventBankSetSummary:
        """Use a training-only free summary without changing set coordinates."""

        if condition.shape != self.condition.shape:
            raise BankConditioningError("free bank-set summary shape changed")
        return EventBankSetSummary(
            mean=self.mean,
            log_variance=self.log_variance,
            induced_positive=self.induced_positive,
            induced_negative=self.induced_negative,
            log_partition=self.log_partition,
            event_mass=self.event_mass,
            condition_override=condition,
        )


@dataclass(frozen=True)
class OutputEventBankSetSummary:
    """All-type and own-type context for one joint dynamic output bank."""

    all_types: EventBankSetSummary
    by_type: tuple[EventBankSetSummary, ...]

    def __post_init__(self) -> None:
        if len(self.by_type) != len(OUTPUT_BANK_TYPES):
            raise BankConditioningError("output bank-set type summary changed")


def program_relative_coordinates(
    native_event_query: torch.Tensor,
    values: torch.Tensor,
    native_mean: torch.Tensor,
) -> torch.Tensor:
    """Return each candidate's complete rank-by-event native coordinate."""

    if (
        native_event_query.ndim != 3
        or native_event_query.shape[0] != G1_RESIDUAL_RANK
        or values.ndim < 2
        or native_event_query.shape[-1] != values.shape[-1]
        or native_mean.shape != (values.shape[-1],)
    ):
        raise BankConditioningError("Program-relative coordinate axes changed")
    centered = values.float() - native_mean.detach().float()
    candidate = centered / centered.square().mean(-1, keepdim=True).clamp_min(
        1e-12
    ).sqrt()
    query = native_event_query.float()
    query = query / query.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    coordinate = torch.einsum("red,...d->...re", query, candidate)
    coordinate = coordinate / math.sqrt(values.shape[-1])
    return coordinate.flatten(-2)


def candidate_metadata(
    frame_positions: torch.Tensor,
    *,
    output: bool,
    like: torch.Tensor,
) -> torch.Tensor:
    """Candidate metadata without inventing an output-type axis for input X."""

    frames = frame_positions.to(like).clamp(0.0, 1.0)
    probes = torch.linspace(-1.0, 1.0, G1_PROBE_COUNT, device=like.device)
    horizons = torch.linspace(-1.0, 1.0, ACTION_HORIZON, device=like.device)
    if not output:
        shape = (frames.shape[0], G1_PROBE_COUNT, ACTION_HORIZON)
        return torch.stack(
            torch.broadcast_tensors(
                frames[:, None, None],
                probes[None, :, None],
                horizons[None, None, :],
            ),
            dim=-1,
        ).expand(*shape, 3)
    types = functional.one_hot(
        torch.arange(len(OUTPUT_BANK_TYPES), device=like.device),
        num_classes=len(OUTPUT_BANK_TYPES),
    ).to(like)
    shape = (
        frames.shape[0],
        G1_PROBE_COUNT,
        ACTION_HORIZON,
        len(OUTPUT_BANK_TYPES),
    )
    return torch.cat(
        (
            frames[:, None, None, None, None].expand(*shape, 1),
            probes[None, :, None, None, None].expand(*shape, 1),
            horizons[None, None, :, None, None].expand(*shape, 1),
            types[None, None, None].expand(*shape, len(OUTPUT_BANK_TYPES)),
        ),
        dim=-1,
    )


def event_candidate_measure(
    frame_measure: torch.Tensor,
    assignment: torch.Tensor,
    *,
    output: bool,
) -> torch.Tensor:
    """Unit-scope event measure; fixed candidate multiplicity cannot bias log-Z."""

    if (
        frame_measure.ndim != 1
        or assignment.ndim != 2
        or assignment.shape[0] != frame_measure.shape[0]
        or torch.any(frame_measure < 0)
        or torch.any(assignment < 0)
    ):
        raise BankConditioningError("event candidate measure axes changed")
    events = assignment.shape[1]
    base = frame_measure.float()[:, None, None]
    base = base.expand(-1, G1_PROBE_COUNT, ACTION_HORIZON)
    base = base / float(G1_PROBE_COUNT * ACTION_HORIZON)
    mass = assignment.float().T.reshape(events, -1, 1, 1) * base
    if output:
        mass = mass[..., None].expand(-1, -1, -1, -1, len(OUTPUT_BANK_TYPES))
        mass = mass / float(len(OUTPUT_BANK_TYPES))
    return mass


class StreamingEventBankSummary:
    """Online moments and antithetic induced attention for one summary scope."""

    def __init__(
        self,
        *,
        events: int,
        coordinate_width: int,
        value_width: int,
        reference: torch.Tensor,
    ) -> None:
        if min(events, coordinate_width, value_width) <= 0:
            raise BankConditioningError("invalid event bank-set summary topology")
        self.events = int(events)
        self.coordinate_width = int(coordinate_width)
        self.value_width = int(value_width)
        self.mass = reference.new_zeros(events, dtype=torch.float32)
        self.first = reference.new_zeros(events, coordinate_width, dtype=torch.float32)
        self.second = reference.new_zeros(events, coordinate_width, dtype=torch.float32)
        self.maximum = reference.new_full(
            (G1_RESIDUAL_RANK, events, 2), -torch.inf, dtype=torch.float32
        )
        self.normalizer = reference.new_zeros(
            G1_RESIDUAL_RANK, events, 2, dtype=torch.float32
        )
        self.weighted = reference.new_zeros(
            G1_RESIDUAL_RANK, events, 2, value_width, dtype=torch.float32
        )
        self.candidate_count = 0

    def add(
        self,
        coordinates: torch.Tensor,
        event_mass: torch.Tensor,
        inducing_query: torch.Tensor,
        summary_values: torch.Tensor,
    ) -> None:
        candidate_shape = coordinates.shape[:-1]
        if (
            coordinates.shape[-1] != self.coordinate_width
            or event_mass.shape != (self.events, *candidate_shape)
            or inducing_query.shape
            != (G1_RESIDUAL_RANK, self.events, self.coordinate_width)
            or summary_values.shape != (*candidate_shape, self.value_width)
        ):
            raise BankConditioningError("event bank-set candidate axes changed")
        coordinate = coordinates.float().reshape(-1, self.coordinate_width)
        values = summary_values.float().reshape(-1, self.value_width)
        mass = event_mass.detach().float().reshape(self.events, -1)
        if (
            torch.any(mass < 0)
            or not bool(torch.isfinite(coordinate).all())
            or not bool(torch.isfinite(values).all())
            or not bool(torch.isfinite(mass).all())
        ):
            raise BankConditioningError("event bank-set stream is invalid")
        self.mass = self.mass + mass.sum(-1)
        self.first = self.first + torch.einsum("en,nk->ek", mass, coordinate)
        self.second = self.second + torch.einsum(
            "en,nk->ek", mass, coordinate.square()
        )
        normalized = functional.layer_norm(coordinate, (self.coordinate_width,))
        score = torch.einsum(
            "rek,nk->ren", inducing_query.float(), normalized
        ) / math.sqrt(self.coordinate_width)
        log_mass = torch.where(
            mass > 0,
            mass.clamp_min(1e-30).log(),
            torch.full_like(mass, -torch.inf),
        )
        logits = log_mass[None, :, None] + torch.stack((score, -score), dim=2)
        chunk_maximum = logits.amax(-1)
        maximum = torch.maximum(self.maximum, chunk_maximum).detach()
        finite = torch.isfinite(maximum)
        old_scale = torch.where(
            torch.isfinite(self.maximum),
            torch.exp(self.maximum - torch.where(finite, maximum, self.maximum)),
            torch.zeros_like(maximum),
        )
        shift = torch.where(finite, maximum, torch.zeros_like(maximum))
        weights = torch.where(
            torch.isfinite(logits),
            torch.exp(logits - shift[..., None]),
            torch.zeros_like(logits),
        )
        self.weighted = self.weighted * old_scale[..., None] + torch.einsum(
            "rebn,nv->rebv", weights, values
        )
        self.normalizer = self.normalizer * old_scale + weights.sum(-1)
        self.maximum = maximum
        self.candidate_count += int(coordinate.shape[0])

    def finalize(self) -> EventBankSetSummary:
        if self.candidate_count <= 0 or not torch.all(self.mass > 0):
            raise BankConditioningError("event bank-set summary has an empty event")
        denominator = self.mass[:, None]
        mean = self.first / denominator
        variance = (self.second / denominator - mean.square()).clamp_min(1e-6)
        induced = self.weighted / self.normalizer.clamp_min(1e-30)[..., None]
        log_partition = self.maximum + self.normalizer.clamp_min(1e-30).log()
        if not all(
            bool(torch.isfinite(value).all())
            for value in (mean, variance, induced, log_partition)
        ):
            raise BankConditioningError("event bank-set summary is non-finite")
        return EventBankSetSummary(
            mean=mean,
            log_variance=variance.log(),
            induced_positive=induced[:, :, 0],
            induced_negative=induced[:, :, 1],
            log_partition=log_partition,
            event_mass=self.mass,
        )


class EventBankSetSummaryStream:
    """Chunk-facing encoder state that retains only online summary statistics."""

    def __init__(
        self,
        *,
        accumulator: StreamingEventBankSummary,
        inducing_query: torch.Tensor,
        value_network: torch.nn.Module,
        output: bool,
    ) -> None:
        self.accumulator = accumulator
        self.inducing_query = inducing_query
        self.value_network = value_network
        self.output = bool(output)

    def add(
        self,
        *,
        coordinates: torch.Tensor,
        metadata: torch.Tensor,
        event_mass: torch.Tensor,
    ) -> None:
        expected_metadata = 7 if self.output else 3
        if metadata.shape != (*coordinates.shape[:-1], expected_metadata):
            raise BankConditioningError("event bank-set metadata axes changed")
        summary_values = self.value_network(
            torch.cat((coordinates, metadata.float()), dim=-1)
        )
        self.accumulator.add(
            coordinates, event_mass, self.inducing_query, summary_values
        )

    def finalize(self) -> EventBankSetSummary:
        return self.accumulator.finalize()


class EventBankSetEncoder(torch.nn.Module):
    """Generate real B0 summaries; S0 bypasses only this learned source."""

    def __init__(
        self,
        *,
        context_width: int,
        coordinate_width: int,
        summary_value_width: int,
        hidden_width: int,
    ) -> None:
        super().__init__()
        if min(context_width, coordinate_width, summary_value_width, hidden_width) <= 0:
            raise BankConditioningError("invalid event bank-set encoder topology")
        self.coordinate_width = int(coordinate_width)
        self.summary_value_width = int(summary_value_width)
        self.inducing = torch.nn.Sequential(
            torch.nn.LayerNorm(context_width),
            torch.nn.Linear(context_width, hidden_width),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, coordinate_width),
        )
        self.input_value = self._value_network(coordinate_width + 3, hidden_width)
        self.output_value = self._value_network(coordinate_width + 7, hidden_width)

    def _value_network(self, width: int, hidden: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, self.summary_value_width),
        )

    def summarize(
        self,
        *,
        coordinates: torch.Tensor,
        metadata: torch.Tensor,
        event_mass: torch.Tensor,
        event_context: torch.Tensor,
        output: bool,
    ) -> EventBankSetSummary:
        stream = self.new_stream(
            event_context=event_context,
            output=output,
            reference=coordinates,
            events=event_mass.shape[0],
        )
        stream.add(
            coordinates=coordinates,
            metadata=metadata,
            event_mass=event_mass,
        )
        return stream.finalize()

    def new_stream(
        self,
        *,
        event_context: torch.Tensor,
        output: bool,
        reference: torch.Tensor,
        events: int,
    ) -> EventBankSetSummaryStream:
        query = self.inducing(event_context.float())
        network = self.output_value if output else self.input_value
        accumulator = StreamingEventBankSummary(
            events=events,
            coordinate_width=self.coordinate_width,
            value_width=self.summary_value_width,
            reference=reference,
        )
        return EventBankSetSummaryStream(
            accumulator=accumulator,
            inducing_query=query,
            value_network=network,
            output=output,
        )
