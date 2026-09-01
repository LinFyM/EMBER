"""Event-conditioned whole-bank interaction for exact native signed pooling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from ember.ecp.bank_conditioning.operator import BankConditioningError
from ember.ecp.bank_conditioning.set_summary import (
    EventBankSetEncoder,
    EventBankSetSummary,
    EventBankSetSummaryStream,
    OutputEventBankSetSummary,
    candidate_metadata,
    event_candidate_measure,
    program_relative_coordinates,
)
from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.native_factors import (
    G1_RESIDUAL_RANK,
    OUTPUT_BANK_TYPES,
)


@dataclass(frozen=True)
class ProgramBankContext:
    """Deployment-visible local Program evidence owned by one video bank."""

    canonical_assignment: torch.Tensor
    frame_positions: torch.Tensor
    local_scene: torch.Tensor
    local_process: torch.Tensor
    local_presence: torch.Tensor
    local_tau: torch.Tensor
    local_sigma: torch.Tensor

    def frame_slice(self, start: int, stop: int) -> ProgramBankContext:
        if not 0 <= start < stop <= self.frame_positions.shape[0]:
            raise BankConditioningError("Program-bank frame slice changed")
        return ProgramBankContext(
            canonical_assignment=self.canonical_assignment[start:stop],
            frame_positions=self.frame_positions[start:stop],
            local_scene=self.local_scene,
            local_process=self.local_process,
            local_presence=self.local_presence,
            local_tau=self.local_tau,
            local_sigma=self.local_sigma,
        )


@dataclass(frozen=True)
class ProgramBankInteractionState:
    """Shared Program queries paired with one bank's local context."""

    context: ProgramBankContext
    rank_event: torch.Tensor
    event_weights: torch.Tensor
    input_event_queries: tuple[torch.Tensor, ...]
    output_event_queries: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class ProgramBankSetSummaries:
    """Finalized input/output summaries for every target in one video."""

    inputs: tuple[EventBankSetSummary, ...]
    outputs: tuple[tuple[OutputEventBankSetSummary, ...], ...]

    def with_condition(self, condition: torch.Tensor) -> ProgramBankSetSummaries:
        """Broadcast one S0 training token across native owners and scopes."""

        return ProgramBankSetSummaries(
            inputs=tuple(value.with_condition(condition) for value in self.inputs),
            outputs=tuple(
                tuple(
                    OutputEventBankSetSummary(
                        all_types=value.all_types.with_condition(condition),
                        by_type=tuple(
                            row.with_condition(condition) for row in value.by_type
                        ),
                    )
                    for value in groups
                )
                for groups in self.outputs
            ),
        )


class EventConditionedBankSetInteraction(torch.nn.Module):
    """Condition exact candidate corrections on the whole current native bank."""

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        program_width: int,
        event_slots: int,
        summary_value_width: int = 16,
        hidden_width: int = 64,
        correction_bound: float = 0.1,
        replay_score_rms: float,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.program_width = int(program_width)
        self.event_slots = int(event_slots)
        self.coordinate_width = G1_RESIDUAL_RANK * self.event_slots
        self.summary_value_width = int(summary_value_width)
        self.hidden_width = int(hidden_width)
        self.correction_bound = float(correction_bound)
        self.replay_score_rms = float(replay_score_rms)
        if (
            not self.owners
            or min(
                self.program_width,
                self.event_slots,
                self.summary_value_width,
                self.hidden_width,
            )
            <= 0
            or not 0.0 < self.correction_bound <= 1.0
            or not math.isfinite(self.replay_score_rms)
            or self.replay_score_rms <= 0.0
        ):
            raise BankConditioningError("invalid bank-set interaction topology")
        self.event_context_width = 4 * self.program_width + 3
        self.summary_width = 2 * (
            self.coordinate_width + self.summary_value_width
        ) + 2
        families = tuple(TargetFamily)
        self.set_encoder = torch.nn.ModuleDict(
            {
                family.value: EventBankSetEncoder(
                    context_width=self.event_context_width,
                    coordinate_width=self.coordinate_width,
                    summary_value_width=self.summary_value_width,
                    hidden_width=self.hidden_width,
                )
                for family in families
            }
        )
        self.input_candidate = torch.nn.ModuleDict(
            {
                family.value: self._candidate_network(2 * self.coordinate_width + 5)
                for family in families
            }
        )
        self.output_candidate = torch.nn.ModuleDict(
            {
                family.value: self._candidate_network(2 * self.coordinate_width + 9)
                for family in families
            }
        )
        self.input_condition = torch.nn.ModuleDict(
            {
                family.value: self._condition_network(
                    self.event_context_width + self.summary_width
                )
                for family in families
            }
        )
        self.output_condition = torch.nn.ModuleDict(
            {
                family.value: self._condition_network(
                    self.event_context_width + 2 * self.summary_width
                )
                for family in families
            }
        )
        slots = torch.empty(
            len(self.owners) + G1_RESIDUAL_RANK + self.event_slots,
            self.program_width,
        )
        torch.nn.init.orthogonal_(slots)
        slots.mul_(math.sqrt(self.program_width))
        owner_end = len(self.owners)
        rank_end = owner_end + G1_RESIDUAL_RANK
        self.owner_slot_context = torch.nn.Parameter(slots[:owner_end].clone())
        self.rank_slot_context = torch.nn.Parameter(
            slots[owner_end:rank_end].clone()
        )
        self.event_slot_context = torch.nn.Parameter(
            slots[rank_end:].clone()
        )

    def _candidate_network(self, width: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, self.hidden_width),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_width, self.hidden_width),
        )

    def _condition_network(self, width: int) -> torch.nn.Sequential:
        """Generate a candidate head directly from Program and bank summary."""

        network = torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, 2 * self.hidden_width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * self.hidden_width, self.hidden_width + 1),
        )
        torch.nn.init.zeros_(network[-1].weight)
        torch.nn.init.zeros_(network[-1].bias)
        return network

    def input_event_delta(
        self, *, family: str, hidden: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        parameters = self.input_condition[family](condition)
        candidate_axes = hidden.ndim - parameters.ndim
        weight = parameters[..., : self.hidden_width].reshape(
            *parameters.shape[:-1], *((1,) * candidate_axes), self.hidden_width
        )
        bias = parameters[..., self.hidden_width].reshape(
            *parameters.shape[:-1], *((1,) * candidate_axes)
        )
        score = (hidden * weight).sum(-1) / math.sqrt(self.hidden_width) + bias
        return self.correction_bound * torch.tanh(score)

    def output_event_delta(
        self, *, family: str, hidden: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        parameters = self.output_condition[family](condition)
        candidate_axes = hidden.ndim - parameters.ndim
        prefix = parameters.shape[:-2]
        types = parameters.shape[-2]
        weight = parameters[..., : self.hidden_width].reshape(
            *prefix,
            *((1,) * candidate_axes),
            types,
            self.hidden_width,
        )
        bias = parameters[..., self.hidden_width].reshape(
            *prefix, *((1,) * candidate_axes), types
        )
        score = (hidden * weight).sum(-1) / math.sqrt(self.hidden_width) + bias
        return self.correction_bound * torch.tanh(score)

    def _validate_context(self, context: ProgramBankContext) -> int:
        frames = int(context.frame_positions.shape[0])
        targets = len(self.owners)
        if (
            frames <= 0
            or context.frame_positions.shape != (frames,)
            or context.canonical_assignment.shape != (frames, self.event_slots)
            or context.local_scene.shape != (targets, self.program_width)
            or context.local_process.shape
            != (self.event_slots, targets, self.program_width)
            or context.local_presence.shape != (self.event_slots,)
            or context.local_tau.shape != (self.event_slots, 2)
            or context.local_sigma.shape
            != (self.event_slots, targets, self.program_width)
        ):
            raise BankConditioningError("Program-bank local context changed")
        return frames

    def _event_context(
        self,
        *,
        target: int,
        program_event_state: torch.Tensor,
        context: ProgramBankContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_context(context)
        if program_event_state.shape != (
            G1_RESIDUAL_RANK,
            self.event_slots,
            self.program_width,
        ):
            raise BankConditioningError("Program bank-set event state changed")
        local = torch.cat(
            (
                context.local_scene[target].float()[None].expand(
                    self.event_slots, -1
                ),
                context.local_process[:, target].float(),
                context.local_sigma[:, target].float(),
                context.local_presence.float()[:, None],
                context.local_tau.float(),
            ),
            dim=-1,
        )
        # The trainable interaction may identify structural rank/event slots, but
        # must not turn the fixed task Program code into an eight-entry function
        # table. Task content still reaches B0/B1 through the frozen native
        # queries, their Program-relative coordinates, base scores, event
        # assignment/weights, and the resulting real-bank summaries.
        owner_slot = self.owner_slot_context[target].float()
        rank_slot = self.rank_slot_context.float()[:, None]
        event_slot = self.event_slot_context.float()[None]
        structural = (
            owner_slot[None, None] + rank_slot + event_slot
        ) / math.sqrt(3.0)
        rank = torch.cat(
            (
                structural,
                local[None].expand(G1_RESIDUAL_RANK, -1, -1),
            ),
            dim=-1,
        )
        inducing_structural = (
            owner_slot[None] + self.event_slot_context.float()
        ) / math.sqrt(2.0)
        inducing = torch.cat((inducing_structural, local), dim=-1)
        return rank, inducing

    def summarize_input(
        self,
        *,
        target: int,
        program_event_state: torch.Tensor,
        native_event_query: torch.Tensor,
        values: torch.Tensor,
        native_mean: torch.Tensor,
        frame_measure: torch.Tensor,
        context: ProgramBankContext,
    ) -> EventBankSetSummary:
        coordinates = program_relative_coordinates(
            native_event_query, values, native_mean
        )
        metadata = candidate_metadata(
            context.frame_positions, output=False, like=values
        )
        mass = event_candidate_measure(
            frame_measure, context.canonical_assignment, output=False
        )
        _, inducing = self._event_context(
            target=target,
            program_event_state=program_event_state,
            context=context,
        )
        return self.set_encoder[self.owners[target].family.value].summarize(
            coordinates=coordinates,
            metadata=metadata,
            event_mass=mass,
            event_context=inducing,
            output=False,
        )

    def summary_stream(
        self,
        *,
        target: int,
        program_event_state: torch.Tensor,
        context: ProgramBankContext,
        output: bool,
        reference: torch.Tensor,
    ) -> EventBankSetSummaryStream:
        """Create one B0 online stream with a query fixed by Program and bank."""

        _, inducing = self._event_context(
            target=target,
            program_event_state=program_event_state,
            context=context,
        )
        encoder = self.set_encoder[self.owners[target].family.value]
        return encoder.new_stream(
            event_context=inducing,
            output=output,
            reference=reference,
            events=self.event_slots,
        )

    def summarize_output(
        self,
        *,
        target: int,
        program_event_state: torch.Tensor,
        native_event_query: torch.Tensor,
        values: torch.Tensor,
        native_mean: torch.Tensor,
        frame_measure: torch.Tensor,
        context: ProgramBankContext,
    ) -> OutputEventBankSetSummary:
        coordinates = program_relative_coordinates(
            native_event_query, values, native_mean
        )
        metadata = candidate_metadata(
            context.frame_positions, output=True, like=values
        )
        mass = event_candidate_measure(
            frame_measure, context.canonical_assignment, output=True
        )
        _, inducing = self._event_context(
            target=target,
            program_event_state=program_event_state,
            context=context,
        )
        encoder = self.set_encoder[self.owners[target].family.value]
        all_types = encoder.summarize(
            coordinates=coordinates,
            metadata=metadata,
            event_mass=mass,
            event_context=inducing,
            output=True,
        )
        by_type = tuple(
            encoder.summarize(
                coordinates=coordinates[..., kind, :],
                metadata=metadata[..., kind, :],
                event_mass=mass[..., kind] * len(OUTPUT_BANK_TYPES),
                event_context=inducing,
                output=True,
            )
            for kind in range(len(OUTPUT_BANK_TYPES))
        )
        return OutputEventBankSetSummary(all_types=all_types, by_type=by_type)

    def _base_score(
        self, base_query: torch.Tensor, centered: torch.Tensor
    ) -> torch.Tensor:
        if base_query.shape != (G1_RESIDUAL_RANK, centered.shape[-1]):
            raise BankConditioningError("bank-set base query axes changed")
        return torch.einsum(
            "rd,...d->r...", base_query.detach().float(), centered.detach().float()
        ) / self.replay_score_rms

    def _collapse_events(
        self,
        event_delta: torch.Tensor,
        event_weights: torch.Tensor,
        assignment: torch.Tensor,
    ) -> torch.Tensor:
        candidate_ndim = event_delta.ndim - 2
        event_assignment = assignment.float().T.reshape(
            1,
            self.event_slots,
            assignment.shape[0],
            *((1,) * (candidate_ndim - 1)),
        )
        weights = event_weights.float().reshape(
            G1_RESIDUAL_RANK,
            self.event_slots,
            *((1,) * candidate_ndim),
        )
        return (event_delta * event_assignment * weights).sum(1)

    def input_logit_corrections(
        self,
        *,
        target: int,
        program_event_state: torch.Tensor,
        native_event_query: torch.Tensor,
        event_weights: torch.Tensor,
        base_query: torch.Tensor,
        values: torch.Tensor,
        native_mean: torch.Tensor,
        context: ProgramBankContext,
        summary: EventBankSetSummary,
    ) -> torch.Tensor:
        rank_context, _ = self._event_context(
            target=target,
            program_event_state=program_event_state,
            context=context,
        )
        coordinates = program_relative_coordinates(
            native_event_query, values, native_mean
        )
        standardized = (
            coordinates[None]
            - summary.mean.reshape(self.event_slots, *((1,) * (coordinates.ndim - 1)), -1)
        ) / (
            0.5
            * summary.log_variance.reshape(
                self.event_slots, *((1,) * (coordinates.ndim - 1)), -1
            )
        ).exp()
        metadata = candidate_metadata(
            context.frame_positions, output=False, like=values
        )
        centered = values.float() - native_mean.detach().float()
        log_norm = centered.square().mean(-1).clamp_min(1e-12).sqrt().log()
        score = self._base_score(base_query, centered)
        candidate_shape = coordinates.shape[:-1]
        features = torch.cat(
            (
                standardized[None].expand(G1_RESIDUAL_RANK, -1, *candidate_shape, -1),
                coordinates.reshape(1, 1, *candidate_shape, -1).expand(
                    G1_RESIDUAL_RANK, self.event_slots, *candidate_shape, -1
                ),
                score.reshape(G1_RESIDUAL_RANK, 1, *candidate_shape, 1).expand(
                    -1, self.event_slots, *candidate_shape, -1
                ),
                log_norm.reshape(1, 1, *candidate_shape, 1).expand(
                    G1_RESIDUAL_RANK, self.event_slots, *candidate_shape, -1
                ),
                metadata.reshape(1, 1, *candidate_shape, 3).expand(
                    G1_RESIDUAL_RANK, self.event_slots, *candidate_shape, -1
                ),
            ),
            dim=-1,
        )
        family = self.owners[target].family.value
        hidden = self.input_candidate[family](features)
        condition = torch.cat(
            (
                rank_context,
                summary.condition[None].expand(G1_RESIDUAL_RANK, -1, -1),
            ),
            dim=-1,
        )
        event_delta = self.input_event_delta(
            family=family,
            hidden=hidden,
            condition=condition,
        )
        correction = self._collapse_events(
            event_delta, event_weights, context.canonical_assignment
        )
        return torch.stack((correction, -correction), dim=1)

    def output_logit_corrections(
        self,
        *,
        target: int,
        program_event_state: torch.Tensor,
        native_event_query: torch.Tensor,
        event_weights: torch.Tensor,
        base_query: torch.Tensor,
        values: torch.Tensor,
        native_mean: torch.Tensor,
        context: ProgramBankContext,
        summary: OutputEventBankSetSummary,
    ) -> torch.Tensor:
        rank_context, _ = self._event_context(
            target=target,
            program_event_state=program_event_state,
            context=context,
        )
        coordinates = program_relative_coordinates(
            native_event_query, values, native_mean
        )
        all_summary = summary.all_types
        candidate_shape = coordinates.shape[:-1]
        standardized = (
            coordinates[None]
            - all_summary.mean.reshape(
                self.event_slots, *((1,) * (coordinates.ndim - 1)), -1
            )
        ) / (
            0.5
            * all_summary.log_variance.reshape(
                self.event_slots, *((1,) * (coordinates.ndim - 1)), -1
            )
        ).exp()
        metadata = candidate_metadata(
            context.frame_positions, output=True, like=values
        )
        centered = values.float() - native_mean.detach().float()
        log_norm = centered.square().mean(-1).clamp_min(1e-12).sqrt().log()
        score = self._base_score(base_query, centered)
        features = torch.cat(
            (
                standardized[None].expand(G1_RESIDUAL_RANK, -1, *candidate_shape, -1),
                coordinates.reshape(1, 1, *candidate_shape, -1).expand(
                    G1_RESIDUAL_RANK, self.event_slots, *candidate_shape, -1
                ),
                score.reshape(G1_RESIDUAL_RANK, 1, *candidate_shape, 1).expand(
                    -1, self.event_slots, *candidate_shape, -1
                ),
                log_norm.reshape(1, 1, *candidate_shape, 1).expand(
                    G1_RESIDUAL_RANK, self.event_slots, *candidate_shape, -1
                ),
                metadata.reshape(1, 1, *candidate_shape, 7).expand(
                    G1_RESIDUAL_RANK, self.event_slots, *candidate_shape, -1
                ),
            ),
            dim=-1,
        )
        family = self.owners[target].family.value
        hidden = self.output_candidate[family](features)
        own = torch.stack(tuple(value.condition for value in summary.by_type), dim=1)
        all_condition = all_summary.condition[:, None].expand(-1, len(OUTPUT_BANK_TYPES), -1)
        condition = torch.cat(
            (
                rank_context[:, :, None].expand(-1, -1, len(OUTPUT_BANK_TYPES), -1),
                all_condition[None].expand(G1_RESIDUAL_RANK, -1, -1, -1),
                own[None].expand(G1_RESIDUAL_RANK, -1, -1, -1),
            ),
            dim=-1,
        )
        event_delta = self.output_event_delta(
            family=family,
            hidden=hidden,
            condition=condition,
        )
        correction = self._collapse_events(
            event_delta, event_weights, context.canonical_assignment
        )
        return torch.stack((correction, -correction), dim=1)
