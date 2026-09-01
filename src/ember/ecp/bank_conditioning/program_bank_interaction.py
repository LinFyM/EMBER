"""Event-conditioned whole-bank interaction for exact native signed pooling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as functional

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
    native_output_group_count,
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
class OutputProgramBankSetConditions:
    """One output group's scope-matched training-only S0 conditions."""

    all_types: torch.Tensor
    by_type: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class ProgramBankSetConditions:
    """S0 condition tree with the same scopes as a real B0 bank response."""

    inputs: tuple[torch.Tensor, ...]
    outputs: tuple[tuple[OutputProgramBankSetConditions, ...], ...]


@dataclass(frozen=True)
class ProgramBankSetSummaries:
    """Finalized input/output summaries for every target in one video."""

    inputs: tuple[EventBankSetSummary, ...]
    outputs: tuple[tuple[OutputEventBankSetSummary, ...], ...]

    def with_condition(
        self, condition: ProgramBankSetConditions
    ) -> ProgramBankSetSummaries:
        """Apply one structured S0 tree without broadcasting across scopes."""

        if len(condition.inputs) != len(self.inputs) or len(condition.outputs) != len(
            self.outputs
        ):
            raise BankConditioningError("free bank-set condition target count changed")
        return ProgramBankSetSummaries(
            inputs=tuple(
                value.with_condition(token)
                for value, token in zip(
                    self.inputs, condition.inputs, strict=True
                )
            ),
            outputs=tuple(
                tuple(
                    OutputEventBankSetSummary(
                        all_types=value.all_types.with_condition(token.all_types),
                        by_type=tuple(
                            row.with_condition(kind)
                            for row, kind in zip(
                                value.by_type, token.by_type, strict=True
                            )
                        ),
                    )
                    for value, token in zip(groups, tokens, strict=True)
                )
                for groups, tokens in zip(
                    self.outputs, condition.outputs, strict=True
                )
            ),
        )


class EventConditionedBankSetInteraction(torch.nn.Module):
    """Use a Program-conditioned whole-bank read to form native primals."""

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        program_width: int,
        event_slots: int,
        summary_value_width: int = 16,
        hidden_width: int = 64,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.program_width = int(program_width)
        self.event_slots = int(event_slots)
        self.coordinate_width = G1_RESIDUAL_RANK * self.event_slots
        self.summary_value_width = int(summary_value_width)
        self.hidden_width = int(hidden_width)
        if (
            not self.owners
            or min(
                self.program_width,
                self.event_slots,
                self.summary_value_width,
                self.hidden_width,
            )
            <= 0
        ):
            raise BankConditioningError("invalid bank-set interaction topology")
        self.b0_query_context_width = 4 * self.program_width
        self.summary_width = 2 * (
            self.coordinate_width + self.summary_value_width
        ) + 2
        families = tuple(TargetFamily)
        self.set_encoder = torch.nn.ModuleDict(
            {
                family.value: EventBankSetEncoder(
                    context_width=self.b0_query_context_width,
                    coordinate_width=self.coordinate_width,
                    summary_value_width=self.summary_value_width,
                    hidden_width=self.hidden_width,
                )
                for family in families
            }
        )
        self.input_primal_gate = torch.nn.ModuleDict(
            {
                family.value: self._primal_gate_network(self.summary_width)
                for family in families
            }
        )
        self.output_primal_gate = torch.nn.ModuleDict(
            {
                family.value: self._primal_gate_network(
                    (1 + len(OUTPUT_BANK_TYPES)) * self.summary_width
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
        self.owner_slot_context = torch.nn.Parameter(
            slots[: len(self.owners)].clone()
        )
        self.rank_slot_context = torch.nn.Parameter(
            slots[len(self.owners) : len(self.owners) + G1_RESIDUAL_RANK].clone()
        )
        self.event_slot_context = torch.nn.Parameter(
            slots[len(self.owners) + G1_RESIDUAL_RANK :].clone()
        )
        self.register_parameter("tasklocal_free_b0_query", None)
        self.tasklocal_free_input_anchor = None
        self.tasklocal_free_output_anchor = None

    def install_tasklocal_free_b0_query(self, initial: torch.Tensor) -> None:
        """Install one diagnostic query per target, shared by every bank scope."""

        expected = (
            len(self.owners),
            G1_RESIDUAL_RANK,
            self.event_slots,
            self.coordinate_width,
        )
        if self.tasklocal_free_b0_query is not None or initial.shape != expected:
            raise BankConditioningError("task-local free B0 query axes changed")
        self.tasklocal_free_b0_query = torch.nn.Parameter(initial.detach().clone())

    def install_tasklocal_free_native_anchor(self) -> None:
        """Add one shared full-native basis without replacing the real anchor."""

        if (
            self.tasklocal_free_input_anchor is not None
            or self.tasklocal_free_output_anchor is not None
        ):
            raise BankConditioningError("task-local free native anchor already exists")
        reference = next(self.parameters())

        def parameter(width: int) -> torch.nn.Parameter:
            return torch.nn.Parameter(
                torch.zeros(
                    G1_RESIDUAL_RANK,
                    self.event_slots,
                    width,
                    device=reference.device,
                    dtype=reference.dtype,
                )
            )

        self.tasklocal_free_input_anchor = torch.nn.ParameterList(
            [parameter(owner.in_features) for owner in self.owners]
        )
        self.tasklocal_free_output_anchor = torch.nn.ModuleList(
            [
                torch.nn.ParameterList(
                    [
                        parameter(
                            owner.out_features // native_output_group_count(owner)
                        )
                        for _ in range(native_output_group_count(owner))
                    ]
                )
                for owner in self.owners
            ]
        )

    def validate_tasklocal_free_gradients(
        self, *, require_native_anchor: bool
    ) -> None:
        """Check the one delayed-gradient step owned by task-local controls."""

        query = self.tasklocal_free_b0_query
        if (
            query is None
            or query.grad is None
            or not bool(torch.isfinite(query.grad).all())
            or not bool(query.grad.abs().sum() > 0)
        ):
            raise BankConditioningError(
                "task-local free B0 query has no finite gradient"
            )
        if not require_native_anchor:
            return
        inputs = self.tasklocal_free_input_anchor
        outputs = self.tasklocal_free_output_anchor
        if inputs is None or outputs is None:
            raise BankConditioningError("task-local free native anchor is absent")
        anchors = (*tuple(inputs.parameters()), *tuple(outputs.parameters()))
        gradients = tuple(value.grad for value in anchors)
        if (
            not gradients
            or any(value is None for value in gradients)
            or not bool(
                torch.stack(tuple(torch.isfinite(value).all() for value in gradients)).all()
            )
            or not bool(
                torch.stack(tuple(value.abs().sum() for value in gradients)).sum() > 0
            )
        ):
            raise BankConditioningError(
                "task-local free native anchor has no finite gradient"
            )

    def _augmented_native_anchor(
        self,
        *,
        target: int,
        group: int | None,
        candidate: torch.Tensor,
    ) -> torch.Tensor:
        inputs = self.tasklocal_free_input_anchor
        outputs = self.tasklocal_free_output_anchor
        if (inputs is None) != (outputs is None):
            raise BankConditioningError("task-local free native anchor ownership changed")
        if inputs is None:
            return candidate
        free = inputs[target] if group is None else outputs[target][group]
        if free.shape != candidate.shape:
            raise BankConditioningError("task-local free native anchor axes changed")
        return candidate + free.to(candidate)

    def _primal_gate_network(self, width: int) -> torch.nn.Sequential:
        network = torch.nn.Sequential(
            torch.nn.LayerNorm(width, elementwise_affine=False),
            torch.nn.Linear(width, self.hidden_width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_width, 1, bias=False),
        )
        torch.nn.init.zeros_(network[-1].weight)
        return network

    @staticmethod
    def _add_native_anchor(
        base: torch.Tensor,
        anchor: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        if (
            base.ndim != 2
            or anchor.shape[:1] != base.shape[:1]
            or anchor.shape[-1] != base.shape[-1]
            or gate.shape != anchor.shape[:2]
            or anchor.ndim != 3
        ):
            raise BankConditioningError("bank-conditioned primal axes changed")
        correction = torch.einsum(
            "re,red->rd", gate.float(), anchor.float()
        )
        return base + correction.to(base)

    def bank_conditioned_primals(
        self,
        *,
        input_primals: tuple[torch.Tensor, ...],
        output_primals: tuple[torch.Tensor, ...],
        summaries: ProgramBankSetSummaries,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Form each current-bank primal before dualization and exact replay."""

        if (
            len(input_primals) != len(self.owners)
            or len(output_primals) != len(self.owners)
            or len(summaries.inputs) != len(self.owners)
            or len(summaries.outputs) != len(self.owners)
        ):
            raise BankConditioningError("bank-conditioned primal target count changed")
        conditioned_inputs = []
        conditioned_outputs = []
        for target, owner in enumerate(self.owners):
            family = owner.family.value
            input_summary = summaries.inputs[target]
            input_gate = self.input_primal_gate[family](
                input_summary.condition
            ).squeeze(-1)
            conditioned_inputs.append(
                self._add_native_anchor(
                    input_primals[target],
                    self._augmented_native_anchor(
                        target=target,
                        group=None,
                        candidate=input_summary.native_anchor,
                    ),
                    input_gate,
                )
            )
            groups = []
            if output_primals[target].shape[0] != len(summaries.outputs[target]):
                raise BankConditioningError(
                    "bank-conditioned output primal group count changed"
                )
            for group, (base, summary) in enumerate(
                zip(
                    output_primals[target],
                    summaries.outputs[target],
                    strict=True,
                )
            ):
                condition = torch.cat(
                    (
                        summary.all_types.condition,
                        *(value.condition for value in summary.by_type),
                    ),
                    dim=-1,
                )
                gate = self.output_primal_gate[family](condition).squeeze(-1)
                groups.append(
                    self._add_native_anchor(
                        base,
                        self._augmented_native_anchor(
                            target=target,
                            group=group,
                            candidate=summary.all_types.native_anchor,
                        ),
                        gate,
                    )
                )
            conditioned_outputs.append(torch.stack(tuple(groups)))
        return tuple(conditioned_inputs), tuple(conditioned_outputs)

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

    def _validate_program_event_state(
        self, program_event_state: torch.Tensor
    ) -> None:
        if program_event_state.shape != (
            G1_RESIDUAL_RANK,
            self.event_slots,
            self.program_width,
        ):
            raise BankConditioningError("Program bank-set event state changed")

    def _structural_context(self, target: int) -> torch.Tensor:
        if not 0 <= target < len(self.owners):
            raise BankConditioningError("Program bank-set target changed")
        owner = self.owner_slot_context.float()[target].reshape(1, 1, -1)
        owner = owner.expand(G1_RESIDUAL_RANK, self.event_slots, -1)
        rank = self.rank_slot_context.float()[:, None].expand(
            -1, self.event_slots, -1
        )
        event = self.event_slot_context.float()[None].expand(
            G1_RESIDUAL_RANK, -1, -1
        )
        return torch.cat((owner, rank, event), dim=-1)

    def _b0_query_context(
        self, *, target: int, program_event_state: torch.Tensor
    ) -> torch.Tensor:
        """Let Program content enter B1 only through a real-bank set read."""

        self._validate_program_event_state(program_event_state)
        return torch.cat(
            (
                functional.layer_norm(
                    program_event_state.float(), (self.program_width,)
                ),
                self._structural_context(target),
            ),
            dim=-1,
        )

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
        self._validate_context(context)
        inducing = self._b0_query_context(
            target=target,
            program_event_state=program_event_state,
        )
        return self.set_encoder[self.owners[target].family.value].summarize(
            coordinates=coordinates,
            metadata=metadata,
            event_mass=mass,
            event_context=inducing,
            output=False,
            native_values=values,
        )

    def summary_stream(
        self,
        *,
        target: int,
        program_event_state: torch.Tensor,
        context: ProgramBankContext,
        output: bool,
        reference: torch.Tensor,
        collect_native: bool = True,
        inducing_query: torch.Tensor | None = None,
        trusted_finite: bool = False,
    ) -> EventBankSetSummaryStream:
        """Create one B0 online stream with a query fixed by Program and bank."""

        self._validate_context(context)
        encoder = self.set_encoder[self.owners[target].family.value]
        return encoder.new_stream(
            event_context=(
                self._b0_query_context(
                    target=target,
                    program_event_state=program_event_state,
                )
                if inducing_query is None
                else None
            ),
            inducing_query=inducing_query,
            output=output,
            reference=reference,
            events=self.event_slots,
            collect_native=collect_native,
            trusted_finite=trusted_finite,
        )

    def b0_inducing_query(
        self, *, target: int, program_event_state: torch.Tensor
    ) -> torch.Tensor:
        """Compute the one target query shared by all of its B0 scopes."""

        if not 0 <= target < len(self.owners):
            raise BankConditioningError("Program bank-set target changed")
        self._validate_program_event_state(program_event_state)
        if self.tasklocal_free_b0_query is not None:
            return self.tasklocal_free_b0_query[target]
        context = self._b0_query_context(
            target=target,
            program_event_state=program_event_state,
        )
        encoder = self.set_encoder[self.owners[target].family.value]
        return encoder.inducing(context.float())

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
        self._validate_context(context)
        inducing = self._b0_query_context(
            target=target,
            program_event_state=program_event_state,
        )
        encoder = self.set_encoder[self.owners[target].family.value]
        all_types = encoder.summarize(
            coordinates=coordinates,
            metadata=metadata,
            event_mass=mass,
            event_context=inducing,
            output=True,
            native_values=values,
        )
        by_type = tuple(
            encoder.summarize(
                coordinates=coordinates[..., kind, :],
                metadata=metadata[..., kind, :],
                event_mass=mass[..., kind] * len(OUTPUT_BANK_TYPES),
                event_context=inducing,
                output=True,
                native_values=None,
            )
            for kind in range(len(OUTPUT_BANK_TYPES))
        )
        return OutputEventBankSetSummary(all_types=all_types, by_type=by_type)
