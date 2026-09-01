"""Compact whole-bank summary and exact native signed replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from ember.ecp.bank_conditioning.operator import StreamingSignedPool
from ember.ecp.bank_conditioning.primal_dual import (
    SpectralNativeCovariance,
    native_candidate_mass,
)
from ember.ecp.bank_conditioning.program_bank_interaction import (
    EventConditionedBankSetInteraction,
    ProgramBankInteractionState,
    ProgramBankSetSummaries,
)
from ember.ecp.bank_conditioning.set_summary import (
    OutputEventBankSetSummary,
    candidate_metadata,
    event_candidate_measure,
    program_relative_coordinates,
)
from ember.ecp.native_factors import (
    G1_RESIDUAL_RANK,
    NativeFactorError,
    NativeOutputBankState,
    OUTPUT_BANK_TYPES,
    native_output_group_count,
)


@dataclass(frozen=True)
class PrimalDualVideoResult:
    input_values: tuple[torch.Tensor, ...]
    output_values: tuple[torch.Tensor, ...]
    frame_measure: torch.Tensor
    group_gains: torch.Tensor
    solve_metrics: torch.Tensor
    conditioning_metrics: torch.Tensor


@dataclass(frozen=True)
class CompactPrimalDualVideo:
    """Frozen raw X/Y plus B0 operator; output-bank types stay implicit."""

    frame_measure: torch.Tensor
    input_operators: tuple[SpectralNativeCovariance, ...]
    output_operators: tuple[tuple[SpectralNativeCovariance, ...], ...]
    input_values: tuple[torch.Tensor, ...]
    output_values: tuple[torch.Tensor, ...]
    final_outputs: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class ReplayPlan:
    input_queries: tuple[torch.Tensor, ...]
    output_queries: tuple[tuple[torch.Tensor, ...], ...]
    frame_measure: torch.Tensor
    group_gains: torch.Tensor
    solve_metrics: torch.Tensor
    conditioning_metrics: torch.Tensor


def summarize_compact_replay(
    operator: Any,
    prepared: CompactPrimalDualVideo,
    *,
    bank_set_interaction: EventConditionedBankSetInteraction,
    interaction_state: ProgramBankInteractionState,
) -> ProgramBankSetSummaries:
    """B0.5 scan over exact candidate sets independently of B1 replay chunks."""

    operator._validate_interaction_state(interaction_state)
    if not operator._interaction_enabled(bank_set_interaction, interaction_state):
        raise NativeFactorError("compact bank-set summary lost interaction")
    inputs = []
    outputs = []
    frames = int(prepared.frame_measure.shape[0])
    summary_frame_chunk = max(1, int(operator.covariance_frame_chunk))
    for target, (owner, x, y) in enumerate(
        zip(
            operator.owners,
            prepared.input_values,
            prepared.output_values,
            strict=True,
        )
    ):
        input_stream = bank_set_interaction.summary_stream(
            target=target,
            program_event_state=interaction_state.rank_event[target],
            context=interaction_state.context,
            output=False,
            reference=x,
        )
        groups = native_output_group_count(owner)
        output_streams = tuple(
            (
                bank_set_interaction.summary_stream(
                    target=target,
                    program_event_state=interaction_state.rank_event[target],
                    context=interaction_state.context,
                    output=True,
                    reference=prepared.output_operators[target][group].mean,
                ),
                tuple(
                    bank_set_interaction.summary_stream(
                        target=target,
                        program_event_state=interaction_state.rank_event[target],
                        context=interaction_state.context,
                        output=True,
                        reference=prepared.output_operators[target][group].mean,
                        collect_native=False,
                    )
                    for _ in OUTPUT_BANK_TYPES
                ),
            )
            for group in range(groups)
        )
        boundary = NativeOutputBankState(final=prepared.final_outputs[target].detach())
        for start in range(0, frames, summary_frame_chunk):
            stop = min(start + summary_frame_chunk, frames)
            context = interaction_state.context.frame_slice(start, stop)
            frame_measure = prepared.frame_measure[start:stop]
            input_coordinates = program_relative_coordinates(
                interaction_state.input_event_queries[target],
                x[start:stop],
                prepared.input_operators[target].mean,
            )
            input_stream.add(
                coordinates=input_coordinates,
                metadata=candidate_metadata(
                    context.frame_positions, output=False, like=x
                ),
                event_mass=event_candidate_measure(
                    frame_measure, context.canonical_assignment, output=False
                ),
                native_values=x[start:stop],
            )
            bank = boundary.build(y[start:stop], start_frame=start)
            grouped = bank.reshape(
                *bank.shape[:-1], groups, owner.out_features // groups
            ).movedim(-2, 0)
            for group, (all_types, by_type) in enumerate(output_streams):
                values = grouped[group]
                coordinates = program_relative_coordinates(
                    interaction_state.output_event_queries[target][group],
                    values,
                    prepared.output_operators[target][group].mean,
                )
                metadata = candidate_metadata(
                    context.frame_positions, output=True, like=values
                )
                mass = event_candidate_measure(
                    frame_measure, context.canonical_assignment, output=True
                )
                all_types.add(
                    coordinates=coordinates,
                    metadata=metadata,
                    event_mass=mass,
                    native_values=values,
                )
                for kind, stream in enumerate(by_type):
                    stream.add(
                        coordinates=coordinates[..., kind, :],
                        metadata=metadata[..., kind, :],
                        event_mass=mass[..., kind] * len(OUTPUT_BANK_TYPES),
                    )
        if boundary.next_frame != frames:
            raise NativeFactorError("compact summary output boundary ended early")
        inputs.append(input_stream.finalize())
        outputs.append(
            tuple(
                OutputEventBankSetSummary(
                    all_types=all_types.finalize(),
                    by_type=tuple(stream.finalize() for stream in by_type),
                )
                for all_types, by_type in output_streams
            )
        )
    return ProgramBankSetSummaries(inputs=tuple(inputs), outputs=tuple(outputs))


def _new_pools(
    operator: Any, plan: ReplayPlan, *, trusted_finite_bias: bool = False
) -> tuple[Any, Any]:
    input_block, output_block = operator._candidate_blocks()
    inputs = tuple(
        StreamingSignedPool(
            query,
            trusted_positive_measure=True,
            trusted_finite_bias=trusted_finite_bias,
            canonical_block_candidates=input_block,
        )
        for query in plan.input_queries
    )
    outputs = tuple(
        tuple(
            StreamingSignedPool(
                query,
                trusted_positive_measure=True,
                trusted_finite_bias=trusted_finite_bias,
                canonical_block_candidates=output_block,
            )
            for query in groups
        )
        for groups in plan.output_queries
    )
    return inputs, outputs


def _replay_target(
    operator: Any,
    prepared: CompactPrimalDualVideo,
    *,
    target: int,
    inputs: Any,
    outputs: Any,
    direct_inputs: Sequence[torch.Tensor] | None,
    direct_outputs: Sequence[Sequence[torch.Tensor]] | None,
) -> None:
    owner = operator.owners[target]
    x = prepared.input_values[target]
    y = prepared.output_values[target]
    boundary = NativeOutputBankState(final=prepared.final_outputs[target].detach())
    groups = native_output_group_count(owner)
    frames = int(prepared.frame_measure.shape[0])
    for start in range(0, frames, operator.covariance_frame_chunk):
        stop = min(start + operator.covariance_frame_chunk, frames)
        x_chunk = x[start:stop]
        y_chunk = y[start:stop]
        x_mass = native_candidate_mass(
            prepared.frame_measure[start:stop], output=False
        )
        y_mass = native_candidate_mass(
            prepared.frame_measure[start:stop], output=True
        )
        input_bias = (
            None
            if direct_inputs is None
            else direct_inputs[target][..., start:stop, :, :]
        )
        inputs[target].add(x_chunk, x_mass, input_bias)
        bank = boundary.build(y_chunk, start_frame=start)
        grouped = bank.reshape(
            *bank.shape[:-1], groups, owner.out_features // groups
        ).movedim(-2, 0)
        for group, accumulator in enumerate(outputs[target]):
            output_bias = (
                None
                if direct_outputs is None
                else direct_outputs[target][group][..., start:stop, :, :, :]
            )
            accumulator.add(grouped[group], y_mass, output_bias)
    if boundary.next_frame != frames:
        raise NativeFactorError("compact output boundary ended early")


def apply_compact_replay(
    operator: Any,
    prepared: CompactPrimalDualVideo,
    input_primals: tuple[torch.Tensor, ...],
    output_primals: tuple[torch.Tensor, ...],
    *,
    direct_input_logit_biases: Sequence[torch.Tensor] | None = None,
    direct_output_logit_biases: Sequence[Sequence[torch.Tensor]] | None = None,
    replay_plan: ReplayPlan | None = None,
) -> PrimalDualVideoResult:
    """Dualize one primal set and replay exactly once over the real X/Y bank."""

    direct_biases = direct_input_logit_biases is not None
    if direct_biases != (direct_output_logit_biases is not None):
        raise NativeFactorError("direct signed-pool bias ownership changed")
    if direct_biases and (
        len(direct_input_logit_biases) != len(operator.owners)
        or len(direct_output_logit_biases) != len(operator.owners)
    ):
        raise NativeFactorError("direct signed-pool target count changed")
    plan = (
        operator._plan(prepared, input_primals, output_primals)
        if replay_plan is None
        else replay_plan
    )
    inputs, outputs = _new_pools(
        operator, plan, trusted_finite_bias=direct_biases
    )
    for target in range(len(operator.owners)):
        _replay_target(
            operator,
            prepared,
            target=target,
            inputs=inputs,
            outputs=outputs,
            direct_inputs=direct_input_logit_biases,
            direct_outputs=direct_output_logit_biases,
        )
    return PrimalDualVideoResult(
        input_values=tuple(value.signed_mean() for value in inputs),
        output_values=tuple(
            torch.cat(tuple(value.signed_mean() for value in groups), dim=-1)
            for groups in outputs
        ),
        frame_measure=plan.frame_measure,
        group_gains=plan.group_gains,
        solve_metrics=plan.solve_metrics,
        conditioning_metrics=plan.conditioning_metrics,
    )
