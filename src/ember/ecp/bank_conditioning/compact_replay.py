"""Exact compact-bank summary and signed replay for EBSRI."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence

import torch
from torch.utils.checkpoint import checkpoint

from ember.ecp.bank_conditioning.batched_interaction import (
    batched_input_corrections,
    batched_output_corrections,
)
from ember.ecp.bank_conditioning.candidate_descriptors import (
    CandidateDescriptor,
    FrozenReplayDescriptors,
)
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
from ember.ecp.contracts import TargetFamily, TargetOwner
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


def _candidate_correction(
    call: Callable[..., torch.Tensor], values: torch.Tensor
) -> torch.Tensor:
    """Recompute large frozen-candidate features during backward."""

    if not torch.is_grad_enabled():
        return call(values=values)
    return checkpoint(
        lambda candidate: call(values=candidate),
        values,
        use_reentrant=False,
        preserve_rng_state=False,
    )


def _batched_candidate_correction(
    call: Callable[..., torch.Tensor],
    descriptors: Sequence[CandidateDescriptor],
) -> torch.Tensor:
    if not torch.is_grad_enabled():
        return call(descriptors=descriptors)
    flat = tuple(
        value
        for descriptor in descriptors
        for value in (
            descriptor.coordinates,
            descriptor.base_score,
            descriptor.log_norm,
        )
    )

    def rebuild(*values: torch.Tensor) -> torch.Tensor:
        rows = tuple(
            CandidateDescriptor(
                coordinates=values[index],
                base_score=values[index + 1],
                log_norm=values[index + 2],
            )
            for index in range(0, len(values), 3)
        )
        return call(descriptors=rows)

    return checkpoint(
        rebuild,
        *flat,
        use_reentrant=False,
        preserve_rng_state=False,
    )


def _batches(values: Sequence[Any], size: int) -> tuple[Sequence[Any], ...]:
    return tuple(values[start : start + size] for start in range(0, len(values), size))


def summarize_compact_replay(
    operator: Any,
    prepared: CompactPrimalDualVideo,
    *,
    plan: ReplayPlan,
    bank_set_interaction: EventConditionedBankSetInteraction,
    interaction_state: ProgramBankInteractionState,
) -> ProgramBankSetSummaries:
    """B0.5 scan over exact input and joint output candidate sets."""

    operator._validate_interaction_state(interaction_state)
    if not operator._interaction_enabled(bank_set_interaction, interaction_state):
        raise NativeFactorError("compact bank-set summary lost interaction")
    inputs = []
    outputs = []
    frames = int(prepared.frame_measure.shape[0])
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
                    reference=y,
                ),
                tuple(
                    bank_set_interaction.summary_stream(
                        target=target,
                        program_event_state=interaction_state.rank_event[target],
                        context=interaction_state.context,
                        output=True,
                        reference=y,
                    )
                    for _ in OUTPUT_BANK_TYPES
                ),
            )
            for _ in range(groups)
        )
        boundary = NativeOutputBankState(final=prepared.final_outputs[target].detach())
        for start in range(0, frames, operator.covariance_frame_chunk):
            stop = min(start + operator.covariance_frame_chunk, frames)
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


def _validate_replay_arguments(
    operator: Any,
    *,
    bank_set_interaction: EventConditionedBankSetInteraction | None,
    interaction_state: ProgramBankInteractionState | None,
    summaries: ProgramBankSetSummaries | None,
    direct_input_logit_biases: Sequence[torch.Tensor] | None,
    direct_output_logit_biases: Sequence[Sequence[torch.Tensor]] | None,
    correction_observer: Callable[[str, TargetOwner, torch.Tensor], None] | None,
) -> tuple[bool, bool]:
    operator._validate_interaction_state(interaction_state)
    interaction_enabled = operator._interaction_enabled(
        bank_set_interaction, interaction_state
    )
    direct_biases = direct_input_logit_biases is not None
    if direct_biases != (direct_output_logit_biases is not None):
        raise NativeFactorError("direct signed-pool bias ownership changed")
    if direct_biases and interaction_enabled:
        raise NativeFactorError("direct and bank-set biases cannot be combined")
    if correction_observer is not None and (
        not interaction_enabled or torch.is_grad_enabled()
    ):
        raise NativeFactorError(
            "correction diagnostics require inference-only bank-set replay"
        )
    if direct_biases and (
        len(direct_input_logit_biases) != len(operator.owners)
        or len(direct_output_logit_biases) != len(operator.owners)
    ):
        raise NativeFactorError("direct signed-pool target count changed")
    if summaries is not None and not interaction_enabled:
        raise NativeFactorError("compact bank-set summary ownership changed")
    return interaction_enabled, direct_biases


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


def _interaction_input_bias(
    *,
    interaction: EventConditionedBankSetInteraction,
    state: ProgramBankInteractionState,
    summaries: ProgramBankSetSummaries,
    plan: ReplayPlan,
    prepared: CompactPrimalDualVideo,
    target: int,
    values: torch.Tensor,
    context: Any,
) -> torch.Tensor:
    return _candidate_correction(
        partial(
            interaction.input_logit_corrections,
            target=target,
            program_event_state=state.rank_event[target],
            native_event_query=state.input_event_queries[target],
            event_weights=state.event_weights[target],
            base_query=plan.input_queries[target],
            native_mean=prepared.input_operators[target].mean,
            context=context,
            summary=summaries.inputs[target],
        ),
        values,
    )


def _interaction_output_bias(
    *,
    interaction: EventConditionedBankSetInteraction,
    state: ProgramBankInteractionState,
    summaries: ProgramBankSetSummaries,
    plan: ReplayPlan,
    prepared: CompactPrimalDualVideo,
    target: int,
    group: int,
    values: torch.Tensor,
    context: Any,
) -> torch.Tensor:
    return _candidate_correction(
        partial(
            interaction.output_logit_corrections,
            target=target,
            program_event_state=state.rank_event[target],
            native_event_query=state.output_event_queries[target][group],
            event_weights=state.event_weights[target],
            base_query=plan.output_queries[target][group],
            native_mean=prepared.output_operators[target][group].mean,
            context=context,
            summary=summaries.outputs[target][group],
        ),
        values,
    )


def _replay_target(
    operator: Any,
    prepared: CompactPrimalDualVideo,
    *,
    plan: ReplayPlan,
    target: int,
    inputs: Any,
    outputs: Any,
    interaction: EventConditionedBankSetInteraction | None,
    state: ProgramBankInteractionState | None,
    summaries: ProgramBankSetSummaries | None,
    direct_inputs: Sequence[torch.Tensor] | None,
    direct_outputs: Sequence[Sequence[torch.Tensor]] | None,
    observer: Callable[[str, TargetOwner, torch.Tensor], None] | None,
) -> None:
    owner = operator.owners[target]
    x = prepared.input_values[target]
    y = prepared.output_values[target]
    boundary = NativeOutputBankState(final=prepared.final_outputs[target].detach())
    groups = native_output_group_count(owner)
    frames = int(prepared.frame_measure.shape[0])
    interaction_enabled = interaction is not None
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
        context = state.context.frame_slice(start, stop) if state is not None else None
        input_bias = (
            _interaction_input_bias(
                interaction=interaction,
                state=state,
                summaries=summaries,
                plan=plan,
                prepared=prepared,
                target=target,
                values=x_chunk,
                context=context,
            )
            if interaction_enabled
            else (
                direct_inputs[target][..., start:stop, :, :]
                if direct_inputs is not None
                else None
            )
        )
        if observer is not None:
            observer("input", owner, input_bias[:, 0])
        inputs[target].add(x_chunk, x_mass, input_bias)
        bank = boundary.build(y_chunk, start_frame=start)
        grouped = bank.reshape(
            *bank.shape[:-1], groups, owner.out_features // groups
        ).movedim(-2, 0)
        for group, accumulator in enumerate(outputs[target]):
            output_bias = (
                _interaction_output_bias(
                    interaction=interaction,
                    state=state,
                    summaries=summaries,
                    plan=plan,
                    prepared=prepared,
                    target=target,
                    group=group,
                    values=grouped[group],
                    context=context,
                )
                if interaction_enabled
                else (
                    direct_outputs[target][group][..., start:stop, :, :, :]
                    if direct_outputs is not None
                    else None
                )
            )
            if observer is not None:
                observer("output", owner, output_bias[:, 0])
            accumulator.add(grouped[group], y_mass, output_bias)
    if boundary.next_frame != frames:
        raise NativeFactorError("compact output boundary ended early")


def _replay_cached_interaction(
    operator: Any,
    prepared: CompactPrimalDualVideo,
    *,
    plan: ReplayPlan,
    inputs: Any,
    outputs: Any,
    interaction: EventConditionedBankSetInteraction,
    state: ProgramBankInteractionState,
    summaries: ProgramBankSetSummaries,
    descriptors: FrozenReplayDescriptors,
    group_batch_size: int,
    observer: Callable[[str, TargetOwner, torch.Tensor], None] | None,
) -> None:
    frames = int(prepared.frame_measure.shape[0])
    device = prepared.frame_measure.device
    rank_context = tuple(
        interaction._event_context(
            target=target,
            program_event_state=state.rank_event[target],
            context=state.context,
        )[0]
        for target in range(len(operator.owners))
    )
    input_buckets = {
        family: tuple(
            target
            for target, owner in enumerate(operator.owners)
            if owner.family is family
        )
        for family in TargetFamily
    }
    output_buckets = {
        family: tuple(
            (target, group)
            for target, owner in enumerate(operator.owners)
            if owner.family is family
            for group in range(native_output_group_count(owner))
        )
        for family in TargetFamily
    }
    boundaries = tuple(
        NativeOutputBankState(final=value.detach()) for value in prepared.final_outputs
    )
    for start in range(0, frames, operator.covariance_frame_chunk):
        stop = min(start + operator.covariance_frame_chunk, frames)
        measure = prepared.frame_measure[start:stop]
        assignment = state.context.canonical_assignment[start:stop]
        input_mass = native_candidate_mass(measure, output=False)
        output_mass = native_candidate_mass(measure, output=True)
        input_metadata = descriptors.metadata_slice(
            start, stop, output=False, device=device
        )
        output_metadata = descriptors.metadata_slice(
            start, stop, output=True, device=device
        )
        grouped_outputs = []
        for target, (owner, y, boundary) in enumerate(
            zip(
                operator.owners,
                prepared.output_values,
                boundaries,
                strict=True,
            )
        ):
            bank = boundary.build(y[start:stop], start_frame=start)
            groups = native_output_group_count(owner)
            grouped_outputs.append(
                bank.reshape(
                    *bank.shape[:-1], groups, owner.out_features // groups
                ).movedim(-2, 0)
            )
        for family, targets in input_buckets.items():
            for batch in _batches(targets, group_batch_size):
                cached = tuple(
                    descriptors.inputs[target].frame_slice(
                        start, stop, device=device
                    )
                    for target in batch
                )
                call = partial(
                    batched_input_corrections,
                    interaction,
                    family=family.value,
                    summaries=tuple(summaries.inputs[target] for target in batch),
                    rank_context=torch.stack(
                        tuple(rank_context[target] for target in batch)
                    ),
                    event_weights=torch.stack(
                        tuple(state.event_weights[target] for target in batch)
                    ),
                    assignment=assignment,
                    metadata=input_metadata,
                )
                correction = _batched_candidate_correction(call, cached)
                for row, target in enumerate(batch):
                    if observer is not None:
                        observer(
                            "input", operator.owners[target], correction[row, :, 0]
                        )
                    inputs[target].add(
                        prepared.input_values[target][start:stop],
                        input_mass,
                        correction[row],
                    )
        for family, entries in output_buckets.items():
            for batch in _batches(entries, group_batch_size):
                cached = tuple(
                    descriptors.outputs[target][group].frame_slice(
                        start, stop, device=device
                    )
                    for target, group in batch
                )
                call = partial(
                    batched_output_corrections,
                    interaction,
                    family=family.value,
                    summaries=tuple(
                        summaries.outputs[target][group]
                        for target, group in batch
                    ),
                    rank_context=torch.stack(
                        tuple(rank_context[target] for target, _ in batch)
                    ),
                    event_weights=torch.stack(
                        tuple(state.event_weights[target] for target, _ in batch)
                    ),
                    assignment=assignment,
                    metadata=output_metadata,
                )
                correction = _batched_candidate_correction(call, cached)
                for row, (target, group) in enumerate(batch):
                    if observer is not None:
                        observer(
                            "output", operator.owners[target], correction[row, :, 0]
                        )
                    outputs[target][group].add(
                        grouped_outputs[target][group],
                        output_mass,
                        correction[row],
                    )
    if any(boundary.next_frame != frames for boundary in boundaries):
        raise NativeFactorError("cached interaction output boundary ended early")


def apply_compact_replay(
    operator: Any,
    prepared: CompactPrimalDualVideo,
    input_primals: tuple[torch.Tensor, ...],
    output_primals: tuple[torch.Tensor, ...],
    *,
    bank_set_interaction: EventConditionedBankSetInteraction | None = None,
    interaction_state: ProgramBankInteractionState | None = None,
    summaries: ProgramBankSetSummaries | None = None,
    direct_input_logit_biases: Sequence[torch.Tensor] | None = None,
    direct_output_logit_biases: Sequence[Sequence[torch.Tensor]] | None = None,
    correction_observer: Callable[[str, TargetOwner, torch.Tensor], None] | None = None,
    frozen_descriptors: FrozenReplayDescriptors | None = None,
    interaction_group_batch_size: int = 1,
    replay_plan: ReplayPlan | None = None,
) -> PrimalDualVideoResult:
    """B1 exact replay over raw X/Y with one finalized summary."""

    interaction_enabled, _ = _validate_replay_arguments(
        operator,
        bank_set_interaction=bank_set_interaction,
        interaction_state=interaction_state,
        summaries=summaries,
        direct_input_logit_biases=direct_input_logit_biases,
        direct_output_logit_biases=direct_output_logit_biases,
        correction_observer=correction_observer,
    )
    plan = (
        operator._plan(prepared, input_primals, output_primals)
        if replay_plan is None
        else replay_plan
    )
    if interaction_enabled and summaries is None:
        summaries = summarize_compact_replay(
            operator,
            prepared,
            plan=plan,
            bank_set_interaction=bank_set_interaction,
            interaction_state=interaction_state,
        )
    if interaction_group_batch_size <= 0 or (
        frozen_descriptors is not None and not interaction_enabled
    ):
        raise NativeFactorError("cached interaction replay contract changed")
    inputs, outputs = _new_pools(
        operator,
        plan,
        trusted_finite_bias=frozen_descriptors is not None,
    )
    if frozen_descriptors is not None:
        _replay_cached_interaction(
            operator,
            prepared,
            plan=plan,
            inputs=inputs,
            outputs=outputs,
            interaction=bank_set_interaction,
            state=interaction_state,
            summaries=summaries,
            descriptors=frozen_descriptors,
            group_batch_size=interaction_group_batch_size,
            observer=correction_observer,
        )
    else:
        for target in range(len(operator.owners)):
            _replay_target(
                operator,
                prepared,
                plan=plan,
                target=target,
                inputs=inputs,
                outputs=outputs,
                interaction=bank_set_interaction,
                state=interaction_state,
                summaries=summaries,
                direct_inputs=direct_input_logit_biases,
                direct_outputs=direct_output_logit_biases,
                observer=correction_observer,
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
