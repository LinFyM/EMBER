"""Memory-bounded backward for singleton cached EBSRI replay."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

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
from ember.ecp.bank_conditioning.primal_dual import native_candidate_mass
from ember.ecp.contracts import TargetFamily
from ember.ecp.native_factors import NativeFactorError, native_output_group_count


def checkpointed_singleton_pool_add(
    pool: StreamingSignedPool,
    *,
    values: torch.Tensor | Callable[[], torch.Tensor],
    mass: torch.Tensor,
    correction: Callable[[], torch.Tensor],
    final_chunk: bool,
) -> None:
    """Retain only online-softmax state and recompute one complete transition."""

    if pool._pending_values.numel() or pool._pending_bias is not None:
        raise NativeFactorError("checkpointed signed replay acquired pending state")
    flat_mass = mass.detach().to(pool.query).reshape(-1)
    candidates = int(flat_mass.shape[0])
    block = pool.canonical_block_candidates
    if block is not None and not (
        candidates == block or (final_chunk and 0 < candidates < block)
    ):
        raise NativeFactorError("checkpointed signed replay block changed")

    def update(
        maximum: torch.Tensor,
        normalizer: torch.Tensor,
        weighted_sum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        native = values() if callable(values) else values
        flat_values = native.detach().to(pool.query).reshape(-1, pool.width)
        if flat_values.shape[0] != candidates:
            raise NativeFactorError("checkpointed signed replay values changed")
        bias = correction().to(pool.query)
        expected = (*pool.query_shape, 2, *mass.shape)
        if bias.shape != expected:
            raise NativeFactorError("checkpointed signed replay bias axes changed")
        return pool._accumulated_state(
            flat_values,
            flat_mass,
            bias.reshape(*pool.query_shape, 2, candidates),
            maximum,
            normalizer,
            weighted_sum,
        )

    pool.maximum, pool.normalizer, pool.weighted_sum = checkpoint(
        update,
        pool.maximum,
        pool.normalizer,
        pool.weighted_sum,
        use_reentrant=False,
        preserve_rng_state=False,
    )
    pool.candidate_count += candidates
    pool._uses_bias = True


def _output_values(
    operator: Any,
    prepared: Any,
    *,
    target: int,
    group: int,
    start: int,
    stop: int,
) -> torch.Tensor:
    """Rebuild one temporal output-bank group without retaining the full bank."""

    owner = operator.owners[target]
    groups = native_output_group_count(owner)
    width = owner.out_features // groups
    column = slice(group * width, (group + 1) * width)
    complete = prepared.output_values[target][..., column]
    raw = complete[start:stop]
    first = complete[0]
    previous = first if start == 0 else complete[start - 1]
    adjacent_previous = torch.cat((previous[None], raw[:-1]), dim=0)
    return torch.stack(
        (
            raw,
            raw - adjacent_previous,
            raw - first,
            prepared.final_outputs[target][..., column].detach() - raw,
        ),
        dim=3,
    )


def _correction(
    call: Callable[..., torch.Tensor],
    descriptor: CandidateDescriptor,
    *,
    start: int,
    stop: int,
    device: torch.device,
) -> torch.Tensor:
    return call(
        descriptors=(descriptor.frame_slice(start, stop, device=device),)
    )[0]


def _replay_inputs(
    operator: Any,
    prepared: Any,
    *,
    inputs: Any,
    interaction: Any,
    state: Any,
    summaries: Any,
    descriptors: FrozenReplayDescriptors,
    structural_gates: tuple[torch.Tensor, ...],
    start: int,
    stop: int,
    assignment: torch.Tensor,
    mass: torch.Tensor,
    metadata: torch.Tensor,
    final_chunk: bool,
) -> None:
    device = prepared.frame_measure.device
    for family in TargetFamily:
        targets = tuple(
            target
            for target, owner in enumerate(operator.owners)
            if owner.family is family
        )
        for target in targets:
            call = partial(
                batched_input_corrections,
                interaction,
                family=family.value,
                summaries=(summaries.inputs[target],),
                structural_gate=structural_gates[target][None],
                event_weights=state.event_weights[target][None],
                assignment=assignment,
                metadata=metadata,
            )
            checkpointed_singleton_pool_add(
                inputs[target],
                values=prepared.input_values[target][start:stop],
                mass=mass,
                correction=partial(
                    _correction,
                    call,
                    descriptors.inputs[target],
                    start=start,
                    stop=stop,
                    device=device,
                ),
                final_chunk=final_chunk,
            )


def _replay_outputs(
    operator: Any,
    prepared: Any,
    *,
    outputs: Any,
    interaction: Any,
    state: Any,
    summaries: Any,
    descriptors: FrozenReplayDescriptors,
    structural_gates: tuple[torch.Tensor, ...],
    start: int,
    stop: int,
    assignment: torch.Tensor,
    mass: torch.Tensor,
    metadata: torch.Tensor,
    final_chunk: bool,
) -> None:
    device = prepared.frame_measure.device
    for family in TargetFamily:
        entries = tuple(
            (target, group)
            for target, owner in enumerate(operator.owners)
            if owner.family is family
            for group in range(native_output_group_count(owner))
        )
        for target, group in entries:
            call = partial(
                batched_output_corrections,
                interaction,
                family=family.value,
                summaries=(summaries.outputs[target][group],),
                structural_gate=structural_gates[target][None],
                event_weights=state.event_weights[target][None],
                assignment=assignment,
                metadata=metadata,
            )
            checkpointed_singleton_pool_add(
                outputs[target][group],
                values=partial(
                    _output_values,
                    operator,
                    prepared,
                    target=target,
                    group=group,
                    start=start,
                    stop=stop,
                ),
                mass=mass,
                correction=partial(
                    _correction,
                    call,
                    descriptors.outputs[target][group],
                    start=start,
                    stop=stop,
                    device=device,
                ),
                final_chunk=final_chunk,
            )


def replay_checkpointed_singletons(
    operator: Any,
    prepared: Any,
    *,
    inputs: Any,
    outputs: Any,
    interaction: Any,
    state: Any,
    summaries: Any,
    descriptors: FrozenReplayDescriptors,
) -> None:
    """Exact singleton replay with peak memory bounded by one frame chunk."""

    frames = int(prepared.frame_measure.shape[0])
    device = prepared.frame_measure.device
    structural_gates = tuple(
        interaction._b1_structural_gate(target)
        for target in range(len(operator.owners))
    )
    for start in range(0, frames, operator.covariance_frame_chunk):
        stop = min(start + operator.covariance_frame_chunk, frames)
        measure = prepared.frame_measure[start:stop]
        assignment = state.context.canonical_assignment[start:stop]
        final_chunk = stop == frames
        _replay_inputs(
            operator,
            prepared,
            inputs=inputs,
            interaction=interaction,
            state=state,
            summaries=summaries,
            descriptors=descriptors,
            structural_gates=structural_gates,
            start=start,
            stop=stop,
            assignment=assignment,
            mass=native_candidate_mass(measure, output=False),
            metadata=descriptors.metadata_slice(
                start, stop, output=False, device=device
            ),
            final_chunk=final_chunk,
        )
        _replay_outputs(
            operator,
            prepared,
            outputs=outputs,
            interaction=interaction,
            state=state,
            summaries=summaries,
            descriptors=descriptors,
            structural_gates=structural_gates,
            start=start,
            stop=stop,
            assignment=assignment,
            mass=native_candidate_mass(measure, output=True),
            metadata=descriptors.metadata_slice(
                start, stop, output=True, device=device
            ),
            final_chunk=final_chunk,
        )
