"""Run-local frozen query-relative candidate descriptors for fixed-route EBSRI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ember.ecp.bank_conditioning.set_summary import (
    candidate_metadata,
    program_relative_coordinates,
)
from ember.ecp.native_factors import NativeFactorError, NativeOutputBankState


@dataclass(frozen=True)
class CandidateDescriptor:
    coordinates: torch.Tensor
    base_score: torch.Tensor
    log_norm: torch.Tensor

    def frame_slice(
        self, start: int, stop: int, *, device: torch.device
    ) -> CandidateDescriptor:
        if not 0 <= start < stop <= self.coordinates.shape[0]:
            raise NativeFactorError("candidate descriptor frame slice changed")
        return CandidateDescriptor(
            coordinates=self.coordinates[start:stop].to(device),
            base_score=self.base_score[:, start:stop].to(device),
            log_norm=self.log_norm[start:stop].to(device),
        )


@dataclass(frozen=True)
class FrozenReplayDescriptors:
    """CPU-resident descriptors; real X/Y remain the only pooling values."""

    input_metadata: torch.Tensor
    output_metadata: torch.Tensor
    inputs: tuple[CandidateDescriptor, ...]
    outputs: tuple[tuple[CandidateDescriptor, ...], ...]

    def metadata_slice(
        self, start: int, stop: int, *, output: bool, device: torch.device
    ) -> torch.Tensor:
        source = self.output_metadata if output else self.input_metadata
        return source[start:stop].to(device)


def _descriptor(
    *,
    values: torch.Tensor,
    native_mean: torch.Tensor,
    native_event_query: torch.Tensor,
    base_query: torch.Tensor,
    replay_score_rms: float,
) -> CandidateDescriptor:
    centered = values.float() - native_mean.detach().float()
    return CandidateDescriptor(
        coordinates=program_relative_coordinates(
            native_event_query, values, native_mean
        ),
        base_score=torch.einsum(
            "rd,...d->r...", base_query.detach().float(), centered
        )
        / replay_score_rms,
        log_norm=centered.square().mean(-1).clamp_min(1e-12).sqrt().log(),
    )


def _freeze(value: CandidateDescriptor) -> CandidateDescriptor:
    return CandidateDescriptor(
        coordinates=value.coordinates.detach().cpu().contiguous(),
        base_score=value.base_score.detach().cpu().contiguous(),
        log_norm=value.log_norm.detach().cpu().contiguous(),
    )


def build_frozen_replay_descriptors(
    operator: Any,
    prepared: Any,
    *,
    plan: Any,
    interaction_state: Any,
) -> FrozenReplayDescriptors:
    """Precompute fixed-route kappa/base-score/log-norm once per video bank."""

    operator._validate_interaction_state(interaction_state)
    frames = int(prepared.frame_measure.shape[0])
    context = interaction_state.context
    input_metadata = candidate_metadata(
        context.frame_positions,
        output=False,
        like=prepared.input_values[0],
    ).detach().cpu().contiguous()
    output_metadata = candidate_metadata(
        context.frame_positions,
        output=True,
        like=prepared.output_values[0],
    ).detach().cpu().contiguous()
    inputs = []
    outputs = []
    with torch.no_grad():
        for target, (owner, x, y) in enumerate(
            zip(operator.owners, prepared.input_values, prepared.output_values, strict=True)
        ):
            input_chunks = []
            output_chunks = [
                [] for _ in range(len(prepared.output_operators[target]))
            ]
            boundary = NativeOutputBankState(
                final=prepared.final_outputs[target].detach()
            )
            for start in range(0, frames, operator.covariance_frame_chunk):
                stop = min(start + operator.covariance_frame_chunk, frames)
                input_chunks.append(
                    _freeze(
                        _descriptor(
                            values=x[start:stop],
                            native_mean=prepared.input_operators[target].mean,
                            native_event_query=(
                                interaction_state.input_event_queries[target]
                            ),
                            base_query=plan.input_queries[target],
                            replay_score_rms=operator.replay_score_rms,
                        )
                    )
                )
                bank = boundary.build(y[start:stop], start_frame=start)
                groups = len(output_chunks)
                grouped = bank.reshape(
                    *bank.shape[:-1], groups, owner.out_features // groups
                ).movedim(-2, 0)
                for group in range(groups):
                    output_chunks[group].append(
                        _freeze(
                            _descriptor(
                                values=grouped[group],
                                native_mean=(
                                    prepared.output_operators[target][group].mean
                                ),
                                native_event_query=(
                                    interaction_state.output_event_queries[target][group]
                                ),
                                base_query=plan.output_queries[target][group],
                                replay_score_rms=operator.replay_score_rms,
                            )
                        )
                    )
            if boundary.next_frame != frames:
                raise NativeFactorError("descriptor output boundary ended early")
            inputs.append(_concatenate(input_chunks))
            outputs.append(
                tuple(_concatenate(chunks) for chunks in output_chunks)
            )
    return FrozenReplayDescriptors(
        input_metadata=input_metadata,
        output_metadata=output_metadata,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
    )


def _concatenate(chunks: list[CandidateDescriptor]) -> CandidateDescriptor:
    if not chunks:
        raise NativeFactorError("candidate descriptor cache is empty")
    return CandidateDescriptor(
        coordinates=torch.cat(tuple(value.coordinates for value in chunks), dim=0),
        base_score=torch.cat(tuple(value.base_score for value in chunks), dim=1),
        log_norm=torch.cat(tuple(value.log_norm for value in chunks), dim=0),
    )
