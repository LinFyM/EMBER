"""Chunked current-video execution of the native primal-to-dual operator."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from ember.ecp.bank_conditioning.operator import StreamingSignedPool
from ember.ecp.bank_conditioning.primal_dual import (
    NativeCovarianceStatistics,
    SpectralNativeCovariance,
    StreamingNativeCovariance,
    batched_spectral_native_covariances,
    native_candidate_mass,
)
from ember.ecp.contracts import TargetOwner
from ember.ecp.native_factors import (
    G1_RESIDUAL_RANK,
    NativeFactorError,
    NativeOutputBankState,
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
class _ReplayPlan:
    input_queries: tuple[torch.Tensor, ...]
    output_queries: tuple[tuple[torch.Tensor, ...], ...]
    frame_measure: torch.Tensor
    group_gains: torch.Tensor
    solve_metrics: torch.Tensor
    conditioning_metrics: torch.Tensor


class PrimalDualVideoOperator:
    """Own B0 covariance, spectral dualization, and exact B1 replay."""

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        program_width: int,
        event_slots: int,
        relative_eigenvalue_floor: float,
        replay_score_rms: float,
    ) -> None:
        self.owners = tuple(owners)
        self.program_width = int(program_width)
        self.event_slots = int(event_slots)
        self.relative_eigenvalue_floor = float(relative_eigenvalue_floor)
        self.replay_score_rms = float(replay_score_rms)

    @staticmethod
    def quadrature(positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim != 1 or positions.numel() <= 0:
            raise NativeFactorError("compiler frame positions changed")
        if positions.numel() == 1:
            return torch.ones_like(positions, dtype=torch.float32)
        points = positions.float()
        if torch.any(points[1:] < points[:-1]):
            raise NativeFactorError("compiler video is not internally ordered")
        gaps = points[1:] - points[:-1]
        weights = torch.empty_like(points)
        weights[0] = gaps[0] * 0.5
        weights[-1] = gaps[-1] * 0.5
        if points.numel() > 2:
            weights[1:-1] = (gaps[:-1] + gaps[1:]) * 0.5
        if float(weights.sum()) <= 1e-8:
            weights.fill_(1.0)
        return weights.clamp_min(1e-8) / weights.sum().clamp_min(1e-8)

    def validate_video(self, video: Any) -> None:
        targets = len(self.owners)
        frames = video.native.frame_count
        if (
            frames <= 0
            or video.canonical_assignment.shape != (frames, self.event_slots)
            or video.frame_positions.shape != (frames,)
            or video.local_scene.shape != (targets, self.program_width)
            or video.local_process.shape
            != (self.event_slots, targets, self.program_width)
            or video.local_presence.shape != (self.event_slots,)
            or video.local_tau.shape != (self.event_slots, 2)
            or video.local_sigma.shape
            != (self.event_slots, targets, self.program_width)
            or len(video.native.final_outputs) != targets
        ):
            raise NativeFactorError("compiler local video contract changed")

    def _new_statistics(self, device: torch.device) -> tuple[list[Any], list[Any]]:
        inputs = [
            StreamingNativeCovariance(width=owner.in_features, device=device)
            for owner in self.owners
        ]
        outputs = [
            tuple(
                StreamingNativeCovariance(
                    width=owner.out_features // native_output_group_count(owner),
                    device=device,
                )
                for _ in range(native_output_group_count(owner))
            )
            for owner in self.owners
        ]
        return inputs, outputs

    def _add_statistics_chunk(
        self,
        *,
        chunk: Any,
        start: int,
        frame_measure: torch.Tensor,
        inputs: list[Any],
        outputs: list[Any],
        boundaries: list[NativeOutputBankState],
    ) -> int:
        stop = start + chunk.frame_count
        if (
            chunk.start_frame != start
            or len(chunk.inputs) != len(self.owners)
            or len(chunk.outputs) != len(self.owners)
        ):
            raise NativeFactorError("compiler B0 native stream changed")
        x_mass = native_candidate_mass(frame_measure[start:stop], output=False)
        y_mass = native_candidate_mass(frame_measure[start:stop], output=True)
        for target, (owner, x, y) in enumerate(
            zip(self.owners, chunk.inputs, chunk.outputs, strict=True)
        ):
            inputs[target].add(x, x_mass)
            bank = boundaries[target].build(y, start_frame=start)
            groups = native_output_group_count(owner)
            grouped = bank.reshape(
                *bank.shape[:-1], groups, owner.out_features // groups
            ).movedim(-2, 0)
            for group, accumulator in enumerate(outputs[target]):
                accumulator.add(grouped[group], y_mass)
        return stop

    def _covariance_statistics(
        self, video: Any
    ) -> tuple[torch.Tensor, tuple[Any, ...], tuple[tuple[Any, ...], ...]]:
        frame_measure = self.quadrature(video.frame_positions)
        inputs, outputs = self._new_statistics(frame_measure.device)
        boundaries = [
            NativeOutputBankState(final=value.detach())
            for value in video.native.final_outputs
        ]
        next_frame = 0
        with torch.no_grad():
            for chunk in video.native.chunks():
                next_frame = self._add_statistics_chunk(
                    chunk=chunk,
                    start=next_frame,
                    frame_measure=frame_measure,
                    inputs=inputs,
                    outputs=outputs,
                    boundaries=boundaries,
                )
        if next_frame != video.native.frame_count or any(
            boundary.next_frame != next_frame for boundary in boundaries
        ):
            raise NativeFactorError("compiler B0 native stream ended early")
        return (
            frame_measure,
            tuple(value.finalize() for value in inputs),
            tuple(tuple(value.finalize() for value in groups) for groups in outputs),
        )

    def _solve_operators(
        self,
        inputs: tuple[NativeCovarianceStatistics, ...],
        outputs: tuple[tuple[NativeCovarianceStatistics, ...], ...],
    ) -> tuple[tuple[Any, ...], tuple[tuple[Any, ...], ...]]:
        entries = [
            (("input", target, 0), value)
            for target, value in enumerate(inputs)
        ]
        entries.extend(
            (("output", target, group), value)
            for target, groups in enumerate(outputs)
            for group, value in enumerate(groups)
        )
        buckets: dict[int, list[Any]] = defaultdict(list)
        for key, value in entries:
            buckets[value.covariance.shape[0]].append((key, value))
        solved = {}
        for rows in buckets.values():
            for start in range(0, len(rows), 8):
                batch = rows[start : start + 8]
                operators = batched_spectral_native_covariances(
                    tuple(row[1] for row in batch),
                    relative_eigenvalue_floor=self.relative_eigenvalue_floor,
                )
                solved.update(
                    (row[0], operator)
                    for row, operator in zip(batch, operators, strict=True)
                )
        return (
            tuple(solved[("input", target, 0)] for target in range(len(inputs))),
            tuple(
                tuple(
                    solved[("output", target, group)]
                    for group in range(len(groups))
                )
                for target, groups in enumerate(outputs)
            ),
        )

    @staticmethod
    def _metric_row(
        side: int,
        target: int,
        group: int,
        operator: SpectralNativeCovariance,
        projection: torch.Tensor,
    ) -> torch.Tensor:
        scalars = projection.new_tensor(
            (
                side,
                target,
                group,
                operator.native_width,
                operator.retained_rank,
            )
        )
        return torch.cat(
            (
                scalars,
                operator.eigenvalue_floor.to(projection).reshape(1),
                operator.retained_condition.to(projection).reshape(1),
                operator.retained_trace_fraction.to(projection).reshape(1),
                projection.detach().min().reshape(1),
            )
        )

    def _queries(
        self,
        input_primals: tuple[torch.Tensor, ...],
        output_primals: tuple[torch.Tensor, ...],
        input_operators: tuple[SpectralNativeCovariance, ...],
        output_operators: tuple[tuple[SpectralNativeCovariance, ...], ...],
    ) -> tuple[Any, ...]:
        input_queries, output_queries, metrics = [], [], []
        projections, raw_rms_values, query_scales = [], [], []
        for target, (primal, operator) in enumerate(
            zip(input_primals, input_operators, strict=True)
        ):
            query, raw_rms, projection = operator.dual_and_score_rms(primal)
            scale = self.replay_score_rms / raw_rms.clamp_min(1e-12)
            input_queries.append(query * scale[:, None])
            projections.append(projection)
            raw_rms_values.append(raw_rms)
            query_scales.append(scale)
            metrics.append(self._metric_row(0, target, 0, operator, projection))
        for target, (primals, operators) in enumerate(
            zip(output_primals, output_operators, strict=True)
        ):
            rows = tuple(
                operator.dual_and_score_rms(primals[group])
                for group, operator in enumerate(operators)
            )
            rms = torch.stack(tuple(row[1] for row in rows))
            scale = self.replay_score_rms / rms.amax(0).clamp_min(1e-12)
            output_queries.append(tuple(row[0] * scale[:, None] for row in rows))
            query_scales.append(scale)
            for group, (operator, row) in enumerate(
                zip(operators, rows, strict=True)
            ):
                projections.append(row[2])
                raw_rms_values.append(row[1])
                metrics.append(self._metric_row(1, target, group, operator, row[2]))
        return (
            tuple(input_queries),
            tuple(output_queries),
            torch.stack(metrics),
            projections,
            raw_rms_values,
            query_scales,
        )

    def _plan(
        self,
        video: Any,
        input_primals: tuple[torch.Tensor, ...],
        output_primals: tuple[torch.Tensor, ...],
    ) -> _ReplayPlan:
        frame, input_stats, output_stats = self._covariance_statistics(video)
        input_operators, output_operators = self._solve_operators(
            input_stats, output_stats
        )
        queries = self._queries(
            input_primals, output_primals, input_operators, output_operators
        )
        operators = (*input_operators, *(row for rows in output_operators for row in rows))
        retained_fractions = frame.new_tensor(
            tuple(op.retained_rank / op.native_width for op in operators)
        )
        retained_traces = torch.stack(
            tuple(op.retained_trace_fraction.to(frame) for op in operators)
        )
        retained_conditions = torch.stack(
            tuple(op.retained_condition.to(frame) for op in operators)
        )
        conditioning = torch.stack(
            (
                retained_fractions.min(),
                retained_traces.min(),
                retained_conditions.max(),
                torch.cat(queries[3]).min(),
                torch.cat(queries[4]).min(),
                torch.cat(queries[5]).max(),
            )
        ).detach()
        group_count = sum(native_output_group_count(owner) for owner in self.owners)
        return _ReplayPlan(
            input_queries=queries[0],
            output_queries=queries[1],
            frame_measure=frame[None, None].expand(
                len(self.owners), G1_RESIDUAL_RANK, -1
            ),
            group_gains=frame.new_ones(group_count, G1_RESIDUAL_RANK),
            solve_metrics=queries[2],
            conditioning_metrics=conditioning,
        )

    def _add_replay_chunk(
        self,
        *,
        chunk: Any,
        start: int,
        frame: torch.Tensor,
        inputs: tuple[Any, ...],
        outputs: tuple[tuple[Any, ...], ...],
        boundaries: list[NativeOutputBankState],
    ) -> int:
        stop = start + chunk.frame_count
        if (
            chunk.start_frame != start
            or stop > frame.shape[0]
            or len(chunk.inputs) != len(self.owners)
            or len(chunk.outputs) != len(self.owners)
        ):
            raise NativeFactorError("compiler B1 native stream changed")
        x_mass = native_candidate_mass(frame[start:stop], output=False)
        y_mass = native_candidate_mass(frame[start:stop], output=True)
        for target, (owner, x, y) in enumerate(
            zip(self.owners, chunk.inputs, chunk.outputs, strict=True)
        ):
            inputs[target].add(x, x_mass)
            bank = boundaries[target].build(y, start_frame=start)
            groups = native_output_group_count(owner)
            grouped = bank.reshape(
                *bank.shape[:-1], groups, owner.out_features // groups
            ).movedim(-2, 0)
            for group, accumulator in enumerate(outputs[target]):
                accumulator.add(grouped[group], y_mass)
        return stop

    def _replay(self, video: Any, plan: _ReplayPlan) -> tuple[Any, Any]:
        frame = self.quadrature(video.frame_positions)
        inputs = tuple(
            StreamingSignedPool(query, trusted_positive_measure=True)
            for query in plan.input_queries
        )
        outputs = tuple(
            tuple(
                StreamingSignedPool(query, trusted_positive_measure=True)
                for query in groups
            )
            for groups in plan.output_queries
        )
        boundaries = [
            NativeOutputBankState(final=value.detach())
            for value in video.native.final_outputs
        ]
        next_frame = 0
        for chunk in video.native.chunks():
            next_frame = self._add_replay_chunk(
                chunk=chunk,
                start=next_frame,
                frame=frame,
                inputs=inputs,
                outputs=outputs,
                boundaries=boundaries,
            )
        if next_frame != video.native.frame_count or any(
            boundary.next_frame != next_frame for boundary in boundaries
        ):
            raise NativeFactorError("compiler B1 native stream ended early")
        return (
            tuple(value.signed_mean() for value in inputs),
            tuple(
                torch.cat(tuple(value.signed_mean() for value in groups), dim=-1)
                for groups in outputs
            ),
        )

    def __call__(
        self,
        video: Any,
        input_primals: tuple[torch.Tensor, ...],
        output_primals: tuple[torch.Tensor, ...],
    ) -> PrimalDualVideoResult:
        plan = self._plan(video, input_primals, output_primals)
        inputs, outputs = self._replay(video, plan)
        return PrimalDualVideoResult(
            input_values=inputs,
            output_values=outputs,
            frame_measure=plan.frame_measure,
            group_gains=plan.group_gains,
            solve_metrics=plan.solve_metrics,
            conditioning_metrics=plan.conditioning_metrics,
        )
