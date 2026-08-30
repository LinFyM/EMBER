"""Chunked current-video execution of the native primal-to-dual operator."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
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
from ember.ecp.contracts import ACTION_HORIZON, TargetOwner
from ember.ecp.native_factors import (
    G1_PROBE_COUNT,
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
class PreparedPrimalDualVideo:
    """Detached current-bank operator reusable across task-local updates."""

    video: Any
    frame_measure: torch.Tensor
    input_operators: tuple[SpectralNativeCovariance, ...]
    output_operators: tuple[tuple[SpectralNativeCovariance, ...], ...]


@dataclass(frozen=True)
class MaterializedPrimalDualVideo:
    """Fit-only fixed-bank replay cache; never used by deployment forward."""

    frame_measure: torch.Tensor
    input_operators: tuple[SpectralNativeCovariance, ...]
    output_operators: tuple[tuple[SpectralNativeCovariance, ...], ...]
    input_values: tuple[torch.Tensor, ...]
    output_values: tuple[torch.Tensor, ...]
    input_mass: torch.Tensor
    output_mass: torch.Tensor


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
        covariance_frame_chunk: int,
        inverse_covariance_power: float = 1.0,
    ) -> None:
        self.owners = tuple(owners)
        self.program_width = int(program_width)
        self.event_slots = int(event_slots)
        self.relative_eigenvalue_floor = float(relative_eigenvalue_floor)
        self.replay_score_rms = float(replay_score_rms)
        self.covariance_frame_chunk = int(covariance_frame_chunk)
        self.inverse_covariance_power = float(inverse_covariance_power)
        if self.inverse_covariance_power not in (0.5, 1.0):
            raise NativeFactorError("invalid native inverse covariance power")

    @staticmethod
    @contextmanager
    def ieee_matmul(device: torch.device):
        """Scope native capture/statistics/replay to the qualified IEEE mode."""

        if device.type != "cuda":
            yield
            return
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

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

    def _candidate_blocks(self) -> tuple[int, int]:
        input_block = (
            self.covariance_frame_chunk * G1_PROBE_COUNT * ACTION_HORIZON
        )
        return input_block, input_block * len(OUTPUT_BANK_TYPES)

    def _new_statistics(self, device: torch.device) -> tuple[list[Any], list[Any]]:
        input_block, output_block = self._candidate_blocks()
        inputs = [
            StreamingNativeCovariance(
                width=owner.in_features,
                device=device,
                canonical_block_candidates=input_block,
            )
            for owner in self.owners
        ]
        outputs = [
            tuple(
                StreamingNativeCovariance(
                    width=owner.out_features // native_output_group_count(owner),
                    device=device,
                    canonical_block_candidates=output_block,
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
            query, raw_rms, projection = operator.dual_and_score_rms(
                primal,
                inverse_covariance_power=self.inverse_covariance_power,
            )
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
                operator.dual_and_score_rms(
                    primals[group],
                    inverse_covariance_power=self.inverse_covariance_power,
                )
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

    def prepare(self, video: Any) -> PreparedPrimalDualVideo:
        """Read B0 once and retain only its detached spectral operator."""

        self.validate_video(video)
        with self.ieee_matmul(video.frame_positions.device):
            frame, input_stats, output_stats = self._covariance_statistics(video)
            input_operators, output_operators = self._solve_operators(
                input_stats, output_stats
            )
        return PreparedPrimalDualVideo(
            video=video,
            frame_measure=frame,
            input_operators=input_operators,
            output_operators=output_operators,
        )

    def materialize(
        self,
        prepared: PreparedPrimalDualVideo,
        *,
        value_dtype: torch.dtype | None = torch.float32,
    ) -> MaterializedPrimalDualVideo:
        """Pack a fixed fit-only bank once for repeated exact softmax replay."""

        if value_dtype not in (None, torch.bfloat16, torch.float32):
            raise NativeFactorError("materialized replay storage dtype changed")

        video = prepared.video
        frame = prepared.frame_measure
        input_blocks: list[list[torch.Tensor]] = [
            [] for _ in self.owners
        ]
        output_blocks: list[list[torch.Tensor]] = [
            [] for _ in self.owners
        ]
        input_mass, output_mass = [], []
        boundaries = [
            NativeOutputBankState(final=value.detach())
            for value in video.native.final_outputs
        ]
        next_frame = 0
        with torch.no_grad():
            for chunk in video.native.chunks():
                stop = next_frame + chunk.frame_count
                if chunk.start_frame != next_frame or stop > frame.shape[0]:
                    raise NativeFactorError("materialized replay stream changed")
                input_mass.append(
                    native_candidate_mass(frame[next_frame:stop], output=False)
                    .reshape(-1)
                    .float()
                )
                output_mass.append(
                    native_candidate_mass(frame[next_frame:stop], output=True)
                    .reshape(-1)
                    .float()
                )
                for target, (owner, x, y) in enumerate(
                    zip(self.owners, chunk.inputs, chunk.outputs, strict=True)
                ):
                    input_value = x.detach().reshape(-1, owner.in_features)
                    if value_dtype is not None:
                        input_value = input_value.to(dtype=value_dtype)
                    input_blocks[target].append(input_value)
                    bank = boundaries[target].build(y, start_frame=next_frame)
                    groups = native_output_group_count(owner)
                    width = owner.out_features // groups
                    grouped = bank.reshape(
                        *bank.shape[:-1], groups, width
                    ).movedim(-2, 0)
                    output_value = grouped.reshape(groups, -1, width)
                    if value_dtype is not None:
                        output_value = output_value.to(dtype=value_dtype)
                    output_blocks[target].append(output_value)
                next_frame = stop
        if next_frame != video.native.frame_count or any(
            boundary.next_frame != next_frame for boundary in boundaries
        ):
            raise NativeFactorError("materialized replay stream ended early")
        return MaterializedPrimalDualVideo(
            frame_measure=frame,
            input_operators=prepared.input_operators,
            output_operators=prepared.output_operators,
            input_values=tuple(torch.cat(rows, dim=0) for rows in input_blocks),
            output_values=tuple(torch.cat(rows, dim=1) for rows in output_blocks),
            input_mass=torch.cat(input_mass),
            output_mass=torch.cat(output_mass),
        )

    def compact(self, prepared: PreparedPrimalDualVideo) -> CompactPrimalDualVideo:
        """Seal raw native values once without expanding four output-bank types."""

        video = prepared.video
        input_blocks: list[list[torch.Tensor]] = [[] for _ in self.owners]
        output_blocks: list[list[torch.Tensor]] = [[] for _ in self.owners]
        next_frame = 0
        with torch.no_grad():
            for chunk in video.native.chunks():
                if chunk.start_frame != next_frame or chunk.frame_count <= 0:
                    raise NativeFactorError("compact replay stream changed")
                for target, (x, y) in enumerate(
                    zip(chunk.inputs, chunk.outputs, strict=True)
                ):
                    input_blocks[target].append(x.detach())
                    output_blocks[target].append(y.detach())
                next_frame += chunk.frame_count
        if next_frame != video.native.frame_count:
            raise NativeFactorError("compact replay stream ended early")
        return CompactPrimalDualVideo(
            frame_measure=prepared.frame_measure,
            input_operators=prepared.input_operators,
            output_operators=prepared.output_operators,
            input_values=tuple(torch.cat(rows, dim=0) for rows in input_blocks),
            output_values=tuple(torch.cat(rows, dim=0) for rows in output_blocks),
            final_outputs=tuple(value.detach() for value in video.native.final_outputs),
        )

    def _plan(
        self,
        prepared: (
            PreparedPrimalDualVideo
            | MaterializedPrimalDualVideo
            | CompactPrimalDualVideo
        ),
        input_primals: tuple[torch.Tensor, ...],
        output_primals: tuple[torch.Tensor, ...],
    ) -> _ReplayPlan:
        frame = prepared.frame_measure
        input_operators = prepared.input_operators
        output_operators = prepared.output_operators
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

    @staticmethod
    def _materialized_signed_pool(
        query: torch.Tensor,
        values: torch.Tensor,
        mass: torch.Tensor,
    ) -> torch.Tensor:
        log_mass = mass.to(query).log()
        if query.ndim == 2 and values.ndim == 2:
            score = query.float() @ values.float().T
            signed = (score + log_mass).softmax(-1) - (
                -score + log_mass
            ).softmax(-1)
            return (signed @ values.float()).to(query)
        if query.ndim == 3 and values.ndim == 3:
            score = torch.einsum("grd,gnd->grn", query.float(), values.float())
            signed = (score + log_mass[None, None]).softmax(-1) - (
                -score + log_mass[None, None]
            ).softmax(-1)
            return torch.einsum(
                "grn,gnd->grd", signed, values.float()
            ).to(query)
        raise NativeFactorError("materialized replay axes changed")

    def apply_materialized(
        self,
        prepared: MaterializedPrimalDualVideo,
        input_primals: tuple[torch.Tensor, ...],
        output_primals: tuple[torch.Tensor, ...],
    ) -> PrimalDualVideoResult:
        """Replay a fixed diagnostic bank without repeated Python chunk launches."""

        with self.ieee_matmul(prepared.frame_measure.device):
            plan = self._plan(prepared, input_primals, output_primals)
            inputs = tuple(
                self._materialized_signed_pool(query, values, prepared.input_mass)
                for query, values in zip(
                    plan.input_queries, prepared.input_values, strict=True
                )
            )
            grouped_outputs = tuple(
                self._materialized_signed_pool(
                    torch.stack(queries), values, prepared.output_mass
                )
                for queries, values in zip(
                    plan.output_queries, prepared.output_values, strict=True
                )
            )
            outputs = tuple(
                value.permute(1, 0, 2).reshape(G1_RESIDUAL_RANK, -1)
                for value in grouped_outputs
            )
        return PrimalDualVideoResult(
            input_values=inputs,
            output_values=outputs,
            frame_measure=plan.frame_measure,
            group_gains=plan.group_gains,
            solve_metrics=plan.solve_metrics,
            conditioning_metrics=plan.conditioning_metrics,
        )

    def apply_compact(
        self,
        prepared: CompactPrimalDualVideo,
        input_primals: tuple[torch.Tensor, ...],
        output_primals: tuple[torch.Tensor, ...],
    ) -> PrimalDualVideoResult:
        """Replay cached raw X/Y with the canonical fixed-microblock reduction."""

        with self.ieee_matmul(prepared.frame_measure.device):
            plan = self._plan(prepared, input_primals, output_primals)
            input_block, output_block = self._candidate_blocks()
            inputs = tuple(
                StreamingSignedPool(
                    query,
                    trusted_positive_measure=True,
                    canonical_block_candidates=input_block,
                )
                for query in plan.input_queries
            )
            outputs = tuple(
                tuple(
                    StreamingSignedPool(
                        query,
                        trusted_positive_measure=True,
                        canonical_block_candidates=output_block,
                    )
                    for query in groups
                )
                for groups in plan.output_queries
            )
            x_mass = native_candidate_mass(prepared.frame_measure, output=False)
            y_mass = native_candidate_mass(prepared.frame_measure, output=True)
            for target, (owner, x, y) in enumerate(
                zip(
                    self.owners,
                    prepared.input_values,
                    prepared.output_values,
                    strict=True,
                )
            ):
                inputs[target].add(x, x_mass)
                boundary = NativeOutputBankState(
                    final=prepared.final_outputs[target].detach()
                )
                bank = boundary.build(y, start_frame=0)
                if boundary.next_frame != prepared.frame_measure.shape[0]:
                    raise NativeFactorError("compact output boundary ended early")
                groups = native_output_group_count(owner)
                grouped = bank.reshape(
                    *bank.shape[:-1], groups, owner.out_features // groups
                ).movedim(-2, 0)
                for group, accumulator in enumerate(outputs[target]):
                    accumulator.add(grouped[group], y_mass)
        return PrimalDualVideoResult(
            input_values=tuple(value.signed_mean() for value in inputs),
            output_values=tuple(
                torch.cat(
                    tuple(value.signed_mean() for value in groups), dim=-1
                )
                for groups in outputs
            ),
            frame_measure=plan.frame_measure,
            group_gains=plan.group_gains,
            solve_metrics=plan.solve_metrics,
            conditioning_metrics=plan.conditioning_metrics,
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
        input_block, output_block = self._candidate_blocks()
        inputs = tuple(
            StreamingSignedPool(
                query,
                trusted_positive_measure=True,
                canonical_block_candidates=input_block,
            )
            for query in plan.input_queries
        )
        outputs = tuple(
            tuple(
                StreamingSignedPool(
                    query,
                    trusted_positive_measure=True,
                    canonical_block_candidates=output_block,
                )
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

    def apply(
        self,
        prepared: PreparedPrimalDualVideo,
        input_primals: tuple[torch.Tensor, ...],
        output_primals: tuple[torch.Tensor, ...],
    ) -> PrimalDualVideoResult:
        with self.ieee_matmul(prepared.frame_measure.device):
            plan = self._plan(prepared, input_primals, output_primals)
            inputs, outputs = self._replay(prepared.video, plan)
        return PrimalDualVideoResult(
            input_values=inputs,
            output_values=outputs,
            frame_measure=plan.frame_measure,
            group_gains=plan.group_gains,
            solve_metrics=plan.solve_metrics,
            conditioning_metrics=plan.conditioning_metrics,
        )

    def __call__(
        self,
        video: Any,
        input_primals: tuple[torch.Tensor, ...],
        output_primals: tuple[torch.Tensor, ...],
    ) -> PrimalDualVideoResult:
        return self.apply(self.prepare(video), input_primals, output_primals)
