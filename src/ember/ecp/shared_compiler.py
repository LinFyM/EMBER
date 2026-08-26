"""Current-bank-conditioned Pass B for the frozen-Program G3 compiler."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.checkpoint import checkpoint

from ember.ecp.bank_conditioning import (
    AnchorProgramState,
    BankStatistics,
    ProgramNativeAnchorScorer,
    SpectralBankQuery,
    StreamingBankStatistics,
    StreamingSignedPool,
    batched_spectral_bank_query,
)
from ember.ecp.contracts import ACTION_HORIZON, TargetOwner
from ember.ecp.native_factors import (
    G1_PROBE_COUNT,
    G1_RESIDUAL_RANK,
    OUTPUT_BANK_TYPES,
    NativeFactorError,
    NativeFactorResidual,
    NativeOutputBankState,
    NativeVideoReadout,
    native_output_group_count,
    rms_normalize,
)
from ember.ecp.natural_program import NaturalProgram


@dataclass(frozen=True)
class SharedCompilerVideo:
    """One independently ordered video and its frozen G2 alignment evidence."""

    native: NativeVideoReadout
    canonical_assignment: torch.Tensor
    frame_positions: torch.Tensor
    local_scene: torch.Tensor
    local_process: torch.Tensor
    local_presence: torch.Tensor
    local_tau: torch.Tensor
    local_sigma: torch.Tensor


@dataclass(frozen=True)
class SharedCompilerOutput:
    residual: NativeFactorResidual
    input_directions: tuple[torch.Tensor, ...]
    output_directions: tuple[torch.Tensor, ...]
    video_weights: torch.Tensor
    frame_measures: tuple[torch.Tensor, ...]
    output_group_gains: tuple[torch.Tensor, ...]
    solve_metrics: torch.Tensor
    global_statistics_enabled: bool


@dataclass(frozen=True)
class _VideoBankPlan:
    input_queries: tuple[torch.Tensor, ...]
    output_queries: tuple[tuple[torch.Tensor, ...], ...]
    frame_measure: torch.Tensor
    group_gains: torch.Tensor
    solve_metrics: torch.Tensor


@dataclass
class _StatisticsStream:
    base_frame: torch.Tensor
    event_frame: torch.Tensor
    frame_measure: torch.Tensor
    input_anchor_queries: tuple[tuple[torch.Tensor, ...], ...]
    input_accumulators: list[StreamingBankStatistics]
    output_accumulators: list[tuple[StreamingBankStatistics, ...]]
    output_anchor_queries: tuple[tuple[torch.Tensor, ...], ...]
    gains: tuple[torch.Tensor, ...]
    boundaries: list[NativeOutputBankState]


class SharedNativeFactorCompiler(torch.nn.Module):
    """Generate one rank-four residual through B0 solve and B1 exact replay.

    The scorer contains no task-, video-, member-, or frame-indexed parameters.
    B0 converts bounded Program/candidate compatibilities into native anchors
    and conditions them on the current video bank. B1 rereads that same bank
    and only then pools its real X/Y values with two explicit softmax branches.
    """

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        program_width: int = 128,
        event_slots: int = 8,
        anchor_width: int = 128,
        relative_eigenvalue_floor: float = 1e-6,
        global_statistics: bool = True,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.program_width = int(program_width)
        self.event_slots = int(event_slots)
        self.anchor_width = int(anchor_width)
        self.relative_eigenvalue_floor = float(relative_eigenvalue_floor)
        self.global_statistics = bool(global_statistics)
        if (
            not self.owners
            or self.program_width <= 0
            or self.event_slots <= 0
            or self.anchor_width <= 0
            or not 0.0 < self.relative_eigenvalue_floor < 1.0
        ):
            raise NativeFactorError("invalid bank-conditioned compiler topology")

        output_counts = tuple(native_output_group_count(owner) for owner in self.owners)
        offsets = [0]
        for count in output_counts:
            offsets.append(offsets[-1] + count)
        self.output_group_slices = tuple(
            slice(start, stop)
            for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
        )
        self.register_buffer(
            "output_group_counts",
            torch.tensor(output_counts, dtype=torch.long),
            persistent=True,
        )
        self.anchor_scorer = ProgramNativeAnchorScorer(
            self.owners,
            program_width=self.program_width,
            event_slots=self.event_slots,
            feature_width=self.anchor_width,
        )
        self.scale_head = torch.nn.Sequential(
            torch.nn.LayerNorm(self.program_width),
            torch.nn.Linear(self.program_width, self.program_width),
            torch.nn.GELU(),
            torch.nn.Linear(self.program_width, 1),
        )
        torch.nn.init.zeros_(self.scale_head[-1].weight)
        initial_scale = float(torch.atanh(torch.tensor(0.1)))
        torch.nn.init.constant_(self.scale_head[-1].bias, initial_scale)

    @staticmethod
    def _quadrature(positions: torch.Tensor) -> torch.Tensor:
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
        return weights.clamp_min(1e-8)

    def _validate_video(self, video: SharedCompilerVideo) -> None:
        owners = len(self.owners)
        frames = video.native.frame_count
        if (
            frames <= 0
            or video.canonical_assignment.shape != (frames, self.event_slots)
            or video.frame_positions.shape != (frames,)
            or video.local_scene.shape != (owners, self.program_width)
            or video.local_process.shape
            != (self.event_slots, owners, self.program_width)
            or video.local_presence.shape != (self.event_slots,)
            or video.local_tau.shape != (self.event_slots, 2)
            or video.local_sigma.shape
            != (self.event_slots, owners, self.program_width)
            or len(video.native.final_outputs) != owners
        ):
            raise NativeFactorError("compiler local video contract changed")

    def _video_measures(
        self,
        video: SharedCompilerVideo,
        event_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assignment = video.canonical_assignment.float().clamp_min(0)
        assignment = assignment / assignment.sum(-1, keepdim=True).clamp_min(1e-8)
        quadrature = self._quadrature(video.frame_positions)
        base = quadrature / quadrature.sum().clamp_min(1e-8)
        event = assignment.transpose(0, 1) * quadrature[None]
        event = event / event.sum(-1, keepdim=True).clamp_min(1e-8)
        frame = torch.einsum("jre,et->jrt", event_weights, event)
        frame = frame / frame.sum(-1, keepdim=True).clamp_min(1e-8)
        return base, event, frame

    @staticmethod
    def _candidate_mass(
        frame_mass: torch.Tensor,
        *,
        output: bool,
    ) -> torch.Tensor:
        shape = (frame_mass.shape[0], G1_PROBE_COUNT, ACTION_HORIZON)
        if output:
            shape = (*shape, len(OUTPUT_BANK_TYPES))
        leading = (frame_mass.shape[0], *((1,) * (len(shape) - 1)))
        return frame_mass.reshape(leading).expand(shape)

    @staticmethod
    def _effective_input_compatibility(
        compatibility: torch.Tensor,
        event_weights: torch.Tensor,
        event_ratio: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum(
            "re,et,rebtph->rbtph",
            event_weights,
            event_ratio,
            compatibility,
        )

    @staticmethod
    def _effective_output_compatibility(
        compatibility: torch.Tensor,
        event_weights: torch.Tensor,
        event_ratio: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum(
            "re,et,grebtphu->grbtphu",
            event_weights,
            event_ratio,
            compatibility,
        )

    @staticmethod
    def _solve_statistics(
        entries: Sequence[tuple[tuple[Any, ...], BankStatistics]],
        *,
        relative_floor: float,
        enabled: bool,
    ) -> dict[tuple[Any, ...], SpectralBankQuery | torch.Tensor]:
        if not enabled:
            return {key: statistics.anchor for key, statistics in entries}
        grouped: dict[
            tuple[int, tuple[int, ...]],
            list[tuple[tuple[Any, ...], BankStatistics]],
        ] = defaultdict(list)
        for key, statistics in entries:
            grouped[
                (statistics.mean.numel(), tuple(statistics.anchor.shape[:-1]))
            ].append((key, statistics))
        solved: dict[tuple[Any, ...], SpectralBankQuery | torch.Tensor] = {}
        for rows in grouped.values():
            queries = batched_spectral_bank_query(
                tuple(row[1] for row in rows),
                relative_eigenvalue_floor=relative_floor,
            )
            solved.update(
                (row[0], query) for row, query in zip(rows, queries, strict=True)
            )
        return solved

    @staticmethod
    def _query_tensor(value: SpectralBankQuery | torch.Tensor) -> torch.Tensor:
        return value.query if isinstance(value, SpectralBankQuery) else value

    def _new_statistics_stream(
        self, video: SharedCompilerVideo, state: AnchorProgramState
    ) -> _StatisticsStream:
        base_frame, event_frame, frame_measure = self._video_measures(
            video, state.event_weights
        )
        output_groups = tuple(
            native_output_group_count(owner) for owner in self.owners
        )
        return _StatisticsStream(
            base_frame=base_frame,
            event_frame=event_frame,
            frame_measure=frame_measure,
            input_anchor_queries=self.anchor_scorer.input_queries(state),
            input_accumulators=[
                StreamingBankStatistics(
                    width=owner.in_features,
                    query_shape=(G1_RESIDUAL_RANK, 2),
                    device=state.rank.device,
                    dtype=torch.float64,
                )
                for owner in self.owners
            ],
            output_accumulators=[
                tuple(
                    StreamingBankStatistics(
                        width=owner.out_features // groups,
                        query_shape=(G1_RESIDUAL_RANK, 2),
                        device=state.rank.device,
                        dtype=torch.float64,
                    )
                    for _ in range(groups)
                )
                for owner, groups in zip(self.owners, output_groups, strict=True)
            ],
            output_anchor_queries=tuple(
                self.anchor_scorer.output_queries(
                    state, target=target, groups=groups
                )
                for target, groups in enumerate(output_groups)
            ),
            gains=tuple(
                self.anchor_scorer.output_group_gains(
                    state, target=target, groups=groups
                )
                for target, groups in enumerate(output_groups)
            ),
            boundaries=[
                NativeOutputBankState(final=value.detach())
                for value in video.native.final_outputs
            ],
        )

    def _input_anchor_compatibility(
        self,
        value: torch.Tensor,
        metadata: torch.Tensor,
        native_query: torch.Tensor,
        metadata_query: torch.Tensor,
        magnitude_query: torch.Tensor,
        weights: torch.Tensor,
        ratio: torch.Tensor,
    ) -> torch.Tensor:
        event = self.anchor_scorer.input_compatibility(
            native_query,
            metadata_query,
            magnitude_query,
            value,
            metadata,
        )
        return self._effective_input_compatibility(event, weights, ratio)

    def _output_anchor_compatibility(
        self,
        value: torch.Tensor,
        metadata: torch.Tensor,
        native_query: torch.Tensor,
        metadata_query: torch.Tensor,
        magnitude_query: torch.Tensor,
        weights: torch.Tensor,
        ratio: torch.Tensor,
    ) -> torch.Tensor:
        event = self.anchor_scorer.output_compatibility(
            native_query,
            metadata_query,
            magnitude_query,
            value,
            metadata,
        )
        return self._effective_output_compatibility(event, weights, ratio)

    def _add_statistics_chunk(
        self,
        *,
        video: SharedCompilerVideo,
        state: AnchorProgramState,
        stream: _StatisticsStream,
        chunk: Any,
        start: int,
    ) -> int:
        stop = start + chunk.frame_count
        assignment = video.canonical_assignment[start:stop].float()
        frame_metadata = self.anchor_scorer.frame_metadata(
            assignment, video.frame_positions[start:stop]
        )
        input_metadata = self.anchor_scorer.candidate_metadata(
            frame_metadata, output=False
        )
        output_metadata = self.anchor_scorer.candidate_metadata(
            frame_metadata, output=True
        )
        event_ratio = stream.event_frame[:, start:stop] / stream.base_frame[
            start:stop
        ][None].clamp_min(1e-12)
        input_mass = self._candidate_mass(
            stream.base_frame[start:stop], output=False
        )
        output_mass = self._candidate_mass(
            stream.base_frame[start:stop], output=True
        )
        for target, (owner, x, y) in enumerate(
            zip(self.owners, chunk.inputs, chunk.outputs, strict=True)
        ):
            input_query = stream.input_anchor_queries[target]
            x_compatibility = checkpoint(
                self._input_anchor_compatibility,
                x,
                input_metadata,
                *input_query,
                state.event_weights[target],
                event_ratio,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            stream.input_accumulators[target].add(x, input_mass, x_compatibility)
            bank = stream.boundaries[target].build(y, start_frame=start)
            groups = native_output_group_count(owner)
            grouped = bank.reshape(
                *bank.shape[:-1], groups, owner.out_features // groups
            ).movedim(-2, 0)
            output_query = stream.output_anchor_queries[target]
            y_compatibility = checkpoint(
                self._output_anchor_compatibility,
                grouped,
                output_metadata[None],
                *output_query,
                state.event_weights[target],
                event_ratio,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            for group, accumulator in enumerate(stream.output_accumulators[target]):
                accumulator.add(grouped[group], output_mass, y_compatibility[group])
        return stop

    def _finalize_statistics_stream(
        self, stream: _StatisticsStream, state: AnchorProgramState
    ) -> _VideoBankPlan:
        entries: list[tuple[tuple[Any, ...], BankStatistics]] = [
            ((target, "input", 0), accumulator.finalize())
            for target, accumulator in enumerate(stream.input_accumulators)
        ]
        entries.extend(
            ((target, "output", group), accumulator.finalize())
            for target, accumulators in enumerate(stream.output_accumulators)
            for group, accumulator in enumerate(accumulators)
        )
        solved = self._solve_statistics(
            entries,
            relative_floor=self.relative_eigenvalue_floor,
            enabled=self.global_statistics,
        )
        input_queries = tuple(
            self._query_tensor(solved[(target, "input", 0)]).float()
            for target in range(len(self.owners))
        )
        output_queries = tuple(
            tuple(
                self._query_tensor(solved[(target, "output", group)]).float()
                * target_gains[group, :, None, None]
                for group in range(target_gains.shape[0])
            )
            for target, target_gains in enumerate(stream.gains)
        )
        diagnostics = tuple(
            value for value in solved.values() if isinstance(value, SpectralBankQuery)
        )
        solve_metrics = (
            state.rank.new_tensor(
                (
                    max(value.relative_residual_maximum for value in diagnostics),
                    min(value.retained_trace_fraction for value in diagnostics),
                    min(value.anchor_projection_minimum for value in diagnostics),
                    min(value.retained_rank for value in diagnostics),
                )
            )
            if diagnostics
            else state.rank.new_tensor((0.0, 1.0, 1.0, 0.0))
        )
        return _VideoBankPlan(
            input_queries=input_queries,
            output_queries=output_queries,
            frame_measure=stream.frame_measure,
            group_gains=torch.cat(stream.gains, dim=0),
            solve_metrics=solve_metrics,
        )

    def _statistics_pass(
        self,
        video: SharedCompilerVideo,
        state: AnchorProgramState,
    ) -> _VideoBankPlan:
        stream = self._new_statistics_stream(video, state)
        next_frame = 0
        for chunk in video.native.chunks():
            stop = next_frame + chunk.frame_count
            valid = all(
                (
                    chunk.start_frame == next_frame,
                    stop <= video.native.frame_count,
                    len(chunk.inputs) == len(self.owners),
                    len(chunk.outputs) == len(self.owners),
                )
            )
            if not valid:
                raise NativeFactorError("compiler B0 native stream changed")
            next_frame = self._add_statistics_chunk(
                video=video,
                state=state,
                stream=stream,
                chunk=chunk,
                start=next_frame,
            )
        complete = next_frame == video.native.frame_count and all(
            boundary.next_frame == next_frame for boundary in stream.boundaries
        )
        if not complete:
            raise NativeFactorError("compiler B0 native stream ended early")
        return self._finalize_statistics_stream(stream, state)

    @staticmethod
    def _measure_bias(
        frame_measure: torch.Tensor,
        base_frame: torch.Tensor,
        *,
        output: bool,
    ) -> torch.Tensor:
        ratio = frame_measure / base_frame[None].clamp_min(1e-12)
        extra = 3 if output else 2
        shape = (ratio.shape[0], ratio.shape[1], *((1,) * extra))
        target = (
            ratio.shape[0],
            ratio.shape[1],
            G1_PROBE_COUNT,
            ACTION_HORIZON,
            *(((len(OUTPUT_BANK_TYPES),) if output else ())),
        )
        return ratio.reshape(shape).expand(target).clamp_min(1e-12).log()

    def _replay_pass(
        self,
        video: SharedCompilerVideo,
        plan: _VideoBankPlan,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        base_frame = self._quadrature(video.frame_positions)
        base_frame = base_frame / base_frame.sum().clamp_min(1e-8)
        input_accumulators = tuple(
            StreamingSignedPool(
                query,
                dtype=torch.float32,
                explicit_branches=True,
            )
            for query in plan.input_queries
        )
        output_accumulators = tuple(
            tuple(
                StreamingSignedPool(
                    query,
                    dtype=torch.float32,
                    explicit_branches=True,
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
            stop = next_frame + chunk.frame_count
            if (
                chunk.start_frame != next_frame
                or stop > video.native.frame_count
                or len(chunk.inputs) != len(self.owners)
                or len(chunk.outputs) != len(self.owners)
            ):
                raise NativeFactorError("compiler B1 native stream changed")
            input_mass = self._candidate_mass(
                base_frame[next_frame:stop], output=False
            )
            output_mass = self._candidate_mass(
                base_frame[next_frame:stop], output=True
            )
            for target, (owner, x, y) in enumerate(
                zip(self.owners, chunk.inputs, chunk.outputs, strict=True)
            ):
                input_bias = self._measure_bias(
                    plan.frame_measure[target, :, next_frame:stop],
                    base_frame[next_frame:stop],
                    output=False,
                )
                input_accumulators[target].add(x, input_mass, input_bias)
                bank = boundaries[target].build(y, start_frame=next_frame)
                groups = native_output_group_count(owner)
                group_width = owner.out_features // groups
                grouped = bank.reshape(
                    *bank.shape[:-1], groups, group_width
                ).movedim(-2, 0)
                output_bias = self._measure_bias(
                    plan.frame_measure[target, :, next_frame:stop],
                    base_frame[next_frame:stop],
                    output=True,
                )
                for group, accumulator in enumerate(output_accumulators[target]):
                    accumulator.add(grouped[group], output_mass, output_bias)
            next_frame = stop
        if next_frame != video.native.frame_count or any(
            boundary.next_frame != next_frame for boundary in boundaries
        ):
            raise NativeFactorError("compiler B1 native stream ended early")
        return (
            tuple(accumulator.signed_mean() for accumulator in input_accumulators),
            tuple(
                torch.cat(
                    tuple(accumulator.signed_mean() for accumulator in groups),
                    dim=-1,
                )
                for groups in output_accumulators
            ),
        )

    def _pool_video(
        self,
        video: SharedCompilerVideo,
        state: AnchorProgramState,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        plan = self._statistics_pass(video, state)
        input_values, output_values = self._replay_pass(video, plan)
        return (
            input_values,
            output_values,
            plan.frame_measure,
            plan.group_gains,
            plan.solve_metrics,
        )

    def forward(
        self,
        program: NaturalProgram,
        videos: Sequence[SharedCompilerVideo],
        *,
        s_ref: torch.Tensor,
    ) -> SharedCompilerOutput:
        if len(videos) not in (1, 2, 4) or s_ref.shape != (len(self.owners),):
            raise NativeFactorError("compiler video set or scale authority changed")
        for video in videos:
            self._validate_video(video)
        state = self.anchor_scorer.program_state(program)

        pooled = tuple(self._pool_video(video, state) for video in videos)
        beta = state.rank.new_full((len(videos),), 1.0 / len(videos))
        scale_logits = self.scale_head(state.rank.detach()).squeeze(-1)
        scales = s_ref[:, None].to(scale_logits) * torch.tanh(scale_logits)
        a_directions = []
        b_directions = []
        b_values = []
        for target in range(len(self.owners)):
            raw_a = sum(
                beta[index] * values[0][target]
                for index, values in enumerate(pooled)
            )
            raw_b = sum(
                beta[index] * values[1][target]
                for index, values in enumerate(pooled)
            )
            a_direction = rms_normalize(raw_a)
            b_direction = rms_normalize(raw_b)
            a_directions.append(a_direction)
            b_directions.append(b_direction)
            b_values.append(b_direction * scales[target, :, None])
        return SharedCompilerOutput(
            residual=NativeFactorResidual(
                a=tuple(a_directions),
                b=tuple(b_values),
                scales=scales,
            ),
            input_directions=tuple(a_directions),
            output_directions=tuple(b_directions),
            video_weights=beta,
            frame_measures=tuple(values[2] for values in pooled),
            output_group_gains=tuple(values[3] for values in pooled),
            solve_metrics=torch.stack(tuple(values[4] for values in pooled)),
            global_statistics_enabled=self.global_statistics,
        )
