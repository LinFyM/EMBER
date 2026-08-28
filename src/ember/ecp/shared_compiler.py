"""Current-bank-conditioned Pass B for the frozen-Program G3 compiler."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from ember.ecp.bank_conditioning import (
    AnchorProgramState,
    FeatureWhiteningPlan,
    FunctionalBankStatistics,
    FunctionalPolarQueries,
    ProgramNativeAnchorScorer,
    StreamingFunctionalBankStatistics,
    StreamingSignedPool,
    batched_functional_polar_queries,
    build_feature_whitening_plan,
    bound_functional_queries,
)
from ember.ecp.bank_conditioning.anchor_solve import (
    ReplayBankPlan,
    build_replay_bank_plan,
    candidate_mass,
    input_event_keys,
    output_event_keys,
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
    conditioning_metrics: torch.Tensor


@dataclass
class _FunctionalStream:
    base_frame: torch.Tensor
    event_frame: torch.Tensor
    frame_measure: torch.Tensor
    input_accumulators: list[StreamingFunctionalBankStatistics]
    output_accumulators: list[tuple[StreamingFunctionalBankStatistics, ...]]
    gains: tuple[torch.Tensor, ...]
    boundaries: list[NativeOutputBankState]
    feature_plan: FeatureWhiteningPlan


@dataclass(frozen=True)
class _FunctionalVideoPlan:
    input_queries: tuple[torch.Tensor, ...]
    output_queries: tuple[torch.Tensor, ...]
    input_statistics: tuple[FunctionalBankStatistics, ...]
    output_statistics: tuple[tuple[FunctionalBankStatistics, ...], ...]
    base_frame: torch.Tensor
    event_frame: torch.Tensor
    frame_measure: torch.Tensor
    group_gains: torch.Tensor
    conditioning_metrics: torch.Tensor
    feature_plan: FeatureWhiteningPlan


@dataclass(frozen=True)
class _PolarRequest:
    identifier: int | tuple[int, int]
    mode: str
    raw: torch.Tensor
    weights: torch.Tensor
    statistics: FunctionalBankStatistics


class SharedNativeFactorCompiler(torch.nn.Module):
    """Generate one rank-four residual through current-bank functional polar.

    The scorer contains no task-, video-, member-, or frame-indexed parameters.
    B0a derives a detached gauge of the actual B0-solve/B1 operator. B0b uses
    the full Program in that gauge to form native anchors. B1 rereads the same
    bank and pools real X/Y values with explicit softmax branches.
    """

    native_dual_matmul_precision = "ieee_fp32"

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        program_width: int = 128,
        event_slots: int = 8,
        anchor_width: int = 128,
        relative_eigenvalue_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.program_width = int(program_width)
        self.event_slots = int(event_slots)
        self.anchor_width = int(anchor_width)
        self.relative_eigenvalue_floor = float(relative_eigenvalue_floor)
        self.anchor_score_bound = self.relative_eigenvalue_floor**0.5
        self.replay_score_rms = 0.02
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

    def _new_functional_stream(
        self,
        video: SharedCompilerVideo,
        state: AnchorProgramState,
        feature_plan: FeatureWhiteningPlan,
    ) -> _FunctionalStream:
        base, event, frame = self._video_measures(video, state.event_weights)
        groups = tuple(native_output_group_count(owner) for owner in self.owners)
        new = lambda width: StreamingFunctionalBankStatistics(
            native_width=width,
            key_width=self.anchor_width,
            events=self.event_slots,
            ranks=G1_RESIDUAL_RANK,
            device=state.rank.device,
            dtype=torch.float32,
        )
        return _FunctionalStream(
            base_frame=base,
            event_frame=event,
            frame_measure=frame,
            input_accumulators=[new(owner.in_features) for owner in self.owners],
            output_accumulators=[
                tuple(new(owner.out_features // count) for _ in range(count))
                for owner, count in zip(self.owners, groups, strict=True)
            ],
            gains=tuple(
                self.anchor_scorer.output_group_gains(
                    state, target=target, groups=count
                )
                for target, count in enumerate(groups)
            ),
            boundaries=[
                NativeOutputBankState(final=value.detach())
                for value in video.native.final_outputs
            ],
            feature_plan=feature_plan,
        )

    def _add_functional_chunk(
        self,
        video: SharedCompilerVideo,
        stream: _FunctionalStream,
        chunk: Any,
        start: int,
    ) -> int:
        stop = start + chunk.frame_count
        assignment = video.canonical_assignment[start:stop].float()
        frame_metadata = self.anchor_scorer.frame_metadata(
            assignment, video.frame_positions[start:stop]
        )
        x_metadata = self.anchor_scorer.candidate_metadata(
            frame_metadata, output=False
        )
        y_metadata = self.anchor_scorer.candidate_metadata(
            frame_metadata, output=True
        )
        x_base = candidate_mass(stream.base_frame[start:stop], output=False)
        y_base = candidate_mass(stream.base_frame[start:stop], output=True)
        x_event = candidate_mass(
            stream.event_frame[:, start:stop], output=False
        )
        y_event = candidate_mass(
            stream.event_frame[:, start:stop], output=True
        )
        with torch.no_grad():
            for target, (owner, x, y) in enumerate(
                zip(self.owners, chunk.inputs, chunk.outputs, strict=True)
            ):
                x_replay = candidate_mass(
                    stream.frame_measure[target, :, start:stop], output=False
                )
                stream.input_accumulators[target].add(
                    x,
                    x_base,
                    x_replay,
                    x_event,
                    input_event_keys(
                        self.anchor_scorer,
                        x,
                        x_metadata,
                        target=target,
                        whitener=stream.feature_plan.input_whiteners[target],
                    ),
                )
                bank = stream.boundaries[target].build(y, start_frame=start)
                count = native_output_group_count(owner)
                grouped = bank.reshape(
                    *bank.shape[:-1], count, owner.out_features // count
                ).movedim(-2, 0)
                y_replay = candidate_mass(
                    stream.frame_measure[target, :, start:stop], output=True
                )
                keys = output_event_keys(
                    self.anchor_scorer,
                    grouped,
                    y_metadata[None],
                    target=target,
                    whiteners=stream.feature_plan.output_whiteners[target],
                ).movedim(0, 1)
                for group, accumulator in enumerate(stream.output_accumulators[target]):
                    accumulator.add(
                        grouped[group], y_base, y_replay, y_event, keys[group]
                    )
        return stop

    def _functional_statistics(
        self, video: SharedCompilerVideo, state: AnchorProgramState
    ) -> tuple[
        _FunctionalStream,
        tuple[FunctionalBankStatistics, ...],
        tuple[tuple[FunctionalBankStatistics, ...], ...],
    ]:
        _, event_frame, _ = self._video_measures(video, state.event_weights)
        feature_plan = build_feature_whitening_plan(
            video=video,
            event_frame=event_frame,
            scorer=self.anchor_scorer,
            owners=self.owners,
            events=self.event_slots,
            width=self.anchor_width,
            relative_eigenvalue_floor=self.relative_eigenvalue_floor,
        )
        stream = self._new_functional_stream(video, state, feature_plan)
        next_frame = 0
        for chunk in video.native.chunks():
            stop = next_frame + chunk.frame_count
            if (
                chunk.start_frame != next_frame
                or stop > video.native.frame_count
                or len(chunk.inputs) != len(self.owners)
                or len(chunk.outputs) != len(self.owners)
            ):
                raise NativeFactorError("compiler B0a native stream changed")
            next_frame = self._add_functional_chunk(
                video, stream, chunk, next_frame
            )
        if next_frame != video.native.frame_count or any(
            boundary.next_frame != next_frame for boundary in stream.boundaries
        ):
            raise NativeFactorError("compiler B0a native stream ended early")
        inputs = tuple(value.finalize() for value in stream.input_accumulators)
        outputs = tuple(
            tuple(value.finalize() for value in groups)
            for groups in stream.output_accumulators
        )
        return stream, inputs, outputs

    def _functional_polar_requests(
        self,
        state: AnchorProgramState,
        stream: _FunctionalStream,
        inputs: tuple[FunctionalBankStatistics, ...],
        outputs: tuple[tuple[FunctionalBankStatistics, ...], ...],
    ) -> tuple[list[_PolarRequest], list[_PolarRequest]]:
        raw_input = self.anchor_scorer.input_queries(state)
        input_requests: list[_PolarRequest] = []
        output_requests: list[_PolarRequest] = []
        for target, statistics in enumerate(inputs):
            output_polar_mode = (
                "global"
                if self.owners[target].family.value == "q"
                else "per_event"
            )
            raw = self.anchor_scorer.input_projected_queries(
                raw_input[target], target=target
            )
            input_requests.append(
                _PolarRequest(
                    identifier=target,
                    mode="per_event",
                    raw=raw,
                    weights=state.event_weights[target],
                    statistics=statistics,
                )
            )
            group_raw = self.anchor_scorer.output_queries(
                state, target=target, groups=len(outputs[target])
            )
            group_raw = self.anchor_scorer.output_projected_queries(
                group_raw, target=target
            )
            group_raw = group_raw * stream.gains[target][
                :, :, None, None, None
            ]
            output_requests.extend(
                _PolarRequest(
                    identifier=(target, group),
                    mode=output_polar_mode,
                    raw=group_raw[group],
                    weights=state.event_weights[target],
                    statistics=statistics,
                )
                for group, statistics in enumerate(outputs[target])
            )
        return input_requests, output_requests

    def _polarize_requests(
        self, requests: Sequence[_PolarRequest]
    ) -> dict[int | tuple[int, int], FunctionalPolarQueries]:
        buckets: dict[tuple[object, ...], list[_PolarRequest]] = defaultdict(list)
        for request in requests:
            statistics = request.statistics
            buckets[
                (
                    request.mode,
                    statistics.mean.numel(),
                    statistics.key_images.shape[-1],
                    statistics.covariance.dtype,
                )
            ].append(request)
        resolved = {}
        for key, rows in buckets.items():
            values = batched_functional_polar_queries(
                tuple(row.raw for row in rows),
                tuple(row.weights for row in rows),
                tuple(row.statistics for row in rows),
                covariance_floor=self.relative_eigenvalue_floor,
                image_floor=self.relative_eigenvalue_floor,
                mode=str(key[0]),
            )
            resolved.update(
                (row.identifier, value)
                for row, value in zip(rows, values, strict=True)
            )
        return resolved

    def _functional_video_plan(
        self,
        state: AnchorProgramState,
        stream: _FunctionalStream,
        inputs: tuple[FunctionalBankStatistics, ...],
        outputs: tuple[tuple[FunctionalBankStatistics, ...], ...],
        input_polar: dict[int | tuple[int, int], FunctionalPolarQueries],
        output_polar: dict[int | tuple[int, int], FunctionalPolarQueries],
    ) -> _FunctionalVideoPlan:
        input_queries = []
        output_queries = []
        metrics = []
        for target in range(len(self.owners)):
            input_value = input_polar[target]
            bounded, _ = bound_functional_queries(
                (input_value.queries,),
                score_bound=self.anchor_score_bound,
            )
            input_queries.append(bounded[0])
            metrics.append(input_value.metrics)
            group_polar = tuple(
                output_polar[(target, group)]
                for group in range(len(outputs[target]))
            )
            bounded, _ = bound_functional_queries(
                tuple(value.queries for value in group_polar),
                score_bound=self.anchor_score_bound,
            )
            output_queries.append(torch.stack(bounded))
            metrics.extend(value.metrics for value in group_polar)
        stacked = torch.stack(metrics)
        conditioning = torch.stack(
            (
                stacked[:, 0].min(),
                stacked[:, 1].min(),
                stacked[:, 2].max(),
                stacked[:, 3].min(),
                stream.feature_plan.metrics[0],
                stream.feature_plan.metrics[1],
            )
        ).to(state.rank)
        return _FunctionalVideoPlan(
            input_queries=tuple(input_queries),
            output_queries=tuple(output_queries),
            input_statistics=inputs,
            output_statistics=outputs,
            base_frame=stream.base_frame,
            event_frame=stream.event_frame,
            frame_measure=stream.frame_measure,
            group_gains=torch.cat(stream.gains, dim=0),
            conditioning_metrics=conditioning,
            feature_plan=stream.feature_plan,
        )

    def _functional_pass(
        self, video: SharedCompilerVideo, state: AnchorProgramState
    ) -> _FunctionalVideoPlan:
        stream, inputs, outputs = self._functional_statistics(video, state)
        input_requests, output_requests = self._functional_polar_requests(
            state, stream, inputs, outputs
        )
        return self._functional_video_plan(
            state,
            stream,
            inputs,
            outputs,
            self._polarize_requests(input_requests),
            self._polarize_requests(output_requests),
        )

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
        plan: ReplayBankPlan,
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
            input_mass = candidate_mass(
                base_frame[next_frame:stop], output=False
            )
            output_mass = candidate_mass(
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
        torch.Tensor,
    ]:
        functional = self._functional_pass(video, state)
        plan = build_replay_bank_plan(
            video=video,
            functional=functional,
            state=state,
            owners=self.owners,
            scorer=self.anchor_scorer,
            relative_floor=self.relative_eigenvalue_floor,
            replay_score_rms=self.replay_score_rms,
        )
        input_values, output_values = self._replay_pass(video, plan)
        return (
            input_values,
            output_values,
            plan.frame_measure,
            plan.group_gains,
            plan.solve_metrics,
            plan.conditioning_metrics,
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
        if s_ref.device.type == "cuda":
            # The retained native covariance can have condition near 1e6.
            # TF32 score products then destroy the signed dual cancellation.
            # Keep this process-wide setting through autograd backward; the
            # frozen policy runs under BF16 and does not need TF32 FP32 GEMMs.
            torch.backends.cuda.matmul.allow_tf32 = False
        for video in videos:
            self._validate_video(video)
        state = self.anchor_scorer.program_state(program)

        pooled = tuple(self._pool_video(video, state) for video in videos)
        beta = state.rank.new_full((len(videos),), 1.0 / len(videos))
        scale_logits = self.scale_head(state.stable_rank.detach()).squeeze(-1)
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
            conditioning_metrics=torch.stack(
                tuple(values[5] for values in pooled)
            ),
        )
