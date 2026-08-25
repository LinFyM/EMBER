"""Content-addressed Pass B for the frozen-Program G3 compiler."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint

from ember.ecp.contracts import ACTION_HORIZON, TargetOwner
from ember.ecp.native_factors import (
    G1_PROBE_COUNT,
    G1_RESIDUAL_RANK,
    OUTPUT_BANK_TYPES,
    NativeFactorError,
    NativeFactorResidual,
    NativeOutputBankState,
    NativeVideoReadout,
    OnlineSoftmaxAccumulator,
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


class NativeContentKey(torch.nn.Module):
    """Map a real native vector and its magnitude to a normalized content key."""

    def __init__(self, width: int, key_width: int) -> None:
        super().__init__()
        self.content = torch.nn.Linear(width, key_width, bias=False)
        self.magnitude = torch.nn.Linear(1, key_width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        magnitude = value.float().square().mean(-1, keepdim=True).sqrt()
        direction = value.float() / magnitude.clamp_min(1e-6)
        return self.content(direction) + self.magnitude(
            magnitude.clamp_min(1e-6).log()
        )


class SharedNativeFactorCompiler(torch.nn.Module):
    """Generate one rank-four residual through shared query-key signed pooling.

    There are no task-, video-, or frame-indexed selection parameters.  Every
    signed weight is the difference of two content logits normalized over the
    real native candidates of one video.  Videos are pooled independently and
    only then combined as a bounded, permutation-invariant probability measure.
    """

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        program_width: int = 128,
        event_slots: int = 8,
        key_width: int = 64,
        maximum_video_correction: float = 0.5,
        video_score_bound: float = 2.0,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.program_width = int(program_width)
        self.event_slots = int(event_slots)
        self.key_width = int(key_width)
        if (
            not self.owners
            or self.program_width <= 0
            or self.event_slots <= 0
            or self.key_width <= 0
            or not 0.0 <= maximum_video_correction < 1.0
            or video_score_bound <= 0.0
        ):
            raise NativeFactorError("invalid shared compiler topology")
        self.maximum_video_correction = float(maximum_video_correction)
        self.video_score_bound = float(video_score_bound)

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

        input_widths = sorted({owner.in_features for owner in self.owners})
        output_widths = sorted(
            {
                owner.out_features // native_output_group_count(owner)
                for owner in self.owners
            }
        )
        self.input_keys = torch.nn.ModuleDict(
            {
                str(width): NativeContentKey(width, self.key_width)
                for width in input_widths
            }
        )
        self.output_keys = torch.nn.ModuleDict(
            {
                str(width): NativeContentKey(width, self.key_width)
                for width in output_widths
            }
        )

        width = self.program_width
        self.owner_embedding = torch.nn.Parameter(torch.empty(len(self.owners), width))
        self.rank_embedding = torch.nn.Parameter(
            torch.empty(G1_RESIDUAL_RANK, width)
        )
        self.group_embedding = torch.nn.Parameter(
            torch.empty(max(output_counts), width)
        )
        self.context = torch.nn.Sequential(
            torch.nn.LayerNorm(4 * width),
            torch.nn.Linear(4 * width, 2 * width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * width, width),
            torch.nn.LayerNorm(width),
        )
        self.rank_context = torch.nn.Sequential(
            torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.LayerNorm(width)
        )
        self.input_query = torch.nn.Linear(width, 2 * self.key_width)
        self.output_query = torch.nn.Linear(width, 2 * self.key_width)
        self.event_query = torch.nn.Linear(width, self.key_width, bias=False)
        self.event_key = torch.nn.Linear(width, self.key_width, bias=False)
        self.uncertainty_penalty = torch.nn.Parameter(torch.tensor(0.0))
        self.scale_head = torch.nn.Linear(width, 1)

        self.event_metadata = torch.nn.Parameter(
            torch.empty(self.event_slots, self.key_width)
        )
        self.probe_metadata = torch.nn.Parameter(
            torch.empty(G1_PROBE_COUNT, self.key_width)
        )
        self.horizon_metadata = torch.nn.Parameter(
            torch.empty(ACTION_HORIZON, self.key_width)
        )
        self.type_metadata = torch.nn.Parameter(
            torch.empty(len(OUTPUT_BANK_TYPES), self.key_width)
        )
        self.time_metadata = torch.nn.Linear(2, self.key_width, bias=False)
        self.input_logit_scale = torch.nn.Parameter(torch.tensor(math.log(4.0)))
        self.output_logit_scale = torch.nn.Parameter(torch.tensor(math.log(4.0)))

        self.video_reliability = torch.nn.Sequential(
            torch.nn.LayerNorm(5 * width),
            torch.nn.Linear(5 * width, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, 1),
        )
        torch.nn.init.normal_(self.owner_embedding, std=width**-0.5)
        torch.nn.init.normal_(self.rank_embedding, std=width**-0.5)
        torch.nn.init.normal_(self.group_embedding, std=width**-0.5)
        for value in (
            self.event_metadata,
            self.probe_metadata,
            self.horizon_metadata,
            self.type_metadata,
        ):
            torch.nn.init.normal_(value, std=self.key_width**-0.5)
        torch.nn.init.zeros_(self.video_reliability[-1].weight)
        torch.nn.init.zeros_(self.video_reliability[-1].bias)
        torch.nn.init.zeros_(self.scale_head.weight)
        torch.nn.init.constant_(self.scale_head.bias, math.atanh(0.1))

    def _validate_program(self, program: NaturalProgram) -> None:
        expected_owner = (len(self.owners), self.program_width)
        if (
            program.p_lang.shape != expected_owner
            or program.p_scene.shape != expected_owner
            or program.p_process.shape
            != (self.event_slots, len(self.owners), self.program_width)
            or program.rho.shape != (self.event_slots,)
            or program.tau.shape != (self.event_slots, 2)
            or program.sigma.shape
            != (self.event_slots, len(self.owners), self.program_width)
        ):
            raise NativeFactorError("shared compiler Program schema changed")

    def _owner_rank_context(
        self, program: NaturalProgram
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mass = program.rho.float().clamp_min(1e-6)
        mass = mass / mass.sum()
        process = torch.einsum("e,ejw->jw", mass, program.p_process.float())
        sigma = torch.einsum("e,ejw->jw", mass, program.sigma.float())
        owner = self.context(
            torch.cat(
                (
                    program.p_lang.float(),
                    program.p_scene.float(),
                    process,
                    sigma,
                ),
                dim=-1,
            )
        )
        rank = self.rank_context(
            owner[:, None] + self.owner_embedding[:, None] + self.rank_embedding[None]
        )
        event_query = functional.normalize(self.event_query(rank), dim=-1)
        event_key = functional.normalize(
            self.event_key(program.p_process.float()).permute(1, 0, 2), dim=-1
        )
        event_logits = torch.einsum("jrd,jed->jre", event_query, event_key)
        event_logits = event_logits * math.sqrt(self.key_width)
        uncertainty = program.sigma.float().square().mean(-1).sqrt().transpose(0, 1)
        event_logits = event_logits + mass.log()[None, None]
        event_logits = event_logits - functional.softplus(
            self.uncertainty_penalty
        ) * uncertainty[:, None]
        return rank, event_logits.softmax(-1), owner

    @staticmethod
    def _quadrature(positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim != 1 or positions.numel() <= 0:
            raise NativeFactorError("shared compiler frame positions changed")
        if positions.numel() == 1:
            return torch.ones_like(positions, dtype=torch.float32)
        points = positions.float()
        if torch.any(points[1:] < points[:-1]):
            raise NativeFactorError("shared compiler video is not internally ordered")
        gaps = points[1:] - points[:-1]
        weights = torch.empty_like(points)
        weights[0] = gaps[0] * 0.5
        weights[-1] = gaps[-1] * 0.5
        if points.numel() > 2:
            weights[1:-1] = (gaps[:-1] + gaps[1:]) * 0.5
        if float(weights.sum()) <= 1e-8:
            weights.fill_(1.0)
        return weights.clamp_min(1e-8)

    def _frame_measure(
        self, video: SharedCompilerVideo, event_weights: torch.Tensor
    ) -> torch.Tensor:
        frames = video.native.frame_count
        if video.canonical_assignment.shape != (frames, self.event_slots):
            raise NativeFactorError("shared compiler canonical assignment changed")
        assignment = video.canonical_assignment.float().clamp_min(0)
        assignment = assignment / assignment.sum(-1, keepdim=True).clamp_min(1e-6)
        measure = torch.einsum("jre,te->jrt", event_weights, assignment)
        measure = measure * self._quadrature(video.frame_positions)[None, None]
        return measure / measure.sum(-1, keepdim=True).clamp_min(1e-8)

    def _frame_metadata(
        self, video: SharedCompilerVideo, start: int, stop: int
    ) -> torch.Tensor:
        assignment = video.canonical_assignment[start:stop].to(self.event_metadata)
        event = assignment @ self.event_metadata
        positions = video.frame_positions[start:stop].to(self.event_metadata)
        time = self.time_metadata(torch.stack((positions, 1.0 - positions), dim=-1))
        return event + time

    def _candidate_metadata(
        self,
        frame: torch.Tensor,
        *,
        output: bool,
    ) -> torch.Tensor:
        value = (
            frame[:, None, None]
            + self.probe_metadata[None, :, None]
            + self.horizon_metadata[None, None]
        )
        if output:
            value = value[:, :, :, None] + self.type_metadata[None, None, None]
        return value

    def _pool_video(
        self,
        video: SharedCompilerVideo,
        *,
        rank_context: torch.Tensor,
        event_weights: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor]:
        owners = len(self.owners)
        frames = video.native.frame_count
        if (
            video.frame_positions.shape != (frames,)
            or video.local_scene.shape != (owners, self.program_width)
            or video.local_process.shape
            != (self.event_slots, owners, self.program_width)
            or video.local_presence.shape != (self.event_slots,)
            or video.local_tau.shape != (self.event_slots, 2)
            or video.local_sigma.shape
            != (self.event_slots, owners, self.program_width)
        ):
            raise NativeFactorError("shared compiler local Program changed")
        frame_measure = self._frame_measure(video, event_weights)
        input_query = self.input_query(rank_context).reshape(
            owners, G1_RESIDUAL_RANK, 2, self.key_width
        )
        input_query = functional.normalize(input_query, dim=-1)
        input_scale = self.input_logit_scale.exp().clamp(max=20.0)

        input_accumulators = [
            OnlineSoftmaxAccumulator(
                ranks=G1_RESIDUAL_RANK,
                width=owner.in_features,
                device=rank_context.device,
            )
            for owner in self.owners
        ]
        output_accumulators = [
            tuple(
                OnlineSoftmaxAccumulator(
                    ranks=G1_RESIDUAL_RANK,
                    width=owner.out_features // native_output_group_count(owner),
                    device=rank_context.device,
                )
                for _ in range(native_output_group_count(owner))
            )
            for owner in self.owners
        ]
        boundaries = [
            NativeOutputBankState(final=value.detach())
            for value in video.native.final_outputs
        ]
        next_frame = 0
        for chunk in video.native.chunks():
            count = chunk.frame_count
            stop = next_frame + count
            if (
                chunk.start_frame != next_frame
                or stop > frames
                or len(chunk.inputs) != owners
                or len(chunk.outputs) != owners
            ):
                raise NativeFactorError("shared compiler native stream changed")
            frame_key = self._frame_metadata(video, next_frame, stop)
            input_metadata = self._candidate_metadata(frame_key, output=False)
            output_metadata = self._candidate_metadata(frame_key, output=True)
            log_measure = frame_measure[:, :, next_frame:stop].clamp_min(1e-12).log()
            for target, (owner, x, y) in enumerate(
                zip(self.owners, chunk.inputs, chunk.outputs, strict=True)
            ):
                x_key = self.input_keys[str(owner.in_features)](x) + input_metadata
                x_key = functional.normalize(x_key.float(), dim=-1)
                x_logits = torch.einsum(
                    "rbd,tphd->rbtph", input_query[target], x_key
                ) * input_scale
                x_logits = x_logits + log_measure[target, :, None, :, None, None]
                input_accumulators[target].add(x_logits, x)

                bank = boundaries[target].build(y, start_frame=next_frame)
                groups = native_output_group_count(owner)
                group_width = owner.out_features // groups
                grouped = bank.reshape(
                    *bank.shape[:-1], groups, group_width
                ).movedim(-2, 0)
                group_context = self.rank_context(
                    rank_context[target][None]
                    + self.group_embedding[:groups, None]
                )
                output_query = self.output_query(group_context).reshape(
                    groups, G1_RESIDUAL_RANK, 2, self.key_width
                )
                output_query = functional.normalize(output_query, dim=-1)
                y_key = self.output_keys[str(group_width)](grouped)
                y_key = functional.normalize(
                    (y_key + output_metadata[None]).float(), dim=-1
                )
                y_logits = torch.einsum(
                    "grbd,gtphud->grbtphu", output_query, y_key
                ) * self.output_logit_scale.exp().clamp(max=20.0)
                y_logits = y_logits + log_measure[
                    target, None, :, None, :, None, None, None
                ]
                for group, accumulator in enumerate(output_accumulators[target]):
                    accumulator.add(y_logits[group], grouped[group])
            next_frame = stop
        if next_frame != frames or any(
            boundary.next_frame != frames for boundary in boundaries
        ):
            raise NativeFactorError("shared compiler native stream ended early")
        return (
            tuple(value.signed_mean() for value in input_accumulators),
            tuple(
                torch.cat(tuple(value.signed_mean() for value in groups), dim=-1)
                for groups in output_accumulators
            ),
            frame_measure,
        )

    def _video_weights(
        self,
        program: NaturalProgram,
        videos: Sequence[SharedCompilerVideo],
        owner_context: torch.Tensor,
    ) -> torch.Tensor:
        count = len(videos)
        if count == 1:
            return owner_context.new_ones(1)
        # Scale/video heads own only their explicit heads.  Their loss path must
        # not re-enter the shared selection context and consume its clip budget.
        aggregate = owner_context.detach().mean(0)
        rows = []
        for video in videos:
            local_process = video.local_process.float().mean((0, 1))
            local_scene = video.local_scene.float().mean(0)
            local_sigma = video.local_sigma.float().mean((0, 1))
            local_tau = video.local_tau.float().mean(0)
            tau_features = functional.pad(
                local_tau, (0, self.program_width - local_tau.numel())
            )
            rows.append(
                torch.cat(
                    (
                        local_process,
                        local_scene,
                        local_sigma,
                        (local_process - aggregate).abs(),
                        tau_features,
                    )
                )
            )
        scores = self.video_reliability(torch.stack(rows)).squeeze(-1)
        scores = self.video_score_bound * torch.tanh(scores)
        learned = scores.softmax(0)
        uniform = torch.full_like(learned, 1.0 / count)
        return (
            (1.0 - self.maximum_video_correction) * uniform
            + self.maximum_video_correction * learned
        )

    def forward(
        self,
        program: NaturalProgram,
        videos: Sequence[SharedCompilerVideo],
        *,
        s_ref: torch.Tensor,
    ) -> SharedCompilerOutput:
        self._validate_program(program)
        if len(videos) not in (1, 2, 4) or s_ref.shape != (len(self.owners),):
            raise NativeFactorError(
                "shared compiler video set or scale authority changed"
            )
        rank_context, event_weights, owner_context = self._owner_rank_context(program)

        def pool(video: SharedCompilerVideo):
            if self.training and torch.is_grad_enabled():
                return checkpoint(
                    lambda rank, events: self._pool_video(
                        video, rank_context=rank, event_weights=events
                    ),
                    rank_context,
                    event_weights,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            return self._pool_video(
                video, rank_context=rank_context, event_weights=event_weights
            )

        pooled = tuple(pool(video) for video in videos)
        beta = self._video_weights(program, videos, owner_context)
        scale_logits = self.scale_head(rank_context.detach()).squeeze(-1)
        scales = s_ref[:, None].to(scale_logits) * torch.tanh(scale_logits)
        a_directions = []
        b_directions = []
        b_values = []
        for target in range(len(self.owners)):
            a = sum(
                beta[index] * value[0][target]
                for index, value in enumerate(pooled)
            )
            b = sum(
                beta[index] * value[1][target]
                for index, value in enumerate(pooled)
            )
            a_direction = rms_normalize(a)
            b_direction = rms_normalize(b)
            a_directions.append(a_direction)
            b_directions.append(b_direction)
            b_values.append(b_direction * scales[target, :, None])
        return SharedCompilerOutput(
            residual=NativeFactorResidual(
                a=tuple(a_directions), b=tuple(b_values), scales=scales
            ),
            input_directions=tuple(a_directions),
            output_directions=tuple(b_directions),
            video_weights=beta,
            frame_measures=tuple(value[2] for value in pooled),
        )
