"""Current-bank primal-to-dual Pass B for the frozen-Program G3 compiler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ember.ecp.bank_conditioning.primal_dual_runtime import (
    PrimalDualVideoOperator,
)
from ember.ecp.bank_conditioning.program_primal import (
    PrimalProgramState,
    ProgramNativePrimalScorer,
)
from ember.ecp.contracts import TargetOwner
from ember.ecp.native_factors import (
    NativeFactorError,
    NativeFactorResidual,
    NativeVideoReadout,
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


class SharedNativeFactorCompiler(torch.nn.Module):
    """Generate one rank-four residual through the current native bank.

    The shared network predicts native *primal* directions from the complete
    Program.  For every video, B0 accumulates global unit-mass X/Y covariance,
    maps the primal into that bank's dual coordinate, and B1 applies exact
    antithetic signed pooling to the same real native candidates.  The final
    output remains one task residual and one complete rank16 adapter.
    """

    native_dual_matmul_precision = "ieee_fp32_no_tf32"

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        program_width: int = 128,
        event_slots: int = 8,
        relative_eigenvalue_floor: float = 1e-6,
        replay_score_rms: float = 0.02,
        covariance_frame_chunk: int = 4,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.program_width = int(program_width)
        self.event_slots = int(event_slots)
        self.relative_eigenvalue_floor = float(relative_eigenvalue_floor)
        self.replay_score_rms = float(replay_score_rms)
        self.covariance_frame_chunk = int(covariance_frame_chunk)
        if (
            not self.owners
            or self.program_width <= 0
            or self.event_slots <= 0
            or not 0.0 < self.relative_eigenvalue_floor < 1.0
            or self.replay_score_rms <= 0.0
            or self.covariance_frame_chunk <= 0
        ):
            raise NativeFactorError("invalid primal-dual compiler topology")
        self.primal_scorer = ProgramNativePrimalScorer(
            self.owners,
            program_width=self.program_width,
            event_slots=self.event_slots,
        )
        self.bank_operator = PrimalDualVideoOperator(
            self.owners,
            program_width=self.program_width,
            event_slots=self.event_slots,
            relative_eigenvalue_floor=self.relative_eigenvalue_floor,
            replay_score_rms=self.replay_score_rms,
            covariance_frame_chunk=self.covariance_frame_chunk,
        )
        self.scale_head = torch.nn.Sequential(
            torch.nn.LayerNorm(self.program_width),
            torch.nn.Linear(self.program_width, self.program_width),
            torch.nn.GELU(),
            torch.nn.Linear(self.program_width, 1),
        )
        torch.nn.init.zeros_(self.scale_head[-1].weight)
        torch.nn.init.constant_(
            self.scale_head[-1].bias,
            float(torch.atanh(torch.tensor(0.1))),
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
            # Retained modes can approach condition 1e6; TF32 destroys the
            # signed cancellation even when the eigensolve itself is valid.
            torch.backends.cuda.matmul.allow_tf32 = False
        for video in videos:
            self.bank_operator.validate_video(video)

        state: PrimalProgramState = self.primal_scorer.program_state(program)
        input_primals = self.primal_scorer.input_primals(state)
        output_primals = self.primal_scorer.output_primals(state)
        pooled = tuple(
            self.bank_operator(video, input_primals, output_primals)
            for video in videos
        )
        beta = state.rank.new_full((len(videos),), 1.0 / len(videos))
        scale_logits = self.scale_head(state.stable_rank.detach()).squeeze(-1)
        scales = s_ref[:, None].to(scale_logits) * torch.tanh(scale_logits)
        input_directions, output_directions, scaled_outputs = [], [], []
        for target in range(len(self.owners)):
            raw_input = torch.stack(
                tuple(row.input_values[target] for row in pooled)
            ).mean(0)
            raw_output = torch.stack(
                tuple(row.output_values[target] for row in pooled)
            ).mean(0)
            input_direction = rms_normalize(raw_input)
            output_direction = rms_normalize(raw_output)
            input_directions.append(input_direction)
            output_directions.append(output_direction)
            scaled_outputs.append(output_direction * scales[target, :, None])
        return SharedCompilerOutput(
            residual=NativeFactorResidual(
                a=tuple(input_directions),
                b=tuple(scaled_outputs),
                scales=scales,
            ),
            input_directions=tuple(input_directions),
            output_directions=tuple(output_directions),
            video_weights=beta,
            frame_measures=tuple(row.frame_measure for row in pooled),
            output_group_gains=tuple(row.group_gains for row in pooled),
            solve_metrics=torch.stack(tuple(row.solve_metrics for row in pooled)),
            conditioning_metrics=torch.stack(
                tuple(row.conditioning_metrics for row in pooled)
            ),
        )
