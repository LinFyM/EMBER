"""Current-bank primal-to-dual Pass B for the frozen-Program G3 compiler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ember.ecp.bank_conditioning.program_bank_interaction import (
    ProgramBankContext,
    ProgramBankInteractionScorer,
    ProgramBankInteractionState,
)
from ember.ecp.bank_conditioning.primal_dual_runtime import (
    CompactPrimalDualVideo,
    MaterializedPrimalDualVideo,
    PrimalDualVideoOperator,
    PrimalDualVideoResult,
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
        inverse_covariance_power: float = 1.0,
        interaction_semantic_width: int = 32,
        interaction_hidden_width: int = 64,
        interaction_correction_bound: float = 0.1,
        scale_prior_ratio: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.program_width = int(program_width)
        self.event_slots = int(event_slots)
        self.relative_eigenvalue_floor = float(relative_eigenvalue_floor)
        self.replay_score_rms = float(replay_score_rms)
        self.covariance_frame_chunk = int(covariance_frame_chunk)
        self.inverse_covariance_power = float(inverse_covariance_power)
        if scale_prior_ratio is None:
            scale_prior_ratio = torch.full(
                (len(self.owners), 4), 0.1, dtype=torch.float32
            )
        scale_prior_ratio = scale_prior_ratio.detach().float()
        if (
            not self.owners
            or self.program_width <= 0
            or self.event_slots <= 0
            or not 0.0 < self.relative_eigenvalue_floor < 1.0
            or self.replay_score_rms <= 0.0
            or self.covariance_frame_chunk <= 0
            or self.inverse_covariance_power not in (0.5, 0.75, 1.0)
            or scale_prior_ratio.shape != (len(self.owners), 4)
            or not bool(torch.isfinite(scale_prior_ratio).all())
            or not bool(torch.all((scale_prior_ratio > 0) & (scale_prior_ratio < 1)))
        ):
            raise NativeFactorError("invalid primal-dual compiler topology")
        self.register_buffer(
            "scale_prior_ratio", scale_prior_ratio.clone(), persistent=True
        )
        self.primal_scorer = ProgramNativePrimalScorer(
            self.owners,
            program_width=self.program_width,
            event_slots=self.event_slots,
        )
        self.interaction_scorer = ProgramBankInteractionScorer(
            self.owners,
            program_width=self.program_width,
            event_slots=self.event_slots,
            semantic_width=interaction_semantic_width,
            hidden_width=interaction_hidden_width,
            correction_bound=interaction_correction_bound,
            replay_score_rms=self.replay_score_rms,
        )
        self.bank_operator = PrimalDualVideoOperator(
            self.owners,
            program_width=self.program_width,
            event_slots=self.event_slots,
            relative_eigenvalue_floor=self.relative_eigenvalue_floor,
            replay_score_rms=self.replay_score_rms,
            covariance_frame_chunk=self.covariance_frame_chunk,
            inverse_covariance_power=self.inverse_covariance_power,
        )
        self.scale_head = torch.nn.Sequential(
            torch.nn.LayerNorm(self.program_width),
            torch.nn.Linear(self.program_width, self.program_width),
            torch.nn.GELU(),
            torch.nn.Linear(self.program_width, 1),
        )
        torch.nn.init.zeros_(self.scale_head[-1].weight)
        torch.nn.init.zeros_(self.scale_head[-1].bias)

    @staticmethod
    def _bank_context(video: SharedCompilerVideo) -> ProgramBankContext:
        return ProgramBankContext(
            canonical_assignment=video.canonical_assignment,
            frame_positions=video.frame_positions,
            local_scene=video.local_scene,
            local_process=video.local_process,
            local_presence=video.local_presence,
            local_tau=video.local_tau,
            local_sigma=video.local_sigma,
        )

    def _interaction_states(
        self,
        state: PrimalProgramState,
        contexts: Sequence[ProgramBankContext],
    ) -> tuple[ProgramBankInteractionState, ...]:
        input_queries = self.primal_scorer.input_event_queries(state)
        output_queries = self.primal_scorer.output_event_queries(state)
        return tuple(
            ProgramBankInteractionState(
                context=context,
                rank_event=state.rank_event,
                event_weights=state.event_weights,
                input_event_queries=input_queries,
                output_event_queries=output_queries,
            )
            for context in contexts
        )

    def forward(
        self,
        program: NaturalProgram,
        videos: Sequence[SharedCompilerVideo],
        *,
        s_ref: torch.Tensor,
        interaction_off: bool = False,
    ) -> SharedCompilerOutput:
        if len(videos) not in (1, 2, 4) or s_ref.shape != (len(self.owners),):
            raise NativeFactorError("compiler video set or scale authority changed")
        for video in videos:
            self.bank_operator.validate_video(video)
        with self.bank_operator.ieee_matmul(s_ref.device):
            state: PrimalProgramState = self.primal_scorer.program_state(program)
            input_primals = self.primal_scorer.input_primals(state)
            output_primals = self.primal_scorer.output_primals(state)
            interaction_states = self._interaction_states(
                state, tuple(self._bank_context(video) for video in videos)
            )
            pooled = tuple(
                self.bank_operator(
                    video,
                    input_primals,
                    output_primals,
                    interaction_scorer=self.interaction_scorer,
                    interaction_state=interaction_state,
                    interaction_off=interaction_off,
                )
                for video, interaction_state in zip(
                    videos, interaction_states, strict=True
                )
            )
            return self._output(state, pooled, s_ref=s_ref)

    def forward_materialized(
        self,
        program: NaturalProgram,
        videos: Sequence[MaterializedPrimalDualVideo],
        *,
        s_ref: torch.Tensor,
    ) -> SharedCompilerOutput:
        """Training-only replay of a frozen bank prepared by canonical B0."""

        if len(videos) not in (1, 2, 4) or s_ref.shape != (len(self.owners),):
            raise NativeFactorError("materialized compiler video set changed")
        with self.bank_operator.ieee_matmul(s_ref.device):
            state: PrimalProgramState = self.primal_scorer.program_state(program)
            input_primals = self.primal_scorer.input_primals(state)
            output_primals = self.primal_scorer.output_primals(state)
            pooled = tuple(
                self.bank_operator.apply_materialized(
                    video,
                    input_primals,
                    output_primals,
                )
                for video in videos
            )
            return self._output(state, pooled, s_ref=s_ref)

    def forward_compact(
        self,
        program: NaturalProgram,
        videos: Sequence[CompactPrimalDualVideo],
        *,
        s_ref: torch.Tensor,
        bank_contexts: Sequence[ProgramBankContext] | None = None,
        interaction_off: bool = False,
    ) -> SharedCompilerOutput:
        """P2 replay of cached raw X/Y without storing expanded output banks."""

        if len(videos) not in (1, 2, 4) or s_ref.shape != (len(self.owners),):
            raise NativeFactorError("compact compiler video set changed")
        if bank_contexts is None:
            if not interaction_off:
                raise NativeFactorError("compact compiler bank context changed")
            bank_contexts = ()
        elif len(bank_contexts) != len(videos):
            raise NativeFactorError("compact compiler bank context changed")
        with self.bank_operator.ieee_matmul(s_ref.device):
            state: PrimalProgramState = self.primal_scorer.program_state(program)
            input_primals = self.primal_scorer.input_primals(state)
            output_primals = self.primal_scorer.output_primals(state)
            interaction_states: tuple[ProgramBankInteractionState | None, ...] = (
                self._interaction_states(state, bank_contexts)
                if bank_contexts
                else (None,) * len(videos)
            )
            pooled = tuple(
                self.bank_operator.apply_compact(
                    video,
                    input_primals,
                    output_primals,
                    interaction_scorer=self.interaction_scorer,
                    interaction_state=interaction_state,
                    interaction_off=interaction_off,
                )
                for video, interaction_state in zip(
                    videos, interaction_states, strict=True
                )
            )
            return self._output(state, pooled, s_ref=s_ref)

    def _output(
        self,
        state: PrimalProgramState,
        pooled: Sequence[PrimalDualVideoResult],
        *,
        s_ref: torch.Tensor,
    ) -> SharedCompilerOutput:
        if len(pooled) not in (1, 2, 4):
            raise NativeFactorError("compiler pooled video set changed")
        beta = state.rank.new_full((len(pooled),), 1.0 / len(pooled))
        scale_logits = torch.atanh(self.scale_prior_ratio.to(state.rank))
        scale_logits = scale_logits + self.scale_head(
            state.stable_rank.detach()
        ).squeeze(-1)
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
