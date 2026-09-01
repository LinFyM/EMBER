"""Canonical PNBTT Writer compiler over frozen PI0.5 native banks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ember.ecp.bank_conditioning.primal_dual_runtime import (
    CompactPrimalDualVideo,
    PrimalDualVideoOperator,
)
from ember.ecp.bank_conditioning.program_bank_interaction import ProgramBankContext
from ember.ecp.bank_conditioning.tangent_transport import (
    NativeBankTangentTransport,
    ProgramTangentQuery,
    TangentTransportResult,
    TangentTransportVideo,
    pnbtt_event_weights,
)
from ember.ecp.contracts import TargetOwner
from ember.ecp.native_factors import NativeFactorError, NativeVideoReadout
from ember.ecp.natural_program import NaturalProgram


@dataclass(frozen=True)
class SharedCompilerVideo:
    """One independently ordered video and its G2 alignment evidence."""

    native: NativeVideoReadout
    canonical_assignment: torch.Tensor
    frame_positions: torch.Tensor
    local_scene: torch.Tensor
    local_process: torch.Tensor
    local_presence: torch.Tensor
    local_tau: torch.Tensor
    local_sigma: torch.Tensor


SharedCompilerOutput = TangentTransportResult


class SharedNativeFactorCompiler(torch.nn.Module):
    """Generate the only rank-four residual through PNBTT.

    ``bank_operator`` is retained temporarily only as the serializer for the
    existing frozen K1 X/Y caches.  It is not on the PNBTT deployment forward;
    B0/B1 are owned solely by ``tangent_transport``.
    """

    native_dual_matmul_precision = "ieee_fp32_no_tf32"

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        program_width: int = 128,
        event_slots: int = 8,
        key_width: int = 128,
        key_hidden_width: int = 128,
        target_key_rank: int = 16,
        query_hidden_width: int = 256,
        covariance_ridge: float = 1e-3,
        native_rms_epsilon: float = 1e-6,
        direction_epsilon: float = 1e-2,
        query_epsilon: float = 1e-4,
        score_epsilon: float = 1e-4,
        replay_chunk_size: int = 2048,
        temperature_by_side: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0),
        type_balance: torch.Tensor | None = None,
        scale_prior_ratio: torch.Tensor | None = None,
        relative_eigenvalue_floor: float = 1e-6,
        replay_score_rms: float = 0.02,
        covariance_frame_chunk: int = 4,
        inverse_covariance_power: float = 1.0,
        **retired_options: object,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.program_width = int(program_width)
        self.event_slots = int(event_slots)
        if type_balance is None:
            type_balance = torch.full((4, 4), 0.25, dtype=torch.float32)
        if scale_prior_ratio is None:
            scale_prior_ratio = torch.full(
                (len(self.owners), 4), 0.1, dtype=torch.float32
            )
        if retired_options:
            unexpected = ", ".join(sorted(retired_options))
            raise NativeFactorError(f"retired compiler option reached PNBTT: {unexpected}")
        self.query = ProgramTangentQuery(
            self.owners,
            program_width=self.program_width,
            event_slots=self.event_slots,
            key_width=int(key_width),
            hidden_width=int(query_hidden_width),
            query_epsilon=float(query_epsilon),
        )
        self.tangent_transport = NativeBankTangentTransport(
            self.owners,
            event_slots=self.event_slots,
            key_width=int(key_width),
            key_hidden_width=int(key_hidden_width),
            target_key_rank=int(target_key_rank),
            covariance_ridge=float(covariance_ridge),
            native_rms_epsilon=float(native_rms_epsilon),
            direction_epsilon=float(direction_epsilon),
            score_epsilon=float(score_epsilon),
            replay_chunk_size=int(replay_chunk_size),
            temperature_by_side=temperature_by_side,
            type_balance=type_balance,
            scale_prior_ratio=scale_prior_ratio,
        )
        self.bank_operator = PrimalDualVideoOperator(
            self.owners,
            program_width=self.program_width,
            event_slots=self.event_slots,
            relative_eigenvalue_floor=float(relative_eigenvalue_floor),
            replay_score_rms=float(replay_score_rms),
            covariance_frame_chunk=int(covariance_frame_chunk),
            inverse_covariance_power=float(inverse_covariance_power),
        )

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

    def _compact_native(self, video: SharedCompilerVideo) -> CompactPrimalDualVideo:
        """Stream only raw frozen X/Y needed by PNBTT; never solve the old dual."""

        self.bank_operator.validate_video(video)
        input_blocks: list[list[torch.Tensor]] = [[] for _ in self.owners]
        output_blocks: list[list[torch.Tensor]] = [[] for _ in self.owners]
        next_frame = 0
        with torch.no_grad():
            for chunk in video.native.chunks():
                if chunk.start_frame != next_frame or chunk.frame_count <= 0:
                    raise NativeFactorError("PNBTT native stream changed")
                for target, (x, y) in enumerate(
                    zip(chunk.inputs, chunk.outputs, strict=True)
                ):
                    input_blocks[target].append(x.detach())
                    output_blocks[target].append(y.detach())
                next_frame += chunk.frame_count
        if next_frame != video.native.frame_count:
            raise NativeFactorError("PNBTT native stream ended early")
        return CompactPrimalDualVideo(
            frame_measure=self.bank_operator.quadrature(video.frame_positions),
            input_operators=(),
            output_operators=(),
            input_values=tuple(torch.cat(rows, dim=0) for rows in input_blocks),
            output_values=tuple(torch.cat(rows, dim=0) for rows in output_blocks),
            final_outputs=tuple(
                value.detach() for value in video.native.final_outputs
            ),
        )

    def forward(
        self,
        program: NaturalProgram,
        videos: Sequence[SharedCompilerVideo],
        *,
        s_ref: torch.Tensor,
        query_override: torch.Tensor | None = None,
    ) -> SharedCompilerOutput:
        if len(videos) not in (1, 2, 4):
            raise NativeFactorError("PNBTT video cardinality changed")
        with self.bank_operator.ieee_matmul(s_ref.device):
            compact = tuple(self._compact_native(video) for video in videos)
            contexts = tuple(self._bank_context(video) for video in videos)
            return self.forward_compact(
                program,
                compact,
                s_ref=s_ref,
                bank_contexts=contexts,
                query_override=query_override,
            )

    def forward_compact(
        self,
        program: NaturalProgram,
        videos: Sequence[CompactPrimalDualVideo],
        *,
        s_ref: torch.Tensor,
        bank_contexts: Sequence[ProgramBankContext],
        query_override: torch.Tensor | None = None,
    ) -> SharedCompilerOutput:
        if len(videos) not in (1, 2, 4) or len(bank_contexts) != len(videos):
            raise NativeFactorError("PNBTT compact video/context cardinality changed")
        queries = self.query(program) if query_override is None else query_override
        tangent_videos = tuple(
            TangentTransportVideo(native=video, context=context)
            for video, context in zip(videos, bank_contexts, strict=True)
        )
        with self.bank_operator.ieee_matmul(s_ref.device):
            return self.tangent_transport(
                queries=queries,
                videos=tangent_videos,
                event_weights=pnbtt_event_weights(program),
                s_ref=s_ref,
            )
