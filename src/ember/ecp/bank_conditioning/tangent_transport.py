"""Program-conditioned real native-bank tangent transport (PNBTT)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, partial
from typing import Sequence

import torch
from torch.utils.checkpoint import checkpoint

from ember.ecp.bank_conditioning.key_value_replay import (
    KeyValueSignedPoolResult,
    differentiable_key_moments,
    safe_rms_normalize,
    signed_key_value_pool,
    whiten_queries,
)
from ember.ecp.bank_conditioning.operator import BankConditioningError
from ember.ecp.bank_conditioning.primal_dual import native_candidate_mass
from ember.ecp.bank_conditioning.primal_dual_runtime import CompactPrimalDualVideo
from ember.ecp.bank_conditioning.program_bank_interaction import ProgramBankContext
from ember.ecp.bank_conditioning.tangent_parameterization import (
    PNBTT_SIDES,
    NativeTangentKey,
    ProgramTangentQuery,
    TaskLocalFreeTangentQuery,
)
from ember.ecp.contracts import ACTION_HORIZON, TargetFamily, TargetOwner
from ember.ecp.native_factors import (
    G1_PROBE_COUNT,
    G1_RESIDUAL_RANK,
    NativeFactorResidual,
    NativeOutputBankState,
    OUTPUT_BANK_TYPES,
    native_output_group_count,
)
from ember.ecp.natural_program import NaturalProgram


PNBTT_FAMILIES = (
    TargetFamily.Q,
    TargetFamily.V,
    TargetFamily.ACTION_IN,
    TargetFamily.ACTION_OUT,
)


@dataclass(frozen=True)
class TangentTransportVideo:
    native: CompactPrimalDualVideo
    context: ProgramBankContext

    @cached_property
    def canonical_order_key(self) -> tuple[bytes, ...]:
        """Content identity used only to stabilize unordered-set reductions."""

        context_tensors = (
            self.context.canonical_assignment,
            self.context.frame_positions,
            self.context.local_scene,
            self.context.local_process,
            self.context.local_presence,
            self.context.local_tau,
            self.context.local_sigma,
        )
        native_tensors = (
            self.native.frame_measure,
            self.native.input_values[0],
            self.native.input_values[-1],
            self.native.output_values[0],
            self.native.output_values[-1],
            self.native.final_outputs[0],
            self.native.final_outputs[-1],
        )
        context = torch.cat(
            tuple(value.detach().float().flatten() for value in context_tensors)
        )
        native_samples = []
        for value in native_tensors:
            flat = value.detach().float().flatten()
            if flat.numel() > 16:
                indices = torch.linspace(
                    0, flat.numel() - 1, 16, device=flat.device
                ).long()
                flat = flat.index_select(0, indices)
            native_samples.append(flat)
        native = torch.cat(native_samples)
        return (
            repr(tuple(tuple(value.shape) for value in context_tensors)).encode(),
            context.contiguous().cpu().numpy().tobytes(),
            repr(tuple(tuple(value.shape) for value in native_tensors)).encode(),
            native.contiguous().cpu().numpy().tobytes(),
        )


@dataclass(frozen=True)
class TangentTransportResult:
    residual: NativeFactorResidual
    input_directions: tuple[torch.Tensor, ...]
    output_directions: tuple[torch.Tensor, ...]
    video_weights: torch.Tensor
    solve_metrics: torch.Tensor
    conditioning_metrics: torch.Tensor


@dataclass(frozen=True)
class _CandidateScope:
    values: torch.Tensor
    metadata: torch.Tensor
    base_mass: torch.Tensor
    event_mass: torch.Tensor
    side_indices: tuple[int, ...]
    output_groups: int


class NativeBankTangentTransport(torch.nn.Module):
    """Joint-K B0/B1 PNBTT owner producing the only rank-four residual."""

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        event_slots: int,
        key_width: int,
        key_hidden_width: int,
        target_key_rank: int,
        covariance_ridge: float,
        native_rms_epsilon: float,
        direction_epsilon: float,
        score_epsilon: float,
        replay_chunk_size: int,
        temperature_by_side: Sequence[float],
        type_balance: torch.Tensor,
        scale_prior_ratio: torch.Tensor,
        residual_rank: int = G1_RESIDUAL_RANK,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.event_slots = int(event_slots)
        self.key_width = int(key_width)
        self.covariance_ridge = float(covariance_ridge)
        self.native_rms_epsilon = float(native_rms_epsilon)
        self.direction_epsilon = float(direction_epsilon)
        self.score_epsilon = float(score_epsilon)
        self.replay_chunk_size = int(replay_chunk_size)
        self.residual_rank = int(residual_rank)
        if self.residual_rank <= 0:
            raise BankConditioningError("PNBTT residual rank must be positive")
        self.key_encoder = NativeTangentKey(
            self.owners,
            key_width=self.key_width,
            hidden_width=int(key_hidden_width),
            target_projection_rank=int(target_key_rank),
        )
        temperatures = torch.tensor(tuple(temperature_by_side), dtype=torch.float32)
        if temperatures.shape != (len(PNBTT_SIDES),) or torch.any(temperatures <= 0):
            raise BankConditioningError("PNBTT temperature contract changed")
        if type_balance.shape != (len(PNBTT_FAMILIES), len(OUTPUT_BANK_TYPES)):
            raise BankConditioningError("PNBTT fixed type balance changed")
        if scale_prior_ratio.shape != (len(self.owners), self.residual_rank):
            raise BankConditioningError("PNBTT frozen scale prior changed")
        self.register_buffer("temperature_by_side", temperatures, persistent=True)
        self.register_buffer("type_balance", type_balance.detach().float(), persistent=True)
        self.register_buffer(
            "scale_prior_ratio", scale_prior_ratio.detach().float(), persistent=True
        )

    @staticmethod
    def _canonical_videos(
        videos: Sequence[TangentTransportVideo],
    ) -> tuple[TangentTransportVideo, ...]:
        """Fix only floating reduction order for a mathematically unordered set."""

        packed = tuple(videos)
        if len(packed) <= 1:
            return packed

        return tuple(sorted(packed, key=lambda video: video.canonical_order_key))

    @staticmethod
    def _equal_video_event_mass(value: torch.Tensor) -> torch.Tensor:
        """Give each valid video-event scope unit mass before the fixed 1/K mix."""

        if value.ndim < 2 or torch.any(value < 0) or not bool(torch.isfinite(value).all()):
            raise BankConditioningError("PNBTT per-video event measure changed")
        total = value.sum(-1, keepdim=True)
        return torch.where(
            total > 0,
            value / total.clamp_min(1e-30),
            torch.zeros_like(value),
        )

    @staticmethod
    def _metadata(context: ProgramBankContext) -> torch.Tensor:
        frames = context.frame_positions.shape[0]
        frame = context.frame_positions.float().mul(2).sub(1)[:, None, None]
        probe = torch.linspace(-1.0, 1.0, G1_PROBE_COUNT, device=frame.device)[
            None, :, None
        ]
        horizon = torch.linspace(-1.0, 1.0, ACTION_HORIZON, device=frame.device)[
            None, None, :
        ]
        return torch.stack(
            (
                frame.expand(frames, G1_PROBE_COUNT, ACTION_HORIZON),
                probe.expand(frames, G1_PROBE_COUNT, ACTION_HORIZON),
                horizon.expand(frames, G1_PROBE_COUNT, ACTION_HORIZON),
            ),
            dim=-1,
        )

    def _input_scope(
        self,
        videos: Sequence[TangentTransportVideo],
        *,
        target: int,
    ) -> _CandidateScope:
        values, metadata, base_masses, event_masses = [], [], [], []
        beta = 1.0 / len(videos)
        for video in videos:
            native, context = video.native, video.context
            frames = native.frame_measure.shape[0]
            if context.canonical_assignment.shape != (frames, self.event_slots):
                raise BankConditioningError("PNBTT event assignment changed")
            candidate_metadata = self._metadata(context)
            value = native.input_values[target]
            base = native_candidate_mass(native.frame_measure, output=False)
            flat_value = value.reshape(-1, value.shape[-1]).detach()
            flat_base = base.reshape(-1).float()
            flat_base = flat_base / flat_base.sum().clamp_min(1e-30) * beta
            assignment = context.canonical_assignment.float().T
            event = (
                assignment[:, :, None, None] * base[None].float()
            ).reshape(self.event_slots, -1)
            event = self._equal_video_event_mass(event) * beta
            values.append(flat_value)
            metadata.append(candidate_metadata.reshape(-1, 3))
            base_masses.append(flat_base)
            event_masses.append(event)
        return _CandidateScope(
            values=torch.cat(values)[None],
            metadata=torch.cat(metadata)[None],
            base_mass=torch.cat(base_masses)[None],
            event_mass=torch.cat(event_masses, dim=-1)[None],
            side_indices=(0,),
            output_groups=1,
        )

    def _output_scope(
        self,
        videos: Sequence[TangentTransportVideo],
        *,
        target: int,
    ) -> _CandidateScope:
        values, metadata, base_masses, event_masses = [], [], [], []
        groups = native_output_group_count(self.owners[target])
        width = self.owners[target].out_features // groups
        types = len(OUTPUT_BANK_TYPES)
        beta = 1.0 / len(videos)
        for video in videos:
            native, context = video.native, video.context
            frames = native.frame_measure.shape[0]
            if context.canonical_assignment.shape != (frames, self.event_slots):
                raise BankConditioningError("PNBTT event assignment changed")
            boundary = NativeOutputBankState(final=native.final_outputs[target])
            bank = boundary.build(native.output_values[target], start_frame=0)
            bank = bank.reshape(*bank.shape[:-1], groups, width).movedim(-2, 0)
            # [group, frame, probe, horizon, type, width] ->
            # [type, group, candidate, width], keeping one joint measure.
            scoped_values = bank.permute(4, 0, 1, 2, 3, 5).reshape(
                types, groups, -1, width
            )
            base = native_candidate_mass(native.frame_measure, output=True)
            normalized_base = (
                base.float()
                / base.float().sum(dim=(0, 1, 2), keepdim=True).clamp_min(1e-30)
                * beta
            )
            scoped_base = normalized_base.permute(3, 0, 1, 2).reshape(
                types, 1, -1
            )
            scoped_base = scoped_base.expand(-1, groups, -1).float()
            assignment = context.canonical_assignment.float().T
            event = assignment[:, :, None, None, None] * base[None].float()
            event = event.permute(4, 0, 1, 2, 3).reshape(
                types, self.event_slots, -1
            )
            event = self._equal_video_event_mass(event) * beta
            event = event[:, None].expand(-1, groups, -1, -1)
            candidate_metadata = self._metadata(context).reshape(-1, 3)
            scoped_metadata = candidate_metadata[None, None].expand(
                types, groups, -1, -1
            )
            values.append(scoped_values.detach())
            base_masses.append(scoped_base)
            event_masses.append(event)
            metadata.append(scoped_metadata)
        return _CandidateScope(
            values=torch.cat(values, dim=2).flatten(0, 1),
            metadata=torch.cat(metadata, dim=2).flatten(0, 1),
            base_mass=torch.cat(base_masses, dim=2).flatten(0, 1),
            event_mass=torch.cat(event_masses, dim=3).flatten(0, 1),
            side_indices=tuple(
                side
                for side in range(1, len(PNBTT_SIDES))
                for _ in range(groups)
            ),
            output_groups=groups,
        )

    def _transport_scope(
        self,
        scope: _CandidateScope,
        *,
        target: int,
        queries: torch.Tensor,
    ) -> tuple[KeyValueSignedPoolResult, torch.Tensor]:
        mean = torch.einsum("sn,snd->sd", scope.base_mass, scope.values.float())
        centered = scope.values.float() - mean[:, None]
        rms = (
            torch.einsum("sn,snd->s", scope.base_mass, centered.square())
            .div(centered.shape[-1])
            .clamp_min(0)
            .sqrt()
        )
        normalized = centered / rms.clamp_min(self.native_rms_epsilon)[:, None, None]
        key_blocks = []
        start = 0
        while start < len(scope.side_indices):
            side = scope.side_indices[start]
            stop = start + 1
            while stop < len(scope.side_indices) and scope.side_indices[stop] == side:
                stop += 1
            key_blocks.append(
                self.key_encoder(
                    target=target,
                    side=side,
                    normalized_values=normalized[start:stop],
                    metadata=scope.metadata[start:stop],
                )
            )
            start = stop
        keys = torch.cat(key_blocks)
        moments = differentiable_key_moments(
            keys, scope.event_mass, ridge=self.covariance_ridge
        )
        whitened = whiten_queries(queries, moments)
        result = signed_key_value_pool(
            keys=keys,
            values=scope.values,
            moments=moments,
            whitened_queries=whitened,
            temperature=torch.stack(
                tuple(self.temperature_by_side[side] for side in scope.side_indices)
            ),
            score_epsilon=self.score_epsilon,
            chunk_size=self.replay_chunk_size,
        )
        diagonal = moments.cholesky.diagonal(dim1=-2, dim2=-1)
        metric = torch.stack(
            (
                result.score_rms.detach().amin(dim=(1, 2)),
                result.score_rms.detach().amax(dim=(1, 2)),
                diagonal.detach().amin(dim=(1, 2)),
                moments.covariance.detach()
                .diagonal(dim1=-2, dim2=-1)
                .sum(-1)
                .mean(-1),
                result.positive_maximum_weight.detach().amax(dim=(1, 2)),
                result.negative_maximum_weight.detach().amax(dim=(1, 2)),
            ),
            dim=-1,
        )
        return result, metric

    def _target_transport(
        self,
        target_queries: torch.Tensor,
        rho: torch.Tensor,
        *,
        videos: Sequence[TangentTransportVideo],
        target: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transport one target so autograd can discard its B0/B1 activations."""

        owner = self.owners[target]
        input_scope = self._input_scope(videos, target=target)
        input_result, input_metric = self._transport_scope(
            input_scope,
            target=target,
            queries=target_queries[:, :, 0][None],
        )
        raw_input = torch.einsum(
            "e,sred->srd", rho, input_result.direction
        )[0]
        groups = native_output_group_count(owner)
        family = PNBTT_FAMILIES.index(owner.family)
        output_scope = self._output_scope(videos, target=target)
        output_queries = torch.cat(
            tuple(
                target_queries[:, :, side][None].expand(groups, -1, -1, -1)
                for side in range(1, len(PNBTT_SIDES))
            ),
            dim=0,
        )
        output_result, output_metric = self._transport_scope(
            output_scope,
            target=target,
            queries=output_queries,
        )
        directions = torch.einsum(
            "e,sred->srd", rho, output_result.direction
        ).reshape(len(OUTPUT_BANK_TYPES), groups, self.residual_rank, -1)
        normalized_types = safe_rms_normalize(
            directions, epsilon=self.direction_epsilon
        )
        combined = torch.einsum(
            "t,tgrd->grd", self.type_balance[family], normalized_types
        )
        raw_output = combined.permute(1, 0, 2).reshape(self.residual_rank, -1)
        return raw_input, raw_output, torch.cat((input_metric, output_metric))

    def forward_target(
        self,
        *,
        target: int,
        target_queries: torch.Tensor,
        videos: Sequence[TangentTransportVideo],
        event_weights: torch.Tensor,
        s_ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Materialize one target for memory-bounded functional chain rule."""

        if (
            not 0 <= int(target) < len(self.owners)
            or target_queries.shape
            != (
                self.residual_rank,
                self.event_slots,
                len(PNBTT_SIDES),
                self.key_width,
            )
            or len(videos) not in (1, 2, 4)
            or event_weights.shape != (self.event_slots,)
            or s_ref.ndim != 0
        ):
            raise BankConditioningError("PNBTT target transport input changed")
        videos = self._canonical_videos(videos)
        rho = event_weights.float().clamp_min(0)
        rho = rho / rho.sum().clamp_min(1e-30)
        raw_input, raw_output, metric = self._target_transport(
            target_queries,
            rho,
            videos=videos,
            target=int(target),
        )
        a = safe_rms_normalize(raw_input, epsilon=self.direction_epsilon)
        direction = safe_rms_normalize(raw_output, epsilon=self.direction_epsilon)
        scale = s_ref.to(self.scale_prior_ratio) * self.scale_prior_ratio[int(target)]
        return a, direction * scale[:, None], metric

    def forward(
        self,
        *,
        queries: torch.Tensor,
        videos: Sequence[TangentTransportVideo],
        event_weights: torch.Tensor,
        s_ref: torch.Tensor,
    ) -> TangentTransportResult:
        if (
            len(videos) not in (1, 2, 4)
            or queries.shape
            != (
                len(self.owners),
                self.residual_rank,
                self.event_slots,
                len(PNBTT_SIDES),
                self.key_width,
            )
            or event_weights.shape != (self.event_slots,)
            or s_ref.shape != (len(self.owners),)
        ):
            raise BankConditioningError("PNBTT transport input contract changed")
        videos = self._canonical_videos(videos)
        rho = event_weights.float().clamp_min(0)
        rho = rho / rho.sum().clamp_min(1e-30)
        raw_inputs, raw_outputs, metrics = [], [], []
        for target in range(len(self.owners)):
            operation = partial(
                self._target_transport,
                videos=videos,
                target=target,
            )
            if torch.is_grad_enabled():
                raw_input, raw_output, metric = checkpoint(
                    operation,
                    queries[target],
                    rho,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                raw_input, raw_output, metric = operation(queries[target], rho)
            raw_inputs.append(raw_input)
            raw_outputs.append(raw_output)
            metrics.extend(metric.unbind(0))

        input_directions = tuple(
            safe_rms_normalize(value, epsilon=self.direction_epsilon)
            for value in raw_inputs
        )
        output_directions = tuple(
            safe_rms_normalize(value, epsilon=self.direction_epsilon)
            for value in raw_outputs
        )
        scales = s_ref[:, None].to(self.scale_prior_ratio) * self.scale_prior_ratio
        residual = NativeFactorResidual(
            a=input_directions,
            b=tuple(
                direction * scales[target, :, None]
                for target, direction in enumerate(output_directions)
            ),
            scales=scales,
        )
        metric_tensor = torch.stack(metrics)
        return TangentTransportResult(
            residual=residual,
            input_directions=input_directions,
            output_directions=output_directions,
            video_weights=s_ref.new_full((len(videos),), 1.0 / len(videos)),
            solve_metrics=metric_tensor,
            conditioning_metrics=torch.stack(
                (
                    metric_tensor[:, 0].amin(),
                    metric_tensor[:, 1].amax(),
                    metric_tensor[:, 2].amin(),
                    metric_tensor[:, 3].mean(),
                    metric_tensor[:, 4].amax(),
                    metric_tensor[:, 5].amax(),
                )
            ),
        )


def pnbtt_event_weights(program: NaturalProgram) -> torch.Tensor:
    """The sole event aggregation authority used by both E1 and deployment."""

    return program.rho
