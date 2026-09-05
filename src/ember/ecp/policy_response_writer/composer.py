"""Unified frozen-policy-evidence to native-factor Writer blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ember.ecp.contracts import ACTION_HORIZON, TargetFamily, TargetOwner
from ember.ecp.native_factors import (
    G1_RESIDUAL_RANK,
    NativeFactorResidual,
    NativeOutputBankState,
    OnlineSoftmaxAccumulator,
    native_output_group_count,
)
from ember.ecp.policy_response_writer.capture import FrozenPolicyResponseVideo
from ember.ecp.policy_response_writer.process import (
    UnifiedPolicyNativeFactorBlock,
    PolicyResponseEvidence,
)


@dataclass(frozen=True)
class _NativeBankChunk:
    """Side-separated bank tokens and untouched native values for real frames."""

    input_values: torch.Tensor
    output_values: torch.Tensor
    input_tokens: torch.Tensor
    output_tokens: torch.Tensor
    position_tokens: torch.Tensor


@dataclass(frozen=True)
class _NativeVideoCandidates:
    frame_count: int
    chunks: tuple[_NativeBankChunk, ...]


@dataclass(frozen=True)
class _FactorVideoState:
    """Static bank-localizing context and frame-relative dynamic innovation."""

    context: torch.Tensor
    innovation: torch.Tensor


class _GroupedOnlineSoftmaxAccumulator:
    """Exact signed pooling with an independent native-output group axis."""

    def __init__(
        self, *, groups: int, ranks: int, width: int, device: torch.device
    ) -> None:
        self.maximum = torch.full(
            (groups, ranks, 2), -torch.inf, dtype=torch.float32, device=device
        )
        self.normalizer = torch.zeros(
            groups, ranks, 2, dtype=torch.float32, device=device
        )
        self.weighted_sum = torch.zeros(
            groups, ranks, 2, width, dtype=torch.float32, device=device
        )

    def add(self, logits: torch.Tensor, values: torch.Tensor) -> None:
        # logits: [rank, branch, frame, group, ...]
        # values: [group, frame, ..., native_width]
        if logits.ndim < 5 or values.ndim != logits.ndim - 1:
            raise ValueError("grouped signed-pooling ranks changed")
        grouped_logits = logits.movedim(3, 0)
        if (
            grouped_logits.shape[:3] != self.maximum.shape
            or values.shape[0] != self.maximum.shape[0]
            or grouped_logits.shape[3:] != values.shape[1:-1]
        ):
            raise ValueError("grouped logits and native values do not align")
        flat_logits = grouped_logits.float().flatten(3)
        flat_values = values.float().flatten(1, -2)
        chunk_maximum = flat_logits.amax(-1)
        maximum = torch.maximum(self.maximum, chunk_maximum)
        old_scale = torch.exp(self.maximum - maximum)
        weights = torch.exp(flat_logits - maximum[..., None])
        self.weighted_sum = self.weighted_sum * old_scale[..., None] + torch.einsum(
            "grbn,gnd->grbd", weights, flat_values
        )
        self.normalizer = self.normalizer * old_scale + weights.sum(-1)
        self.maximum = maximum

    def signed_mean(self) -> torch.Tensor:
        if torch.any(self.normalizer <= 0):
            raise ValueError("grouped signed pooling received no candidates")
        means = self.weighted_sum / self.normalizer[..., None]
        signed = means[:, :, 0] - means[:, :, 1]
        return signed.permute(1, 0, 2).flatten(1)


def _effective_update_mean_square(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("native factor shapes changed")
    a32 = a.float()
    b32 = b.float()
    a_gram = a32 @ a32.transpose(0, 1)
    b_gram = b32 @ b32.transpose(0, 1)
    squared_frobenius = (a_gram * b_gram).sum()
    return (squared_frobenius / float(a.shape[1] * b.shape[1])).clamp_min(0.0)


def _effective_update_rms(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return matrix RMS of ``b.T @ a`` without materializing the matrix."""

    return _effective_update_mean_square(a, b).sqrt()


def _effective_update_cap_factor(
    a: torch.Tensor,
    b: torch.Tensor,
    cap: torch.Tensor,
) -> torch.Tensor:
    """Bound the complete target update without changing its direction."""

    mean_square = _effective_update_mean_square(a, b)
    denominator = mean_square.clamp_min(torch.finfo(mean_square.dtype).tiny).sqrt()
    return torch.clamp(cap.to(mean_square) / denominator, max=1.0)




class UnifiedPolicyNativeFactorGenerator(torch.nn.Module):
    """Generate rank-four X/Y with one repeated factor-latent block type."""

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        width: int = 128,
        heads: int = 4,
        block_depth: int = 2,
        pooling_frame_chunk: int = 4,
        task_local: bool = False,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.width = width
        self.pooling_frame_chunk = pooling_frame_chunk
        if (
            not owners
            or width % heads
            or block_depth <= 0
            or pooling_frame_chunk <= 0
        ):
            raise ValueError("unified policy-native factor topology changed")
        input_widths = sorted({owner.in_features for owner in owners})
        output_widths = sorted(
            {owner.out_features // native_output_group_count(owner) for owner in owners}
        )
        self.input_projection = torch.nn.ModuleDict(
            {
                str(value): torch.nn.Linear(value, width, bias=False)
                for value in input_widths
            }
        )
        self.output_projection = torch.nn.ModuleDict(
            {
                str(value): torch.nn.Linear(value, width, bias=False)
                for value in output_widths
            }
        )
        maximum_groups = max(native_output_group_count(owner) for owner in owners)
        self.rank_embedding = torch.nn.Parameter(
            torch.empty(G1_RESIDUAL_RANK, width)
        )
        self.factor_side_embedding = torch.nn.Parameter(torch.empty(2, width))
        self.owner_embedding = torch.nn.Parameter(torch.empty(len(owners), width))
        self.family_embedding = torch.nn.Embedding(len(TargetFamily), width)
        self.probe_embedding = torch.nn.Embedding(2, width)
        self.horizon_embedding = torch.nn.Embedding(ACTION_HORIZON, width)
        self.bank_type_embedding = torch.nn.Embedding(5, width)
        self.group_embedding = torch.nn.Embedding(maximum_groups, width)
        self.position_projection = torch.nn.Linear(2, width, bias=False)
        self.native_key_norm = torch.nn.LayerNorm(width)
        self.blocks = torch.nn.ModuleList(
            UnifiedPolicyNativeFactorBlock(width, heads)
            for _ in range(block_depth)
        )
        # B shares its context query, so zero innovation closes the complete
        # update. A uses distinct context queries over the current native bank.
        self.input_signed_query = torch.nn.Linear(width, 3 * width, bias=False)
        self.output_signed_query = torch.nn.Linear(width, 3 * width, bias=False)
        self.task_query = (
            torch.nn.Parameter(
                torch.zeros(len(owners), G1_RESIDUAL_RANK, 2, width)
            )
            if task_local
            else None
        )
        family_order = tuple(TargetFamily)
        self.register_buffer(
            "family_ids",
            torch.tensor([family_order.index(owner.family) for owner in owners]),
            persistent=False,
        )
        torch.nn.init.normal_(self.rank_embedding, std=width**-0.5)
        torch.nn.init.normal_(self.factor_side_embedding, std=width**-0.5)
        torch.nn.init.normal_(self.owner_embedding, std=width**-0.5)
        # Preserve the established asymmetric initialization and later RNG draws.
        with torch.random.fork_rng(devices=[]):
            expanded = torch.nn.Linear(width, 4 * width, bias=False)
        with torch.no_grad():
            expanded.weight[:width].copy_(self.input_signed_query.weight[:width])
            expanded.weight[2 * width:].copy_(self.input_signed_query.weight[width:])
        self.input_signed_query = expanded

    def _owner_bias(self, target: int) -> torch.Tensor:
        return self.owner_embedding[target] + self.family_embedding(
            self.family_ids[target]
        )

    def _seed(self, target: int) -> torch.Tensor:
        return (
            self.rank_embedding[:, None]
            + self.factor_side_embedding[None]
            + self._owner_bias(target)
        )

    def _position(self, video: FrozenPolicyResponseVideo) -> torch.Tensor:
        position = video.frame_positions.float()
        if position.shape != (video.frame_count,):
            raise ValueError("unified factor frame positions changed")
        return self.position_projection(torch.stack((position, position.square()), -1))

    def _input_context(self, target: int, values: torch.Tensor) -> torch.Tensor:
        owner = self.owners[target]
        key = self.input_projection[str(owner.in_features)](values)
        key = key + self._owner_bias(target)
        key = key + self.probe_embedding.weight[None, :, None]
        key = key + self.horizon_embedding.weight[None, None]
        return key + self.bank_type_embedding.weight[0]

    def _output_context(self, target: int, values: torch.Tensor) -> torch.Tensor:
        # values: [group, frame, probe, horizon, bank_type, native_width]
        key = self.output_projection[str(values.shape[-1])](values)
        key = key + self._owner_bias(target)
        key = key + self.probe_embedding.weight[None, None, :, None, None]
        key = key + self.horizon_embedding.weight[None, None, None, :, None]
        key = key + self.bank_type_embedding.weight[1:][None, None, None, None]
        return key + self.group_embedding.weight[
            : values.shape[0], None, None, None, None
        ]

    def _pooling_keys(
        self, chunk: _NativeBankChunk
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_keys = self.native_key_norm(
            chunk.input_tokens + chunk.position_tokens[:, None, None]
        )
        output_keys = self.native_key_norm(
            chunk.output_tokens
            + chunk.position_tokens[None, :, None, None, None]
        )
        return input_keys, output_keys

    def _bank_candidates(
        self,
        target: int,
        videos: Sequence[FrozenPolicyResponseVideo],
    ) -> tuple[_NativeVideoCandidates, ...]:
        candidates = []
        owner = self.owners[target]
        groups = native_output_group_count(owner)
        group_width = owner.out_features // groups
        for video in videos:
            positions = self._position(video)
            boundary = NativeOutputBankState(final=video.final_outputs[target])
            chunks = []
            for start in range(0, video.frame_count, self.pooling_frame_chunk):
                stop = min(start + self.pooling_frame_chunk, video.frame_count)
                input_values = video.native_inputs[target][start:stop]
                output = boundary.build(
                    video.native_outputs[target][start:stop], start_frame=start
                )
                grouped = output.reshape(
                    *output.shape[:-1], groups, group_width
                ).movedim(-2, 0)
                chunks.append(
                    _NativeBankChunk(
                        input_values=input_values,
                        output_values=grouped,
                        input_tokens=self._input_context(target, input_values),
                        output_tokens=self._output_context(target, grouped),
                        position_tokens=positions[start:stop],
                    )
                )
            if boundary.next_frame != video.frame_count:
                raise ValueError("unified factor bank stream ended early")
            candidates.append(
                _NativeVideoCandidates(
                    frame_count=video.frame_count,
                    chunks=tuple(chunks),
                )
            )
        if not candidates:
            raise ValueError("unified factor received no current-video bank")
        return tuple(candidates)

    def _decode_target(
        self,
        target: int,
        videos: Sequence[FrozenPolicyResponseVideo],
        evidence: Sequence[PolicyResponseEvidence],
        candidates: Sequence[_NativeVideoCandidates],
    ) -> tuple[_FactorVideoState, ...]:
        if len(videos) != len(evidence) or len(videos) != len(candidates):
            raise ValueError("unified factor video/evidence count changed")
        states = []
        selected_evidence = []
        for video, tokens in zip(videos, evidence, strict=True):
            if (
                tokens.patches.shape[0] != video.frame_count
                or tokens.language.shape[0] != video.frame_count
                or tokens.language_valid.shape != tokens.language.shape[:2]
                or tokens.response.shape[:2]
                != (video.frame_count, len(self.owners))
                or tokens.response.shape[-1] != self.width
            ):
                raise ValueError("unified factor evidence axes changed")
            state = self._seed(target)[None].expand(
                video.frame_count, -1, -1, -1
            )
            if self.task_query is not None:
                state = state + self.task_query[target][None]
            states.append(state)
            selected_evidence.append(
                PolicyResponseEvidence(
                    patches=tokens.patches,
                    language=tokens.language,
                    language_valid=tokens.language_valid,
                    response=tokens.response[:, target],
                )
            )

        positions = tuple(video.frame_positions for video in videos)
        input_chunks = tuple(
            tuple(chunk.input_tokens.flatten(1, -2) for chunk in video.chunks)
            for video in candidates
        )
        output_chunks = tuple(
            tuple(
                chunk.output_tokens.movedim(0, 1).flatten(1, -2)
                for chunk in video.chunks
            )
            for video in candidates
        )
        values = tuple(states)
        for block in self.blocks:
            values = block(
                values,
                positions,
                selected_evidence,
                input_chunks,
                output_chunks,
            )
        output = []
        for value in values:
            context = value.mean(0)
            output.append(
                _FactorVideoState(
                    context=context,
                    innovation=value - context[None],
                )
            )
        return tuple(output)

    def _branch_logits(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        *,
        log_base_mass: float,
    ) -> torch.Tensor:
        if (
            keys.ndim < 3
            or keys.shape[0] != queries.shape[0]
            or keys.shape[-1] != self.width
            or queries.ndim != 4
            or queries.shape[1:] != (G1_RESIDUAL_RANK, 2, self.width)
        ):
            raise ValueError("signed native-attention axes changed")
        logits = torch.einsum("frbd,f...d->rbf...", queries, keys)
        return logits.float() / math.sqrt(self.width) + log_base_mass

    def _checkpointed_branch_logits(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        *,
        log_base_mass: float,
    ) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return self._branch_logits(
                keys, queries, log_base_mass=log_base_mass
            )

        def score(local_keys: torch.Tensor, local_queries: torch.Tensor) -> torch.Tensor:
            return self._branch_logits(
                local_keys, local_queries, log_base_mass=log_base_mass
            )

        return checkpoint(score, keys, queries, use_reentrant=False)

    def _signed_queries(
        self,
        projection: torch.nn.Linear,
        context: torch.Tensor,
        innovation: torch.Tensor,
    ) -> torch.Tensor:
        if (
            projection.in_features != self.width
            or projection.out_features not in (3 * self.width, 4 * self.width)
            or context.shape != (G1_RESIDUAL_RANK, self.width)
            or innovation.ndim != 3
            or innovation.shape[1:] != (G1_RESIDUAL_RANK, self.width)
        ):
            raise ValueError("common-base signed query axes changed")
        base_width = projection.out_features - 2 * self.width
        base = F.linear(context, projection.weight[:base_width]).reshape(
            G1_RESIDUAL_RANK, base_width // self.width, self.width
        )
        offsets = F.linear(
            innovation, projection.weight[base_width:]
        ).reshape(
            innovation.shape[0], G1_RESIDUAL_RANK, 2, self.width
        )
        return base[None] + offsets

    def _pool_target(
        self,
        target: int,
        frame_states: Sequence[_FactorVideoState],
        videos: Sequence[_NativeVideoCandidates],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        owner = self.owners[target]
        groups = native_output_group_count(owner)
        group_width = owner.out_features // groups
        device = frame_states[0].innovation.device
        input_accumulator = OnlineSoftmaxAccumulator(
            ranks=G1_RESIDUAL_RANK,
            width=owner.in_features,
            device=device,
        )
        output_accumulator = _GroupedOnlineSoftmaxAccumulator(
            groups=groups,
            ranks=G1_RESIDUAL_RANK,
            width=group_width,
            device=device,
        )
        video_count = len(videos)
        if len(frame_states) != video_count:
            raise ValueError("unified factor frame-state set changed")
        for video, frame_state in zip(videos, frame_states, strict=True):
            if (
                frame_state.context.shape
                != (G1_RESIDUAL_RANK, 2, self.width)
                or frame_state.innovation.shape
                != (
                    video.frame_count,
                    G1_RESIDUAL_RANK,
                    2,
                    self.width,
                )
            ):
                raise ValueError("unified factor-side frame state changed")
            input_queries = self._signed_queries(
                self.input_signed_query,
                frame_state.context[:, 0],
                frame_state.innovation[:, :, 0],
            )
            output_queries = self._signed_queries(
                self.output_signed_query,
                frame_state.context[:, 1],
                frame_state.innovation[:, :, 1],
            )
            input_mass = -math.log(
                video_count * video.frame_count * 2 * ACTION_HORIZON
            )
            output_mass = -math.log(
                video_count * video.frame_count * 2 * ACTION_HORIZON * 4
            )
            frame_offset = 0
            for chunk in video.chunks:
                frame_count = chunk.input_values.shape[0]
                frame_slice = slice(frame_offset, frame_offset + frame_count)
                input_keys, output_keys = self._pooling_keys(chunk)
                input_accumulator.add(
                    self._checkpointed_branch_logits(
                        input_keys,
                        input_queries[frame_slice],
                        log_base_mass=input_mass,
                    ),
                    chunk.input_values,
                )
                output_accumulator.add(
                    self._checkpointed_branch_logits(
                        output_keys.movedim(0, 1),
                        output_queries[frame_slice],
                        log_base_mass=output_mass,
                    ),
                    chunk.output_values,
                )
                frame_offset += frame_count
            if frame_offset != video.frame_count:
                raise ValueError("unified aligned frame stream ended early")
        return input_accumulator.signed_mean(), output_accumulator.signed_mean()

    def forward(
        self,
        videos: Sequence[FrozenPolicyResponseVideo],
        evidence: Sequence[PolicyResponseEvidence],
        *,
        s_ref: torch.Tensor,
    ) -> NativeFactorResidual:
        values = tuple(videos)
        memories = tuple(evidence)
        if (
            not values
            or len(values) != len(memories)
            or s_ref.shape != (len(self.owners),)
        ):
            raise ValueError("unified factor video set or scale authority changed")
        a_values = []
        b_values = []
        scales = []
        for target in range(len(self.owners)):
            candidates = self._bank_candidates(target, values)
            frame_states = self._decode_target(
                target, values, memories, candidates
            )
            a, b = self._pool_target(target, frame_states, candidates)
            cap_factor = _effective_update_cap_factor(a, b, s_ref[target])
            b = b * cap_factor.to(b)
            rank_scale = (
                a.float().square().mean(-1).sqrt()
                * b.float().square().mean(-1).sqrt()
            )
            a_values.append(a)
            b_values.append(b)
            scales.append(rank_scale)
        return NativeFactorResidual(
            a=tuple(a_values),
            b=tuple(b_values),
            scales=torch.stack(scales),
        )

    @torch.no_grad()
    def initialize_from_stage0(self, stage0: torch.nn.Module) -> dict[str, object]:
        """Reuse G2 structural embeddings and the first policy-evidence read."""

        binding = stage0.encoder.binding
        projector = stage0.encoder.observer.projector
        self.owner_embedding.copy_(binding.owner_embedding)
        self.family_embedding.weight.copy_(projector.family_embedding.weight)
        self.horizon_embedding.weight.copy_(binding.horizon_embedding)
        first = self.blocks[0].policy_attention
        width = self.width
        first.in_proj_weight[:width].copy_(binding.event_query.weight)
        first.in_proj_weight[width : 2 * width].copy_(binding.policy_key.weight)
        first.in_proj_weight[2 * width :].copy_(binding.policy_value.weight)
        first.in_proj_bias.zero_()
        first.out_proj.weight.copy_(
            torch.eye(width, device=first.out_proj.weight.device)
        )
        first.out_proj.bias.zero_()
        return {
            "reused": [
                "owner_family_horizon_embeddings",
                "first_parallel_policy_evidence_attention",
            ]
        }
