"""Native-bank-conditioned temporal composition of rank-four LoRA factors.

Each repeatable block gives explicit X/Y factor-side states access to the same
frame's complete native bank, then models the ordered teacher-frame sequence.
The final states score the untouched X/Y values directly through exact signed
pooling.  No event summary, global video code, gain, or calibration path sits
between the frozen PI0.5 evidence and the factors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
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
    GatedMLP,
    PolicyResponseFrameOutput,
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


class NativeTemporalFactorBlock(torch.nn.Module):
    """One copyable side-native read, frame-time, rank-side transformer block."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.bank_query_norm = torch.nn.LayerNorm(width)
        self.bank_memory_norm = torch.nn.LayerNorm(width)
        self.bank_attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.temporal_norm = torch.nn.LayerNorm(width)
        self.temporal_position = torch.nn.Linear(2, width, bias=False)
        self.temporal_attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.factor_norm = torch.nn.LayerNorm(width)
        self.factor_attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.mlp = GatedMLP(width)

    def _position(self, value: torch.Tensor) -> torch.Tensor:
        position = value.clamp(0.0, 1.0)
        return self.temporal_position(
            torch.stack((position, position.square()), dim=-1)
        )

    def _bank_read(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        memory = self.bank_memory_norm(memory)
        attended, _ = self.bank_attention(
            self.bank_query_norm(query), memory, memory, need_weights=False
        )
        return attended

    def _checkpointed_bank_read(
        self, query: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return self._bank_read(query, memory)
        return checkpoint(self._bank_read, query, memory, use_reentrant=False)

    def forward(
        self,
        frame_states: Sequence[torch.Tensor],
        frame_positions: Sequence[torch.Tensor],
        input_bank_chunks: Sequence[Sequence[torch.Tensor]],
        output_bank_chunks: Sequence[Sequence[torch.Tensor]],
    ) -> tuple[torch.Tensor, ...]:
        frames = tuple(frame_states)
        positions = tuple(frame_positions)
        input_banks = tuple(tuple(chunks) for chunks in input_bank_chunks)
        output_banks = tuple(tuple(chunks) for chunks in output_bank_chunks)
        if (
            not frames
            or len(frames) != len(positions)
            or len(frames) != len(input_banks)
            or len(frames) != len(output_banks)
        ):
            raise ValueError("native-temporal video set changed")

        output = []
        for frame, position, input_chunks, output_chunks in zip(
            frames, positions, input_banks, output_banks, strict=True
        ):
            if frame.ndim != 4 or frame.shape[2] != 2:
                raise ValueError("native-temporal factor-side axes changed")
            frame_count, ranks, _, width = frame.shape
            if (
                position.shape != (frame_count,)
                or not input_chunks
                or len(input_chunks) != len(output_chunks)
                or any(
                    left.ndim != 3
                    or right.ndim != 3
                    or left.shape[0] != right.shape[0]
                    or left.shape[-1] != width
                    or right.shape[-1] != width
                    or not left.shape[0]
                    or not left.shape[1]
                    or not right.shape[1]
                    for left, right in zip(
                        input_chunks, output_chunks, strict=True
                    )
                )
                or sum(chunk.shape[0] for chunk in input_chunks) != frame_count
            ):
                raise ValueError("native-temporal bank axes changed")

            reads = []
            offset = 0
            for input_memory, output_memory in zip(
                input_chunks, output_chunks, strict=True
            ):
                count = input_memory.shape[0]
                local = frame[offset : offset + count]
                x_read = self._checkpointed_bank_read(
                    local[:, :, 0], input_memory
                )
                y_read = self._checkpointed_bank_read(
                    local[:, :, 1], output_memory
                )
                reads.append(torch.stack((x_read, y_read), dim=2))
                offset += count
            value = frame + torch.cat(tuple(reads), dim=0)

            temporal = value.permute(1, 2, 0, 3).reshape(
                ranks * 2, frame_count, width
            )
            normalized = self.temporal_norm(temporal)
            query_key = normalized + self._position(position.to(frame))[None]
            attended, _ = self.temporal_attention(
                query_key, query_key, normalized, need_weights=False
            )
            value = (temporal + attended).reshape(
                ranks, 2, frame_count, width
            ).permute(2, 0, 1, 3)

            factors = value.reshape(frame_count, ranks * 2, width)
            normalized = self.factor_norm(factors)
            attended, _ = self.factor_attention(
                normalized, normalized, normalized, need_weights=False
            )
            output.append(
                self.mlp(factors + attended).reshape(frame_count, ranks, 2, width)
            )
        return tuple(output)


class NativeTemporalFactorComposer(torch.nn.Module):
    """Map frame responses and current-video native banks to rank-four X/Y."""

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
            raise ValueError("native-temporal composer topology changed")
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
            NativeTemporalFactorBlock(width, heads) for _ in range(block_depth)
        )
        self.input_signed_query = torch.nn.Linear(width, 2 * width, bias=False)
        self.output_signed_query = torch.nn.Linear(width, 2 * width, bias=False)
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
        torch.nn.init.normal_(self.factor_side_embedding, std=width**-0.5)
        torch.nn.init.normal_(self.owner_embedding, std=width**-0.5)

    def _owner_bias(self, target: int) -> torch.Tensor:
        return self.owner_embedding[target] + self.family_embedding(
            self.family_ids[target]
        )

    def _position(self, video: FrozenPolicyResponseVideo) -> torch.Tensor:
        position = video.frame_positions.float()
        if position.shape != (video.frame_count,):
            raise ValueError("native composer frame positions changed")
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
                raise ValueError("native composer bank stream ended early")
            candidates.append(
                _NativeVideoCandidates(
                    frame_count=video.frame_count,
                    chunks=tuple(chunks),
                )
            )
        if not candidates:
            raise ValueError("native composer received no current-video bank")
        return tuple(candidates)

    def _decode_target(
        self,
        target: int,
        videos: Sequence[FrozenPolicyResponseVideo],
        frames: Sequence[PolicyResponseFrameOutput],
        candidates: Sequence[_NativeVideoCandidates],
    ) -> tuple[torch.Tensor, ...]:
        if len(videos) != len(frames) or len(videos) != len(candidates):
            raise ValueError("native composer video/frame count changed")
        states = []
        for video, encoded in zip(videos, frames, strict=True):
            if encoded.frame_tokens.shape != (
                video.frame_count,
                len(self.owners),
                G1_RESIDUAL_RANK,
                self.width,
            ):
                raise ValueError("native composer encoded-frame axes changed")
            state = (
                encoded.frame_tokens[:, target, :, None]
                + self.factor_side_embedding[None, None]
            )
            if self.task_query is not None:
                state = state + self.task_query[target][None]
            states.append(state)

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
            values = block(values, positions, input_chunks, output_chunks)
        return tuple(value - value.mean(0, keepdim=True) for value in values)

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
        self, projection: torch.nn.Linear, state: torch.Tensor
    ) -> torch.Tensor:
        frames = state.shape[0]
        return projection(state).reshape(
            frames, G1_RESIDUAL_RANK, 2, self.width
        )

    def _pool_target(
        self,
        target: int,
        frame_states: Sequence[torch.Tensor],
        videos: Sequence[_NativeVideoCandidates],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        owner = self.owners[target]
        groups = native_output_group_count(owner)
        group_width = owner.out_features // groups
        device = frame_states[0].device
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
            raise ValueError("native composer frame-state set changed")
        for video, frame_state in zip(videos, frame_states, strict=True):
            if frame_state.shape != (
                video.frame_count,
                G1_RESIDUAL_RANK,
                2,
                self.width,
            ):
                raise ValueError("native composer factor-side frame state changed")
            input_queries = self._signed_queries(
                self.input_signed_query, frame_state[:, :, 0]
            )
            output_queries = self._signed_queries(
                self.output_signed_query, frame_state[:, :, 1]
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
                raise ValueError("native composer aligned frame stream ended early")
        return input_accumulator.signed_mean(), output_accumulator.signed_mean()

    def forward(
        self,
        videos: Sequence[FrozenPolicyResponseVideo],
        frames: Sequence[PolicyResponseFrameOutput],
        *,
        s_ref: torch.Tensor,
    ) -> NativeFactorResidual:
        values = tuple(videos)
        encoded = tuple(frames)
        if (
            not values
            or len(values) != len(encoded)
            or s_ref.shape != (len(self.owners),)
        ):
            raise ValueError("native composer video set or scale authority changed")
        a_values = []
        b_values = []
        scales = []
        for target in range(len(self.owners)):
            candidates = self._bank_candidates(target, values)
            frame_states = self._decode_target(
                target, values, encoded, candidates
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
