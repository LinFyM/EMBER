"""Frame-local bank-conditioned native-factor composition.

Each copyable block lets every real-frame rank token read that frame's complete
native X/Y bank, then the video's ordered policy-response events. The resulting
dynamic frame states score the same untouched X/Y values once through exact
signed pooling. There is no global bank summary or analytic transform chain;
the only post-pooling operation is the established target update cap.
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
    PolicyResponseProcessOutput,
)


@dataclass(frozen=True)
class _NativeBankChunk:
    """Position-free bank tokens and untouched native values for a frame chunk."""

    input_values: torch.Tensor
    output_values: torch.Tensor
    context_tokens: torch.Tensor
    input_candidate_count: int
    frame_positions: torch.Tensor


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


class FrameBankFactorBlock(torch.nn.Module):
    """One repeatable frame-bank, ordered-event, rank, and MLP block."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.bank_query_norm = torch.nn.LayerNorm(width)
        self.bank_memory_norm = torch.nn.LayerNorm(width)
        self.bank_attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, bias=False, batch_first=True
        )
        self.event_query_norm = torch.nn.LayerNorm(width)
        self.temporal_position = torch.nn.Linear(2, width, bias=False)
        self.event_attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, bias=False, batch_first=True
        )
        self.rank_attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, bias=False, batch_first=True
        )
        # Every value projection on the dynamic path is bias-free. Structural
        # and positional content enters queries/keys only, so zero dynamic
        # evidence remains zero through an arbitrarily deep stack.
        self.mlp = GatedMLP(width, bias=False)

    def _position(self, value: torch.Tensor) -> torch.Tensor:
        position = value.clamp(0.0, 1.0)
        return self.temporal_position(
            torch.stack((position, position.square()), dim=-1)
        )

    def _bank_read(
        self, query: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        normalized_memory = self.bank_memory_norm(memory)
        attended, _ = self.bank_attention(
            self.bank_query_norm(query),
            normalized_memory,
            normalized_memory,
            need_weights=False,
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
        structural_query: torch.Tensor,
        video_events: Sequence[torch.Tensor],
        frame_states: Sequence[torch.Tensor],
        frame_positions: Sequence[torch.Tensor],
        bank_chunks: Sequence[Sequence[torch.Tensor]],
    ) -> tuple[torch.Tensor, ...]:
        events = tuple(video_events)
        frames = tuple(frame_states)
        positions = tuple(frame_positions)
        banks = tuple(tuple(chunks) for chunks in bank_chunks)
        if structural_query.ndim != 2:
            raise ValueError("frame-bank structural query axes changed")
        ranks, width = structural_query.shape
        if (
            not events
            or len(events) != len(frames)
            or len(events) != len(positions)
            or len(events) != len(banks)
            or any(
                event.ndim != 3
                or event.shape[0] != ranks
                or event.shape[-1] != width
                or not event.shape[1]
                for event in events
            )
            or any(
                frame.ndim != 3
                or frame.shape[1:] != (ranks, width)
                or position.shape != frame.shape[:1]
                or not chunks
                or any(
                    chunk.ndim != 3
                    or chunk.shape[-1] != width
                    or not chunk.shape[0]
                    or not chunk.shape[1]
                    for chunk in chunks
                )
                or sum(chunk.shape[0] for chunk in chunks) != frame.shape[0]
                for frame, position, chunks in zip(
                    frames, positions, banks, strict=True
                )
            )
        ):
            raise ValueError("frame-bank factor block axes changed")

        output = []
        for frame, event, position, chunks in zip(
            frames, events, positions, banks, strict=True
        ):
            reads = []
            offset = 0
            for memory in chunks:
                count = memory.shape[0]
                query = frame[offset : offset + count] + structural_query[None]
                reads.append(self._checkpointed_bank_read(query, memory))
                offset += count
            bank_read = torch.cat(tuple(reads), dim=0)
            # A bank property that is constant through the video is not motion.
            # Centering within each independently encoded video makes this a
            # structural invariant rather than a learned anti-static loss.
            value = frame + bank_read - bank_read.mean(0, keepdim=True)

            frame_rows = value.permute(1, 0, 2)
            frame_query = self.event_query_norm(
                frame_rows + structural_query[:, None]
            )
            frame_query = frame_query + self._position(position.to(frame))[None]
            event_position = torch.linspace(
                0.0,
                1.0,
                event.shape[1],
                device=event.device,
                dtype=event.dtype,
            )
            event_key = event + self._position(event_position)[None]
            attended, _ = self.event_attention(
                frame_query, event_key, event, need_weights=False
            )
            value = value + attended.permute(1, 0, 2)

            rank_query_key = value + structural_query[None]
            attended, _ = self.rank_attention(
                rank_query_key, rank_query_key, value, need_weights=False
            )
            output.append(self.mlp(value + attended))
        return tuple(output)


class CurrentVideoNativeFactorComposer(torch.nn.Module):
    """Map event and bank tokens directly to current-video rank-four X/Y factors."""

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
            raise ValueError("native composer topology changed")
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
        self.rank_queries = torch.nn.Parameter(torch.empty(G1_RESIDUAL_RANK, width))
        self.owner_embedding = torch.nn.Parameter(torch.empty(len(owners), width))
        self.family_embedding = torch.nn.Embedding(len(TargetFamily), width)
        self.probe_embedding = torch.nn.Embedding(2, width)
        self.horizon_embedding = torch.nn.Embedding(ACTION_HORIZON, width)
        self.bank_type_embedding = torch.nn.Embedding(5, width)
        self.group_embedding = torch.nn.Embedding(maximum_groups, width)
        self.position_projection = torch.nn.Linear(2, width, bias=False)
        self.native_key_norm = torch.nn.LayerNorm(width)
        self.query_seed = torch.nn.Sequential(
            torch.nn.Linear(4 * width, width),
            torch.nn.LayerNorm(width),
        )
        self.blocks = torch.nn.ModuleList(
            FrameBankFactorBlock(width, heads) for _ in range(block_depth)
        )
        self.input_base_query = torch.nn.Linear(width, width, bias=False)
        self.input_contrast_query = torch.nn.Linear(width, width, bias=False)
        self.output_base_query = torch.nn.Linear(width, width, bias=False)
        # No bias: if event content is zero the two Y distributions coincide.
        self.output_contrast_query = torch.nn.Linear(width, width, bias=False)
        self.task_query = (
            torch.nn.Parameter(torch.zeros(len(owners), G1_RESIDUAL_RANK, width))
            if task_local
            else None
        )
        family_order = tuple(TargetFamily)
        self.register_buffer(
            "family_ids",
            torch.tensor([family_order.index(owner.family) for owner in owners]),
            persistent=False,
        )
        torch.nn.init.normal_(self.rank_queries, std=width**-0.5)
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

    def _input_context(
        self,
        target: int,
        values: torch.Tensor,
    ) -> torch.Tensor:
        owner = self.owners[target]
        key = self.input_projection[str(owner.in_features)](values)
        key = key + self._owner_bias(target)
        key = key + self.probe_embedding.weight[None, :, None]
        key = key + self.horizon_embedding.weight[None, None]
        key = key + self.bank_type_embedding.weight[0]
        return key

    def _output_context(
        self,
        target: int,
        values: torch.Tensor,
    ) -> torch.Tensor:
        # values: [group, frame, probe, horizon, bank_type, native_width]
        key = self.output_projection[str(values.shape[-1])](values)
        key = key + self._owner_bias(target)
        key = key + self.probe_embedding.weight[None, None, :, None, None]
        key = key + self.horizon_embedding.weight[None, None, None, :, None]
        key = key + self.bank_type_embedding.weight[1:][None, None, None, None]
        key = key + self.group_embedding.weight[
            : values.shape[0], None, None, None, None
        ]
        return key

    def _pooling_keys(
        self, chunk: _NativeBankChunk
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frames = chunk.input_values.shape[0]
        input_context = chunk.context_tokens[
            :, : chunk.input_candidate_count
        ].reshape(*chunk.input_values.shape[:-1], self.width)
        output_context = chunk.context_tokens[
            :, chunk.input_candidate_count :
        ].reshape(
            frames,
            chunk.output_values.shape[0],
            *chunk.output_values.shape[2:-1],
            self.width,
        ).movedim(1, 0)
        input_keys = self.native_key_norm(
            input_context + chunk.frame_positions[:, None, None]
        )
        output_keys = self.native_key_norm(
            output_context + chunk.frame_positions[None, :, None, None, None]
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
                local_position = positions[start:stop]
                input_values = video.native_inputs[target][start:stop]
                input_context = self._input_context(target, input_values)
                output = boundary.build(
                    video.native_outputs[target][start:stop], start_frame=start
                )
                grouped = output.reshape(
                    *output.shape[:-1], groups, group_width
                ).movedim(-2, 0)
                output_context = self._output_context(target, grouped)
                input_count = input_context[0].numel() // self.width
                context_tokens = torch.cat(
                    (
                        input_context.flatten(1, -2),
                        output_context.movedim(0, 1).flatten(1, -2),
                    ),
                    dim=1,
                )
                chunks.append(
                    _NativeBankChunk(
                        input_values=input_values,
                        output_values=grouped,
                        context_tokens=context_tokens,
                        input_candidate_count=input_count,
                        frame_positions=local_position,
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

    def _video_events(
        self,
        target: int,
        processes: Sequence[PolicyResponseProcessOutput],
    ) -> tuple[torch.Tensor, ...]:
        rows = []
        for process in processes:
            event = process.events[:, target]
            if event.ndim != 3 or event.shape[1:] != (
                G1_RESIDUAL_RANK,
                self.width,
            ):
                raise ValueError("native composer event tokens changed")
            rows.append(event.permute(1, 0, 2))
        return tuple(rows)

    def _decode_target(
        self,
        target: int,
        videos: Sequence[FrozenPolicyResponseVideo],
        processes: Sequence[PolicyResponseProcessOutput],
        candidates: Sequence[_NativeVideoCandidates],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if len(videos) != len(processes) or len(videos) != len(candidates):
            raise ValueError("native composer video/process count changed")
        for video, process in zip(videos, processes, strict=True):
            if (
                process.frame_tokens.shape[:3]
                != (video.frame_count, len(self.owners), G1_RESIDUAL_RANK)
                or process.frame_innovations.shape
                != (
                    video.frame_count,
                    len(self.owners),
                    G1_RESIDUAL_RANK,
                    self.width,
                )
            ):
                raise ValueError("native composer process/video axes changed")
        language = torch.stack(
            tuple(process.owner_language[target] for process in processes)
        ).mean(0)
        rank = self.rank_queries
        if self.task_query is not None:
            rank = rank + self.task_query[target]
        ranks = rank.shape[0]
        owner = self.owner_embedding[target].expand(ranks, -1)
        family = self.family_embedding(self.family_ids[target]).expand(ranks, -1)
        language = language.expand(ranks, -1)
        query = self.query_seed(torch.cat((rank, owner, family, language), dim=-1))
        events = self._video_events(target, processes)
        frame_states = tuple(
            process.frame_innovations[:, target] for process in processes
        )
        positions = tuple(video.frame_positions for video in videos)
        bank_chunks = tuple(
            tuple(chunk.context_tokens for chunk in video.chunks)
            for video in candidates
        )
        for block in self.blocks:
            frame_states = block(
                query, events, frame_states, positions, bank_chunks
            )
        return query, frame_states

    def _branch_logits(
        self,
        keys: torch.Tensor,
        base_query: torch.Tensor,
        contrast_query: torch.Tensor,
        *,
        log_base_mass: float,
    ) -> torch.Tensor:
        if (
            keys.ndim < 2
            or keys.shape[-1] != self.width
            or base_query.shape != (G1_RESIDUAL_RANK, self.width)
            or contrast_query.shape
            != (keys.shape[0], G1_RESIDUAL_RANK, self.width)
        ):
            raise ValueError("signed native-attention axes changed")
        scale = math.sqrt(self.width)
        base = torch.einsum("rd,f...d->rf...", base_query, keys) / scale
        contrast = (
            torch.einsum("frd,f...d->rf...", contrast_query, keys) / scale
        )
        return (
            torch.stack((base + contrast, base - contrast), dim=1).float()
            + log_base_mass
        )

    def _checkpointed_branch_logits(
        self,
        keys: torch.Tensor,
        base_query: torch.Tensor,
        contrast_query: torch.Tensor,
        *,
        log_base_mass: float,
    ) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return self._branch_logits(
                keys,
                base_query,
                contrast_query,
                log_base_mass=log_base_mass,
            )

        def score(
            local_keys: torch.Tensor,
            local_base: torch.Tensor,
            local_contrast: torch.Tensor,
        ) -> torch.Tensor:
            return self._branch_logits(
                local_keys,
                local_base,
                local_contrast,
                log_base_mass=log_base_mass,
            )

        return checkpoint(
            score, keys, base_query, contrast_query, use_reentrant=False
        )

    def _pool_target(
        self,
        target: int,
        query: torch.Tensor,
        frame_states: Sequence[torch.Tensor],
        videos: Sequence[_NativeVideoCandidates],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        owner = self.owners[target]
        groups = native_output_group_count(owner)
        group_width = owner.out_features // groups
        input_accumulator = OnlineSoftmaxAccumulator(
            ranks=G1_RESIDUAL_RANK,
            width=owner.in_features,
            device=query.device,
        )
        output_accumulator = _GroupedOnlineSoftmaxAccumulator(
            groups=groups,
            ranks=G1_RESIDUAL_RANK,
            width=group_width,
            device=query.device,
        )
        input_base = self.input_base_query(query)
        output_base = self.output_base_query(query)
        video_count = len(videos)
        if len(frame_states) != video_count:
            raise ValueError("native composer frame-state set changed")
        for video, frame_state in zip(videos, frame_states, strict=True):
            if frame_state.shape != (
                video.frame_count,
                G1_RESIDUAL_RANK,
                self.width,
            ):
                raise ValueError("native composer aligned frame state changed")
            input_contrast = self.input_contrast_query(frame_state)
            output_contrast = self.output_contrast_query(frame_state)
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
                        input_base,
                        input_contrast[frame_slice],
                        log_base_mass=input_mass,
                    ),
                    chunk.input_values,
                )
                output_accumulator.add(
                    self._checkpointed_branch_logits(
                        output_keys.movedim(0, 1),
                        output_base,
                        output_contrast[frame_slice],
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
        processes: Sequence[PolicyResponseProcessOutput],
        *,
        s_ref: torch.Tensor,
    ) -> NativeFactorResidual:
        values = tuple(videos)
        programs = tuple(processes)
        if (
            not values
            or len(values) != len(programs)
            or s_ref.shape != (len(self.owners),)
        ):
            raise ValueError("native composer video set or scale authority changed")
        a_values = []
        b_values = []
        scales = []
        for target in range(len(self.owners)):
            candidates = self._bank_candidates(target, values)
            query, frame_states = self._decode_target(
                target, values, programs, candidates
            )
            a, b = self._pool_target(target, query, frame_states, candidates)
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
