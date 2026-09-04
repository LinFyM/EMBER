"""Frame-aligned event-conditioned native-factor composition.

A short stack of identical learned blocks aligns ordered policy-response events
back to their real teacher frames. The resulting frame states score the complete
current-video native bank once, through exact signed pooling of untouched X/Y
values. There is no preliminary bank read or chain of analytic transforms; the
only post-pooling operation is the established target update cap.
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
    """Projected keys and untouched native values for one frame chunk."""

    input_values: torch.Tensor
    input_keys: torch.Tensor
    output_values: torch.Tensor
    output_keys: torch.Tensor


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


class FrameAlignedFactorBlock(torch.nn.Module):
    """One repeatable event, rank, and frame-alignment Transformer block."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.query_norm = torch.nn.LayerNorm(width)
        self.event_attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, bias=False, batch_first=True
        )
        self.rank_norm = torch.nn.LayerNorm(width)
        self.rank_attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.query_mlp = GatedMLP(width)
        self.temporal_position = torch.nn.Linear(2, width, bias=False)
        # Values on the dynamic path are bias-free. A static repeated video
        # therefore cannot manufacture a non-zero factor contrast from position
        # or from the static language/owner query.
        self.frame_event_attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, bias=False, batch_first=True
        )
        self.frame_mlp = GatedMLP(width, bias=False)

    def _position(self, value: torch.Tensor) -> torch.Tensor:
        position = value.clamp(0.0, 1.0)
        return self.temporal_position(
            torch.stack((position, position.square()), dim=-1)
        )

    def forward(
        self,
        query: torch.Tensor,
        video_events: Sequence[torch.Tensor],
        frame_states: Sequence[torch.Tensor],
        frame_positions: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        events = tuple(video_events)
        frames = tuple(frame_states)
        positions = tuple(frame_positions)
        if query.ndim != 2:
            raise ValueError("frame-aligned factor query axes changed")
        ranks, width = query.shape
        if (
            not events
            or len(events) != len(frames)
            or len(events) != len(positions)
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
                for frame, position in zip(frames, positions, strict=True)
            )
        ):
            raise ValueError("frame-aligned factor block axes changed")

        event_memory = torch.cat(events, dim=1)
        event_query = self.query_norm(query)[:, None]
        attended, _ = self.event_attention(
            event_query, event_memory, event_memory, need_weights=False
        )
        value = query + attended[:, 0]
        normalized = self.rank_norm(value)[None]
        attended, _ = self.rank_attention(
            normalized, normalized, normalized, need_weights=False
        )
        value = self.query_mlp(value + attended[0])

        aligned = []
        for frame, event, position in zip(
            frames, events, positions, strict=True
        ):
            frame_rows = frame.permute(1, 0, 2)
            frame_query = (
                frame_rows
                + value[:, None]
                + self._position(position.to(frame))
            )
            event_position = torch.linspace(
                0.0,
                1.0,
                event.shape[1],
                device=event.device,
                dtype=event.dtype,
            )
            event_key = event + self._position(event_position)[None]
            attended, _ = self.frame_event_attention(
                frame_query, event_key, event, need_weights=False
            )
            aligned.append(
                self.frame_mlp(frame + attended.permute(1, 0, 2))
            )
        return value, tuple(aligned)


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
            FrameAlignedFactorBlock(width, heads) for _ in range(block_depth)
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

    def _input_keys(
        self,
        target: int,
        values: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        owner = self.owners[target]
        key = self.input_projection[str(owner.in_features)](values)
        key = key + self._owner_bias(target)
        key = key + self.probe_embedding.weight[None, :, None]
        key = key + self.horizon_embedding.weight[None, None]
        key = key + self.bank_type_embedding.weight[0]
        return self.native_key_norm(key + positions[:, None, None])

    def _output_keys(
        self,
        target: int,
        values: torch.Tensor,
        positions: torch.Tensor,
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
        return self.native_key_norm(
            key + positions[None, :, None, None, None]
        )

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
                input_keys = self._input_keys(target, input_values, local_position)
                output = boundary.build(
                    video.native_outputs[target][start:stop], start_frame=start
                )
                grouped = output.reshape(
                    *output.shape[:-1], groups, group_width
                ).movedim(-2, 0)
                output_keys = self._output_keys(target, grouped, local_position)
                chunks.append(
                    _NativeBankChunk(
                        input_values=input_values,
                        input_keys=input_keys,
                        output_values=grouped,
                        output_keys=output_keys,
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
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if len(videos) != len(processes):
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
        for block in self.blocks:
            query, frame_states = block(query, events, frame_states, positions)
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
                input_accumulator.add(
                    self._checkpointed_branch_logits(
                        chunk.input_keys,
                        input_base,
                        input_contrast[frame_slice],
                        log_base_mass=input_mass,
                    ),
                    chunk.input_values,
                )
                output_accumulator.add(
                    self._checkpointed_branch_logits(
                        chunk.output_keys.movedim(0, 1),
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
            query, frame_states = self._decode_target(target, values, programs)
            candidates = self._bank_candidates(target, values)
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
