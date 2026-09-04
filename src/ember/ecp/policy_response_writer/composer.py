"""Direct event-conditioned native-factor composition.

The composer deliberately has no relation marginal, analytic transport,
factor normalization, or separate gain network. A short stack of identical
blocks reads dynamic event tokens and the complete current-video native bank.
Two signed-attention heads then pool raw X and Y values directly into rank-four
factors. The only post-pooling operation is the established target update cap.
"""

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
    GatedMLP,
    PolicyResponseProcessOutput,
)


DENSE_BANK_ATTENTION_TOKEN_LIMIT = 192 * 1024
STREAMING_BANK_BLOCK_TOKEN_LIMIT = 128 * 1024


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


class RankBankContextBlock(torch.nn.Module):
    """One repeatable event read, native-bank read, rank attention, and MLP."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.query_norm = torch.nn.LayerNorm(width)
        # Dynamic values must remain exactly zero for a static repeated video.
        self.event_norm = torch.nn.Identity()
        self.event_attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, bias=False, batch_first=True
        )
        self.bank_norm = torch.nn.LayerNorm(width)
        self.bank_attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, bias=False, batch_first=True
        )
        self.rank_norm = torch.nn.LayerNorm(width)
        self.rank_attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.mlp = GatedMLP(width)

    def forward(
        self,
        query: torch.Tensor,
        event_memory: torch.Tensor,
        bank_memory: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            query.ndim != 2
            or event_memory.ndim != 3
            or event_memory.shape[0] != query.shape[0]
            or event_memory.shape[-1] != query.shape[-1]
            or not bank_memory
        ):
            raise ValueError("rank-bank context axes changed")
        event_query = self.query_norm(query)[:, None]
        event = self.event_norm(event_memory)
        attended, _ = self.event_attention(
            event_query, event, event, need_weights=False
        )
        event_delta = attended[:, 0]
        value = query + event_delta

        bank_query = self.query_norm(value)[None]
        bank = tuple(bank_memory)
        if torch.is_grad_enabled() and any(chunk.requires_grad for chunk in bank):

            def read(
                local_query: torch.Tensor, *local_bank: torch.Tensor
            ) -> torch.Tensor:
                return self._exact_bank_attention(local_query, local_bank)

            attended = checkpoint(read, bank_query, *bank, use_reentrant=False)
        else:
            attended = self._exact_bank_attention(bank_query, bank)
        value = value + attended[0]

        normalized = self.rank_norm(value)[None]
        attended, _ = self.rank_attention(
            normalized, normalized, normalized, need_weights=False
        )
        return self.mlp(value + attended[0]), event_delta

    def _exact_bank_attention(
        self,
        query: torch.Tensor,
        memory: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Read each video bank exactly, then aggregate the video set equally."""

        reads = tuple(self._single_bank_attention(query, value) for value in memory)
        if not reads:
            raise ValueError("native-bank attention received no videos")
        return torch.stack(reads).mean(0)

    def _single_bank_attention(
        self, query: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        if memory.shape[0] <= DENSE_BANK_ATTENTION_TOKEN_LIMIT:
            return self._dense_bank_attention(query, memory)
        return self._streaming_bank_attention(query, memory)

    def _dense_bank_attention(
        self, query: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """Exact fused SDPA over every token in one video's native bank."""

        if (
            query.ndim != 3
            or query.shape[0] != 1
            or memory.ndim != 2
            or memory.shape[-1] != query.shape[-1]
            or not memory.shape[0]
        ):
            raise ValueError("dense native-bank attention inputs changed")
        width = query.shape[-1]
        heads = self.bank_attention.num_heads
        head_width = width // heads
        weight = self.bank_attention.in_proj_weight
        bias = self.bank_attention.in_proj_bias
        q_bias = None if bias is None else bias[:width]
        k_bias = None if bias is None else bias[width : 2 * width]
        v_bias = None if bias is None else bias[2 * width :]
        projected_query = F.linear(query[0], weight[:width], q_bias)
        normalized = self.bank_norm(memory)
        key = F.linear(normalized, weight[width : 2 * width], k_bias)
        value = F.linear(normalized, weight[2 * width :], v_bias)
        projected_query = projected_query.reshape(-1, heads, head_width).permute(
            1, 0, 2
        )
        key = key.reshape(-1, heads, head_width).permute(1, 0, 2)
        value = value.reshape(-1, heads, head_width).permute(1, 0, 2)
        attended = F.scaled_dot_product_attention(
            projected_query[None],
            key[None],
            value[None],
            dropout_p=0.0,
            is_causal=False,
        )[0]
        attended = attended.permute(1, 0, 2).reshape(1, query.shape[1], width)
        return self.bank_attention.out_proj(attended)

    def _streaming_bank_attention(
        self, query: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """The same attention reduced exactly in bounded token blocks."""

        if (
            query.ndim != 3
            or query.shape[0] != 1
            or memory.ndim != 2
            or memory.shape[-1] != query.shape[-1]
            or not memory.shape[0]
        ):
            raise ValueError("streaming native-bank attention inputs changed")
        width = query.shape[-1]
        heads = self.bank_attention.num_heads
        head_width = width // heads
        weight = self.bank_attention.in_proj_weight
        bias = self.bank_attention.in_proj_bias
        q_bias = None if bias is None else bias[:width]
        k_bias = None if bias is None else bias[width : 2 * width]
        v_bias = None if bias is None else bias[2 * width :]
        projected_query = F.linear(query[0], weight[:width], q_bias)
        projected_query = projected_query.reshape(-1, heads, head_width).permute(
            1, 0, 2
        )
        maximum = None
        denominator = None
        numerator = None
        for start in range(0, memory.shape[0], STREAMING_BANK_BLOCK_TOKEN_LIMIT):
            chunk = self.bank_norm(
                memory[start : start + STREAMING_BANK_BLOCK_TOKEN_LIMIT]
            )
            key = F.linear(chunk, weight[width : 2 * width], k_bias)
            value = F.linear(chunk, weight[2 * width :], v_bias)
            key = key.reshape(-1, heads, head_width).permute(1, 0, 2)
            value = value.reshape(-1, heads, head_width).permute(1, 0, 2)
            score = torch.einsum(
                "hqd,hnd->hqn", projected_query, key
            ).float() / math.sqrt(head_width)
            local_maximum = score.amax(-1)
            if maximum is None:
                maximum = local_maximum
                mass = torch.exp(score - maximum[..., None])
                denominator = mass.sum(-1)
                numerator = torch.einsum("hqn,hnd->hqd", mass, value.float())
                continue
            updated_maximum = torch.maximum(maximum, local_maximum)
            previous_scale = torch.exp(maximum - updated_maximum)
            mass = torch.exp(score - updated_maximum[..., None])
            denominator = denominator * previous_scale + mass.sum(-1)
            numerator = numerator * previous_scale[..., None] + torch.einsum(
                "hqn,hnd->hqd", mass, value.float()
            )
            maximum = updated_maximum
        if denominator is None or numerator is None:
            raise ValueError("streaming native-bank attention received no tokens")
        attended = (numerator / denominator[..., None]).permute(1, 0, 2)
        attended = attended.reshape(1, query.shape[1], width).to(query.dtype)
        return self.bank_attention.out_proj(attended)


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
            RankBankContextBlock(width, heads) for _ in range(block_depth)
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
        processes: Sequence[PolicyResponseProcessOutput],
    ) -> tuple[tuple[torch.Tensor, ...], tuple[_NativeVideoCandidates, ...]]:
        memories = []
        candidates = []
        owner = self.owners[target]
        groups = native_output_group_count(owner)
        group_width = owner.out_features // groups
        for video, process in zip(videos, processes, strict=True):
            if (
                process.frame_tokens.shape[:3]
                != (video.frame_count, len(self.owners), G1_RESIDUAL_RANK)
                or process.events.ndim != 4
                or process.events.shape[1:3]
                != (len(self.owners), G1_RESIDUAL_RANK)
                or process.events.shape[-1] != self.width
            ):
                raise ValueError("native composer process/video axes changed")
            positions = self._position(video)
            boundary = NativeOutputBankState(final=video.final_outputs[target])
            chunks = []
            memory = []
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
                memory.extend(
                    (
                        input_keys.reshape(-1, self.width),
                        output_keys.reshape(-1, self.width),
                    )
                )
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
            memories.append(torch.cat(tuple(memory), dim=0))
            candidates.append(
                _NativeVideoCandidates(
                    frame_count=video.frame_count,
                    chunks=tuple(chunks),
                )
            )
        if not memories:
            raise ValueError("native composer received no current-video bank")
        return tuple(memories), tuple(candidates)

    def _event_tokens(
        self,
        target: int,
        processes: Sequence[PolicyResponseProcessOutput],
    ) -> torch.Tensor:
        rows = []
        for process in processes:
            event = process.events[:, target]
            if event.ndim != 3 or event.shape[1:] != (
                G1_RESIDUAL_RANK,
                self.width,
            ):
                raise ValueError("native composer event tokens changed")
            rows.append(event.permute(1, 0, 2))
        return torch.cat(tuple(rows), dim=1)

    def _query_target(
        self,
        target: int,
        processes: Sequence[PolicyResponseProcessOutput],
        bank_memory: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        event_memory = self._event_tokens(target, processes)
        dynamic = torch.zeros_like(query)
        for block in self.blocks:
            query, event_delta = block(query, event_memory, bank_memory)
            dynamic = dynamic + event_delta
        return query, dynamic

    def _branch_logits(
        self,
        keys: torch.Tensor,
        base_query: torch.Tensor,
        contrast_query: torch.Tensor,
        *,
        log_base_mass: float,
    ) -> torch.Tensor:
        if keys.shape[-1] != self.width or base_query.shape != contrast_query.shape:
            raise ValueError("signed native-attention axes changed")
        scale = math.sqrt(self.width)
        base = torch.einsum("rd,...d->r...", base_query, keys) / scale
        contrast = torch.einsum("rd,...d->r...", contrast_query, keys) / scale
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
        dynamic: torch.Tensor,
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
        input_contrast = self.input_contrast_query(query)
        output_base = self.output_base_query(query)
        output_contrast = self.output_contrast_query(dynamic)
        video_count = len(videos)
        for video in videos:
            input_mass = -math.log(
                video_count * video.frame_count * 2 * ACTION_HORIZON
            )
            output_mass = -math.log(
                video_count * video.frame_count * 2 * ACTION_HORIZON * 4
            )
            input_keys = torch.cat(
                tuple(chunk.input_keys for chunk in video.chunks), dim=0
            )
            input_values = torch.cat(
                tuple(chunk.input_values for chunk in video.chunks), dim=0
            )
            input_accumulator.add(
                self._checkpointed_branch_logits(
                    input_keys,
                    input_base,
                    input_contrast,
                    log_base_mass=input_mass,
                ),
                input_values,
            )
            output_keys = torch.cat(
                tuple(chunk.output_keys for chunk in video.chunks), dim=1
            )
            output_values = torch.cat(
                tuple(chunk.output_values for chunk in video.chunks), dim=1
            )
            output_accumulator.add(
                self._checkpointed_branch_logits(
                    output_keys.movedim(0, 1),
                    output_base,
                    output_contrast,
                    log_base_mass=output_mass,
                ),
                output_values,
            )
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
            bank_memory, candidates = self._bank_candidates(target, values, programs)
            query, dynamic = self._query_target(target, programs, bank_memory)
            a, b = self._pool_target(target, query, dynamic, candidates)
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
