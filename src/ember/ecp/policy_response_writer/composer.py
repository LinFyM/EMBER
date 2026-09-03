"""Current-video native factor composer with exact signed X/Y pooling."""

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
    rms_normalize,
)
from ember.ecp.policy_response_writer.capture import FrozenPolicyResponseVideo
from ember.ecp.policy_response_writer.process import (
    GatedMLP,
    PolicyResponseProcessOutput,
)


DENSE_BANK_ATTENTION_TOKEN_LIMIT = 192 * 1024
STREAMING_BANK_BLOCK_TOKEN_LIMIT = 128 * 1024
PROCESS_RELATION_COUNT = 4


@dataclass(frozen=True)
class _NativeBankChunk:
    """Projected keys and raw values shared by context read and signed pooling."""

    input_values: torch.Tensor
    input_keys: torch.Tensor
    output_values: torch.Tensor
    output_keys: torch.Tensor
    innovation: torch.Tensor


@dataclass(frozen=True)
class _NativeVideoCandidates:
    frame_count: int
    chunks: tuple[_NativeBankChunk, ...]


class _GroupedOnlineSoftmaxAccumulator:
    """Exact signed-pooling statistics with an independent group axis."""

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
        # logits is [rank, branch, frame, group, ...]; values keeps group
        # first so every native output group has its own exact normalization.
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
        # Restore [rank, group * native-width], matching the deployed target.
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
    """Bound the complete target update, not each rank component separately."""

    mean_square = _effective_update_mean_square(a, b)
    denominator = mean_square.clamp_min(torch.finfo(mean_square.dtype).tiny).sqrt()
    return torch.clamp(cap.to(mean_square) / denominator, max=1.0)


class RankBankContextBlock(torch.nn.Module):
    """One copyable event read, bank read, rank attention, and gated MLP."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.query_norm = torch.nn.LayerNorm(width)
        self.event_norm = torch.nn.LayerNorm(width)
        self.event_attention = torch.nn.MultiheadAttention(
            width, heads, batch_first=True
        )
        self.bank_norm = torch.nn.LayerNorm(width)
        self.bank_attention = torch.nn.MultiheadAttention(
            width, heads, batch_first=True
        )
        self.rank_norm = torch.nn.LayerNorm(width)
        self.rank_attention = torch.nn.MultiheadAttention(
            width, heads, batch_first=True
        )
        self.mlp = GatedMLP(width)

    def forward(
        self,
        query: torch.Tensor,
        event_memory: torch.Tensor,
        bank_memory: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        q = self.query_norm(query)[None]
        event = self.event_norm(event_memory)[None]
        attended, _ = self.event_attention(q, event, event, need_weights=False)
        value = query + attended[0]
        q = self.query_norm(value)[None]
        bank = tuple(bank_memory)
        if torch.is_grad_enabled() and any(chunk.requires_grad for chunk in bank):
            attended = checkpoint(
                self._exact_bank_attention,
                q,
                bank,
                use_reentrant=False,
            )
        else:
            attended = self._exact_bank_attention(q, bank)
        value = value + attended[0]
        normalized = self.rank_norm(value)[None]
        attended, _ = self.rank_attention(
            normalized, normalized, normalized, need_weights=False
        )
        return self.mlp(value + attended[0])

    def _exact_bank_attention(
        self,
        query: torch.Tensor,
        memory: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Use one fused full-bank read when its transient memory is bounded."""

        token_count = sum(int(chunk.shape[0]) for chunk in memory)
        if token_count <= DENSE_BANK_ATTENTION_TOKEN_LIMIT:
            return self._dense_bank_attention(query, memory)
        return self._streaming_bank_attention(query, memory)

    def _dense_bank_attention(
        self,
        query: torch.Tensor,
        memory: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Exact SDPA over the concatenated ordered bank, with no axis pooling."""

        if query.ndim != 3 or query.shape[0] != 1 or not memory:
            raise ValueError("dense native-bank attention inputs changed")
        width = query.shape[-1]
        if any(chunk.ndim != 2 or chunk.shape[-1] != width for chunk in memory):
            raise ValueError("dense native-bank memory changed")
        heads = self.bank_attention.num_heads
        head_width = width // heads
        weight = self.bank_attention.in_proj_weight
        bias = self.bank_attention.in_proj_bias
        q_bias = None if bias is None else bias[:width]
        k_bias = None if bias is None else bias[width : 2 * width]
        v_bias = None if bias is None else bias[2 * width :]
        projected_query = F.linear(query[0], weight[:width], q_bias)
        bank = self.bank_norm(torch.cat(tuple(memory), dim=0))
        key = F.linear(bank, weight[width : 2 * width], k_bias)
        value = F.linear(bank, weight[2 * width :], v_bias)
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
        self,
        query: torch.Tensor,
        memory: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Exact cross-attention over full bank tokens in bounded fused blocks."""

        if query.ndim != 3 or query.shape[0] != 1 or not memory:
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
        # The capture boundary deliberately emits small frame chunks.  They
        # are useful while native X/Y are being constructed, but replaying
        # every one as a separate LayerNorm/projection/reduction leaves long
        # videos launch-bound.  Coalesce adjacent chunks only for this exact
        # online-softmax reduction.  Token order and every frame/probe/horizon/
        # bank-type value remain present; the bound controls transient memory.
        blocks = []
        pending = []
        pending_tokens = 0
        for chunk in memory:
            if chunk.ndim != 2 or chunk.shape[-1] != width:
                raise ValueError("streaming native-bank memory changed")
            chunk_tokens = int(chunk.shape[0])
            if pending and (
                pending_tokens + chunk_tokens > STREAMING_BANK_BLOCK_TOKEN_LIMIT
            ):
                blocks.append(torch.cat(tuple(pending), dim=0))
                pending = []
                pending_tokens = 0
            pending.append(chunk)
            pending_tokens += chunk_tokens
        if pending:
            blocks.append(torch.cat(tuple(pending), dim=0))

        for chunk in blocks:
            chunk = self.bank_norm(chunk)
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
    """Compose rank-four factors without a solve, transport, or free residual."""

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
        if width % heads or block_depth <= 0 or pooling_frame_chunk <= 0:
            raise ValueError("native composer block topology changed")
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
        self.event_projection = torch.nn.Linear(2 * width, width, bias=False)
        self.occupancy_projection = torch.nn.Linear(2, width, bias=False)
        self.blocks = torch.nn.ModuleList(
            RankBankContextBlock(width, heads) for _ in range(block_depth)
        )
        self.common_query = torch.nn.Linear(width, width, bias=False)
        self.innovation_key = torch.nn.Linear(width, width, bias=False)
        self.input_positive_query = torch.nn.Linear(width, width, bias=False)
        self.input_negative_query = torch.nn.Linear(width, width, bias=False)
        self.output_positive_query = torch.nn.Linear(width, width, bias=False)
        self.output_negative_query = torch.nn.Linear(width, width, bias=False)
        self.scale_head = torch.nn.Linear(width, 1)
        self.output_norm = torch.nn.LayerNorm(width)
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
        torch.nn.init.zeros_(self.scale_head.weight)
        torch.nn.init.zeros_(self.scale_head.bias)
        # The relation path is the sole new parameter in the matched arm.
        # Fork CPU RNG so adding it does not perturb any established Composer
        # parameter initialized under the same formal seed.
        with torch.random.fork_rng(devices=[]):
            self.relation_embedding = torch.nn.Embedding(PROCESS_RELATION_COUNT, width)
            torch.nn.init.normal_(self.relation_embedding.weight, std=width**-0.5)

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
        key = self.input_projection[str(owner.in_features)](rms_normalize(values))
        key = key + self._owner_bias(target)
        key = key + self.probe_embedding.weight[None, :, None]
        key = key + self.horizon_embedding.weight[None, None]
        key = key + self.bank_type_embedding.weight[0]
        return self.output_norm(key + positions[:, None, None])

    def _output_keys(
        self,
        target: int,
        values: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # values is [group, frame, probe, horizon, bank_type, native_width].
        width = values.shape[-1]
        key = self.output_projection[str(width)](rms_normalize(values))
        key = key + self._owner_bias(target)
        key = key + self.probe_embedding.weight[None, None, :, None, None]
        key = key + self.horizon_embedding.weight[None, None, None, :, None]
        key = key + self.bank_type_embedding.weight[1:][None, None, None, None]
        key = (
            key + self.group_embedding.weight[: values.shape[0], None, None, None, None]
        )
        return self.output_norm(key + positions[None, :, None, None, None])

    def _bank_candidates(
        self,
        target: int,
        videos: Sequence[FrozenPolicyResponseVideo],
        processes: Sequence[PolicyResponseProcessOutput],
    ) -> tuple[tuple[torch.Tensor, ...], tuple[_NativeVideoCandidates, ...]]:
        tokens = []
        candidates = []
        owner = self.owners[target]
        groups = native_output_group_count(owner)
        group_width = owner.out_features // groups
        for video, process in zip(videos, processes, strict=True):
            if process.frame_innovation.shape[:2] != (
                video.frame_count,
                len(self.owners),
            ):
                raise ValueError("native composer process/video frames changed")
            if (
                process.assignment.ndim != 3
                or process.assignment.shape[1:]
                != (video.frame_count, PROCESS_RELATION_COUNT)
                or process.innovations.shape[:2]
                != (process.assignment.shape[0], len(self.owners))
            ):
                raise ValueError("native composer event-relation assignment changed")
            # Preserve the explicit relation axis promised by the Process ->
            # Composer contract. Summing this axis recovers the old frame
            # innovation, but doing so before candidate scoring erased which
            # scene transition supported each dynamic direction.
            relation_innovation = torch.einsum(
                "etm,ejd->tmjd", process.assignment, process.innovations
            )
            positions = self._position(video)
            boundary = NativeOutputBankState(final=video.final_outputs[target])
            chunks = []
            for start in range(0, video.frame_count, self.pooling_frame_chunk):
                stop = min(start + self.pooling_frame_chunk, video.frame_count)
                local_position = positions[start:stop]
                input_values = video.native_inputs[target][start:stop]
                input_key = self._input_keys(target, input_values, local_position)
                # Keep every relative Action Expert horizon available to the
                # process-conditioned bank read.  Averaging this axis here
                # would erase the same 50-step response structure that the
                # full Process path is required to preserve.
                tokens.append(input_key.reshape(-1, self.width))
                output = boundary.build(
                    video.native_outputs[target][start:stop], start_frame=start
                )
                grouped = output.reshape(
                    *output.shape[:-1], groups, group_width
                ).movedim(-2, 0)
                output_key = self._output_keys(target, grouped, local_position)
                tokens.append(output_key.reshape(-1, self.width))
                chunks.append(
                    _NativeBankChunk(
                        input_values=input_values,
                        input_keys=input_key,
                        output_values=grouped,
                        output_keys=output_key,
                        innovation=relation_innovation[start:stop, :, target],
                    )
                )
            if boundary.next_frame != video.frame_count:
                raise ValueError("native composer context stream ended early")
            candidates.append(
                _NativeVideoCandidates(
                    frame_count=video.frame_count,
                    chunks=tuple(chunks),
                )
            )
        if not tokens:
            raise ValueError("native composer received no bank context")
        return tuple(tokens), tuple(candidates)

    def _event_tokens(
        self,
        target: int,
        processes: Sequence[PolicyResponseProcessOutput],
    ) -> torch.Tensor:
        rows = []
        for process in processes:
            occupancy = process.occupancy / process.occupancy.sum().clamp_min(1e-6)
            meta = torch.stack((occupancy, process.presence), -1)
            rows.append(
                self.event_projection(
                    torch.cat(
                        (
                            process.events[:, target],
                            process.innovations[:, target],
                        ),
                        -1,
                    )
                )
                + self.occupancy_projection(meta)
            )
        return torch.cat(rows)

    def _query_target(
        self,
        target: int,
        processes: Sequence[PolicyResponseProcessOutput],
        bank_memory: torch.Tensor,
    ) -> torch.Tensor:
        common = torch.stack(
            tuple(process.common[target] for process in processes)
        ).mean(0)
        language = torch.stack(
            tuple(process.owner_language[target] for process in processes)
        ).mean(0)
        query = self.rank_queries + self._owner_bias(target) + common + language
        if self.task_query is not None:
            query = query + self.task_query[target]
        event_memory = self._event_tokens(target, processes)
        for block in self.blocks:
            query = block(query, event_memory, bank_memory)
        return query

    def _branch_logits(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        relation_innovation: torch.Tensor,
        *,
        log_base_mass: float,
        output: bool,
        projected_queries: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None,
    ) -> torch.Tensor:
        if projected_queries is None:
            positive_head = (
                self.output_positive_query if output else self.input_positive_query
            )
            negative_head = (
                self.output_negative_query if output else self.input_negative_query
            )
            projected_queries = (
                self.common_query(query),
                positive_head(query),
                negative_head(query),
            )
        common_query, positive_query, negative_query = projected_queries
        common = torch.einsum("rd,...d->r...", common_query, keys) / math.sqrt(
            self.width
        )
        if (
            relation_innovation.ndim != 3
            or relation_innovation.shape[:2] != (keys.shape[0], PROCESS_RELATION_COUNT)
            or relation_innovation.shape[-1] != self.width
        ):
            raise ValueError("native composer relation innovation changed")
        # Relation identity may modulate dynamic evidence but cannot create a
        # branch-specific bias of its own. Therefore D == 0 still gives
        # identical positive/negative distributions and zero mobile value,
        # while functional credit now reaches alpha(e,t,m).
        # Raw X/Y do not have a relation axis. This marginalization is exactly
        # equivalent to repeating each raw candidate four times with 1/4 base
        # mass, then pooling the shared value. Score one relation at a time so
        # the relation x native-token x hidden-width intermediate is never
        # materialized fourfold; logaddexp preserves the same exact marginal.
        relation_mass = log_base_mass - math.log(PROCESS_RELATION_COUNT)
        relation_scale = (1.0 + torch.tanh(self.relation_embedding.weight)).to(
            relation_innovation
        )
        key_feature = torch.tanh(keys)
        marginal = None
        for relation in range(PROCESS_RELATION_COUNT):
            innovation = (
                self.innovation_key(relation_innovation[:, relation])
                * relation_scale[relation]
            )
            while innovation.ndim < keys.ndim:
                innovation = innovation.unsqueeze(1)
            branch_feature = innovation * key_feature
            positive = torch.einsum(
                "rd,...d->r...", positive_query, branch_feature
            ) / math.sqrt(self.width)
            negative = torch.einsum(
                "rd,...d->r...", negative_query, branch_feature
            ) / math.sqrt(self.width)
            logits = torch.stack(
                (
                    common + positive + relation_mass,
                    common + negative + relation_mass,
                ),
                dim=1,
            ).float()
            marginal = logits if marginal is None else torch.logaddexp(marginal, logits)
        if marginal is None:
            raise RuntimeError("native composer relation marginal is empty")
        return marginal

    def _checkpointed_branch_logits(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        relation_innovation: torch.Tensor,
        projected_queries: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        log_base_mass: float,
        output: bool,
    ) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return self._branch_logits(
                query,
                keys,
                relation_innovation,
                log_base_mass=log_base_mass,
                output=output,
                projected_queries=projected_queries,
            )

        def score(
            local_query: torch.Tensor,
            local_keys: torch.Tensor,
            local_innovation: torch.Tensor,
            common_query: torch.Tensor,
            positive_query: torch.Tensor,
            negative_query: torch.Tensor,
        ) -> torch.Tensor:
            return self._branch_logits(
                local_query,
                local_keys,
                local_innovation,
                log_base_mass=log_base_mass,
                output=output,
                projected_queries=(common_query, positive_query, negative_query),
            )

        return checkpoint(
            score,
            query,
            keys,
            relation_innovation,
            *projected_queries,
            use_reentrant=False,
        )

    def _pool_target(
        self,
        target: int,
        query: torch.Tensor,
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
        common_query = self.common_query(query)
        input_queries = (
            common_query,
            self.input_positive_query(query),
            self.input_negative_query(query),
        )
        output_queries = (
            common_query,
            self.output_positive_query(query),
            self.output_negative_query(query),
        )
        video_count = len(videos)
        for video in videos:
            input_mass = -math.log(video_count * video.frame_count * 2 * ACTION_HORIZON)
            output_mass = -math.log(
                video_count * video.frame_count * 2 * ACTION_HORIZON * 4
            )
            # Native boundary construction remains chunked. Once its projected
            # tensors are resident, concatenate only their ordered frame axis
            # so each bank group uses one query projection and reduction. No
            # frame, probe, horizon, bank-type, or native-value axis is pooled.
            innovation = torch.cat(
                tuple(chunk.innovation for chunk in video.chunks), dim=0
            )
            input_keys = torch.cat(
                tuple(chunk.input_keys for chunk in video.chunks), dim=0
            )
            input_values = torch.cat(
                tuple(chunk.input_values for chunk in video.chunks), dim=0
            )
            input_accumulator.add(
                self._checkpointed_branch_logits(
                    query,
                    input_keys,
                    innovation,
                    input_queries,
                    log_base_mass=input_mass,
                    output=False,
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
                    query,
                    output_keys.movedim(0, 1),
                    innovation,
                    output_queries,
                    log_base_mass=output_mass,
                    output=True,
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
            query = self._query_target(target, programs, bank_memory)
            a, b = self._pool_target(target, query, candidates)
            # Match the G1-proven asymmetric optimization geometry: native A/B
            # directions exist at initialization, while the effective B scale
            # is exactly zero.  Step 1 opens the scale head and step 2 delivers
            # functional credit to the full current-bank/process path.
            scale = s_ref[target].to(query) * torch.tanh(
                self.scale_head(query).squeeze(-1)
            )
            a = rms_normalize(a, epsilon=1e-6)
            b = rms_normalize(b, epsilon=1e-6) * scale[:, None]
            cap_factor = _effective_update_cap_factor(a, b, s_ref[target])
            b = b * cap_factor.to(b)
            scale = scale * cap_factor.to(scale)
            a_values.append(a)
            b_values.append(b)
            scales.append(scale)
        return NativeFactorResidual(
            a=tuple(a_values),
            b=tuple(b_values),
            scales=torch.stack(scales),
        )
