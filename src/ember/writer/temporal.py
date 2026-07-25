"""Absolute-time plan/revision encoding for the canonical PI05 Writer."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class VariableEpisodeInputError(ValueError):
    """Raised when a variable-length forecast batch violates its contract."""


class RMSNorm(torch.nn.Module):
    """Small dtype-stable RMS normalization."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        if width <= 0:
            raise VariableEpisodeInputError("RMSNorm width must be positive")
        self.weight = torch.nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(
            value.to(torch.float32).square().mean(dim=-1, keepdim=True) + self.eps
        ).to(value.dtype)
        return value * scale * self.weight


class RevisionContentBlock(torch.nn.Module):
    """Read directed revision events without copying a static query to output."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise VariableEpisodeInputError("invalid revision-read dimensions")
        self.query_norm = RMSNorm(width)
        self.memory_norm = RMSNorm(width)
        self.attention = torch.nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            batch_first=True,
            bias=False,
        )
        self.ffn_norm = RMSNorm(width)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, 4 * width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(4 * width, width, bias=False),
        )

    def forward(
        self,
        routing: torch.Tensor,
        memory: torch.Tensor,
        valid_memory: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(routing),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=~valid_memory,
            need_weights=False,
        )
        content = attended
        return content + self.ffn(self.ffn_norm(content))


class PlanRevisionEncoder(torch.nn.Module):
    """Convert overlapping action plans into two tokens per absolute time."""

    def __init__(
        self,
        *,
        action_width: int,
        horizon: int,
        width: int,
        heads: int,
        maximum_revision_count: int,
    ) -> None:
        super().__init__()
        if (
            min(
                action_width,
                horizon,
                width,
                heads,
                maximum_revision_count,
            )
            <= 0
            or width % heads
        ):
            raise VariableEpisodeInputError("invalid Plan/Revision dimensions")
        self.action_width = int(action_width)
        self.horizon = int(horizon)
        self.width = int(width)
        self.maximum_revision_count = int(maximum_revision_count)
        self.plan_encoder = torch.nn.Sequential(
            torch.nn.Linear(action_width + 1, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, width),
        )
        self.event_encoder = torch.nn.Sequential(
            torch.nn.Linear(3 * action_width + 3, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, width),
        )
        self.revision_query = torch.nn.Parameter(torch.empty(1, 1, width))
        self.revision_reader = RevisionContentBlock(width, heads)
        self.plan_norm = RMSNorm(width)
        self.revision_norm = RMSNorm(width)
        gate_width = max(width // 4, 1)
        self.stability_gate = torch.nn.Sequential(
            torch.nn.Linear(4, gate_width),
            torch.nn.GELU(),
            torch.nn.Linear(gate_width, width),
        )
        self.no_revision = torch.nn.Parameter(torch.empty(width))
        torch.nn.init.normal_(self.revision_query, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.no_revision, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.stability_gate[-1].weight)
        torch.nn.init.zeros_(self.stability_gate[-1].bias)

    def _validate(
        self,
        plans: torch.Tensor,
        frame_indices: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> None:
        if (
            plans.ndim != 4
            or plans.shape[2:] != (self.horizon, self.action_width)
            or frame_indices.shape != plans.shape[:2]
            or frame_indices.dtype != torch.long
            or frame_mask.shape != plans.shape[:2]
            or frame_mask.dtype != torch.bool
            or not bool(frame_mask[:, 0].all())
            or bool((frame_mask[:, 1:] & ~frame_mask[:, :-1]).any())
        ):
            raise VariableEpisodeInputError("invalid packed action forecasts")
        for row in range(plans.shape[0]):
            active = frame_indices[row, frame_mask[row]]
            if (
                active.numel() == 0
                or int(active[0]) != 0
                or bool((active[1:] <= active[:-1]).any())
            ):
                raise VariableEpisodeInputError(
                    "forecast frame indices must start at zero and increase"
                )

    @staticmethod
    def _gather_plan_actions(
        plans: torch.Tensor,
        frame_ids: torch.Tensor,
        leads: torch.Tensor,
    ) -> torch.Tensor:
        batch, times = frame_ids.shape
        batch_ids = torch.arange(batch, device=plans.device)[:, None].expand(
            batch, times
        )
        selected = plans[batch_ids, frame_ids]
        return torch.gather(
            selected,
            2,
            leads[..., None, None].expand(
                batch,
                times,
                1,
                plans.shape[-1],
            ),
        ).squeeze(2)

    def _time_layout(
        self,
        plans: torch.Tensor,
        frame_indices: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        last = frame_indices.masked_fill(~frame_mask, -1).max(dim=1).values
        lengths = last + self.horizon
        absolute_time = torch.arange(int(lengths.max()), device=plans.device)
        lead_grid = absolute_time[None, :, None] - frame_indices[:, None, :]
        coverage = (
            frame_mask[:, None, :]
            & (lead_grid >= 0)
            & (lead_grid < self.horizon)
        )
        valid_time = absolute_time[None, :] < lengths[:, None]
        if bool((valid_time & ~coverage.any(dim=-1)).any()):
            raise VariableEpisodeInputError("forecast sequence left an uncovered time")
        return absolute_time, coverage, valid_time

    def _plan_tokens(
        self,
        plans: torch.Tensor,
        frame_indices: torch.Tensor,
        absolute_time: torch.Tensor,
        coverage: torch.Tensor,
    ) -> torch.Tensor:
        frame_grid = frame_indices[:, None, :]
        latest_frame = frame_grid.masked_fill(~coverage, -1).argmax(dim=-1)
        latest_start = torch.gather(frame_indices, 1, latest_frame)
        latest_lead = absolute_time[None, :] - latest_start
        latest_action = self._gather_plan_actions(
            plans,
            latest_frame,
            latest_lead.clamp(0, self.horizon - 1),
        )
        normalized_lead = (
            latest_lead.to(torch.float32) / max(self.horizon - 1, 1)
        )[..., None]
        encoded = self.plan_encoder(
            torch.cat((latest_action, normalized_lead), dim=-1)
        )
        return self.plan_norm(encoded)

    def _revision_event_inputs(
        self,
        plans: torch.Tensor,
        frame_indices: torch.Tensor,
        frame_mask: torch.Tensor,
        absolute_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, frame_count = plans.shape[:2]
        maximum_time = absolute_time.numel()
        if frame_count == 1:
            event_mask = torch.zeros(
                batch,
                maximum_time,
                1,
                dtype=torch.bool,
                device=plans.device,
            )
            delta = plans.new_zeros(
                batch,
                maximum_time,
                1,
                self.action_width,
            )
            features = plans.new_zeros(
                batch,
                maximum_time,
                1,
                3 * self.action_width + 3,
            )
            return features, event_mask, delta

        old_time = frame_indices[:, :-1]
        new_time = frame_indices[:, 1:]
        adjacent = frame_mask[:, :-1] & frame_mask[:, 1:]
        old_lead = absolute_time[None, :, None] - old_time[:, None, :]
        new_lead = absolute_time[None, :, None] - new_time[:, None, :]
        event_mask = (
            adjacent[:, None, :]
            & (old_lead >= 0)
            & (old_lead < self.horizon)
            & (new_lead >= 0)
            & (new_lead < self.horizon)
        )
        old_index = old_lead.permute(0, 2, 1).clamp(0, self.horizon - 1)
        new_index = new_lead.permute(0, 2, 1).clamp(0, self.horizon - 1)
        gather_shape = (
            batch,
            frame_count - 1,
            maximum_time,
            self.action_width,
        )
        old_action = torch.gather(
            plans[:, :-1],
            2,
            old_index[..., None].expand(gather_shape),
        ).permute(0, 2, 1, 3)
        new_action = torch.gather(
            plans[:, 1:],
            2,
            new_index[..., None].expand(gather_shape),
        ).permute(0, 2, 1, 3)
        delta = new_action - old_action
        delta_time = (new_time - old_time)[:, None, :].expand(
            batch,
            maximum_time,
            frame_count - 1,
        )
        features = torch.cat(
            (
                old_action,
                new_action,
                delta,
                (
                    old_lead.to(torch.float32) / max(self.horizon - 1, 1)
                )[..., None],
                (
                    new_lead.to(torch.float32) / max(self.horizon - 1, 1)
                )[..., None],
                delta_time.to(torch.float32)[..., None],
            ),
            dim=-1,
        )
        return features, event_mask, delta

    def _revision_tokens(
        self,
        event_features: torch.Tensor,
        event_mask: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        batch, maximum_time, event_count = event_mask.shape
        flat_events = self.event_encoder(event_features).reshape(
            batch * maximum_time,
            event_count,
            self.width,
        )
        flat_mask = event_mask.reshape(batch * maximum_time, event_count)
        has_revision = flat_mask.any(dim=1)
        safe_mask = flat_mask.clone()
        safe_mask[~has_revision, 0] = True
        flat_events = flat_events.masked_fill(
            (~flat_mask & has_revision[:, None])[..., None],
            0.0,
        )
        query = self.revision_query.expand(batch * maximum_time, -1, -1)
        directed_content = self.revision_reader(
            query,
            flat_events,
            safe_mask,
        )[:, 0]

        count = event_mask.sum(dim=-1).to(torch.float32)
        if bool((count > self.maximum_revision_count).any()):
            raise VariableEpisodeInputError(
                "revision count exceeds its sealed stride/horizon bound"
            )
        delta_norm = torch.linalg.vector_norm(delta.to(torch.float32), dim=-1)
        masked_norm = delta_norm * event_mask
        denominator = count.clamp_min(1.0)
        mean = masked_norm.sum(dim=-1) / denominator
        variance = (
            ((delta_norm - mean[..., None]).square() * event_mask).sum(dim=-1)
            / denominator
        )
        maximum = delta_norm.masked_fill(~event_mask, 0.0).max(dim=-1).values
        standard_deviation = torch.where(
            count > 1,
            variance.clamp_min(1e-12).sqrt(),
            torch.zeros_like(variance),
        )
        bounded_stability = torch.stack(
            (
                count / self.maximum_revision_count,
                mean / (1.0 + mean),
                standard_deviation / (1.0 + standard_deviation),
                maximum / (1.0 + maximum),
            ),
            dim=-1,
        )
        revision = torch.where(
            has_revision[:, None],
            directed_content,
            self.no_revision[None].expand_as(directed_content),
        ).reshape(batch, maximum_time, self.width)
        gate = 1.0 + 0.25 * torch.tanh(
            self.stability_gate(bounded_stability)
        )
        return self.revision_norm(revision) * gate.to(revision.dtype)

    def forward(
        self,
        plans: torch.Tensor,
        frame_indices: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return interleaved tokens, absolute positions and a valid-token mask."""

        self._validate(plans, frame_indices, frame_mask)
        batch = plans.shape[0]
        absolute_time, coverage, valid_time = self._time_layout(
            plans,
            frame_indices,
            frame_mask,
        )
        plan_tokens = self._plan_tokens(
            plans,
            frame_indices,
            absolute_time,
            coverage,
        )
        event_features, event_mask, delta = self._revision_event_inputs(
            plans,
            frame_indices,
            frame_mask,
            absolute_time,
        )
        revision = self._revision_tokens(event_features, event_mask, delta)
        maximum_time = absolute_time.numel()
        tokens = torch.stack((plan_tokens, revision), dim=2).reshape(
            batch,
            2 * maximum_time,
            self.width,
        )
        positions = absolute_time[None, :, None].expand(batch, -1, 2).reshape(
            batch,
            2 * maximum_time,
        )
        valid_tokens = valid_time[..., None].expand(batch, -1, 2).reshape(
            batch,
            2 * maximum_time,
        )
        return tokens, positions, valid_tokens


def _apply_rope(value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Apply one-dimensional RoPE to ``[B,H,T,D]`` query or key tensors."""

    width = value.shape[-1]
    if width % 2 or positions.shape != (value.shape[0], value.shape[2]):
        raise VariableEpisodeInputError("invalid temporal RoPE request")
    inverse_frequency = torch.exp(
        torch.arange(0, width, 2, device=value.device, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    angles = (
        positions.to(torch.float32)[:, None, :, None]
        * inverse_frequency[None, None, None]
    )
    cosine = torch.cos(angles).to(value.dtype)
    sine = torch.sin(angles).to(value.dtype)
    even, odd = value[..., 0::2], value[..., 1::2]
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine),
        dim=-1,
    ).flatten(-2)


class RotaryTemporalBlock(torch.nn.Module):
    """Pre-norm self-attention with true absolute-time RoPE."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if (
            min(width, heads) <= 0
            or width % heads
            or (width // heads) % 2
        ):
            raise VariableEpisodeInputError("invalid temporal block dimensions")
        self.heads = int(heads)
        self.head_width = width // heads
        self.attention_norm = RMSNorm(width)
        self.qkv = torch.nn.Linear(width, 3 * width)
        self.output = torch.nn.Linear(width, width)
        self.ffn_norm = RMSNorm(width)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, 4 * width),
            torch.nn.GELU(),
            torch.nn.Linear(4 * width, width),
        )

    def forward(
        self,
        value: torch.Tensor,
        positions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, width = value.shape
        query, key, content = self.qkv(self.attention_norm(value)).chunk(3, dim=-1)

        def heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch,
                tokens,
                self.heads,
                self.head_width,
            ).transpose(1, 2)

        query = _apply_rope(heads(query), positions)
        key = _apply_rope(heads(key), positions)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            heads(content),
            attn_mask=valid_mask[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, width)
        value = value + self.output(attended)
        value = value + self.ffn(self.ffn_norm(value))
        return value.masked_fill(~valid_mask[..., None], 0.0)


class VariableTimeTemporalEncoder(torch.nn.Module):
    """Encode interleaved Plan/Revision tokens without fixed video length."""

    def __init__(self, *, width: int, heads: int, blocks: int) -> None:
        super().__init__()
        if min(width, heads, blocks) <= 0:
            raise VariableEpisodeInputError("invalid temporal encoder dimensions")
        self.token_type = torch.nn.Parameter(torch.empty(2, width))
        torch.nn.init.normal_(self.token_type, mean=0.0, std=0.02)
        self.blocks = torch.nn.ModuleList(
            RotaryTemporalBlock(width, heads) for _ in range(blocks)
        )
        self.output_norm = RMSNorm(width)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            tokens.ndim != 3
            or positions.shape != tokens.shape[:2]
            or valid_mask.shape != tokens.shape[:2]
            or valid_mask.dtype != torch.bool
            or tokens.shape[1] % 2
        ):
            raise VariableEpisodeInputError("invalid temporal token batch")
        token_type = torch.arange(tokens.shape[1], device=tokens.device) % 2
        value = tokens + self.token_type.index_select(0, token_type)[None]
        value = value.masked_fill(~valid_mask[..., None], 0.0)
        for block in self.blocks:
            value = block(value, positions, valid_mask)
        return self.output_norm(value).masked_fill(~valid_mask[..., None], 0.0)


class ContentOnlyQueryBlock(torch.nn.Module):
    """Route LoRA slots while keeping their residual values memory-derived."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.self_norm = RMSNorm(width)
        self.self_attention = torch.nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            batch_first=True,
            bias=False,
        )
        self.cross_norm = RMSNorm(width)
        self.memory_norm = RMSNorm(width)
        self.cross_attention = torch.nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            batch_first=True,
            bias=False,
        )
        self.ffn_norm = RMSNorm(width)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, 4 * width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(4 * width, width, bias=False),
        )

    def forward(
        self,
        content: torch.Tensor,
        routing: torch.Tensor,
        memory: torch.Tensor,
        valid_memory: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.self_norm(content)
        addressed = normalized + routing
        attended, _ = self.self_attention(
            addressed,
            addressed,
            normalized,
            need_weights=False,
        )
        content = content + attended
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.cross_attention(
            self.cross_norm(content) + routing,
            normalized_memory,
            normalized_memory,
            key_padding_mask=~valid_memory,
            need_weights=False,
        )
        content = content + attended
        return content + self.ffn(self.ffn_norm(content))


class LoRAQueryDecoder(torch.nn.Module):
    """Decode 288 expert and 32 boundary-projection rank-slot states."""

    EXPERT_LAYERS = 18
    RANK = 16
    QUERY_COUNT = EXPERT_LAYERS * RANK + 2 * RANK

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        blocks: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if min(width, heads, blocks) <= 0 or width % heads:
            raise VariableEpisodeInputError("invalid LoRA query decoder dimensions")
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)

        def parameter(rows: int) -> torch.nn.Parameter:
            value = torch.empty(rows, width)
            value.normal_(mean=0.0, std=0.02, generator=generator)
            return torch.nn.Parameter(value)

        self.query_table = parameter(self.QUERY_COUNT)
        self.module_identity = parameter(3)
        self.layer_identity = parameter(self.EXPERT_LAYERS)
        self.rank_identity = parameter(self.RANK)
        self.routing_norm = RMSNorm(width)
        self.blocks = torch.nn.ModuleList(
            ContentOnlyQueryBlock(width, heads) for _ in range(blocks)
        )
        self.output_norm = RMSNorm(width)

    def _initial_queries(self) -> torch.Tensor:
        expert = (
            self.query_table[: self.EXPERT_LAYERS * self.RANK].reshape(
                self.EXPERT_LAYERS,
                self.RANK,
                -1,
            )
            + self.module_identity[0]
            + self.layer_identity[:, None]
            + self.rank_identity[None]
        ).reshape(self.EXPERT_LAYERS * self.RANK, -1)
        action_in = (
            self.query_table[
                self.EXPERT_LAYERS * self.RANK :
                self.EXPERT_LAYERS * self.RANK + self.RANK
            ]
            + self.module_identity[1]
            + self.rank_identity
        )
        action_out = (
            self.query_table[-self.RANK :]
            + self.module_identity[2]
            + self.rank_identity
        )
        return torch.cat((expert, action_in, action_out), dim=0)

    def forward(
        self,
        memory: torch.Tensor,
        valid_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            memory.ndim != 3
            or valid_memory.shape != memory.shape[:2]
            or valid_memory.dtype != torch.bool
            or not bool(valid_memory.any(dim=1).all())
        ):
            raise VariableEpisodeInputError("invalid LoRA decoder memory")
        routing = self.routing_norm(self._initial_queries())[None].expand(
            memory.shape[0],
            -1,
            -1,
        )
        content = memory.new_zeros(
            memory.shape[0],
            self.QUERY_COUNT,
            memory.shape[-1],
        )
        for block in self.blocks:
            content = block(content, routing, memory, valid_memory)
        content = self.output_norm(content)
        expert_stop = self.EXPERT_LAYERS * self.RANK
        expert = content[:, :expert_stop].reshape(
            memory.shape[0],
            self.EXPERT_LAYERS,
            self.RANK,
            -1,
        )
        return (
            expert,
            content[:, expert_stop : expert_stop + self.RANK],
            content[:, -self.RANK :],
        )
