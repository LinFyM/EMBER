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
    """Read Plan-relative residual values through routing-only event metadata."""

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
        event_routing: torch.Tensor,
        event_values: torch.Tensor,
        valid_memory: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(routing),
            self.memory_norm(event_routing),
            event_values,
            key_padding_mask=~valid_memory,
            need_weights=False,
        )
        content = attended
        return content + self.ffn(self.ffn_norm(content))


class ForecastBeliefEncoder(torch.nn.Module):
    """Build one fixed-layout Plan/Revision belief token per control time."""

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
            or width % 2
        ):
            raise VariableEpisodeInputError("invalid forecast-belief dimensions")
        self.action_width = int(action_width)
        self.horizon = int(horizon)
        self.width = int(width)
        self.branch_width = width // 2
        self.maximum_revision_count = int(maximum_revision_count)
        self.plan_encoder = torch.nn.Sequential(
            torch.nn.Linear(action_width, width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(width, self.branch_width, bias=False),
        )
        self.event_value_encoder = torch.nn.Sequential(
            torch.nn.Linear(2 * action_width, width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(width, width, bias=False),
        )
        self.event_routing_encoder = torch.nn.Sequential(
            torch.nn.Linear(3, width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(width, width, bias=False),
        )
        self.revision_query = torch.nn.Parameter(torch.empty(1, 1, width))
        self.revision_reader = RevisionContentBlock(width, heads)
        statistic_width = max(width // 4, 1)
        self.revision_statistics_encoder = torch.nn.Sequential(
            torch.nn.Linear(3, statistic_width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(statistic_width, width, bias=False),
        )
        self.revision_output = torch.nn.Linear(
            width,
            self.branch_width,
            bias=False,
        )
        self.plan_norm = RMSNorm(self.branch_width)
        self.revision_norm = RMSNorm(self.branch_width)
        torch.nn.init.normal_(self.revision_query, mean=0.0, std=0.02)

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

    def _forecast_layout(
        self,
        plans: torch.Tensor,
        frame_indices: torch.Tensor,
        absolute_time: torch.Tensor,
        coverage: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch, frames = frame_indices.shape
        times = absolute_time.numel()
        frame_grid = frame_indices[:, None, :].expand(batch, times, frames)
        latest_frame = frame_grid.masked_fill(~coverage, -1).argmax(dim=-1)
        latest_start = torch.gather(frame_indices, 1, latest_frame)
        latest_lead = absolute_time[None, :] - latest_start
        lead_grid = absolute_time[None, :, None] - frame_indices[:, None, :]
        gather_indices = lead_grid.permute(0, 2, 1).clamp(
            0,
            self.horizon - 1,
        )
        all_actions = torch.gather(
            plans,
            2,
            gather_indices[..., None].expand(
                batch,
                frames,
                times,
                self.action_width,
            ),
        ).permute(0, 2, 1, 3)
        latest_action = torch.gather(
            all_actions,
            2,
            latest_frame[..., None, None].expand(
                batch,
                times,
                1,
                self.action_width,
            ),
        ).squeeze(2)
        frame_ids = torch.arange(frames, device=plans.device)[None, None]
        earlier_mask = coverage & (frame_ids != latest_frame[..., None])
        residual = latest_action[..., None, :] - all_actions
        age_gap = latest_start[..., None] - frame_indices[:, None, :]
        routing_features = torch.stack(
            (
                lead_grid.to(torch.float32) / max(self.horizon - 1, 1),
                latest_lead.to(torch.float32)[..., None].expand_as(lead_grid)
                / max(self.horizon - 1, 1),
                age_gap.to(torch.float32) / max(self.horizon - 1, 1),
            ),
            dim=-1,
        )
        return (
            latest_action,
            latest_lead,
            residual,
            routing_features,
            earlier_mask,
        )

    def _revision_branch(
        self,
        residual: torch.Tensor,
        event_routing: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, maximum_time, event_count, _ = residual.shape
        value_features = torch.cat((residual, residual.abs()), dim=-1)
        flat_values = self.event_value_encoder(value_features).reshape(
            batch * maximum_time,
            event_count,
            self.width,
        )
        flat_routing = self.event_routing_encoder(event_routing).reshape(
            batch * maximum_time,
            event_count,
            self.width,
        )
        flat_mask = event_mask.reshape(batch * maximum_time, event_count)
        has_revision = flat_mask.any(dim=1)
        safe_mask = flat_mask.clone()
        safe_mask[~has_revision, 0] = True
        flat_values = flat_values.masked_fill(
            (~flat_mask)[..., None],
            0.0,
        )
        flat_routing = flat_routing.masked_fill(
            (~safe_mask)[..., None],
            0.0,
        )
        query = self.revision_query.expand(batch * maximum_time, -1, -1)
        directed_content = self.revision_reader(
            query,
            flat_routing,
            flat_values,
            safe_mask,
        )[:, 0]

        count = event_mask.sum(dim=-1).to(torch.float32)
        if bool((count > self.maximum_revision_count).any()):
            raise VariableEpisodeInputError(
                "revision count exceeds its sealed stride/horizon bound"
            )
        residual_float = residual.to(torch.float32)
        event_mask_float = event_mask.to(torch.float32)
        scalar_denominator = (
            count * self.action_width
        ).clamp_min(1.0)
        absolute = residual_float.abs() * event_mask_float[..., None]
        mean_absolute = absolute.sum(dim=(-1, -2)) / scalar_denominator
        masked_residual = residual_float * event_mask_float[..., None]
        rms = torch.linalg.vector_norm(
            masked_residual,
            dim=(-1, -2),
        ) / scalar_denominator.sqrt()
        maximum_absolute = absolute.amax(dim=(-1, -2))
        raw_statistics = torch.stack(
            (mean_absolute, rms, maximum_absolute),
            dim=-1,
        )
        statistic_content = self.revision_statistics_encoder(
            raw_statistics.to(dtype=directed_content.dtype)
        ).reshape(batch * maximum_time, self.width)
        combined = directed_content + statistic_content
        projected = self.revision_output(combined)
        strength = rms.detach()
        revision = (
            self.revision_norm(projected)
            * strength.reshape(-1, 1).to(dtype=projected.dtype)
        )
        revision = revision.masked_fill(
            ~has_revision[:, None],
            0.0,
        ).reshape(batch, maximum_time, self.branch_width)
        return (
            revision,
            count,
            strength,
        )

    def forward(
        self,
        plans: torch.Tensor,
        frame_indices: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Belief tokens, positions, valid mask, and Q/K routing features."""

        self._validate(plans, frame_indices, frame_mask)
        batch = plans.shape[0]
        absolute_time, coverage, valid_time = self._time_layout(
            plans,
            frame_indices,
            frame_mask,
        )
        (
            latest_action,
            latest_lead,
            residual,
            event_routing,
            event_mask,
        ) = self._forecast_layout(
            plans,
            frame_indices,
            absolute_time,
            coverage,
        )
        plan = self.plan_norm(self.plan_encoder(latest_action))
        revision, count, strength = self._revision_branch(
            residual,
            event_routing,
            event_mask,
        )
        maximum_time = absolute_time.numel()
        belief = torch.cat((plan, revision), dim=-1).masked_fill(
            ~valid_time[..., None],
            0.0,
        )
        positions = absolute_time[None].expand(batch, maximum_time)
        routing = torch.stack(
            (
                latest_lead.to(torch.float32) / max(self.horizon - 1, 1),
                count / self.maximum_revision_count,
                strength,
            ),
            dim=-1,
        ).masked_fill(
            ~valid_time[..., None],
            0.0,
        )
        return belief, positions, valid_time, routing


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
    """Zero-preserving content attention with routing-only Q/K metadata."""

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
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.value = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)
        self.ffn_norm = RMSNorm(width)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, 4 * width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(4 * width, width, bias=False),
        )

    def forward(
        self,
        value: torch.Tensor,
        positions: torch.Tensor,
        valid_mask: torch.Tensor,
        routing: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, width = value.shape
        addressed = self.attention_norm(value) + routing
        query = self.query(addressed)
        key = self.key(addressed)
        content = self.value(value)

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
    """Encode one Belief token per time without metadata entering content."""

    def __init__(self, *, width: int, heads: int, blocks: int) -> None:
        super().__init__()
        if min(width, heads, blocks) <= 0:
            raise VariableEpisodeInputError("invalid temporal encoder dimensions")
        self.routing_encoder = torch.nn.Sequential(
            torch.nn.Linear(3, width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(width, width, bias=False),
        )
        self.blocks = torch.nn.ModuleList(
            RotaryTemporalBlock(width, heads) for _ in range(blocks)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        valid_mask: torch.Tensor,
        routing_features: torch.Tensor,
    ) -> torch.Tensor:
        if (
            tokens.ndim != 3
            or positions.shape != tokens.shape[:2]
            or valid_mask.shape != tokens.shape[:2]
            or valid_mask.dtype != torch.bool
            or routing_features.shape != (*tokens.shape[:2], 3)
        ):
            raise VariableEpisodeInputError("invalid temporal token batch")
        routing = self.routing_encoder(
            routing_features.to(dtype=tokens.dtype)
        ).masked_fill(~valid_mask[..., None], 0.0)
        value = tokens
        value = value.masked_fill(~valid_mask[..., None], 0.0)
        for block in self.blocks:
            value = block(value, positions, valid_mask, routing)
        return value.masked_fill(~valid_mask[..., None], 0.0)


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
            content,
            need_weights=False,
        )
        content = content + attended
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.cross_attention(
            self.cross_norm(content) + routing,
            normalized_memory,
            memory,
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
