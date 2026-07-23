"""Variable-length temporal aggregation for PI05 Action-Memory states."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class VariableEpisodeInputError(ValueError):
    """Raised when a variable-length Action-Memory batch is malformed."""


def sinusoidal_positions(
    positions: torch.Tensor, width: int, dtype: torch.dtype
) -> torch.Tensor:
    """Return deterministic absolute positions at an arbitrary integer stride."""

    if positions.ndim != 1 or width <= 0:
        raise VariableEpisodeInputError("invalid temporal position request")
    half = width // 2
    frequencies = torch.exp(
        torch.arange(half, device=positions.device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(half - 1, 1))
    )
    angles = positions.to(torch.float32)[:, None] * frequencies[None]
    value = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    if value.shape[-1] < width:
        value = F.pad(value, (0, width - value.shape[-1]))
    return value.to(dtype=dtype)


class RMSNorm(torch.nn.Module):
    """Bias-free RMS normalization."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        if width <= 0:
            raise VariableEpisodeInputError("RMSNorm width must be positive")
        self.weight = torch.nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value * torch.rsqrt(
            value.to(torch.float32).square().mean(dim=-1, keepdim=True) + self.eps
        ).to(value.dtype)
        return normalized * self.weight


class ConditionOnlyBlock(torch.nn.Module):
    """A bias-free pre-norm block whose output always depends on its input."""

    def __init__(self, width: int, heads: int, expansion: int = 4) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads or expansion <= 0:
            raise VariableEpisodeInputError("invalid condition-only block dimensions")
        self.attention_norm = RMSNorm(width)
        self.attention = torch.nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.ffn_norm = RMSNorm(width)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, expansion * width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(expansion * width, width, bias=False),
        )

    def forward(
        self, value: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        normalized = self.attention_norm(value)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        value = value + attended
        return value + self.ffn(self.ffn_norm(value))


class ActionMemoryTemporalEncoder(torch.nn.Module):
    """Aggregate ``[B,T,layer,slot,1024]`` into one vector per layer/slot.

    Every ``(layer, slot)`` trajectory is a separate member of a flattened
    batch during temporal attention.  The shared temporal weights therefore
    process all 288 trajectories in parallel without allowing them to attend
    to one another.  Explicit layer and slot mixing happens only after temporal
    pooling.
    """

    def __init__(
        self,
        *,
        input_width: int,
        hidden_width: int,
        expert_layers: int,
        memory_slots: int,
        attention_heads: int,
        temporal_blocks: int,
    ) -> None:
        super().__init__()
        dimensions = (
            input_width,
            hidden_width,
            expert_layers,
            memory_slots,
            attention_heads,
            temporal_blocks,
        )
        if any(value <= 0 for value in dimensions) or hidden_width % attention_heads:
            raise VariableEpisodeInputError("invalid Action-Memory temporal dimensions")
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.expert_layers = int(expert_layers)
        self.memory_slots = int(memory_slots)

        self.input_norm = RMSNorm(input_width)
        self.input_projection = torch.nn.Linear(
            input_width, hidden_width, bias=False
        )
        self.time_modulation = torch.nn.Linear(
            hidden_width, hidden_width, bias=False
        )
        self.layer_modulation = torch.nn.Embedding(expert_layers, hidden_width)
        self.slot_modulation = torch.nn.Embedding(memory_slots, hidden_width)
        torch.nn.init.zeros_(self.time_modulation.weight)
        torch.nn.init.zeros_(self.layer_modulation.weight)
        torch.nn.init.zeros_(self.slot_modulation.weight)

        self.temporal = torch.nn.ModuleList(
            ConditionOnlyBlock(hidden_width, attention_heads)
            for _ in range(temporal_blocks)
        )
        self.temporal_score = torch.nn.Linear(hidden_width, 1, bias=False)
        self.layer_mixer = ConditionOnlyBlock(hidden_width, attention_heads)
        self.slot_mixer = ConditionOnlyBlock(hidden_width, attention_heads)

    def forward(
        self,
        states: torch.Tensor,
        frame_indices: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[B, expert_layers, memory_slots, hidden_width]``."""

        if (
            states.ndim != 5
            or states.shape[2:4] != (self.expert_layers, self.memory_slots)
            or states.shape[-1] != self.input_width
            or frame_indices.shape != frame_mask.shape
            or states.shape[:2] != frame_mask.shape
            or frame_indices.dtype != torch.long
            or frame_mask.dtype != torch.bool
            or not bool(frame_mask.any(dim=1).all())
        ):
            raise VariableEpisodeInputError("invalid Action-Memory trajectory batch")
        value = self.input_projection(self.input_norm(states))
        batch, frames = frame_mask.shape
        positions = sinusoidal_positions(
            frame_indices.reshape(-1), self.hidden_width, value.dtype
        ).reshape(batch, frames, self.hidden_width)
        time_gate = torch.tanh(self.time_modulation(positions))
        layer_gate = torch.tanh(self.layer_modulation.weight)
        slot_gate = torch.tanh(self.slot_modulation.weight)

        # Identity signals modulate condition-derived states.  They cannot form
        # a query-only route that emits a public adapter without video content.
        value = value * (
            1.0
            + time_gate[:, :, None, None]
            + layer_gate[None, None, :, None]
            + slot_gate[None, None, None, :]
        )

        value = value.permute(0, 2, 3, 1, 4).reshape(
            batch * self.expert_layers * self.memory_slots,
            frames,
            self.hidden_width,
        )
        padding = (
            ~frame_mask[:, None, None, :]
            .expand(batch, self.expert_layers, self.memory_slots, frames)
            .reshape(batch * self.expert_layers * self.memory_slots, frames)
        )
        for block in self.temporal:
            value = block(value, padding)

        scores = self.temporal_score(value).squeeze(-1).masked_fill(
            padding, float("-inf")
        )
        pooled = torch.sum(torch.softmax(scores, dim=-1)[..., None] * value, dim=1)
        pooled = pooled.reshape(
            batch, self.expert_layers, self.memory_slots, self.hidden_width
        )

        layer_view = pooled.permute(0, 2, 1, 3).reshape(
            batch * self.memory_slots, self.expert_layers, self.hidden_width
        )
        layer_view = self.layer_mixer(layer_view)
        pooled = layer_view.reshape(
            batch, self.memory_slots, self.expert_layers, self.hidden_width
        ).permute(0, 2, 1, 3)

        slot_view = pooled.reshape(
            batch * self.expert_layers, self.memory_slots, self.hidden_width
        )
        slot_view = self.slot_mixer(slot_view)
        return slot_view.reshape(
            batch, self.expert_layers, self.memory_slots, self.hidden_width
        )
