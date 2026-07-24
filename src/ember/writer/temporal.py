"""Variable-length temporal aggregation for PI05 Action-Memory states."""

from __future__ import annotations

import torch


class VariableEpisodeInputError(ValueError):
    """Raised when a variable-length Action-Memory batch is malformed."""


class RMSNorm(torch.nn.Module):
    """RMS normalization used by the conditional temporal path."""

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
    """A pre-norm block with no independent output or adapter branch."""

    def __init__(
        self,
        width: int,
        heads: int,
        expansion: int = 4,
        *,
        linear_bias: bool,
    ) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads or expansion <= 0:
            raise VariableEpisodeInputError("invalid condition-only block dimensions")
        self.attention_norm = RMSNorm(width)
        self.attention = torch.nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            bias=linear_bias,
            batch_first=True,
        )
        self.ffn_norm = RMSNorm(width)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, expansion * width, bias=linear_bias),
            torch.nn.GELU(),
            torch.nn.Linear(expansion * width, width, bias=linear_bias),
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


def _apply_rope(value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Apply one-dimensional rotary positions to ``[N,H,T,D]`` Q or K."""

    width = value.shape[-1]
    if width % 2 or positions.shape != (value.shape[0], value.shape[2]):
        raise VariableEpisodeInputError("invalid temporal RoPE request")
    inverse_frequency = torch.exp(
        torch.arange(0, width, 2, device=value.device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10_000.0, device=value.device)) / width)
    )
    angles = positions.to(torch.float32)[:, None, :, None] * inverse_frequency[
        None, None, None
    ]
    cosine = torch.cos(angles).to(value.dtype)
    sine = torch.sin(angles).to(value.dtype)
    even = value[..., 0::2]
    odd = value[..., 1::2]
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine),
        dim=-1,
    ).flatten(-2)


class RotaryConditionOnlyBlock(torch.nn.Module):
    """A temporal pre-norm block whose Q/K carry signed relative time."""

    def __init__(
        self,
        width: int,
        heads: int,
        expansion: int = 4,
        *,
        linear_bias: bool,
    ) -> None:
        super().__init__()
        if (
            width <= 0
            or heads <= 0
            or width % heads
            or (width // heads) % 2
            or expansion <= 0
        ):
            raise VariableEpisodeInputError("invalid rotary block dimensions")
        self.heads = int(heads)
        self.head_width = width // heads
        self.attention_norm = RMSNorm(width)
        self.qkv = torch.nn.Linear(width, 3 * width, bias=linear_bias)
        self.output = torch.nn.Linear(width, width, bias=linear_bias)
        self.ffn_norm = RMSNorm(width)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, expansion * width, bias=linear_bias),
            torch.nn.GELU(),
            torch.nn.Linear(expansion * width, width, bias=linear_bias),
        )

    def forward(
        self,
        value: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, width = value.shape
        query, key, content = self.qkv(self.attention_norm(value)).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch, tokens, self.heads, self.head_width
            ).transpose(1, 2)

        query = _apply_rope(split_heads(query), positions)
        key = _apply_rope(split_heads(key), positions)
        content = split_heads(content)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            content,
            attn_mask=(~padding_mask)[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, width)
        value = value + self.output(attended)
        return value + self.ffn(self.ffn_norm(value))


class TemporalMemoryPool(torch.nn.Module):
    """Compress a variable trajectory with learned condition-only queries."""

    def __init__(
        self,
        width: int,
        heads: int,
        memory_tokens: int,
        *,
        linear_bias: bool,
    ) -> None:
        super().__init__()
        if min(width, heads, memory_tokens) <= 0 or width % heads:
            raise VariableEpisodeInputError("invalid temporal memory dimensions")
        self.memory_tokens = torch.nn.Parameter(torch.empty(memory_tokens, width))
        torch.nn.init.normal_(self.memory_tokens, mean=0.0, std=0.02)
        self.query_norm = RMSNorm(width)
        self.value_norm = RMSNorm(width)
        self.attention = torch.nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            bias=linear_bias,
            batch_first=True,
        )
        self.ffn_norm = RMSNorm(width)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, 4 * width, bias=linear_bias),
            torch.nn.GELU(),
            torch.nn.Linear(4 * width, width, bias=linear_bias),
        )

    def forward(
        self, value: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        queries = self.memory_tokens[None].expand(value.shape[0], -1, -1)
        normalized = self.value_norm(value)
        # Queries control what is read, but their residual is not propagated.
        # Every output token therefore remains conditioned on video states.
        attended, _ = self.attention(
            self.query_norm(queries),
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return attended + self.ffn(self.ffn_norm(attended))


class ActionMemoryTemporalEncoder(torch.nn.Module):
    """Aggregate ``[B,T,layer,slot,1024]`` into temporal memory tokens.

    Every ``(layer, slot)`` trajectory is a separate member of a flattened
    batch during temporal attention.  The shared temporal weights therefore
    process all 288 trajectories in parallel without allowing them to attend
    to one another.  Explicit layer and slot mixing happens only after temporal
    memory extraction.
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
        temporal_memory_tokens: int,
        frame_stride: int,
        conditional_linear_bias: bool,
    ) -> None:
        super().__init__()
        dimensions = (
            input_width,
            hidden_width,
            expert_layers,
            memory_slots,
            attention_heads,
            temporal_blocks,
            temporal_memory_tokens,
            frame_stride,
        )
        if any(value <= 0 for value in dimensions) or hidden_width % attention_heads:
            raise VariableEpisodeInputError("invalid Action-Memory temporal dimensions")
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.expert_layers = int(expert_layers)
        self.memory_slots = int(memory_slots)
        self.temporal_memory_tokens = int(temporal_memory_tokens)
        self.frame_stride = int(frame_stride)

        self.input_norm = RMSNorm(input_width)
        self.input_projection = torch.nn.Linear(
            input_width, hidden_width, bias=conditional_linear_bias
        )
        self.layer_modulation = torch.nn.Embedding(expert_layers, hidden_width)
        self.slot_modulation = torch.nn.Embedding(memory_slots, hidden_width)
        torch.nn.init.zeros_(self.layer_modulation.weight)
        torch.nn.init.zeros_(self.slot_modulation.weight)

        self.temporal = torch.nn.ModuleList(
            RotaryConditionOnlyBlock(
                hidden_width,
                attention_heads,
                linear_bias=conditional_linear_bias,
            )
            for _ in range(temporal_blocks)
        )
        self.temporal_memory = TemporalMemoryPool(
            hidden_width,
            attention_heads,
            temporal_memory_tokens,
            linear_bias=conditional_linear_bias,
        )
        self.layer_mixer = ConditionOnlyBlock(
            hidden_width,
            attention_heads,
            linear_bias=conditional_linear_bias,
        )
        self.slot_mixer = ConditionOnlyBlock(
            hidden_width,
            attention_heads,
            linear_bias=conditional_linear_bias,
        )

    def forward(
        self,
        states: torch.Tensor,
        frame_indices: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[B, expert_layers, memory_slots, memories, hidden]``."""

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
        layer_gate = torch.tanh(self.layer_modulation.weight)
        slot_gate = torch.tanh(self.slot_modulation.weight)

        # Identity signals modulate condition-derived states. They cannot form
        # a query-only route that emits a public adapter without video content.
        value = value * (
            1.0
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
        positions = (
            frame_indices.to(torch.float32)
            .div(float(self.frame_stride))[:, None, None, :]
            .expand(batch, self.expert_layers, self.memory_slots, frames)
            .reshape(batch * self.expert_layers * self.memory_slots, frames)
        )
        for block in self.temporal:
            value = block(value, positions, padding)

        pooled = self.temporal_memory(value, padding)
        pooled = pooled.reshape(
            batch,
            self.expert_layers,
            self.memory_slots,
            self.temporal_memory_tokens,
            self.hidden_width,
        )

        layer_view = pooled.permute(0, 2, 3, 1, 4).reshape(
            batch * self.memory_slots * self.temporal_memory_tokens,
            self.expert_layers,
            self.hidden_width,
        )
        layer_view = self.layer_mixer(layer_view)
        pooled = layer_view.reshape(
            batch,
            self.memory_slots,
            self.temporal_memory_tokens,
            self.expert_layers,
            self.hidden_width,
        ).permute(0, 3, 1, 2, 4)

        slot_view = pooled.permute(0, 1, 3, 2, 4).reshape(
            batch * self.expert_layers * self.temporal_memory_tokens,
            self.memory_slots,
            self.hidden_width,
        )
        slot_view = self.slot_mixer(slot_view)
        return slot_view.reshape(
            batch,
            self.expert_layers,
            self.temporal_memory_tokens,
            self.memory_slots,
            self.hidden_width,
        ).permute(0, 1, 3, 2, 4)
