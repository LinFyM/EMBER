"""Joint video content and continuous complete-LoRA parameter decoding."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class VariableEpisodeInputError(ValueError):
    """Raised when a variable-length video batch violates its contract."""


class RMSNorm(torch.nn.Module):
    """Normalize each content token without introducing an additive value."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        if width <= 0 or eps <= 0:
            raise VariableEpisodeInputError("RMSNorm width and epsilon must be positive")
        self.weight = torch.nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return value * scale.to(value.dtype) * self.weight.to(value.dtype)


def _apply_rope(value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Apply raw-frame RoPE to query or key tensors shaped [B,H,T,D]."""
    width = value.shape[-1]
    frequency = torch.exp(torch.arange(0, width, 2, device=value.device).float()
                          * (-math.log(10_000.0) / width))
    angles = positions.float()[:, None, :, None] * frequency[None, None, None]
    cosine, sine = angles.cos().to(value.dtype), angles.sin().to(value.dtype)
    even, odd = value[..., 0::2], value[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2)


def token_role_addresses(
    type_embeddings: torch.Tensor, semantic_tokens: int, horizon: int,
) -> torch.Tensor:
    """Shared type addresses plus separate, fixed within-type sinusoidal indices."""
    if (type_embeddings.ndim != 2 or type_embeddings.shape[0] != 2
            or type_embeddings.shape[1] <= 0 or type_embeddings.shape[1] % 2
            or semantic_tokens <= 0 or horizon <= 0):
        raise VariableEpisodeInputError("invalid semantic/action role addresses")
    width = type_embeddings.shape[1]
    frequency = torch.exp(torch.arange(0, width, 2, device=type_embeddings.device).float()
                          * (-math.log(10_000.0) / width))
    addresses = []
    for kind, count in enumerate((semantic_tokens, horizon)):
        angles = torch.arange(count, device=type_embeddings.device).float()[:, None] * frequency[None]
        fixed = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        addresses.append(type_embeddings[kind][None] + fixed.to(type_embeddings.dtype))
    return torch.cat(addresses, dim=0)


def _split_heads(value: torch.Tensor, heads: int) -> torch.Tensor:
    batch, tokens, width = value.shape
    return value.reshape(batch, tokens, heads, width // heads).transpose(1, 2)


def _merge_heads(value: torch.Tensor) -> torch.Tensor:
    batch, heads, tokens, width = value.shape
    return value.transpose(1, 2).reshape(batch, tokens, heads * width)


def _memory_mask(memory, valid_frames, valid_roles, role_addresses, width):
    if (memory.ndim != 4 or min(memory.shape[:3]) <= 0 or memory.shape[-1] != width
            or valid_frames.shape != memory.shape[:2] or valid_frames.dtype != torch.bool
            or valid_roles.shape != (memory.shape[0], memory.shape[2]) or valid_roles.dtype != torch.bool
            or role_addresses.shape != memory.shape[2:]
            or not bool(valid_frames.any(dim=1).all()) or not bool(valid_roles.any(dim=1).all())):
        raise VariableEpisodeInputError("invalid joint video memory or padding")
    return valid_frames[:, :, None] & valid_roles[:, None, :]


class _ContentAttention(torch.nn.Module):
    """Q/K addressing is explicit and cannot enter the separate Value input."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise VariableEpisodeInputError("invalid attention dimensions")
        self.heads = heads
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.value = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)

    def forward(self, query_key, memory_key, memory_value, valid_memory=None, positions=None):
        query = _split_heads(self.query(query_key), self.heads)
        key = _split_heads(self.key(memory_key), self.heads)
        if positions is not None:
            query, key = _apply_rope(query, positions), _apply_rope(key, positions)
        value = _split_heads(self.value(memory_value), self.heads)
        allowed = None if valid_memory is None else valid_memory[:, None, None, :]
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed, dropout_p=0.0)
        return self.output(_merge_heads(attended))


def _feed_forward(width: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(torch.nn.Linear(width, 4 * width, bias=False), torch.nn.GELU(),
                               torch.nn.Linear(4 * width, width, bias=False))


class _JointVideoBlock(torch.nn.Module):
    """Mix all roles within a frame, then all frames within a role."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.role_norm = RMSNorm(width)
        self.time_norm = RMSNorm(width)
        self.ffn_norm = RMSNorm(width)
        self.role_attention = _ContentAttention(width, heads)
        self.time_attention = _ContentAttention(width, heads)
        self.ffn = _feed_forward(width)

    def forward(self, memory, frame_positions, valid_frames, valid_roles, role_addresses, mask):
        batch, frames, roles, width = memory.shape
        normalized = self.role_norm(memory).reshape(batch * frames, roles, width)
        addressed = normalized + role_addresses[None].to(normalized.dtype)
        role_keys = valid_roles[:, None].expand(batch, frames, roles).reshape(batch * frames, roles)
        update = self.role_attention(addressed, addressed, normalized, role_keys)
        memory = (memory + update.reshape_as(memory)).masked_fill(~mask[..., None], 0.0)

        normalized = self.time_norm(memory).transpose(1, 2).reshape(batch * roles, frames, width)
        positions = frame_positions[:, None].expand(batch, roles, frames).reshape(batch * roles, frames)
        time_keys = valid_frames[:, None].expand(batch, roles, frames).reshape(batch * roles, frames)
        update = self.time_attention(normalized, normalized, normalized, time_keys, positions)
        memory = memory + update.reshape(batch, roles, frames, width).transpose(1, 2)
        memory = memory + self.ffn(self.ffn_norm(memory))
        return memory.masked_fill(~mask[..., None], 0.0)


class JointVideoStack(torch.nn.Module):
    """Replicable joint blocks with frame time only in temporal Q/K."""

    def __init__(self, width: int, heads: int, blocks: int) -> None:
        super().__init__()
        if min(width, heads, blocks) <= 0 or width % heads or (width // heads) % 2:
            raise VariableEpisodeInputError("invalid joint video stack dimensions")
        self.width = width
        self.blocks = torch.nn.ModuleList(_JointVideoBlock(width, heads) for _ in range(blocks))

    def forward(self, memory, frame_positions, valid_frames, valid_roles, role_addresses):
        mask = _memory_mask(memory, valid_frames, valid_roles, role_addresses, self.width)
        if frame_positions.shape != memory.shape[:2] or frame_positions.dtype != torch.long:
            raise VariableEpisodeInputError("joint video time must contain raw frame indices")
        content = memory.masked_fill(~mask[..., None], 0.0)
        for block in self.blocks:
            content = block(content, frame_positions, valid_frames, valid_roles, role_addresses, mask)
        return content


class _ParameterDecoderBlock(torch.nn.Module):
    """Write a content read and slot coordination back into one parameter state."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.query_norm = RMSNorm(width)
        self.memory_norm = RMSNorm(width)
        self.self_norm = RMSNorm(width)
        self.ffn_norm = RMSNorm(width)
        self.cross_attention = _ContentAttention(width, heads)
        self.self_attention = _ContentAttention(width, heads)
        self.ffn = _feed_forward(width)

    def forward(self, content, memory, valid_memory, role_addresses, routing):
        content = content + self.cross_attention(
            self.query_norm(content) + routing,
            self.memory_norm(memory) + role_addresses[None].to(memory.dtype),
            memory, valid_memory,
        )
        addressed = self.self_norm(content) + routing
        content = content + self.self_attention(addressed, addressed, content)
        return content + self.ffn(self.ffn_norm(content))


class ContinuousParameterDecoder(torch.nn.Module):
    """Initialize 320 slots from semantic content, then read complete joint memory."""

    EXPERT_LAYERS = 18
    RANK = 16
    QUERY_COUNT = EXPERT_LAYERS * RANK + 2 * RANK

    def __init__(self, width: int, heads: int, blocks: int, initialization_seed: int) -> None:
        super().__init__()
        if min(width, heads, blocks) <= 0 or width % heads:
            raise VariableEpisodeInputError("invalid continuous parameter decoder dimensions")
        self.width = width
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)

        def parameter(rows):
            value = torch.empty(rows, width)
            value.normal_(mean=0.0, std=0.02, generator=generator)
            return torch.nn.Parameter(value)

        self.query_table = parameter(self.QUERY_COUNT)
        self.module_identity = parameter(3)
        self.layer_identity = parameter(self.EXPERT_LAYERS)
        self.rank_identity = parameter(self.RANK)
        self.routing_norm = RMSNorm(width)
        self.blocks = torch.nn.ModuleList(_ParameterDecoderBlock(width, heads) for _ in range(blocks))
        self.output_norm = RMSNorm(width)

    def _routing(self):
        stop = self.EXPERT_LAYERS * self.RANK
        expert = (self.query_table[:stop].reshape(self.EXPERT_LAYERS, self.RANK, -1)
                  + self.module_identity[0] + self.layer_identity[:, None]
                  + self.rank_identity[None]).reshape(stop, -1)
        action_in = self.query_table[stop:stop + self.RANK] + self.module_identity[1] + self.rank_identity
        action_out = self.query_table[-self.RANK:] + self.module_identity[2] + self.rank_identity
        return self.routing_norm(torch.cat((expert, action_in, action_out), dim=0))

    def forward(self, memory, valid_frames, valid_roles, role_addresses, semantic_tokens: int):
        mask = _memory_mask(memory, valid_frames, valid_roles, role_addresses, self.width)
        if (not 0 < semantic_tokens < memory.shape[2]
                or not bool(valid_roles[:, :semantic_tokens].any(dim=1).all())):
            raise VariableEpisodeInputError("parameter initialization requires valid semantic content")
        batch, frames, _, width = memory.shape
        memory = memory.masked_fill(~mask[..., None], 0.0)
        routing = self._routing()[None].to(memory.dtype)
        content = memory.new_zeros(batch, self.QUERY_COUNT, width)
        for index, block in enumerate(self.blocks):
            count = semantic_tokens if index == 0 else memory.shape[2]
            values = memory[:, :, :count].reshape(batch, frames * count, width)
            allowed = mask[:, :, :count].reshape(batch, frames * count)
            addresses = role_addresses[:count][None].expand(frames, count, width).reshape(frames * count, width)
            content = block(content, values, allowed, addresses, routing)
        content = self.output_norm(content)
        stop = self.EXPERT_LAYERS * self.RANK
        return (content[:, :stop].reshape(batch, self.EXPERT_LAYERS, self.RANK, width),
                content[:, stop:stop + self.RANK], content[:, -self.RANK:])
