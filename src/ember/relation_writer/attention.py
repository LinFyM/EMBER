"""Shared width-256 attention operators for relation G and independent F."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

WIDTH = 256
HEADS = 8
HEAD_WIDTH = WIDTH // HEADS


def split_heads(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(*value.shape[:-1], HEADS, HEAD_WIDTH).transpose(-3, -2)


def merge_heads(value: torch.Tensor) -> torch.Tensor:
    return value.transpose(-3, -2).flatten(-2)


def check_frames(frame_indices, count: int, device: torch.device) -> torch.Tensor:
    indices = torch.as_tensor(frame_indices, device=device)
    if indices.shape != (count,) or count == 0:
        raise ValueError("relations require one real frame index per frame")
    if not bool(torch.isfinite(indices).all()) or not bool((indices[1:] > indices[:-1]).all()):
        raise ValueError("relation frame indices must be finite and strictly increasing")
    return indices


def rotary(value: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
    """Adjacent-pair RoPE on Q/K only, at real frame_index / stride5."""
    frequency = torch.exp(torch.arange(0, HEAD_WIDTH, 2, device=value.device).float()
                          * (-math.log(10000.0) / HEAD_WIDTH))
    phase = frame_indices.float()[:, None] / 5.0 * frequency[None]
    even, odd = value.float()[..., 0::2], value.float()[..., 1::2]
    output = torch.stack((even * phase.cos() - odd * phase.sin(),
                          even * phase.sin() + odd * phase.cos()), -1)
    return output.flatten(-2).to(value.dtype)


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(WIDTH, WIDTH)
        self.k = nn.Linear(WIDTH, WIDTH)
        self.v = nn.Linear(WIDTH, WIDTH)
        self.out = nn.Linear(WIDTH, WIDTH)

    def forward(self, query: torch.Tensor, memory: torch.Tensor,
                frame_indices: torch.Tensor | None = None,
                key_valid: torch.Tensor | None = None) -> torch.Tensor:
        q, k = split_heads(self.q(query)), split_heads(self.k(memory))
        if frame_indices is not None:
            q, k = rotary(q, frame_indices), rotary(k, frame_indices)
        mask = None if key_valid is None else key_valid[..., None, None, :].bool()
        values = F.scaled_dot_product_attention(q, k, split_heads(self.v(memory)),
                                                attn_mask=mask, dropout_p=0.0)
        return self.out(merge_heads(values))


def feed_forward() -> nn.Sequential:
    return nn.Sequential(nn.Linear(WIDTH, 1024), nn.GELU(), nn.Linear(1024, WIDTH))


class CrossAttentionBlock(nn.Module):
    """Pre-LN residual cross-attention and FFN, with no positional Value."""
    def __init__(self) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(WIDTH)
        self.memory_norm = nn.LayerNorm(WIDTH)
        self.attention = Attention()
        self.ffn_norm = nn.LayerNorm(WIDTH)
        self.ffn = feed_forward()

    def forward(self, query: torch.Tensor, memory: torch.Tensor,
                key_valid: torch.Tensor | None = None) -> torch.Tensor:
        query = query + self.attention(self.query_norm(query), self.memory_norm(memory),
                                       key_valid=key_valid)
        return query + self.ffn(self.ffn_norm(query))


class TemporalBlock(nn.Module):
    """Bidirectional per-entity temporal attention, Q/K-only real-time RoPE."""
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH)
        self.attention = Attention()
        self.ffn_norm = nn.LayerNorm(WIDTH)
        self.ffn = feed_forward()

    def forward(self, nodes: torch.Tensor, frame_indices: torch.Tensor,
                key_valid: torch.Tensor | None = None) -> torch.Tensor:
        normal = self.norm(nodes)
        nodes = nodes + self.attention(normal, normal, frame_indices, key_valid)
        return nodes + self.ffn(self.ffn_norm(nodes))


def presence_attention(scores: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
    """Existence-weighted selection; a completely absent memory returns zero."""
    weights = scores.float().softmax(-1) * presence.float()[..., None, None, :]
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
