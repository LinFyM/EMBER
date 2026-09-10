"""Strong full-H per-frame evidence for the order-free video-set comparator."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ember.writer.attention import Attention, RotaryBlock, feed_forward


class FrameEvidenceGroup(nn.Module):
    """Mix the native action horizon and ground it in this frame's real Z."""

    def __init__(self, width: int, heads: int, visual_width: int) -> None:
        super().__init__()
        self.horizon = RotaryBlock(width, heads, causal=False)
        self.visual_query_norm, self.language_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.language_query = nn.Linear(width, width)
        self.visual_read = Attention(width, heads, visual_width)
        self.ffn_norm, self.ffn = nn.LayerNorm(width), feed_forward(width)

    def forward(self, states: Tensor, h_positions: Tensor, language: Tensor,
                visual: Tensor, valid: Tensor) -> Tensor:
        states = self.horizon(states, h_positions)
        query = self.visual_query_norm(states) + self.language_query(self.language_norm(language))[:, None, :]
        states = states + self.visual_read(query, visual, visual, valid[:, None, None, :])
        return states + self.ffn(self.ffn_norm(states))


class FrameEvidenceEncoder(nn.Module):
    """No frame ordinal, temporal state, directed pair, or video length input."""

    def __init__(self, width: int, heads: int, horizon: int, visual_width: int,
                 blocks: int, activation_checkpoint: bool) -> None:
        super().__init__()
        self.activation_checkpoint = activation_checkpoint
        self.groups = nn.ModuleList([FrameEvidenceGroup(width, heads, visual_width) for _ in range(blocks)])
        self.read_norm, self.language_query = nn.LayerNorm(width), nn.Linear(width, width)
        self.horizon_read = Attention(width, heads)
        self.register_buffer("h_positions", torch.arange(horizon), persistent=False)

    def forward(self, states: Tensor, language: Tensor, visual: Tensor, valid: Tensor) -> Tensor:
        for group in self.groups:
            args = (states, self.h_positions, language, visual, valid)
            if self.activation_checkpoint and torch.is_grad_enabled():
                states = checkpoint(group, *args, use_reentrant=False)
            else:
                states = group(*args)
        normalized = self.read_norm(states)
        return self.horizon_read(self.language_query(language)[:, None, :], normalized, normalized).squeeze(-2)
