"""Projected attention and pre-norm blocks shared by the horizon Writer."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def position_encoding(positions: Tensor, width: int, dtype: torch.dtype) -> Tensor:
    frequencies = torch.exp(-math.log(10000) * torch.arange(0, width, 2, device=positions.device).float() / width)
    phase = positions.float()[..., None] * frequencies
    return torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)[..., :width].to(dtype)


def feed_forward(width: int, input_width: int | None = None) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_width or width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))


class Attention(nn.Module):
    """K/V may be projected once and reused by several edge batches."""

    def __init__(self, width: int, heads: int, memory_width: int | None = None,
                 *, zero_value: bool = False) -> None:
        super().__init__()
        self.heads, self.head_width = heads, width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(memory_width or width, width)
        # Q/K may use static conditions while zero process Values stay zero.
        self.value = nn.Linear(memory_width or width, width, bias=not zero_value)
        self.output = nn.Linear(width, width, bias=not zero_value)

    def split_heads(self, value: Tensor) -> Tensor:
        return value.unflatten(-1, (self.heads, self.head_width)).transpose(-3, -2)

    def project_memory(self, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        return self.split_heads(self.key(key)), self.split_heads(self.value(value))

    def read_projected(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None,
                       *, return_log_distribution: bool = False):
        queries = self.split_heads(self.query(query))
        result = F.scaled_dot_product_attention(queries, key, value, attn_mask=mask)
        output = self.output(result.transpose(-3, -2).flatten(-2))
        if not return_log_distribution:
            return output
        # Same Q/K as the actual Value read. FP32 log probabilities keep the
        # auxiliary spatial KL stable without changing SDPA or its output.
        logits = (queries.float() @ key.float().transpose(-2, -1)) / math.sqrt(self.head_width)
        if mask is not None:
            logits = logits.masked_fill(~mask, -torch.inf) if mask.dtype == torch.bool else logits + mask
        log_probabilities = logits.float().log_softmax(-1)
        supported = torch.isfinite(log_probabilities).any((-3, -2))
        # Avoid logsumexp(all -inf)'s undefined derivative at fully masked
        # task/padding positions; those positions have exactly zero read mass.
        safe_probabilities = torch.where(supported[..., None, None, :], log_probabilities, 0.)
        log_distribution = safe_probabilities.logsumexp((-3, -2)) - math.log(queries.shape[-3] * queries.shape[-2])
        log_distribution = log_distribution.masked_fill(~supported, -torch.inf)
        return output, log_distribution

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None,
                *, return_log_distribution: bool = False):
        return self.read_projected(query, *self.project_memory(key, value), mask,
                                   return_log_distribution=return_log_distribution)

    def _rope(self, value: Tensor, positions: Tensor) -> Tensor:
        frequencies = torch.exp(-math.log(10000) * torch.arange(0, self.head_width, 2, device=value.device).float() / self.head_width)
        phase = positions.float()[..., None] * frequencies
        cosine, sine = phase.cos().to(value.dtype), phase.sin().to(value.dtype)
        pairs = value.unflatten(-1, (-1, 2))
        left, right = pairs.unbind(-1)
        return torch.stack((left * cosine - right * sine, left * sine + right * cosine), -1).flatten(-2)

    def self_rotary(self, value: Tensor, positions: Tensor, causal: bool) -> Tensor:
        query = self._rope(self.split_heads(self.query(value)), positions)
        key, values = self.project_memory(value, value)
        key = self._rope(key, positions)
        result = F.scaled_dot_product_attention(query, key, values, is_causal=causal)
        return self.output(result.transpose(-3, -2).flatten(-2))


class RotaryBlock(nn.Module):
    def __init__(self, width: int, heads: int, *, causal: bool) -> None:
        super().__init__()
        self.causal = causal
        self.attention_norm, self.ffn_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention, self.ffn = Attention(width, heads), feed_forward(width)

    def forward(self, value: Tensor, positions: Tensor) -> Tensor:
        value = value + self.attention.self_rotary(self.attention_norm(value), positions, self.causal)
        return value + self.ffn(self.ffn_norm(value))


class CompilerBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.cross_norm, self.self_norm, self.ffn_norm = (nn.LayerNorm(width) for _ in range(3))
        self.memory_norm = nn.LayerNorm(width)
        self.cross, self.self_attention = Attention(width, heads), Attention(width, heads)
        self.ffn = feed_forward(width)

    def forward(self, query: Tensor, memory: Tensor, routing: Tensor, prior: Tensor,
                query_bias: Tensor | None = None) -> Tensor:
        values = self.memory_norm(memory)
        lookup = query if query_bias is None else query + query_bias
        query = query + self.cross(self.cross_norm(lookup), values + routing, values, prior)
        normalized = self.self_norm(query)
        query = query + self.self_attention(normalized, normalized, normalized)
        return query + self.ffn(self.ffn_norm(query))
