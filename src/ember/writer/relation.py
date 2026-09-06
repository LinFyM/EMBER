"""Synchronous, same-layer local relations over a single ordered video."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def directional_correspondence(scores: Tensor) -> tuple[Tensor, Tensor]:
    """Normalize a chronological score and its transpose independently."""
    return scores.softmax(-1), scores.transpose(-1, -2).softmax(-1)


def relative_correspondence(attention: Tensor) -> Tensor:
    """Reindex [..., head, h, g] to [..., head, h, delta + H - 1]."""
    horizon = attention.shape[-1]
    h = torch.arange(horizon, device=attention.device)
    offsets = h[None, :] - h[:, None] + horizon - 1
    offsets = offsets.expand(*attention.shape[:-2], horizon, horizon)
    rho = attention.new_zeros(*attention.shape[:-1], 2 * horizon - 1)
    return rho.scatter(-1, offsets, attention)


class PairMessage(nn.Module):
    """One GELU MLP jointly consumes x, matched content, rho, and signed gap."""

    def __init__(self, width: int, heads: int, horizon: int) -> None:
        super().__init__()
        self.current = nn.Linear(width, width)
        self.matched = nn.Linear(width, width, bias=False)
        self.gap = nn.Linear(1, width, bias=False)
        self.relative = nn.Parameter(torch.empty(heads, 2 * horizon - 1, width))
        nn.init.normal_(self.relative, std=(heads * (2 * horizon - 1)) ** -0.5)
        self.output = nn.Linear(width, width)

    def relative_read(self, attention: Tensor) -> Tensor:
        # This bounded rho tensor avoids broadcasting a [H,H,width] table over
        # every edge/layer. Its linear read is exactly sum_g A[h,g] e[g-h].
        rho = relative_correspondence(attention).transpose(-3, -2).flatten(-2)
        weight = self.relative.flatten(0, 1).transpose(0, 1)
        return F.linear(rho, weight)

    def forward(self, current: Tensor, matched: Tensor, attention: Tensor, gap: Tensor) -> Tensor:
        hidden = self.current(current) + self.matched(matched)
        hidden = hidden + self.relative_read(attention)
        hidden = hidden + self.gap(gap[:, None, None, None] / 5)
        return self.output(F.gelu(hidden))


class LocalRelationBlock(nn.Module):
    """Shared across native layers; all edges read the same previous U."""

    def __init__(
        self, width: int, heads: int, horizon: int, radius: int,
        edge_chunk: int, activation_checkpoint: bool,
    ) -> None:
        super().__init__()
        self.width, self.heads, self.horizon = width, heads, horizon
        self.radius, self.edge_chunk = radius, edge_chunk
        self.activation_checkpoint = activation_checkpoint
        self.norm = nn.LayerNorm(width)
        self.content = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.bias = nn.Sequential(nn.Linear(2, width // heads), nn.GELU(),
                                  nn.Linear(width // heads, heads))
        self.message = PairMessage(width, heads, horizon)
        self.neighbor_query = nn.Linear(width, width)
        self.neighbor_key = nn.Linear(width, width)
        self.neighbor_value = nn.Linear(width, width)
        self.neighbor_output = nn.Linear(width, width, bias=False)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))
        positions = torch.arange(horizon)
        self.register_buffer("offsets", positions[None, :] - positions[:, None], persistent=False)

    def _heads(self, value: Tensor) -> Tensor:
        return value.unflatten(-1, (self.heads, self.width // self.heads)).transpose(-3, -2)

    def chronological_score(self, late: Tensor, early: Tensor, gap: Tensor) -> Tensor:
        """Inputs are preprojected [..., heads, H, head_width] features."""
        score = late @ early.transpose(-1, -2) / math.sqrt(self.width // self.heads)
        shape = (len(gap), self.horizon, self.horizon)
        joint = torch.stack((gap[:, None, None].expand(shape) / 5,
                             self.offsets.to(gap).expand(shape) / self.horizon), dim=-1)
        bias = self.bias(joint).permute(0, 3, 1, 2).unsqueeze(1)
        return score + bias

    def _pair_messages(
        self, x: Tensor, content: Tensor, value: Tensor,
        late: Tensor, early: Tensor, gap: Tensor,
    ) -> tuple[Tensor, Tensor]:
        score = self.chronological_score(content[late], content[early], gap)
        to_late, to_early = directional_correspondence(score)
        late_read = (to_late @ value[early]).transpose(-3, -2).flatten(-2)
        early_read = (to_early @ value[late]).transpose(-3, -2).flatten(-2)
        return (self.message(x[late], late_read, to_late, gap),
                self.message(x[early], early_read, to_early, -gap))

    def _edges(self, length: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        distance = torch.arange(1, min(self.radius, length - 1) + 1, device=device)
        late = torch.arange(length, device=device)[:, None].expand(-1, len(distance))
        early = late - distance
        valid = early >= 0
        return late[valid], early[valid], distance.expand_as(late)[valid]

    def _messages(self, x: Tensor, frame_indices: Tensor) -> tuple[Tensor, Tensor]:
        length, layers, horizon, width = x.shape
        late, early, distance = self._edges(length, x.device)
        content, value = self._heads(self.content(x)), self._heads(self.value(x))
        messages, destinations = [], []
        for start in range(0, len(late), self.edge_chunk):
            end = start + self.edge_chunk
            left, right = late[start:end], early[start:end]
            gap = (frame_indices[left] - frame_indices[right]).to(x.dtype)
            arguments = (x, content, value, left, right, gap)
            if self.activation_checkpoint and torch.is_grad_enabled():
                outputs = checkpoint(self._pair_messages, *arguments, use_reentrant=False)
            else:
                outputs = self._pair_messages(*arguments)
            messages.extend(outputs)
            slot = distance[start:end] - 1
            destinations.extend((left * (2 * self.radius) + slot,
                                 right * (2 * self.radius) + self.radius + slot))
        indices = torch.cat(destinations)
        shape = (length * 2 * self.radius, layers, horizon, width)
        values = torch.cat(messages)
        packed = values.new_zeros(shape).index_copy(0, indices, values)
        mask = torch.zeros(length * 2 * self.radius, dtype=torch.bool, device=x.device)
        mask[indices] = True
        return packed.unflatten(0, (length, 2 * self.radius)), mask.view(length, -1)

    def _neighbor_update(self, x: Tensor, messages: Tensor, mask: Tensor) -> Tensor:
        # [T, neighbor, J, H, d] -> [T, J, H, head, neighbor, head_width].
        keys = self.neighbor_key(messages).unflatten(-1, (self.heads, self.width // self.heads))
        values = self.neighbor_value(messages).unflatten(-1, (self.heads, self.width // self.heads))
        keys, values = keys.permute(0, 2, 3, 4, 1, 5), values.permute(0, 2, 3, 4, 1, 5)
        query = self.neighbor_query(x).unflatten(-1, (self.heads, self.width // self.heads))
        update = F.scaled_dot_product_attention(
            query.unsqueeze(-2), keys, values, attn_mask=mask[:, None, None, None, None, :],
        ).squeeze(-2).flatten(-2)
        return self.neighbor_output(update)

    def forward(self, states: Tensor, frame_indices: Tensor) -> Tensor:
        x = self.norm(states)
        if len(states) > 1 and self.radius > 0:
            messages, mask = self._messages(x, frame_indices)
            states = states + self._neighbor_update(x, messages, mask)
        # For T=1 there is no attention call and the neighbor update is exactly 0.
        return states + self.ffn(self.ffn_norm(states))
