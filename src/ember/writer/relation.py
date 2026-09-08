"""Past-edge correspondence, complete H queries, visual verification and short GRU."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ember.writer.attention import Attention, RotaryBlock, feed_forward


def relative_correspondence(attention: Tensor) -> Tensor:
    """Reindex [..., head, h, g] to [..., head, h, g-h+H-1], retaining mass."""
    horizon = attention.shape[-1]
    h = torch.arange(horizon, device=attention.device)
    offsets = (h[None, :] - h[:, None] + horizon - 1).expand_as(attention)
    return attention.new_zeros(*attention.shape[:-1], 2 * horizon - 1).scatter(-1, offsets, attention)


def past_edges(length: int, radius: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Chronological slots per current frame; no synthetic neighbors."""
    current = torch.arange(length, device=device)[:, None].expand(-1, radius)
    slot = torch.arange(radius, device=device)[None, :].expand(length, -1)
    past = (current - radius).clamp_min(0) + slot
    valid = past < current
    return current[valid], past[valid], slot[valid]


class LocalRelationBlock(nn.Module):
    def __init__(self, width: int, heads: int, horizon: int, radius: int, edge_chunk: int,
                 activation_checkpoint: bool, visual_width: int) -> None:
        super().__init__()
        self.width, self.heads, self.horizon = width, heads, horizon
        self.radius, self.edge_chunk, self.activation_checkpoint = radius, edge_chunk, activation_checkpoint
        self.norm = nn.LayerNorm(width)
        self.content, self.value = (nn.Linear(width, width, bias=False) for _ in range(2))
        self.bias = nn.Sequential(nn.Linear(2, width // heads), nn.GELU(), nn.Linear(width // heads, heads))
        nn.init.zeros_(self.bias[-1].weight)
        nn.init.zeros_(self.bias[-1].bias)
        self.null = nn.Sequential(nn.Linear(width + 1, width), nn.GELU(), nn.Linear(width, heads))
        self.matched = nn.Linear(width, width, bias=False)
        self.relative_read = nn.Linear(heads * (2 * horizon - 1), width, bias=False)
        self.query_input = feed_forward(width, 6 * width + heads + 1)
        self.horizon_query = RotaryBlock(width, heads, causal=False)
        self.visual_read = Attention(width, heads, visual_width)
        self.visual_roles = nn.Parameter(torch.randn(2, width) * 0.02)
        self.message = feed_forward(width, 7 * width + heads + 1)
        self.message_norm, self.initial_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.time_input = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.initial = nn.Linear(width, width)
        self.gru = nn.GRUCell(width, width)
        self.neighbor_output = nn.Linear(width, width, bias=False)
        self.ffn_norm, self.ffn = nn.LayerNorm(width), feed_forward(width)
        h = torch.arange(horizon)
        self.register_buffer("offsets", h[None, :] - h[:, None], persistent=False)
        self.register_buffer("h_positions", h, persistent=False)

    def _heads(self, value: Tensor) -> Tensor:
        return value.unflatten(-1, (self.heads, self.width // self.heads)).transpose(-3, -2)

    def correspondence(self, current: Tensor, late: Tensor, early: Tensor, gap: Tensor) -> Tensor:
        gap = gap.to(current.dtype)
        scores = late @ early.transpose(-1, -2) / math.sqrt(self.width // self.heads)
        shape = (len(gap), self.horizon, self.horizon)
        joint = torch.stack((gap[:, None, None].expand(shape) / 5,
                             (self.offsets.to(gap) - gap[:, None, None]) / self.horizon), -1)
        scores = scores + self.bias(joint).permute(0, 3, 1, 2)
        null_input = torch.cat((current, (gap[:, None, None] / 5).expand(-1, self.horizon, 1)), -1)
        null = self.null(null_input).transpose(-1, -2).unsqueeze(-1)
        return torch.cat((scores, null), -1).softmax(-1)[..., :-1]

    def form_query(self, current: Tensor, past_value: Tensor, attention: Tensor,
                   gap: Tensor, horizon_embedding: Tensor, language: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        gap = gap.to(current.dtype)
        matched = self.matched((attention @ past_value).transpose(-3, -2).flatten(-2))
        relative = self.relative_read(relative_correspondence(attention).transpose(-3, -2).flatten(-2))
        mass = attention.sum(-1).transpose(-1, -2)
        fields = (current, matched, current - matched, relative, mass,
                  horizon_embedding.expand_as(current), language[:, None, :].expand_as(current),
                  (gap[:, None, None] / 5).expand(-1, self.horizon, 1))
        query = self.horizon_query(self.query_input(torch.cat(fields, -1)), self.h_positions)
        return query, matched, relative, mass

    def _pair_messages(self, states: Tensor, normalized: Tensor, content: Tensor, values: Tensor,
                       visual_key: Tensor, visual_value: Tensor, visual_mask: Tensor,
                       current: Tensor, past: Tensor, gap: Tensor,
                       horizon_embedding: Tensor, language: Tensor) -> Tensor:
        attention = self.correspondence(normalized[current], content[current], content[past], gap)
        query, matched, relative, mass = self.form_query(
            states[current], values[past], attention, gap, horizon_embedding, language[current])
        z_past = self.visual_read.read_projected(query + self.visual_roles[0], visual_key[past],
                                               visual_value[past], visual_mask[past, None, None, :])
        z_now = self.visual_read.read_projected(query + self.visual_roles[1], visual_key[current],
                                              visual_value[current], visual_mask[current, None, None, :])
        gap = gap.to(states.dtype)
        fields = (query, states[current], matched, relative, mass,
                  (gap[:, None, None] / 5).expand(-1, self.horizon, 1), z_past, z_now, z_now - z_past)
        return self.message(torch.cat(fields, -1))

    def aggregate(self, states: Tensor, times: Tensor, messages: Tensor,
                  current: Tensor, past: Tensor, slots: Tensor) -> Tensor:
        """Only the <=4 neighbor slots recur; all real (t,h) cells are batched."""
        initial = self.initial(self.initial_norm(states)).tanh()
        hidden = initial
        for slot in range(min(self.radius, len(states) - 1)):
            selected = slots == slot
            targets, sources = current[selected], past[selected]
            gap = (times[targets] - times[sources]) / 5
            previous_gap = (times[sources] - times[sources - 1]) / 5 if slot else torch.zeros_like(gap)
            gamma = torch.stack((gap, previous_gap, torch.full_like(gap, float(slot > 0))), -1)
            inputs = self.message_norm(messages[selected]) + self.time_input(gamma.to(states.dtype))[:, None, :]
            updated = self.gru(inputs.flatten(0, 1), hidden[targets].flatten(0, 1)).reshape(-1, self.horizon, self.width)
            hidden = hidden.index_copy(0, targets, updated)
        updated = states + self.neighbor_output(hidden - initial)
        return updated + self.ffn(self.ffn_norm(updated))

    def forward(self, states: Tensor, times: Tensor, language: Tensor,
                visual_tokens: Tensor, visual_mask: Tensor, horizon_embedding: Tensor) -> Tensor:
        normalized = self.norm(states)
        content, values = self._heads(self.content(normalized)), self._heads(self.value(states))
        current, past, slots = past_edges(len(states), self.radius, states.device)
        if not len(current):
            return states + self.ffn(self.ffn_norm(states))
        # These projections occur once per frame/group, outside the edge loop.
        key, value = self.visual_read.project_memory(visual_tokens, visual_tokens)
        chunks = []
        for start in range(0, len(current), self.edge_chunk):
            late, early = current[start:start + self.edge_chunk], past[start:start + self.edge_chunk]
            args = (states, normalized, content, values, key, value, visual_mask, late, early,
                    times[late] - times[early], horizon_embedding, language)
            if self.activation_checkpoint and torch.is_grad_enabled():
                chunks.append(checkpoint(self._pair_messages, *args, use_reentrant=False))
            else:
                chunks.append(self._pair_messages(*args))
        return self.aggregate(states, times, torch.cat(chunks), current, past, slots)
