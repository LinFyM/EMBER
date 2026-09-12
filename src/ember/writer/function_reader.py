"""Training-only local action flow matching from shared four-frame video evidence."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ember.writer.attention import CompilerBlock, position_encoding


class LocalActionReader(nn.Module):
    """Predict local velocity without owning the deployment Writer or its Compiler.

    All four frames' task tokens form one memory. The caller supplies fixed
    relative-time routing for ordered evidence, or zeros for the static set
    reference. Action positions identify outputs, never particular input frames.
    """

    def __init__(self, width: int = 256, heads: int = 8, depth: int = 2,
                 horizon: int = 15, action_dim: int = 7) -> None:
        super().__init__()
        if min(width, heads, depth, horizon, action_dim) <= 0 or width % heads:
            raise ValueError("local reader dimensions must be positive and width divisible by heads")
        self.width, self.horizon, self.action_dim = width, horizon, action_dim
        self.action_projection = nn.Linear(action_dim, width)
        self.routing_projection = nn.Linear(width, width, bias=False)
        self.blocks = nn.ModuleList([CompilerBlock(width, heads) for _ in range(depth)])
        self.output = nn.Linear(width, action_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, noisy_actions: Tensor, time: Tensor, memory: Tensor,
                routing: Tensor, prior: Tensor) -> Tensor:
        if (noisy_actions.ndim != 3 or noisy_actions.shape[1:] != (self.horizon, self.action_dim)
                or not len(noisy_actions) or time.shape != noisy_actions.shape[:1]):
            raise ValueError("local action queries require [B,horizon,action_dim] and flow time [B]")
        if (memory.ndim != 2 or memory.shape[-1] != self.width or not len(memory)
                or routing.shape != memory.shape or prior.shape != (1, len(memory))):
            raise ValueError("local video memory and routing require [N,width] and prior [1,N]")
        query = self.action_projection(noisy_actions)
        action_positions = torch.arange(self.horizon, device=query.device)
        query = (query + position_encoding(action_positions, self.width, query.dtype)[None]
                 + position_encoding(time, self.width, query.dtype)[:, None, :])
        batch = len(query)
        # Fixed input routing is projected only by this head. Zero routing stays zero.
        routes = self.routing_projection(routing)[None].expand(batch, -1, -1)
        values = memory[None].expand(batch, -1, -1)
        attention_prior = prior.reshape(1, 1, 1, -1).expand(batch, 1, 1, -1)
        for block in self.blocks:
            query = block(query, values, routes, attention_prior)
        return self.output(query)
