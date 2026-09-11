"""Training-only execution-state queries into the shared video representation."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ember.writer.attention import Attention


class ExecutionVideoReader(nn.Module):
    """Predict the full source velocity residual; never an input to the Compiler.

    A constant video read can support a common state-dependent correction. This
    is an admitted competing solution, not structural proof of video use.
    """

    def __init__(self, width: int = 256, heads: int = 8, query_width: int = 1024,
                 action_width: int = 32) -> None:
        super().__init__()
        self.query_width, self.action_width = query_width, action_width
        self.query_norm = nn.LayerNorm(query_width)
        self.query_projection = nn.Linear(query_width, width)
        self.memory_norm = nn.LayerNorm(width)
        self.read = Attention(width, heads)
        self.action_query = nn.Linear(query_width, width, bias=False)
        self.content = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, action_width, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, query: Tensor, base: Tensor, memory: Tensor, prior: Tensor) -> Tensor:
        if (query.ndim != 3 or query.shape[-1] != self.query_width
                or base.shape != (*query.shape[:2], self.action_width)
                or memory.ndim != 2
                or prior.shape != (1, len(memory))):
            raise ValueError("execution reader lost its native query or shared video memory")
        normalized = self.query_norm(query)
        values = self.memory_norm(memory)
        keys, contents = self.read.project_memory(values, values)
        # Shared video memory is projected once, then broadcast without copying
        # to equal-rank SDPA operands for the physical action-query microbatch.
        keys = keys[None].expand(len(query), -1, -1, -1)
        contents = contents[None].expand(len(query), -1, -1, -1)
        read = self.read.read_projected(self.query_projection(normalized), keys, contents, prior)
        residual = self.output(F.gelu(self.action_query(normalized)) * self.content(read))
        return base + residual.float()


def distillation_weight(config: dict, step: int) -> float:
    if not config["enabled"]:
        return 0.0
    start, end, maximum = config["distill_start"], config["distill_end"], config["distill_max"]
    if not (0 <= start < end and 0 <= maximum < 1 and step >= 0):
        raise ValueError("invalid registered distillation schedule")
    return float(maximum) * min(1.0, max(0.0, (step - start) / (end - start)))
