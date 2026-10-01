"""Shared identity-initialized LoRA parameters used by the canonical Writer."""
from __future__ import annotations
from typing import Mapping
import torch
from ember.writer.errors import WriterModelError

class DirectLoRAParameters(torch.nn.Module):
    """One shared, fresh identity-initialized LoRA parameterization without a compiler."""

    def __init__(self, template_state: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        if not template_state:
            raise WriterModelError("direct LoRA baseline requires a complete identity template")
        self.names = tuple(sorted(template_state))
        self.values = torch.nn.ParameterList(
            torch.nn.Parameter(template_state[name].detach().clone().contiguous())
            for name in self.names
        )

    def forward(self) -> dict[str, torch.Tensor]:
        return dict(zip(self.names, self.values, strict=True))
