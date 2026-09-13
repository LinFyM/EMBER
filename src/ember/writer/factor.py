"""Shared native-shape heads for one complete, freely generated task LoRA."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract, identity_lora_state


def _head(width: int, hidden: int, output: int) -> nn.Sequential:
    head = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, output))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class _FactorGroup(nn.Module):
    """One A/B map per native shape, shared across target and rank queries."""

    def __init__(self, indices, contract: LoRAContract, width: int, hidden: int, identity) -> None:
        super().__init__()
        targets = [contract.targets[index] for index in indices]
        self.names = tuple(target.name for target in targets)
        self.register_buffer("indices", torch.tensor(indices), persistent=False)
        self.register_buffer("identity_a", torch.stack([identity[name + LORA_A_SUFFIX] for name in self.names]))
        self.a = _head(width, hidden, targets[0].in_features)
        self.b = _head(width, hidden, targets[0].out_features)

    def forward(self, codes: Tensor) -> tuple[Tensor, Tensor]:
        a = self.identity_a + self.a(codes).float()
        b = self.b(codes).transpose(-1, -2).float()
        return a, b


class FactorLoRADecoder(nn.Module):
    """Legal identity plus free A/B; no task labels, fixed-X span, or calibration."""

    def __init__(self, contract: LoRAContract, width: int, hidden: int) -> None:
        super().__init__()
        self.contract, self.width = contract, width
        grouped = {}
        for index, target in enumerate(contract.targets):
            grouped.setdefault((target.in_features, target.out_features), []).append(index)
        identity = identity_lora_state(contract)
        self.groups = nn.ModuleList([
            _FactorGroup(indices, contract, width, hidden, identity) for indices in grouped.values()
        ])

    def forward(self, codes: Tensor) -> dict[str, Tensor]:
        if codes.shape != (len(self.contract.targets), self.contract.rank, self.width):
            raise ValueError("factor codes must cover every target/rank in the native contract")
        generated = {}
        for group in self.groups:
            a, b = group(codes[group.indices])
            for index, name in enumerate(group.names):
                generated[name + LORA_A_SUFFIX] = a[index]
                generated[name + LORA_B_SUFFIX] = b[index]
        return {target.name + suffix: generated[target.name + suffix]
                for target in self.contract.targets for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX)}
