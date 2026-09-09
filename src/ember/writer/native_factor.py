"""Complete native A/B factors with an isolated within-target rank binding."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract, identity_lora_state


class _NativeGroup(nn.Module):
    """Batch targets of identical native shape without sharing their D tensors."""

    def __init__(self, indices: list[int], contract: LoRAContract, identity: dict, width: int, *, share_ranks: bool) -> None:
        super().__init__()
        targets = [contract.targets[index] for index in indices]
        self.names = tuple(target.name for target in targets)
        self.register_buffer("indices", torch.tensor(indices), persistent=False)
        self.register_buffer("identity_a", torch.stack([identity[name + LORA_A_SUFFIX] for name in self.names]))
        rank_owners = 1 if share_ranks else contract.rank
        self.a_factors = nn.Parameter(torch.zeros(len(targets), rank_owners, targets[0].in_features, width))
        self.b_factors = nn.Parameter(torch.zeros(len(targets), rank_owners, targets[0].out_features, width))

    def forward(self, a_code: Tensor, b_code: Tensor) -> tuple[Tensor, Tensor]:
        a = self.identity_a + torch.matmul(self.a_factors, a_code[..., None]).squeeze(-1)
        b = torch.matmul(self.b_factors, b_code[..., None]).squeeze(-1).transpose(-1, -2)
        return a, b


class NativeFactorLoRADecoder(nn.Module):
    """A0 + D_A GELU(U_A c), D_B GELU(U_B c); no task dictionary."""

    def __init__(self, contract: LoRAContract, width: int, factor_width: int = 256, *, share_ranks: bool = False) -> None:
        super().__init__()
        self.contract, self.width = contract, width
        self.a_code, self.b_code = (nn.Linear(width, factor_width, bias=False) for _ in range(2))
        grouped: dict[tuple[int, int], list[int]] = {}
        for index, target in enumerate(contract.targets):
            grouped.setdefault((target.in_features, target.out_features), []).append(index)
        identity = identity_lora_state(contract)
        self.groups = nn.ModuleList([
            _NativeGroup(indices, contract, identity, factor_width, share_ranks=share_ranks)
            for indices in grouped.values()
        ])

    def forward(self, codes: Tensor) -> dict[str, Tensor]:
        if codes.shape != (len(self.contract.targets), self.contract.rank, self.width):
            raise ValueError("native codes must cover every target and rank slot with the configured width")
        a_code, b_code = F.gelu(self.a_code(codes)), F.gelu(self.b_code(codes))
        generated = {}
        for group in self.groups:
            a, b = group(a_code[group.indices], b_code[group.indices])
            for index, name in enumerate(group.names):
                generated[name + LORA_A_SUFFIX] = a[index]
                generated[name + LORA_B_SUFFIX] = b[index]
        return {target.name + suffix: generated[target.name + suffix]
                for target in self.contract.targets for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX)}
