"""Independent native-channel linear factor heads for every target/rank slot."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract, identity_lora_state


class _FactorGroup(nn.Module):
    """Batch equal native shapes without sharing their output parameters."""

    def __init__(self, indices: list[int], contract: LoRAContract, identity: dict, width: int) -> None:
        super().__init__()
        targets = [contract.targets[index] for index in indices]
        self.names = tuple(target.name for target in targets)
        self.register_buffer("indices", torch.tensor(indices), persistent=False)
        self.register_buffer("identity_a", torch.stack([identity[name + LORA_A_SUFFIX] for name in self.names]))
        shape = (len(targets), contract.rank)
        self.a_weight = nn.Parameter(torch.zeros(*shape, targets[0].in_features, width))
        self.b_weight = nn.Parameter(torch.zeros(*shape, targets[0].out_features, width))


class NativeFactorLoRADecoder(nn.Module):
    """Paired codes produce delta-A/B; public A0 and zero heads start at identity.

    Each target/rank learns its own fixed native subspace. Its code coefficients
    depend on the whole input video set; no task-specific parameter is stored.
    """

    def __init__(self, contract: LoRAContract, width: int, factor_width: int) -> None:
        super().__init__()
        self.contract = contract
        self.a_code = nn.Linear(width, factor_width, bias=False)
        self.b_code = nn.Linear(width, factor_width, bias=False)
        grouped: dict[tuple[int, int], list[int]] = {}
        for index, target in enumerate(contract.targets):
            grouped.setdefault((target.in_features, target.out_features), []).append(index)
        identity = identity_lora_state(contract)
        self.groups = nn.ModuleList([
            _FactorGroup(indices, contract, identity, factor_width)
            for indices in grouped.values()
        ])

    def forward(self, codes: Tensor) -> dict[str, Tensor]:
        if codes.shape != (len(self.contract.targets), self.contract.rank, self.a_code.in_features):
            raise ValueError("factor codes must cover every target and rank slot at the configured width")
        a_code, b_code = self.a_code(codes), self.b_code(codes)
        generated = {}
        for group in self.groups:
            a = group.identity_a + torch.einsum("grp,grnp->grn", a_code[group.indices], group.a_weight)
            b = torch.einsum("grp,grnp->grn", b_code[group.indices], group.b_weight).transpose(-1, -2)
            for index, name in enumerate(group.names):
                generated[name + LORA_A_SUFFIX] = a[index]
                generated[name + LORA_B_SUFFIX] = b[index]
        return {target.name + suffix: generated[target.name + suffix]
                for target in self.contract.targets for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX)}
