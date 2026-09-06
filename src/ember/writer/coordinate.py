"""Shared coordinate-conditioned scalar A/B decoder for a complete LoRA."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract, identity_lora_state


class _CoordinateGroup(nn.Module):
    """Targets with one native (input, output) shape share a batched layout."""

    def __init__(self, indices: list[int], contract: LoRAContract, identity: dict, width: int) -> None:
        super().__init__()
        targets = [contract.targets[index] for index in indices]
        self.names = tuple(target.name for target in targets)
        self.register_buffer("indices", torch.tensor(indices), persistent=False)
        self.register_buffer("identity_a", torch.stack([identity[name + LORA_A_SUFFIX] for name in self.names]))
        self.a_coordinates = nn.Parameter(torch.empty(len(targets), targets[0].in_features, width))
        self.b_coordinates = nn.Parameter(torch.empty(len(targets), targets[0].out_features, width))
        # These are additive GELU coordinates, not token embeddings followed by
        # LayerNorm. Unit-scale offsets expose distinct native-channel slopes;
        # zero scalar readouts still give the exact public identity LoRA.
        nn.init.normal_(self.a_coordinates)
        nn.init.normal_(self.b_coordinates)


class CoordinateLoRADecoder(nn.Module):
    """A0 is public/fixed; paired codes jointly generate all delta-A and B."""

    def __init__(
        self, contract: LoRAContract, width: int, coordinate_width: int,
        coordinate_chunk: int, activation_checkpoint: bool,
    ) -> None:
        super().__init__()
        self.contract = contract
        self.coordinate_chunk = coordinate_chunk
        self.activation_checkpoint = activation_checkpoint
        self.a_code = nn.Linear(width, coordinate_width, bias=False)
        self.b_code = nn.Linear(width, coordinate_width, bias=False)
        self.a_readout = nn.Parameter(torch.zeros(coordinate_width))
        self.b_readout = nn.Parameter(torch.zeros(coordinate_width))
        grouped: dict[tuple[int, int], list[int]] = {}
        for index, target in enumerate(contract.targets):
            grouped.setdefault((target.in_features, target.out_features), []).append(index)
        identity = identity_lora_state(contract)
        self.groups = nn.ModuleList([
            _CoordinateGroup(indices, contract, identity, coordinate_width)
            for indices in grouped.values()
        ])

    @staticmethod
    def _tile(code: Tensor, coordinates: Tensor, readout: Tensor) -> Tensor:
        hidden = F.gelu(code.unsqueeze(-2) + coordinates.unsqueeze(-3))
        return hidden @ readout

    def _decode(self, code: Tensor, coordinates: Tensor, readout: Tensor) -> Tensor:
        tiles = []
        for start in range(0, coordinates.shape[1], self.coordinate_chunk):
            arguments = (code, coordinates[:, start:start + self.coordinate_chunk], readout)
            if self.activation_checkpoint and torch.is_grad_enabled():
                tiles.append(checkpoint(self._tile, *arguments, use_reentrant=False))
            else:
                tiles.append(self._tile(*arguments))
        return torch.cat(tiles, dim=-1)

    def forward(self, codes: Tensor) -> dict[str, Tensor]:
        if codes.shape[:2] != (len(self.contract.targets), self.contract.rank):
            raise ValueError("coordinate codes must cover every target and rank slot")
        a_code, b_code = self.a_code(codes), self.b_code(codes)
        generated = {}
        for group in self.groups:
            a = group.identity_a + self._decode(a_code[group.indices], group.a_coordinates, self.a_readout)
            b = self._decode(b_code[group.indices], group.b_coordinates, self.b_readout).transpose(-1, -2)
            for index, name in enumerate(group.names):
                generated[name + LORA_A_SUFFIX] = a[index]
                generated[name + LORA_B_SUFFIX] = b[index]
        return {target.name + suffix: generated[target.name + suffix]
                for target in self.contract.targets for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX)}
