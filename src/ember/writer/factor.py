"""A supervised local field and its exact contraction into one complete LoRA."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract


class _FieldGroup(nn.Module):
    """Share local prediction and output-basis maps across one native shape."""

    def __init__(self, inputs: int, outputs: int, width: int, hidden: int, rank: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(inputs, width, bias=False)
        self.input_norm, self.local_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.coefficients = nn.Sequential(
            nn.Linear(2 * width, hidden), nn.GELU(), nn.Linear(hidden, rank),
        )
        self.basis = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, outputs))
        # r=0 and U!=0 is a legal complete-LoRA identity. The first update
        # learns r; all upstream modules receive its actual credit thereafter.
        nn.init.zeros_(self.coefficients[-1].weight)
        nn.init.zeros_(self.coefficients[-1].bias)

    def forward(self, codes: Tensor, local: Tensor, inputs: Tensor, indices: Tensor,
                unit: Tensor, chunk_size: int) -> tuple[Tensor, Tensor, Tensor]:
        x = inputs.to(local.device, non_blocking=True)
        coefficients = []
        context = codes.mean(0)
        for start in range(0, len(local), chunk_size):
            address = self.input_norm(self.input_projection(x[start:start + chunk_size]))
            condition = self.local_norm(local[start:start + chunk_size] + context)
            coefficients.append(self.coefficients(torch.cat((condition, address), -1)))
        r = torch.cat(coefficients).float()
        # Keep the native physical scale of X and the actual field. There is
        # no separately normalized A, free parameter head, or auxiliary exit.
        b = self.basis(codes).transpose(0, 1).float() * unit
        with torch.autocast(local.device.type, enabled=False):
            a = torch.einsum("thr,thd->rd", r, x.float()) / len(local)
            fields = torch.einsum("or,thr->tho", b, r[indices.to(local.device)])
        return a, b, fields


class LocalFieldLoRADecoder(nn.Module):
    """Every emitted local prediction is the same U r used in B A."""

    def __init__(self, contract: LoRAContract, width: int, hidden: int, unit: float,
                 *, chunk_size: int, activation_checkpoint: bool) -> None:
        super().__init__()
        self.contract, self.width = contract, width
        self.chunk_size, self.activation_checkpoint = chunk_size, activation_checkpoint
        self.register_buffer("field_unit", torch.tensor(unit, dtype=torch.float32))
        shapes = tuple(dict.fromkeys((target.in_features, target.out_features) for target in contract.targets))
        self.groups = nn.ModuleList([
            _FieldGroup(inputs, outputs, width, hidden, contract.rank) for inputs, outputs in shapes
        ])
        self.group_indices = tuple(shapes.index((target.in_features, target.out_features))
                                   for target in contract.targets)

    def forward(self, codes: Tensor, local: Tensor, native: Mapping[str, Tensor],
                indices: Tensor | None = None) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        if codes.shape != (len(self.contract.targets), self.contract.rank, self.width):
            raise ValueError("field codes must cover every native target/rank")
        if local.ndim != 3 or not len(local) or local.shape[-1] != self.width:
            raise ValueError("the local field requires complete frame/horizon evidence")
        if set(native) != {target.name for target in self.contract.targets}:
            raise ValueError("local contraction requires every actual native input")
        selected = torch.empty(0, dtype=torch.long) if indices is None else indices
        if (selected.ndim != 1 or selected.dtype != torch.long or len(selected.unique()) != len(selected)
                or (len(selected) and (selected.min() < 0 or selected.max() >= len(local)))):
            raise ValueError("field supervision indices must be distinct valid frame ordinals")
        state, fields = {}, {}
        for index, (target, group_index) in enumerate(zip(self.contract.targets, self.group_indices, strict=True)):
            x = native[target.name]
            if x.shape != (*local.shape[:2], target.in_features) or x.requires_grad:
                raise ValueError("bare native input must preserve its frozen same-position coordinates")
            group = self.groups[group_index]
            args = (codes[index], local, x, selected, self.field_unit, self.chunk_size)
            if self.activation_checkpoint and torch.is_grad_enabled():
                a, b, field = checkpoint(group, *args, use_reentrant=False)
            else:
                a, b, field = group(*args)
            state[target.name + LORA_A_SUFFIX], state[target.name + LORA_B_SUFFIX] = a, b
            if indices is not None:
                fields[target.name] = field
        return state, fields
