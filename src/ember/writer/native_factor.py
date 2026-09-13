"""Forward-only native input/correction pairing into one complete rank-r LoRA."""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract


class _NativeGroup(nn.Module):
    """Batch equal-shaped targets while retaining their own native-coordinate maps."""

    def __init__(self, indices: list[int], contract: LoRAContract, width: int, key_width: int,
                 output_units: Sequence[float]) -> None:
        super().__init__()
        targets = [contract.targets[index] for index in indices]
        self.names, self.positions = tuple(target.name for target in targets), tuple(indices)
        self.register_buffer("indices", torch.tensor(indices), persistent=False)
        self.register_buffer("output_units", torch.tensor([output_units[index] for index in indices])[:, None, None])
        self.key_factors = nn.Parameter(torch.empty(len(targets), key_width, targets[0].in_features))
        for weight in self.key_factors:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.key_norm = nn.LayerNorm(key_width)
        # B=0 gives identity without differentiating an SVD at a zero matrix.
        self.b_factors = nn.Parameter(torch.zeros(len(targets), targets[0].out_features, width))

    def forward(self, route: Tensor, content: Tensor, videos: Sequence[Sequence[Tensor]]) -> tuple[Tensor, Tensor]:
        a = None
        for video in videos:
            native = torch.stack([video[index].flatten(0, 1) for index in self.positions]).to(
                device=route.device, dtype=torch.float32, non_blocking=True)
            keys = self.key_norm(torch.matmul(native, self.key_factors.transpose(-1, -2)))
            coefficients = torch.tanh(torch.matmul(route, keys.transpose(-1, -2)) / math.sqrt(keys.shape[-1]))
            # Keep the native X Value in its source coordinate/precision. The
            # signed coefficients and the corresponding X position stay paired.
            with torch.autocast(route.device.type, enabled=False):
                part = torch.matmul(coefficients.float(), native) / (native.shape[1] * len(videos))
            a = part if a is None else a + part
        # Fix the arbitrary A/B row gauge. Zero rows stay zero; physical X is
        # unchanged and still paired with its own correction coefficient.
        a = F.normalize(a, dim=-1)
        b = torch.matmul(self.b_factors, F.gelu(content).transpose(-1, -2)).float() * self.output_units
        return a, b


class NativeFactorLoRADecoder(nn.Module):
    """Unit-row A from RX/N and B in frozen shared training-target units."""

    def __init__(self, contract: LoRAContract, width: int, factor_width: int = 256, *,
                 output_units: Sequence[float]) -> None:
        super().__init__()
        self.contract, self.width = contract, width
        if len(output_units) != len(contract.targets) or any(not math.isfinite(value) or value <= 0 for value in output_units):
            raise ValueError("every native target requires a positive shared output unit")
        self.route_projection = nn.Linear(width, factor_width, bias=False)
        self.route_norm = nn.LayerNorm(factor_width)
        grouped: dict[tuple[int, int], list[int]] = {}
        for index, target in enumerate(contract.targets):
            grouped.setdefault((target.in_features, target.out_features), []).append(index)
        self.groups = nn.ModuleList([
            _NativeGroup(indices, contract, width, factor_width, output_units) for indices in grouped.values()
        ])

    def _validate(self, codes: Tensor, videos: Sequence[Sequence[Tensor]]) -> None:
        if codes.shape != (len(self.contract.targets), self.contract.rank, self.width):
            raise ValueError("native codes must cover every target/rank with the configured width")
        if not videos:
            raise ValueError("native input pairing requires one or more complete videos")
        for video in videos:
            if len(video) != len(self.contract.targets):
                raise ValueError("native input pairing lost a target")
            layout = video[0].shape[:2]
            for value, target in zip(video, self.contract.targets, strict=True):
                if (value.ndim != 3 or min(value.shape[:2]) <= 0 or value.shape[:2] != layout
                        or value.shape[-1] != target.in_features or value.requires_grad):
                    raise ValueError("native inputs must be frozen same-frame full-horizon source tensors")

    def forward(self, codes: Tensor, native_inputs: Sequence[Sequence[Tensor]]) -> dict[str, Tensor]:
        self._validate(codes, native_inputs)
        route = self.route_norm(self.route_projection(codes))
        generated = {}
        for group in self.groups:
            a, b = group(route[group.indices], codes[group.indices], native_inputs)
            for index, name in enumerate(group.names):
                generated[name + LORA_A_SUFFIX] = a[index]
                generated[name + LORA_B_SUFFIX] = b[index]
        return {target.name + suffix: generated[target.name + suffix]
                for target in self.contract.targets for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX)}
