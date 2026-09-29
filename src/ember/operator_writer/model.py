"""One public rank-128 LoRA and a video-written B residual in its actual A space."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.pi05_lora import Pi05LoRAContract
from ember.writer.model import DirectLoRAParameters
from . import change_clock as clock_contract


class TargetWrite(nn.Module):
    """One target's no-bias address/value maps and FP32 delta-rule memory."""

    def __init__(self, in_width: int, out_width: int, *, change_clock: bool = False) -> None:
        super().__init__()
        self.change_clock = change_clock
        self.p = nn.Linear(128, 256, bias=False)
        self.c = nn.Linear(1024, 256, bias=False)
        self.d = nn.Linear(1024, 256, bias=False)
        self.o = nn.Linear(256, out_width, bias=False)
        nn.init.zeros_(self.o.weight)

    def forward(self, address: torch.Tensor, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or h.shape != (x.shape[0], 50, 1024) or x.shape[-1] != address.shape[1]:
            raise ValueError("native target input or full suffix changed")
        with torch.autocast(device_type=x.device.type, enabled=False):
            x, h, address = x.float(), h.float(), address.float()
            normalized = h * torch.rsqrt(h.square().mean(dim=-1, keepdim=True) + 1e-6)
            memory = torch.zeros(self.o.out_features, 128, dtype=torch.float32, device=x.device)
            for t in range(len(x) - 1):
                key = F.normalize(F.linear(x[t], address), dim=-1, eps=1e-6).T
                change = normalized[t + 1] - normalized[t]
                value = self.o(F.gelu(self.p(key.T) + self.c(normalized[t])) * self.d(change)).T
                erase = memory @ key
                if self.change_clock:
                    d = torch.linalg.vector_norm(change, dim=-1) / (1024 ** 0.5)
                    erase = erase * (-torch.expm1(-d))[None, :]
                memory = memory + (value - erase) @ key.T / 50
            return memory


class OperatorReadWrite(nn.Module):
    """T uses execution A for teacher keys; U has an independent cloned key S."""

    def __init__(self, contract: Pi05LoRAContract, template: dict[str, torch.Tensor], mode: str) -> None:
        super().__init__()
        if mode not in {"T", "U", clock_contract.MODE} or len(contract.targets) != 38 or contract.rank != 128:
            raise ValueError("bounded operator mode or complete rank changed")
        validate_lora_state(template, contract)
        self.mode = mode
        self.names = tuple(target.name for target in contract.targets)
        self.common = DirectLoRAParameters(template)
        common = self.common()
        # S is a separate parameter only in U, initialized without consuming RNG.
        self.separate_keys = (nn.ParameterList([
            nn.Parameter(common[name + LORA_A_SUFFIX].detach().clone()) for name in self.names
        ]) if mode == "U" else None)
        self.writes = nn.ModuleList()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            for target in contract.targets:
                self.writes.append(TargetWrite(target.in_features, target.out_features,
                                               change_clock=mode == clock_contract.MODE))
        self.register_buffer("probe", torch.randn(50, 32, generator=torch.Generator(device="cpu").manual_seed(1729)),
                             persistent=True)

    def public_state(self) -> dict[str, torch.Tensor]:
        return self.common()

    def forward(self, native_inputs: dict[str, torch.Tensor], h: torch.Tensor) -> dict[str, torch.Tensor]:
        common = self.public_state()
        if set(native_inputs) != set(self.names):
            raise ValueError("native teaching lost a complete LoRA target")
        result = dict(common)
        for index, name in enumerate(self.names):
            key = (self.separate_keys[index] if self.separate_keys is not None
                   else common[name + LORA_A_SUFFIX])
            written = self.writes[index](key, native_inputs[name], h)
            result[name + LORA_B_SUFFIX] = common[name + LORA_B_SUFFIX] + written
        return result
