"""Canonical public/native Writer and its registered conditional A/B compiler."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.pi05_lora import Pi05LoRAContract
from ember.writer.model import DirectLoRAParameters
from . import change_clock as clock_contract
from .value_context import ValueContext
from .conditional_read_write import CausalInterpreter, ConditionalTarget
from . import prefix_change as prefix_contract
from torch.utils.checkpoint import checkpoint


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
        self.u: nn.Linear | None = None
        self.e: nn.Linear | None = None

    def forward(self, address: torch.Tensor, x: torch.Tensor, h: torch.Tensor,
                context: torch.Tensor | None = None, prefix_change=None, *, passive=False) -> torch.Tensor:
        if x.ndim != 3 or h.shape != (x.shape[0], 50, 1024) or x.shape[-1] != address.shape[1]:
            raise ValueError("native target input or full suffix changed")
        if (self.u is None) != (context is None) or context is not None and context.shape != (len(x), 50, 256):
            raise ValueError("target Value context or its projection is incomplete")
        if (self.e is None) != (prefix_change is None) or prefix_change is not None and prefix_change.shape != (len(x)-1, 50, 1024):
            raise ValueError("target prefix response or E is incomplete")
        with torch.autocast(device_type=x.device.type, enabled=False):
            x, h, address = x.float(), h.float(), address.float()
            normalized = h * torch.rsqrt(h.square().mean(dim=-1, keepdim=True) + 1e-6)
            memory = torch.zeros(self.o.out_features, 128, dtype=torch.float32, device=x.device)
            extra_memory = torch.zeros_like(memory) if passive else None
            statistics = {"D_dH_norm": [], "E_dP_norm": []} if passive else None
            for t in range(len(x) - 1):
                key = F.normalize(F.linear(x[t], address), dim=-1, eps=1e-6).T
                change = normalized[t + 1] - normalized[t]
                gate = self.p(key.T) + self.c(normalized[t])
                if context is not None:
                    gate = gate + self.u(context[t].float())
                gate = F.gelu(gate)
                dynamic = self.d(change)
                extra = self.e(prefix_change[t].float()) if self.e is not None else None
                value = self.o(gate * (dynamic if extra is None else dynamic + extra)).T
                erase = memory @ key
                if self.change_clock:
                    d = torch.linalg.vector_norm(change, dim=-1) / (1024 ** 0.5)
                    erase = erase * (-torch.expm1(-d))[None, :]
                memory = memory + (value - erase) @ key.T / 50
                if passive:
                    extra_value = self.o(gate * extra).T
                    extra_memory = extra_memory + (extra_value - extra_memory @ key) @ key.T / 50
                    statistics["D_dH_norm"].append(dynamic.norm(dim=-1).detach().cpu())
                    statistics["E_dP_norm"].append(extra.norm(dim=-1).detach().cpu())
            if passive:
                self.last_prefix_statistics = {**{k: torch.stack(v) for k, v in statistics.items()},
                    "M_original_norm": float((memory-extra_memory).norm()),
                    "M_prefix_norm": float(extra_memory.norm()), "M_total_norm": float(memory.norm())}
            return memory


class OperatorReadWrite(nn.Module):
    """Own one complete LoRA, preserving sealed historical checkpoint layouts."""

    def __init__(self, contract: Pi05LoRAContract, template: dict[str, torch.Tensor], mode: str) -> None:
        super().__init__()
        if mode not in {"T", "U", "context", "self_read", "conditional_read_write", clock_contract.MODE, prefix_contract.MODE} or len(contract.targets) != 38 or contract.rank != 128:
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
        self.interpreter: CausalInterpreter | None = None
        self.conditional_targets = nn.ModuleList()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            for target in (() if mode == "conditional_read_write" else contract.targets):
                self.writes.append(TargetWrite(target.in_features, target.out_features,
                                               change_clock=mode == clock_contract.MODE))
        self.value_context: ValueContext | None = None
        if mode == prefix_contract.MODE:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(7)
                for write in self.writes:
                    write.e = nn.Linear(1024, 256, bias=False, device="cpu")
                    nn.init.zeros_(write.e.weight)
        if mode in ("context", "self_read"):
            # Complete the old P/C/D/O sequence before an independent new RNG scope.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(7)
                self.value_context = ValueContext()
                for write in self.writes:
                    write.u = nn.Linear(256, 256, bias=False, device="cpu")
                    nn.init.zeros_(write.u.weight)
        if mode == "conditional_read_write":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(7)
                self.interpreter = CausalInterpreter()
                self.conditional_targets.extend(
                    ConditionalTarget(target.in_features, target.out_features) for target in contract.targets)
        self.register_buffer("probe", torch.randn(50, 32, generator=torch.Generator(device="cpu").manual_seed(1729)),
                             persistent=True)

    def public_state(self) -> dict[str, torch.Tensor]:
        return self.common()

    def forward(self, native_inputs: dict[str, torch.Tensor], h: torch.Tensor,
                frame_indices=None, *, capture_mechanism: bool = False,
                target_executor=None, prefix_change=None, capture_prefix_stats=False) -> dict[str, torch.Tensor]:
        common = self.public_state()
        if set(native_inputs) != set(self.names):
            raise ValueError("native teaching lost a complete LoRA target")
        if self.interpreter is not None:
            c, d = self.interpreter(h, frame_indices)
            if target_executor is not None:
                if capture_mechanism:
                    raise ValueError("mechanism readout uses the complete local compiler")
                return target_executor(self, native_inputs, h, c, d)
            result, targets = {}, {}
            for name, unit in zip(self.names, self.conditional_targets, strict=True):
                a0, b0 = common[name + LORA_A_SUFFIX], common[name + LORA_B_SUFFIX]
                inputs = (a0, b0, native_inputs[name], h, c, d)
                a, b, s, m = (checkpoint(unit, *inputs, use_reentrant=False, preserve_rng_state=False)
                              if torch.is_grad_enabled() else unit(*inputs))
                result[name + LORA_A_SUFFIX], result[name + LORA_B_SUFFIX] = a, b
                if capture_mechanism:
                    targets[name] = {"A0": a0, "B0": b0, "S": s, "M": m}
            if capture_mechanism:
                self.last_mechanism = {"c": c, "d": d, "targets": targets}
            return result
        context = self.value_context(h, frame_indices) if self.value_context is not None else None
        if capture_prefix_stats:
            if self.mode != prefix_contract.MODE or torch.is_grad_enabled():
                raise ValueError("passive prefix statistics require authorized frozen materialization")
            normalized = prefix_contract.rms(h)
            self.last_prefix_statistics = {
                "dP_norm": prefix_change.float().norm(dim=-1).detach().cpu(),
                "dH_norm": (normalized[1:]-normalized[:-1]).norm(dim=-1).detach().cpu(),
                "targets": {}}
        result = dict(common)
        for index, name in enumerate(self.names):
            key = (self.separate_keys[index] if self.separate_keys is not None
                   else common[name + LORA_A_SUFFIX])
            written = self.writes[index](key, native_inputs[name], h, context,
                                         prefix_change, passive=capture_prefix_stats)
            if capture_prefix_stats:
                self.last_prefix_statistics["targets"][name] = self.writes[index].last_prefix_statistics
            result[name + LORA_B_SUFFIX] = common[name + LORA_B_SUFFIX] + written
        return result
