"""Compact functional code to one complete PI0.5 task LoRA."""

from __future__ import annotations

import math
from typing import Mapping

import torch

from ember.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    LoRAContract,
    LoRATarget,
    validate_lora_state,
)


class FunctionalAdapterDecoderError(RuntimeError):
    """Raised when the fixed decoder leaves the complete-LoRA contract."""


def _factor_family(target: LoRATarget, factor: str) -> str:
    if target.name.endswith(".q_proj"):
        owner = "q"
    elif target.name.endswith(".v_proj"):
        owner = "v"
    elif target.name.endswith(".action_in_proj"):
        owner = "action_in"
    elif target.name.endswith(".action_out_proj"):
        owner = "action_out"
    else:
        raise FunctionalAdapterDecoderError(
            f"unsupported functional-adapter target: {target.name}"
        )
    return f"{owner}_{factor}"


def _factor_width(target: LoRATarget, factor: str) -> int:
    return target.in_features if factor == "a" else target.out_features


class FunctionalCodebook(torch.nn.Module):
    """Privileged meta-task codes with an explicit centered whitened gauge."""

    def __init__(self, task_count: int, code_width: int, *, seed: int) -> None:
        super().__init__()
        if not 0 < code_width < task_count:
            raise FunctionalAdapterDecoderError(
                "functional code width must be below the number of meta tasks"
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            values = torch.randn(task_count, code_width)
            values = values - values.mean(dim=0, keepdim=True)
            orthogonal, _ = torch.linalg.qr(values, mode="reduced")
            values = orthogonal * math.sqrt(task_count - 1)
        self.weight = torch.nn.Parameter(values)

    def forward(self, task_indices: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(task_indices, self.weight)

    def gauge_loss(self) -> torch.Tensor:
        centered = self.weight - self.weight.mean(dim=0, keepdim=True)
        covariance = centered.transpose(0, 1) @ centered
        covariance = covariance / (self.weight.shape[0] - 1)
        identity = torch.eye(
            covariance.shape[0],
            dtype=covariance.dtype,
            device=covariance.device,
        )
        return self.weight.mean(dim=0).square().mean() + (
            covariance - identity
        ).square().mean()


class FunctionalAdapterDecoder(torch.nn.Module):
    """Address a compact task code into all 76 native LoRA tensors."""

    def __init__(
        self,
        contract: LoRAContract,
        template_state: Mapping[str, torch.Tensor],
        *,
        code_width: int = 32,
        address_width: int = 64,
        hidden_width: int = 256,
        initialization_seed: int = 7,
    ) -> None:
        super().__init__()
        if min(code_width, address_width, hidden_width) <= 0:
            raise FunctionalAdapterDecoderError("invalid decoder dimensions")
        validate_lora_state(template_state, contract)
        self.contract = contract
        self.code_width = int(code_width)
        self.address_width = int(address_width)
        self.hidden_width = int(hidden_width)
        self.target_count = len(contract.targets)
        self.rank = int(contract.rank)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed))
            self.target_addresses = torch.nn.Embedding(
                self.target_count, self.address_width
            )
            self.rank_addresses = torch.nn.Embedding(self.rank, self.address_width)
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(
                    self.code_width + 2 * self.address_width,
                    self.hidden_width,
                    bias=False,
                ),
                torch.nn.GELU(),
                torch.nn.Linear(self.hidden_width, self.hidden_width, bias=False),
                torch.nn.GELU(),
            )
            self.factor_heads = self._build_factor_heads(contract.targets)
        for head in self.factor_heads.values():
            torch.nn.init.zeros_(head.weight)
        self._template_buffers: dict[str, str] = {}
        for index, name in enumerate(sorted(template_state)):
            buffer_name = f"template_{index:03d}"
            self.register_buffer(
                buffer_name,
                template_state[name].detach().clone().contiguous(),
                persistent=True,
            )
            self._template_buffers[name] = buffer_name

    def _build_factor_heads(
        self, targets: tuple[LoRATarget, ...]
    ) -> torch.nn.ModuleDict:
        widths: dict[str, int] = {}
        for target in targets:
            for factor in ("a", "b"):
                family = _factor_family(target, factor)
                width = _factor_width(target, factor)
                if family in widths and widths[family] != width:
                    raise FunctionalAdapterDecoderError(
                        f"factor family width changed: {family}"
                    )
                widths[family] = width
        if set(widths) != {
            "q_a",
            "q_b",
            "v_a",
            "v_b",
            "action_in_a",
            "action_in_b",
            "action_out_a",
            "action_out_b",
        }:
            raise FunctionalAdapterDecoderError(
                "decoder did not resolve all PI0.5 factor families"
            )
        return torch.nn.ModuleDict(
            {
                family: torch.nn.Linear(self.hidden_width, width, bias=False)
                for family, width in widths.items()
            }
        )

    def template_state(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, buffer_name)
            for name, buffer_name in self._template_buffers.items()
        }

    def _addressed_hidden(self, code: torch.Tensor) -> torch.Tensor:
        batch = code.shape[0]
        targets = self.target_addresses.weight[:, None, :].expand(
            self.target_count, self.rank, -1
        )
        ranks = self.rank_addresses.weight[None, :, :].expand(
            self.target_count, self.rank, -1
        )
        addresses = torch.cat((targets, ranks), dim=-1)
        expanded_code = code[:, None, None, :].expand(
            batch, self.target_count, self.rank, self.code_width
        )
        inputs = torch.cat(
            (expanded_code, addresses[None].expand(batch, -1, -1, -1)),
            dim=-1,
        )
        return self.trunk(inputs)

    def forward(self, code: torch.Tensor) -> dict[str, torch.Tensor]:
        squeeze = code.ndim == 1
        if squeeze:
            code = code[None]
        if code.ndim != 2 or code.shape[1] != self.code_width:
            raise FunctionalAdapterDecoderError(
                f"functional code must have trailing width {self.code_width}"
            )
        hidden = self._addressed_hidden(code)
        result: dict[str, torch.Tensor] = {}
        templates = self.template_state()
        for target_index, target in enumerate(self.contract.targets):
            addressed = hidden[:, target_index]
            for factor, suffix in (("a", LORA_A_SUFFIX), ("b", LORA_B_SUFFIX)):
                name = target.name + suffix
                residual = self.factor_heads[_factor_family(target, factor)](addressed)
                if factor == "b":
                    residual = residual.transpose(-1, -2)
                template = templates[name]
                generated = template[None] + residual.to(template.dtype)
                result[name] = generated[0] if squeeze else generated
        if squeeze:
            validate_lora_state(result, self.contract)
        return result


def relative_effective_update_loss(
    candidate: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    contract: LoRAContract,
) -> torch.Tensor:
    """Gauge-invariant warm-start loss on effective BA updates."""

    numerator: torch.Tensor | None = None
    denominator: torch.Tensor | None = None
    for owner in contract.targets:
        name_a = owner.name + LORA_A_SUFFIX
        name_b = owner.name + LORA_B_SUFFIX
        candidate_ba = candidate[name_b].float() @ candidate[name_a].float()
        target_ba = target[name_b].float() @ target[name_a].float()
        error = (candidate_ba - target_ba).square().sum()
        energy = target_ba.square().sum()
        numerator = error if numerator is None else numerator + error
        denominator = energy if denominator is None else denominator + energy
    if numerator is None or denominator is None:
        raise FunctionalAdapterDecoderError("effective update panel is empty")
    return numerator / denominator.clamp_min(1e-12)
