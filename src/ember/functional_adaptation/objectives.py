"""Gauge-invariant objectives for fitting the fixed adapter decoder."""

from __future__ import annotations

from typing import Mapping

import torch

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract


def effective_update_probes(
    contract: LoRAContract,
    *,
    probe_count: int,
    seed: int,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Create a small deterministic input panel for every LoRA target."""

    if probe_count <= 0:
        raise ValueError("effective-update probe count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return {
        target.name: (
            torch.randn(target.in_features, probe_count, generator=generator)
            / target.in_features**0.5
        ).to(device)
        for target in contract.targets
    }


def effective_update_probe_loss(
    candidate: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    probes: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Match BA on a compact input panel without choosing an A/B gauge."""

    numerator: torch.Tensor | None = None
    denominator: torch.Tensor | None = None
    for owner in contract.targets:
        probe = probes[owner.name]
        name_a = owner.name + LORA_A_SUFFIX
        name_b = owner.name + LORA_B_SUFFIX
        candidate_response = candidate[name_b].float() @ (
            candidate[name_a].float() @ probe.float()
        )
        target_response = target[name_b].float() @ (
            target[name_a].float() @ probe.float()
        )
        error = (candidate_response - target_response).square().sum()
        energy = target_response.square().sum()
        numerator = error if numerator is None else numerator + error
        denominator = energy if denominator is None else denominator + energy
    if numerator is None or denominator is None:
        raise ValueError("effective-update probe panel is empty")
    return numerator / denominator.clamp_min(1e-8)
