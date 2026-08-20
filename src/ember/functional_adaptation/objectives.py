"""Gauge-invariant objectives for fitting the fixed adapter decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract
from ember.functional_adaptation.inference import FunctionalCodePosterior


@dataclass(frozen=True)
class CodeInferenceLoss:
    total: torch.Tensor
    combined_code: torch.Tensor
    language_code: torch.Tensor
    video_code: torch.Tensor
    correct_confidence: torch.Tensor
    control_confidence: torch.Tensor
    control_update: torch.Tensor
    same_task_consistency: torch.Tensor
    action_alignment: torch.Tensor


def functional_code_inference_loss(
    correct: FunctionalCodePosterior,
    target_code: torch.Tensor,
    *,
    weights: Mapping[str, float],
    control: FunctionalCodePosterior | None = None,
    same_task_other: FunctionalCodePosterior | None = None,
    action_phase_targets: torch.Tensor | None = None,
) -> CodeInferenceLoss:
    """Train honest language/video baselines and a process-sensitive posterior."""

    target = target_code.to(correct.combined_code)
    if target.ndim == 1:
        target = target[None]
    if target.shape != correct.combined_code.shape:
        raise ValueError("functional code target changed shape")
    mse = torch.nn.functional.mse_loss
    combined = mse(correct.combined_code.float(), target.float())
    language = mse(correct.language_code.float(), target.float())
    video = mse(correct.video_code.float(), target.float())
    correct_confidence = torch.nn.functional.binary_cross_entropy_with_logits(
        correct.posterior_confidence_logits.float(),
        torch.ones_like(correct.posterior_confidence_logits, dtype=torch.float32),
    )
    zero = combined.new_zeros(())
    control_confidence = zero
    control_update = zero
    if control is not None:
        control_confidence = torch.nn.functional.binary_cross_entropy_with_logits(
            control.posterior_confidence_logits.float(),
            torch.zeros_like(control.posterior_confidence_logits, dtype=torch.float32),
        )
        control_update = mse(
            control.combined_code.float(), control.language_code.float()
        )
    same_task_consistency = zero
    if same_task_other is not None:
        same_task_consistency = mse(
            same_task_other.combined_code.float(), correct.combined_code.float()
        ) + mse(same_task_other.combined_code.float(), target.float())
    action_alignment = zero
    if action_phase_targets is not None:
        action_alignment = mse(
            correct.action_phase_predictions.float(),
            action_phase_targets.to(correct.action_phase_predictions).float(),
        )
    terms = {
        "combined_code": combined,
        "language_code": language,
        "video_code": video,
        "correct_confidence": correct_confidence,
        "control_confidence": control_confidence,
        "control_update": control_update,
        "same_task_consistency": same_task_consistency,
        "action_alignment": action_alignment,
    }
    missing = set(terms) - set(weights)
    if missing:
        raise ValueError(f"missing code-inference loss weights: {sorted(missing)}")
    total = sum(float(weights[name]) * value for name, value in terms.items())
    return CodeInferenceLoss(total=total, **terms)


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
