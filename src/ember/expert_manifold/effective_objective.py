"""Gauge-invariant policy-effective supervision for generated PI05 LoRAs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from ember.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    LoRAContract,
    expected_lora_state_shapes,
)

from .contract import ExpertManifoldError


@dataclass(frozen=True)
class EffectiveAlignment:
    """Exact global effective-update statistics for one generated/target pair."""

    generated_norm: torch.Tensor
    target_norm: torch.Tensor
    inner_product: torch.Tensor
    cosine: torch.Tensor


@dataclass(frozen=True)
class EffectiveExpertLoss:
    """Direction and robust log-energy losses for one correct Writer output."""

    total: torch.Tensor
    direction: torch.Tensor
    log_norm: torch.Tensor
    alignment: EffectiveAlignment


@dataclass(frozen=True)
class EffectiveRankingLoss:
    """Bounded-gradient correct-over-counterfactual expert ranking."""

    loss: torch.Tensor
    margin: torch.Tensor
    correct: EffectiveAlignment
    counterfactual: EffectiveAlignment


@dataclass(frozen=True)
class EffectiveAuxiliaryGradients:
    """Unweighted output-space gradients for expert and ranking supervision."""

    expert: EffectiveExpertLoss
    ranking: EffectiveRankingLoss
    correct_expert: Mapping[str, torch.Tensor]
    correct_ranking: Mapping[str, torch.Tensor]
    counterfactual_ranking: Mapping[str, torch.Tensor]


def _validate_state(
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    *,
    label: str,
) -> tuple[int, ...]:
    expected = expected_lora_state_shapes(contract)
    if set(state) != set(expected):
        raise ExpertManifoldError(f"{label} LoRA tensor names changed")
    leading: tuple[int, ...] | None = None
    for name, shape in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape[-2:]) != shape:
            raise ExpertManifoldError(f"{label} LoRA tensor shape changed: {name}")
        current = tuple(value.shape[:-2])
        if leading is None:
            leading = current
        elif current != leading:
            raise ExpertManifoldError(f"{label} LoRA leading axes changed")
        if not value.is_floating_point():
            raise ExpertManifoldError(f"{label} LoRA tensor is not floating point")
    return () if leading is None else leading


def _target_for_leading(
    value: torch.Tensor,
    leading: tuple[int, ...],
    *,
    label: str,
) -> torch.Tensor:
    observed = tuple(value.shape[:-2])
    if observed == leading:
        return value
    if observed:
        raise ExpertManifoldError(f"{label} target LoRA cannot broadcast")
    return value.reshape(*(1 for _ in leading), *value.shape)


def effective_alignment(
    generated: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    *,
    epsilon: float = 1e-12,
) -> EffectiveAlignment:
    """Compare all 38 effective ``BA`` updates without materializing them."""

    if epsilon <= 0:
        raise ExpertManifoldError("effective-alignment epsilon must be positive")
    leading = _validate_state(generated, contract, label="generated")
    target_leading = _validate_state(target, contract, label="expert target")
    if target_leading not in ((), leading):
        raise ExpertManifoldError("expert target LoRA leading axes changed")

    generated_norm_sq: torch.Tensor | None = None
    target_norm_sq: torch.Tensor | None = None
    inner_product: torch.Tensor | None = None
    for item in contract.targets:
        a_name = item.name + LORA_A_SUFFIX
        b_name = item.name + LORA_B_SUFFIX
        a = generated[a_name].to(dtype=torch.float32)
        b = generated[b_name].to(dtype=torch.float32)
        expert_a = _target_for_leading(
            target[a_name].to(device=a.device, dtype=torch.float32),
            leading,
            label="A",
        )
        expert_b = _target_for_leading(
            target[b_name].to(device=b.device, dtype=torch.float32),
            leading,
            label="B",
        )
        generated_gram = b.transpose(-2, -1) @ b
        generated_row_gram = a @ a.transpose(-2, -1)
        expert_gram = expert_b.transpose(-2, -1) @ expert_b
        expert_row_gram = expert_a @ expert_a.transpose(-2, -1)
        generated_part = (generated_gram * generated_row_gram).sum(dim=(-2, -1))
        target_part = (expert_gram * expert_row_gram).sum(dim=(-2, -1))
        cross_part = (
            (b.transpose(-2, -1) @ expert_b)
            * (a @ expert_a.transpose(-2, -1))
        ).sum(dim=(-2, -1))
        generated_norm_sq = (
            generated_part
            if generated_norm_sq is None
            else generated_norm_sq + generated_part
        )
        target_norm_sq = (
            target_part if target_norm_sq is None else target_norm_sq + target_part
        )
        inner_product = (
            cross_part if inner_product is None else inner_product + cross_part
        )
    if generated_norm_sq is None or target_norm_sq is None or inner_product is None:
        raise ExpertManifoldError("effective alignment received no LoRA targets")
    if bool((target_norm_sq <= epsilon).any()):
        raise ExpertManifoldError("expert target has zero policy-effective energy")
    generated_norm = generated_norm_sq.clamp_min(0).sqrt()
    target_norm = target_norm_sq.clamp_min(0).sqrt()
    denominator = (generated_norm * target_norm).clamp_min(epsilon)
    cosine = (inner_product / denominator).clamp(-1.0, 1.0)
    values = (generated_norm, target_norm, inner_product, cosine)
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ExpertManifoldError("policy-effective alignment is non-finite")
    return EffectiveAlignment(
        generated_norm=generated_norm,
        target_norm=target_norm,
        inner_product=inner_product,
        cosine=cosine,
    )


def effective_expert_loss(
    generated: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    *,
    norm_weight: float,
    smooth_l1_beta: float,
    epsilon: float = 1e-12,
) -> EffectiveExpertLoss:
    """Attract a correct generated update to one task expert in effective space."""

    if norm_weight < 0 or smooth_l1_beta <= 0:
        raise ExpertManifoldError("invalid effective expert-loss weights")
    alignment = effective_alignment(generated, target, contract, epsilon=epsilon)
    direction = (1.0 - alignment.cosine).mean()
    log_ratio = torch.log(
        alignment.generated_norm.clamp_min(epsilon)
        / alignment.target_norm.clamp_min(epsilon)
    )
    log_norm = F.smooth_l1_loss(
        log_ratio,
        torch.zeros_like(log_ratio),
        beta=smooth_l1_beta,
        reduction="mean",
    )
    total = direction + norm_weight * log_norm
    if not bool(torch.isfinite(total)):
        raise ExpertManifoldError("effective expert loss is non-finite")
    return EffectiveExpertLoss(
        total=total,
        direction=direction,
        log_norm=log_norm,
        alignment=alignment,
    )


def effective_counterfactual_ranking_loss(
    correct: Mapping[str, torch.Tensor],
    counterfactual: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    *,
    required_margin: float,
    temperature: float,
    epsilon: float = 1e-12,
) -> EffectiveRankingLoss:
    """Require correct video LoRA to be more expert-aligned than a negative arm."""

    if not 0 <= required_margin <= 2 or temperature <= 0:
        raise ExpertManifoldError("invalid policy-effective ranking parameters")
    correct_alignment = effective_alignment(correct, target, contract, epsilon=epsilon)
    counterfactual_alignment = effective_alignment(
        counterfactual, target, contract, epsilon=epsilon
    )
    if correct_alignment.cosine.shape != counterfactual_alignment.cosine.shape:
        raise ExpertManifoldError("counterfactual alignment batch changed")
    margin = correct_alignment.cosine - counterfactual_alignment.cosine
    loss = (
        F.softplus((required_margin - margin) / temperature) * temperature
    ).mean()
    if not bool(torch.isfinite(loss)):
        raise ExpertManifoldError("policy-effective ranking loss is non-finite")
    return EffectiveRankingLoss(
        loss=loss,
        margin=margin,
        correct=correct_alignment,
        counterfactual=counterfactual_alignment,
    )


def effective_auxiliary_output_gradients(
    correct: Mapping[str, torch.Tensor],
    counterfactual: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    *,
    norm_weight: float,
    smooth_l1_beta: float,
    required_margin: float,
    temperature: float,
    epsilon: float = 1e-12,
) -> EffectiveAuxiliaryGradients:
    """Differentiate both auxiliaries only to generated LoRA output tensors."""

    names = tuple(correct)
    if set(counterfactual) != set(names):
        raise ExpertManifoldError("counterfactual LoRA tensor names changed")
    expert = effective_expert_loss(
        correct,
        target,
        contract,
        norm_weight=norm_weight,
        smooth_l1_beta=smooth_l1_beta,
        epsilon=epsilon,
    )
    ranking = effective_counterfactual_ranking_loss(
        correct,
        counterfactual,
        target,
        contract,
        required_margin=required_margin,
        temperature=temperature,
        epsilon=epsilon,
    )
    correct_values = tuple(correct[name] for name in names)
    counterfactual_values = tuple(counterfactual[name] for name in names)
    expert_gradients = torch.autograd.grad(expert.total, correct_values)
    ranking_gradients = torch.autograd.grad(
        ranking.loss,
        (*correct_values, *counterfactual_values),
    )
    split = len(names)
    values = (*expert_gradients, *ranking_gradients)
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise ExpertManifoldError("policy-effective auxiliary gradient is non-finite")
    return EffectiveAuxiliaryGradients(
        expert=expert,
        ranking=ranking,
        correct_expert=dict(zip(names, expert_gradients, strict=True)),
        correct_ranking=dict(zip(names, ranking_gradients[:split], strict=True)),
        counterfactual_ranking=dict(
            zip(names, ranking_gradients[split:], strict=True)
        ),
    )
