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
    generated_norm_sq: torch.Tensor
    target_norm_sq: torch.Tensor
    inner_product: torch.Tensor
    cosine: torch.Tensor
    projection_coefficient: torch.Tensor
    per_target_inner_product: torch.Tensor
    per_target_target_norm_sq: torch.Tensor


@dataclass(frozen=True)
class EffectiveProjectionLoss:
    """Robust expert-component completion loss for one correct Writer output."""

    total: torch.Tensor
    coefficient: torch.Tensor
    alignment: EffectiveAlignment


@dataclass(frozen=True)
class EffectiveTangentState:
    """One condition's displacement from its same-input frozen v6 output."""

    loss: torch.Tensor
    alignment: EffectiveAlignment
    anchor_alignment: EffectiveAlignment
    delta_norm: torch.Tensor
    directional_coefficient: torch.Tensor
    directional_component_norm: torch.Tensor
    directional_to_anchor_ratio: torch.Tensor
    orthogonal_delta_norm: torch.Tensor
    orthogonal_to_anchor_ratio: torch.Tensor
    orthogonal_to_direction_ratio: torch.Tensor
    orthogonal_clamp_correction: torch.Tensor


@dataclass(frozen=True)
class EffectiveTangentTubeLoss:
    """Expert completion plus a two-condition, same-input tangent tube."""

    total: torch.Tensor
    completion: EffectiveProjectionLoss
    tube: torch.Tensor
    correct: EffectiveTangentState
    counterfactual: EffectiveTangentState

    @property
    def coefficient(self) -> torch.Tensor:
        return self.completion.coefficient

    @property
    def alignment(self) -> EffectiveAlignment:
        return self.completion.alignment


@dataclass(frozen=True)
class EffectiveRankingLoss:
    """Smooth correct-over-counterfactual expert-component ranking."""

    loss: torch.Tensor
    margin: torch.Tensor
    correct: EffectiveAlignment
    counterfactual: EffectiveAlignment


@dataclass(frozen=True)
class EffectiveAuxiliaryGradients:
    """Unweighted output-space gradients for projection and ranking."""

    projection: EffectiveTangentTubeLoss
    ranking: EffectiveRankingLoss
    correct_completion: Mapping[str, torch.Tensor]
    correct_projection: Mapping[str, torch.Tensor]
    counterfactual_projection: Mapping[str, torch.Tensor]
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


def _effective_alignment(
    generated: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    *,
    epsilon: float = 1e-12,
) -> EffectiveAlignment:

    if epsilon <= 0:
        raise ExpertManifoldError("effective-alignment epsilon must be positive")
    leading = _validate_state(generated, contract, label="generated")
    target_leading = _validate_state(target, contract, label="expert target")
    if target_leading not in ((), leading):
        raise ExpertManifoldError("expert target LoRA leading axes changed")

    generated_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    cross_parts: list[torch.Tensor] = []
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
        generated_parts.append(generated_part)
        target_parts.append(target_part)
        cross_parts.append(cross_part)
    if not generated_parts:
        raise ExpertManifoldError("effective alignment received no LoRA targets")
    per_target_generated_norm_sq = torch.stack(generated_parts, dim=-1)
    per_target_target_norm_sq = torch.stack(target_parts, dim=-1)
    per_target_inner_product = torch.stack(cross_parts, dim=-1)
    generated_norm_sq = per_target_generated_norm_sq.sum(dim=-1)
    target_norm_sq = per_target_target_norm_sq.sum(dim=-1)
    inner_product = per_target_inner_product.sum(dim=-1)
    generated_norm = generated_norm_sq.clamp_min(0).sqrt()
    target_norm = target_norm_sq.clamp_min(0).sqrt()
    denominator = (generated_norm * target_norm).clamp_min(epsilon)
    cosine = (inner_product / denominator).clamp(-1.0, 1.0)
    projection_coefficient = inner_product / (target_norm_sq + epsilon)
    return EffectiveAlignment(
        generated_norm=generated_norm,
        target_norm=target_norm,
        generated_norm_sq=generated_norm_sq,
        target_norm_sq=target_norm_sq,
        inner_product=inner_product,
        cosine=cosine,
        projection_coefficient=projection_coefficient,
        per_target_inner_product=per_target_inner_product,
        per_target_target_norm_sq=per_target_target_norm_sq,
    )


def _validate_alignment(value: EffectiveAlignment, *, epsilon: float) -> None:
    checks = torch.stack(
        (
            (value.target_norm > epsilon).all(),
            torch.isfinite(value.generated_norm).all(),
            torch.isfinite(value.target_norm).all(),
            torch.isfinite(value.generated_norm_sq).all(),
            torch.isfinite(value.target_norm_sq).all(),
            torch.isfinite(value.inner_product).all(),
            torch.isfinite(value.cosine).all(),
            torch.isfinite(value.projection_coefficient).all(),
            torch.isfinite(value.per_target_inner_product).all(),
            torch.isfinite(value.per_target_target_norm_sq).all(),
        )
    ).detach().to(device="cpu").tolist()
    if not checks[0]:
        raise ExpertManifoldError("expert target has zero policy-effective energy")
    if not all(checks[1:]):
        raise ExpertManifoldError("policy-effective alignment is non-finite")


def effective_alignment(
    generated: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    *,
    epsilon: float = 1e-12,
) -> EffectiveAlignment:
    """Compare all 38 effective ``BA`` updates without materializing them."""

    value = _effective_alignment(
        generated,
        target,
        contract,
        epsilon=epsilon,
    )
    _validate_alignment(value, epsilon=epsilon)
    return value


def _projection_loss_from_alignment(
    alignment: EffectiveAlignment,
    *,
    smooth_l1_beta: float,
) -> EffectiveProjectionLoss:
    total = F.smooth_l1_loss(
        alignment.projection_coefficient,
        torch.ones_like(alignment.projection_coefficient),
        beta=smooth_l1_beta,
        reduction="mean",
    )
    return EffectiveProjectionLoss(
        total=total,
        coefficient=alignment.projection_coefficient,
        alignment=alignment,
    )


def effective_projection_loss(
    generated: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    *,
    smooth_l1_beta: float,
    epsilon: float = 1e-12,
) -> EffectiveProjectionLoss:
    """Complete only the task-expert component of one generated update."""

    if smooth_l1_beta <= 0:
        raise ExpertManifoldError("invalid effective projection-loss beta")
    return _projection_loss_from_alignment(
        effective_alignment(generated, target, contract, epsilon=epsilon),
        smooth_l1_beta=smooth_l1_beta,
    )


def _effective_cross_inner_product(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    *,
    left_label: str,
    right_label: str,
) -> torch.Tensor:
    leading = _validate_state(left, contract, label=left_label)
    if _validate_state(right, contract, label=right_label) != leading:
        raise ExpertManifoldError("condition-local LoRA leading axes changed")
    parts: list[torch.Tensor] = []
    for item in contract.targets:
        a_name = item.name + LORA_A_SUFFIX
        b_name = item.name + LORA_B_SUFFIX
        left_a = left[a_name].to(dtype=torch.float32)
        left_b = left[b_name].to(dtype=torch.float32)
        right_a = right[a_name].to(device=left_a.device, dtype=torch.float32)
        right_b = right[b_name].to(device=left_b.device, dtype=torch.float32)
        parts.append(
            (
                (left_b.transpose(-2, -1) @ right_b)
                * (left_a @ right_a.transpose(-2, -1))
            ).sum(dim=(-2, -1))
        )
    if not parts:
        raise ExpertManifoldError("condition-local tangent received no LoRA targets")
    return torch.stack(parts, dim=-1).sum(dim=-1)


def _effective_tangent_state(
    generated: Mapping[str, torch.Tensor],
    anchor: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    *,
    smooth_l1_beta: float,
    epsilon: float,
) -> EffectiveTangentState:
    if any(value.requires_grad for value in anchor.values()):
        raise ExpertManifoldError("condition-local v6 anchor is not frozen")
    alignment = _effective_alignment(generated, target, contract, epsilon=epsilon)
    anchor_alignment = _effective_alignment(anchor, target, contract, epsilon=epsilon)
    generated_anchor_inner = _effective_cross_inner_product(
        generated,
        anchor,
        contract,
        left_label="generated",
        right_label="condition-local v6 anchor",
    )
    target_norm_sq = alignment.target_norm_sq
    projection_denominator = target_norm_sq.clamp_min(epsilon)
    delta_target_inner = alignment.inner_product - anchor_alignment.inner_product
    directional_coefficient = delta_target_inner / projection_denominator
    raw_delta_norm_sq = (
        alignment.generated_norm_sq
        + anchor_alignment.generated_norm_sq
        - 2.0 * generated_anchor_inner
    )
    delta_norm_sq = raw_delta_norm_sq.clamp_min(0.0)
    raw_orthogonal_delta_norm_sq = (
        raw_delta_norm_sq
        - delta_target_inner.square() / projection_denominator
    )
    orthogonal_delta_norm_sq = raw_orthogonal_delta_norm_sq.clamp_min(0.0)
    delta_norm = delta_norm_sq.sqrt()
    orthogonal_delta_norm = orthogonal_delta_norm_sq.sqrt()
    directional_component_norm = directional_coefficient.abs() * alignment.target_norm
    loss = (
        orthogonal_delta_norm_sq
        / (2.0 * smooth_l1_beta * projection_denominator)
    ).mean()
    return EffectiveTangentState(
        loss=loss,
        alignment=alignment,
        anchor_alignment=anchor_alignment,
        delta_norm=delta_norm,
        directional_coefficient=directional_coefficient,
        directional_component_norm=directional_component_norm,
        directional_to_anchor_ratio=(
            directional_component_norm
            / anchor_alignment.generated_norm.clamp_min(epsilon)
        ),
        orthogonal_delta_norm=orthogonal_delta_norm,
        orthogonal_to_anchor_ratio=(
            orthogonal_delta_norm / anchor_alignment.generated_norm.clamp_min(epsilon)
        ),
        orthogonal_to_direction_ratio=(
            orthogonal_delta_norm / (directional_component_norm + epsilon)
        ),
        orthogonal_clamp_correction=(
            orthogonal_delta_norm_sq - raw_orthogonal_delta_norm_sq
        ),
    )


def _condition_local_tangent_tube_loss(
    correct: Mapping[str, torch.Tensor],
    counterfactual: Mapping[str, torch.Tensor],
    correct_anchor: Mapping[str, torch.Tensor],
    counterfactual_anchor: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    *,
    smooth_l1_beta: float,
    epsilon: float,
) -> EffectiveTangentTubeLoss:
    correct_state = _effective_tangent_state(
        correct,
        correct_anchor,
        target,
        contract,
        smooth_l1_beta=smooth_l1_beta,
        epsilon=epsilon,
    )
    counterfactual_state = _effective_tangent_state(
        counterfactual,
        counterfactual_anchor,
        target,
        contract,
        smooth_l1_beta=smooth_l1_beta,
        epsilon=epsilon,
    )
    completion = _projection_loss_from_alignment(
        correct_state.alignment,
        smooth_l1_beta=smooth_l1_beta,
    )
    tube = 0.5 * (correct_state.loss + counterfactual_state.loss)
    return EffectiveTangentTubeLoss(
        total=completion.total + tube,
        completion=completion,
        tube=tube,
        correct=correct_state,
        counterfactual=counterfactual_state,
    )


def effective_condition_local_tangent_tube_loss(
    correct: Mapping[str, torch.Tensor],
    counterfactual: Mapping[str, torch.Tensor],
    correct_anchor: Mapping[str, torch.Tensor],
    counterfactual_anchor: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    *,
    smooth_l1_beta: float,
    epsilon: float = 1e-12,
) -> EffectiveTangentTubeLoss:
    """Constrain each condition's increment to the task-expert tangent."""

    if smooth_l1_beta <= 0 or epsilon <= 0:
        raise ExpertManifoldError("invalid condition-local tangent tube")
    value = _condition_local_tangent_tube_loss(
        correct,
        counterfactual,
        correct_anchor,
        counterfactual_anchor,
        target,
        contract,
        smooth_l1_beta=smooth_l1_beta,
        epsilon=epsilon,
    )
    for alignment in (
        value.correct.alignment,
        value.correct.anchor_alignment,
        value.counterfactual.alignment,
        value.counterfactual.anchor_alignment,
    ):
        _validate_alignment(alignment, epsilon=epsilon)
    if not bool(
        (value.correct.alignment.target_norm_sq > epsilon).all()
    ):
        raise ExpertManifoldError("expert target has zero tangent denominator")
    tensors = (
        value.total,
        value.completion.total,
        value.tube,
        value.correct.delta_norm,
        value.correct.directional_coefficient,
        value.correct.directional_to_anchor_ratio,
        value.correct.orthogonal_delta_norm,
        value.correct.orthogonal_to_anchor_ratio,
        value.correct.orthogonal_to_direction_ratio,
        value.correct.orthogonal_clamp_correction,
        value.counterfactual.delta_norm,
        value.counterfactual.directional_coefficient,
        value.counterfactual.directional_to_anchor_ratio,
        value.counterfactual.orthogonal_delta_norm,
        value.counterfactual.orthogonal_to_anchor_ratio,
        value.counterfactual.orthogonal_to_direction_ratio,
        value.counterfactual.orthogonal_clamp_correction,
    )
    if not all(bool(torch.isfinite(item).all()) for item in tensors):
        raise ExpertManifoldError("condition-local tangent tube is non-finite")
    return value


def _ranking_loss_from_alignments(
    correct: EffectiveAlignment,
    counterfactual: EffectiveAlignment,
    *,
    required_margin: float,
    temperature: float,
) -> EffectiveRankingLoss:
    if correct.projection_coefficient.shape != counterfactual.projection_coefficient.shape:
        raise ExpertManifoldError("counterfactual alignment batch changed")
    margin = (
        correct.projection_coefficient
        - counterfactual.projection_coefficient
    )
    loss = (
        F.softplus((required_margin - margin) / temperature) * temperature
    ).mean()
    return EffectiveRankingLoss(
        loss=loss,
        margin=margin,
        correct=correct,
        counterfactual=counterfactual,
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
    return _ranking_loss_from_alignments(
        correct_alignment,
        counterfactual_alignment,
        required_margin=required_margin,
        temperature=temperature,
    )


def effective_auxiliary_output_gradients(
    correct: Mapping[str, torch.Tensor],
    counterfactual: Mapping[str, torch.Tensor],
    correct_anchor: Mapping[str, torch.Tensor],
    counterfactual_anchor: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    *,
    smooth_l1_beta: float,
    required_margin: float,
    temperature: float,
    epsilon: float = 1e-12,
) -> EffectiveAuxiliaryGradients:
    """Differentiate both auxiliaries only to generated LoRA output tensors."""

    if (
        smooth_l1_beta <= 0
        or not 0 <= required_margin <= 2
        or temperature <= 0
        or epsilon <= 0
    ):
        raise ExpertManifoldError("invalid policy-effective auxiliary objective")
    names = tuple(correct)
    if set(counterfactual) != set(names):
        raise ExpertManifoldError("counterfactual LoRA tensor names changed")
    projection = _condition_local_tangent_tube_loss(
        correct,
        counterfactual,
        correct_anchor,
        counterfactual_anchor,
        target,
        contract,
        smooth_l1_beta=smooth_l1_beta,
        epsilon=epsilon,
    )
    ranking = _ranking_loss_from_alignments(
        projection.correct.alignment,
        projection.counterfactual.alignment,
        required_margin=required_margin,
        temperature=temperature,
    )
    correct_values = tuple(correct[name] for name in names)
    counterfactual_values = tuple(counterfactual[name] for name in names)
    completion_gradients = torch.autograd.grad(
        projection.completion.total,
        correct_values,
        retain_graph=True,
    )
    projection_gradients = torch.autograd.grad(
        projection.total,
        (*correct_values, *counterfactual_values),
        retain_graph=True,
    )
    ranking_gradients = torch.autograd.grad(
        ranking.loss,
        (*correct_values, *counterfactual_values),
    )
    split = len(names)
    return EffectiveAuxiliaryGradients(
        projection=projection,
        ranking=ranking,
        correct_completion=dict(zip(names, completion_gradients, strict=True)),
        correct_projection=dict(
            zip(names, projection_gradients[:split], strict=True)
        ),
        counterfactual_projection=dict(
            zip(names, projection_gradients[split:], strict=True)
        ),
        correct_ranking=dict(zip(names, ranking_gradients[:split], strict=True)),
        counterfactual_ranking=dict(
            zip(names, ranking_gradients[split:], strict=True)
        ),
    )
