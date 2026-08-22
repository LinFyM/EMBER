"""Action-informed factor coordinates for task-equal Stage 1 outcome credit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from ember.ecp.contracts import TargetOwner
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX
from ember.reward.protocol import RewardProtocolError


@dataclass(frozen=True)
class ActionGuidedFactorPerturbation:
    """One paired complete-LoRA perturbation in 38 owner-local directions."""

    epsilon: torch.Tensor
    sigma: float
    directions: Mapping[str, torch.Tensor]
    direction_norm_sq: torch.Tensor
    plus_state: Mapping[str, torch.Tensor]
    minus_state: Mapping[str, torch.Tensor]
    active_owners: int


def _rademacher(count: int, *, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (
        torch.randint(0, 2, (1, count), generator=generator, dtype=torch.float32)
        .mul_(2.0)
        .sub_(1.0)
        .to(device)
    )


def action_guided_factor_perturbation(
    state: Mapping[str, torch.Tensor],
    action_gradients: Mapping[str, torch.Tensor],
    owners: tuple[TargetOwner, ...],
    *,
    sigma: float,
    seed: int,
) -> ActionGuidedFactorPerturbation:
    """Use exact action-loss descent as one relative factor direction per owner.

    Each A/B pair is jointly L2-normalized, then rescaled to the base pair's
    factor norm. Consequently ``sigma`` is the same relative factor change for
    every active owner despite different target shapes and families.
    """

    expected = {
        owner.target_name + suffix
        for owner in owners
        for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX)
    }
    if set(state) != expected or set(action_gradients) != expected or sigma <= 0:
        raise RewardProtocolError("invalid action-guided factor surface")
    first = next(iter(state.values()))
    epsilon = _rademacher(len(owners), seed=seed, device=first.device)
    directions: dict[str, torch.Tensor] = {}
    plus: dict[str, torch.Tensor] = {}
    minus: dict[str, torch.Tensor] = {}
    norm_squares = []
    active = 0
    for owner in owners:
        name_a = owner.target_name + LORA_A_SUFFIX
        name_b = owner.target_name + LORA_B_SUFFIX
        base_a = state[name_a].detach().float()
        base_b = state[name_b].detach().float()
        gradient_a = action_gradients[name_a].detach().float()
        gradient_b = action_gradients[name_b].detach().float()
        base_norm_sq = base_a.square().sum() + base_b.square().sum()
        gradient_norm_sq = gradient_a.square().sum() + gradient_b.square().sum()
        if not bool(torch.isfinite(base_norm_sq + gradient_norm_sq)) or float(
            base_norm_sq
        ) <= 0:
            raise RewardProtocolError("non-finite action-guided factor direction")
        if float(gradient_norm_sq) > 0:
            scale = (base_norm_sq / gradient_norm_sq).sqrt()
            direction_a = -gradient_a * scale
            direction_b = -gradient_b * scale
            active += 1
        else:
            direction_a = torch.zeros_like(base_a)
            direction_b = torch.zeros_like(base_b)
            epsilon[:, owner.index] = 0
        coefficient = sigma * epsilon[0, owner.index]
        directions[name_a] = direction_a
        directions[name_b] = direction_b
        plus[name_a] = (base_a + coefficient * direction_a).to(state[name_a])
        plus[name_b] = (base_b + coefficient * direction_b).to(state[name_b])
        minus[name_a] = (base_a - coefficient * direction_a).to(state[name_a])
        minus[name_b] = (base_b - coefficient * direction_b).to(state[name_b])
        norm_squares.append(base_norm_sq)
    return ActionGuidedFactorPerturbation(
        epsilon=epsilon,
        sigma=float(sigma),
        directions=directions,
        direction_norm_sq=torch.stack(norm_squares),
        plus_state=plus,
        minus_state=minus,
        active_owners=active,
    )


def action_guided_outcome_leaf_gradients(
    perturbation: ActionGuidedFactorPerturbation,
    owners: tuple[TargetOwner, ...],
    coordinate_gradient: torch.Tensor,
    *,
    weight: float,
) -> dict[str, torch.Tensor]:
    """Project reward ascent onto the action-informed directions.

    The returned tensors are loss gradients. For every active owner their inner
    product with its proposal direction equals the negative weighted reward
    coordinate gradient, so optimizer descent performs reward ascent without
    treating reward as a deployment input.
    """

    if (
        coordinate_gradient.shape != (1, len(owners))
        or perturbation.direction_norm_sq.shape != (len(owners),)
        or weight <= 0
        or not bool(torch.isfinite(coordinate_gradient).all())
    ):
        raise RewardProtocolError("invalid action-guided outcome gradient")
    result: dict[str, torch.Tensor] = {}
    for owner in owners:
        denominator = perturbation.direction_norm_sq[owner.index].clamp_min(1e-20)
        coefficient = (
            -weight * coordinate_gradient[0, owner.index].float() / denominator
        )
        for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX):
            name = owner.target_name + suffix
            result[name] = (coefficient * perturbation.directions[name]).to(
                perturbation.plus_state[name]
            )
    return result
