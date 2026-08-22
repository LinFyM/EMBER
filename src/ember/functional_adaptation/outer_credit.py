"""Paired train-task closed-loop credit for a fixed functional code decoder."""

from __future__ import annotations

from typing import Sequence

import torch

from ember.reward.credit import (
    AntitheticCredit,
    paired_antithetic_credit,
)
from ember.reward.protocol import RewardProtocolError
from ember.reward.rollout import RewardTrajectory


AntitheticCodeCredit = AntitheticCredit


def paired_antithetic_code_credit(
    plus: Sequence[RewardTrajectory],
    minus: Sequence[RewardTrajectory],
    epsilon: torch.Tensor,
    *,
    sigma: float,
    success_weight: float,
    progress_weight: float,
    success_efficiency_weight: float,
) -> AntitheticCodeCredit:
    """Estimate dJ/dz from exact K2 common-random-number rollout arms."""
    return paired_antithetic_credit(
        plus,
        minus,
        epsilon,
        sigma=sigma,
        success_weight=success_weight,
        progress_weight=progress_weight,
        success_efficiency_weight=success_efficiency_weight,
    )


def outer_credit_surrogate_loss(
    code: torch.Tensor,
    credit: AntitheticCodeCredit,
    *,
    anchor_code: torch.Tensor,
    anchor_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transport rollout credit through the amortized Writer, not the decoder."""

    target = anchor_code.to(code)
    if (
        code.ndim != 2
        or code.shape[0] != 1
        or credit.gradient.shape != code.shape
        or target.shape not in {code.shape, code.shape[1:]}
        or anchor_weight < 0
    ):
        raise RewardProtocolError("invalid outer-credit Writer surrogate")
    if target.ndim == 1:
        target = target[None]
    anchor = torch.nn.functional.mse_loss(code.float(), target.float())
    ascent = -(code.float() * credit.gradient.to(code)).sum()
    return ascent + anchor_weight * anchor, anchor
