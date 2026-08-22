"""Common-random antithetic credit for structured policy coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ember.reward.protocol import RewardProtocolError, SUITE_HORIZONS
from ember.reward.rollout import RewardTrajectory


@dataclass(frozen=True)
class AntitheticCredit:
    gradient: torch.Tensor
    plus_scores: tuple[float, ...]
    minus_scores: tuple[float, ...]
    lane_advantages: tuple[float, ...]
    plus_successes: int
    minus_successes: int
    plus_progress_mean: float
    minus_progress_mean: float

    @property
    def mean_advantage(self) -> float:
        return sum(self.lane_advantages) / len(self.lane_advantages)


def trajectory_outer_score(
    trajectory: RewardTrajectory,
    *,
    success_weight: float,
    progress_weight: float,
    success_efficiency_weight: float,
) -> tuple[float, float]:
    """Score terminal success, peak BDDL progress, and successful efficiency."""

    count = int(trajectory.goal_predicate_count)
    peak = int(trajectory.goal_predicate_peak)
    if (
        success_weight <= 0
        or progress_weight < 0
        or success_efficiency_weight < 0
        or count <= 0
        or not 0 <= peak <= count
    ):
        raise RewardProtocolError("invalid outer-credit trajectory score")
    progress = peak / count
    efficiency = (
        1.0 - trajectory.steps / SUITE_HORIZONS[trajectory.suite]
        if trajectory.success
        else 0.0
    )
    return (
        success_weight * float(trajectory.success)
        + success_efficiency_weight * efficiency
        + progress_weight * progress,
        progress,
    )


def paired_antithetic_credit(
    plus: Sequence[RewardTrajectory],
    minus: Sequence[RewardTrajectory],
    epsilon: torch.Tensor,
    *,
    sigma: float,
    success_weight: float,
    progress_weight: float,
    success_efficiency_weight: float,
) -> AntitheticCredit:
    """Estimate the reward gradient of one flattened structured coordinate."""

    if (
        len(plus) != 2
        or len(minus) != 2
        or epsilon.ndim != 2
        or epsilon.shape[0] != 1
        or sigma <= 0
        or not bool(torch.isfinite(epsilon).all())
    ):
        raise RewardProtocolError("invalid antithetic outer-credit panel")
    for left, right in zip(plus, minus, strict=True):
        shared_replans = min(
            len(left.policy_noise_seeds), len(right.policy_noise_seeds)
        )
        if (
            left.suite != right.suite
            or left.task_id != right.task_id
            or left.global_task_id != right.global_task_id
            or left.rollout_cursor != right.rollout_cursor
            or left.env_seed != right.env_seed
            or left.policy_seed_root != right.policy_seed_root
            or shared_replans <= 0
            or left.policy_noise_seeds[:shared_replans]
            != right.policy_noise_seeds[:shared_replans]
        ):
            raise RewardProtocolError("outer-credit policy arms lost exact pairing")
    plus_values = tuple(
        trajectory_outer_score(
            row,
            success_weight=success_weight,
            progress_weight=progress_weight,
            success_efficiency_weight=success_efficiency_weight,
        )
        for row in plus
    )
    minus_values = tuple(
        trajectory_outer_score(
            row,
            success_weight=success_weight,
            progress_weight=progress_weight,
            success_efficiency_weight=success_efficiency_weight,
        )
        for row in minus
    )
    plus_scores = tuple(value[0] for value in plus_values)
    minus_scores = tuple(value[0] for value in minus_values)
    advantages = tuple(
        left - right for left, right in zip(plus_scores, minus_scores, strict=True)
    )
    mean_advantage = sum(advantages) / len(advantages)
    return AntitheticCredit(
        gradient=epsilon.float() * (mean_advantage / (2.0 * sigma)),
        plus_scores=plus_scores,
        minus_scores=minus_scores,
        lane_advantages=advantages,
        plus_successes=sum(int(row.success) for row in plus),
        minus_successes=sum(int(row.success) for row in minus),
        plus_progress_mean=sum(value[1] for value in plus_values) / len(plus_values),
        minus_progress_mean=sum(value[1] for value in minus_values)
        / len(minus_values),
    )
