from __future__ import annotations

from dataclasses import replace

import torch

from ember.functional_adaptation.outer_credit import (
    outer_credit_surrogate_loss,
    paired_antithetic_code_credit,
)
from ember.reward.rollout import RewardTrajectory


def _trajectory(*, success: bool, peak: int, seeds: tuple[int, ...]) -> RewardTrajectory:
    return RewardTrajectory(
        suite="libero_goal",
        task_id=2,
        global_task_id=22,
        adaptation_seed=7,
        rollout_cursor=3,
        env_seed=11,
        policy_seed_root=13,
        success=success,
        steps=5,
        reward_sum=float(success),
        dummy_settling_steps=10,
        policy_noise_seeds=seeds,
        observations=(),
        action_chunks=(),
        valid_action_steps=(),
        goal_predicate_count=2,
        goal_predicate_peak=peak,
    )


def test_antithetic_outer_credit_uses_paired_success_and_progress() -> None:
    plus = (
        _trajectory(success=True, peak=2, seeds=(17,)),
        _trajectory(success=False, peak=1, seeds=(19, 23)),
    )
    minus = (
        _trajectory(success=False, peak=0, seeds=(17, 29)),
        _trajectory(success=False, peak=1, seeds=(19, 23, 31)),
    )
    plus = (plus[0], replace(plus[1], rollout_cursor=4, env_seed=12))
    minus = (minus[0], replace(minus[1], rollout_cursor=4, env_seed=12))
    epsilon = torch.tensor([[1.0, -2.0]])
    credit = paired_antithetic_code_credit(
        plus,
        minus,
        epsilon,
        sigma=0.5,
        success_weight=1.0,
        progress_weight=0.25,
        success_efficiency_weight=0.0,
    )
    assert credit.plus_successes == 1
    assert credit.minus_successes == 0
    assert credit.mean_advantage == 0.625
    torch.testing.assert_close(credit.gradient, epsilon * 0.625)

    code = torch.tensor([[0.25, -0.5]], requires_grad=True)
    loss, anchor = outer_credit_surrogate_loss(
        code,
        credit,
        anchor_code=torch.zeros(2),
        anchor_weight=0.1,
    )
    loss.backward()
    assert float(anchor.detach()) > 0
    assert code.grad is not None
