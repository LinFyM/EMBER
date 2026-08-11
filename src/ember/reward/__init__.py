"""Canonical exact-paired closed-loop outcome mechanics for PCUG."""

from ember.reward.protocol import (
    RewardTask,
    reward_credit_environment_seed,
    reward_credit_policy_noise_seed,
)
from ember.reward.rollout import (
    RandomResetEnvironmentPool,
    RewardRolloutOutcome,
    collect_paired_reward_arm_outcomes,
)

__all__ = [
    "RewardTask",
    "RewardRolloutOutcome",
    "RandomResetEnvironmentPool",
    "collect_paired_reward_arm_outcomes",
    "reward_credit_environment_seed",
    "reward_credit_policy_noise_seed",
]
