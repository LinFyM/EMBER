"""Canonical K4 outcome and constant-memory landmark mechanics for SRTP."""

from ember.reward.protocol import (
    RewardTask,
    reward_credit_environment_seed,
    reward_credit_policy_noise_seed,
    reward_tangent_landmark_index,
)
from ember.reward.rollout import (
    RandomResetEnvironmentPool,
    RewardOccupancyLandmark,
    RewardRolloutOutcome,
    collect_randomized_reward_outcomes,
)

__all__ = [
    "RewardTask",
    "RewardOccupancyLandmark",
    "RewardRolloutOutcome",
    "RandomResetEnvironmentPool",
    "collect_randomized_reward_outcomes",
    "reward_credit_environment_seed",
    "reward_credit_policy_noise_seed",
    "reward_tangent_landmark_index",
]
