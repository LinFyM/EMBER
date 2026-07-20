from __future__ import annotations

import torch

from ember.libero_reward_rollout import (
    RewardTrajectory,
    successful_trajectory_batch,
)


def _trajectory(
    *, success: bool, steps: int, valid: tuple[int, ...]
) -> RewardTrajectory:
    observations = tuple(
        {"observation.state": torch.full((1, 3), float(index))}
        for index in range(len(valid))
    )
    actions = tuple(torch.ones(1, 50, 7) for _ in valid)
    if not success:
        observations = ()
        actions = ()
        valid = ()
    return RewardTrajectory(
        task_id=7,
        env_seed=11,
        policy_seed=13,
        success=success,
        steps=steps,
        reward_sum=float(success),
        observations=observations,
        action_chunks=actions,
        valid_action_steps=valid,
    )


def test_successful_trajectory_batch_masks_unexecuted_tail() -> None:
    trajectories = (
        _trajectory(success=False, steps=400, valid=()),
        _trajectory(success=True, steps=63, valid=(50, 13)),
        _trajectory(success=True, steps=7, valid=(7,)),
    )
    batch, episode_ids = successful_trajectory_batch(
        trajectories, torch.device("cpu")
    )
    assert batch["action"].shape == (3, 50, 7)
    assert batch["observation.state"].shape == (3, 3)
    assert batch["action_is_pad"].sum(dim=1).tolist() == [0, 37, 43]
    assert episode_ids.tolist() == [0, 0, 1]


def test_failed_trajectory_ledger_retains_seeds_but_not_observations() -> None:
    trajectory = _trajectory(success=False, steps=400, valid=())
    assert trajectory.observations == ()
    assert trajectory.ledger_row() == {
        "task_id": 7,
        "env_seed": 11,
        "policy_seed": 13,
        "success": False,
        "steps": 400,
        "reward_sum": 0.0,
        "action_chunk_count": 0,
        "valid_action_steps": [],
    }
