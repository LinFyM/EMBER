from __future__ import annotations

import torch

from ember.libero_reward_rollout import RewardTrajectory
from ember.task_local_rl_update import equal_episode_flow_loss


class _PerChunkPolicy(torch.nn.Module):
    def forward(self, batch, reduction="mean"):
        assert reduction == "none"
        losses = batch["marker"].flatten()
        return losses, {"loss": float(losses.mean())}


def _trajectory(markers: tuple[float, ...]) -> RewardTrajectory:
    return RewardTrajectory(
        task_id=7,
        env_seed=11,
        policy_seed=13,
        success=True,
        steps=50,
        reward_sum=1.0,
        observations=tuple(
            {"marker": torch.tensor([[marker]])} for marker in markers
        ),
        action_chunks=tuple(
            torch.zeros((1, 2, 1)) for _ in markers
        ),
        valid_action_steps=tuple(2 for _ in markers),
    )


def test_task_local_loss_weights_successful_episodes_equally() -> None:
    loss, details = equal_episode_flow_loss(
        _PerChunkPolicy(),
        (_trajectory((1.0, 3.0)), _trajectory((9.0,))),
        torch.device("cpu"),
    )
    torch.testing.assert_close(loss, torch.tensor(5.5))
    assert details["successful_episodes"] == 2
    assert details["successful_chunks"] == 3
