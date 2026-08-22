"""Load retained successful evaluator trajectories for functional re-query."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from ember.reward.protocol import RewardProtocolError
from ember.reward.rollout import RewardTrajectory


def load_successful_occupancy_trajectory(
    *,
    row: Mapping[str, Any],
    suite: str,
    task_id: int,
    global_task_id: int,
    replan_steps: int,
) -> RewardTrajectory:
    """Load one successful evaluator replay without inventing new supervision."""

    record = row.get("occupancy_trajectory", {})
    path = Path(str(record.get("path", "")))
    if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)):
        raise RewardProtocolError("successful occupancy sidecar changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    observations = tuple(payload.get("observations", ()))
    actions = tuple(payload.get("action_chunks", ()))
    seeds = tuple(int(value) for value in payload.get("policy_noise_seeds", ()))
    steps = int(payload.get("steps", -1))
    valid_steps = tuple(
        min(replan_steps, max(steps - index * replan_steps, 0))
        for index in range(len(observations))
    )
    if (
        payload.get("schema_version") != "ember_writer_occupancy_trajectory_v1"
        or not bool(payload.get("success"))
        or not bool(row.get("success"))
        or len(observations) != len(actions)
        or len(observations) != len(seeds)
        or len(observations) != int(record.get("replans", -1))
        or not valid_steps
        or min(valid_steps) <= 0
        or sum(valid_steps) != steps
    ):
        raise RewardProtocolError("successful occupancy replay changed")
    return RewardTrajectory(
        suite=suite,
        task_id=task_id,
        global_task_id=global_task_id,
        adaptation_seed=0,
        rollout_cursor=int(row["init_state_id"]),
        env_seed=int(row["env_seed"]),
        policy_seed_root=int(row["policy_seed_root"]),
        success=True,
        steps=steps,
        reward_sum=1.0,
        dummy_settling_steps=10,
        policy_noise_seeds=seeds,
        observations=observations,
        action_chunks=actions,
        valid_action_steps=valid_steps,
    )
