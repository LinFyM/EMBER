from __future__ import annotations

from pathlib import Path

from ember.task_local_rl_protocol import (
    load_task_local_rl_config,
    rank_assignments,
    rollout_seed,
    select_adaptation_checkpoint,
    task_arms,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_task_local_units_and_seeds_are_matched_across_arms() -> None:
    config = load_task_local_rl_config(
        REPO_ROOT / "configs/task_local_lora_rl_v1.json"
    )
    units = task_arms(config)
    assert len(units) == 20
    assert {unit.arm for unit in units} == {"identity", "writer"}
    assignments = rank_assignments(units, 8)
    assert tuple(map(len, assignments)) == (3, 3, 3, 3, 2, 2, 2, 2)
    assert len({unit for values in assignments for unit in values}) == 20
    for task_id in config["role"]["task_ids"]:
        identity = rollout_seed(17, task_id, 3, 2)
        writer = rollout_seed(17, task_id, 3, 2)
        assert identity == writer


def test_task_local_selection_uses_segment_rate_then_earliest_cursor() -> None:
    candidates = (
        {"interaction_cursor": 40, "segment_successes": 1, "segment_rollouts": 8},
        {"interaction_cursor": 80, "segment_successes": 2, "segment_rollouts": 8},
        {"interaction_cursor": 120, "segment_successes": 1, "segment_rollouts": 4},
    )
    assert select_adaptation_checkpoint(candidates) is candidates[1]
