from __future__ import annotations

from pathlib import Path

from ember.writer_rl_protocol import (
    environment_seed,
    load_writer_rl_config,
    policy_seed,
    rank_rollout_count,
    rank_task_assignments,
    schedule_summary,
    source_task_ids,
    task_for_update,
    updates_per_cycle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_writer_rl_schedule_is_70_task_no_replacement() -> None:
    config = load_writer_rl_config(REPO_ROOT / "configs/writer_only_rl_v1.json")
    task_ids = source_task_ids(config)
    assignments = rank_task_assignments(task_ids, 8)
    assert tuple(map(len, assignments)) == (9, 9, 9, 9, 9, 9, 8, 8)
    assert updates_per_cycle(task_ids, 8) == 9
    scheduled = [
        task_for_update(task_ids, 8, rank, update)
        for update in range(9)
        for rank in range(8)
    ]
    observed = [value[0] for value in scheduled if value is not None]
    assert len(observed) == len(set(observed)) == 70
    assert set(observed) == set(task_ids)
    assert sum(value is None for value in scheduled) == 2
    assert rank_rollout_count(task_ids, 8, 0, 9, 4) == 36
    assert rank_rollout_count(task_ids, 8, 6, 9, 4) == 32


def test_writer_rl_coverage_and_seed_schedules_are_explicit() -> None:
    config = load_writer_rl_config(REPO_ROOT / "configs/writer_only_rl_v1.json")
    task_ids = source_task_ids(config)
    assert schedule_summary(task_ids, 8, 9, 4) == {
        "next_update": 9,
        "completed_full_task_cycles": 1,
        "cycle_slot_cursor": 0,
        "declared_task_count": 70,
        "tasks_with_interactions": 70,
        "min_rollouts_per_task": 4,
        "max_rollouts_per_task": 4,
        "total_rollouts": 280,
    }
    assert environment_seed(10, 2, 7, 3) == 200_713
    assert policy_seed(20, 2, 7, 3) == 200_723
