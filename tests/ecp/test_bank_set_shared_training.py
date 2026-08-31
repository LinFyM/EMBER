from types import SimpleNamespace

import pytest
import torch

from ember.ecp.joint_program_primal.bank_set_shared_training import (
    ALL_INTERACTION_TASKS,
    ARM_SCHEDULE,
    GRADIENT_META_TASKS,
    GRADIENT_TARGET_TASKS,
    GRADIENT_TASKS,
    HELD_INTERACTION_TASKS,
    WRONG_TASK_RING,
    _apply_task_profile,
    _validate_shared_config,
    _validate_task_cursors,
    balanced_task_assignments,
    shared_task_group,
    task_cursor_counts,
)


def test_s2_schedule_is_role_task_and_world_size_invariant() -> None:
    groups = tuple(shared_task_group(step) for step in range(4))
    assert all(len(group) == len(set(group)) == 6 for group in groups)
    assert all(len(set(group).intersection(GRADIENT_META_TASKS)) == 3 for group in groups)
    assert all(len(set(group).intersection(GRADIENT_TARGET_TASKS)) == 3 for group in groups)
    assert not any(set(group).intersection(HELD_INTERACTION_TASKS) for group in groups)
    counts = {task: sum(task in group for group in groups) for task in GRADIENT_TASKS}
    assert set(counts.values()) == {3}

    for world_size in range(1, 7):
        group = groups[0]
        costs = {task: task % 37 + 1 for task in group}
        assignments = balanced_task_assignments(group, costs, world_size)
        assert {task for row in assignments for task in row} == set(group)
        assert sum(len(row) for row in assignments) == 6
        assert sum(1.0 / 6.0 for row in assignments for _ in row) == pytest.approx(1.0)


def test_s2_each_task_has_its_own_four_beat_arm_cursor() -> None:
    observed = {task: [] for task in GRADIENT_TASKS}
    for step in range(12):
        before = task_cursor_counts(step)
        for task in shared_task_group(step):
            observed[task].append(ARM_SCHEDULE[before[task] % len(ARM_SCHEDULE)])
    for arms in observed.values():
        assert tuple(arms[:8]) == 2 * ARM_SCHEDULE
    assert task_cursor_counts(4) == {task: 3 for task in GRADIENT_TASKS}
    assert task_cursor_counts(8) == {task: 6 for task in GRADIENT_TASKS}


def test_s2_cursor_buffer_is_checkpoint_reconstructible() -> None:
    runtime = SimpleNamespace(
        optimizer_steps=8,
        writer_state=SimpleNamespace(
            task_arm_cursors=torch.tensor(
                [task_cursor_counts(8)[task] for task in GRADIENT_TASKS],
                dtype=torch.int64,
            )
        ),
    )
    assert _validate_task_cursors(runtime) == task_cursor_counts(8)
    runtime.writer_state.task_arm_cursors[0] += 1
    with pytest.raises(ValueError, match="disagree with macro"):
        _validate_task_cursors(runtime)


def _config() -> dict:
    profiles = {
        str(task): {
            "replay_frame_chunk_size": 16,
            "interaction_group_batch_size": 8,
            "functional_policy_microbatch_size": 2,
        }
        for task in ALL_INTERACTION_TASKS
    }
    profiles["1"].update(
        replay_frame_chunk_size=4, interaction_group_batch_size=16
    )
    profiles["93"].update(
        replay_frame_chunk_size=32, interaction_group_batch_size=4
    )
    profiles["94"].update(
        replay_frame_chunk_size=32, interaction_group_batch_size=4
    )
    return {
        "task_split": {
            "gradient_meta": list(GRADIENT_META_TASKS),
            "gradient_target": list(GRADIENT_TARGET_TASKS),
        },
        "shared_training": {
            "gradient_task_ids": list(GRADIENT_TASKS),
            "held_interaction_task_ids": list(HELD_INTERACTION_TASKS),
            "wrong_task_by_task": {
                str(task): wrong for task, wrong in WRONG_TASK_RING.items()
            },
            "arm_schedule": list(ARM_SCHEDULE),
            "task_profiles": profiles,
        },
        "model": {},
        "optimization": {},
    }


def test_s2_profiles_switch_b1_per_task_and_held_never_enters_train_ring() -> None:
    config = _config()
    _validate_shared_config(config)
    operator = SimpleNamespace(covariance_frame_chunk=-1)
    runtime = SimpleNamespace(
        config=config,
        compiler=SimpleNamespace(bank_operator=operator),
    )
    assert _apply_task_profile(runtime, 1)["replay_frame_chunk_size"] == 4
    assert operator.covariance_frame_chunk == 4
    assert _apply_task_profile(runtime, 8)["interaction_group_batch_size"] == 8
    assert operator.covariance_frame_chunk == 16
    assert _apply_task_profile(runtime, 94)["interaction_group_batch_size"] == 4
    assert operator.covariance_frame_chunk == 32

    config["shared_training"]["wrong_task_by_task"]["8"] = 1
    with pytest.raises(ValueError, match="task contract changed"):
        _validate_shared_config(config)
