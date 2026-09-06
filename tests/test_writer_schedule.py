from ember.writer.task_execution import assignment_makespan, cost_balanced_task_assignment
from ember.writer.task_schedule import (
    counted_task_group, task_group_counts, task_occurrence_schedule, training_video_demos,
)


def test_task_batch_size_and_role_ratio_are_experiment_settings() -> None:
    meta = tuple(range(10))
    target = tuple(range(20, 26))
    assert task_group_counts(
        {
            "global_tasks_per_update": 5,
            "tasks_per_update_by_role": {"meta": 4, "target": 1},
        },
        meta=meta,
        target=target,
    ) == (4, 1)
    group = counted_task_group((meta, target), (4, 1), 0, seed=17)
    assert len(group) == 5
    assert len(set(group).intersection(meta)) == 4
    assert len(set(group).intersection(target)) == 1


def test_task_local_occurrence_drives_k_without_global_step_aliasing() -> None:
    groups = ((1, 2), (3, 4), (1, 3), (2, 4), (1, 4))
    occurrences = task_occurrence_schedule(groups)
    assert [row.get(1) for row in occurrences if 1 in row] == [0, 1, 2]
    selected = [
        training_video_demos(
            (0, 1, 2, 3),
            task_occurrence=row[1],
            task=1,
            cardinalities=(1, 2, 4),
            seed=19,
        )
        for row in occurrences
        if 1 in row
    ]
    assert {len(value) for value in selected} == {1, 2, 4}


def test_dynamic_cost_assignment_reduces_tail_without_changing_tasks() -> None:
    costs = {0: 19, 1: 17, 2: 13, 3: 11, 4: 7, 5: 5}
    eligibility = {task: (0, 1, 2) for task in costs}
    assignment = cost_balanced_task_assignment(
        tuple(costs), costs, eligibility, world_size=3
    )
    assert {task for row in assignment for task in row} == set(costs)
    assert assignment_makespan(assignment, costs) <= 25
