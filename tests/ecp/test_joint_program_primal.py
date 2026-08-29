from types import SimpleNamespace

from ember.ecp.joint_program_primal.train_step import joint_task_group


def test_joint_task_schedule_is_role_balanced_and_task_equal() -> None:
    runtime = SimpleNamespace(
        config={
            "task_split": {
                "gradient_meta": [1, 8, 9, 32, 52],
                "gradient_target": [72, 73, 75, 93, 94],
            }
        }
    )
    groups = tuple(joint_task_group(runtime, step) for step in range(5))
    assert all(len(group) == len(set(group)) == 6 for group in groups)
    assert all(sum(task < 72 for task in group) == 3 for group in groups)
    counts = {
        task: sum(task in group for group in groups)
        for task in (1, 8, 9, 32, 52, 72, 73, 75, 93, 94)
    }
    assert set(counts.values()) == {3}
