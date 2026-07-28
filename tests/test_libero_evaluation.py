from __future__ import annotations

import pytest

from ember.libero_evaluation import (
    EvaluationContractError,
    aggregate_rows,
    batched_with_padding,
    partition_fixed_state_ids,
    validate_complete_rows,
)


def test_rank_striding_covers_every_fixed_state_once() -> None:
    assignments = [set(partition_fixed_state_ids(50, 8, rank)) for rank in range(8)]
    assert set.union(*assignments) == set(range(50))
    assert sum(len(values) for values in assignments) == 50
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(assignments)
        for right in assignments[index + 1 :]
    )
    assert {len(values) for values in assignments} == {6, 7}
    assert all(len(batched_with_padding(values, 4)) == 2 for values in assignments)


def test_complete_rows_preserve_raw_counts() -> None:
    rows = [
        {
            "task_id": task_id,
            "init_state_id": state_id,
            "language": f"task {task_id}",
            "success": state_id == 0,
        }
        for task_id in (2, 5)
        for state_id in range(3)
    ]
    complete = validate_complete_rows(reversed(rows), (2, 5), 3)
    metrics = aggregate_rows(complete)
    assert metrics["overall"] == {"successes": 2, "episodes": 6, "success_rate": 1 / 3}
    assert [item["successes"] for item in metrics["per_task"]] == [1, 1]
    with pytest.raises(EvaluationContractError, match="incomplete"):
        validate_complete_rows(rows[:-1], (2, 5), 3)
