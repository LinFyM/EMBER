"""Deterministic statistics for already validated paired rollout rows."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_target_data import SUITE_ORDER


EpisodeKey = tuple[str, int, int]
TaskKey = tuple[str, int]


def _fail(message: str) -> None:
    raise Pi05EvaluationError(message)


def suite_sort_key(suite: str) -> tuple[int, str]:
    try:
        return SUITE_ORDER.index(suite), suite
    except ValueError:
        return len(SUITE_ORDER), suite


def episode_key(row: Mapping[str, Any]) -> EpisodeKey:
    return str(row["suite"]), int(row["task_id"]), int(row["init_state_id"])


def task_key(row: Mapping[str, Any]) -> TaskKey:
    return str(row["suite"]), int(row["task_id"])


def index_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[EpisodeKey, Mapping[str, Any]]:
    indexed: dict[EpisodeKey, Mapping[str, Any]] = {}
    for row in rows:
        key = episode_key(row)
        if key in indexed or type(row.get("success")) is not bool:
            _fail("paired analysis rows are duplicated or have an invalid outcome")
        indexed[key] = row
    return indexed


def _task_sort_key(row: Mapping[str, Any]) -> tuple[int, str, int]:
    suite = str(row["suite"])
    return (*suite_sort_key(suite), int(row["task_id"]))


def _summary_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(row["success"]) for row in rows)
    episodes = len(rows)
    return {
        "successes": successes,
        "episodes": episodes,
        "success_rate": successes / episodes if episodes else None,
    }


def summarize_panel(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one already validated, uniquely keyed rollout panel."""

    indexed = index_rows(rows)
    task_keys = sorted(
        {task_key(row) for row in indexed.values()},
        key=lambda key: (*suite_sort_key(key[0]), key[1]),
    )
    per_task: list[dict[str, Any]] = []
    for suite, task_id in task_keys:
        selected = [
            row for row in indexed.values() if task_key(row) == (suite, task_id)
        ]
        first = selected[0]
        per_task.append(
            {
                "suite": suite,
                "task_id": task_id,
                "split_role": first.get("split_role"),
                "language": first.get("language"),
                **_summary_row(selected),
            }
        )
    per_suite = []
    for suite in sorted({key[0] for key in task_keys}, key=suite_sort_key):
        selected = [
            row for row in indexed.values() if str(row["suite"]) == suite
        ]
        per_suite.append({"suite": suite, **_summary_row(selected)})
    top3 = sorted(
        per_task,
        key=lambda row: (-int(row["successes"]), *_task_sort_key(row)),
    )[:3]
    total_successes = sum(int(row["successes"]) for row in per_task)
    top3_successes = sum(int(row["successes"]) for row in top3)
    return {
        "overall": _summary_row(list(indexed.values())),
        "per_task": per_task,
        "per_suite": per_suite,
        "task_breadth": {
            f"at_least_{threshold}": sum(
                int(row["successes"]) >= threshold for row in per_task
            )
            for threshold in (1, 5, 10)
        },
        "nonzero_task_breadth": sum(
            int(row["successes"]) > 0 for row in per_task
        ),
        "top3_tasks": top3,
        "top3_successes": top3_successes,
        "top3_success_share": (
            top3_successes / total_successes if total_successes else None
        ),
    }


def exact_mcnemar_two_sided_p(gained: int, lost: int) -> float:
    """Exact two-sided binomial McNemar p-value for discordant pairs."""

    if gained < 0 or lost < 0:
        _fail("McNemar discordant counts must be non-negative")
    discordant = gained + lost
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(gained, lost) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def _outcome_counts(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    retained_success = sum(
        bool(left["success"]) and bool(right["success"])
        for left, right in pairs
    )
    gained = sum(
        not bool(left["success"]) and bool(right["success"])
        for left, right in pairs
    )
    lost = sum(
        bool(left["success"]) and not bool(right["success"])
        for left, right in pairs
    )
    retained_failure = len(pairs) - retained_success - gained - lost
    union = retained_success + gained + lost
    return {
        "episodes": len(pairs),
        "retained_success": retained_success,
        "gained": gained,
        "lost": lost,
        "retained_failure": retained_failure,
        "net": gained - lost,
        "churn": gained + lost,
        "churn_rate": (gained + lost) / len(pairs) if pairs else None,
        "success_set_jaccard": retained_success / union if union else 1.0,
        "mcnemar_exact_two_sided_p": exact_mcnemar_two_sided_p(gained, lost),
    }


def paired_transition_summary(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare two panels; gained means left failure became right success."""

    left, right = index_rows(left_rows), index_rows(right_rows)
    if set(left) != set(right):
        _fail("paired panels do not have identical episode keys")
    keys = sorted(
        left,
        key=lambda key: (*suite_sort_key(key[0]), key[1], key[2]),
    )
    pairs = [(left[key], right[key]) for key in keys]
    task_keys = sorted(
        {key[:2] for key in keys},
        key=lambda key: (*suite_sort_key(key[0]), key[1]),
    )
    per_task = []
    for suite, task_id in task_keys:
        selected = [
            pair
            for key, pair in zip(keys, pairs)
            if key[:2] == (suite, task_id)
        ]
        per_task.append(
            {"suite": suite, "task_id": task_id, **_outcome_counts(selected)}
        )
    per_suite = []
    for suite in sorted({key[0] for key in task_keys}, key=suite_sort_key):
        selected = [
            pair for key, pair in zip(keys, pairs) if key[0] == suite
        ]
        per_suite.append({"suite": suite, **_outcome_counts(selected)})
    return {
        "overall": _outcome_counts(pairs),
        "per_task": per_task,
        "per_suite": per_suite,
    }


def control_outcome_summary(
    correct: Sequence[Mapping[str, Any]],
    control: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Rename control-to-correct transitions for a causality comparison."""

    transition = paired_transition_summary(control, correct)

    def rename(value: Mapping[str, Any]) -> dict[str, Any]:
        copied = {
            key: value[key]
            for key in (
                "episodes",
                "churn",
                "churn_rate",
                "success_set_jaccard",
                "mcnemar_exact_two_sided_p",
            )
        }
        return {
            **copied,
            "both_success": value["retained_success"],
            "correct_only": value["gained"],
            "control_only": value["lost"],
            "both_failure": value["retained_failure"],
            "correct_minus_control": value["net"],
        }

    per_task = [
        {
            "suite": row["suite"],
            "task_id": row["task_id"],
            **rename(row),
        }
        for row in transition["per_task"]
    ]
    positive = [
        int(row["correct_minus_control"])
        for row in per_task
        if int(row["correct_minus_control"]) > 0
    ]
    return {
        "overall": rename(transition["overall"]),
        "per_task": per_task,
        "per_suite": [
            {"suite": row["suite"], **rename(row)}
            for row in transition["per_suite"]
        ],
        "positive_contributing_task_count": len(positive),
        "largest_positive_task_contribution_share": (
            max(positive) / sum(positive) if positive else None
        ),
    }
