"""Pure task and aggregate summaries for frozen Stage 1 support audits."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _mean(rows: Sequence[Mapping[str, Any]], name: str) -> float:
    return sum(float(row[name]) for row in rows) / len(rows)


def _panel_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    candidate = _mean(rows, "candidate_response")
    source = _mean(rows, "source_response")
    shared = _mean(rows, "shared_response")
    consensus = _mean(rows, "consensus_response")
    return {
        "panels": len(rows),
        "candidate_response": candidate,
        "source_response": source,
        "shared_response": shared,
        "consensus_response": consensus,
        "candidate_to_source_ratio": candidate / max(source, 1e-12),
        "candidate_to_shared_ratio": candidate / max(shared, 1e-12),
        "candidate_panel_wins_over_source": sum(
            float(row["candidate_response"]) < float(row["source_response"])
            for row in rows
        ),
        "candidate_panel_wins_over_shared": sum(
            float(row["candidate_response"]) < float(row["shared_response"])
            for row in rows
        ),
    }


def summarize_task_policy_support(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "all": _panel_summary(rows),
        "successful": _panel_summary(
            [row for row in rows if row["kind"] == "successful"]
        ),
        "learner": _panel_summary(
            [row for row in rows if row["kind"] == "learner"]
        ),
    }


def _aggregate_tasks(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def aggregate(kind: str) -> dict[str, Any] | None:
        summaries = [task["summary"][kind] for task in tasks]
        summaries = [summary for summary in summaries if summary is not None]
        if not summaries:
            return None
        candidate = _mean(summaries, "candidate_response")
        source = _mean(summaries, "source_response")
        shared = _mean(summaries, "shared_response")
        return {
            "tasks": len(summaries),
            "panels": sum(int(summary["panels"]) for summary in summaries),
            "candidate_response": candidate,
            "source_response": source,
            "shared_response": shared,
            "consensus_response": _mean(summaries, "consensus_response"),
            "candidate_to_source_ratio": candidate / max(source, 1e-12),
            "candidate_to_shared_ratio": candidate / max(shared, 1e-12),
            "candidate_tasks_better_than_source": sum(
                float(summary["candidate_response"])
                < float(summary["source_response"])
                for summary in summaries
            ),
            "candidate_tasks_better_than_shared": sum(
                float(summary["candidate_response"])
                < float(summary["shared_response"])
                for summary in summaries
            ),
        }

    return {
        "all": aggregate("all"),
        "successful": aggregate("successful"),
        "learner": aggregate("learner"),
    }


def summarize_policy_support_audit(
    *, tasks: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit = [task for task in tasks if task["fold_role"] == "fit"]
    held = [task for task in tasks if task["fold_role"] == "held_transform_only"]
    aggregates = {"fit19": _aggregate_tasks(fit), "held5": _aggregate_tasks(held)}
    fit_all = aggregates["fit19"]["all"]
    held_all = aggregates["held5"]["all"]
    conditions = {
        "fit_aggregate_better_than_source": fit_all["candidate_response"]
        < fit_all["source_response"],
        "fit_aggregate_better_than_shared": fit_all["candidate_response"]
        < fit_all["shared_response"],
        "held_aggregate_better_than_source": held_all["candidate_response"]
        < held_all["source_response"],
        "held_aggregate_better_than_shared": held_all["candidate_response"]
        < held_all["shared_response"],
        "fit_task_breadth_over_source": fit_all["candidate_tasks_better_than_source"]
        >= int(thresholds["minimum_fit_tasks_better_than_source"]),
        "fit_task_breadth_over_shared": fit_all["candidate_tasks_better_than_shared"]
        >= int(thresholds["minimum_fit_tasks_better_than_shared"]),
        "held_task_breadth_over_source": held_all[
            "candidate_tasks_better_than_source"
        ]
        >= int(thresholds["minimum_held_tasks_better_than_source"]),
        "held_task_breadth_over_shared": held_all[
            "candidate_tasks_better_than_shared"
        ]
        >= int(thresholds["minimum_held_tasks_better_than_shared"]),
    }
    return aggregates, {
        "thresholds": {name: int(value) for name, value in thresholds.items()},
        "conditions": conditions,
        "passed": all(conditions.values()),
    }
