"""Strict cross-method and per-task-only Writer benchmark comparisons."""

from __future__ import annotations

import copy
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from ember.pi05_eval_queue import publish_json_exclusive
from ember.pi05_eval.paired_metrics import paired_transition_summary, summarize_panel


BENCHMARK_COMPARISON_SCHEMA = "ember_pi05_writer_benchmark_comparison_v1"
V6_FAST_143_TASKS = (
    ("libero_spatial", 1),
    ("libero_spatial", 3),
    ("libero_object", 1),
    ("libero_object", 3),
    ("libero_goal", 3),
    ("libero_goal", 6),
    ("libero_10", 1),
    ("libero_10", 2),
)
V6_FAST_143_PER_TASK = (0, 3, 46, 37, 0, 36, 20, 1)


def v6_fast_143_reference(
    task_keys: Sequence[tuple[str, int]],
) -> dict[str, Any]:
    """Return the retained count-only reference without inventing row pairing."""

    if tuple(task_keys) != V6_FAST_143_TASKS:
        from ember.pi05_assets import Pi05EvaluationError

        raise Pi05EvaluationError("v6-fast reference requires the canonical 8 tasks")
    return {
        "successes": sum(V6_FAST_143_PER_TASK),
        "comparison_scope": "per_task_counts_only",
        "reason_not_episode_paired": (
            "historical v6-fast teacher schedule differs from current strict panels"
        ),
        "per_task": [
            {"suite": suite, "task_id": task_id, "successes": successes}
            for (suite, task_id), successes in zip(
                task_keys, V6_FAST_143_PER_TASK, strict=True
            )
        ],
        "provenance": {
            "label": "v6-fast task-complete macro400",
            "aggregate": 143,
            "source": "docs/research_history.md",
        },
    }


def _benchmark_pairing(
    reference: Mapping[tuple[str, int, int], Mapping[str, Any]],
    candidate: Mapping[tuple[str, int, int], Mapping[str, Any]],
    shared: Any,
) -> None:
    if set(reference) != set(candidate):
        shared._fail("benchmark roots do not contain identical episode keys")
    row_fields = (
        "language",
        "split_role",
        "env_seed",
        "policy_seed_root",
    )
    writer_fields = (
        "teacher_reference_demo_indices",
        "teacher_demo_indices",
        "teacher_video_order_seeds",
        "video_suite",
        "video_task_id",
        "video_global_task_id",
        "video_split_role",
    )
    for key in reference:
        left, right = reference[key], candidate[key]
        if any(left.get(field) != right.get(field) for field in row_fields) or any(
            left.get("writer", {}).get(field) != right.get("writer", {}).get(field)
            for field in writer_fields
        ) or not shared._common_noise_prefix(left, right):
            shared._fail(
                "strict benchmark comparison changed state, RNG, language, or actual teacher video"
            )


def _benchmark_contract_projection(result: Mapping[str, Any]) -> dict[str, Any]:
    projection = copy.deepcopy(result["paired_control"])
    projection.pop("git", None)
    projection.pop("parallel", None)
    projection.pop("writer", None)
    normalization = projection.get("normalization", {})
    if isinstance(normalization, dict) and isinstance(normalization.get("path"), str):
        normalization["path"] = _config_authority_location(normalization["path"])
    tokenizer = projection.get("tokenizer", {})
    if isinstance(tokenizer, dict) and isinstance(
        tokenizer.get("manifest_path"), str
    ):
        tokenizer["manifest_path"] = _config_authority_location(
            tokenizer["manifest_path"]
        )
    return projection


def _config_authority_location(value: str) -> str:
    """Ignore only the frozen-worktree prefix of repository config authorities."""

    parts = PurePosixPath(value).parts
    indices = [index for index, part in enumerate(parts) if part == "configs"]
    if not indices:
        return value
    return PurePosixPath(*parts[indices[-1] :]).as_posix()


def _per_task_reference(
    value: Mapping[str, Any],
    *,
    expected_tasks: Sequence[tuple[str, int]],
    shared: Any,
) -> dict[str, Any]:
    rows = list(value.get("per_task", ()))
    observed_tasks = [(str(row.get("suite", "")), int(row.get("task_id", -1))) for row in rows]
    successes = [int(row.get("successes", -1)) for row in rows]
    total = int(value.get("successes", -1))
    if (
        value.get("comparison_scope") != "per_task_counts_only"
        or observed_tasks != list(expected_tasks)
        or any(count < 0 or count > 50 for count in successes)
        or sum(successes) != total
        or not value.get("reason_not_episode_paired")
    ):
        shared._fail("per-task benchmark reference is incomplete or claims false pairing")
    return {
        "successes": total,
        "nonzero_task_breadth": sum(count > 0 for count in successes),
        "per_task": [
            {"suite": suite, "task_id": task_id, "successes": count}
            for (suite, task_id), count in zip(expected_tasks, successes, strict=True)
        ],
        "comparison_scope": "per_task_counts_only",
        "reason_not_episode_paired": str(value["reason_not_episode_paired"]),
        "provenance": dict(value.get("provenance", {})),
    }


def _per_task_delta(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    reference_rows = {
        (str(row["suite"]), int(row["task_id"])): int(row["successes"])
        for row in reference["per_task"]
    }
    candidate_rows = {
        (str(row["suite"]), int(row["task_id"])): int(row["successes"])
        for row in candidate["per_task"]
    }
    per_task = [
        {
            "suite": suite,
            "task_id": task_id,
            "reference_successes": reference_rows[(suite, task_id)],
            "candidate_successes": count,
            "success_delta": count - reference_rows[(suite, task_id)],
        }
        for (suite, task_id), count in candidate_rows.items()
    ]
    suites = []
    for suite in dict.fromkeys(row["suite"] for row in per_task):
        selected = [row for row in per_task if row["suite"] == suite]
        suites.append(
            {
                "suite": suite,
                "reference_successes": sum(row["reference_successes"] for row in selected),
                "candidate_successes": sum(row["candidate_successes"] for row in selected),
                "success_delta": sum(row["success_delta"] for row in selected),
            }
        )
    return {
        "success_delta": int(candidate["overall"]["successes"])
        - int(reference["successes"]),
        "breadth_delta": int(candidate["nonzero_task_breadth"])
        - int(reference["nonzero_task_breadth"]),
        "per_task": per_task,
        "per_suite": suites,
        "gained_lost_churn": None,
        "pairing_warning": (
            "episode identities are not strictly paired; gained, lost, and churn are unavailable"
        ),
    }


def benchmark_reference_comparison(
    candidate_result: Mapping[str, Any],
    *,
    strict_references: Mapping[str, Mapping[str, Any]] | None = None,
    per_task_references: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare one correct400 candidate to paired roots and count-only history."""

    from ember.pi05_eval import analysis as shared

    strict_references = strict_references or {}
    per_task_references = per_task_references or {}
    if not strict_references and not per_task_references:
        shared._fail("benchmark comparison requires at least one reference")
    labels = [*strict_references, *per_task_references]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        shared._fail("benchmark reference labels are empty or duplicated")
    candidate_index = shared._formal_panel_index(candidate_result)
    if candidate_result["adapter"].get("video_condition") != "correct":
        shared._fail("benchmark comparison candidate must be a correct-video root")
    candidate_rows = list(candidate_index.values())
    candidate_summary = summarize_panel(candidate_rows)
    candidate_tasks = [
        (str(row["suite"]), int(row["task_id"]))
        for row in candidate_summary["per_task"]
    ]
    paired = {}
    for label, result in strict_references.items():
        reference_index = shared._formal_panel_index(result)
        if result["adapter"].get("video_condition") != "correct":
            shared._fail("strict benchmark reference must be a correct-video root")
        if _benchmark_contract_projection(result) != _benchmark_contract_projection(
            candidate_result
        ):
            shared._fail("strict benchmark comparison changed its shared policy contract")
        _benchmark_pairing(reference_index, candidate_index, shared)
        reference_rows = list(reference_index.values())
        paired[label] = {
            "reference_family": shared._writer_family(result["adapter"])[0],
            "reference": summarize_panel(reference_rows),
            "reference_to_candidate": paired_transition_summary(
                reference_rows, candidate_rows
            ),
            "comparison_scope": "strict_episode_paired",
        }
    count_only = {}
    for label, value in per_task_references.items():
        reference = _per_task_reference(
            value, expected_tasks=candidate_tasks, shared=shared
        )
        count_only[label] = {
            "reference": reference,
            "reference_to_candidate": _per_task_delta(reference, candidate_summary),
        }
    adapter = candidate_result["adapter"]
    return {
        "schema_version": BENCHMARK_COMPARISON_SCHEMA,
        "candidate": {
            "method_family": shared._writer_family(adapter)[0],
            "method_macro": int(adapter["writer_asset"]["method_macro"]),
            "summary": candidate_summary,
        },
        "strict_paired_references": paired,
        "per_task_only_references": count_only,
        "metric_definitions": {
            "retained_success": "reference and candidate both succeed on the identical episode",
            "gained": "reference fails and candidate succeeds on the identical episode",
            "lost": "reference succeeds and candidate fails on the identical episode",
            "churn": "gained plus lost",
            "nonzero_task_breadth": "tasks with at least one success",
            "count_only_boundary": (
                "per-task success deltas do not imply episode-level gained, lost, or churn"
            ),
        },
    }


def analyze_benchmark_references(
    candidate_root: Path,
    *,
    strict_reference_roots: Mapping[str, Path],
    output_path: Path,
    include_v6_fast_143: bool = True,
) -> dict[str, Any]:
    """Reaggregate retained roots and publish one immutable comparison artifact."""

    from ember.pi05_eval import analysis as shared

    roots = [
        candidate_root.resolve(),
        *(path.resolve() for path in strict_reference_roots.values()),
    ]
    if len(set(roots)) != len(roots):
        shared._fail("benchmark candidate and reference roots must be unique")
    validated = shared._validated_roots(roots)
    candidate = validated[str(roots[0])]
    shared._formal_panel_index(candidate)
    _, paired, _ = shared._formal_adapter(candidate)
    _, task_keys = shared._formal_tasks(paired)
    per_task = (
        {"v6_fast_143": v6_fast_143_reference(task_keys)}
        if include_v6_fast_143
        else {}
    )
    result = benchmark_reference_comparison(
        candidate,
        strict_references={
            label: validated[str(path.resolve())]
            for label, path in strict_reference_roots.items()
        },
        per_task_references=per_task,
    )
    publish_json_exclusive(output_path.resolve(), result)
    return result
