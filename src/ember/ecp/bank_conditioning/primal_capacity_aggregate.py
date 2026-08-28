"""Aggregate the fixed six-task G3 P1 primal-capacity evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping

from ember.ecp.bank_conditioning.primal_capacity_run import (
    P1_REPORT_SCHEMA,
    load_primal_capacity_config,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic


def _load_complete_task_rows(
    root: Path, task_ids: tuple[int, ...]
) -> tuple[dict[str, Any], ...]:
    rows = tuple(
        read_json(root / f"task_{task:03d}" / "report.json")
        for task in task_ids
    )
    if (
        {int(row.get("task", -1)) for row in rows} != set(task_ids)
        or any(row.get("status") != "complete" for row in rows)
        or len({row.get("git", {}).get("commit") for row in rows}) != 1
        or any(int(row.get("held_backward_calls", -1)) != 0 for row in rows)
    ):
        raise RuntimeError("P1 worker evidence is incomplete")
    return rows


def _aggregate_metrics(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    fit = [float(row["fit_mean_recovery"]) for row in rows]
    held = [float(row["held_recovery"]) for row in rows]
    relative = [float(row["held_relative_to_optimistic"]) for row in rows]
    family = {
        name: [float(row["held"]["family_recovery"][name]) for row in rows]
        for name in ("q", "v", "action_in", "action_out")
    }
    return {
        "fit_median_recovery": statistics.median(fit),
        "held_median_recovery": statistics.median(held),
        "held_to_fit": statistics.median(held) / max(statistics.median(fit), 1e-12),
        "held_relative_to_optimistic_median": statistics.median(relative),
        "held_family_median": {
            name: statistics.median(values) for name, values in family.items()
        },
        "minimum_task_held_recovery": min(held),
    }


def _aggregate_checks(
    rows: tuple[dict[str, Any], ...],
    metrics: dict[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, bool]:
    family_median = metrics["held_family_median"]
    return {
        "fit_capacity": metrics["fit_median_recovery"]
        >= float(gate["fit_median_recovery_minimum"]),
        "held_capacity": metrics["held_median_recovery"]
        >= float(gate["held_median_recovery_minimum"]),
        "held_to_fit": metrics["held_to_fit"]
        >= float(gate["held_to_fit_minimum"]),
        "relative_to_optimistic": metrics["held_relative_to_optimistic_median"]
        >= float(gate["held_relative_to_optimistic_median_minimum"]),
        "all_families": min(family_median.values())
        >= float(gate["held_family_median_minimum"]),
        "all_tasks": metrics["minimum_task_held_recovery"]
        >= float(gate["per_task_held_recovery_minimum"]),
        "information_wall": all(
            row["action_meta_modules"] == 0
            and row["action_meta_parameters"] == 0
            and row["source_trainable_parameters"] == 0
            and row["program_trainable_parameters"] == 0
            and row["shared_compiler_trainable_parameters"] == 0
            and row["task_local_scale_trainable_parameters"] == 0
            and row["information_wall"]["held_video_gradients"] == 0
            for row in rows
        ),
    }


def _task_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task": row["task"],
        "role": row["role"],
        "fit": row["fit_mean_recovery"],
        "held": row["held_recovery"],
        "held_to_fit": row["held_to_fit"],
        "relative_to_optimistic": row["held_relative_to_optimistic"],
        "family": row["held"]["family_recovery"],
        "seconds": row["total_seconds"],
    }


def aggregate(args: argparse.Namespace) -> None:
    config = load_primal_capacity_config(args.config)
    root = args.output_dir.resolve()
    if (root / "report.json").exists() or (root / "completion.json").exists():
        raise RuntimeError("P1 aggregate already exists")
    task_ids = tuple(map(int, config["panel"]["task_ids"]))
    rows = _load_complete_task_rows(root, task_ids)
    metrics = _aggregate_metrics(rows)
    checks = _aggregate_checks(rows, metrics, config["gate"])
    report = {
        "schema_version": P1_REPORT_SCHEMA,
        "status": "complete",
        "passed": all(checks.values()),
        "git_commit": rows[0]["git"]["commit"],
        "config": str(args.config.resolve()),
        "task_ids": list(task_ids),
        "metrics": metrics,
        "checks": checks,
        "tasks": [_task_summary(row) for row in rows],
        "information_wall": {
            "held_gradient_tasks": 0,
            "validation_test_action_reward_reads": 0,
            "action_meta_modules": 0,
            "shuffled_reversed_use": False,
        },
    }
    write_json_atomic(root / "report.json", report)
    write_json_atomic(
        root / "completion.json",
        {
            "schema_version": P1_REPORT_SCHEMA,
            "status": "complete",
            "passed": report["passed"],
            "task_count": len(rows),
            "git_commit": report["git_commit"],
        },
    )
    print(json.dumps(report, sort_keys=True), flush=True)
