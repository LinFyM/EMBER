"""Aggregation and pre-registered Gates for F2/F3 mapping evidence."""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Mapping, Sequence

from ember.ecp.shared_compiler_assets import load_shared_compiler_config
from ember.ecp.bank_conditioning.mapping import load_mapping_split
from ember.ecp.bank_conditioning.mapping_eval_runtime import (
    EVALUATION_SCHEMA,
    FAMILY_NAMES,
    SPLIT_NAMES,
    balanced_mapping_assignments,
    labeled_mapping_conditions,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(map(float, values))
    if not ordered or not 0.0 <= fraction <= 1.0:
        raise ValueError("mapping evaluation quantile input changed")
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "median": _quantile(values, 0.5),
        "p10": _quantile(values, 0.1),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize_mapping_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        selected = [row for row in rows if row["split"] == split_name]
        by_task: dict[int, list[Mapping[str, Any]]] = {}
        for row in selected:
            by_task.setdefault(int(row["authority_id"]), []).append(row)
        task_rows = []
        for task, task_conditions in sorted(by_task.items()):
            task_rows.append(
                {
                    "authority_id": task,
                    "condition_count": len(task_conditions),
                    "mean_best_recovery": sum(
                        float(row["mean_best_recovery"])
                        for row in task_conditions
                    )
                    / len(task_conditions),
                    "family_recovery": {
                        family: sum(
                            float(row["best_family_recovery"][family])
                            for row in task_conditions
                        )
                        / len(task_conditions)
                        for family in FAMILY_NAMES
                    },
                }
            )
        result[split_name] = {
            "condition_count": len(selected),
            "task_count": len(task_rows),
            "condition_recovery": _distribution(
                [float(row["mean_best_recovery"]) for row in selected]
            ),
            "task_recovery": _distribution(
                [float(row["mean_best_recovery"]) for row in task_rows]
            ),
            "task_family_recovery": {
                family: _distribution(
                    [float(row["family_recovery"][family]) for row in task_rows]
                )
                for family in FAMILY_NAMES
            },
            "tasks": task_rows,
        }
    return result


def _gate_report(
    *,
    phase: str,
    macro: int,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
) -> dict[str, Any]:
    gate = config["mapping_gate"]
    held = summary["video_holdout"]["task_recovery"]
    task_held = summary["task_holdout"]["task_recovery"]
    families = summary["video_holdout"]["task_family_recovery"]
    if phase == "f2":
        checks = {
            "held_video_median": held["median"]
            >= float(gate["f2_held_video_median_minimum"]),
            "held_video_each_family_median": min(
                float(values["median"]) for values in families.values()
            )
            >= float(gate["f2_family_minimum"]),
            "task_holdout_task_mean_median": task_held["median"]
            >= float(gate["f2_task_holdout_minimum"]),
        }
        return {
            "phase": phase,
            "primary_checks": checks,
            "primary_pass": all(checks.values()),
            "gate_pass": all(checks.values()),
            "adjacent_checkpoint": "not_required",
        }

    fit_median = float(summary["fit"]["task_recovery"]["median"])
    held_to_fit = float(held["median"]) / fit_median if fit_median > 0 else None
    checks = {
        "held_video_median": held["median"]
        >= float(gate["f3_held_video_median_minimum"]),
        "held_video_p10": held["p10"]
        >= float(gate["f3_held_video_p10_minimum"]),
        "held_to_fit_ratio": held_to_fit is not None
        and held_to_fit >= float(gate["f3_held_to_fit_minimum"]),
    }
    stability: dict[str, Any] = {"status": "pending_adjacent_checkpoint"}
    stability_pass = False
    if previous is not None:
        checkpoints = list(map(int, config["formal_run"]["checkpoint_macros"]))
        previous_macro = int(previous.get("checkpoint_macro", -1))
        adjacent = (
            previous_macro in checkpoints
            and macro in checkpoints
            and checkpoints.index(macro) == checkpoints.index(previous_macro) + 1
        )
        current_tasks = {
            int(row["authority_id"]): float(row["mean_best_recovery"])
            for row in summary["video_holdout"]["tasks"]
        }
        previous_tasks = {
            int(row["authority_id"]): float(row["mean_best_recovery"])
            for row in previous["summary"]["video_holdout"]["tasks"]
        }
        deltas = [
            abs(current_tasks[task] - previous_tasks[task])
            for task in sorted(current_tasks)
        ] if set(current_tasks) == set(previous_tasks) else []
        median_delta = _quantile(deltas, 0.5) if deltas else math.inf
        median_drop = float(
            previous["summary"]["video_holdout"]["task_recovery"]["median"]
        ) - float(held["median"])
        rule = gate["adjacent_checkpoint_stability"]
        previous_primary = bool(previous.get("gate", {}).get("primary_pass"))
        stability_pass = (
            adjacent
            and previous_primary
            and all(checks.values())
            and median_delta
            <= float(rule["maximum_median_absolute_task_delta"])
            and median_drop <= float(rule["maximum_held_median_drop"])
        )
        stability = {
            "status": "evaluated",
            "previous_checkpoint_macro": previous_macro,
            "checkpoint_macros_are_adjacent": adjacent,
            "both_primary_pass": previous_primary and all(checks.values()),
            "median_absolute_task_delta": median_delta,
            "held_median_drop": median_drop,
            "pass": stability_pass,
        }
    return {
        "phase": phase,
        "held_to_fit_ratio": held_to_fit,
        "primary_checks": checks,
        "primary_pass": all(checks.values()),
        "adjacent_checkpoint": stability,
        "gate_pass": all(checks.values()) and stability_pass,
    }


def aggregate_mapping_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    config = load_shared_compiler_config(args.config)
    split = load_mapping_split(config, asset_root=args.asset_root)
    expected = balanced_mapping_assignments(
        labeled_mapping_conditions(split), args.worker_count
    )
    rows: list[dict[str, Any]] = []
    worker_records = []
    expected_keys = set()
    for worker, assigned in enumerate(expected):
        contract = read_json(args.output_dir / f"worker_{worker:02d}_contract.json")
        completion = read_json(
            args.output_dir / f"worker_{worker:02d}_completion.json"
        )
        worker_rows = [
            json.loads(line)
            for line in (
                args.output_dir / f"worker_{worker:02d}_rows.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assigned_keys = {
            (name, condition.authority_id, condition.video_demo)
            for name, condition in assigned
        }
        observed_keys = {
            (str(row["split"]), int(row["authority_id"]), int(row["video_demo"]))
            for row in worker_rows
        }
        if (
            contract.get("schema_version") != EVALUATION_SCHEMA
            or completion.get("schema_version") != EVALUATION_SCHEMA
            or int(contract.get("worker_index", -1)) != worker
            or int(contract.get("worker_count", -1)) != args.worker_count
            or int(contract.get("condition_count", -1)) != len(assigned)
            or assigned_keys != observed_keys
            or any(key in expected_keys for key in observed_keys)
        ):
            raise ValueError("mapping evaluation worker evidence changed")
        expected_keys.update(observed_keys)
        rows.extend(worker_rows)
        worker_records.append(completion)
    if len(rows) != 451 or len(expected_keys) != 451:
        raise ValueError("mapping evaluation did not cover all sealed conditions")
    macros = {int(row["checkpoint_macro"]) for row in worker_records}
    phases = {str(row["phase"]) for row in worker_records}
    checkpoints = {str(row["checkpoint"]) for row in worker_records}
    if len(macros) != 1 or phases != {args.phase} or len(checkpoints) != 1:
        raise ValueError("mapping evaluation workers used different authorities")
    summary = summarize_mapping_rows(rows)
    previous = read_json(args.previous_report) if args.previous_report else None
    macro = next(iter(macros))
    report = {
        "schema_version": EVALUATION_SCHEMA,
        "phase": args.phase,
        "checkpoint": next(iter(checkpoints)),
        "checkpoint_macro": macro,
        "condition_count": len(rows),
        "worker_count": args.worker_count,
        "worker_records": worker_records,
        "summary": summary,
        "gate": _gate_report(
            phase=args.phase,
            macro=macro,
            config=config,
            summary=summary,
            previous=previous,
        ),
        "metric_contract": {
            "condition_member_selection": (
                "one best verified member selected by complete four-family update"
            ),
            "task_weighting": "average videos within task then equal task aggregate",
            "f3_ratio": "held-video task-median divided by fit task-median",
            "held_gradients": 0,
            "shuffled_or_reversed_use": False,
        },
    }
    write_json_atomic(args.output_dir / "report.json", report)
    return report
