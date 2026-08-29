"""Aggregate the fixed J3 functional evaluation and apply its Gate."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.ecp.joint_program_primal.evaluation import (
    FAMILY_NAMES,
    J2_EVALUATION_SCHEMA,
    load_joint_program_primal_gate,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic


J2_GATE_REPORT_SCHEMA = "ember_ecp_counterfactual_program_primal_gate_report_v1"


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(map(float, values))
    if not ordered or not 0.0 <= fraction <= 1.0:
        raise ValueError("J2 Gate quantile input changed")
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, float] | None:
    rows = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not rows:
        return None
    return {
        "count": len(rows),
        "mean": statistics.fmean(rows),
        "median": _quantile(rows, 0.5),
        "p10": _quantile(rows, 0.1),
        "minimum": min(rows),
        "maximum": max(rows),
    }


def _recovery(row: Mapping[str, Any]) -> float | None:
    value = row.get("functional_recovery")
    return None if value is None else float(value)


def _margin(
    controls: Mapping[str, Any], left: str, right: str
) -> float | None:
    lhs, rhs = _recovery(controls[left]), _recovery(controls[right])
    return None if lhs is None or rhs is None else lhs - rhs


def _interaction(controls: Mapping[str, Any]) -> float | None:
    values = [
        _recovery(controls[name])
        for name in (
            "primary_correct",
            "wrong_program_correct_bank",
            "correct_program_wrong_bank",
            "wrong_program_wrong_bank",
        )
    ]
    if any(value is None for value in values):
        return None
    correct, wrong_program, wrong_bank, wrong_both = map(float, values)
    return correct - wrong_program - wrong_bank + wrong_both


def _last_metric(path: Path) -> dict[str, Any]:
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    if not isinstance(last, dict):
        raise ValueError("J2 training metrics are empty")
    return last


def _load_workers(output_dir: Path, worker_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks, workers = [], []
    for worker in range(worker_count):
        root = output_dir / f"worker_{worker:02d}"
        payload = read_json(root / "result.json")
        completion = read_json(root / "completion.json")
        rows = payload.get("tasks", [])
        if (
            payload.get("schema_version") != J2_EVALUATION_SCHEMA
            or payload.get("status") != "complete"
            or int(payload.get("worker_index", -1)) != worker
            or int(payload.get("worker_count", -1)) != worker_count
            or completion.get("schema_version") != J2_EVALUATION_SCHEMA
            or int(completion.get("worker_index", -1)) != worker
            or int(completion.get("task_count", -1)) != len(rows)
        ):
            raise ValueError("J2 evaluation worker evidence changed")
        tasks.extend(rows)
        workers.append(payload)
    if len(tasks) != 12 or {int(row["task"]) for row in tasks} != {
        1, 2, 8, 9, 32, 52, 72, 73, 74, 75, 93, 94
    }:
        raise ValueError("J2 evaluation task coverage changed")
    checkpoints = {int(row["checkpoint"]["optimizer_step"]) for row in workers}
    if len(checkpoints) != 1:
        raise ValueError("J2 evaluation workers used different checkpoints")
    return sorted(tasks, key=lambda row: int(row["task"])), workers


def _family_value(row: Mapping[str, Any], family: str) -> float:
    diagnostics = row["family_diagnostic"]
    if row["split"] == "gradient":
        selected = [diagnostics[str(row["held_video"])]]
    else:
        selected = list(diagnostics.values())
    return statistics.fmean(
        float(value["family_recovery"][family]) for value in selected
    )


def _checks(
    *,
    gate: Mapping[str, Any],
    summary: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
) -> tuple[dict[str, bool], dict[str, Any]]:
    thresholds = gate["gate"]
    train = summary["gradient_train"]
    held = summary["gradient_held_video"]
    task_held = summary["true_task_held"]
    family = summary["family"]
    causal = summary["causal_controls"]
    checks = {
        "gradient_train_median": train is not None
        and train["count"] == 10
        and train["median"] >= float(thresholds["train_median_minimum"]),
        "held_video_median": held is not None
        and held["count"] == 10
        and held["median"] >= float(thresholds["held_video_median_minimum"]),
        "true_task_held_mean": task_held is not None
        and task_held["count"] == 2
        and task_held["mean"] >= float(thresholds["true_task_held_mean_minimum"]),
        "true_task_held_each": task_held is not None
        and task_held["count"] == 2
        and task_held["minimum"] >= float(thresholds["true_task_held_each_minimum"]),
        "held_to_train": summary["held_to_train"] is not None
        and summary["held_to_train"] >= float(thresholds["held_to_train_minimum"]),
        **{
            f"family_{name}": family[name] is not None
            and family[name]["count"] == 12
            and family[name]["median"] >= float(thresholds[f"{name}_minimum"])
            for name in FAMILY_NAMES
        },
        "full_over_language": causal["full_over_language"] is not None
        and causal["full_over_language"]["count"] == 12
        and causal["full_over_language"]["median"]
        >= float(thresholds["full_over_language_minimum"]),
        "full_over_endpoints": causal["full_over_endpoints"] is not None
        and causal["full_over_endpoints"]["count"] == 12
        and causal["full_over_endpoints"]["median"]
        >= float(thresholds["full_over_endpoints_minimum"]),
        "correct_wrong_program": causal["correct_wrong_program"] is not None
        and causal["correct_wrong_program"]["count"] == 12
        and causal["correct_wrong_program"]["median"]
        >= float(thresholds["correct_wrong_program_margin_minimum"]),
        "correct_wrong_bank": causal["correct_wrong_bank"] is not None
        and causal["correct_wrong_bank"]["count"] == 12
        and causal["correct_wrong_bank"]["median"]
        >= float(thresholds["correct_wrong_bank_margin_minimum"]),
        "interaction": causal["interaction"] is not None
        and causal["interaction"]["count"] == 12
        and causal["interaction"]["median"]
        >= float(thresholds["interaction_minimum"]),
        "same_task_other_retention": summary["same_task_retention"] is not None
        and summary["same_task_retention"]["count"] == 10
        and summary["same_task_retention"]["median"]
        >= float(thresholds["same_task_other_retention_minimum"]),
        "event_noncollapse": summary["events"]["median_active_events"]
        >= float(thresholds["minimum_median_active_events"])
        and summary["events"]["one_event_fraction"]
        <= float(thresholds["maximum_one_event_fraction"]),
        "K1_identity_and_information_wall": summary["information_wall_pass"],
        "evaluation_throughput": summary["evaluation_to_training_wall"]
        <= float(thresholds["evaluation_to_training_wall_maximum"]),
    }
    stability: dict[str, Any] = {"status": "pending_adjacent_checkpoint", "pass": False}
    if previous is not None:
        previous_step = int(previous["checkpoint"]["optimizer_step"])
        current_step = int(summary["checkpoint_optimizer_step"])
        drop = float(previous["summary"]["gradient_train"]["median"]) - float(
            summary["gradient_train"]["median"]
        )
        stable = (
            previous_step == 70
            and current_step == 110
            and drop <= float(thresholds["maximum_checkpoint_task_median_drop"])
            and bool(previous.get("primary_pass"))
        )
        stability = {
            "status": "evaluated",
            "previous_optimizer_step": previous_step,
            "current_optimizer_step": current_step,
            "gradient_train_median_drop": drop,
            "maximum_allowed_drop": float(
                thresholds["maximum_checkpoint_task_median_drop"]
            ),
            "previous_primary_pass": bool(previous.get("primary_pass")),
            "pass": stable,
        }
    return checks, stability


def aggregate_evaluation(
    *,
    output_dir: Path,
    gate_config: Path,
    compiler_run: Path,
    worker_count: int,
    previous_report: Path | None = None,
) -> dict[str, Any]:
    gate = load_joint_program_primal_gate(gate_config)
    tasks, workers = _load_workers(output_dir, worker_count)
    gradient = [row for row in tasks if row["split"] == "gradient"]
    task_held = [row for row in tasks if row["split"] == "true_task_held"]
    if len(gradient) != 10 or len(task_held) != 2:
        raise ValueError("J2 evaluation split counts changed")
    train_values = [row["functional_summary"]["fit_recovery"] for row in gradient]
    held_values = [
        row["functional_summary"]["held_video_recovery"] for row in gradient
    ]
    task_held_values = [
        row["functional_summary"]["true_task_held_recovery"] for row in task_held
    ]
    train = _distribution(train_values)
    held = _distribution(held_values)
    task_held_summary = _distribution(task_held_values)
    held_to_train = (
        held["median"] / train["median"]
        if held is not None and train is not None and train["median"] > 0
        else None
    )
    family = {
        name: _distribution([_family_value(row, name) for row in tasks])
        for name in FAMILY_NAMES
    }
    full_over_language = []
    full_over_endpoints = []
    wrong_program = []
    wrong_bank = []
    interaction = []
    for row in tasks:
        controls = row["controls"]
        full_over_language.append(
            _margin(controls, "primary_correct", "language_only")
        )
        full_over_endpoints.append(
            _margin(controls, "primary_correct", "endpoints")
        )
        wrong_program.append(
            _margin(controls, "primary_correct", "wrong_program_correct_bank")
        )
        wrong_bank.append(
            _margin(controls, "primary_correct", "correct_program_wrong_bank")
        )
        interaction.append(_interaction(controls))
    active = [
        int(value)
        for row in tasks
        for value in row["active_events"].values()
    ]
    wall_pass = all(
        row["information_wall"]
        == {
            "deployment_native_teacher_tensor_reads": 0,
            "panel_b_backward_calls": 0,
            "same_task_held_backward_calls": 0,
            "true_task_held_backward_calls": 0,
            "action_meta_installed": False,
            "single_complete_rank16": True,
            "K1_identity": True,
            "shuffled_or_reversed_use": False,
        }
        for row in tasks
    )
    training_metric = _last_metric(compiler_run / "metrics.jsonl")
    evaluation_wall = max(float(row["elapsed_seconds"]) for row in workers)
    training_wall = float(training_metric["elapsed_seconds"])
    checkpoint_step = int(workers[0]["checkpoint"]["optimizer_step"])
    summary = {
        "checkpoint_optimizer_step": checkpoint_step,
        "gradient_train": train,
        "gradient_held_video": held,
        "true_task_held": task_held_summary,
        "held_to_train": held_to_train,
        "family": family,
        "causal_controls": {
            "full_over_language": _distribution(full_over_language),
            "full_over_endpoints": _distribution(full_over_endpoints),
            "correct_wrong_program": _distribution(wrong_program),
            "correct_wrong_bank": _distribution(wrong_bank),
            "interaction": _distribution(interaction),
        },
        "same_task_retention": _distribution(
            [
                float(row["functional_summary"]["same_task_raw_benefit_retention"])
                for row in gradient
            ]
        ),
        "events": {
            "count": len(active),
            "median_active_events": _quantile(active, 0.5),
            "one_event_fraction": sum(value <= 1 for value in active) / len(active),
        },
        "information_wall_pass": wall_pass,
        "evaluation_wall_seconds": evaluation_wall,
        "training_wall_seconds": training_wall,
        "evaluation_to_training_wall": evaluation_wall / max(training_wall, 1e-12),
        "tasks": tasks,
    }
    previous = read_json(previous_report) if previous_report is not None else None
    checks, stability = _checks(gate=gate, summary=summary, previous=previous)
    primary_pass = all(checks.values())
    report = {
        "schema_version": J2_GATE_REPORT_SCHEMA,
        "status": "complete",
        "checkpoint": dict(workers[0]["checkpoint"]),
        "summary": summary,
        "checks": checks,
        "primary_pass": primary_pass,
        "adjacent_checkpoint": stability,
        "gate_pass": primary_pass and bool(stability["pass"]),
        "worker_count": worker_count,
        "worker_commits": sorted({row["git"]["commit"] for row in workers}),
        "gate_config": {"path": str(gate_config), "bytes": gate_config.stat().st_size},
    }
    write_json_atomic(output_dir / "aggregate.json", report)
    write_json_atomic(
        output_dir / "completion.json",
        {
            "schema_version": J2_GATE_REPORT_SCHEMA,
            "checkpoint_optimizer_step": checkpoint_step,
            "primary_pass": primary_pass,
            "gate_pass": report["gate_pass"],
        },
    )
    return report
