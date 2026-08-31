"""Panel-B aggregation and Gate for EBSRI S2 shared LOTO."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.ecp.joint_program_primal.bank_set_shared_contract import (
    BANK_SET_SHARED_GRADIENT_META_TASKS,
    BANK_SET_SHARED_GRADIENT_TARGET_TASKS,
    BANK_SET_SHARED_HELD_META_TASKS,
    BANK_SET_SHARED_HELD_TARGET_TASKS,
    BANK_SET_SHARED_RUN_SCHEMA,
    BANK_SET_SHARED_TASKS,
    load_bank_set_shared_config,
)
from ember.ecp.joint_program_primal.bank_set_shared_evaluation import (
    BANK_SET_SHARED_JOB_RESULT_SCHEMA,
    BANK_SET_SHARED_QUEUE_SCHEMA,
    BANK_SET_SHARED_WORKER_SCHEMA,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic


BANK_SET_SHARED_AGGREGATE_SCHEMA = (
    "ember_ecp_event_bank_set_shared_loto_aggregate_v1"
)
FAMILY_NAMES = ("q", "v", "action_in", "action_out")
_EXPECTED_WALL = {
    "panel_b_backward_calls": 0,
    "held_interaction_task_backward_calls": 0,
    "same_task_held_backward_calls": 0,
    "wrong_fit1_backward_calls": 0,
    "result_or_action_gradient_calls": 0,
    "forbidden_task_reads": 0,
    "validation_or_test_reads": 0,
    "action_meta_installed": False,
    "shuffled_or_reversed_use": False,
    "single_complete_rank16": True,
    "adapter_rank": 16,
    "adapter_target_count": 38,
    "adapter_tensor_count": 76,
}


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    if not rows or not all(map(math.isfinite, rows)):
        raise ValueError("S2 aggregate received an invalid distribution")
    return {
        "count": len(rows),
        "median": statistics.median(rows),
        "minimum": min(rows),
        "maximum": max(rows),
        "values": rows,
    }


def _wall_valid(value: Mapping[str, Any]) -> bool:
    return all(value.get(key) == expected for key, expected in _EXPECTED_WALL.items())


def _raw_panel_valid(panel: Mapping[str, Any]) -> bool:
    rows = panel.get("rows", ())
    if len(rows) != 16:
        return False
    for index, row in enumerate(rows):
        if (
            int(row.get("visit", -1)) != index
            or not math.isfinite(float(row.get("carrier_loss", math.nan)))
            or not math.isfinite(float(row.get("generated_loss", math.nan)))
        ):
            return False
    return float(panel.get("free_primal_benefit", 0.0)) > 0.0


def _load_evidence(
    output_dir: Path, config_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    queue = read_json(output_dir / "queue.json")
    expected_config = {
        "path": str(config_path.resolve()),
        "bytes": config_path.stat().st_size,
    }
    if (
        queue.get("schema_version") != BANK_SET_SHARED_QUEUE_SCHEMA
        or queue.get("status") != "ready"
        or queue.get("config") != expected_config
    ):
        raise ValueError("S2 aggregate queue schema changed")
    compiler_contract = read_json(Path(queue["compiler_run"]) / "run_contract.json")
    compiler_authority = queue.get("compiler_authority", {})
    training_commit = str(compiler_contract.get("git", {}).get("commit", ""))
    if not all(
        (
            compiler_contract.get("schema_version") == BANK_SET_SHARED_RUN_SCHEMA,
            compiler_contract.get("config") == expected_config,
            compiler_authority.get("run_contract_schema")
            == BANK_SET_SHARED_RUN_SCHEMA,
            compiler_authority.get("training_commit") == training_commit,
            training_commit,
            all(
                str(row.get("training_commit", "")) == training_commit
                for row in queue.get("checkpoints", ())
            ),
        )
    ):
        raise ValueError("S2 aggregate compiler authority changed")
    jobs = {str(row["id"]): row for row in queue.get("jobs", ())}
    result_paths = sorted((output_dir / "results").glob("*.json"))
    if len(jobs) != 100 or {path.stem for path in result_paths} != set(jobs):
        raise ValueError("S2 aggregate job coverage changed")
    worker_ids: list[str] = []
    for index in range(int(queue["worker_count"])):
        worker = read_json(output_dir / "workers" / f"worker_{index:02d}.json")
        if (
            worker.get("schema_version") != BANK_SET_SHARED_WORKER_SCHEMA
            or worker.get("status") != "complete"
            or int(worker.get("worker_index", -1)) != index
        ):
            raise ValueError("S2 aggregate worker completion changed")
        worker_ids.extend(map(str, worker.get("completed_job_ids", ())))
    if len(worker_ids) != len(set(worker_ids)) or set(worker_ids) != set(jobs):
        raise ValueError("S2 dynamic queue ownership changed")

    results = []
    for path in result_paths:
        payload = read_json(path)
        job = jobs[path.stem]
        metrics = payload.get("metrics", {})
        target = metrics.get("target_authority", {})
        expected_target = (
            "each_bank_frozen_r5_base_residual"
            if str(job["arm"]).startswith("correct")
            else "task_wrong_fit0_one_round_functional_free_delta_suppressive_teacher"
        )
        family = metrics.get("family_recovery", {})
        valid = all(
            (
                payload.get("schema_version") == BANK_SET_SHARED_JOB_RESULT_SCHEMA,
                payload.get("status") == "complete",
                payload.get("job") == job,
                payload.get("checkpoint", {}).get("optimizer_step")
                == job["checkpoint_optimizer_step"],
                int(metrics.get("task", -1)) == int(job["task"]),
                metrics.get("arm") == job["arm"],
                math.isfinite(float(metrics.get("functional_recovery", math.nan))),
                _raw_panel_valid(metrics.get("panel_b", {})),
                set(family) == set(FAMILY_NAMES),
                all(math.isfinite(float(value)) for value in family.values()),
                isinstance(metrics.get("effective_rank4"), Mapping),
                target.get("effective_target") == expected_target,
                target.get("family_denominator")
                == "wrong_fit0_r5_base_to_suppressive_teacher_squared_distance",
                target.get("cached_on_cpu") is True,
                target.get("real_bank_cached") is False,
                _wall_valid(metrics.get("information_wall", {})),
                int(
                    payload.get("bank_lifecycle", {}).get(
                        "resident_real_bank_count_after_release", -1
                    )
                )
                == 0,
            )
        )
        if not valid:
            raise ValueError(f"invalid S2 job result: {path}")
        results.append(payload)
    return queue, results


def _task_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_arm = {
        str(row["job"]["arm"]): float(row["metrics"]["functional_recovery"])
        for row in rows
    }
    if set(by_arm) != {
        "correct_fit0",
        "correct_fit1",
        "correct_held",
        "wrong_fit0",
        "wrong_fit1",
    }:
        raise ValueError("S2 task arm coverage changed")
    correct_fit = statistics.median(
        (by_arm["correct_fit0"], by_arm["correct_fit1"])
    )
    wrong = statistics.median((by_arm["wrong_fit0"], by_arm["wrong_fit1"]))
    correct_all = (
        by_arm["correct_fit0"],
        by_arm["correct_fit1"],
        by_arm["correct_held"],
    )
    wrong_all = (by_arm["wrong_fit0"], by_arm["wrong_fit1"])
    family = {
        name: {
            arm: float(row["metrics"]["family_recovery"][name])
            for arm, row in ((str(value["job"]["arm"]), value) for value in rows)
        }
        for name in FAMILY_NAMES
    }
    return {
        "task": int(rows[0]["job"]["task"]),
        "role": str(rows[0]["job"]["role"]),
        "split": str(rows[0]["job"]["split"]),
        "arms": by_arm,
        "correct_fit": correct_fit,
        "same_task_held": by_arm["correct_held"],
        "wrong": wrong,
        "margin": min(correct_all) - max(wrong_all),
        "all_correct_views_better_than_all_wrong_views": min(correct_all)
        > max(wrong_all),
        "family_recovery": family,
    }


def _checkpoint_tasks(
    results: Sequence[Mapping[str, Any]], step: int
) -> dict[int, dict[str, Any]]:
    output = {}
    for task in BANK_SET_SHARED_TASKS:
        rows = [
            row
            for row in results
            if int(row["job"]["checkpoint_optimizer_step"]) == step
            and int(row["job"]["task"]) == task
        ]
        if len(rows) != 5:
            raise ValueError("S2 checkpoint task coverage changed")
        output[task] = _task_summary(rows)
    return output


def _role_summary(
    *, role: str, tasks: Mapping[int, Mapping[str, Any]], gate: Mapping[str, Any]
) -> dict[str, Any]:
    if role == "meta":
        gradient_ids = BANK_SET_SHARED_GRADIENT_META_TASKS
        held_ids = BANK_SET_SHARED_HELD_META_TASKS
    elif role == "target":
        gradient_ids = BANK_SET_SHARED_GRADIENT_TARGET_TASKS
        held_ids = BANK_SET_SHARED_HELD_TARGET_TASKS
    else:
        raise ValueError("S2 role changed")
    gradient = [tasks[task] for task in gradient_ids]
    held = [tasks[task] for task in held_ids]
    metrics = {
        "gradient_correct_fit": _distribution([row["correct_fit"] for row in gradient]),
        "gradient_same_task_held": _distribution(
            [row["same_task_held"] for row in gradient]
        ),
        "gradient_wrong": _distribution([row["wrong"] for row in gradient]),
        "gradient_margin": _distribution([row["margin"] for row in gradient]),
        "held_correct_fit": _distribution([row["correct_fit"] for row in held]),
        "held_same_task_held": _distribution([row["same_task_held"] for row in held]),
        "held_wrong": _distribution([row["wrong"] for row in held]),
        "held_margin": _distribution([row["margin"] for row in held]),
    }
    train_correct = float(metrics["gradient_correct_fit"]["median"])
    held_correct = float(metrics["held_correct_fit"]["median"])
    held_to_train = held_correct / train_correct if train_correct > 0 else None
    checks = {
        "gradient_correct_fit": train_correct
        >= float(gate["correct_fit_median_minimum"]),
        "gradient_same_task_held": metrics["gradient_same_task_held"]["median"]
        >= float(gate["same_task_held_median_minimum"]),
        "gradient_wrong": metrics["gradient_wrong"]["median"]
        <= float(gate["wrong_median_maximum"]),
        "gradient_margin": metrics["gradient_margin"]["median"]
        >= float(gate["margin_median_minimum"]),
        "held_correct_fit": held_correct >= float(gate["correct_fit_median_minimum"]),
        "held_same_task_held": metrics["held_same_task_held"]["median"]
        >= float(gate["same_task_held_median_minimum"]),
        "held_wrong": metrics["held_wrong"]["median"]
        <= float(gate["wrong_median_maximum"]),
        "held_margin": metrics["held_margin"]["median"]
        >= float(gate["margin_median_minimum"]),
        "held_each_correct_better_than_wrong": all(
            bool(row["all_correct_views_better_than_all_wrong_views"])
            for row in held
        ),
        "held_to_gradient_correct_fit": held_to_train is not None
        and held_to_train >= float(gate["held_to_gradient_correct_fit_minimum"]),
    }
    family = {
        name: {
            "gradient_correct_fit": _distribution(
                [
                    statistics.median(
                        (
                            row["family_recovery"][name]["correct_fit0"],
                            row["family_recovery"][name]["correct_fit1"],
                        )
                    )
                    for row in gradient
                ]
            ),
            "held_correct_fit": _distribution(
                [
                    statistics.median(
                        (
                            row["family_recovery"][name]["correct_fit0"],
                            row["family_recovery"][name]["correct_fit1"],
                        )
                    )
                    for row in held
                ]
            ),
        }
        for name in FAMILY_NAMES
    }
    return {
        "role": role,
        "gradient_task_ids": list(gradient_ids),
        "held_interaction_task_ids": list(held_ids),
        **metrics,
        "held_to_gradient_correct_fit": held_to_train,
        "family_diagnostic": family,
        "checks": checks,
        "pass": all(checks.values()),
    }


def aggregate_shared_evaluation(
    *, output_dir: Path, config_path: Path
) -> dict[str, Any]:
    """Seal both adjacent checkpoints; Panel B is the only scientific verdict."""

    config = load_bank_set_shared_config(config_path)
    queue, results = _load_evidence(output_dir, config_path)
    steps = list(map(int, config["evaluation"]["checkpoint_optimizer_steps"]))
    checkpoint_rows = []
    task_rows_by_step = {}
    for step in steps:
        tasks = _checkpoint_tasks(results, step)
        task_rows_by_step[step] = tasks
        roles = {
            role: _role_summary(role=role, tasks=tasks, gate=config["gate"])
            for role in ("meta", "target")
        }
        checkpoint_rows.append(
            {
                "optimizer_step": step,
                "roles": roles,
                "primary_pass": all(row["pass"] for row in roles.values()),
                "tasks": {str(task): tasks[task] for task in BANK_SET_SHARED_TASKS},
            }
        )
    earlier, later = checkpoint_rows
    stability_roles = {}
    for role in ("meta", "target"):
        previous = earlier["roles"][role]
        current = later["roles"][role]
        gradient_drop = (
            previous["gradient_correct_fit"]["median"]
            - current["gradient_correct_fit"]["median"]
        )
        held_drop = (
            previous["held_correct_fit"]["median"]
            - current["held_correct_fit"]["median"]
        )
        stable = max(gradient_drop, held_drop) <= float(
            config["gate"]["maximum_later_correct_fit_drop"]
        )
        stability_roles[role] = {
            "gradient_correct_fit_drop": gradient_drop,
            "held_interaction_correct_fit_drop": held_drop,
            "pass": stable,
        }
    stability_pass = all(row["pass"] for row in stability_roles.values())
    primary_pass = all(row["primary_pass"] for row in checkpoint_rows)
    report = {
        "schema_version": BANK_SET_SHARED_AGGREGATE_SCHEMA,
        "status": "complete",
        "stage": config["stage"],
        "functional_authority": "panel_b",
        "checkpoint_reports": checkpoint_rows,
        "adjacent_checkpoint": {
            "earlier_optimizer_step": steps[0],
            "later_optimizer_step": steps[1],
            "roles": stability_roles,
            "pass": stability_pass,
        },
        "primary_pass": primary_pass,
        "gate_pass": primary_pass and stability_pass,
        "queue": {
            "schema_version": queue["schema_version"],
            "config": dict(queue["config"]),
            "compiler_authority": dict(queue["compiler_authority"]),
            "job_count": len(queue["jobs"]),
            "worker_count": int(queue["worker_count"]),
            "policy": queue["queue_policy"],
        },
        "scientific_scope": {
            "fixed_route_shared_interaction_loto_only": True,
            "natural_program_or_g3_pass": False,
            "fresh_all_ten_task_refit_authorized": primary_pass and stability_pass,
        },
    }
    write_json_atomic(output_dir / "aggregate.json", report)
    write_json_atomic(output_dir / "completion.json", {
        "schema_version": BANK_SET_SHARED_AGGREGATE_SCHEMA,
        "primary_pass": primary_pass,
        "adjacent_checkpoint_pass": stability_pass,
        "gate_pass": report["gate_pass"],
    })
    return report
