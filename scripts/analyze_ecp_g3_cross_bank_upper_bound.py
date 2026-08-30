#!/usr/bin/env python3
"""Test whether a successful fixed-route primal needs its own task's bank."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.ecp.joint_program_primal.evaluation import (
    _complete_state,
    _normalized,
    _panel_value,
    _positive_control_losses,
    _task_conditions,
    _wrong_task,
)
from ember.ecp.joint_program_primal.routing_control import (
    ROUTING_TASK_IDS,
    fixed_routing_program,
    prepare_routing_control_runtime,
)
from ember.ecp.joint_program_primal.routing_control_evaluation import (
    _checkpoint_authority,
    load_routing_control_gate,
    routing_task_assignments,
)
from ember.ecp.joint_program_primal.train_step import prepare_joint_condition
from ember.ecp.native_materialization import residual_lora_state
from ember.ecp.shared_compiler_span import _low_rank_geometry
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import initialize_distributed


SCHEMA = "ember_ecp_g3_cross_bank_upper_bound_v1"


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    rows = tuple(map(float, values))
    if not rows:
        raise ValueError("cross-bank distribution is empty")
    ordered = sorted(rows)
    p10_index = max(0, int(0.1 * len(ordered)) - 1)
    return {
        "count": len(rows),
        "minimum": min(rows),
        "p10": ordered[p10_index],
        "median": statistics.median(rows),
        "mean": statistics.fmean(rows),
        "maximum": max(rows),
    }


def _r5_rows(path: Path) -> dict[int, Mapping[str, Any]]:
    report = read_json(path.resolve())
    rows = {
        int(row["task"]): row
        for row in report.get("summary", {}).get("tasks", ())
    }
    if (
        report.get("gate_pass") is not True
        or int(report.get("checkpoint", {}).get("optimizer_step", -1)) != 110
        or set(rows) != set(ROUTING_TASK_IDS)
    ):
        raise ValueError("R5 passed aggregate authority changed")
    return rows


def _evaluate_task(
    runtime: Any,
    *,
    task_id: int,
    positive_root: Path,
    r5_row: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    first = _task_conditions(runtime, task_id)[0]
    wrong_task = _wrong_task(runtime, task_id)
    wrong_first = _task_conditions(runtime, wrong_task)[0]
    free_reference, free_authority = _positive_control_losses(
        positive_root, task_id
    )
    correct_record = r5_row["controls"]["primary_correct"]
    if int(r5_row["fit_videos"][0]) != int(first.video_demo):
        raise ValueError("R5 primary view authority changed")

    correct_prepared, _ = prepare_joint_condition(runtime, first)
    wrong_prepared, _ = prepare_joint_condition(runtime, wrong_first)
    teacher_reads = runtime.native_teachers.tensor_reads
    with torch.inference_mode():
        program = fixed_routing_program(runtime, task_id)
        _, correct_output = _complete_state(
            runtime, program=program, bank=correct_prepared
        )
        wrong_state, wrong_output = _complete_state(
            runtime, program=program, bank=wrong_prepared
        )
    wrong_record = _normalized(
        _panel_value(runtime, task_id=task_id, state=wrong_state),
        free_reference[first.video_demo],
    )
    if runtime.native_teachers.tensor_reads != teacher_reads:
        raise RuntimeError("cross-bank diagnostic read native teachers")

    correct_residual = residual_lora_state(
        correct_output.residual, runtime.rank4_contract, canonicalize=False
    )
    wrong_residual = residual_lora_state(
        wrong_output.residual, runtime.rank4_contract, canonicalize=False
    )
    geometry = _low_rank_geometry(wrong_residual, correct_residual, runtime)
    correct_recovery = float(correct_record["functional_recovery"])
    wrong_recovery = float(wrong_record["functional_recovery"])
    correct_benefit = float(correct_record["benefit_over_carrier"])
    wrong_benefit = float(wrong_record["benefit_over_carrier"])
    return {
        "task": task_id,
        "role": runtime.panels[task_id].role,
        "correct_video_demo": int(first.video_demo),
        "wrong_task": wrong_task,
        "wrong_video_demo": int(wrong_first.video_demo),
        "correct": dict(correct_record),
        "wrong_bank": wrong_record,
        "correct_minus_wrong_recovery": correct_recovery - wrong_recovery,
        "wrong_to_correct_benefit_retention": wrong_benefit
        / max(abs(correct_benefit), 1e-12),
        "wrong_to_correct_update_geometry": geometry,
        "free_primal_authority": free_authority,
        "native_teacher_tensor_reads": 0,
        "panel_b_backward_calls": 0,
        "action_meta_installed": False,
        "single_complete_rank16": True,
        "elapsed_seconds": time.monotonic() - started,
    }


def worker(args: argparse.Namespace) -> None:
    state = git_state(Path(__file__).resolve().parents[1])
    if (
        not git_state_is_clean_pushed_or_frozen_authority(state)
        or state.get("branch") != ""
        or state.get("upstream") is not None
    ):
        raise ValueError("cross-bank evidence requires clean detached authority")
    gate = load_routing_control_gate(args.gate_config)
    positive_root = (
        args.asset_root / gate["authorities"]["positive_control_root"]
    ).resolve()
    r5_rows = _r5_rows(args.r5_aggregate)
    if args.worker_index < 0 or args.worker_index >= args.worker_count:
        raise ValueError("cross-bank worker index changed")
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size != 1:
        raise ValueError("cross-bank workers must be independent single-GPU jobs")
    runtime_args = argparse.Namespace(
        config=args.config,
        base_config=args.base_config,
        mode="profile",
        phase="joint",
        task=None,
        asset_root=args.asset_root,
        source_run=args.source_run,
        checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer_path,
        data_root=args.data_root,
        output_dir=args.output_dir / f"worker_{args.worker_index:02d}_runtime",
        condition_cache_root=args.condition_cache_root,
        resume=None,
        stop_after_step=1,
        log_every=1,
        skip_routing_initialization=True,
    )
    runtime = None
    try:
        runtime = prepare_routing_control_runtime(runtime_args, context)
        checkpoint = _checkpoint_authority(
            runtime,
            compiler_run=args.compiler_run,
            compiler_checkpoint=args.compiler_checkpoint,
            gate=gate,
        )
        assignments = routing_task_assignments(
            args.worker_count, gate["evaluation"]["task_cost_seconds"]
        )
        rows = []
        for task_id in assignments[args.worker_index]:
            rows.append(
                _evaluate_task(
                    runtime,
                    task_id=task_id,
                    positive_root=positive_root,
                    r5_row=r5_rows[task_id],
                )
            )
            runtime.panel_batch_cache.clear()
            torch.cuda.empty_cache()
        worker_dir = args.output_dir / f"worker_{args.worker_index:02d}"
        if worker_dir.exists():
            raise ValueError("cross-bank worker output already exists")
        worker_dir.mkdir(parents=True)
        write_json_atomic(
            worker_dir / "result.json",
            {
                "schema_version": SCHEMA,
                "status": "complete",
                "worker_index": args.worker_index,
                "worker_count": args.worker_count,
                "assignments": [list(row) for row in assignments],
                "checkpoint": checkpoint,
                "tasks": rows,
                "git": {"commit": state["commit"], "branch": state["branch"]},
                "physical_visible_device": __import__("os").environ.get(
                    "CUDA_VISIBLE_DEVICES"
                ),
            },
        )
        write_json_atomic(
            worker_dir / "completion.json",
            {
                "schema_version": SCHEMA,
                "worker_index": args.worker_index,
                "task_count": len(rows),
            },
        )
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def aggregate(args: argparse.Namespace) -> None:
    rows = []
    commits = set()
    for worker_index in range(args.worker_count):
        root = args.output_dir / f"worker_{worker_index:02d}"
        result = read_json(root / "result.json")
        completion = read_json(root / "completion.json")
        if (
            result.get("schema_version") != SCHEMA
            or result.get("status") != "complete"
            or result.get("worker_index") != worker_index
            or completion.get("schema_version") != SCHEMA
            or completion.get("worker_index") != worker_index
        ):
            raise ValueError("cross-bank worker evidence changed")
        rows.extend(result["tasks"])
        commits.add(result["git"]["commit"])
    if len(rows) != len(ROUTING_TASK_IDS) or {row["task"] for row in rows} != set(
        ROUTING_TASK_IDS
    ) or len(commits) != 1:
        raise ValueError("cross-bank aggregate task or commit authority changed")
    rows.sort(key=lambda row: row["task"])
    margins = [float(row["correct_minus_wrong_recovery"]) for row in rows]
    retentions = [float(row["wrong_to_correct_benefit_retention"]) for row in rows]
    cosines = [
        float(row["wrong_to_correct_update_geometry"]["overall"]["update_cosine"])
        for row in rows
    ]
    summary = {
        "correct_recovery": _distribution(
            [float(row["correct"]["functional_recovery"]) for row in rows]
        ),
        "wrong_bank_recovery": _distribution(
            [float(row["wrong_bank"]["functional_recovery"]) for row in rows]
        ),
        "correct_minus_wrong_recovery": _distribution(margins),
        "wrong_to_correct_benefit_retention": _distribution(retentions),
        "wrong_to_correct_update_cosine": _distribution(cosines),
        "correct_bank_better_count": sum(value > 0 for value in margins),
        "bank_margin_at_least_0_10_count": sum(value >= 0.10 for value in margins),
        "wrong_bank_positive_count": sum(
            float(row["wrong_bank"]["functional_recovery"]) > 0 for row in rows
        ),
    }
    operator_bank_interaction_identifiable = (
        float(summary["correct_minus_wrong_recovery"]["median"]) >= 0.10
        and int(summary["correct_bank_better_count"]) >= 8
    )
    write_json_atomic(
        args.output_dir / "aggregate.json",
        {
            "schema_version": SCHEMA,
            "status": "complete",
            "worker_count": args.worker_count,
            "git_commit": next(iter(commits)),
            "operator_bank_interaction_identifiable": (
                operator_bank_interaction_identifiable
            ),
            "decision_rule": {
                "median_correct_minus_wrong_recovery": 0.10,
                "minimum_correct_bank_better_tasks": 8,
            },
            "summary": summary,
            "tasks": rows,
        },
    )
    print(
        {
            "operator_bank_interaction_identifiable": (
                operator_bank_interaction_identifiable
            ),
            "summary": summary,
        }
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    work = commands.add_parser("worker")
    for name in (
        "config",
        "gate_config",
        "base_config",
        "asset_root",
        "source_run",
        "checkpoint",
        "tokenizer_path",
        "data_root",
        "compiler_run",
        "compiler_checkpoint",
        "condition_cache_root",
        "r5_aggregate",
        "output_dir",
    ):
        work.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    work.add_argument("--worker-index", type=int, required=True)
    work.add_argument("--worker-count", type=int, required=True)
    collect = commands.add_parser("aggregate")
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--worker-count", type=int, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "worker":
        worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
