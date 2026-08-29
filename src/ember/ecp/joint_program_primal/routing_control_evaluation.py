"""Efficient paired Gate for the training-only routing-token boundary control."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA, checkpoint_macro
from ember.ecp.joint_program_primal.evaluation import (
    FAMILY_NAMES,
    _complete_state,
    _family_record,
    _mean_recovery,
    _normalized,
    _panel_value,
    _positive_control_losses,
    _task_conditions,
)
from ember.ecp.joint_program_primal.evaluation_gate import (
    _distribution,
    _last_metric,
)
from ember.ecp.joint_program_primal.routing_control import (
    ROUTING_CONTROL_RUN_SCHEMA,
    ROUTING_CONTROL_STAGE,
    ROUTING_TASK_IDS,
    fixed_routing_program,
    load_routing_control_config,
    prepare_routing_control_runtime,
)
from ember.ecp.joint_program_primal.train_step import prepare_joint_condition
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import initialize_distributed


ROUTING_GATE_SCHEMA = "ember_ecp_routing_token_control_gate_v1"
ROUTING_EVALUATION_SCHEMA = "ember_ecp_routing_token_control_evaluation_task_v1"
ROUTING_GATE_REPORT_SCHEMA = "ember_ecp_routing_token_control_gate_report_v1"


def load_routing_control_gate(path: Path) -> dict[str, Any]:
    config = read_json(path.resolve())
    evaluation = config.get("evaluation", {})
    wall = config.get("information_wall", {})
    task_cost_seconds = evaluation.get("task_cost_seconds", {})
    if (
        config.get("schema_version") != ROUTING_GATE_SCHEMA
        or config.get("status") != "training_only_routing_boundary_control"
        or config.get("checkpoint_optimizer_steps") != [70, 110]
        or evaluation.get("evaluated_task_ids") != list(ROUTING_TASK_IDS)
        or evaluation.get("functional_panel") != "panel_b"
        or evaluation.get("panel_visits") != 16
        or evaluation.get("wrong_token_pairing")
        != "next_gradient_task_within_same_role_cyclic"
        or set(map(int, task_cost_seconds)) != set(ROUTING_TASK_IDS)
        or min(map(float, task_cost_seconds.values()), default=0.0) <= 0.0
        or wall.get("fixed_routing_token_training_only") is not True
        or wall.get("deployment_candidate") is not False
        or wall.get("action_meta_installed") is not False
        or wall.get("single_complete_rank16") is not True
        or wall.get("shuffled_or_reversed_use") is not False
    ):
        raise ValueError("unsupported routing-token control Gate config")
    return config


def _training_world_size(
    runtime: Any,
    run_contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> int:
    world_size = int(manifest.get("world_size", -1))
    topology = run_contract.get("world_topology", ())
    allowed = set(map(int, runtime.config["profile"]["allowed_world_sizes"]))
    if (
        world_size not in allowed
        or not isinstance(topology, list)
        or len(topology) != world_size
        or sorted(int(row.get("rank", -1)) for row in topology)
        != list(range(world_size))
    ):
        raise ValueError("routing-control training world authority changed")
    return world_size


def _checkpoint_authority(
    runtime: Any,
    *,
    compiler_run: Path,
    compiler_checkpoint: Path,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    compiler_run = compiler_run.resolve()
    compiler_checkpoint = compiler_checkpoint.resolve()
    step = checkpoint_macro(compiler_checkpoint)
    run_contract = read_json(compiler_run / "run_contract.json")
    manifest = read_json(compiler_checkpoint / "checkpoint_manifest.json")
    training_world_size = _training_world_size(runtime, run_contract, manifest)
    files = manifest.get("files", {})
    expected = {
        "ecp.safetensors",
        "trainer_state.pt",
        *(
            f"rank_{rank:02d}_state.pt"
            for rank in range(training_world_size)
        ),
    }
    if (
        compiler_checkpoint.parent.parent != compiler_run
        or step not in set(map(int, gate["checkpoint_optimizer_steps"]))
        or run_contract.get("schema_version") != ROUTING_CONTROL_RUN_SCHEMA
        or run_contract.get("stage") != ROUTING_CONTROL_STAGE
        or run_contract.get("phase") != "joint"
        or run_contract.get("mode") != "formal"
        or run_contract.get("diagnostic", {}).get("deployment_candidate") is not False
        or manifest.get("schema_version") != ECP_CHECKPOINT_SCHEMA
        or manifest.get("stage") != ROUTING_CONTROL_STAGE
        or int(manifest.get("next_macro", -1)) != step
        or manifest.get("run_contract_schema") != ROUTING_CONTROL_RUN_SCHEMA
        or set(files) != expected
    ):
        raise ValueError("routing-control checkpoint authority changed")
    for name, record in files.items():
        candidate = compiler_checkpoint / name
        if not candidate.is_file() or candidate.stat().st_size != int(record["bytes"]):
            raise ValueError(f"routing-control checkpoint file changed: {name}")
    runtime.writer_state.load_state_dict(
        load_file(
            str(compiler_checkpoint / "ecp.safetensors"),
            device=str(runtime.context.device),
        ),
        strict=True,
    )
    runtime.writer_state.requires_grad_(False).eval()
    runtime.program.eval()
    runtime.compiler.eval()
    if any(parameter.requires_grad for parameter in runtime.policy.parameters()):
        raise RuntimeError("routing-control evaluator unfroze source policy")
    return {
        "optimizer_step": step,
        "path": str(compiler_checkpoint),
        "training_commit": str(run_contract["git"]["commit"]),
        "world_size": training_world_size,
        "tensor_bytes": int(files["ecp.safetensors"]["bytes"]),
    }


def _wrong_task(runtime: Any, task_id: int) -> int:
    split = runtime.config["task_split"]
    candidates = tuple(
        map(
            int,
            split[
                "gradient_meta"
                if task_id in set(map(int, split["gradient_meta"]))
                else "gradient_target"
            ],
        )
    )
    if len(candidates) != 5 or task_id not in candidates:
        raise ValueError("routing-control wrong-token role panel changed")
    return candidates[(candidates.index(task_id) + 1) % len(candidates)]


def routing_task_assignments(
    worker_count: int,
    task_cost_seconds: Mapping[str, Any],
) -> tuple[tuple[int, ...], ...]:
    if not 1 <= worker_count <= 6:
        raise ValueError("routing-control evaluator worker count changed")
    costs = {int(task): float(value) for task, value in task_cost_seconds.items()}
    if set(costs) != set(ROUTING_TASK_IDS) or min(costs.values()) <= 0.0:
        raise ValueError("routing-control evaluator cost authority changed")
    rows: list[list[int]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    maximum_tasks = (len(ROUTING_TASK_IDS) + worker_count - 1) // worker_count
    for task in sorted(ROUTING_TASK_IDS, key=lambda value: (-costs[value], value)):
        eligible = [
            worker
            for worker in range(worker_count)
            if len(rows[worker]) < maximum_tasks
        ]
        worker = min(eligible, key=lambda value: (loads[value], value))
        rows[worker].append(task)
        loads[worker] += costs[task]
    return tuple(tuple(sorted(row)) for row in rows)


def _evaluate_task(
    runtime: Any,
    *,
    task_id: int,
    gate: Mapping[str, Any],
    positive_control_root: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    first, second, held = _task_conditions(runtime, task_id)
    correct_conditions = (first, second, held)
    free_reference, free_authority = _positive_control_losses(
        positive_control_root, task_id
    )
    correct_rows: dict[int, dict[str, Any]] = {}
    outputs = {}
    for condition in correct_conditions:
        prepared, _ = prepare_joint_condition(runtime, condition)
        with torch.inference_mode():
            state, output = _complete_state(
                runtime,
                program=fixed_routing_program(runtime, task_id),
                bank=prepared,
            )
        correct_rows[condition.video_demo] = _normalized(
            _panel_value(runtime, task_id=task_id, state=state),
            free_reference[condition.video_demo],
        )
        outputs[condition.video_demo] = output
        del state, prepared

    wrong_task = _wrong_task(runtime, task_id)
    primary_prepared, _ = prepare_joint_condition(runtime, first)
    with torch.inference_mode():
        wrong_state, _ = _complete_state(
            runtime,
            program=fixed_routing_program(runtime, wrong_task),
            bank=primary_prepared,
        )
    wrong_program = _normalized(
        _panel_value(runtime, task_id=task_id, state=wrong_state),
        free_reference[first.video_demo],
    )
    del wrong_state, primary_prepared

    indices = tuple(map(int, gate["evaluation"]["selected_family_report_targets"]))
    family = {
        str(condition.video_demo): _family_record(
            runtime,
            task_id=task_id,
            condition=condition,
            output=outputs[condition.video_demo],
            indices=indices,
        )
        for condition in correct_conditions
    }
    fit_recovery = _mean_recovery(
        [correct_rows[row.video_demo] for row in (first, second)]
    )
    held_recovery = correct_rows[held.video_demo]["functional_recovery"]
    return {
        "schema_version": ROUTING_EVALUATION_SCHEMA,
        "task": task_id,
        "role": runtime.panels[task_id].role,
        "fit_videos": [first.video_demo, second.video_demo],
        "held_video": held.video_demo,
        "correct": {str(key): value for key, value in correct_rows.items()},
        "functional_summary": {
            "fit_recovery": fit_recovery,
            "held_video_recovery": (
                None if held_recovery is None else float(held_recovery)
            ),
            "same_task_raw_benefit_retention": (
                float(correct_rows[held.video_demo]["benefit_over_carrier"])
                / max(
                    statistics.fmean(
                        float(correct_rows[row.video_demo]["benefit_over_carrier"])
                        for row in (first, second)
                    ),
                    1e-12,
                )
            ),
        },
        "controls": {
            "primary_correct": correct_rows[first.video_demo],
            "wrong_token_correct_bank": wrong_program,
            "wrong_task": wrong_task,
        },
        "family_diagnostic": family,
        "free_primal_authority": free_authority,
        "information_wall": {
            "deployment_native_teacher_tensor_reads": 0,
            "panel_b_backward_calls": 0,
            "same_task_held_backward_calls": 0,
            "action_meta_installed": False,
            "single_complete_rank16": True,
            "K1_identity": True,
            "shuffled_or_reversed_use": False,
            "fixed_routing_token_training_only": True,
            "deployment_candidate": False,
        },
        "task_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
            runtime.context.device
        ),
        "max_cuda_reserved_bytes": torch.cuda.max_memory_reserved(
            runtime.context.device
        ),
    }


def evaluate_routing_worker(args: argparse.Namespace) -> None:
    state = git_state(Path(__file__).resolve().parents[4])
    if (
        not git_state_is_clean_pushed_or_frozen_authority(state)
        or state.get("branch") != ""
        or state.get("upstream") is not None
    ):
        raise ValueError("formal routing-control evaluation requires detached authority")
    gate = load_routing_control_gate(args.gate_config)
    if args.config != (args.asset_root / gate["training_config"]).resolve():
        raise ValueError("routing-control evaluator training authority changed")
    load_routing_control_config(args.config)
    positive_root = (
        args.asset_root / gate["authorities"]["positive_control_root"]
    ).resolve()
    if args.worker_index < 0 or args.worker_index >= args.worker_count:
        raise ValueError("routing-control evaluator worker index changed")
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size != 1:
        raise ValueError("routing-control workers must be independent single GPUs")
    if (
        len(args.compiler_checkpoints) != 2
        or len(args.output_dirs) != 2
        or [checkpoint_macro(path) for path in args.compiler_checkpoints]
        != list(map(int, gate["checkpoint_optimizer_steps"]))
    ):
        raise ValueError("routing-control paired checkpoint panel changed")
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
        output_dir=args.output_dirs[0]
        / f"worker_{args.worker_index:02d}_runtime",
        condition_cache_root=args.condition_cache_root,
        resume=None,
        stop_after_step=1,
        log_every=1,
        skip_routing_initialization=True,
    )
    runtime = None
    started = time.monotonic()
    try:
        runtime = prepare_routing_control_runtime(runtime_args, context)
        assignments = routing_task_assignments(
            args.worker_count,
            gate["evaluation"]["task_cost_seconds"],
        )
        setup_seconds = time.monotonic() - started
        for compiler_checkpoint, output_dir in zip(
            args.compiler_checkpoints, args.output_dirs, strict=True
        ):
            checkpoint_started = time.monotonic()
            checkpoint = _checkpoint_authority(
                runtime,
                compiler_run=args.compiler_run,
                compiler_checkpoint=compiler_checkpoint,
                gate=gate,
            )
            torch.cuda.reset_peak_memory_stats(context.device)
            rows = []
            for task_id in assignments[args.worker_index]:
                rows.append(
                    _evaluate_task(
                        runtime,
                        task_id=task_id,
                        gate=gate,
                        positive_control_root=positive_root,
                    )
                )
                runtime.panel_batch_cache.clear()
            evaluation_seconds = time.monotonic() - checkpoint_started
            worker_dir = output_dir / f"worker_{args.worker_index:02d}"
            if worker_dir.exists():
                raise ValueError("routing-control worker output already exists")
            worker_dir.mkdir(parents=True)
            payload = {
                "schema_version": ROUTING_EVALUATION_SCHEMA,
                "status": "complete",
                "worker_index": args.worker_index,
                "worker_count": args.worker_count,
                "assignments": [list(row) for row in assignments],
                "checkpoint": checkpoint,
                "tasks": rows,
                "shared_runtime_setup_seconds": setup_seconds,
                "checkpoint_evaluation_seconds": evaluation_seconds,
                "elapsed_seconds": setup_seconds / 2.0 + evaluation_seconds,
                "physical_visible_device": __import__("os").environ.get(
                    "CUDA_VISIBLE_DEVICES"
                ),
                "git": {"commit": state["commit"], "branch": state["branch"]},
            }
            write_json_atomic(worker_dir / "result.json", payload)
            write_json_atomic(
                worker_dir / "completion.json",
                {
                    "schema_version": ROUTING_EVALUATION_SCHEMA,
                    "worker_index": args.worker_index,
                    "task_count": len(rows),
                },
            )
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _load_workers(
    output_dir: Path, worker_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks, workers = [], []
    for worker in range(worker_count):
        root = output_dir / f"worker_{worker:02d}"
        payload = read_json(root / "result.json")
        completion = read_json(root / "completion.json")
        rows = payload.get("tasks", [])
        if (
            payload.get("schema_version") != ROUTING_EVALUATION_SCHEMA
            or payload.get("status") != "complete"
            or int(payload.get("worker_index", -1)) != worker
            or int(payload.get("worker_count", -1)) != worker_count
            or completion.get("schema_version") != ROUTING_EVALUATION_SCHEMA
            or int(completion.get("worker_index", -1)) != worker
            or int(completion.get("task_count", -1)) != len(rows)
        ):
            raise ValueError("routing-control worker evidence changed")
        tasks.extend(rows)
        workers.append(payload)
    if len(tasks) != 10 or {int(row["task"]) for row in tasks} != set(
        ROUTING_TASK_IDS
    ):
        raise ValueError("routing-control task coverage changed")
    if len({int(row["checkpoint"]["optimizer_step"]) for row in workers}) != 1:
        raise ValueError("routing-control workers used different checkpoints")
    return sorted(tasks, key=lambda row: int(row["task"])), workers


def _family_value(row: Mapping[str, Any], family: str) -> float:
    return float(
        row["family_diagnostic"][str(row["held_video"])]["family_recovery"][
            family
        ]
    )


def _checks(
    gate: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[str, bool]:
    thresholds = gate["gate"]
    train = summary["gradient_train"]
    held = summary["gradient_held_video"]
    family = summary["family"]
    return {
        "gradient_train_median": train is not None
        and train["count"] == 10
        and train["median"] >= float(thresholds["train_median_minimum"]),
        "held_video_median": held is not None
        and held["count"] == 10
        and held["median"] >= float(thresholds["held_video_median_minimum"]),
        "held_to_train": summary["held_to_train"] is not None
        and summary["held_to_train"] >= float(thresholds["held_to_train_minimum"]),
        **{
            f"family_{name}": family[name] is not None
            and family[name]["count"] == 10
            and family[name]["median"] >= float(thresholds[f"{name}_minimum"])
            for name in FAMILY_NAMES
        },
        "wrong_token_margin": summary["wrong_token_margin"] is not None
        and summary["wrong_token_margin"]["count"] == 10
        and summary["wrong_token_margin"]["median"]
        >= float(thresholds["wrong_token_margin_minimum"]),
        "same_task_other_retention": summary["same_task_retention"] is not None
        and summary["same_task_retention"]["count"] == 10
        and summary["same_task_retention"]["median"]
        >= float(thresholds["same_task_other_retention_minimum"]),
        "information_wall": bool(summary["information_wall_pass"]),
        "evaluation_throughput": summary["evaluation_to_training_wall"]
        <= float(thresholds["evaluation_to_training_wall_maximum"]),
    }


def _evaluation_summary(
    *,
    tasks: Sequence[Mapping[str, Any]],
    workers: Sequence[Mapping[str, Any]],
    compiler_run: Path,
) -> dict[str, Any]:
    train = _distribution(
        [row["functional_summary"]["fit_recovery"] for row in tasks]
    )
    held = _distribution(
        [row["functional_summary"]["held_video_recovery"] for row in tasks]
    )
    held_to_train = (
        held["median"] / train["median"]
        if held is not None and train is not None and train["median"] > 0
        else None
    )
    family = {
        name: _distribution([_family_value(row, name) for row in tasks])
        for name in FAMILY_NAMES
    }
    wrong_token = _distribution(
        [
            float(row["controls"]["primary_correct"]["functional_recovery"])
            - float(
                row["controls"]["wrong_token_correct_bank"][
                    "functional_recovery"
                ]
            )
            for row in tasks
        ]
    )
    expected_wall = {
        "deployment_native_teacher_tensor_reads": 0,
        "panel_b_backward_calls": 0,
        "same_task_held_backward_calls": 0,
        "action_meta_installed": False,
        "single_complete_rank16": True,
        "K1_identity": True,
        "shuffled_or_reversed_use": False,
        "fixed_routing_token_training_only": True,
        "deployment_candidate": False,
    }
    training_metric = _last_metric(compiler_run / "metrics.jsonl")
    evaluation_wall = max(float(row["elapsed_seconds"]) for row in workers)
    training_wall = float(training_metric["elapsed_seconds"])
    return {
        "checkpoint_optimizer_step": int(workers[0]["checkpoint"]["optimizer_step"]),
        "gradient_train": train,
        "gradient_held_video": held,
        "held_to_train": held_to_train,
        "family": family,
        "wrong_token_margin": wrong_token,
        "same_task_retention": _distribution(
            [
                float(row["functional_summary"]["same_task_raw_benefit_retention"])
                for row in tasks
            ]
        ),
        "information_wall_pass": all(
            row["information_wall"] == expected_wall for row in tasks
        ),
        "evaluation_wall_seconds": evaluation_wall,
        "training_wall_seconds": training_wall,
        "evaluation_to_training_wall": evaluation_wall / max(training_wall, 1e-12),
        "tasks": tasks,
    }


def aggregate_routing_evaluation(
    *,
    output_dir: Path,
    gate_config: Path,
    compiler_run: Path,
    worker_count: int,
    previous_report: Path | None = None,
) -> dict[str, Any]:
    gate = load_routing_control_gate(gate_config)
    tasks, workers = _load_workers(output_dir, worker_count)
    summary = _evaluation_summary(
        tasks=tasks, workers=workers, compiler_run=compiler_run
    )
    checkpoint_step = int(summary["checkpoint_optimizer_step"])
    train = summary["gradient_train"]
    checks = _checks(gate, summary)
    primary_pass = all(checks.values())
    stability: dict[str, Any] = {
        "status": "pending_adjacent_checkpoint",
        "pass": False,
    }
    if previous_report is not None:
        previous = read_json(previous_report)
        drop = float(previous["summary"]["gradient_train"]["median"]) - float(
            train["median"]
        )
        stable = (
            int(previous["checkpoint"]["optimizer_step"]) == 70
            and checkpoint_step == 110
            and bool(previous.get("primary_pass"))
            and drop
            <= float(gate["gate"]["maximum_checkpoint_task_median_drop"])
        )
        stability = {
            "status": "evaluated",
            "previous_optimizer_step": 70,
            "current_optimizer_step": checkpoint_step,
            "gradient_train_median_drop": drop,
            "previous_primary_pass": bool(previous.get("primary_pass")),
            "pass": stable,
        }
    report = {
        "schema_version": ROUTING_GATE_REPORT_SCHEMA,
        "status": "complete",
        "checkpoint": dict(workers[0]["checkpoint"]),
        "summary": summary,
        "checks": checks,
        "primary_pass": primary_pass,
        "adjacent_checkpoint": stability,
        "gate_pass": primary_pass and bool(stability["pass"]),
        "scientific_scope": {
            "deployment_candidate": False,
            "can_pass_g3": False,
            "question": "scorer usability given perfectly separated training-only routing content",
        },
        "worker_count": worker_count,
        "worker_commits": sorted({row["git"]["commit"] for row in workers}),
        "gate_config": {
            "path": str(gate_config),
            "bytes": gate_config.stat().st_size,
        },
    }
    write_json_atomic(output_dir / "aggregate.json", report)
    write_json_atomic(
        output_dir / "completion.json",
        {
            "schema_version": ROUTING_GATE_REPORT_SCHEMA,
            "checkpoint_optimizer_step": checkpoint_step,
            "primary_pass": primary_pass,
            "gate_pass": report["gate_pass"],
        },
    )
    return report
