"""Per-bank dynamic evaluation queue for EBSRI S2 shared LOTO."""

from __future__ import annotations

import importlib
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.ecp.bank_conditioning.mapping import load_mapping_split
from ember.ecp.joint_program_primal.bank_set_shared_contract import (
    BANK_SET_SHARED_ARMS,
    BANK_SET_SHARED_TASKS,
    checkpoint_authority,
    load_bank_set_shared_config,
)
from ember.ecp.shared_compiler_assets import load_shared_compiler_config
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import initialize_distributed


BANK_SET_SHARED_QUEUE_SCHEMA = "ember_ecp_event_bank_set_shared_eval_queue_v1"
BANK_SET_SHARED_JOB_RESULT_SCHEMA = (
    "ember_ecp_event_bank_set_shared_eval_job_result_v1"
)
BANK_SET_SHARED_WORKER_SCHEMA = "ember_ecp_event_bank_set_shared_eval_worker_v1"
_BACKEND_MODULE = "ember.ecp.joint_program_primal.bank_set_shared_training"


def _task_axes(config: Mapping[str, Any], task: int) -> tuple[str, str]:
    shared = config["shared_training"]
    role = "meta" if task in {1, 8, 9, 32, 52} else "target"
    split = (
        "gradient"
        if task in set(map(int, shared["gradient_task_ids"]))
        else "interaction_held"
    )
    return role, split


def build_job_queue(
    *,
    config_path: Path,
    base_config_path: Path,
    asset_root: Path,
    checkpoints: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build all 100 task-arm-checkpoint jobs from the sealed video split."""

    config = load_bank_set_shared_config(config_path)
    base = load_shared_compiler_config(base_config_path)
    split = load_mapping_split(base, asset_root=asset_root.resolve())
    fit_by_task = split.fit_by_task
    held_by_task = split.video_held_by_task
    wrong_map = {
        int(task): int(wrong)
        for task, wrong in config["shared_training"]["wrong_task_by_task"].items()
    }
    wrong_map.update(
        {
            int(task): int(wrong)
            for task, wrong in config["shared_training"][
                "evaluation_wrong_task_by_task"
            ].items()
        }
    )
    jobs = []
    for checkpoint in checkpoints:
        step = int(checkpoint["optimizer_step"])
        for task in BANK_SET_SHARED_TASKS:
            correct = fit_by_task[task]
            held = held_by_task[task]
            wrong_task = wrong_map[task]
            wrong = fit_by_task[wrong_task]
            if len(correct) < 2 or len(held) != 1 or len(wrong) < 2:
                raise ValueError("S2 task video panel changed")
            profile = config["shared_training"]["task_profiles"][str(task)]
            observed_frames = {
                "correct_fit0": int(correct[0].sampled_frames),
                "correct_fit1": int(correct[1].sampled_frames),
                "correct_held": int(held[0].sampled_frames),
            }
            if observed_frames != profile["correct_arm_sampled_frames"]:
                raise ValueError(f"S2 task {task} frozen-cache profile changed")
            arm_conditions = {
                "correct_fit0": (task, correct[0]),
                "correct_fit1": (task, correct[1]),
                "correct_held": (task, held[0]),
                "wrong_fit0": (wrong_task, wrong[0]),
                "wrong_fit1": (wrong_task, wrong[1]),
            }
            role, task_split = _task_axes(config, task)
            for arm, (bank_task, condition) in arm_conditions.items():
                jobs.append(
                    {
                        "id": f"s{step:04d}_t{task:03d}_{arm}",
                        "task": task,
                        "role": role,
                        "split": task_split,
                        "arm": arm,
                        "program_task": task,
                        "bank_task": bank_task,
                        "video_demo": int(condition.video_demo),
                        "sampled_frames": int(condition.sampled_frames),
                        "receives_gradient": False,
                        "checkpoint_optimizer_step": step,
                        "checkpoint_path": str(checkpoint["path"]),
                        "estimated_cost": int(condition.sampled_frames)
                        * int(config["evaluation"]["panel_visits"]),
                    }
                )
    arm_order = {name: index for index, name in enumerate(BANK_SET_SHARED_ARMS)}
    step_order = {int(row["optimizer_step"]): index for index, row in enumerate(checkpoints)}
    return sorted(
        jobs,
        key=lambda row: (
            step_order[int(row["checkpoint_optimizer_step"])],
            -int(row["estimated_cost"]),
            int(row["task"]),
            arm_order[str(row["arm"])],
        ),
    )


def prepare_job_queue(
    *,
    config_path: Path,
    base_config_path: Path,
    asset_root: Path,
    compiler_run: Path,
    checkpoint_paths: Sequence[Path],
    output_dir: Path,
    worker_count: int,
) -> dict[str, Any]:
    if not 1 <= worker_count <= 6 or len(checkpoint_paths) != 2:
        raise ValueError("S2 evaluator worker/checkpoint count changed")
    checkpoints = [
        checkpoint_authority(
            config_path=config_path,
            compiler_run=compiler_run,
            checkpoint=path,
        )
        for path in checkpoint_paths
    ]
    expected_steps = load_bank_set_shared_config(config_path)["evaluation"][
        "checkpoint_optimizer_steps"
    ]
    if [row["optimizer_step"] for row in checkpoints] != expected_steps:
        raise ValueError("S2 adjacent checkpoint order changed")
    jobs = build_job_queue(
        config_path=config_path,
        base_config_path=base_config_path,
        asset_root=asset_root,
        checkpoints=checkpoints,
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("S2 evaluation output root is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("claims", "results", "workers"):
        (output_dir / name).mkdir()
    payload = {
        "schema_version": BANK_SET_SHARED_QUEUE_SCHEMA,
        "status": "ready",
        "worker_count": worker_count,
        "config": {"path": str(config_path), "bytes": config_path.stat().st_size},
        "base_config": str(base_config_path),
        "compiler_run": str(compiler_run),
        "checkpoints": checkpoints,
        "queue_policy": "persistent_workers_atomic_dynamic_claim_long_first",
        "jobs": jobs,
    }
    write_json_atomic(output_dir / "queue.json", payload)
    return payload


def _claim_job(
    output_dir: Path, jobs: Sequence[Mapping[str, Any]], worker_index: int
) -> dict[str, Any] | None:
    for row in jobs:
        job = dict(row)
        job_id = str(job["id"])
        if (output_dir / "results" / f"{job_id}.json").is_file():
            continue
        claim = output_dir / "claims" / job_id
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        write_json_atomic(claim / "claim.json", {
            "worker_index": worker_index, "pid": os.getpid(), "claimed_at": time.time()
        })
        return job
    return None


def _validate_backend_row(row: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    panel = row.get("panel_b", {})
    target = row.get("target_authority", {})
    expected_target = (
        "each_bank_frozen_r5_base_residual"
        if str(job["arm"]).startswith("correct")
        else "task_wrong_fit0_one_round_functional_free_delta_suppressive_teacher"
    )
    if (
        int(row.get("task", -1)) != int(job["task"])
        or row.get("arm") != job["arm"]
        or len(panel.get("rows", ())) != 16
        or not isinstance(row.get("functional_recovery"), (int, float))
        or target.get("effective_target") != expected_target
        or target.get("family_denominator")
        != "wrong_fit0_r5_base_to_suppressive_teacher_squared_distance"
    ):
        raise ValueError(f"S2 backend result contract changed for {job['id']}")


def evaluate_worker(args: Any) -> dict[str, Any]:
    """Persistently claim jobs; one real bank is prepared and released per job."""

    state = git_state(Path(__file__).resolve().parents[4])
    if args.mode == "formal" and (
        not git_state_is_clean_pushed_or_frozen_authority(state)
        or state.get("branch") != ""
        or state.get("upstream") is not None
    ):
        raise ValueError("formal S2 evaluation requires detached frozen authority")
    queue = read_json(args.output_dir / "queue.json")
    if (
        queue.get("schema_version") != BANK_SET_SHARED_QUEUE_SCHEMA
        or queue.get("status") != "ready"
        or int(queue.get("worker_count", -1)) != args.worker_count
        or not 0 <= args.worker_index < args.worker_count
    ):
        raise ValueError("S2 evaluation queue authority changed")
    backend = importlib.import_module(_BACKEND_MODULE)
    required = (
        "prepare_shared_evaluation_runtime",
        "prepare_shared_target_cache",
        "load_shared_checkpoint",
        "evaluate_shared_job",
        "release_shared_job",
    )
    if any(not callable(getattr(backend, name, None)) for name in required):
        raise RuntimeError("S2 training backend omitted the evaluation interface")
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size != 1:
        raise ValueError("S2 evaluator workers must be independent single GPUs")
    runtime = None
    completed = []
    started = time.monotonic()
    target_cache: dict[int, Any] = {}
    loaded_step = None
    try:
        runtime = backend.prepare_shared_evaluation_runtime(args, context)
        while (job := _claim_job(args.output_dir, queue["jobs"], args.worker_index)):
            step = int(job["checkpoint_optimizer_step"])
            if loaded_step != step:
                backend.load_shared_checkpoint(runtime, Path(job["checkpoint_path"]))
                loaded_step = step
            task = int(job["task"])
            if task not in target_cache:
                prepared_targets = backend.prepare_shared_target_cache(
                    runtime, (task,), distributed=False
                )
                if isinstance(prepared_targets, Mapping):
                    target_cache.update(prepared_targets)
                else:
                    target_cache[task] = prepared_targets
                if task not in target_cache:
                    raise RuntimeError("S2 backend target cache omitted requested task")
            tick = time.monotonic()
            release = None
            try:
                row = backend.evaluate_shared_job(
                    runtime, job, target_cache=target_cache
                )
                _validate_backend_row(row, job)
            finally:
                release = backend.release_shared_job(runtime, job)
                torch.cuda.empty_cache()
            lifecycle = dict(release or {})
            if int(lifecycle.get("resident_real_bank_count_after_release", -1)) != 0:
                raise RuntimeError("S2 backend retained a real bank after one job")
            payload = {
                "schema_version": BANK_SET_SHARED_JOB_RESULT_SCHEMA,
                "status": "complete",
                "job": job,
                "checkpoint": next(
                    row
                    for row in queue["checkpoints"]
                    if int(row["optimizer_step"]) == step
                ),
                "metrics": dict(row),
                "bank_lifecycle": lifecycle,
                "job_seconds": time.monotonic() - tick,
                "worker_index": args.worker_index,
                "git": {"commit": state["commit"], "branch": state["branch"]},
            }
            write_json_atomic(
                args.output_dir / "results" / f"{job['id']}.json", payload
            )
            completed.append(str(job["id"]))
    finally:
        if runtime is not None and callable(getattr(runtime, "close", None)):
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    worker = {
        "schema_version": BANK_SET_SHARED_WORKER_SCHEMA,
        "status": "complete",
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "completed_job_ids": completed,
        "target_cache_task_count": len(target_cache),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json_atomic(
        args.output_dir / "workers" / f"worker_{args.worker_index:02d}.json", worker
    )
    return worker
