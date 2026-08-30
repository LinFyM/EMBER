#!/usr/bin/env python3
"""Preseal the exact-language wrong-bank cache used by the G3 interaction Gate."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ember.ecp.joint_program_primal.evaluation import _task_conditions
from ember.ecp.joint_program_primal.routing_control import (
    ROUTING_TASK_IDS,
    prepare_routing_control_runtime,
)
from ember.ecp.joint_program_primal.train_step import (
    prepare_program_bank_condition,
)
from ember.ecp.joint_program_primal.training import build_parser, finalize_args
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_source_checkpoint import write_json_atomic
from ember.pi05_source_setup import initialize_distributed


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_SEAL_SCHEMA = "ember_ecp_program_bank_condition_cache_seal_v1"


def _pairs(runtime: Any) -> tuple[tuple[int, int, Any], ...]:
    meta = tuple(map(int, runtime.config["task_split"]["gradient_meta"]))
    target = tuple(map(int, runtime.config["task_split"]["gradient_target"]))
    rows = []
    for program_task in ROUTING_TASK_IDS:
        role_tasks = meta if program_task in meta else target
        for bank_task in role_tasks:
            if bank_task == program_task:
                continue
            for condition in runtime.task_conditions[bank_task].fit_views:
                rows.append((program_task, bank_task, condition))
        unseen_task = 2 if program_task in meta else 74
        unseen_condition = _task_conditions(runtime, unseen_task)[0]
        rows.append((program_task, unseen_task, unseen_condition))
    if (
        len(rows) != 90
        or len(
            {
                (program, bank, condition.video_demo)
                for program, bank, condition in rows
            }
        )
        != 90
    ):
        raise RuntimeError("interaction cache-seal allowlist changed")
    return tuple(rows)


def _assignments(
    rows: tuple[tuple[int, int, Any], ...], worker_count: int
) -> tuple[tuple[tuple[int, int, Any], ...], ...]:
    if not 1 <= worker_count <= 6:
        raise ValueError("interaction cache-seal worker count changed")
    assigned: list[list[tuple[int, int, Any]]] = [
        [] for _ in range(worker_count)
    ]
    loads = [0] * worker_count
    for row in sorted(
        rows,
        key=lambda value: (
            -int(value[2].sampled_frames),
            value[0],
            value[1],
            value[2].video_demo,
        ),
    ):
        worker = min(range(worker_count), key=lambda value: (loads[value], value))
        assigned[worker].append(row)
        loads[worker] += int(row[2].sampled_frames)
    return tuple(tuple(row) for row in assigned)


def main() -> int:
    parser = build_parser()
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    args = finalize_args(parser.parse_args())
    if (
        args.mode != "profile"
        or args.phase != "joint"
        or args.task is not None
        or args.resume is not None
        or args.stop_after_step is not None
        or not 0 <= args.worker_index < args.worker_count
    ):
        raise ValueError("interaction cache-seal invocation changed")
    state = git_state(REPO_ROOT)
    if (
        not git_state_is_clean_pushed_or_frozen_authority(state)
        or state.get("branch") != ""
        or state.get("upstream") is not None
    ):
        raise ValueError("interaction cache seal requires detached authority")
    setattr(args, "skip_routing_initialization", True)
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size != 1:
        raise ValueError("interaction cache-seal workers must be independent")
    runtime = None
    started = time.monotonic()
    try:
        runtime = prepare_routing_control_runtime(args, context)
        assignments = _assignments(_pairs(runtime), args.worker_count)
        teacher_reads = runtime.native_teachers.tensor_reads
        records = []
        for program_task, bank_task, condition in assignments[args.worker_index]:
            tick = time.monotonic()
            prepared, metrics = prepare_program_bank_condition(
                runtime,
                language_authority_id=program_task,
                bank_condition=condition,
            )
            if (
                int(metrics.get("conditioning_language_authority_id", -1))
                != program_task
                or int(metrics.get("video_bank_authority_id", -1)) != bank_task
                or prepared.program is not None
                or prepared.evidence is None
            ):
                raise RuntimeError("interaction cache-seal pairing changed")
            del prepared
            records.append(
                {
                    "program_task": program_task,
                    "bank_task": bank_task,
                    "bank_video_demo": condition.video_demo,
                    "sampled_frames": condition.sampled_frames,
                    "cache_state": metrics["frozen_condition_cache"],
                    "file_bytes": metrics["frozen_condition_cache_file_bytes"],
                    "build_seconds": metrics[
                        "frozen_condition_cache_build_seconds"
                    ],
                    "load_seconds": metrics[
                        "frozen_condition_cache_load_seconds"
                    ],
                    "elapsed_seconds": time.monotonic() - tick,
                }
            )
        torch.cuda.synchronize(context.device)
        if runtime.native_teachers.tensor_reads != teacher_reads:
            raise RuntimeError("interaction cache seal read factor teachers")
        payload = {
            "schema_version": CACHE_SEAL_SCHEMA,
            "status": "complete",
            "worker_index": args.worker_index,
            "worker_count": args.worker_count,
            "git": {"commit": state["commit"], "branch": state["branch"]},
            "program_bank_condition_cache_root": str(
                args.program_bank_condition_cache_root
            ),
            "pair_count": len(records),
            "built_count": sum(row["cache_state"] == "built" for row in records),
            "hit_count": sum(row["cache_state"] == "hit" for row in records),
            "file_bytes": sum(int(row["file_bytes"]) for row in records),
            "native_teacher_tensor_reads": runtime.native_teachers.tensor_reads,
            "elapsed_seconds": time.monotonic() - started,
            "records": records,
        }
        write_json_atomic(args.output_dir / "cache_seal.json", payload)
        write_json_atomic(
            args.output_dir / "completion.json",
            {
                "schema_version": CACHE_SEAL_SCHEMA,
                "status": "complete",
                "worker_index": args.worker_index,
                "pair_count": len(records),
            },
        )
        print(json.dumps({key: payload[key] for key in (
            "worker_index", "pair_count", "built_count", "hit_count",
            "file_bytes", "elapsed_seconds",
        )}, sort_keys=True), flush=True)
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
