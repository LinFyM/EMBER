#!/usr/bin/env python3
"""Canonical cost-balanced multi-GPU evaluator for frozen PI0.5 policies."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.launcher import (
    evaluator_gpus_are_eligible as _evaluator_gpus_are_eligible,
    gpu_preflight as _gpu_preflight,
    spawn_worker_processes,
)
from ember.pi05_eval.preparation import (
    prepare_evaluation_run,
    shards_from_contract as _shards_from_contract,
)
from ember.pi05_eval.recovery import (
    active_worker_pids as _active_worker_pids,
    record_launcher_failure as _record_launcher_failure,
    validate_resume_inputs as _validate_resume_inputs,
    worker_ids as _worker_ids,
)
from ember.pi05_eval_contract import RUNTIME_REPLICA_PROFILES, load_run_contract
from ember.pi05_eval_queue import (
    initialize_queue,
    publish_json_exclusive,
    queue_summary,
    read_json_with_size,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/pi05_target_evaluation_v1.json"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--role",
        choices=(
            "all_targets",
            "development_train",
            "seen_panel",
            "validation",
            "test",
            "final_source",
            "nonheld_meta",
            "nonheld_meta_train",
            "nonheld_meta_validation",
        ),
        required=True,
    )
    parser.add_argument("--mode", choices=("smoke", "screen", "formal"), required=True)
    parser.add_argument("--state-count", type=int, required=True)
    parser.add_argument(
        "--replicas-per-gpu",
        type=int,
        choices=RUNTIME_REPLICA_PROFILES,
        required=True,
    )
    parser.add_argument("--gpu-indices", help="Comma-separated physical GPU indices.")
    parser.add_argument("--source-sft-config", type=Path)
    parser.add_argument("--source-sft-checkpoint", type=Path)
    parser.add_argument("--task-expert-config", type=Path)
    parser.add_argument("--task-expert-bank-root", type=Path)
    parser.add_argument("--task-expert-step", type=_positive_int)
    parser.add_argument("--occupancy-capture-selection", type=Path)
    parser.add_argument("--task-subset-selection", type=Path)
    parser.add_argument("--capture-stage-predicates", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        _add_prepare_arguments(commands.add_parser(name))
    for name in ("start", "resume", "aggregate"):
        commands.add_parser(name).add_argument("--output-dir", type=Path, required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--output-dir", type=Path, required=True)
    worker.add_argument("--worker-id", required=True)
    return parser.parse_args()


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def prepare_run(args: argparse.Namespace) -> dict[str, Any]:
    summary = prepare_evaluation_run(
        args,
        repo_root=REPO_ROOT,
        command=sys.argv,
    )
    print(json.dumps(summary, sort_keys=True))
    return summary


def _finalize_aggregate(output_dir: Path) -> dict[str, Any]:
    from ember.pi05_eval_results import aggregate_run

    result = aggregate_run(output_dir)
    completion, completion_bytes = read_json_with_size(
        output_dir / "launcher_completion.json"
    )
    _, results_bytes = read_json_with_size(output_dir / "results.json")
    summary = {
        "schema_version": "ember_pi05_eval_run_summary_v1",
        "contract_reference": result["contract_reference"],
        "launcher_completion_bytes": completion_bytes,
        "results_bytes": results_bytes,
        "invocation_id": completion["invocation_id"],
        "launcher_started_unix": completion["started_unix"],
        "completed_unix": completion["finished_unix"],
        "panel_active_wall_seconds": result["overall"]["evaluation_wall_seconds"],
        "successes": result["overall"]["successes"],
        "episodes": result["overall"]["episodes"],
        "effective_rollouts_per_second": result["overall"][
            "effective_rollouts_per_second"
        ],
    }
    path = output_dir / "run_summary.json"
    if path.exists():
        observed, _ = read_json_with_size(path)
        if observed != summary:
            raise Pi05EvaluationError("existing evaluator run summary differs")
    else:
        publish_json_exclusive(path, summary)
    print(json.dumps({"event": "complete", **result["overall"]}, sort_keys=True))
    return result


def _recover_locked_queue(
    output_dir: Path, *, resume: bool
) -> tuple[dict[str, Any], bool]:
    active = _active_worker_pids(output_dir)
    if active:
        raise Pi05EvaluationError(f"PI05 evaluator workers are already active: {active}")
    contract = load_run_contract(output_dir / "run_contract.json")
    _validate_resume_inputs(contract)
    shards = _shards_from_contract(contract)
    initialize_queue(
        output_dir / "queue.sqlite3",
        shards,
        contract_reference=contract["contract_reference"],
        recover_claims=resume,
        retry_failed=resume,
    )
    results_exists = (output_dir / "results.json").exists()
    completion_exists = (output_dir / "launcher_completion.json").exists()
    if results_exists and not completion_exists:
        raise Pi05EvaluationError("PI05 evaluation has unowned aggregate results")
    queue = queue_summary(output_dir / "queue.sqlite3")
    complete = queue["status_counts"] == {"complete": len(shards)}
    if complete and not completion_exists:
        raise Pi05EvaluationError("queue completed without launcher completion evidence")
    if completion_exists and not complete:
        raise Pi05EvaluationError("launcher completion exists for an incomplete queue")
    return contract, bool(completion_exists and complete)


def _fail_launcher_invocation(
    output_dir: Path,
    *,
    invocation_id: str,
    started_unix: float,
    processes: Mapping[str, subprocess.Popen[bytes]],
    return_codes: dict[str, int],
    queue: dict[str, Any],
    launch_error: BaseException | None,
) -> None:
    finished_unix = time.time()
    error_text = repr(launch_error) if launch_error is not None else None
    _append_jsonl(
        output_dir / "invocations.jsonl",
        {
            "event": "failed",
            "unix": finished_unix,
            "invocation_id": invocation_id,
            "wall_seconds": finished_unix - started_unix,
            "return_codes": return_codes,
            "error": error_text,
        },
    )
    failure = _record_launcher_failure(
        output_dir,
        return_codes=return_codes,
        queue=queue,
        invocation_id=invocation_id,
        worker_pids={worker_id: process.pid for worker_id, process in processes.items()},
        error=error_text,
    )
    if launch_error is not None and not isinstance(launch_error, Exception):
        raise launch_error
    raise Pi05EvaluationError(f"PI05 evaluator workers failed; evidence: {failure}")


def _publish_launcher_completion(
    output_dir: Path,
    *,
    contract: Mapping[str, Any],
    invocation_id: str,
    started_unix: float,
    worker_ids: tuple[str, ...],
    processes: Mapping[str, subprocess.Popen[bytes]],
    return_codes: dict[str, int],
    queue: dict[str, Any],
    preflight: Mapping[str, Any],
) -> None:
    finished_unix = time.time()
    completion = {
        "schema_version": "ember_pi05_eval_launcher_completion_v1",
        "contract_reference": contract["contract_reference"],
        "invocation_id": invocation_id,
        "started_unix": started_unix,
        "finished_unix": finished_unix,
        "wall_seconds": finished_unix - started_unix,
        "worker_ids": list(worker_ids),
        "worker_pids": {worker_id: process.pid for worker_id, process in processes.items()},
        "return_codes": return_codes,
        "queue": queue,
        "preflight": dict(preflight),
    }
    publish_json_exclusive(output_dir / "launcher_completion.json", completion)
    _append_jsonl(
        output_dir / "invocations.jsonl",
        {
            "event": "completed",
            "unix": finished_unix,
            "invocation_id": invocation_id,
            "wall_seconds": finished_unix - started_unix,
            "return_codes": return_codes,
        },
    )


def _start_workers_locked(output_dir: Path, *, resume: bool) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    contract, ready_to_aggregate = _recover_locked_queue(output_dir, resume=resume)
    if ready_to_aggregate:
        return _finalize_aggregate(output_dir)
    physical_gpu_ids = tuple(int(value) for value in contract["parallel"]["physical_gpu_ids"])
    preflight = _gpu_preflight(physical_gpu_ids)
    if not _evaluator_gpus_are_eligible(preflight):
        raise Pi05EvaluationError("selected GPUs do not satisfy evaluator admission limits")
    if preflight.get("device_names") != ["NVIDIA A40"] * len(physical_gpu_ids):
        raise Pi05EvaluationError("PI05 evaluation requires NVIDIA A40 devices")
    worker_ids = _worker_ids(
        int(contract["parallel"]["replicas_per_gpu"]), physical_gpu_ids
    )
    invocation_id = uuid.uuid4().hex
    started_unix = time.time()
    _append_jsonl(
        output_dir / "invocations.jsonl",
        {
            "event": "resume_started" if resume else "started",
            "unix": started_unix,
            "invocation_id": invocation_id,
            "argv": sys.argv,
            "contract_reference": contract["contract_reference"],
            "worker_ids": worker_ids,
            "preflight": preflight,
        },
    )
    processes, return_codes, launch_error = spawn_worker_processes(
        output_dir,
        contract,
        worker_ids,
        invocation_id=invocation_id,
        repo_root=REPO_ROOT,
        script_path=Path(__file__).resolve(),
    )
    queue = queue_summary(output_dir / "queue.sqlite3")
    failed = (
        launch_error is not None
        or len(return_codes) != len(worker_ids)
        or any(code != 0 for code in return_codes.values())
        or set(queue["status_counts"]) != {"complete"}
    )
    if failed:
        _fail_launcher_invocation(
            output_dir,
            invocation_id=invocation_id,
            started_unix=started_unix,
            processes=processes,
            return_codes=return_codes,
            queue=queue,
            launch_error=launch_error,
        )
    _publish_launcher_completion(
        output_dir,
        contract=contract,
        invocation_id=invocation_id,
        started_unix=started_unix,
        worker_ids=worker_ids,
        processes=processes,
        return_codes=return_codes,
        queue=queue,
        preflight=preflight,
    )
    return _finalize_aggregate(output_dir)


def start_workers(output_dir: Path, *, resume: bool) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    lock_path = output_dir / ".launcher.lock"
    try:
        lock = lock_path.open("a+b")
    except OSError as error:
        raise Pi05EvaluationError(f"PI05 evaluation run is not prepared: {output_dir}") from error
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Pi05EvaluationError("another evaluator launcher owns this run") from error
        return _start_workers_locked(output_dir, resume=resume)


def main() -> int:
    args = parse_args()
    if args.command in {"prepare", "run"}:
        prepare_run(args)
        if args.command == "run":
            start_workers(args.output_dir, resume=False)
    elif args.command == "start":
        start_workers(args.output_dir, resume=False)
    elif args.command == "resume":
        start_workers(args.output_dir, resume=True)
    elif args.command == "worker":
        from ember.pi05_evaluation import run_worker

        print(json.dumps(run_worker(output_dir=args.output_dir, worker_id=args.worker_id)))
    else:
        _finalize_aggregate(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
