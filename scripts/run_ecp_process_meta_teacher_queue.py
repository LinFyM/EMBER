#!/usr/bin/env python3
"""Prepare and run the six-worker ECP process-teacher Gate A queue."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping

from ember.ecp.process_meta import (
    ProcessMetaError,
    load_process_meta_authority,
)
from ember.pi05_assets import write_json_atomic
from ember.pi05_eval.launcher import (
    evaluator_gpus_are_eligible,
    gpu_preflight,
    terminate_owned_workers,
)
from ember.pi05_eval_queue import (
    EvaluationShard,
    initialize_queue,
    queue_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = REPO_ROOT / "scripts/collect_ecp_process_meta_teacher.py"
CONTRACT_SCHEMA = "ember_ecp_process_meta_teacher_queue_run_v1"


def _gpu_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "GPU IDs must be comma-separated integers"
        ) from error
    if len(result) != 6 or len(set(result)) != 6 or min(result) < 0:
        raise argparse.ArgumentTypeError("Gate A requires six distinct GPU IDs")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--physical-gpu-ids", type=_gpu_ids, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    start = commands.add_parser("start")
    start.add_argument("--output-dir", type=Path, required=True)
    return parser


def _git_authority() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ProcessMetaError("Gate A must be prepared from a clean frozen worktree")
    return {"commit": commit, "worktree": str(REPO_ROOT), "clean": True}


def _authority(manifest: Path) -> Any:
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    source_root = REPO_ROOT / str(raw["source_policy_authority"])
    source_contract = json.loads(
        (source_root / "run_contract.json").read_text(encoding="utf-8")
    )
    return load_process_meta_authority(
        manifest,
        repo_root=REPO_ROOT,
        libero_init_root=Path(source_contract["libero_paths"]["init_states"]),
    )


def _shards(authority: Any, variant_name: str) -> tuple[EvaluationShard, ...]:
    horizon = int(authority.rollout["horizon"])
    return tuple(
        EvaluationShard(
            job_id=f"{variant_name}-state{state_id:03d}",
            ordinal=state_id,
            suite=variant_name,
            task_id=authority.family.base_task_id,
            horizon=horizon,
            init_state_ids=(state_id,),
            estimated_cost=horizon,
        )
        for state_id in authority.family.init_state_ids
    )


def _queue_path(output_dir: Path, variant_name: str) -> Path:
    return output_dir / "queues" / f"{variant_name}.sqlite3"


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ProcessMetaError("Gate A output root already exists")
    authority = _authority(manifest)
    variants = tuple(variant.name for variant in authority.family.variants)
    workers = []
    for variant_name, gpu_group in zip(
        variants,
        (args.physical_gpu_ids[:3], args.physical_gpu_ids[3:]),
        strict=True,
    ):
        queue_path = _queue_path(output_dir, variant_name)
        initialize_queue(
            queue_path,
            _shards(authority, variant_name),
            contract_reference=f"{CONTRACT_SCHEMA}:{authority.family.family_id}:{variant_name}",
        )
        workers.extend(
            {
                "worker_id": f"{variant_name}-gpu{gpu}",
                "variant_name": variant_name,
                "physical_gpu_id": gpu,
                "queue_path": str(queue_path),
            }
            for gpu in gpu_group
        )
    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "status": "prepared",
        "prepared_unix": time.time(),
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "teacher_mode": "phase_expert",
        "family_id": authority.family.family_id,
        "variant_names": list(variants),
        "physical_gpu_ids": list(args.physical_gpu_ids),
        "workers": workers,
        "queue_policy": {
            "jobs_per_variant": len(authority.family.init_state_ids),
            "states_per_job": 1,
            "persistent_workers_per_variant": 3,
            "ordering": "estimated_cost_desc_then_state_ordinal",
        },
        "git": _git_authority(),
    }
    write_json_atomic(output_dir / "run_contract.json", contract)
    return contract


def _spawn_workers(
    output_dir: Path,
    contract: Mapping[str, Any],
) -> tuple[dict[str, subprocess.Popen[bytes]], dict[str, int]]:
    processes: dict[str, subprocess.Popen[bytes]] = {}
    (output_dir / "worker_logs").mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for worker in contract["workers"]:
            worker_id = str(worker["worker_id"])
            gpu = int(worker["physical_gpu_id"])
            log = stack.enter_context(
                (output_dir / "worker_logs" / f"{worker_id}.log").open("ab")
            )
            environment = os.environ.copy()
            environment.update(
                PYTHONPATH=str(REPO_ROOT / "src"),
                CUDA_DEVICE_ORDER="PCI_BUS_ID",
                CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS="8",
            )
            command = [
                sys.executable,
                str(COLLECTOR),
                "--manifest",
                str(contract["manifest"]),
                "--variant",
                str(worker["variant_name"]),
                "--queue-path",
                str(worker["queue_path"]),
                "--worker-id",
                worker_id,
                "--physical-gpu-id",
                str(gpu),
                "--output-dir",
                str(output_dir),
                "--teacher-mode",
                str(contract["teacher_mode"]),
            ]
            processes[worker_id] = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        while True:
            observed = {
                worker_id: process.poll() for worker_id, process in processes.items()
            }
            if any(code not in (None, 0) for code in observed.values()):
                terminate_owned_workers(processes)
                break
            if all(code == 0 for code in observed.values()):
                break
            time.sleep(0.5)
        return_codes = {
            worker_id: int(process.wait()) for worker_id, process in processes.items()
        }
    return processes, return_codes


def start(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    contract_path = output_dir / "run_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or contract.get("status") != "prepared"
        or contract.get("git") != _git_authority()
        or (output_dir / "launcher_completion.json").exists()
    ):
        raise ProcessMetaError("Gate A prepared authority changed")
    lock_path = output_dir / ".launcher.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ProcessMetaError("another Gate A launcher owns this run") from error
        physical_gpu_ids = tuple(int(value) for value in contract["physical_gpu_ids"])
        preflight = gpu_preflight(physical_gpu_ids)
        if (
            not evaluator_gpus_are_eligible(preflight)
            or preflight.get("device_names") != ["NVIDIA A40"] * 6
        ):
            raise ProcessMetaError("selected Gate A GPUs are no longer eligible")
        write_json_atomic(output_dir / "launcher_preflight.json", preflight)
        started = time.time()
        processes, return_codes = _spawn_workers(output_dir, contract)
        queues = {
            variant_name: queue_summary(_queue_path(output_dir, variant_name))
            for variant_name in contract["variant_names"]
        }
        passed_runtime = bool(
            len(return_codes) == 6
            and all(code == 0 for code in return_codes.values())
            and all(
                summary["status_counts"] == {"complete": 50}
                and summary["completed_rows"] == 50
                for summary in queues.values()
            )
        )
        completion = {
            "schema_version": "ember_ecp_process_meta_teacher_queue_completion_v1",
            "status": "complete" if passed_runtime else "engineering_failed",
            "started_unix": started,
            "finished_unix": time.time(),
            "wall_seconds": time.time() - started,
            "worker_pids": {
                worker_id: process.pid for worker_id, process in processes.items()
            },
            "return_codes": return_codes,
            "queues": queues,
            "preflight": preflight,
        }
        write_json_atomic(output_dir / "launcher_completion.json", completion)
        if not passed_runtime:
            raise ProcessMetaError("Gate A workers did not complete the fixed panel")
        return completion


def main() -> None:
    args = build_parser().parse_args()
    result = prepare(args) if args.command == "prepare" else start(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
