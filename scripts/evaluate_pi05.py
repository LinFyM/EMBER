#!/usr/bin/env python3
"""Canonical cost-balanced multi-GPU evaluator for frozen PI05 policies."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pwd
import subprocess
import sys
import time
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.libero_evaluation import sha256_file
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import (
    build_run_contract,
    git_state,
    inspect_installed_target_tasks,
    inspect_source_checkpoint,
    inspect_tokenizer,
    load_evaluation_authorities,
    load_run_contract,
)
from ember.pi05_eval_queue import (
    EvaluationTask,
    build_cost_balanced_shards,
    failed_jobs,
    initialize_queue,
    publish_json_exclusive,
    queue_summary,
    read_json_with_sha256,
    validate_worker_layout,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/pi05_target_evaluation_v1.json"


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--role",
        choices=("all_targets", "development_train", "validation", "test", "final_source"),
        required=True,
    )
    parser.add_argument("--mode", choices=("smoke", "screen", "formal"), required=True)
    parser.add_argument("--state-count", type=int, required=True)
    parser.add_argument("--replicas-per-gpu", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--as-writer-config", type=Path)
    parser.add_argument("--as-writer-checkpoint", type=Path)
    parser.add_argument("--writer-feature-cache", type=Path)
    parser.add_argument(
        "--writer-video-condition",
        choices=("correct", "cross_suite_wrong"),
    )
    parser.add_argument("--source-sft-config", type=Path)
    parser.add_argument("--source-sft-checkpoint", type=Path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    _add_prepare_arguments(prepare)
    run = commands.add_parser("run")
    _add_prepare_arguments(run)
    start = commands.add_parser("start")
    start.add_argument("--output-dir", type=Path, required=True)
    resume = commands.add_parser("resume")
    resume.add_argument("--output-dir", type=Path, required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--output-dir", type=Path, required=True)
    worker.add_argument("--worker-id", required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _shards_from_contract(contract: dict[str, Any]) -> tuple[Any, ...]:
    tasks = tuple(
        EvaluationTask(
            suite=row["suite"],
            task_id=int(row["task_id"]),
            horizon=int(row["horizon"]),
            init_state_ids=tuple(int(value) for value in row["init_state_ids"]),
        )
        for row in contract["tasks"]
    )
    return build_cost_balanced_shards(
        tasks,
        env_batch_size=int(contract["parallel"]["envs_per_replica"]),
        target_cost=int(contract["parallel"]["shard_target_cost"]),
    )


def _all_or_none(values: Sequence[Any], label: str) -> bool:
    if any(value is not None for value in values) and not all(
        value is not None for value in values
    ):
        raise Pi05EvaluationError(f"{label} evaluation requires all declared assets")
    return all(value is not None for value in values)


def _writer_requested(args: argparse.Namespace) -> bool:
    return _all_or_none(
        (args.as_writer_config, args.as_writer_checkpoint, args.writer_feature_cache,
         args.writer_video_condition),
        "AS-Writer",
    )


def _source_sft_requested(args: argparse.Namespace) -> bool:
    return _all_or_none(
        (args.source_sft_config, args.source_sft_checkpoint), "Source-SFT"
    )


def _adapter_requests(args: argparse.Namespace) -> tuple[bool, bool]:
    writer_requested = _writer_requested(args)
    source_sft_requested = _source_sft_requested(args)
    if writer_requested and source_sft_requested:
        raise Pi05EvaluationError("AS-Writer and Source-SFT adapters are mutually exclusive")
    return writer_requested, source_sft_requested


def _inspect_writer_adapter(
    *,
    config_path: Path,
    checkpoint: Path,
    feature_cache: Path,
    source: Mapping[str, Any],
    tasks: Sequence[Any],
    video_condition: str,
    video_seed: int,
    require_formal: bool,
) -> dict[str, Any]:
    from ember.lora import LoRAContractError
    from ember.writer.feature_cache import FeatureCacheError
    from ember.writer.inference import inspect_as_writer_evaluation
    from ember.writer.model import WriterModelError

    try:
        return inspect_as_writer_evaluation(
            config_path=config_path,
            checkpoint=checkpoint,
            feature_cache=feature_cache,
            source=source,
            task_keys=tuple((task.suite, int(task.task_id)) for task in tasks),
            video_condition=video_condition,
            video_seed=video_seed,
            require_formal=require_formal,
        )
    except (FeatureCacheError, LoRAContractError, WriterModelError) as error:
        raise Pi05EvaluationError(str(error)) from error


def _inspect_source_sft_adapter(
    *,
    config_path: Path,
    checkpoint: Path,
    source: Mapping[str, Any],
    tasks: Sequence[Any],
    evaluation_role: str,
    require_formal: bool,
) -> dict[str, Any]:
    from ember.source_sft.inference import inspect_source_sft_evaluation

    return inspect_source_sft_evaluation(
        config_path=config_path,
        checkpoint=checkpoint,
        source=source,
        task_keys=tuple((task.suite, int(task.task_id)) for task in tasks),
        evaluation_role=evaluation_role,
        require_formal=require_formal,
    )


def prepare_run(args: argparse.Namespace) -> dict[str, Any]:
    writer_requested, source_sft_requested = _adapter_requests(args)
    adapter_requested = writer_requested or source_sft_requested
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise Pi05EvaluationError(f"PI05 evaluation output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    authorities = load_evaluation_authorities(args.config, REPO_ROOT)
    formal_count = int(authorities.config["environment"]["fixed_init_state_count"])
    if args.mode == "formal" and args.state_count != formal_count:
        raise Pi05EvaluationError("formal PI05 evaluation requires all 50 fixed states")
    if args.mode == "screen" and not adapter_requested and args.role != "all_targets":
        raise Pi05EvaluationError("source-base screen must cover all 40 target tasks")
    tasks, libero_paths = inspect_installed_target_tasks(
        authorities,
        role=args.role,
        state_count=args.state_count,
        libero_config_dir=output_dir / "libero_config",
    )
    model = inspect_source_checkpoint(
        authorities,
        args.source_run,
        args.checkpoint,
        evaluation_mode=args.mode,
    )
    tokenizer = inspect_tokenizer(authorities, args.tokenizer_path)
    adapter = None
    if writer_requested:
        adapter = _inspect_writer_adapter(
            config_path=args.as_writer_config.resolve(),
            checkpoint=args.as_writer_checkpoint.resolve(),
            feature_cache=args.writer_feature_cache.resolve(),
            source=model,
            tasks=tasks,
            video_condition=str(args.writer_video_condition),
            video_seed=int(authorities.config["rng"]["inference_seed"]),
            require_formal=args.mode != "smoke",
        )
    elif source_sft_requested:
        adapter = _inspect_source_sft_adapter(
            config_path=args.source_sft_config.resolve(),
            checkpoint=args.source_sft_checkpoint.resolve(),
            source=model,
            tasks=tasks,
            evaluation_role=args.role,
            require_formal=args.mode != "smoke",
        )
    contract = build_run_contract(
        authorities=authorities,
        tasks=tasks,
        libero_paths=libero_paths,
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        role=args.role,
        mode=args.mode,
        replicas_per_gpu=args.replicas_per_gpu,
        command=sys.argv,
        adapter=adapter,
    )
    publish_json_exclusive(output_dir / "run_contract.json", contract)
    shards = _shards_from_contract(contract)
    initialize_queue(
        output_dir / "queue.sqlite3",
        shards,
        contract_sha256=contract["contract_sha256"],
    )
    summary = {
        "event": "prepared",
        "contract_sha256": contract["contract_sha256"],
        "tasks": len(tasks),
        "states": sum(len(task.init_state_ids) for task in tasks),
        "shards": len(shards),
        "replicas_per_gpu": args.replicas_per_gpu,
        "arm": contract["arm"],
        "output_dir": str(output_dir),
    }
    print(json.dumps(summary, sort_keys=True))
    return summary


def _active_worker_pids(output_dir: Path) -> list[int]:
    needle = str(output_dir.resolve()).encode()
    active = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = path.read_bytes()
        except OSError:
            continue
        if b"evaluate_pi05.py" in command and b"worker" in command and needle in command:
            active.append(int(path.parent.name))
    return sorted(active)


def _gpu_preflight() -> dict[str, Any]:
    """Check storage first, then sample live GPU ownership immediately before spawn."""

    import torch

    personal_bytes = int(
        subprocess.run(
            ["du", "-sb", "/data/ymdai"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.split()[0]
    )
    data_capacity = subprocess.run(
        ["df", "-B1", "--output=size,used,avail,pcent,target", "/data"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()[-1].split()
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    if len(gpu_query) != 8:
        raise Pi05EvaluationError(f"PI05 evaluation requires eight GPUs, found {len(gpu_query)}")
    applications = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    if applications:
        owned = []
        for application in applications:
            fields = [value.strip() for value in application.split(",")]
            owner = "unknown"
            if len(fields) >= 2 and fields[1].isdigit():
                try:
                    status = Path(f"/proc/{fields[1]}/status").read_text(encoding="utf-8")
                    uid_line = next(line for line in status.splitlines() if line.startswith("Uid:"))
                    owner = pwd.getpwuid(int(uid_line.split()[1])).pw_name
                except (OSError, KeyError, StopIteration, ValueError):
                    pass
            owned.append(f"{application}, owner={owner}")
        raise Pi05EvaluationError(
            "GPU preflight found existing compute processes; refusing to interfere: "
            + " | ".join(owned)
        )
    return {
        "unix": time.time(),
        "gpus": gpu_query,
        "compute_applications": [],
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "personal_bytes": personal_bytes,
        "personal_cap_bytes": 500_000_000_000,
        "data_filesystem": {
            "size": int(data_capacity[0]),
            "used": int(data_capacity[1]),
            "available": int(data_capacity[2]),
            "percent": data_capacity[3],
            "mount": data_capacity[4],
        },
    }


def _validate_resume_inputs(contract: dict[str, Any]) -> None:
    config_path = Path(contract["authorities"]["config_path"])
    authorities = load_evaluation_authorities(config_path, REPO_ROOT)
    current_git = git_state(REPO_ROOT)
    if (
        current_git["commit"] != contract["git"]["commit"]
        or contract["mode"] != "smoke" and current_git["dirty_paths"]
    ):
        raise Pi05EvaluationError("evaluator checkout differs from the sealed run commit")
    model = inspect_source_checkpoint(
        authorities,
        Path(contract["model"]["source_run"]),
        Path(contract["model"]["checkpoint"]),
        evaluation_mode=contract["mode"],
    )
    tokenizer = inspect_tokenizer(authorities, Path(contract["tokenizer"]["path"]))
    if model != contract["model"] or tokenizer != contract["tokenizer"]:
        raise Pi05EvaluationError("evaluation model or tokenizer changed after prepare")
    normalization = Path(contract["normalization"]["path"])
    if sha256_file(normalization) != contract["normalization"]["sha256"]:
        raise Pi05EvaluationError("evaluation normalization changed after prepare")
    adapter = contract.get("adapter")
    if adapter is not None:
        tasks = tuple(
            argparse.Namespace(suite=row["suite"], task_id=int(row["task_id"]))
            for row in contract["tasks"]
        )
        if adapter.get("kind") == "shared_source_sft_lora":
            observed = _inspect_source_sft_adapter(
                config_path=Path(adapter["config"]["path"]),
                checkpoint=Path(adapter["checkpoint"]["path"]),
                source=model,
                tasks=tasks,
                evaluation_role=str(adapter["evaluation_role"]),
                require_formal=contract["mode"] != "smoke",
            )
        else:
            observed = _inspect_writer_adapter(
                config_path=Path(adapter["config"]["path"]),
                checkpoint=Path(adapter["checkpoint"]["path"]),
                feature_cache=Path(adapter["feature_cache"]["root"]),
                source=model,
                tasks=tasks,
                video_condition=str(adapter["video_condition"]),
                video_seed=int(adapter["video_schedule"]["seed"]),
                require_formal=contract["mode"] != "smoke",
            )
        if observed != adapter:
            raise Pi05EvaluationError("evaluation adapter assets changed after prepare")


def _worker_ids(replicas_per_gpu: int) -> tuple[str, ...]:
    values = tuple(
        f"{gpu}-r{replica}" for gpu in range(8) for replica in range(replicas_per_gpu)
    )
    validate_worker_layout(values, replicas_per_gpu)
    return values


def _record_launcher_failure(
    output_dir: Path,
    *,
    return_codes: dict[str, int],
    queue: dict[str, Any],
    invocation_id: str,
    worker_pids: dict[str, int],
    error: str | None = None,
) -> Path:
    logs = []
    for path in sorted((output_dir / "worker_logs").glob("*.log")):
        logs.append(
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    path = output_dir / "failures" / f"launcher_{time.time_ns()}.json"
    publish_json_exclusive(
        path,
        {
            "schema_version": "ember_pi05_eval_launcher_failure_v1",
            "unix": time.time(),
            "invocation_id": invocation_id,
            "error": error,
            "worker_pids": worker_pids,
            "return_codes": return_codes,
            "queue": queue,
            "failed_jobs": list(failed_jobs(output_dir / "queue.sqlite3")),
            "worker_logs": logs,
        },
    )
    return path


def _terminate_owned_workers(processes: dict[str, subprocess.Popen[bytes]]) -> None:
    """Stop only workers created by this launcher after a local launch failure."""

    live = [process for process in processes.values() if process.poll() is None]
    for process in live:
        process.terminate()
    deadline = time.monotonic() + 10.0
    while live and time.monotonic() < deadline:
        live = [process for process in live if process.poll() is None]
        if live:
            time.sleep(0.1)
    for process in live:
        process.kill()
    for process in processes.values():
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _finalize_aggregate(output_dir: Path) -> dict[str, Any]:
    from ember.pi05_eval_results import aggregate_run

    result = aggregate_run(output_dir)
    completion, completion_sha256 = read_json_with_sha256(
        output_dir / "launcher_completion.json"
    )
    _, results_sha256 = read_json_with_sha256(output_dir / "results.json")
    summary = {
        "schema_version": "ember_pi05_eval_run_summary_v1",
        "contract_sha256": result["contract_sha256"],
        "launcher_completion_sha256": completion_sha256,
        "results_sha256": results_sha256,
        "invocation_id": completion["invocation_id"],
        "launcher_started_unix": completion["started_unix"],
        "completed_unix": completion["finished_unix"],
        "panel_active_wall_seconds": completion["wall_seconds"],
        "successes": result["overall"]["successes"],
        "episodes": result["overall"]["episodes"],
        "effective_rollouts_per_second": result["overall"][
            "effective_rollouts_per_second"
        ],
    }
    summary_path = output_dir / "run_summary.json"
    if summary_path.exists():
        observed, _ = read_json_with_sha256(summary_path)
        if observed != summary:
            raise Pi05EvaluationError("existing evaluator run summary differs")
    else:
        publish_json_exclusive(summary_path, summary)
    print(json.dumps({"event": "complete", **result["overall"]}, sort_keys=True))
    return result


def start_workers(output_dir: Path, *, resume: bool) -> dict[str, Any]:
    """Own the launcher before inspecting or mutating any queue state."""

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
            raise Pi05EvaluationError("another PI05 evaluator launcher owns this run") from error
        return _start_workers_locked(output_dir, resume=resume)


def _recover_locked_queue(
    output_dir: Path, *, resume: bool
) -> tuple[dict[str, Any], tuple[Any, ...], bool]:
    active = _active_worker_pids(output_dir)
    if active:
        raise Pi05EvaluationError(f"PI05 evaluator workers are already active: {active}")
    contract = load_run_contract(output_dir / "run_contract.json")
    _validate_resume_inputs(contract)
    shards = _shards_from_contract(contract)
    initialize_queue(
        output_dir / "queue.sqlite3",
        shards,
        contract_sha256=contract["contract_sha256"],
        recover_claims=resume,
        retry_failed=resume,
    )
    results_exists = (output_dir / "results.json").exists()
    completion_exists = (output_dir / "launcher_completion.json").exists()
    if results_exists and not completion_exists:
        raise Pi05EvaluationError("PI05 evaluation already has unowned aggregate results")
    queue = queue_summary(output_dir / "queue.sqlite3")
    complete = queue["status_counts"] == {"complete": len(shards)}
    if complete and not completion_exists:
        raise Pi05EvaluationError(
            "queue completed without exact launcher timing/return-code evidence; "
            "preserve this root and rerun the panel in a new output"
        )
    if completion_exists and not complete:
        raise Pi05EvaluationError("launcher completion exists for an incomplete queue")
    return contract, shards, bool(completion_exists and complete)


def _spawn_worker_processes(
    output_dir: Path,
    contract: Mapping[str, Any],
    worker_ids: Sequence[str],
    *,
    invocation_id: str,
) -> tuple[
    dict[str, subprocess.Popen[bytes]],
    dict[str, int],
    BaseException | None,
]:
    replicas = int(contract["parallel"]["replicas_per_gpu"])
    processes: dict[str, subprocess.Popen[bytes]] = {}
    return_codes: dict[str, int] = {}
    launch_error: BaseException | None = None
    (output_dir / "worker_logs").mkdir(parents=True, exist_ok=True)
    try:
        with ExitStack() as stack:
            for worker_id in worker_ids:
                gpu = worker_id.split("-r", 1)[0]
                log = stack.enter_context(
                    (output_dir / "worker_logs" / f"{worker_id}.log").open("ab")
                )
                environment = os.environ.copy()
                environment.update(
                    PYTHONPATH=str(REPO_ROOT / "src"),
                    CUDA_DEVICE_ORDER="PCI_BUS_ID",
                    CUDA_VISIBLE_DEVICES=gpu,
                    OMP_NUM_THREADS=str(
                        contract["parallel"]["omp_threads_per_worker"][str(replicas)]
                    ),
                    EMBER_PI05_EVAL_INVOCATION_ID=invocation_id,
                )
                processes[worker_id] = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "worker",
                        "--output-dir",
                        str(output_dir),
                        "--worker-id",
                        worker_id,
                    ],
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            while True:
                return_codes = {
                    worker_id: code
                    for worker_id, process in processes.items()
                    if (code := process.poll()) is not None
                }
                if any(code != 0 for code in return_codes.values()):
                    _terminate_owned_workers(processes)
                    break
                if len(return_codes) == len(processes):
                    break
                time.sleep(0.2)
            return_codes = {
                worker_id: int(process.wait())
                for worker_id, process in processes.items()
            }
    except BaseException as error:
        launch_error = error
        _terminate_owned_workers(processes)
        return_codes = {
            worker_id: int(process.returncode)
            for worker_id, process in processes.items()
            if process.returncode is not None
        }
    return processes, return_codes, launch_error


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
    worker_ids: Sequence[str],
    processes: Mapping[str, subprocess.Popen[bytes]],
    return_codes: dict[str, int],
    queue: dict[str, Any],
    preflight: Mapping[str, Any],
) -> None:
    finished_unix = time.time()
    completion = {
        "schema_version": "ember_pi05_eval_launcher_completion_v1",
        "contract_sha256": contract["contract_sha256"],
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
    contract, _, ready_to_aggregate = _recover_locked_queue(output_dir, resume=resume)
    if ready_to_aggregate:
        return _finalize_aggregate(output_dir)
    preflight = _gpu_preflight()
    if preflight["personal_bytes"] >= preflight["personal_cap_bytes"]:
        raise Pi05EvaluationError("personal storage already exceeds the 500GB hard cap")
    worker_ids = _worker_ids(int(contract["parallel"]["replicas_per_gpu"]))
    invocation_id = uuid.uuid4().hex
    started_unix = time.time()
    _append_jsonl(
        output_dir / "invocations.jsonl",
        {
            "event": "resume_started" if resume else "started",
            "unix": started_unix,
            "invocation_id": invocation_id,
            "argv": sys.argv,
            "contract_sha256": contract["contract_sha256"],
            "worker_ids": worker_ids,
            "preflight": preflight,
        },
    )
    processes, return_codes, launch_error = _spawn_worker_processes(
        output_dir,
        contract,
        worker_ids,
        invocation_id=invocation_id,
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
