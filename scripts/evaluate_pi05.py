#!/usr/bin/env python3
"""Canonical cost-balanced multi-GPU evaluator for frozen PI05 policies."""

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
from typing import Any, Mapping, Sequence

from ember.pi05_assets import Pi05EvaluationError
from ember.expert_manifold.v6_prior_checkpoint import V6_PRIOR_CHECKPOINT_SCHEMA
from ember.expert_manifold.v6_prior_contract import (
    V6_PRIOR_CONFIG_SCHEMA,
    V6_PRIOR_RUN_SCHEMA,
)
from ember.expert_manifold.video_schedule import VIDEO_CONDITIONS
from ember.pi05_eval.launcher import (
    gpu_preflight as _gpu_preflight,
    spawn_worker_processes,
    terminate_owned_workers as _terminate_owned_workers,
)
from ember.pi05_eval.reward_credit_gate import (
    validate_registered_reward_credit_output as _validate_registered_reward_credit_output,
)
from ember.pi05_eval.preparation import (
    parse_gpu_indices as _parse_gpu_indices,
    prepare_evaluation_run,
    shards_from_contract as _shards_from_contract,
)
from ember.eval_adapters import (
    adapter_requests as _adapter_requests,
    inspect_source_sft_adapter as _inspect_source_sft_adapter,
    inspect_task_expert_adapter as _inspect_task_expert_adapter,
    inspect_expert_manifold_writer_adapter as _inspect_expert_manifold_writer_adapter,
)
from ember.pi05_eval_contract import (
    RUNTIME_REPLICA_PROFILES,
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
    inspect_source_checkpoint,
    inspect_tokenizer,
    load_evaluation_authorities,
    load_run_contract,
    SEEN_PANEL_RELATIVE_PATH,
)
from ember.pi05_eval_queue import (
    failed_jobs,
    initialize_queue,
    publish_json_exclusive,
    queue_summary,
    read_json_with_size,
    validate_worker_layout,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/pi05_target_evaluation_v1.json"
WRITER_PROFILE_PREFLIGHT = "writer_generation_preflight.json"
WRITER_PROFILE_WORKER_LOG = "writer_generation_profile_worker.log"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _add_writer_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--writer-generators-per-gpu",
        type=int,
        choices=RUNTIME_REPLICA_PROFILES,
        default=1,
        help=(
            "Writer-LoRA generator processes per GPU; independent of rollout "
            "replicas and no greater than --replicas-per-gpu."
        ),
    )
    parser.add_argument(
        "--writer-generation-batch-size",
        type=_positive_int,
        default=8,
        help=(
            "Episode LoRAs dispatched together to each Writer process; choose the "
            "highest-throughput profiled value with useful device-memory headroom."
        ),
    )
    parser.add_argument(
        "--writer-lora-cache-root",
        type=Path,
        help="Optional reusable cache root shared by rollout-topology profiles.",
    )


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
    _add_writer_runtime_arguments(parser)
    parser.add_argument(
        "--gpu-indices",
        help=(
            "Comma-separated live-checked physical GPU indices to use on one host; "
            "all listed devices must satisfy the configured topology and live-idle checks."
        ),
    )
    parser.add_argument("--source-sft-config", type=Path)
    parser.add_argument("--source-sft-checkpoint", type=Path)
    parser.add_argument("--task-expert-config", type=Path)
    parser.add_argument("--task-expert-bank-root", type=Path)
    parser.add_argument("--task-expert-step", type=_positive_int)
    parser.add_argument("--expert-manifold-config", type=Path)
    parser.add_argument("--expert-manifold-checkpoint", type=Path)
    parser.add_argument("--expert-manifold-video-data-root", type=Path)
    parser.add_argument(
        "--expert-manifold-video-condition",
        choices=tuple(sorted(VIDEO_CONDITIONS)),
    )
    parser.add_argument(
        "--expert-manifold-video-sampling",
        choices=("with_replacement", "without_replacement"),
        default="without_replacement",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    _add_prepare_arguments(prepare)
    run = commands.add_parser("run")
    _add_prepare_arguments(run)
    profile_writer = commands.add_parser("profile-writer-generation")
    _add_prepare_arguments(profile_writer)
    profile_writer.add_argument(
        "--profile-batch-sizes",
        default="8,16,32",
        help="Ascending comma-separated actual Writer forward batches.",
    )
    profile_writer.add_argument(
        "--profile-warmup-runs",
        type=_positive_int,
        default=1,
    )
    profile_writer.add_argument(
        "--profile-measured-runs",
        type=_positive_int,
        default=2,
    )
    profile_worker = commands.add_parser("profile-writer-worker")
    profile_worker.add_argument("--output-dir", type=Path, required=True)
    profile_worker.add_argument("--worker-id", required=True)
    profile_worker.add_argument("--profile-batch-sizes", required=True)
    profile_worker.add_argument(
        "--profile-warmup-runs",
        type=_positive_int,
        required=True,
    )
    profile_worker.add_argument(
        "--profile-measured-runs",
        type=_positive_int,
        required=True,
    )
    start = commands.add_parser("start")
    start.add_argument("--output-dir", type=Path, required=True)
    resume = commands.add_parser("resume")
    resume.add_argument("--output-dir", type=Path, required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--output-dir", type=Path, required=True)
    worker.add_argument("--worker-id", required=True)
    worker.add_argument("--writer-generator", action="store_true")
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--output-dir", type=Path, required=True)
    checkpoint_curve = commands.add_parser("checkpoint-curve")
    checkpoint_curve.add_argument("--root", type=Path, action="append", required=True)
    checkpoint_curve.add_argument("--output", type=Path, required=True)
    historical_transition = commands.add_parser("historical-baseline-transition")
    historical_transition.add_argument("--legacy-root", type=Path, required=True)
    historical_transition.add_argument("--current-root", type=Path, required=True)
    historical_transition.add_argument("--output", type=Path, required=True)
    six_arm_audit = commands.add_parser("six-arm-audit")
    six_arm_audit.add_argument("--root", type=Path, action="append", required=True)
    six_arm_audit.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def prepare_run(
    args: argparse.Namespace,
    *,
    create_evaluation_queue: bool = True,
) -> dict[str, Any]:
    summary = prepare_evaluation_run(
        args,
        repo_root=REPO_ROOT,
        command=sys.argv,
        create_evaluation_queue=create_evaluation_queue,
    )
    print(json.dumps(summary, sort_keys=True))
    return summary


def _profile_batch_sizes(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise Pi05EvaluationError(
            "Writer profile batch sizes must be comma-separated integers"
        ) from error
    if (
        result != tuple(sorted(set(result)))
        or len(result) < 3
        or result[0] < 8
        or not {8, 16, 32}.issubset(result)
        or any(item <= 0 for item in result)
    ):
        raise Pi05EvaluationError("Writer profile batch sizes are invalid")
    return result


def _profile_worker_launch(
    *,
    output_dir: Path,
    contract: Mapping[str, Any],
    physical_gpu: int,
    batch_sizes: Sequence[int],
    warmup_runs: int,
    measured_runs: int,
) -> tuple[list[str], dict[str, str]]:
    replicas = int(contract["parallel"]["replicas_per_gpu"])
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(REPO_ROOT / "src"),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=str(physical_gpu),
        OMP_NUM_THREADS=str(
            contract["parallel"]["omp_threads_per_worker"][str(replicas)]
        ),
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "profile-writer-worker",
        "--output-dir",
        str(output_dir.resolve()),
        "--worker-id",
        f"{physical_gpu}-r0",
        "--profile-batch-sizes",
        ",".join(str(value) for value in batch_sizes),
        "--profile-warmup-runs",
        str(warmup_runs),
        "--profile-measured-runs",
        str(measured_runs),
    ]
    return command, environment


def _preflight_gpu_identity(
    preflight: Mapping[str, Any]
) -> tuple[tuple[str, ...], ...]:
    identities = []
    for row in preflight.get("gpus", ()):
        fields = tuple(value.strip() for value in str(row).split(",")[:3])
        if len(fields) != 3:
            raise Pi05EvaluationError("Writer profile GPU identity is invalid")
        identities.append(fields)
    return tuple(identities)


def profile_writer_worker_run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    contract = load_run_contract(output_dir / "run_contract.json")
    contract_git = contract.get("git", {})
    live_git = git_state(REPO_ROOT)
    physical = tuple(int(value) for value in contract["parallel"]["physical_gpu_ids"])
    launcher_preflight, _ = read_json_with_size(output_dir / WRITER_PROFILE_PREFLIGHT)
    live_preflight = _gpu_preflight(physical)
    sizes = _profile_batch_sizes(args.profile_batch_sizes)
    if (
        len(physical) != 1
        or not git_state_is_clean_pushed_or_frozen_authority(live_git)
        or live_git.get("commit") != contract_git.get("commit")
        or not git_state_is_clean_pushed_or_frozen_authority(contract_git)
        or args.worker_id != f"{physical[0]}-r0"
        or launcher_preflight.get("physical_gpu_ids") != list(physical)
        or launcher_preflight.get("compute_applications") != []
        or launcher_preflight.get("device_names") != ["NVIDIA A40"]
        or live_preflight.get("physical_gpu_ids") != list(physical)
        or live_preflight.get("compute_applications") != []
        or live_preflight.get("device_names") != ["NVIDIA A40"]
        or _preflight_gpu_identity(live_preflight)
        != _preflight_gpu_identity(launcher_preflight)
    ):
        raise Pi05EvaluationError("Writer profile worker preflight changed")

    from ember.pi05_evaluation import _initialize_worker
    from ember.writer.evaluation_runtime import profile_writer_generation

    runtime = _initialize_worker(
        output_dir,
        args.worker_id,
        writer_generation=True,
    )
    try:
        result = profile_writer_generation(
            runtime,
            batch_sizes=sizes,
            warmup_runs=int(args.profile_warmup_runs),
            measured_runs=int(args.profile_measured_runs),
            preflight=live_preflight,
        )
    finally:
        runtime.pool.close()
    print(
        json.dumps(
            {
                "event": "writer_generation_profile_worker_complete",
                "root": result["root"],
                "physical_gpu": result["physical_gpu"],
                "selected_writer_model_batch_size": result[
                    "selected_writer_model_batch_size"
                ],
            },
            sort_keys=True,
        )
    )
    return result


def profile_writer_run(args: argparse.Namespace) -> dict[str, Any]:
    sizes = _profile_batch_sizes(args.profile_batch_sizes)
    writer_kind, source_sft_requested = _adapter_requests(args)
    physical_args = _parse_gpu_indices(args.gpu_indices)
    state = git_state(REPO_ROOT)
    if (
        args.mode != "smoke"
        or args.role != "validation"
        or args.state_count < (sizes[-1] + 7) // 8
        or args.replicas_per_gpu != 1
        or args.writer_generators_per_gpu != 1
        or writer_kind != "expert_manifold_writer"
        or source_sft_requested
        or args.expert_manifold_video_condition != "correct"
        or args.expert_manifold_video_sampling != "without_replacement"
        or physical_args is None
        or len(physical_args) != 1
        or int(args.profile_warmup_runs) < 1
        or int(args.profile_measured_runs) < 2
        or not git_state_is_clean_pushed_or_frozen_authority(state)
    ):
        raise Pi05EvaluationError(
            "Writer generation profile requires the clean pushed validation/correct "
            "single-A40 smoke contract"
        )
    args.writer_generation_batch_size = sizes[-1]
    prepare_run(args, create_evaluation_queue=False)
    contract = load_run_contract(args.output_dir.resolve() / "run_contract.json")
    physical = tuple(int(value) for value in contract["parallel"]["physical_gpu_ids"])
    if len(physical) != 1:
        raise Pi05EvaluationError("Writer generation profile requires one physical GPU")
    preflight = _gpu_preflight(physical)
    if preflight.get("compute_applications") != []:
        raise Pi05EvaluationError(
            "Writer generation profile requires an idle physical GPU"
        )
    if preflight.get("device_names") != ["NVIDIA A40"]:
        raise Pi05EvaluationError(
            "Writer generation profile requires one physical NVIDIA A40"
        )
    publish_json_exclusive(
        args.output_dir.resolve() / WRITER_PROFILE_PREFLIGHT,
        preflight,
    )
    command, environment = _profile_worker_launch(
        output_dir=args.output_dir,
        contract=contract,
        physical_gpu=physical[0],
        batch_sizes=sizes,
        warmup_runs=int(args.profile_warmup_runs),
        measured_runs=int(args.profile_measured_runs),
    )
    worker_log = args.output_dir.resolve() / WRITER_PROFILE_WORKER_LOG
    with worker_log.open("ab") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise Pi05EvaluationError(
            "Writer generation profile worker failed with return code "
            f"{completed.returncode}; inspect {worker_log}"
        )
    result, _ = read_json_with_size(
        args.output_dir.resolve() / "writer_generation_profile.json"
    )
    if (
        result.get("root") != str(args.output_dir.resolve())
        or int(result.get("physical_gpu", -1)) != physical[0]
        or result.get("preflight", {}).get("physical_gpu_ids") != list(physical)
        or result.get("preflight", {}).get("compute_applications") != []
        or result.get("preflight", {}).get("device_names") != ["NVIDIA A40"]
        or _preflight_gpu_identity(result.get("preflight", {}))
        != _preflight_gpu_identity(preflight)
    ):
        raise Pi05EvaluationError("Writer generation profile worker result changed")
    print(
        json.dumps(
            {
                "event": "writer_generation_profile_complete",
                "root": result["root"],
                "selected_writer_model_batch_size": result[
                    "selected_writer_model_batch_size"
                ],
                "writer_generation_measurements": result[
                    "writer_generation_measurements"
                ],
            },
            sort_keys=True,
        )
    )
    return result


def _active_worker_pids(output_dir: Path) -> list[int]:
    needle = str(output_dir.resolve()).encode()
    active = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = path.read_bytes()
        except OSError:
            continue
        if (
            b"evaluate_pi05.py" in command
            and b"worker" in command
            and needle in command
        ):
            active.append(int(path.parent.name))
    return sorted(active)


def _validate_resume_inputs(contract: dict[str, Any]) -> None:
    config_path = Path(contract["authorities"]["config_path"])
    authorities = load_evaluation_authorities(config_path, REPO_ROOT)
    current_git = git_state(REPO_ROOT)
    if (
        current_git["commit"] != contract["git"]["commit"]
        or contract["mode"] != "smoke"
        and current_git["dirty_paths"]
    ):
        raise Pi05EvaluationError(
            "evaluator checkout differs from the sealed run commit"
        )
    expected_role_authority = None
    if contract.get("role") == "seen_panel":
        expected_role_authority = {
            "path": str(REPO_ROOT / SEEN_PANEL_RELATIVE_PATH),
            "bytes": (REPO_ROOT / SEEN_PANEL_RELATIVE_PATH).stat().st_size,
            "schema_version": authorities.seen_panel.get("schema_version"),
        }
    if contract.get("role_authority") != expected_role_authority:
        raise Pi05EvaluationError("evaluation role authority changed after prepare")
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
    if not normalization.is_file() or normalization.stat().st_size != int(
        contract["normalization"]["bytes"]
    ):
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
        elif adapter.get("kind") == "task_local_expert_bank":
            observed = _inspect_task_expert_adapter(
                config_path=Path(adapter["config"]["path"]),
                bank_root=Path(adapter["bank_root"]),
                step=int(adapter["step"]),
                source=model,
                tasks=tasks,
                evaluation_role=str(contract["role"]),
                require_formal=contract["mode"] != "smoke",
            )
        elif adapter.get("kind") == "expert_manifold_writer":
            observed = _inspect_expert_manifold_writer_adapter(
                config_path=Path(adapter["config"]["path"]),
                checkpoint=Path(adapter["writer_asset"]["checkpoint"]),
                video_data_root=Path(adapter["video_data"]["root"]),
                source=model,
                tasks=tasks,
                video_condition=str(adapter["video_condition"]),
                video_seed=int(adapter["video_schedule"]["seed"]),
                video_sampling_mode=str(adapter["video_schedule"]["sampling_mode"]),
                require_formal=contract["mode"] != "smoke",
            )
        else:
            raise Pi05EvaluationError("evaluation adapter kind changed after prepare")
        if observed != adapter:
            raise Pi05EvaluationError("evaluation adapter assets changed after prepare")


def _worker_ids(
    replicas_per_gpu: int, physical_gpu_ids: Sequence[int]
) -> tuple[str, ...]:
    values = tuple(
        f"{gpu}-r{replica}"
        for gpu in physical_gpu_ids
        for replica in range(replicas_per_gpu)
    )
    validate_worker_layout(
        values,
        replicas_per_gpu,
        physical_gpu_ids=physical_gpu_ids,
    )
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
    summary_path = output_dir / "run_summary.json"
    if summary_path.exists():
        observed, _ = read_json_with_size(summary_path)
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
        raise Pi05EvaluationError(
            f"PI05 evaluation run is not prepared: {output_dir}"
        ) from error
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Pi05EvaluationError(
                "another PI05 evaluator launcher owns this run"
            ) from error
        return _start_workers_locked(output_dir, resume=resume)


def _recover_locked_queue(
    output_dir: Path, *, resume: bool
) -> tuple[dict[str, Any], tuple[Any, ...], bool]:
    active = _active_worker_pids(output_dir)
    if active:
        raise Pi05EvaluationError(
            f"PI05 evaluator workers are already active: {active}"
        )
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
        raise Pi05EvaluationError(
            "PI05 evaluation already has unowned aggregate results"
        )
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
        worker_pids={
            worker_id: process.pid for worker_id, process in processes.items()
        },
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
        "contract_reference": contract["contract_reference"],
        "invocation_id": invocation_id,
        "started_unix": started_unix,
        "finished_unix": finished_unix,
        "wall_seconds": finished_unix - started_unix,
        "worker_ids": list(worker_ids),
        "worker_pids": {
            worker_id: process.pid for worker_id, process in processes.items()
        },
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
    physical_gpu_ids = tuple(
        int(value)
        for value in contract["parallel"].get(
            "physical_gpu_ids",
            range(int(contract["parallel"]["physical_gpu_count"])),
        )
    )
    preflight = _gpu_preflight(physical_gpu_ids)
    if preflight.get("compute_applications") != []:
        raise Pi05EvaluationError(
            "PI05 evaluation requires every selected physical GPU to be idle"
        )
    if preflight.get("device_names") != ["NVIDIA A40"] * len(physical_gpu_ids):
        raise Pi05EvaluationError(
            "PI05 evaluation requires every selected physical GPU to be an NVIDIA A40"
        )
    projected_new_bytes = 0
    if contract.get("writer_lora_cache") is not None:
        from ember.writer.evaluation_cache import writer_cache_manifest_is_ready

        if not writer_cache_manifest_is_ready(contract):
            projected_new_bytes = int(
                contract["writer_lora_cache"]["estimated_peak_new_bytes"]
            )
    preflight["projected_new_bytes"] = projected_new_bytes
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


def main() -> int:
    args = parse_args()
    if args.command == "profile-writer-generation":
        profile_writer_run(args)
    elif args.command == "profile-writer-worker":
        profile_writer_worker_run(args)
    elif args.command in {"prepare", "run"}:
        prepare_run(args)
        if args.command == "run":
            start_workers(args.output_dir, resume=False)
    elif args.command == "start":
        start_workers(args.output_dir, resume=False)
    elif args.command == "resume":
        start_workers(args.output_dir, resume=True)
    elif args.command == "worker":
        from ember.pi05_evaluation import run_worker

        print(
            json.dumps(
                run_worker(
                    output_dir=args.output_dir,
                    worker_id=args.worker_id,
                    writer_generator=args.writer_generator,
                )
            )
        )
    elif args.command == "checkpoint-curve":
        from ember.pi05_eval.analysis import analyze_checkpoint_curve

        print(
            json.dumps(analyze_checkpoint_curve(args.root, args.output), sort_keys=True)
        )
    elif args.command == "historical-baseline-transition":
        from ember.pi05_eval.analysis import analyze_historical_baseline_transition

        print(
            json.dumps(
                analyze_historical_baseline_transition(
                    args.legacy_root,
                    args.current_root,
                    args.output,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "six-arm-audit":
        from ember.pi05_eval.analysis import audit_six_arms

        print(json.dumps(audit_six_arms(args.root, args.output), sort_keys=True))
    else:
        _finalize_aggregate(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
