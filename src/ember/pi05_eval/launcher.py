"""Live preflight and staged worker-process ownership for PI05 evaluation."""

from __future__ import annotations

import os
import pwd
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ember.pi05_assets import Pi05EvaluationError
from ember.eval_adapters import ARCHIVAL_WRITER_CACHE_KIND, WRITER_ADAPTER_KINDS


MAX_COSCHEDULED_GPU_UTILIZATION_PERCENT = 10
MAX_COSCHEDULED_GPU_MEMORY_USED_MIB = 8 * 1024
MIN_EVALUATOR_GPU_FREE_MEMORY_MIB = 32 * 1024


def _storage_root() -> Path:
    """Return the host-local root used for the personal storage cap."""

    configured = os.environ.get("EMBER_STORAGE_ROOT")
    if not configured:
        raise Pi05EvaluationError("EMBER_STORAGE_ROOT must be set")
    return Path(configured).expanduser().resolve()


def gpu_preflight(physical_gpu_ids: Sequence[int]) -> dict[str, Any]:
    """Record storage, CUDA runtime, GPU telemetry, and co-scheduled processes."""

    import torch

    selected_indices = tuple(int(value) for value in physical_gpu_ids)
    if (
        not selected_indices
        or len(set(selected_indices)) != len(selected_indices)
        or any(value < 0 for value in selected_indices)
    ):
        raise Pi05EvaluationError(
            "PI05 evaluation preflight GPU selection is invalid"
        )
    nvidia_selection = ",".join(str(value) for value in selected_indices)
    storage_root = _storage_root()
    data_capacity = subprocess.run(
        [
            "df",
            "-B1",
            "--output=size,used,avail,pcent,target",
            str(storage_root),
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()[-1].split()
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            nvidia_selection,
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,"
            "temperature.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    gpu_by_index: dict[int, str] = {}
    uuid_by_index: dict[int, str] = {}
    name_by_index: dict[int, str] = {}
    telemetry_by_index: dict[int, dict[str, int | str]] = {}
    for row in gpu_query:
        fields = [value.strip() for value in row.split(",")]
        if len(fields) != 8 or not fields[0].isdigit():
            raise Pi05EvaluationError(f"invalid nvidia-smi GPU row: {row}")
        index = int(fields[0])
        gpu_by_index[index] = row
        uuid_by_index[index] = fields[1]
        name_by_index[index] = fields[2]
        telemetry_by_index[index] = {
            "physical_gpu": index,
            "uuid": fields[1],
            "memory_used_mib": int(fields[3]),
            "memory_total_mib": int(fields[4]),
            "utilization_percent": int(fields[5]),
        }
    missing = sorted(set(physical_gpu_ids) - set(gpu_by_index))
    if missing:
        raise Pi05EvaluationError(
            f"PI05 evaluation physical GPUs are unavailable: {missing}"
        )
    applications = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            nvidia_selection,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    selected_uuids = {uuid_by_index[index] for index in physical_gpu_ids}
    selected = [
        row
        for row in applications
        if row.split(",", 1)[0].strip() in selected_uuids
    ]
    owned_applications = []
    for application in selected:
        fields = [value.strip() for value in application.split(",")]
        owner = "unknown"
        if len(fields) >= 2 and fields[1].isdigit():
            try:
                status = Path(f"/proc/{fields[1]}/status").read_text(encoding="utf-8")
                uid_line = next(
                    line for line in status.splitlines() if line.startswith("Uid:")
                )
                owner = pwd.getpwuid(int(uid_line.split()[1])).pw_name
            except (OSError, KeyError, StopIteration, ValueError):
                pass
        owned_applications.append(f"{application}, owner={owner}")
    return {
        "unix": time.time(),
        "physical_gpu_ids": list(physical_gpu_ids),
        "gpus": [gpu_by_index[index] for index in physical_gpu_ids],
        "gpu_telemetry": [
            telemetry_by_index[index] for index in physical_gpu_ids
        ],
        "device_names": [name_by_index[index] for index in physical_gpu_ids],
        "compute_applications": owned_applications,
        "gpu_admission_policy": {
            "max_utilization_percent": MAX_COSCHEDULED_GPU_UTILIZATION_PERCENT,
            "max_memory_used_mib": MAX_COSCHEDULED_GPU_MEMORY_USED_MIB,
            "min_free_memory_mib": MIN_EVALUATOR_GPU_FREE_MEMORY_MIB,
        },
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "storage_root": str(storage_root),
        "storage_accounting": "filesystem_capacity_only_no_recursive_personal_scan",
        "data_filesystem": {
            "size": int(data_capacity[0]),
            "used": int(data_capacity[1]),
            "available": int(data_capacity[2]),
            "percent": data_capacity[3],
            "mount": data_capacity[4],
        },
    }


def evaluator_gpus_are_eligible(preflight: Mapping[str, Any]) -> bool:
    telemetry = preflight.get("gpu_telemetry", ())
    expected = preflight.get("physical_gpu_ids", ())
    return len(telemetry) == len(expected) and all(
        int(row["utilization_percent"])
        <= MAX_COSCHEDULED_GPU_UTILIZATION_PERCENT
        and int(row["memory_used_mib"])
        <= MAX_COSCHEDULED_GPU_MEMORY_USED_MIB
        and int(row["memory_total_mib"]) - int(row["memory_used_mib"])
        >= MIN_EVALUATOR_GPU_FREE_MEMORY_MIB
        for row in telemetry
    )


def terminate_owned_workers(
    processes: Mapping[str, subprocess.Popen[bytes]],
) -> None:
    """Stop only subprocesses created by the current launcher invocation."""

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


def _writer_generator_ids(
    contract: Mapping[str, Any],
    physical_gpu_ids: Sequence[int],
) -> tuple[str, ...]:
    generators = int(contract["parallel"].get("writer_generators_per_gpu", 0))
    return tuple(
        f"{gpu}-r{replica}"
        for gpu in physical_gpu_ids
        for replica in range(generators)
    )


def _spawn_one(
    *,
    stack: ExitStack,
    processes: dict[str, subprocess.Popen[bytes]],
    output_dir: Path,
    contract: Mapping[str, Any],
    worker_id: str,
    writer_generator: bool,
    invocation_id: str,
    repo_root: Path,
    script_path: Path,
) -> None:
    replicas = int(contract["parallel"]["replicas_per_gpu"])
    gpu = worker_id.split("-r", 1)[0]
    log = stack.enter_context(
        (output_dir / "worker_logs" / f"{worker_id}.log").open("ab")
    )
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(repo_root / "src"),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=gpu,
        OMP_NUM_THREADS=str(
            contract["parallel"]["omp_threads_per_worker"][str(replicas)]
        ),
        EMBER_PI05_EVAL_INVOCATION_ID=invocation_id,
    )
    command = [
        sys.executable,
        str(script_path),
        "worker",
        "--output-dir",
        str(output_dir),
        "--worker-id",
        worker_id,
    ]
    if writer_generator:
        command.append("--writer-generator")
    processes[worker_id] = subprocess.Popen(
        command,
        cwd=repo_root,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def _stage_writer_generators(
    contract: Mapping[str, Any],
    *,
    invocation_id: str,
    processes: Mapping[str, subprocess.Popen[bytes]],
    spawn: Callable[[str, bool], None],
) -> tuple[str, ...]:
    adapter = contract.get("adapter")
    if not (
        isinstance(adapter, Mapping)
        and adapter.get("kind") in WRITER_ADAPTER_KINDS
    ):
        return ()
    from ember.writer.evaluation_cache import (
        finalize_writer_cache,
        generator_marker_path,
        writer_cache_manifest_is_ready,
    )

    if writer_cache_manifest_is_ready(contract):
        return ()
    if adapter.get("kind") == ARCHIVAL_WRITER_CACHE_KIND:
        raise Pi05EvaluationError(
            "archival Writer projection must be imported and sealed before launch"
        )
    physical_gpu_ids = tuple(
        int(value) for value in contract["parallel"]["physical_gpu_ids"]
    )
    generator_ids = _writer_generator_ids(contract, physical_gpu_ids)
    for worker_id in generator_ids:
        spawn(worker_id, True)
    marker_paths = tuple(
        generator_marker_path(contract, invocation_id, worker_id)
        for worker_id in generator_ids
    )
    while not all(path.is_file() for path in marker_paths):
        exited = {
            worker_id: process.poll()
            for worker_id, process in processes.items()
            if process.poll() is not None
        }
        if exited:
            raise Pi05EvaluationError(
                f"Writer generator exited before sealing its cache: {exited}"
            )
        time.sleep(0.2)
    finalize_writer_cache(
        contract,
        invocation_id=invocation_id,
        worker_ids=generator_ids,
    )
    return generator_ids


def _wait_for_workers(
    processes: Mapping[str, subprocess.Popen[bytes]],
) -> dict[str, int]:
    while True:
        return_codes = {
            worker_id: code
            for worker_id, process in processes.items()
            if (code := process.poll()) is not None
        }
        if any(code != 0 for code in return_codes.values()):
            terminate_owned_workers(processes)
            break
        if len(return_codes) == len(processes):
            break
        time.sleep(0.2)
    return {
        worker_id: int(process.wait())
        for worker_id, process in processes.items()
    }


def spawn_worker_processes(
    output_dir: Path,
    contract: Mapping[str, Any],
    worker_ids: Sequence[str],
    *,
    invocation_id: str,
    repo_root: Path,
    script_path: Path,
) -> tuple[
    dict[str, subprocess.Popen[bytes]],
    dict[str, int],
    BaseException | None,
]:
    """Stage Writer generators first, then scale out rollout-only workers."""

    processes: dict[str, subprocess.Popen[bytes]] = {}
    return_codes: dict[str, int] = {}
    launch_error: BaseException | None = None
    (output_dir / "worker_logs").mkdir(parents=True, exist_ok=True)
    try:
        with ExitStack() as stack:
            def spawn(worker_id: str, writer_generator: bool) -> None:
                _spawn_one(
                    stack=stack,
                    processes=processes,
                    output_dir=output_dir,
                    contract=contract,
                    worker_id=worker_id,
                    writer_generator=writer_generator,
                    invocation_id=invocation_id,
                    repo_root=repo_root,
                    script_path=script_path,
                )

            generator_ids = _stage_writer_generators(
                contract,
                invocation_id=invocation_id,
                processes=processes,
                spawn=spawn,
            )
            for worker_id in worker_ids:
                if worker_id not in generator_ids:
                    spawn(worker_id, False)
            return_codes = _wait_for_workers(processes)
    except BaseException as error:
        launch_error = error
        terminate_owned_workers(processes)
        return_codes = {
            worker_id: int(process.returncode)
            for worker_id, process in processes.items()
            if process.returncode is not None
        }
    return processes, return_codes, launch_error
