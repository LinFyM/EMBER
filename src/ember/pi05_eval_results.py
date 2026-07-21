"""Strict aggregation and worker-topology evidence for PI05 evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import load_run_contract
from ember.pi05_eval_queue import (
    EvaluationShard,
    completed_jobs,
    publish_json_exclusive,
    queue_summary,
    read_json_with_sha256,
)
from ember.pi05_evaluation import task_lookup, validate_shard_result


AGGREGATE_SCHEMA = "ember_pi05_target_eval_results_v1"


def _expected_worker_ids(contract: Mapping[str, Any]) -> tuple[str, ...]:
    replicas = int(contract["parallel"]["replicas_per_gpu"])
    return tuple(
        f"{gpu}-r{replica}"
        for gpu in range(8)
        for replica in range(replicas)
    )


def _worker_lifecycle(
    output_dir: Path,
    contract: Mapping[str, Any],
    *,
    invocation_id: str,
    worker_id: str,
) -> dict[str, Any]:
    path = output_dir / "workers" / f"{worker_id}.jsonl"
    try:
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise Pi05EvaluationError(f"invalid worker lifecycle evidence: {worker_id}") from error
    selected = [row for row in events if row.get("invocation_id") == invocation_id]
    by_event = {row.get("event"): row for row in selected}
    if (
        len(selected) != 3
        or set(by_event) != {"process_started", "ready", "finished"}
        or any(
            row.get("worker_id") != worker_id
            or row.get("contract_sha256") != contract["contract_sha256"]
            for row in selected
        )
    ):
        raise Pi05EvaluationError(f"worker lifecycle is incomplete: {worker_id}")
    process = by_event["process_started"]
    ready = by_event["ready"]
    finished = by_event["finished"]
    gpu_text, replica_text = worker_id.split("-r", 1)
    physical_gpu = int(gpu_text)
    expected_numa = 0 if physical_gpu < 4 else 1
    if (
        int(ready.get("physical_gpu", -1)) != physical_gpu
        or int(ready.get("replica", -1)) != int(replica_text)
        or not ready.get("gpu_uuid")
        or ready.get("numa_node") != expected_numa
        or not ready.get("cpu_affinity")
        or not float(process["unix"]) <= float(ready["unix"]) <= float(finished["unix"])
    ):
        raise Pi05EvaluationError(f"worker topology evidence changed: {worker_id}")
    return {
        "worker_id": worker_id,
        "pid": int(process["pid"]),
        "physical_gpu": physical_gpu,
        "gpu_uuid": ready["gpu_uuid"],
        "replica": int(ready["replica"]),
        "numa_node": int(ready["numa_node"]),
        "cpu_affinity": ready["cpu_affinity"],
        "process_started_unix": float(process["unix"]),
        "ready_unix": float(ready["unix"]),
        "finished_unix": float(finished["unix"]),
        "model_load_seconds": float(ready["model_load_seconds"]),
        "completed_shards": int(finished["completed_shards"]),
        "adopted_shards": int(finished["adopted_shards"]),
    }


def _validated_worker_lifecycles(
    output_dir: Path,
    contract: Mapping[str, Any],
    launcher: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected_ids = _expected_worker_ids(contract)
    if tuple(launcher.get("worker_ids", ())) != expected_ids or launcher.get(
        "return_codes"
    ) != {worker_id: 0 for worker_id in expected_ids}:
        raise Pi05EvaluationError("launcher did not complete the sealed worker topology")
    invocation_id = str(launcher.get("invocation_id", ""))
    lifecycles = [
        _worker_lifecycle(
            output_dir,
            contract,
            invocation_id=invocation_id,
            worker_id=worker_id,
        )
        for worker_id in expected_ids
    ]
    if len({row["pid"] for row in lifecycles}) != len(lifecycles):
        raise Pi05EvaluationError("worker lifecycle reused a process across CUDA roles")
    gpu_uuids = {
        gpu: {row["gpu_uuid"] for row in lifecycles if row["physical_gpu"] == gpu}
        for gpu in range(8)
    }
    if any(len(values) != 1 for values in gpu_uuids.values()) or len(
        {next(iter(values)) for values in gpu_uuids.values()}
    ) != 8:
        raise Pi05EvaluationError("worker lifecycle GPU UUID mapping is not eight-device symmetric")
    return lifecycles


def _load_shard_records(
    output_dir: Path,
    contract: Mapping[str, Any],
    jobs: tuple[dict[str, Any], ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float], list[float]]:
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    starts: list[float] = []
    finishes: list[float] = []
    for job in jobs:
        shard_value = json.loads(job["payload"])
        shard_value["init_state_ids"] = tuple(shard_value["init_state_ids"])
        shard = EvaluationShard(**shard_value)
        relative = Path(job["rows_path"])
        path = output_dir / relative
        try:
            path.resolve().relative_to(output_dir)
        except ValueError as error:
            raise Pi05EvaluationError("evaluation shard path escaped output root") from error
        payload, digest = read_json_with_sha256(path)
        if digest != job["rows_sha256"]:
            raise Pi05EvaluationError(f"raw evaluation shard hash changed: {shard.job_id}")
        shard_rows = validate_shard_result(payload, contract=contract, shard=shard)
        raw_successes = sum(bool(row["success"]) for row in shard_rows)
        if int(job["row_count"]) != len(shard_rows) or int(job["successes"]) != raw_successes:
            raise Pi05EvaluationError(f"queue summary differs from raw shard: {shard.job_id}")
        rows.extend(shard_rows)
        starts.append(float(payload["started_unix"]))
        finishes.append(float(payload["finished_unix"]))
        records.append(
            {
                "job_id": shard.job_id,
                "suite": shard.suite,
                "task_id": shard.task_id,
                "state_count": len(shard.init_state_ids),
                "rows_path": relative.as_posix(),
                "rows_sha256": digest,
                "producer_worker_id": payload["producer"]["worker_id"],
                "completing_worker_id": job["worker_id"],
                "attempt": int(job["attempt"]),
                "successes": raw_successes,
            }
        )
    return rows, records, starts, finishes


def _per_task_rows(
    rows: list[dict[str, Any]],
    tasks: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    values = []
    for key in sorted(tasks):
        selected = [row for row in rows if (row["suite"], int(row["task_id"])) == key]
        successes = sum(bool(row["success"]) for row in selected)
        values.append(
            {
                "suite": key[0],
                "task_id": key[1],
                "split_role": tasks[key]["split_role"],
                "language": tasks[key]["language"],
                "successes": successes,
                "episodes": len(selected),
                "success_rate": successes / len(selected),
            }
        )
    return values


def aggregate_run(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    contract = load_run_contract(output_dir / "run_contract.json")
    queue_path = output_dir / "queue.sqlite3"
    summary = queue_summary(queue_path)
    jobs = completed_jobs(queue_path)
    if summary["status_counts"] != {"complete": len(jobs)}:
        raise Pi05EvaluationError(f"PI05 evaluation queue is incomplete: {summary}")
    tasks = task_lookup(contract)
    rows, shard_records, starts, finishes = _load_shard_records(
        output_dir, contract, jobs
    )
    expected = {
        (task["suite"], int(task["task_id"]), int(state_id))
        for task in contract["tasks"]
        for state_id in task["init_state_ids"]
    }
    actual = {
        (row["suite"], int(row["task_id"]), int(row["init_state_id"])) for row in rows
    }
    if actual != expected or len(rows) != len(expected):
        raise Pi05EvaluationError("PI05 aggregate does not cover every task/state exactly once")
    shard_window_seconds = max(finishes) - min(starts)
    launcher: dict[str, Any] | None = None
    workers: list[dict[str, Any]] = []
    if contract.get("parallel", {}).get("worker_count"):
        launcher, _ = read_json_with_sha256(output_dir / "launcher_completion.json")
        if (
            launcher.get("schema_version") != "ember_pi05_eval_launcher_completion_v1"
            or launcher.get("contract_sha256") != contract["contract_sha256"]
        ):
            raise Pi05EvaluationError("launcher completion evidence changed")
        workers = _validated_worker_lifecycles(output_dir, contract, launcher)
        evaluation_seconds = float(launcher["wall_seconds"])
        if (
            evaluation_seconds <= 0
            or float(launcher.get("finished_unix", 0))
            - float(launcher.get("started_unix", 0))
            != evaluation_seconds
            or sum(row["completed_shards"] for row in workers) != len(jobs)
        ):
            raise Pi05EvaluationError("launcher timing or completed-shard evidence changed")
    else:
        evaluation_seconds = shard_window_seconds
    successes = sum(bool(row["success"]) for row in rows)
    result = {
        "schema_version": AGGREGATE_SCHEMA,
        "contract_sha256": contract["contract_sha256"],
        "arm": contract["arm"],
        "role": contract["role"],
        "mode": contract["mode"],
        "model": contract["model"],
        "normalization": contract["normalization"],
        "tokenizer": contract["tokenizer"],
        "overall": {
            "successes": successes,
            "episodes": len(rows),
            "success_rate": successes / len(rows),
            "evaluation_wall_seconds": evaluation_seconds,
            "shard_execution_window_seconds": shard_window_seconds,
            "effective_rollouts_per_second": len(rows) / evaluation_seconds,
            "episodes_per_hour": len(rows) * 3600.0 / evaluation_seconds,
        },
        "per_task": _per_task_rows(rows, tasks),
        "launcher": launcher,
        "workers": workers,
        "shards": shard_records,
        "rows": sorted(
            rows,
            key=lambda row: (
                row["suite"],
                int(row["task_id"]),
                int(row["init_state_id"]),
            ),
        ),
    }
    path = output_dir / "results.json"
    if path.exists():
        observed, _ = read_json_with_sha256(path)
        if observed != result:
            raise Pi05EvaluationError("existing aggregate differs from immutable raw shards")
    else:
        publish_json_exclusive(path, result)
    return result
