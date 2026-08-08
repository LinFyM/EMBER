"""Strict aggregation and worker-topology evidence for PI05 evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import load_run_contract
from ember.pi05_eval.launcher_evidence import launcher_attempt_summary
from ember.pi05_eval_queue import (
    EvaluationShard,
    completed_jobs,
    publish_json_exclusive,
    queue_summary,
    read_json_with_size,
)
from ember.pi05_evaluation import task_lookup, validate_shard_result


AGGREGATE_SCHEMA = "ember_pi05_target_eval_results_v2"


def _expected_worker_ids(contract: Mapping[str, Any]) -> tuple[str, ...]:
    replicas = int(contract["parallel"]["replicas_per_gpu"])
    physical_gpu_ids = contract["parallel"].get(
        "physical_gpu_ids",
        range(int(contract["parallel"].get("physical_gpu_count", 8))),
    )
    return tuple(
        f"{gpu}-r{replica}"
        for gpu in physical_gpu_ids
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
    ready = by_event.get("ready", {})
    writer_generator = bool(ready.get("writer_generator", False))
    expected_events = {"process_started", "ready", "finished"}
    if writer_generator:
        expected_events.update(
            {"writer_generation_finished", "rollout_ready_with_retained_policy"}
        )
    if (
        len(selected) != len(expected_events)
        or set(by_event) != expected_events
        or any(
            row.get("worker_id") != worker_id
            or row.get("contract_reference") != contract["contract_reference"]
            for row in selected
        )
    ):
        raise Pi05EvaluationError(f"worker lifecycle is incomplete: {worker_id}")
    process = by_event["process_started"]
    finished = by_event["finished"]
    generation = by_event.get("writer_generation_finished")
    rollout_ready = by_event.get("rollout_ready_with_retained_policy", ready)
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
        or not float(ready["unix"])
        <= float(rollout_ready["unix"])
        <= float(finished["unix"])
        or (
            generation is not None
            and not float(ready["unix"])
            <= float(generation["unix"])
            <= float(rollout_ready["unix"])
        )
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
        "rollout_ready_unix": float(rollout_ready["unix"]),
        "finished_unix": float(finished["unix"]),
        "model_load_seconds": float(ready["model_load_seconds"]),
        "writer_generator": writer_generator,
        "writer_generation": (
            {
                key: generation[key]
                for key in (
                    "assigned_entries",
                    "generated_entries",
                    "reused_entries",
                    "generated_batches",
                    "generation_batch_size",
                    "generation_wall_seconds",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                    "post_release_allocated_bytes",
                    "post_release_reserved_bytes",
                    "source_policy_reused_for_rollout",
                    "writer_modules_released",
                )
            }
            if generation is not None
            else None
        ),
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
    physical_gpu_ids = tuple(
        int(value)
        for value in contract["parallel"].get(
            "physical_gpu_ids",
            range(int(contract["parallel"].get("physical_gpu_count", 8))),
        )
    )
    gpu_uuids = {
        gpu: {row["gpu_uuid"] for row in lifecycles if row["physical_gpu"] == gpu}
        for gpu in physical_gpu_ids
    }
    physical_gpu_count = len(physical_gpu_ids)
    if any(len(values) != 1 for values in gpu_uuids.values()) or len(
        {next(iter(values)) for values in gpu_uuids.values()}
    ) != physical_gpu_count:
        raise Pi05EvaluationError("worker lifecycle GPU UUID mapping is not device symmetric")
    return lifecycles


def _writer_generation_summary(workers: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    generated = [
        row["writer_generation"]
        for row in workers
        if row.get("writer_generation") is not None
    ]
    if not generated:
        return None
    return {
        "generator_workers": len(generated),
        "assigned_entries": sum(int(row["assigned_entries"]) for row in generated),
        "generated_entries": sum(int(row["generated_entries"]) for row in generated),
        "reused_entries": sum(int(row["reused_entries"]) for row in generated),
        "generated_batches": sum(int(row["generated_batches"]) for row in generated),
        "generation_batch_size": sorted(
            {int(row["generation_batch_size"]) for row in generated}
        ),
        "max_worker_generation_wall_seconds": max(
            float(row["generation_wall_seconds"]) for row in generated
        ),
        "max_peak_allocated_bytes": max(
            int(row["peak_allocated_bytes"]) for row in generated
        ),
        "max_peak_reserved_bytes": max(
            int(row["peak_reserved_bytes"]) for row in generated
        ),
        "all_source_policy_processes_reused_for_rollout": all(
            row["source_policy_reused_for_rollout"] is True for row in generated
        ),
        "all_writer_modules_released": all(
            row["writer_modules_released"] is True for row in generated
        ),
    }


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
        payload, artifact_bytes = read_json_with_size(path)
        if artifact_bytes != int(job["rows_bytes"]):
            raise Pi05EvaluationError(f"raw evaluation shard size changed: {shard.job_id}")
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
                "rows_bytes": artifact_bytes,
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
        value = {
                "suite": key[0],
                "task_id": key[1],
                "split_role": tasks[key]["split_role"],
                "language": tasks[key]["language"],
                "successes": successes,
                "episodes": len(selected),
                "success_rate": successes / len(selected),
            }
        writer_rows = [row["writer"] for row in selected if row.get("writer") is not None]
        if writer_rows:
            demo_counts: dict[str, int] = {}
            demo_set_counts: dict[str, int] = {}
            for writer in writer_rows:
                demos = (
                    tuple(int(demo) for demo in writer["teacher_demo_indices"])
                    if "teacher_demo_indices" in writer
                    else (int(writer["teacher_demo_index"]),)
                )
                for demo in demos:
                    key = str(demo)
                    demo_counts[key] = demo_counts.get(key, 0) + 1
                set_key = ",".join(str(demo) for demo in demos)
                demo_set_counts[set_key] = demo_set_counts.get(set_key, 0) + 1
            value["writer"] = {
                "condition": writer_rows[0]["condition"],
                "unique_teacher_videos": len(demo_counts),
                "teacher_demo_counts": dict(sorted(demo_counts.items(), key=lambda item: int(item[0]))),
                "generation_wall_seconds": sum(
                    float(writer["writer_generation_seconds"]) for writer in writer_rows
                ),
            }
            if "teacher_demo_indices" in writer_rows[0]:
                value["writer"].update(
                    {
                        "videos_per_condition": len(writer_rows[0]["teacher_demo_indices"]),
                        "unique_teacher_video_sets": len(demo_set_counts),
                        "teacher_demo_set_counts": dict(sorted(demo_set_counts.items())),
                    }
                )
        values.append(value)
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
    launcher_attempts: dict[str, Any] | None = None
    workers: list[dict[str, Any]] = []
    if contract.get("parallel", {}).get("worker_count"):
        launcher, _ = read_json_with_size(output_dir / "launcher_completion.json")
        if (
            launcher.get("schema_version") != "ember_pi05_eval_launcher_completion_v1"
            or launcher.get("contract_reference") != contract["contract_reference"]
        ):
            raise Pi05EvaluationError("launcher completion evidence changed")
        workers = _validated_worker_lifecycles(output_dir, contract, launcher)
        launcher_attempts = launcher_attempt_summary(
            output_dir,
            contract,
            launcher,
            workers,
            total_shards=len(jobs),
        )
        evaluation_seconds = float(launcher_attempts["active_wall_seconds"])
        if (
            evaluation_seconds <= 0
            or float(launcher.get("finished_unix", 0))
            - float(launcher.get("started_unix", 0))
            != float(launcher["wall_seconds"])
        ):
            raise Pi05EvaluationError("launcher timing or completed-shard evidence changed")
    else:
        evaluation_seconds = shard_window_seconds
    successes = sum(bool(row["success"]) for row in rows)
    result = {
        "schema_version": AGGREGATE_SCHEMA,
        "contract_reference": contract["contract_reference"],
        "arm": contract["arm"],
        "paired_control": contract.get("paired_control"),
        "role": contract["role"],
        "mode": contract["mode"],
        "model": contract["model"],
        "adapter": contract.get("adapter"),
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
            "rollout_only_effective_rollouts_per_second": (
                len(rows) / shard_window_seconds
            ),
            "rollout_only_episodes_per_hour": (
                len(rows) * 3600.0 / shard_window_seconds
            ),
        },
        "writer_generation": _writer_generation_summary(workers),
        "per_task": _per_task_rows(rows, tasks),
        "launcher": launcher,
        "launcher_attempts": launcher_attempts,
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
        observed, _ = read_json_with_size(path)
        if observed != result:
            raise Pi05EvaluationError("existing aggregate differs from immutable raw shards")
    else:
        publish_json_exclusive(path, result)
    return result
