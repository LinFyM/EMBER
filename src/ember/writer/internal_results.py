"""Fail-closed distributed result sealing for AS-Writer internal analysis."""

from __future__ import annotations

import sys
import time
import traceback
from collections import defaultdict
from datetime import timedelta
from math import ceil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch.distributed as dist

from ember.pi05_source_checkpoint import canonical_hash, read_json, write_json_atomic
from ember.writer.internal_metrics import CONDITIONS, validate_finite_tree
from ember.writer.model import WriterModelError


CONTROL_TIMEOUT = timedelta(hours=3)
FAILURE_SCHEMA = "ember_as_writer_internal_analysis_failure_v1"


def lpt_assignment(
    cost_by_task: Mapping[int, int],
    *,
    world_size: int,
) -> dict[int, list[int]]:
    """Assign eight whole tasks across the explicit distributed topology."""

    if (
        len(cost_by_task) != 8
        or not 1 <= world_size <= len(cost_by_task)
        or any(value <= 0 for value in cost_by_task.values())
    ):
        raise WriterModelError("internal-analysis LPT contract changed")
    result = {rank: [] for rank in range(world_size)}
    loads = [0] * world_size
    maximum_tasks_per_rank = ceil(len(cost_by_task) / world_size)
    for task_id, cost in sorted(cost_by_task.items(), key=lambda item: (-item[1], item[0])):
        rank = min(
            (
                value
                for value in range(world_size)
                if len(result[value]) < maximum_tasks_per_rank
            ),
            key=lambda value: (loads[value], value),
        )
        result[rank].append(task_id)
        loads[rank] += cost
    if (
        any(not tasks for tasks in result.values())
        or any(len(tasks) > maximum_tasks_per_rank for tasks in result.values())
        or {task for tasks in result.values() for task in tasks} != set(cost_by_task)
    ):
        raise WriterModelError("internal-analysis LPT lost rank/task ownership")
    return result


def create_control_group(world_size: int) -> Any | None:
    return None if world_size == 1 else dist.new_group(backend="gloo", timeout=CONTROL_TIMEOUT)


def broadcast(value: Any, *, rank: int, world_size: int, group: Any | None) -> Any:
    payload = [value if rank == 0 else None]
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0, group=group)
    if isinstance(payload[0], Mapping) and "error" in payload[0]:
        raise WriterModelError(str(payload[0]["error"]))
    return payload[0]


def barrier(world_size: int, group: Any | None) -> None:
    if world_size > 1:
        dist.barrier(group=group)


def seal_rank_rows(
    output_dir: Path,
    *,
    rank: int,
    world_size: int,
    assigned_task_ids: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    group: Any | None,
) -> None:
    validate_finite_tree(rows, f"rank_{rank}")
    write_json_atomic(
        output_dir / f"rows_rank_{rank:02d}.json",
        {"rank": rank, "assigned_task_ids": list(assigned_task_ids), "rows": list(rows)},
    )
    barrier(world_size, group)


def record_failure(output_dir: Path, rank: int, error: BaseException) -> None:
    record = {
        "schema_version": FAILURE_SCHEMA,
        "rank": rank,
        "error": repr(error),
        "traceback": traceback.format_exc(),
    }
    print(record["traceback"], file=sys.stderr, flush=True)
    try:
        write_json_atomic(output_dir / f"failure_rank_{rank:02d}.json", record)
    except Exception as write_error:
        print(f"failed to persist internal-analysis error: {write_error!r}", file=sys.stderr)


def validate_rank_payloads(
    payloads: Sequence[Mapping[str, Any]],
    references_per_task: int,
    *,
    world_size: int,
) -> list[dict[str, Any]]:
    """Require the exact distributed 8-task Cartesian panel."""

    if (
        len(payloads) != world_size
        or not 1 <= world_size <= 8
        or not 1 <= references_per_task <= 50
    ):
        raise WriterModelError("internal-analysis rank payload count changed")
    rows: list[dict[str, Any]] = []
    owned: set[int] = set()
    maximum_tasks_per_rank = ceil(8 / world_size)
    for expected_rank, payload in enumerate(payloads):
        task_ids = [int(value) for value in payload.get("assigned_task_ids", [])]
        local = list(payload.get("rows", []))
        if (
            int(payload.get("rank", -1)) != expected_rank
            or not 1 <= len(task_ids) <= maximum_tasks_per_rank
            or len(set(task_ids)) != len(task_ids)
            or owned.intersection(task_ids)
            or len(local) != len(task_ids) * references_per_task
            or any(int(row.get("global_task_id", -1)) not in task_ids for row in local)
        ):
            raise WriterModelError("internal-analysis rank ownership changed")
        owned.update(task_ids)
        rows.extend(dict(row) for row in local)
    keys = {(int(row["global_task_id"]), int(row["reference_ordinal"])) for row in rows}
    if len(owned) != 8 or keys != {(task, ref) for task in owned for ref in range(references_per_task)}:
        raise WriterModelError("internal-analysis Cartesian coverage changed")
    for row in rows:
        conditions = [value["condition"] for value in row.get("conditions", [])]
        wall = row.get("information_wall", {})
        if tuple(conditions) != CONDITIONS or any(int(value) != 0 for value in wall.values()):
            raise WriterModelError("internal-analysis pairing or information wall changed")
    validate_finite_tree(rows)
    return sorted(rows, key=lambda row: (int(row["global_task_id"]), int(row["reference_ordinal"])))


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate_numeric(records: Sequence[Any]) -> Any:
    if not records:
        return None
    if all(isinstance(value, (int, float, np.number)) and not isinstance(value, bool) for value in records):
        return _numeric_summary([float(value) for value in records])
    if all(isinstance(value, Mapping) for value in records):
        keys = sorted(set.intersection(*(set(value) for value in records)))
        return {key: aggregate_numeric([value[key] for value in records]) for key in keys}
    return None


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["global_task_id"])].append(row)
    return {
        "rows": len(rows),
        "tasks": len(grouped),
        "global_numeric": aggregate_numeric(rows),
        "per_task_numeric": {
            str(task): aggregate_numeric(values) for task, values in sorted(grouped.items())
        },
    }


def finalize(
    output_dir: Path,
    *,
    rank: int,
    world_size: int,
    references_per_task: int,
    result_schema: str,
    started: float,
    group: Any | None,
) -> None:
    status: Any = None
    if rank == 0:
        try:
            payloads = [read_json(output_dir / f"rows_rank_{value:02d}.json") for value in range(world_size)]
            rows = validate_rank_payloads(
                payloads,
                references_per_task,
                world_size=world_size,
            )
            summary = summarize_rows(rows)
            result = {
                "schema_version": result_schema,
                "rows": rows,
                "summary": summary,
                "task_count": 8,
                "references_per_task": references_per_task,
                "conditions": list(CONDITIONS),
                "rollouts": 0,
                "wall_seconds": time.monotonic() - started,
                "run_contract_sha256": canonical_hash(read_json(output_dir / "run_contract.json")),
            }
            validate_finite_tree(result)
            write_json_atomic(output_dir / "analysis.json", result)
            write_json_atomic(output_dir / "summary.json", summary)
            status = {"ok": True}
        except Exception as error:
            status = {"error": repr(error)}
    broadcast(status, rank=rank, world_size=world_size, group=group)
