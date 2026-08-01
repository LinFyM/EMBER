"""Distributed control and atomic result sealing for UCP internal analysis."""

from __future__ import annotations

import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch.distributed as dist

from ember.pi05_source_checkpoint import canonical_hash, read_json, write_json_atomic
from ember.writer.ucp_analysis_summary import (
    summarize_records,
    validate_finite_tree,
    validate_rank_payloads,
)


CONTROL_GROUP_TIMEOUT = timedelta(hours=2)
FAILURE_SCHEMA = "ember_pi05_ucp_internal_analysis_failure_v1"


def control_group_contract() -> dict[str, Any]:
    return {
        "backend": "gloo",
        "timeout_seconds": int(CONTROL_GROUP_TIMEOUT.total_seconds()),
        "purpose": "variable-duration rank coordination only",
    }


def create_control_group(context: Any) -> Any | None:
    if context.world_size <= 1:
        return None
    return dist.new_group(backend="gloo", timeout=CONTROL_GROUP_TIMEOUT)


def destroy_process_groups(control_group: Any | None) -> None:
    """Orderly cleanup for a successful run; failures exit through torchrun."""

    if control_group is not None:
        dist.destroy_process_group(control_group)
    if dist.is_initialized():
        dist.destroy_process_group()


def broadcast_value(
    context: Any,
    value: Any,
    *,
    control_group: Any | None,
) -> Any:
    payload = [value if context.is_main else None]
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, group=control_group)
    if isinstance(payload[0], Mapping) and "error" in payload[0]:
        from ember.writer.model import WriterModelError

        raise WriterModelError(str(payload[0]["error"]))
    return payload[0]


def control_barrier(context: Any, control_group: Any | None) -> None:
    if context.world_size > 1:
        dist.barrier(group=control_group)


def seal_local_rows(
    output_dir: Path,
    context: Any,
    local_results: Sequence[Mapping[str, Any]],
    control_group: Any | None,
) -> None:
    validate_finite_tree(local_results, f"rank_{context.rank}")
    write_json_atomic(output_dir / f"rows_rank_{context.rank:02d}.json", {
        "rank": context.rank,
        "rows": list(local_results),
    })
    control_barrier(context, control_group)


def record_local_failure(
    output_dir: Path,
    rank: int,
    error: BaseException,
) -> None:
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
        print(
            f"failed to persist rank-local UCP error: {write_error!r}",
            file=sys.stderr,
            flush=True,
        )


def finalize_results(
    *,
    output_dir: Path,
    context: Any,
    references_per_task: int,
    conditions: Sequence[str],
    result_schema: str,
    started: float,
    control_group: Any | None,
) -> None:
    status: Any = None
    if context.is_main:
        try:
            payloads = [
                read_json(output_dir / f"rows_rank_{rank:02d}.json")
                for rank in range(context.world_size)
            ]
            rows = validate_rank_payloads(payloads, references_per_task)
            result = {
                "schema_version": result_schema,
                "rows": rows,
                "summary": summarize_records(rows),
                "task_count": 8,
                "references_per_task": references_per_task,
                "conditions": list(conditions),
                "rollouts": 0,
                "wall_seconds": time.monotonic() - started,
                "run_contract_sha256": canonical_hash(
                    read_json(output_dir / "run_contract.json")
                ),
            }
            validate_finite_tree(result)
            write_json_atomic(output_dir / "analysis.json", result)
            write_json_atomic(output_dir / "summary.json", result["summary"])
            status = {"ok": True}
        except Exception as error:
            status = {"error": repr(error)}
    broadcast_value(context, status, control_group=control_group)
