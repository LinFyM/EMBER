"""Seal and recover rank-local internal-analysis results."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import canonical_hash, read_json, write_json_atomic
from ember.writer.model import WriterModelError


SummaryFunction = Callable[[Sequence[Mapping[str, Any]]], dict[str, Any]]


def _rank_payloads(
    output_dir: Path,
    contract: Mapping[str, Any],
    *,
    wait: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = [
        output_dir / f"rank_{rank:02d}_rows.json"
        for rank in range(int(contract["world_size"]))
    ]
    if wait:
        deadline = time.monotonic() + 3600
        while not all(path.is_file() for path in paths):
            if time.monotonic() >= deadline:
                raise WriterModelError("analysis rank sealing timed out")
            time.sleep(1)
    if not all(path.is_file() for path in paths):
        raise WriterModelError("existing analysis rank payloads are incomplete")
    payloads = [read_json(path) for path in paths]
    combined = [row for payload in payloads for row in payload["rows"]]
    combined.sort(
        key=lambda row: (int(row["checkpoint_cursor"]), int(row["global_task_id"]))
    )
    expected = len(contract["checkpoints"]) * len(contract["tasks"])
    if len(combined) != expected:
        raise WriterModelError("analysis Cartesian result coverage changed")
    return payloads, combined


def finalize_internal_analysis(
    output_dir: Path,
    contract: Mapping[str, Any],
    *,
    result_schema: str,
    summary_function: SummaryFunction,
    started: float,
    wait_for_ranks: bool,
    aggregation: Mapping[str, Any] | None = None,
) -> None:
    rank_payloads, combined = _rank_payloads(
        output_dir, contract, wait=wait_for_ranks
    )
    summary = summary_function(combined)
    completion = {
        "rows": len(combined),
        "tasks": len(contract["tasks"]),
        "checkpoints": len(contract["checkpoints"]),
        "world_size": int(contract["world_size"]),
        "wall_seconds": time.monotonic() - started,
        "max_cuda_reserved_bytes": max(
            int(payload["max_cuda_reserved_bytes"]) for payload in rank_payloads
        ),
        "target_action_reads": 0,
        "validation_or_test_reads": 0,
        "rank_payloads_reused": aggregation is not None,
    }
    result = {
        "schema_version": result_schema,
        "run_contract_sha256": canonical_hash(contract),
        "rows": combined,
        "summary": summary,
        "completion": completion,
    }
    if aggregation is not None:
        result["aggregation"] = dict(aggregation)
    write_json_atomic(output_dir / "results.json", result)
    write_json_atomic(
        output_dir / "completion.json",
        {**completion, "results_payload_sha256": canonical_hash(result)},
    )
    print(json.dumps(result["summary"], sort_keys=True), flush=True)


def recover_internal_analysis_aggregation(
    args: Any,
    *,
    repo_root: Path,
    result_schema: str,
    summary_function: SummaryFunction,
) -> None:
    contract = read_json(args.output_dir / "run_contract.json")
    state = git_state(repo_root)
    if args.mode == "formal" and (
        state["dirty_paths"] or state["commit"] != state["origin_main"]
    ):
        raise WriterModelError("formal aggregation recovery requires pushed clean code")
    if (
        Path(contract["training_run"]["path"]) != args.training_run
        or [str(row["path"]) for row in contract["checkpoints"]]
        != [str(path) for path in args.checkpoints]
        or Path(contract["tokenizer_path"]) != args.tokenizer_path
        or Path(contract["data_root"]) != args.data_root
    ):
        raise WriterModelError("aggregation recovery inputs crossed the sealed run")
    aggregation_contract = {
        "schema_version": "ember_pi05_internal_analysis_aggregation_recovery_v1",
        "command": list(os.sys.argv),
        "git": state,
        "source_run_contract_sha256": canonical_hash(contract),
        "rank_payload_count": int(contract["world_size"]),
    }
    write_json_atomic(args.output_dir / "aggregation_contract.json", aggregation_contract)
    started = time.monotonic()
    finalize_internal_analysis(
        args.output_dir,
        contract,
        result_schema=result_schema,
        summary_function=summary_function,
        started=started,
        wait_for_ranks=False,
        aggregation={"contract_sha256": canonical_hash(aggregation_contract)},
    )
