"""Durable finished rows survive failure later in the same queue shard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_queue import publish_json_exclusive, read_json_with_size

SCHEMA = "ember_pi05_completed_episode_v1"


def receipt_path(contract: Mapping, suite: str, task_id: int, state: int) -> Path | None:
    root = contract.get("output_dir")
    if root is None:
        return None
    return Path(root) / "completed_episodes" / f"{suite}_task_{task_id:02d}_state_{state:03d}.json"


def save_completed_row(contract: Mapping, row: Mapping) -> None:
    path = receipt_path(contract, row["suite"], int(row["task_id"]), int(row["init_state_id"]))
    if path is None:
        return
    payload = {"schema_version": SCHEMA, "contract_reference": contract["contract_reference"],
               "row": dict(row)}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        observed, _ = read_json_with_size(path)
        if observed != payload:
            raise Pi05EvaluationError("finished episode would overwrite a different completed row")
    else:
        publish_json_exclusive(path, payload)


def load_completed_rows(contract: Mapping, shard: Any, task: Mapping, validate) -> list[dict]:
    rows = []
    for state in shard.init_state_ids:
        path = receipt_path(contract, shard.suite, shard.task_id, int(state))
        if path is None or not path.exists():
            continue
        payload, _ = read_json_with_size(path)
        row = payload.get("row")
        if (payload.get("schema_version") != SCHEMA
                or payload.get("contract_reference") != contract["contract_reference"]
                or not isinstance(row, dict) or row.get("init_state_id") != int(state)):
            raise Pi05EvaluationError("completed episode receipt changed its contract or identity")
        validate(row, contract=contract, shard=shard, task=task)
        rows.append(row)
    return rows
