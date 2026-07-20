"""Sealed task and RNG scheduling for official LIBERO fresh evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


class EvaluationContractError(RuntimeError):
    """Raised when evaluation would cross a split or reproducibility boundary."""


@dataclass(frozen=True)
class EvaluationRole:
    name: str
    required_split: str
    task_ids: tuple[int, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_evaluation_config(path: Path, repo_root: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "ember_source_base_eval_v1":
        raise EvaluationContractError("unsupported source-base evaluation schema")
    protocol = config.get("protocol", {})
    for name in ("split", "manifest"):
        authority = repo_root / protocol[name]
        if sha256_file(authority) != protocol[f"{name}_sha256"]:
            raise EvaluationContractError(f"sealed {name} authority hash changed")
    environment = config.get("environment", {})
    required = {
        "suite": "libero_90",
        "fixed_init_states": True,
        "fixed_init_state_count": 50,
        "dummy_settling_steps": 10,
        "max_horizon": 400,
        "terminate_on_success": True,
    }
    for key, expected in required.items():
        if environment.get(key) != expected:
            raise EvaluationContractError(
                f"evaluation environment {key} must remain {expected!r}"
            )
    parallel = config.get("parallel", {})
    if parallel.get("world_size") != 8 or parallel.get("policy_processes_per_gpu") != 1:
        raise EvaluationContractError("evaluation requires eight symmetric policy ranks")
    if int(parallel.get("envs_per_rank", 0)) <= 0:
        raise EvaluationContractError("envs_per_rank must be positive")
    return config


def resolve_role(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    role_name: str,
) -> EvaluationRole:
    """Resolve only train-development or validation; test is structurally absent."""

    roles = config.get("roles", {})
    if role_name not in roles:
        raise EvaluationContractError(
            f"role {role_name!r} is not available; reporting-only test remains sealed"
        )
    role = roles[role_name]
    task_ids = tuple(int(task_id) for task_id in role.get("task_ids", []))
    required_split = str(role.get("required_split"))
    if required_split not in {"train", "validation"}:
        raise EvaluationContractError("active evaluator cannot access test tasks")
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise EvaluationContractError("evaluation role task IDs must be non-empty and unique")
    split_by_task = {
        int(record["task_index"]): str(record["split"])
        for record in manifest.get("tasks", [])
    }
    mismatched = [
        task_id for task_id in task_ids if split_by_task.get(task_id) != required_split
    ]
    if mismatched:
        raise EvaluationContractError(
            f"role {role_name} crosses its {required_split} split: {mismatched}"
        )
    return EvaluationRole(role_name, required_split, task_ids)


def partition_fixed_state_ids(
    state_count: int, world_size: int, rank: int
) -> tuple[int, ...]:
    if state_count <= 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise EvaluationContractError("invalid fixed-state partition dimensions")
    return tuple(range(rank, state_count, world_size))


def batched_with_padding(
    values: Iterable[int], batch_size: int
) -> tuple[tuple[tuple[int, ...], int], ...]:
    values = tuple(values)
    if not values or batch_size <= 0:
        raise EvaluationContractError("cannot batch an empty fixed-state assignment")
    result: list[tuple[tuple[int, ...], int]] = []
    for start in range(0, len(values), batch_size):
        valid = values[start : start + batch_size]
        padded = valid + (valid[-1],) * (batch_size - len(valid))
        result.append((padded, len(valid)))
    return tuple(result)


def environment_seed(base_seed: int, task_id: int, init_state_id: int) -> int:
    return base_seed + task_id * 1_000 + init_state_id


def policy_seed(base_seed: int, task_id: int, batch_index: int, rank: int) -> int:
    return base_seed + task_id * 100_000 + batch_index * 100 + rank


def validate_complete_rows(
    rows: Iterable[Mapping[str, Any]], task_ids: Iterable[int], state_count: int
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in rows]
    task_ids = tuple(task_ids)
    expected = {
        (task_id, state_id)
        for task_id in task_ids
        for state_id in range(state_count)
    }
    actual = {(int(row["task_id"]), int(row["init_state_id"])) for row in rows}
    if actual != expected or len(rows) != len(expected):
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise EvaluationContractError(
            f"fresh evaluation rows are incomplete; missing={missing[:5]} extra={extra[:5]}"
        )
    return sorted(rows, key=lambda row: (int(row["task_id"]), int(row["init_state_id"])))


def aggregate_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["task_id"]), []).append(row)
    per_task: list[dict[str, Any]] = []
    total_successes = 0
    total_episodes = 0
    for task_id in sorted(grouped):
        task_rows = grouped[task_id]
        successes = sum(bool(row["success"]) for row in task_rows)
        total_successes += successes
        total_episodes += len(task_rows)
        per_task.append(
            {
                "task_id": task_id,
                "language": task_rows[0]["language"],
                "successes": successes,
                "episodes": len(task_rows),
                "success_rate": successes / len(task_rows),
            }
        )
    return {
        "per_task": per_task,
        "overall": {
            "successes": total_successes,
            "episodes": total_episodes,
            "success_rate": total_successes / total_episodes,
        },
    }
