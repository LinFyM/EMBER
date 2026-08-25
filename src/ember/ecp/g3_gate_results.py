"""Load and pair formal strict250 arms for the G3 closed-loop Gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.pi05_source_checkpoint import read_json


def _rows(
    result: Mapping[str, Any], held_keys: set[tuple[str, int]]
) -> dict[tuple[str, int, int], dict[str, Any]]:
    selected = {
        (str(row["suite"]), int(row["task_id"]), int(row["init_state_id"])): dict(
            row
        )
        for row in result.get("rows", ())
        if (str(row["suite"]), int(row["task_id"])) in held_keys
    }
    expected = {
        (suite, task_id, state) for suite, task_id in held_keys for state in range(50)
    }
    if len(selected) != 250 or set(selected) != expected:
        raise ValueError("G3 arm is not exact held5 strict250")
    return selected


def _normalized_tokenizer(result: Mapping[str, Any]) -> dict[str, Any]:
    cell = result.get("tokenizer", {})
    return {
        "path": str(Path(str(cell.get("path", ""))).resolve()),
        "bytes": int(cell.get("bytes", -1)),
        "manifest_name": Path(str(cell.get("manifest_path", ""))).name,
    }


def _normalized_normalization(result: Mapping[str, Any]) -> dict[str, Any]:
    cell = result.get("normalization", {})
    return {
        "name": Path(str(cell.get("path", ""))).name,
        "bytes": int(cell.get("bytes", -1)),
        "source_only_numeric_reads": cell.get("source_only_numeric_reads"),
        "validation_or_test_numeric_reads": cell.get(
            "validation_or_test_numeric_reads"
        ),
    }


def _task_authority(
    contract: Mapping[str, Any], held: set[tuple[str, int]]
) -> dict[Any, Any]:
    fields = (
        "language",
        "bddl_file",
        "bddl_bytes",
        "init_states_file",
        "init_states_bytes",
        "horizon",
        "init_state_ids",
    )
    return {
        (str(row["suite"]), int(row["task_id"])): {
            field: row.get(field) for field in fields
        }
        for row in contract.get("tasks", ())
        if (str(row["suite"]), int(row["task_id"])) in held
    }


def _arm(*, name: str, path: Path, held_keys: set[tuple[str, int]]) -> dict[str, Any]:
    path = path.resolve()
    result = read_json(path)
    contract = read_json(path.parent / "run_contract.json")
    rows = _rows(result, held_keys)
    if (
        result.get("schema_version") != "ember_pi05_target_eval_results_v2"
        or result.get("mode") != "formal"
        or result.get("role") != "development_train"
        or contract.get("mode") != "formal"
        or contract.get("role") != "development_train"
        or result.get("contract_reference") != contract.get("contract_reference")
    ):
        raise ValueError(f"G3 {name} arm is not a formal development-train result")
    per_task = {
        key: sum(
            bool(row["success"])
            for row_key, row in rows.items()
            if row_key[:2] == key
        )
        for key in held_keys
    }
    return {
        "name": name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "contract_reference": str(result["contract_reference"]),
        "arm": str(result["arm"]),
        "successes": sum(bool(row["success"]) for row in rows.values()),
        "breadth": sum(value > 0 for value in per_task.values()),
        "per_task": per_task,
        "rows": rows,
        "model": result.get("model"),
        "tokenizer": _normalized_tokenizer(result),
        "normalization": _normalized_normalization(result),
        "rng": contract.get("rng"),
        "task_authority": _task_authority(contract, held_keys),
        "adapter": result.get("adapter") or {},
    }


def _paired_authority(arms: Sequence[Mapping[str, Any]]) -> None:
    reference = arms[0]
    fields = ("model", "tokenizer", "normalization", "rng", "task_authority")
    for arm in arms[1:]:
        if any(arm[field] != reference[field] for field in fields):
            raise ValueError(
                "G3 source, task, normalization, tokenizer, or RNG differs"
            )
    for key, base in reference["rows"].items():
        identity = (
            base["language"],
            int(base["env_seed"]),
            int(base["policy_seed_root"]),
        )
        for arm in arms[1:]:
            row = arm["rows"][key]
            if (
                row["language"],
                int(row["env_seed"]),
                int(row["policy_seed_root"]),
            ) != identity:
                raise ValueError("G3 paired row identity differs")
            common = min(
                len(base["policy_noise_seeds"]), len(row["policy_noise_seeds"])
            )
            if (
                base["policy_noise_seeds"][:common]
                != row["policy_noise_seeds"][:common]
            ):
                raise ValueError("G3 paired policy-noise schedule differs")


def load_paired_g3_arms(
    paths: Mapping[str, Path], held_keys: set[tuple[str, int]]
) -> dict[str, dict[str, Any]]:
    arms = {
        name: _arm(name=name, path=path, held_keys=held_keys)
        for name, path in paths.items()
    }
    _paired_authority(tuple(arms.values()))
    return arms
