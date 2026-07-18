"""Seal and verify the LIBERO-90 split from pinned public task specifications."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ember.contracts import load_contract, validate_contract
from ember.libero_split import (
    MINIMUM_SOURCE_ROLE_OCCURRENCES,
    SEARCH_CANDIDATE_COUNT,
    SEARCH_SEED,
    SplitResealError,
    _sorted_split,
    audit_split,
    search_split,
)
from ember.libero_task_factors import FACTOR_ROLE_DEFINITIONS, FACTOR_SCHEMA, factor_task


TASK_NAME_RE = re.compile(r"^([A-Z]+(?:_[A-Z]+)*_SCENE\d+)_(.+)$")


def _git_blob_sha(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def load_task_specs(task_map_path: Path, *, expected_git_blob_sha: str) -> list[dict[str, Any]]:
    task_map_path = task_map_path.resolve()
    observed = _git_blob_sha(task_map_path)
    if observed != expected_git_blob_sha:
        raise SplitResealError(
            f"task-map blob mismatch: expected {expected_git_blob_sha}, observed {observed}"
        )
    tree = ast.parse(task_map_path.read_text(encoding="utf-8"), filename=str(task_map_path))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "libero_task_map"
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise SplitResealError("task-map authority must define libero_task_map exactly once")
    task_map = ast.literal_eval(assignments[0].value)
    names = task_map.get("libero_90") if isinstance(task_map, dict) else None
    if not isinstance(names, list) or len(names) != 90 or len(set(names)) != 90:
        raise SplitResealError("task-map authority must contain 90 unique LIBERO-90 tasks")
    records = []
    for index, name in enumerate(names):
        matched = TASK_NAME_RE.fullmatch(name)
        if matched is None:
            raise SplitResealError(f"task name lacks canonical scene/specification: {index}")
        scene, normalized = matched.groups()
        records.append(
            factor_task(
                task_index=index,
                scene=scene,
                language=normalized.replace("_", " "),
            )
        )
    return records


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def specification_surface_sha256(tasks: Sequence[Mapping[str, Any]]) -> str:
    surface = [
        {
            "task_index": int(task["task_index"]),
            "scene": str(task["scene"]),
            "language": str(task["language"]),
        }
        for task in sorted(tasks, key=lambda task: int(task["task_index"]))
    ]
    return _sha256_json(surface)


def build_reseal_record(
    contract: Mapping[str, Any],
    *,
    task_map_path: Path,
    seed: int = SEARCH_SEED,
    candidate_count: int = SEARCH_CANDIDATE_COUNT,
) -> dict[str, Any]:
    validate_contract(contract)
    tasks = load_task_specs(
        task_map_path,
        expected_git_blob_sha=contract["upstreams"]["libero_task_map"]["git_blob_sha"],
    )
    reseal_contract = contract.get("split_reseal", {})
    prior_split = reseal_contract.get("prior_split", contract["splits"])
    minimum = int(
        reseal_contract.get(
            "minimum_source_role_occurrences", MINIMUM_SOURCE_ROLE_OCCURRENCES
        )
    )
    configured_seed = int(reseal_contract.get("search_seed", seed))
    configured_candidates = int(
        reseal_contract.get("search_candidate_count", candidate_count)
    )
    result = search_split(
        tasks,
        prior_split=prior_split,
        sizes={"source": 60, "validation": 15, "held_out": 15},
        minimum=minimum,
        seed=configured_seed,
        candidate_count=configured_candidates,
    )
    return {
        "schema_version": 1,
        "status": "permanently_resealed_before_libero90_policy_outcomes",
        "recovery_class": "benchmark_design_specification_coverage",
        "authority": {
            "task_map_git_blob_sha": contract["upstreams"]["libero_task_map"][
                "git_blob_sha"
            ],
            "allowed_fields": ["task_index", "scene", "language"],
            "numeric_or_privileged_fields_read": [],
            "specification_surface_sha256": specification_surface_sha256(tasks),
        },
        "factor_contract": {
            "schema": FACTOR_SCHEMA,
            "role_definitions": FACTOR_ROLE_DEFINITIONS,
            "minimum_source_role_occurrences": minimum,
            "unknown_or_ambiguous_template_policy": "fail_closed",
            "task_count": len(tasks),
        },
        "search_contract": {
            "algorithm": result["algorithm"],
            "seed": result["seed"],
            "candidate_count": result["candidate_count"],
            "split_sizes": {"source": 60, "validation": 15, "held_out": 15},
            "combined_objective_priority": [
                "maximize_novel_full_composition_count",
                "maximize_same_scene_hard_negative_count",
                "maximize_same_scene_source_count",
                "minimize_scene_distribution_penalty",
                "minimize_difficulty_distribution_penalty",
                "maximize_prior_evaluation_retention",
                "lexicographically_smallest_task_ids",
            ],
            "partition_objective_priority": [
                "maximize_scenes_represented_in_both_validation_and_held",
                "minimize_validation_held_scene_imbalance",
                "minimize_validation_held_difficulty_imbalance",
                "minimize_validation_held_primitive_role_imbalance",
                "maximize_prior_exact_role_retention",
                "lexicographically_smallest_validation_task_ids",
            ],
        },
        "prior_split": _sorted_split(prior_split),
        "active_split": result["split"],
        "gate_minus_one_thresholds": dict(contract["gate_minus_one"]["thresholds"]),
        "selection_diagnostics": {
            "combined_objective": result["combined_objective"],
            "partition_objective": result["partition_objective"],
            "audit": result["audit"],
        },
        "tasks": tasks,
    }


def load_sealed_record(record_path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Load the permanent record and verify its hash, factors, split, and audit."""

    record_path = record_path.resolve()
    observed_bytes = record_path.read_bytes()
    record = json.loads(observed_bytes)
    if observed_bytes != _canonical_json(record):
        raise SplitResealError("reseal record is not canonical sorted JSON")
    configured = contract["split_reseal"]
    if hashlib.sha256(observed_bytes).hexdigest() != configured["record_sha256"]:
        raise SplitResealError("reseal record SHA256 differs from the active contract")
    if record.get("status") != configured["status"]:
        raise SplitResealError("reseal record status differs from the active contract")
    if record.get("gate_minus_one_thresholds") != contract["gate_minus_one"]["thresholds"]:
        raise SplitResealError("reseal record changes the frozen Gate -1 thresholds")
    if _sorted_split(contract["splits"]) != record.get("active_split"):
        raise SplitResealError("active contract split differs from the sealed record")
    tasks = record.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 90:
        raise SplitResealError("reseal factor table must contain exactly 90 tasks")
    for task in tasks:
        expected = factor_task(
            task_index=task["task_index"],
            scene=task["scene"],
            language=task["language"],
        )
        if task != expected:
            raise SplitResealError(
                f"sealed factors differ from strict parser at task {task['task_index']}"
            )
    if specification_surface_sha256(tasks) != configured["specification_surface_sha256"]:
        raise SplitResealError("sealed specification surface SHA256 differs from the contract")
    observed_audit = audit_split(
        tasks,
        **record["active_split"],
        minimum=configured["minimum_source_role_occurrences"],
    )
    if observed_audit != record["selection_diagnostics"]["audit"]:
        raise SplitResealError("sealed split audit differs from deterministic recomputation")
    if not observed_audit["mechanics_valid"]:
        raise SplitResealError("sealed split no longer satisfies role coverage")
    return record


def verify_reseal_record(
    record_path: Path,
    contract: Mapping[str, Any],
    *,
    task_map_path: Path,
) -> dict[str, Any]:
    record_path = record_path.resolve()
    observed_bytes = record_path.read_bytes()
    record = load_sealed_record(record_path, contract)
    expected = build_reseal_record(contract, task_map_path=task_map_path)
    if record != expected:
        raise SplitResealError("reseal record differs from deterministic regeneration")
    return {
        "status": record["status"],
        "record_sha256": hashlib.sha256(observed_bytes).hexdigest(),
        "specification_surface_sha256": record["authority"][
            "specification_surface_sha256"
        ],
        "mechanics_valid": record["selection_diagnostics"]["audit"]["mechanics_valid"],
    }


def _write_new(path: Path, value: Any) -> None:
    path = path.resolve()
    if path.exists():
        raise SplitResealError(f"refusing to overwrite reseal record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(value))
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path("configs/phase0.toml"))
    parser.add_argument("--task-map", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    contract = load_contract(args.contract)
    if args.output is not None:
        record = build_reseal_record(contract, task_map_path=args.task_map)
        _write_new(args.output, record)
        result = {
            "status": record["status"],
            "output": str(args.output.resolve()),
            "record_sha256": hashlib.sha256(_canonical_json(record)).hexdigest(),
            "split": record["active_split"],
        }
    else:
        result = verify_reseal_record(args.verify, contract, task_map_path=args.task_map)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
