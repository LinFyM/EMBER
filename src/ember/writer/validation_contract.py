"""Frozen Writer validation authority, rank assignment, and paired statistics."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ember.writer.core import load_writer_contract, sha256_file


class WriterValidationError(RuntimeError):
    """Raised when Writer validation leaves its frozen contract."""


def require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise WriterValidationError(f"{label} changed: {actual!r} != {expected!r}")


def load_validation_contract(path: Path, *, repo_root: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            spec = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise WriterValidationError("Writer validation contract is unreadable") from error
    require(spec.get("schema_version"), 1, "validation schema")
    status = spec.get("status")
    if status not in {
        "predeclared_before_writer_step1000_or_validation_outcomes",
        "predeclared_physical_norm_recovery_validation_before_closed_loop_outcomes",
    }:
        raise WriterValidationError("validation predeclaration is not frozen")
    authority = spec["authority"]
    for relative_key, hash_key in (
        ("writer_contract_relative_path", "writer_contract_sha256"),
        ("split_reseal_relative_path", "split_reseal_sha256"),
    ):
        bound = repo_root / authority[relative_key]
        require(sha256_file(bound), authority[hash_key], hash_key)
    upstream = load_writer_contract(
        repo_root / authority["writer_contract_relative_path"],
        phase0_path=repo_root / "configs/phase0.toml",
        split_path=repo_root / "configs/libero90_split_reseal.json",
        gate_zero_path=repo_root / "configs/gate_zero_oracle_pilot.toml",
        mature_lora_path=repo_root / "configs/gate_zero_mature_lora_positive_control.toml",
    )
    evaluation = spec["evaluation"]
    direct = spec["direct_baseline"]
    require(evaluation["task_ids"], upstream["validation"]["task_ids"], "validation tasks")
    require(direct["task_ids"], evaluation["task_ids"], "direct-baseline tasks")
    expected_writer_arm = (
        "writer_physical_norm_recovery"
        if "reuse_baseline" in spec
        else "writer_cold_start"
    )
    require(
        evaluation["arms"],
        ["frozen_base", expected_writer_arm, "matched_direct_task_local_lora"],
        "validation arms",
    )
    if "reuse_baseline" in spec:
        require(
            spec["reuse_baseline"]["arms"],
            ["frozen_base", "matched_direct_task_local_lora"],
            "reused baseline arms",
        )
    for key in ("rank", "alpha", "dropout", "expected_parameter_count"):
        require(spec["lora"][key], upstream["lora"][key], f"LoRA {key}")
    require(spec["lora"]["target_count"], upstream["lora"]["target_count"], "LoRA targets")
    require(evaluation["rollouts_per_task_arm"], 64, "validation denominator")
    require(
        evaluation["rollouts_per_policy_seed"] * len(evaluation["policy_rng_seeds"]),
        evaluation["rollouts_per_task_arm"],
        "validation rollout decomposition",
    )
    require(spec["parallel"]["world_size"], 8, "validation world size")
    require(authority["test_held_numeric_access"], False, "test/held access")
    require(evaluation["test_held_access"], False, "evaluation test/held access")
    return spec


def validation_work_for_rank(
    spec: Mapping[str, Any], *, rank: int, world_size: int
) -> dict[str, Any]:
    if world_size != spec["parallel"]["world_size"] or not 0 <= rank < world_size:
        raise WriterValidationError("validation rank topology changed")
    tasks = list(spec["evaluation"]["task_ids"])
    if "reuse_baseline" in spec:
        pairs = [
            (task_id, arm)
            for task_id in tasks
            for arm in spec["evaluation"]["arms"]
        ]
        return {
            "direct_fit_task": None,
            "evaluation_arms": pairs[rank::world_size],
        }
    direct_ranks = list(spec["parallel"]["direct_fit_ranks"])
    direct_task = tasks[direct_ranks.index(rank)] if rank in direct_ranks else None
    if direct_task is not None:
        arms = [(direct_task, "matched_direct_task_local_lora")]
    else:
        fixed = [
            (task, arm)
            for task in tasks
            for arm in ("frozen_base", "writer_cold_start")
        ]
        ranks = list(spec["parallel"]["base_writer_eval_ranks"])
        offset = ranks.index(rank)
        arms = fixed[offset:: len(ranks)]
    return {"direct_fit_task": direct_task, "evaluation_arms": arms}


def _paired_interval(
    left: Sequence[bool],
    right: Sequence[bool],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if len(left) != len(right) or not left or replicates < 1000:
        raise WriterValidationError("paired bootstrap input changed")
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(difference, size=(replicates, len(difference)), replace=True).mean(1)
    return {
        "episodes": len(left),
        "paired_success_rate_difference": float(difference.mean()),
        "paired_bootstrap_95_interval": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "paired_wins": int(np.sum(difference > 0)),
        "paired_losses": int(np.sum(difference < 0)),
        "paired_ties": int(np.sum(difference == 0)),
    }


def _wilson(successes: int, episodes: int) -> list[float]:
    z = 1.959963984540054
    p = successes / episodes
    denominator = 1 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / denominator
    radius = z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes**2)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _success_summary(cell: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(row["success"]) for row in cell)
    return {
        "successes": successes,
        "episodes": len(cell),
        "success_rate": successes / len(cell),
        "wilson_95_interval": _wilson(successes, len(cell)),
    }


def _paired_cell_interval(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    *,
    episode_fields: Sequence[str],
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    episode_key = lambda row: tuple(row[field] for field in episode_fields)
    left = {episode_key(row): bool(row["success"]) for row in left_rows}
    right = {episode_key(row): bool(row["success"]) for row in right_rows}
    if set(left) != set(right):
        raise WriterValidationError("validation arms are not episode-paired")
    ordered = sorted(left)
    return _paired_interval(
        [left[key] for key in ordered],
        [right[key] for key in ordered],
        seed=seed,
        replicates=replicates,
    )


def _aggregate_per_task(
    grouped: Mapping[tuple[int, str, int], Sequence[Mapping[str, Any]]],
    *,
    task_ids: Sequence[int],
    arms: Sequence[str],
    horizons: Sequence[int],
    expected_rollouts: int,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    per_task: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for task_id in task_ids:
        per_task[str(task_id)] = {}
        comparisons[str(task_id)] = {}
        for arm in arms:
            per_task[str(task_id)][arm] = {}
            if arm != "frozen_base":
                comparisons[str(task_id)][arm] = {}
            for horizon in horizons:
                cell = grouped.get((task_id, arm, horizon), [])
                if len(cell) != expected_rollouts:
                    raise WriterValidationError(
                        f"task {task_id} arm {arm} h{horizon} has {len(cell)} rows"
                    )
                per_task[str(task_id)][arm][str(horizon)] = _success_summary(cell)
                if arm != "frozen_base":
                    comparisons[str(task_id)][arm][str(horizon)] = _paired_cell_interval(
                        cell,
                        grouped.get((task_id, "frozen_base", horizon), []),
                        episode_fields=(
                            "policy_rng_seed",
                            "evaluator_seed",
                            "physical_init_state_index",
                        ),
                        seed=bootstrap_seed + task_id * 101 + horizon,
                        replicates=bootstrap_replicates,
                    )
    return per_task, comparisons


def _aggregate_overall(
    grouped: Mapping[tuple[int, str, int], Sequence[Mapping[str, Any]]],
    *,
    task_ids: Sequence[int],
    arms: Sequence[str],
    horizons: Sequence[int],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    overall: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for arm in arms:
        overall[arm] = {}
        if arm != "frozen_base":
            comparisons[arm] = {}
        for horizon in horizons:
            cell = [
                row
                for task_id in task_ids
                for row in grouped[(task_id, arm, horizon)]
            ]
            overall[arm][str(horizon)] = _success_summary(cell)
            if arm != "frozen_base":
                base = [
                    row
                    for task_id in task_ids
                    for row in grouped[(task_id, "frozen_base", horizon)]
                ]
                comparisons[arm][str(horizon)] = _paired_cell_interval(
                    cell,
                    base,
                    episode_fields=(
                        "task_id",
                        "policy_rng_seed",
                        "evaluator_seed",
                        "physical_init_state_index",
                    ),
                    seed=bootstrap_seed + horizon,
                    replicates=bootstrap_replicates,
                )
    return overall, comparisons


def aggregate_validation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_ids: Sequence[int],
    arms: Sequence[str],
    horizons: Sequence[int],
    expected_rollouts: int,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    grain = (
        "task_id",
        "arm",
        "execution_horizon",
        "policy_rng_seed",
        "evaluator_seed",
        "physical_init_state_index",
    )
    keys = [tuple(row[key] for key in grain) for row in rows]
    if len(keys) != len(set(keys)):
        raise WriterValidationError("validation rows contain duplicate episode grain")
    grouped: dict[tuple[int, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row["task_id"]), str(row["arm"]), int(row["execution_horizon"]))
        grouped.setdefault(key, []).append(row)
    per_task, comparisons = _aggregate_per_task(
        grouped,
        task_ids=task_ids,
        arms=arms,
        horizons=horizons,
        expected_rollouts=expected_rollouts,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    overall, overall_comparisons = _aggregate_overall(
        grouped,
        task_ids=task_ids,
        arms=arms,
        horizons=horizons,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    return {
        "per_task": per_task,
        "paired_vs_frozen_base": comparisons,
        "overall": overall,
        "overall_paired_vs_frozen_base": overall_comparisons,
        "raw_episode_rows": len(rows),
    }
