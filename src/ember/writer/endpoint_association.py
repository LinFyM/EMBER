"""Pre-registered post-hoc association audit for endpoint10 diagnostics."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ember.pi05_source_checkpoint import (
    canonical_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.writer.endpoint_provenance import SEALED_PANEL_PAYLOAD_SHA256
from ember.writer.endpoint_validation import ENDPOINT_SUMMARY_SCHEMA, METRICS
from ember.writer.model import WriterModelError


ASSOCIATION_SCHEMA = "ember_pi05_endpoint_primary_association_v1"
PRIMARY_METRIC = "rollout10_executed5_valid_normalized_mse"
PERMUTATION_REPETITIONS = 100_000
PERMUTATION_SEED_LABEL = "endpoint10-primary-candidate-label-permutation-v1"
EXPECTED_CURSORS = {
    "v52_new": (150, 200, 350, 400),
    "ucp_raw": (50, 100, 150, 200),
    "v6_fast": (50, 100, 150, 200, 250, 300, 350, 400),
    "v52_old": (900,),
    "v6_old": (900,),
}
MULTI_CHECKPOINT_FAMILIES = ("v52_new", "ucp_raw", "v6_fast")
EXPECTED_TASK_IDS = (1, 3, 11, 13, 23, 26, 31, 32)


def _finite_scalar(label: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise WriterModelError(f"endpoint association {label} changed") from error
    if not math.isfinite(result):
        raise WriterModelError(f"endpoint association {label} is non-finite")
    return result


def _finite_vector(label: str, values: Sequence[Any]) -> np.ndarray:
    result = np.asarray(
        [_finite_scalar(f"{label} value", value) for value in values],
        dtype=np.float64,
    )
    if result.ndim != 1 or result.size < 2:
        raise WriterModelError(f"endpoint association {label} shape changed")
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic one-based average ranks, including exact ties."""

    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise WriterModelError("endpoint association rank input changed")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def _pearson(values: np.ndarray, outcomes: np.ndarray) -> float | None:
    if values.shape != outcomes.shape or values.ndim != 1 or values.size < 2:
        raise WriterModelError("endpoint association Pearson shape changed")
    left = values - values.mean()
    right = outcomes - outcomes.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    if denominator == 0.0:
        return None
    result = float((left @ right) / denominator)
    if not math.isfinite(result):
        raise WriterModelError("endpoint association Pearson is non-finite")
    return max(-1.0, min(1.0, result))


def _spearman(values: np.ndarray, outcomes: np.ndarray) -> float | None:
    return _pearson(_average_ranks(values), _average_ranks(outcomes))


def _permutation_seed() -> int:
    digest = hashlib.sha256(
        f"{SEALED_PANEL_PAYLOAD_SHA256}:{PERMUTATION_SEED_LABEL}".encode()
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _permutation_pvalue(
    qualities: np.ndarray,
    outcomes: np.ndarray,
    observed: float,
) -> tuple[float, int]:
    """Compute the fixed two-sided Monte Carlo candidate-label p-value."""

    quality_ranks = _average_ranks(qualities)
    outcome_ranks = _average_ranks(outcomes)
    generator = np.random.Generator(np.random.PCG64(_permutation_seed()))
    exceedances = 0
    threshold = abs(observed)
    for _ in range(PERMUTATION_REPETITIONS):
        # Ranking commutes exactly with a label permutation, including ties.
        permuted = generator.permutation(outcome_ranks)
        correlation = _pearson(quality_ranks, permuted)
        if correlation is None:
            raise WriterModelError("endpoint association permutation is undefined")
        exceedances += int(abs(correlation) >= threshold)
    return (
        (1 + exceedances) / (PERMUTATION_REPETITIONS + 1),
        exceedances,
    )


def _candidate_task_outcomes(candidate: Mapping[str, Any]) -> dict[int, int]:
    rows = candidate.get("correct400_per_task", [])
    result = {
        int(row.get("global_task_id", -1)): int(row.get("successes", -1))
        for row in rows
    }
    if (
        tuple(sorted(result)) != EXPECTED_TASK_IDS
        or len(rows) != len(result)
        or any(not 0 <= value <= 50 for value in result.values())
        or any(int(row.get("episodes", -1)) != 50 for row in rows)
    ):
        raise WriterModelError("endpoint association task outcomes changed")
    return result


def _candidate_task_qualities(candidate: Mapping[str, Any]) -> dict[int, float]:
    per_task = candidate.get("per_task", {})
    if tuple(sorted(map(int, per_task))) != EXPECTED_TASK_IDS:
        raise WriterModelError("endpoint association task metrics changed")
    result = {}
    for task_id in EXPECTED_TASK_IDS:
        row = per_task[str(task_id)]
        if int(row.get("rows", -1)) != 64:
            raise WriterModelError("endpoint association task row count changed")
        result[task_id] = _finite_scalar(
            f"task {task_id} primary quality",
            row.get("metrics", {}).get(PRIMARY_METRIC, {}).get("quality"),
        )
    return result


def _validate_candidates(summary: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    candidates = tuple(summary.get("candidates", ()))
    observed: dict[str, list[int]] = {}
    pairing = set()
    ids = set()
    for candidate in candidates:
        family = str(candidate.get("family", ""))
        cursor = int(candidate.get("checkpoint_cursor", -1))
        candidate_id = str(candidate.get("candidate_id", ""))
        observed.setdefault(family, []).append(cursor)
        pairing.add(str(candidate.get("outcome_pairing_sha256", "")))
        ids.add(candidate_id)
        correct = int(candidate.get("correct400", -1))
        if (
            candidate_id != f"{family}_step{cursor:08d}"
            or int(candidate.get("rows", -1)) != 512
            or not 0 <= correct <= 400
            or not 0 <= int(candidate.get("task_breadth", -1)) <= 8
            or correct != sum(_candidate_task_outcomes(candidate).values())
        ):
            raise WriterModelError("endpoint association candidate changed")
        _candidate_task_qualities(candidate)
        _finite_scalar(
            f"{candidate_id} primary quality",
            candidate.get("aggregate", {}).get(PRIMARY_METRIC, {}).get("quality"),
        )
    normalized = {
        family: tuple(sorted(cursors)) for family, cursors in observed.items()
    }
    if (
        normalized != EXPECTED_CURSORS
        or len(candidates) != len(ids)
        or len(candidates) != 18
        or len(pairing) != 1
        or "" in pairing
    ):
        raise WriterModelError("endpoint association candidate set changed")
    return tuple(sorted(candidates, key=lambda row: str(row["candidate_id"])))


def _correlation_record(
    qualities: Sequence[Any], outcomes: Sequence[Any]
) -> dict[str, Any]:
    quality_values = _finite_vector("quality", qualities)
    outcome_values = _finite_vector("outcome", outcomes)
    return {
        "count": int(quality_values.size),
        "pearson": _pearson(quality_values, outcome_values),
        "spearman": _spearman(quality_values, outcome_values),
    }


def _source_contract(endpoint_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = read_json(endpoint_root / "summary.json")
    contract = read_json(endpoint_root / "run_contract.json")
    expected_rows = 18 * 512
    valid = (
        summary.get("schema_version") == ENDPOINT_SUMMARY_SCHEMA
        and summary.get("metrics") == list(METRICS)
        and summary.get("primary_metric") == PRIMARY_METRIC
        and int(summary.get("row_count", -1)) == expected_rows
        and int(summary.get("validation_action_rows_read", -1)) == expected_rows
        and int(summary.get("test_action_reads", -1)) == 0
        and summary.get("environment_constructed") is False
        and summary.get("parameter_gradients_computed") is False
        and contract.get("mode") == "formal"
        and int(contract.get("world_size", -1)) == 4
        and contract.get("physical_gpu_ids") == [4, 5, 6, 7]
        and contract.get("max_groups_per_task") is None
        and contract.get("panel", {}).get("manifest_payload_sha256")
        == SEALED_PANEL_PAYLOAD_SHA256
        and contract.get("information_wall", {}).get("environment_constructed")
        is False
        and contract.get("information_wall", {}).get(
            "parameter_gradients_computed"
        )
        is False
    )
    if not valid:
        raise WriterModelError("endpoint association formal source changed")
    return summary, contract


def analyze_endpoint_association(
    endpoint_root: Path,
    preregistration: Path,
) -> dict[str, Any]:
    """Apply the immutable four-part gate to one completed formal endpoint run."""

    endpoint_root = endpoint_root.resolve()
    preregistration = preregistration.resolve()
    summary, contract = _source_contract(endpoint_root)
    candidates = _validate_candidates(summary)
    qualities = _finite_vector(
        "global qualities",
        [candidate["aggregate"][PRIMARY_METRIC]["quality"] for candidate in candidates],
    )
    outcomes = _finite_vector(
        "global outcomes", [candidate["correct400"] for candidate in candidates]
    )
    global_spearman = _spearman(qualities, outcomes)
    if global_spearman is None:
        raise WriterModelError("endpoint association global Spearman is undefined")
    permutation_p, permutation_exceedances = _permutation_pvalue(
        qualities, outcomes, global_spearman
    )

    family_records = {}
    pooled_quality = []
    pooled_outcome = []
    for family in MULTI_CHECKPOINT_FAMILIES:
        selected = [row for row in candidates if row["family"] == family]
        family_quality = _finite_vector(
            f"{family} quality",
            [row["aggregate"][PRIMARY_METRIC]["quality"] for row in selected],
        )
        family_outcome = _finite_vector(
            f"{family} outcome", [row["correct400"] for row in selected]
        )
        family_records[family] = _correlation_record(
            family_quality, family_outcome
        )
        pooled_quality.extend((family_quality - family_quality.mean()).tolist())
        pooled_outcome.extend((family_outcome - family_outcome.mean()).tolist())
    pooled = _correlation_record(pooled_quality, pooled_outcome)

    by_key = {
        (str(row["family"]), int(row["checkpoint_cursor"])): row
        for row in candidates
    }
    recipe_comparisons = (
        ("v52_old_step900_gt_v52_new_macro150", ("v52_old", 900), ("v52_new", 150)),
        ("v6_fast_macro150_gt_v6_old_step900", ("v6_fast", 150), ("v6_old", 900)),
    )
    recipe = {}
    for name, left_key, right_key in recipe_comparisons:
        left = _finite_scalar(
            f"{name} left", by_key[left_key]["aggregate"][PRIMARY_METRIC]["quality"]
        )
        right = _finite_scalar(
            f"{name} right", by_key[right_key]["aggregate"][PRIMARY_METRIC]["quality"]
        )
        recipe[name] = {
            "left_quality": left,
            "right_quality": right,
            "difference": left - right,
            "pass": left > right,
        }

    task_records = {}
    task_spearman = []
    for task_id in EXPECTED_TASK_IDS:
        record = _correlation_record(
            [_candidate_task_qualities(row)[task_id] for row in candidates],
            [_candidate_task_outcomes(row)[task_id] for row in candidates],
        )
        task_records[str(task_id)] = record
        task_spearman.append(record["spearman"])
    tasks_defined = all(value is not None for value in task_spearman)
    finite_task_spearman = [float(value) for value in task_spearman if value is not None]
    task_median = (
        float(np.median(np.asarray(finite_task_spearman, dtype=np.float64)))
        if tasks_defined
        else None
    )
    nonnegative_tasks = sum(value >= 0.0 for value in finite_task_spearman)

    global_pass = global_spearman >= 0.45 and permutation_p <= 0.05
    family_spearman = [family_records[name]["spearman"] for name in MULTI_CHECKPOINT_FAMILIES]
    family_pass = (
        pooled["pearson"] is not None
        and pooled["pearson"] >= 0.30
        and pooled["spearman"] is not None
        and pooled["spearman"] >= 0.30
        and all(value is not None and value > 0.0 for value in family_spearman)
        and sum(value is not None and value >= 0.30 for value in family_spearman) >= 2
    )
    recipe_pass = all(record["pass"] for record in recipe.values())
    task_pass = (
        tasks_defined
        and task_median is not None
        and task_median > 0.0
        and nonnegative_tasks >= 6
    )
    gates = {
        "global": global_pass,
        "family": family_pass,
        "recipe_direction": recipe_pass,
        "per_task": task_pass,
    }
    payload = {
        "schema_version": ASSOCIATION_SCHEMA,
        "source": {
            "endpoint_root": str(endpoint_root),
            "summary_file_sha256": sha256_file(endpoint_root / "summary.json"),
            "rows_file_sha256": sha256_file(endpoint_root / "rows.json"),
            "run_contract_file_sha256": sha256_file(
                endpoint_root / "run_contract.json"
            ),
            "run_contract_payload_sha256": canonical_hash(contract),
            "preregistration": str(preregistration),
            "preregistration_file_sha256": sha256_file(preregistration),
        },
        "contract": {
            "panel_payload_sha256": SEALED_PANEL_PAYLOAD_SHA256,
            "primary_metric": PRIMARY_METRIC,
            "permutation_repetitions": PERMUTATION_REPETITIONS,
            "permutation_seed_label": PERMUTATION_SEED_LABEL,
            "permutation_seed": str(_permutation_seed()),
            "candidate_count": len(candidates),
            "multi_checkpoint_families": list(MULTI_CHECKPOINT_FAMILIES),
        },
        "candidates": [
            {
                "candidate_id": row["candidate_id"],
                "family": row["family"],
                "checkpoint_cursor": row["checkpoint_cursor"],
                "primary_quality": row["aggregate"][PRIMARY_METRIC]["quality"],
                "correct400": row["correct400"],
            }
            for row in candidates
        ],
        "global": {
            "spearman": global_spearman,
            "permutation_p_two_sided": permutation_p,
            "permutation_exceedances": permutation_exceedances,
        },
        "within_family": {
            "pooled_demeaned": pooled,
            "families": family_records,
        },
        "recipe_direction": recipe,
        "per_task": {
            "tasks": task_records,
            "median_spearman": task_median,
            "nonnegative_task_count": nonnegative_tasks,
            "all_defined": tasks_defined,
        },
        "gates": {**gates, "all": all(gates.values())},
    }
    payload["canonical_payload_sha256"] = canonical_hash(payload)
    return payload


def write_endpoint_association(
    endpoint_root: Path,
    preregistration: Path,
    output: Path,
) -> dict[str, Any]:
    payload = analyze_endpoint_association(endpoint_root, preregistration)
    write_json_atomic(output.resolve(), payload)
    return payload
