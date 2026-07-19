"""Fail-closed Gate-0 evidence design and paired binary statistics.

This module owns the post-smoke evidence contract.  The historical n=8
trainers and their immutable packets remain provenance; they are not a second
scientific decision path.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


class GateZeroEvidenceError(RuntimeError):
    """Raised when Gate-0 evidence is too small, overlaps, or changes authority."""


def _require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GateZeroEvidenceError(f"Gate-0 evidence contract changed: {label}")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_confirmation_candidates(
    split: Mapping[str, Any], *, development_task_ids: Sequence[int]
) -> list[int]:
    """Choose one source task per non-development primitive signature, result blind."""

    source = set(split.get("active_split", {}).get("source", []))
    development = set(development_task_ids)
    tasks = split.get("tasks", [])
    if not source or not development <= source or not isinstance(tasks, list):
        raise GateZeroEvidenceError("invalid source split or development task authority")
    development_signatures = {
        row["order_signature"]
        for row in tasks
        if row.get("task_index") in development
    }
    by_signature: dict[str, list[int]] = defaultdict(list)
    for row in tasks:
        task_id = row.get("task_index")
        signature = row.get("order_signature")
        if (
            task_id in source - development
            and isinstance(signature, str)
            and signature not in development_signatures
        ):
            by_signature[signature].append(task_id)
    if not by_signature:
        raise GateZeroEvidenceError("source specification has no confirmation candidates")
    return sorted(min(task_ids) for task_ids in by_signature.values())


def deterministic_state_partition(*, task_id: int, seed: int) -> dict[str, Any]:
    """Make a result-blind 32/16/2 physical-init partition over LIBERO's 50 states."""

    if task_id < 0 or seed < 0:
        raise GateZeroEvidenceError("invalid task/partition seed")
    ordered = sorted(
        range(50),
        key=lambda index: hashlib.sha256(
            f"{seed}:{task_id}:{index}".encode("ascii")
        ).digest(),
    )
    payload = {
        "task_id": task_id,
        "seed": seed,
        "train": ordered[:32],
        "development": ordered[32:48],
        "reserve": ordered[48:],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def select_confirmation_tasks(
    spec: Mapping[str, Any],
    split: Mapping[str, Any],
    base_success_counts: Mapping[int, int],
) -> dict[str, Any]:
    """Apply the frozen base-only competence/headroom rule after the audit."""

    selection = spec["confirmation_selection"]
    candidates = selection["candidate_task_ids"]
    if set(base_success_counts) != set(candidates):
        raise GateZeroEvidenceError("base audit does not cover the frozen candidate pool")
    episodes = selection["audit_rollouts_per_task"]
    if any(
        not isinstance(value, int) or not 0 <= value <= episodes
        for value in base_success_counts.values()
    ):
        raise GateZeroEvidenceError("base audit success counts are invalid")
    eligible = [
        task_id
        for task_id in candidates
        if base_success_counts[task_id] >= selection["minimum_base_successes"]
        and episodes - base_success_counts[task_id]
        >= selection["minimum_base_failures"]
    ]
    if len(eligible) < selection["minimum_selected_tasks"]:
        raise GateZeroEvidenceError("base-only audit found too few confirmation tasks")
    ranked = sorted(
        eligible,
        key=lambda task_id: (
            abs(base_success_counts[task_id] / episodes - 0.5),
            task_id,
        ),
    )
    selected = sorted(ranked[: selection["maximum_selected_tasks"]])
    signatures = {
        int(row["task_index"]): row["order_signature"]
        for row in split["tasks"]
        if row.get("task_index") in selected
    }
    if len(signatures) != len(selected) or len(set(signatures.values())) != len(selected):
        raise GateZeroEvidenceError("selected confirmation primitives are not distinct")
    result = {
        "outcome_authority": "frozen_base_only",
        "audit_episodes_per_task": episodes,
        "base_success_counts": {
            str(task_id): base_success_counts[task_id] for task_id in candidates
        },
        "eligible_task_ids": sorted(eligible),
        "selection_rank": ranked,
        "selected_task_ids": selected,
        "selected_order_signatures": {
            str(task_id): signatures[task_id] for task_id in selected
        },
    }
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    return {**result, "sha256": hashlib.sha256(encoded).hexdigest()}


def _validate_gate_minus_one(spec: Mapping[str, Any]) -> None:
    value = spec.get("gate_minus_one_resolution", {})
    _require(value.get("status"), "passed_with_residuals", "Gate -1 status")
    _require(value.get("ordered_balanced_accuracy"), 19 / 24, "ordered accuracy")
    _require(value.get("wrong_video_specificity"), 19 / 24, "wrong-video specificity")
    _require(value.get("bidirectional_pairs_correct"), 15, "paired correct")
    _require(value.get("bidirectional_pairs_total"), 24, "paired total")
    _require(value.get("original_content_threshold"), 0.8, "original threshold")
    _require(value.get("drop_last_residual_preserved"), True, "drop-last residual")
    _require(
        value.get("additional_compute_to_reach_threshold_authorized"),
        False,
        "Gate -1 compute stop",
    )


def _validate_evaluation_contract(spec: Mapping[str, Any]) -> None:
    value = spec.get("evaluation", {})
    _require(value.get("development_task_ids"), [3, 4], "development tasks")
    _require(
        value.get("arms"),
        ["frozen_base", "supervised_lora", "zero_init_rl", "supervised_init_rl"],
        "matched arms",
    )
    _require(value.get("smoke_rollouts_per_task_arm"), 8, "smoke size")
    _require(value.get("smoke_may_select_or_reject_mechanism"), False, "smoke boundary")
    if value.get("minimum_rollouts_per_task_arm", 0) < 32:
        raise GateZeroEvidenceError("Gate-0 decision requires at least 32 rollouts/task/arm")
    if value.get("minimum_policy_rng_seeds", 0) < 2:
        raise GateZeroEvidenceError("Gate-0 decision requires multiple policy RNG seeds")
    if value.get("minimum_training_seeds", 0) < 2:
        raise GateZeroEvidenceError("Gate-0 decision requires independent training seeds")
    _require(value.get("preferred_training_seeds"), 3, "preferred training seeds")
    _require(value.get("primary_execution_horizon"), 16, "primary horizon")
    _require(value.get("deployment_robustness_horizon"), 50, "robustness horizon")
    _require(value.get("binary_success_is_primary"), True, "primary success metric")
    _require(value.get("paired_episode_rows_required"), True, "paired row retention")
    _require(value.get("bootstrap_and_exact_intervals_required"), True, "intervals")
    _require(value.get("validation_numeric_access"), False, "validation isolation")
    _require(value.get("held_numeric_access"), False, "held isolation")
    _require(value.get("locked_numeric_access"), False, "locked isolation")


def _validate_algorithm_boundary(spec: Mapping[str, Any]) -> None:
    pilot = spec.get("custom_pilot", {})
    _require(pilot.get("name"), "custom_chunk_level_flow_loss_ppo_pilot", "pilot name")
    _require(pilot.get("flow_samples_averaged_before_ratio"), 8, "pilot ratio semantics")
    if "flow_sample_group_size" in pilot:
        raise GateZeroEvidenceError("misleading pilot group-size field is forbidden")
    faithful = spec.get("faithful_fpo_plus_core", {})
    expected = {
        "required_before_ordinary_rl_negative_claim": True,
        "ratio_granularity": "per_flow_sample",
        "cfm_loss_average_group_size": 1,
        "modified_huber_matches_mse_below_delta": True,
        "huber_delta": 0.5,
        "old_cfm_loss_clamp": 4.0,
        "log_ratio_clamp": 5.0,
        "execution_horizon": 16,
        "flow_samples_per_transition": 8,
        "trust_region": "ppo",
    }
    for key, value in expected.items():
        _require(faithful.get(key), value, f"faithful_fpo_plus_core.{key}")


def validate_gate_zero_evidence_spec(
    spec: dict[str, Any], split: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the result-blind repair before any new Gate decision outcome."""

    _require(spec.get("schema_version"), 1, "schema")
    _require(
        spec.get("status"),
        "predeclared_after_n8_atomic_stop_before_new_gate_decision_outcome",
        "status",
    )
    _require(spec.get("surface"), "libero90_source_only_gate_zero_evidence_repair", "surface")
    authority = spec.get("authority", {})
    for key in (
        "split_reseal_sha256",
        "horizon_coverage_contract_sha256",
        "horizon_coverage_stage24_result_sha256",
        "video_information_result_sha256",
    ):
        if not _is_sha256(authority.get(key)):
            raise GateZeroEvidenceError(f"invalid SHA256 authority: {key}")
    for key in (
        "horizon_coverage_contract_relative_path",
        "horizon_coverage_stage24_result_relative_path",
        "video_information_result_relative_path",
    ):
        if not isinstance(authority.get(key), str) or not authority[key]:
            raise GateZeroEvidenceError(f"invalid authority path: {key}")
    for surface in ("validation", "held", "locked"):
        _require(authority.get(f"{surface}_numeric_access"), False, f"{surface} access")
    _validate_gate_minus_one(spec)
    _validate_evaluation_contract(spec)
    _validate_algorithm_boundary(spec)
    confirmation = spec.get("confirmation_selection", {})
    development = spec["evaluation"]["development_task_ids"]
    expected = canonical_confirmation_candidates(split, development_task_ids=development)
    _require(confirmation.get("candidate_task_ids"), expected, "confirmation candidates")
    if set(expected) & set(development):
        raise GateZeroEvidenceError("development and confirmation task candidates overlap")
    _require(confirmation.get("selection_uses_base_outcomes_only"), True, "selection authority")
    _require(confirmation.get("minimum_selected_tasks"), 2, "minimum confirmation tasks")
    _require(confirmation.get("maximum_selected_tasks"), 4, "maximum confirmation tasks")
    _require(confirmation.get("audit_rollouts_per_task"), 32, "difficulty audit size")
    _require(confirmation.get("minimum_base_successes"), 4, "competence floor")
    _require(confirmation.get("minimum_base_failures"), 8, "headroom floor")
    partition = spec.get("state_partition", {})
    _require(partition.get("seed"), 20260719, "partition seed")
    _require(partition.get("physical_state_count"), 50, "state count")
    _require(partition.get("train_count"), 32, "train count")
    _require(partition.get("development_count"), 16, "development count")
    _require(partition.get("reserve_count"), 2, "reserve count")
    return spec


def load_gate_zero_evidence_spec(path: Path, split_path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            spec = tomllib.load(handle)
        split = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, json.JSONDecodeError) as error:
        raise GateZeroEvidenceError("Gate-0 evidence authority is unreadable") from error
    if hashlib.sha256(split_path.read_bytes()).hexdigest() != spec.get("authority", {}).get(
        "split_reseal_sha256"
    ):
        raise GateZeroEvidenceError("split reseal hash changed")
    return validate_gate_zero_evidence_spec(spec, split)


def validate_bound_authority(
    spec: Mapping[str, Any], *, repo_root: Path, output_root: Path
) -> dict[str, str]:
    """Hash-check immutable repository and external evidence before a new run."""

    authority = spec.get("authority", {})
    bindings = (
        (
            repo_root,
            "horizon_coverage_contract_relative_path",
            "horizon_coverage_contract_sha256",
        ),
        (
            output_root,
            "horizon_coverage_stage24_result_relative_path",
            "horizon_coverage_stage24_result_sha256",
        ),
        (
            output_root,
            "video_information_result_relative_path",
            "video_information_result_sha256",
        ),
    )
    verified: dict[str, str] = {}
    for root, path_key, sha_key in bindings:
        relative = authority.get(path_key)
        expected = authority.get(sha_key)
        if not isinstance(relative, str) or not _is_sha256(expected):
            raise GateZeroEvidenceError(f"invalid bound authority: {path_key}")
        path = root / relative
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise GateZeroEvidenceError(f"bound authority is unreadable: {path_key}") from error
        if actual != expected:
            raise GateZeroEvidenceError(f"bound authority hash changed: {path_key}")
        verified[path_key] = actual
    return verified


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    return sum(
        math.comb(n, index)
        * probability**index
        * (1.0 - probability) ** (n - index)
        for index in range(k + 1)
    )


def _clopper_pearson(k: int, n: int, confidence: float = 0.95) -> list[float]:
    if not 0 <= k <= n or n <= 0 or not 0 < confidence < 1:
        raise GateZeroEvidenceError("invalid exact interval input")
    alpha = 1.0 - confidence
    if k == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            survival = 1.0 - _binomial_cdf(k - 1, n, mid)
            if survival < alpha / 2.0:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2.0
    if k == n:
        upper = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if _binomial_cdf(k, n, mid) > alpha / 2.0:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2.0
    return [lower, upper]


def paired_binary_summary(
    left: Sequence[bool],
    right: Sequence[bool],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    """Report paired net gain, paired bootstrap, and exact discordant interval."""

    if (
        len(left) != len(right)
        or len(left) < 2
        or bootstrap_replicates < 1000
        or bootstrap_seed < 0
        or any(not isinstance(value, (bool, np.bool_)) for value in (*left, *right))
    ):
        raise GateZeroEvidenceError("invalid paired binary evidence")
    differences = np.asarray(left, dtype=np.int8) - np.asarray(right, dtype=np.int8)
    wins = int((differences == 1).sum())
    losses = int((differences == -1).sum())
    discordant = wins + losses
    rng = np.random.default_rng(bootstrap_seed)
    draws = differences[
        rng.integers(0, len(differences), size=(bootstrap_replicates, len(differences)))
    ].mean(axis=1)
    return {
        "episodes": len(differences),
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": int((differences == 0).sum()),
        "net_gain_pp": float(differences.mean() * 100.0),
        "paired_bootstrap_ci95_pp": (np.quantile(draws, [0.025, 0.975]) * 100.0).tolist(),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "exact_discordant_trials": discordant,
        "exact_conditional_win_rate_ci95": (
            _clopper_pearson(wins, discordant) if discordant else [0.0, 1.0]
        ),
    }


_EVALUATION_ROW_FIELDS = {
    "surface",
    "task_id",
    "arm",
    "training_seed",
    "policy_rng_seed",
    "evaluator_seed",
    "physical_init_state_index",
    "physical_init_state_sha256",
    "execution_horizon",
    "success",
    "grasp",
    "correct_object_or_region",
    "drawer_closed",
    "time_to_success",
    "progress_fraction",
    "action_drift_to_base",
    "action_drift_to_initialization",
}


def _validate_evaluation_row(
    row: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> tuple[int, str, int, int]:
    if not _EVALUATION_ROW_FIELDS <= set(row) or row["surface"] != "development":
        raise GateZeroEvidenceError("evaluation row authority is incomplete")
    if row["task_id"] not in evaluation["development_task_ids"]:
        raise GateZeroEvidenceError("evaluation row escaped development tasks")
    if row["arm"] not in evaluation["arms"]:
        raise GateZeroEvidenceError("evaluation row has an unknown arm")
    horizons = {
        evaluation["primary_execution_horizon"],
        evaluation["deployment_robustness_horizon"],
    }
    if row["execution_horizon"] not in horizons:
        raise GateZeroEvidenceError("evaluation horizon changed")
    if not isinstance(row["success"], (bool, np.bool_)):
        raise GateZeroEvidenceError("binary success changed type")
    for name in ("grasp", "correct_object_or_region", "drawer_closed"):
        if row[name] is not None and not isinstance(row[name], (bool, np.bool_)):
            raise GateZeroEvidenceError(f"{name} diagnostic changed type")
    if not _is_sha256(row["physical_init_state_sha256"]):
        raise GateZeroEvidenceError("physical init-state hash is invalid")
    if not 0.0 <= float(row["progress_fraction"]) <= 1.0:
        raise GateZeroEvidenceError("progress fraction is invalid")
    if row["success"] and (
        row["time_to_success"] is None or float(row["time_to_success"]) < 0
    ):
        raise GateZeroEvidenceError("successful episode lacks time-to-success")
    for name in ("action_drift_to_base", "action_drift_to_initialization"):
        value = float(row[name])
        if not math.isfinite(value) or value < 0:
            raise GateZeroEvidenceError(f"{name} is invalid")
    return (
        int(row["task_id"]),
        str(row["arm"]),
        int(row["training_seed"]),
        int(row["execution_horizon"]),
    )


def _validate_episode_pairing(
    groups: Mapping[tuple[int, str, int, int], Sequence[Mapping[str, Any]]]
) -> None:
    pairing: dict[tuple[int, int, int], set[tuple[Any, ...]]] = {}
    for (task_id, _arm, training_seed, horizon), rows in groups.items():
        key = (task_id, training_seed, horizon)
        episode_keys = {
            (
                row["evaluator_seed"],
                row["policy_rng_seed"],
                row["physical_init_state_index"],
                row["physical_init_state_sha256"],
            )
            for row in rows
        }
        if len(episode_keys) != len(rows):
            raise GateZeroEvidenceError("evaluation arm repeats a paired episode")
        if key in pairing and pairing[key] != episode_keys:
            raise GateZeroEvidenceError("matched arms do not share paired episodes")
        pairing[key] = episode_keys


def validate_evaluation_records(
    records: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Reject n=8, unpaired arms, or insufficient policy/training seeds."""

    evaluation = spec["evaluation"]
    groups: dict[tuple[int, str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[_validate_evaluation_row(row, evaluation)].append(row)
    expected_groups = (
        len(evaluation["development_task_ids"])
        * len(evaluation["arms"])
        * len({row["training_seed"] for row in records})
        * 2
    )
    if len(groups) != expected_groups:
        raise GateZeroEvidenceError("evaluation arm/task/seed/horizon grid is incomplete")
    _validate_episode_pairing(groups)
    minimum_count = min(len(rows) for rows in groups.values())
    if minimum_count < evaluation["minimum_rollouts_per_task_arm"]:
        raise GateZeroEvidenceError("n=8 smoke cannot support a Gate-0 decision")
    policy_counts = [len({row["policy_rng_seed"] for row in rows}) for rows in groups.values()]
    if min(policy_counts) < evaluation["minimum_policy_rng_seeds"]:
        raise GateZeroEvidenceError("evaluation lacks multiple policy RNG seeds")
    training_seeds = {row["training_seed"] for row in records}
    if len(training_seeds) < evaluation["minimum_training_seeds"]:
        raise GateZeroEvidenceError("candidate lacks independent training-seed replication")
    return {
        "minimum_count": minimum_count,
        "training_seed_count": len(training_seeds),
        "policy_rng_seed_count_minimum": min(policy_counts),
        "primary_execution_horizon": evaluation["primary_execution_horizon"],
        "deployment_robustness_horizon": evaluation["deployment_robustness_horizon"],
    }
