"""Freeze Gate 0 oracle selection and evaluate its locked source report."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ember.gate_zero_oracle_artifacts import (
    atomic_json,
    sha256_file,
    validate_selected_artifact,
)
from ember.gate_zero_oracle_contract import load_oracle_execution_spec


FIT_RESULT = "fit_selection_result.json"
SELECTED_MANIFEST = "selected_manifest.json"
TRAINABLE_STATE = "trainable_state.safetensors"
BASE_FLOW_MATCH_RTOL = 1e-4
BASE_FLOW_MATCH_ATOL = 1e-8


class GateZeroOracleReportError(RuntimeError):
    """Raised when selection freeze or locked reporting violates its seal."""


def canonical_report_shards() -> list[list[tuple[int, str]]]:
    """Return the frozen four-shard report schedule in launch order."""

    return [
        [(3, "frozen_base"), (3, "own_adapter")],
        [(3, "swapped_adapter"), (3, "partial_upper_bound")],
        [(4, "frozen_base"), (4, "own_adapter")],
        [(4, "swapped_adapter"), (4, "partial_upper_bound")],
    ]


def assigned_report_arms(*, rank: int, world_size: int) -> list[tuple[int, str]]:
    if world_size not in {1, 2, 4} or rank < 0 or rank >= world_size:
        raise GateZeroOracleReportError("invalid locked-report topology")
    shards = canonical_report_shards()
    return [arm for index, shard in enumerate(shards) if index % world_size == rank for arm in shard]


def _load_parent(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroOracleReportError("invalid Gate 0 parent contract") from error
    if not isinstance(value, dict):
        raise GateZeroOracleReportError("invalid Gate 0 parent contract")
    return value


def _verify_checksums(output: Path) -> None:
    checksum_path = output / "checksums.sha256"
    try:
        rows = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise GateZeroOracleReportError("fit output lacks checksums") from error
    seen: set[str] = set()
    for row in rows:
        try:
            digest, relative = row.split("  ", 1)
        except ValueError as error:
            raise GateZeroOracleReportError("fit checksum row is malformed") from error
        if len(digest) != 64 or relative in seen or Path(relative).is_absolute():
            raise GateZeroOracleReportError("fit checksum authority is malformed")
        candidate = (output / relative).resolve()
        try:
            candidate.relative_to(output.resolve())
        except ValueError as error:
            raise GateZeroOracleReportError("fit checksum path escaped output") from error
        if not candidate.is_file() or sha256_file(candidate) != digest:
            raise GateZeroOracleReportError(f"fit checksum failed: {relative}")
        seen.add(relative)
    required = {
        FIT_RESULT,
        f"selected/{SELECTED_MANIFEST}",
        f"selected/{TRAINABLE_STATE}",
    }
    if not required <= seen:
        raise GateZeroOracleReportError("fit checksums omit selected evidence")


def _fit_evidence(
    output: Path,
    *,
    variant: str,
    task_id: int,
    expected_authorities: Mapping[str, str],
) -> dict[str, Any]:
    output = output.resolve()
    if not output.is_dir():
        raise GateZeroOracleReportError("frozen fit output is missing")
    _verify_checksums(output)
    selected_dir = output / "selected"
    selected = validate_selected_artifact(
        selected_dir, expected={"variant": variant, "task_id": task_id}
    )
    try:
        result = json.loads((output / FIT_RESULT).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateZeroOracleReportError("fit selection result is invalid") from error
    expected = {
        "schema_version": 1,
        "status": "oracle_fit_selection_complete_pending_global_report_grant",
        "variant": variant,
        "task_id": task_id,
        "gate_zero_authorized": False,
        "writer_authorized": False,
        "final_writer_target_contract_sealed": False,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise GateZeroOracleReportError(f"fit result changed {key}")
    authorities = result.get("authorities")
    if not isinstance(authorities, dict):
        raise GateZeroOracleReportError("fit result lacks authorities")
    for key, value in expected_authorities.items():
        if authorities.get(key) != value:
            raise GateZeroOracleReportError(f"fit authority changed {key}")
    selection = result.get("selection", {})
    selected_manifest_sha = sha256_file(selected_dir / SELECTED_MANIFEST)
    if (
        selection.get("selected_step") != selected["selected_step"]
        or selection.get("selected_trainable_state_sha256")
        != selected["trainable_state_sha256"]
        or selection.get("selected_manifest_sha256") != selected_manifest_sha
        or selection.get("locked_report_accessed") is not False
    ):
        raise GateZeroOracleReportError("fit result and selected artifact differ")
    return {
        "fit_output_name": output.name,
        "fit_result_sha256": sha256_file(output / FIT_RESULT),
        "selected_step": selected["selected_step"],
        "selected_manifest_sha256": selected_manifest_sha,
        "selected_trainable_state_sha256": selected["trainable_state_sha256"],
        "selected_trainable_parameters": selected["trainable_parameters"],
        "selected_query_metrics": selected["selected_metrics"],
    }


def create_selection_freeze_grant(
    *,
    execution_path: Path,
    parent_path: Path,
    phase0_path: Path,
    competence_path: Path,
    fit_outputs: Mapping[tuple[str, int], Path],
    grant_path: Path,
) -> dict[str, Any]:
    """Validate all primary/capacity selections before opening report data."""

    spec = load_oracle_execution_spec(
        execution_path,
        gate_zero_path=parent_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
    )
    parent = _load_parent(parent_path)
    required = {
        (variant, task_id)
        for variant in ("lora", "partial_upper_bound")
        for task_id in spec["task_ids"]
    }
    if set(fit_outputs) != required:
        raise GateZeroOracleReportError("selection freeze requires exactly four frozen fit outputs")
    if not grant_path.is_absolute() or grant_path.exists() or grant_path.parent.exists():
        raise GateZeroOracleReportError("selection-freeze output must be a fresh absolute directory")
    expected_authorities = {
        "execution_contract_sha256": sha256_file(execution_path),
        "gate_zero_contract_sha256": sha256_file(parent_path),
        "phase0_contract_sha256": sha256_file(phase0_path),
        "source_competence_contract_sha256": sha256_file(competence_path),
    }
    evidence = {
        f"{variant}:task{task_id}": _fit_evidence(
            Path(fit_outputs[(variant, task_id)]),
            variant=variant,
            task_id=task_id,
            expected_authorities=expected_authorities,
        )
        for variant, task_id in sorted(required)
    }
    lora = {
        str(task_id): evidence[f"lora:task{task_id}"]["selected_trainable_state_sha256"]
        for task_id in spec["task_ids"]
    }
    partial = {
        str(task_id): evidence[f"partial_upper_bound:task{task_id}"][
            "selected_trainable_state_sha256"
        ]
        for task_id in spec["task_ids"]
    }
    grant = {
        "schema_version": 1,
        "status": spec["selection_freeze"]["grant_status"],
        "pilot_name": parent["name"],
        "task_ids": spec["task_ids"],
        "report_access_authorized": True,
        "selection_changes_after_grant_forbidden": True,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        **expected_authorities,
        "fit_evidence": evidence,
        "selected_adapter_sha256_by_task": lora,
        "selected_capacity_upper_bound_sha256_by_task": partial,
        "gate_zero_authorized": False,
        "writer_authorized": False,
        "final_writer_target_contract_sealed": False,
        "validation_numeric_access": False,
        "held_numeric_access": False,
    }
    grant_path.parent.mkdir(parents=True, exist_ok=False)
    atomic_json(grant_path, grant)
    checksum = grant_path.parent / "checksums.sha256"
    checksum.write_text(
        f"{sha256_file(grant_path)}  {grant_path.name}\n", encoding="utf-8"
    )
    return grant


def validate_selection_freeze_grant(
    *,
    grant_path: Path,
    execution_path: Path,
    parent_path: Path,
    phase0_path: Path,
    competence_path: Path,
    fit_outputs: Mapping[tuple[str, int], Path],
) -> dict[str, Any]:
    """Revalidate the immutable grant and the selected state files it opened."""

    try:
        grant = json.loads(grant_path.read_text(encoding="utf-8"))
        checksum = (grant_path.parent / "checksums.sha256").read_text(
            encoding="utf-8"
        ).split()
    except (OSError, json.JSONDecodeError) as error:
        raise GateZeroOracleReportError("selection-freeze grant is invalid") from error
    parent = _load_parent(parent_path)
    expected = {
        "schema_version": 1,
        "status": "oracle_selection_frozen_before_report_access",
        "pilot_name": parent["name"],
        "task_ids": [3, 4],
        "report_access_authorized": True,
        "selection_changes_after_grant_forbidden": True,
        "execution_contract_sha256": sha256_file(execution_path),
        "gate_zero_contract_sha256": sha256_file(parent_path),
        "phase0_contract_sha256": sha256_file(phase0_path),
        "source_competence_contract_sha256": sha256_file(competence_path),
        "gate_zero_authorized": False,
        "writer_authorized": False,
        "final_writer_target_contract_sealed": False,
        "validation_numeric_access": False,
        "held_numeric_access": False,
    }
    for key, value in expected.items():
        if grant.get(key) != value:
            raise GateZeroOracleReportError(f"selection-freeze grant changed {key}")
    if checksum != [sha256_file(grant_path), grant_path.name]:
        raise GateZeroOracleReportError("selection-freeze grant checksum changed")
    required = {
        (variant, task_id)
        for variant in ("lora", "partial_upper_bound")
        for task_id in (3, 4)
    }
    if set(fit_outputs) != required:
        raise GateZeroOracleReportError("grant validation requires four fit outputs")
    for variant, task_id in required:
        selected = validate_selected_artifact(
            Path(fit_outputs[(variant, task_id)]) / "selected",
            expected={"variant": variant, "task_id": task_id},
        )
        key = (
            "selected_adapter_sha256_by_task"
            if variant == "lora"
            else "selected_capacity_upper_bound_sha256_by_task"
        )
        if grant.get(key, {}).get(str(task_id)) != selected["trainable_state_sha256"]:
            raise GateZeroOracleReportError("grant selected-state hash changed")
    return grant


def _positive_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value > 0


def _validate_arm(arm: dict[str, Any]) -> None:
    successes = arm.get("successes")
    row_digest = arm.get("offline_row_keys_sha256")
    valid = [
        arm.get("mechanics_valid") is True,
        isinstance(successes, list) and len(successes) == 8,
        isinstance(successes, list)
        and all(isinstance(value, bool) for value in successes),
        _positive_finite(arm.get("offline_flow_mse")),
        _positive_finite(arm.get("base_offline_flow_mse")),
        isinstance(arm.get("offline_sample_count"), int)
        and arm["offline_sample_count"] > 0,
        isinstance(row_digest, str) and len(row_digest) == 64,
        arm.get("official_rollout_init_state_indices") == list(range(16, 24)),
        arm.get("seeds") == list(range(5400, 5408)),
    ]
    if not all(valid):
        raise GateZeroOracleReportError("locked report arm is mechanically invalid")


def _validate_task_matching(
    mapping: Mapping[tuple[int, str], dict[str, Any]], task_id: int
) -> None:
    conditions = (
        "frozen_base",
        "own_adapter",
        "swapped_adapter",
        "partial_upper_bound",
    )
    task_arms = [mapping[(task_id, condition)] for condition in conditions]
    if len({arm["offline_row_keys_sha256"] for arm in task_arms}) != 1:
        raise GateZeroOracleReportError("locked report rows differ across arms")
    if len({arm["offline_sample_count"] for arm in task_arms}) != 1:
        raise GateZeroOracleReportError("locked report sample counts differ across arms")
    base_values = [float(arm["base_offline_flow_mse"]) for arm in task_arms]
    if any(
        not math.isclose(
            value,
            base_values[0],
            rel_tol=BASE_FLOW_MATCH_RTOL,
            abs_tol=BASE_FLOW_MATCH_ATOL,
        )
        for value in base_values[1:]
    ):
        raise GateZeroOracleReportError("locked report base loss differs across arms")


def _arm_map(arms: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    expected = {arm for shard in canonical_report_shards() for arm in shard}
    mapping: dict[tuple[int, str], dict[str, Any]] = {}
    for arm in arms:
        key = (arm.get("task_id"), arm.get("condition"))
        if key in mapping or key not in expected:
            raise GateZeroOracleReportError("locked report arm set changed")
        _validate_arm(arm)
        mapping[key] = arm
    if set(mapping) != expected:
        raise GateZeroOracleReportError("locked report arms are incomplete")
    for task_id in (3, 4):
        _validate_task_matching(mapping, task_id)
    return mapping


def decide_gate_zero_report(
    *,
    arms: list[dict[str, Any]],
    selected_lora_drift: Mapping[int, float],
    thresholds: Mapping[str, float | int],
) -> dict[str, Any]:
    """Apply the frozen two-task Gate 0 pilot decision without Writer authority."""

    mapping = _arm_map(arms)
    if set(selected_lora_drift) != {3, 4} or any(
        not math.isfinite(float(value)) or value < 0 for value in selected_lora_drift.values()
    ):
        raise GateZeroOracleReportError("selected LoRA drift evidence is invalid")
    task_rows: dict[str, dict[str, float | int]] = {}
    for task_id in (3, 4):
        base = mapping[(task_id, "frozen_base")]
        own = mapping[(task_id, "own_adapter")]
        swapped = mapping[(task_id, "swapped_adapter")]
        partial = mapping[(task_id, "partial_upper_bound")]
        base_success = sum(base["successes"])
        own_success = sum(own["successes"])
        base_flow = float(base["offline_flow_mse"])
        own_flow = float(own["offline_flow_mse"])
        task_rows[str(task_id)] = {
            "base_successes": base_success,
            "own_successes": own_success,
            "swapped_successes": sum(swapped["successes"]),
            "partial_upper_bound_successes": sum(partial["successes"]),
            "success_gain_pp": (own_success - base_success) * 100.0 / 8.0,
            "locked_action_loss_reduction_fraction": (base_flow - own_flow) / base_flow,
            "selected_action_drift_proxy": float(selected_lora_drift[task_id]),
        }
    rows = list(task_rows.values())
    median_success = statistics.median(float(row["success_gain_pp"]) for row in rows)
    median_loss = statistics.median(
        float(row["locked_action_loss_reduction_fraction"]) for row in rows
    )
    median_drift = statistics.median(
        float(row["selected_action_drift_proxy"]) for row in rows
    )
    positive_count = sum(float(row["success_gain_pp"]) > 0 for row in rows)
    positive_fraction = positive_count / len(rows)
    checks = {
        "median_success_gain": median_success
        >= float(thresholds["median_success_gain_pp_min"]),
        "median_locked_action_loss_reduction": median_loss
        >= float(thresholds["median_locked_action_loss_reduction_fraction_min"]),
        "positive_task_fraction": positive_fraction
        >= float(thresholds["positive_task_fraction_min"]),
        "two_task_positive_count": positive_count
        >= int(thresholds["two_task_positive_count_required"]),
        "selection_drift": median_drift
        <= float(thresholds["median_selection_drift_proxy_max"]),
    }
    passed = all(checks.values())
    partial_only = not passed and all(
        int(row["partial_upper_bound_successes"]) > int(row["base_successes"])
        for row in rows
    )
    return {
        "status": "gate_zero_pilot_passed" if passed else "gate_zero_pilot_failed",
        "gate_zero_pilot_passed": passed,
        "gate_zero_authorized": False,
        "writer_authorized": False,
        "final_writer_target_contract_sealed": False,
        "failure_class": (
            None
            if passed
            else (
                "primary_lora_contract_too_narrow_trigger_bounded_recovery"
                if partial_only
                else "task_local_lora_oracle_utility_not_established"
            )
        ),
        "task_metrics": task_rows,
        "aggregate": {
            "median_success_gain_pp": median_success,
            "median_locked_action_loss_reduction_fraction": median_loss,
            "positive_task_count": positive_count,
            "positive_task_fraction": positive_fraction,
            "median_selection_drift_proxy": median_drift,
        },
        "threshold_checks": checks,
        "mechanics_tolerances": {
            "base_offline_flow_mse_rtol": BASE_FLOW_MATCH_RTOL,
            "base_offline_flow_mse_atol": BASE_FLOW_MATCH_ATOL,
        },
        "partial_upper_bound_is_non_matched_diagnostic": True,
    }


def _freeze_cli(arguments: argparse.Namespace) -> dict[str, Any]:
    outputs = {
        ("lora", 3): arguments.lora_task3,
        ("lora", 4): arguments.lora_task4,
        ("partial_upper_bound", 3): arguments.partial_task3,
        ("partial_upper_bound", 4): arguments.partial_task4,
    }
    return create_selection_freeze_grant(
        execution_path=arguments.config,
        parent_path=arguments.gate_zero_contract,
        phase0_path=arguments.phase0_contract,
        competence_path=arguments.source_competence_contract,
        fit_outputs=outputs,
        grant_path=arguments.grant_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="freeze all four query-selected states")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--gate-zero-contract", type=Path, required=True)
    freeze.add_argument("--phase0-contract", type=Path, required=True)
    freeze.add_argument("--source-competence-contract", type=Path, required=True)
    freeze.add_argument("--lora-task3", type=Path, required=True)
    freeze.add_argument("--lora-task4", type=Path, required=True)
    freeze.add_argument("--partial-task3", type=Path, required=True)
    freeze.add_argument("--partial-task4", type=Path, required=True)
    freeze.add_argument("--grant-path", type=Path, required=True)
    arguments = parser.parse_args()
    result = _freeze_cli(arguments)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
