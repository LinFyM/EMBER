"""Ceiling-aware source closed-loop contract for the mature LoRA candidate."""

from __future__ import annotations

import hashlib
import math
import tomllib
from pathlib import Path
from typing import Any, Mapping

from safetensors.torch import load_file

from ember.gate_zero_support.mature_lora_lr_contract import (
    EXPECTED_VARIANT,
    load_mature_lora_lr_recovery_spec,
)
from ember.gate_zero_oracle_artifacts import (
    RECOVERY_MANIFEST,
    TRAINABLE_STATE,
    sha256_file,
    validate_candidate_artifact,
    validate_recovery_artifact,
)


EXPECTED_NAME = "smolvla_libero90_gate_zero_mature_lora_headroom_screen_v1"
EXPECTED_STATUS = "pending_owner_gate_design_decision_after_zero_outcome_mechanical_failure"
EXPECTED_STAGE = "mature_lora_headroom_control"
SELECTED_STEP = 1_000


class GateZeroMatureLoraHeadroomContractError(RuntimeError):
    """Raised when the ceiling-aware closed-loop contract changes."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GateZeroMatureLoraHeadroomContractError(
            f"{label} changed: {actual!r} != {expected!r}"
        )


def _require_hash(authority: Mapping[str, Any], key: str, path: Path, label: str) -> None:
    value = authority.get(key)
    if not isinstance(value, str) or len(value) != 64 or _sha256(path) != value:
        raise GateZeroMatureLoraHeadroomContractError(f"{label} SHA256 changed")


def _load_toml(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroMatureLoraHeadroomContractError(f"invalid {label} TOML") from error
    if not isinstance(value, dict):
        raise GateZeroMatureLoraHeadroomContractError(f"invalid {label} TOML")
    return value


def _validate_authority(
    *,
    raw: Mapping[str, Any],
    fit_spec: Mapping[str, Any],
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
    fit_path: Path,
    ladder_path: Path,
) -> Mapping[str, Any]:
    authority = raw.get("authority", {})
    for key, authority_path, label in (
        ("gate_zero_contract_sha256", gate_zero_path, "Gate 0 contract"),
        ("phase0_contract_sha256", phase0_path, "Phase 0 contract"),
        ("source_competence_contract_sha256", competence_path, "source competence"),
        ("fit_contract_sha256", fit_path, "mature-LoRA fit contract"),
        ("fit_ladder_sha256", ladder_path, "mature-LoRA fit ladder"),
    ):
        _require_hash(authority, key, authority_path, label)
    for key in (
        "source_competence_result_relative_path",
        "source_competence_result_sha256",
        "source_base_output_relative_path",
        "source_base_checkpoint_step",
        "source_base_checkpoint_role",
        "source_base_checkpoint_manifest_sha256",
    ):
        _require_equal(
            authority.get(key), fit_spec["authority"].get(key), f"authority {key}"
        )
    for key in (
        "validation_numeric_access",
        "held_numeric_access",
        "locked_report_numeric_reuse_for_selection",
        "closed_loop_accessed_before_predeclaration",
    ):
        _require_equal(authority.get(key), False, key)
    for task_id in (3, 4):
        for kind in (
            "candidate_manifest",
            "trainable_state",
            "recovery_manifest",
            "telemetry",
        ):
            value = authority.get(f"fit_task{task_id}_{kind}_sha256")
            if not isinstance(value, str) or len(value) != 64:
                raise GateZeroMatureLoraHeadroomContractError(
                    f"task-{task_id} {kind} authority is invalid"
                )
    return authority


def _validate_fit_boundary(
    raw: Mapping[str, Any], fit_spec: Mapping[str, Any]
) -> None:
    expected_outcome = {
        "selected_step": 1_000,
        "task3_query_reduction_fraction": 0.05797786803520661,
        "task4_query_reduction_fraction": 0.04236216536177371,
        "median_query_reduction_fraction": 0.05017001669849016,
        "task3_action_drift_proxy": 0.010563420131802559,
        "task4_action_drift_proxy": 0.008580253459513187,
        "offline_success_criterion_passed": True,
        "fit_resume_to_step2000": False,
        "gate_zero_authorized": False,
        "writer_authorized": False,
    }
    _require_equal(raw.get("fit_outcome_boundary", {}), expected_outcome, "fit outcome boundary")
    repair = raw.get("headroom_repair", {})
    _require_equal(
        repair.get("repair_decided_before_new_lora_closed_loop_outcomes"),
        True,
        "result-blind headroom repair",
    )
    if "8/8" not in repair.get("reason", "") or "do not lower thresholds" not in repair.get(
        "failure_with_absent_task4_headroom", ""
    ):
        raise GateZeroMatureLoraHeadroomContractError("headroom recovery language changed")
    _require_equal(
        raw.get("fit"),
        {EXPECTED_VARIANT: fit_spec["fit"][EXPECTED_VARIANT]},
        "frozen LoRA state contract",
    )
    _require_equal(raw.get("selection"), fit_spec["selection"], "query selection")
    staged = raw.get("staged_selection", {})
    for key, expected in (
        ("selected_step", SELECTED_STEP),
        ("same_training_trajectory_required", True),
        ("candidate_and_recovery_must_both_validate", True),
        ("fit_resume_to_step2000_authorized", False),
        ("selection_changes_after_grant_forbidden", True),
    ):
        _require_equal(staged.get(key), expected, f"staged selection {key}")


def _validate_rollout_decision(raw: Mapping[str, Any]) -> None:
    expected_rollout = {
        "role": "paired_source_closed_loop_headroom_control",
        "init_state_indices": list(range(40, 48)),
        "batch_size": 8,
        "seed_start": 5800,
        "warmup_seed_start": 5792,
        "policy_rng_seed": 2026071836,
        "conditions": ["frozen_base", EXPECTED_VARIANT],
        "maintenance_task_id": 3,
        "improvement_task_id": 4,
        "locked_report_demos_accessed": False,
        "selection_changes_after_access_forbidden": True,
    }
    _require_equal(raw.get("screening_rollout", {}), expected_rollout, "screening rollout")
    expected_recovery = {
        "failure_run_id": "gate0_mature_lora_headroom_screen_20260718_211501",
        "failure_git_commit": "5681819b22aa93d634384cdd3760e004c0e1742a",
        "failure_contract_sha256": "ba3ee431cb093170bb3c58460db076c95227ee92a3f33d0efb1fdeb3fc2f132f",
        "failure_rank0_packet_sha256": "c6fc083a7c7a11c7cbcc55b9132bbf5f79b92ff57a0532fd48c280358445463c",
        "failure_rank1_packet_sha256": "7ba077985e43bfa58daed16711c17fa43cee454174140b41e5474afd414d991e",
        "failure_telemetry_sha256": "d2b56c2863e5ca5d72231018d3068c6859117c62175481c3ecbacd747e63d479",
        "failure_message": "last warm-up seeds must precede report seeds",
        "failed_before_any_episode": True,
        "old_warmup_seed_start": 5760,
        "corrected_last_warmup_seed_start": 5792,
        "report_seed_start_unchanged": 5800,
        "repair_changes_scientific_surface": False,
    }
    _require_equal(raw.get("mechanical_recovery", {}), expected_recovery, "mechanical recovery")
    expected_owner_decision = {
        "reason": (
            "replacing the unreachable two-of-two positive-success Gate changes "
            "the scientific completion standard and has multiple defensible designs"
        ),
        "options": [
            "A: ceiling-aware task3 non-harm plus task4 success improvement",
            "B: replace task3 using base-competence-only non-ceiling source selection and retain two-of-two positive success gain",
            "C: task3 predeclared fine-grained functional metric plus task4 success improvement",
        ],
        "choice_must_precede_new_closed_loop_outcome": True,
        "validation_numeric_access": False,
        "held_numeric_access": False,
    }
    _require_equal(raw.get("owner_decision", {}), expected_owner_decision, "owner decision")
    expected_decision = {
        "maintenance_task_id": 3,
        "improvement_task_id": 4,
        "minimum_improvement_headroom_failures": 2,
        "minimum_improvement_net_wins": 2,
        "minimum_maintenance_net_wins": 0,
        "minimum_aggregate_net_wins": 2,
        "minimum_each_query_reduction_fraction": 0.02,
        "maximum_each_selection_drift_proxy": 0.02,
        "pass_authorizes_gate_zero": True,
        "pass_seals_final_writer_target_contract": True,
        "pass_authorizes_direct_writer": True,
        "failure_is_not_a_final_ember_negative": True,
    }
    _require_equal(raw.get("decision", {}), expected_decision, "headroom decision")
    _require_equal(raw.get("parallel", {}).get("maximum_concurrent_gpus"), 4, "GPU ceiling")
    _require_equal(raw.get("resources", {}).get("minimum_free_memory_mib"), 10_240, "OOM headroom")
    _require_equal(
        raw.get("resources", {}).get("retain_one_video_per_rollout_arm"),
        True,
        "rollout video retention",
    )


def load_mature_lora_headroom_spec(
    path: Path,
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
) -> dict[str, Any]:
    """Load the one source-only behavioral screen authorized by the 1k ladder."""

    raw = _load_toml(path, "mature-LoRA headroom screen")
    for key, expected in (
        ("schema_version", 1),
        ("name", EXPECTED_NAME),
        ("status", EXPECTED_STATUS),
        ("screening_stage", EXPECTED_STAGE),
        ("task_ids", [3, 4]),
        ("variants", [EXPECTED_VARIANT]),
        ("writer_authorized_before_closed_loop", False),
        ("owner_decision_required", True),
        ("screening_rollout_authorized", False),
    ):
        _require_equal(raw.get(key), expected, key)

    fit_path = gate_zero_path.with_name("gate_zero_mature_lora_lr_recovery.toml")
    ladder_path = gate_zero_path.with_name(
        "gate_zero_mature_lora_lr_recovery_ladder.toml"
    )
    fit_spec = load_mature_lora_lr_recovery_spec(
        fit_path,
        gate_zero_path=gate_zero_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
    )
    _validate_authority(
        raw=raw,
        fit_spec=fit_spec,
        gate_zero_path=gate_zero_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
        fit_path=fit_path,
        ladder_path=ladder_path,
    )
    _validate_fit_boundary(raw, fit_spec)
    _validate_rollout_decision(raw)
    return raw


def _path_has_tail(path: Path, expected_tail: Path) -> bool:
    return tuple(path.parts[-len(expected_tail.parts) :]) == expected_tail.parts


def collect_staged_fit_evidence(
    *,
    spec: Mapping[str, Any],
    fit_outputs: Mapping[tuple[str, int], Path],
) -> dict[str, dict[str, Any]]:
    """Validate the immutable step-1000 candidates without fabricating a final fit."""

    required = {(EXPECTED_VARIANT, task_id) for task_id in (3, 4)}
    if set(fit_outputs) != required:
        raise GateZeroMatureLoraHeadroomContractError(
            "headroom grant requires exactly both staged task outputs"
        )
    authority = spec["authority"]
    fit_tail = Path(authority["fit_root_relative_path"])
    evidence: dict[str, dict[str, Any]] = {}
    for task_id in (3, 4):
        output = Path(fit_outputs[(EXPECTED_VARIANT, task_id)]).resolve()
        expected_output_tail = fit_tail / f"{EXPECTED_VARIANT}_task{task_id}"
        if not output.is_dir() or not _path_has_tail(output, expected_output_tail):
            raise GateZeroMatureLoraHeadroomContractError(
                f"task-{task_id} staged output authority changed"
            )
        candidate_dir = output / "candidates" / f"{SELECTED_STEP:06d}"
        recovery_dir = output / "recovery" / f"{SELECTED_STEP:06d}"
        candidate = validate_candidate_artifact(
            candidate_dir,
            expected={
                "variant": EXPECTED_VARIANT,
                "task_id": task_id,
                "step": SELECTED_STEP,
            },
        )
        recovery = validate_recovery_artifact(
            recovery_dir,
            expected={
                "variant": EXPECTED_VARIANT,
                "task_id": task_id,
                "step": SELECTED_STEP,
            },
        )
        if (output / "recovery" / "last").resolve(strict=True) != recovery_dir:
            raise GateZeroMatureLoraHeadroomContractError(
                f"task-{task_id} recovery pointer changed"
            )
        paths = {
            "candidate_manifest": candidate_dir / "candidate_manifest.json",
            "trainable_state": candidate_dir / TRAINABLE_STATE,
            "recovery_manifest": recovery_dir / RECOVERY_MANIFEST,
            "telemetry": output / authority[f"fit_task{task_id}_telemetry_filename"],
        }
        for kind, artifact in paths.items():
            if (
                not artifact.is_file()
                or sha256_file(artifact)
                != authority[f"fit_task{task_id}_{kind}_sha256"]
            ):
                raise GateZeroMatureLoraHeadroomContractError(
                    f"task-{task_id} {kind} authority changed"
                )
        expected_authorities = {
            "execution_contract_sha256": authority["fit_contract_sha256"],
            "gate_zero_contract_sha256": authority["gate_zero_contract_sha256"],
            "phase0_contract_sha256": authority["phase0_contract_sha256"],
            "source_competence_contract_sha256": authority[
                "source_competence_contract_sha256"
            ],
            "source_competence_result_sha256": authority[
                "source_competence_result_sha256"
            ],
            "source_base_checkpoint_manifest_sha256": authority[
                "source_base_checkpoint_manifest_sha256"
            ],
            "validation_numeric_access": False,
            "held_numeric_access": False,
        }
        for key, value in expected_authorities.items():
            _require_equal(candidate.get("authorities", {}).get(key), value, f"candidate {key}")
            _require_equal(recovery.get("authorities", {}).get(key), value, f"recovery {key}")
        _require_equal(candidate.get("trainable_parameters"), 1_485_312, "LoRA parameters")
        state_hash = candidate["files"][TRAINABLE_STATE]["sha256"]
        _require_equal(
            state_hash,
            authority[f"fit_task{task_id}_trainable_state_sha256"],
            "candidate state hash",
        )
        evidence[f"{EXPECTED_VARIANT}:task{task_id}"] = {
            "fit_output_name": output.name,
            "selection_source": "validated_staged_step1000_candidate",
            "selected_step": SELECTED_STEP,
            "selected_manifest_sha256": sha256_file(paths["candidate_manifest"]),
            "selected_trainable_state_sha256": state_hash,
            "selected_trainable_parameters": candidate["trainable_parameters"],
            "selected_query_metrics": candidate["metrics"],
            "recovery_manifest_sha256": sha256_file(paths["recovery_manifest"]),
        }
    return evidence


def load_staged_candidate_state(
    *,
    output: Path,
    task_id: int,
    grant: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the grant-bound step-1000 state for the existing rollout runtime."""

    candidate_dir = output / "candidates" / f"{SELECTED_STEP:06d}"
    candidate = validate_candidate_artifact(
        candidate_dir,
        expected={"variant": EXPECTED_VARIANT, "task_id": task_id, "step": SELECTED_STEP},
    )
    expected = grant["fit_evidence"][f"{EXPECTED_VARIANT}:task{task_id}"]
    manifest_hash = sha256_file(candidate_dir / "candidate_manifest.json")
    state_hash = candidate["files"][TRAINABLE_STATE]["sha256"]
    if (
        manifest_hash != expected["selected_manifest_sha256"]
        or state_hash != expected["selected_trainable_state_sha256"]
    ):
        raise GateZeroMatureLoraHeadroomContractError(
            "staged candidate differs from screening grant"
        )
    selected = {
        "selected_step": SELECTED_STEP,
        "trainable_state_sha256": state_hash,
    }
    return load_file(candidate_dir / TRAINABLE_STATE), selected


def _query_metrics(
    grant: Mapping[str, Any], *, variant: str, task_id: int
) -> tuple[float, float]:
    metrics = grant.get("fit_evidence", {}).get(f"{variant}:task{task_id}", {}).get(
        "selected_query_metrics", {}
    )
    base = metrics.get("base_query_flow_mse")
    query = metrics.get("query_flow_mse")
    drift = metrics.get("action_drift_proxy")
    if (
        not all(isinstance(value, (int, float)) for value in (base, query, drift))
        or not all(math.isfinite(float(value)) for value in (base, query, drift))
        or float(base) <= 0
        or float(query) < 0
        or float(drift) < 0
    ):
        raise GateZeroMatureLoraHeadroomContractError("invalid grant query metrics")
    return (float(base) - float(query)) / float(base), float(drift)


def decide_mature_lora_headroom(
    *,
    arms: list[dict[str, Any]],
    grant: Mapping[str, Any],
    variant: str,
    parameter_count: int,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Require paired task-4 gains while retaining ceiling task 3."""

    if variant != EXPECTED_VARIANT or parameter_count != 1_485_312:
        raise GateZeroMatureLoraHeadroomContractError("headroom variant changed")
    expected_keys = {(task_id, condition) for task_id in (3, 4) for condition in ("frozen_base", variant)}
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for arm in arms:
        key = (arm.get("task_id"), arm.get("condition"))
        successes = arm.get("successes")
        if (
            key not in expected_keys
            or key in by_key
            or arm.get("mechanics_valid") is not True
            or not isinstance(successes, list)
            or len(successes) != 8
            or any(not isinstance(value, bool) for value in successes)
            or arm.get("official_rollout_init_state_indices") != list(range(40, 48))
            or arm.get("seeds") != list(range(5800, 5808))
        ):
            raise GateZeroMatureLoraHeadroomContractError("invalid paired rollout arm")
        by_key[key] = arm
    if set(by_key) != expected_keys:
        raise GateZeroMatureLoraHeadroomContractError("paired rollout arms are incomplete")

    task_metrics: dict[str, dict[str, Any]] = {}
    aggregate_net_wins = 0
    query_checks = []
    drift_checks = []
    for task_id in (3, 4):
        base = by_key[(task_id, "frozen_base")]["successes"]
        own = by_key[(task_id, variant)]["successes"]
        paired_wins = sum((not base_value) and own_value for base_value, own_value in zip(base, own, strict=True))
        paired_losses = sum(base_value and (not own_value) for base_value, own_value in zip(base, own, strict=True))
        net = paired_wins - paired_losses
        reduction, drift = _query_metrics(grant, variant=variant, task_id=task_id)
        aggregate_net_wins += net
        query_checks.append(reduction >= thresholds["minimum_each_query_reduction_fraction"])
        drift_checks.append(drift <= thresholds["maximum_each_selection_drift_proxy"])
        task_metrics[str(task_id)] = {
            "base_successes": sum(base),
            "own_successes": sum(own),
            "paired_wins": paired_wins,
            "paired_losses": paired_losses,
            "paired_net_wins": net,
            "success_gain_pp": 100.0 * net / len(base),
            "query_loss_reduction_fraction": reduction,
            "selection_drift_proxy": drift,
        }

    maintenance = str(thresholds["maintenance_task_id"])
    improvement = str(thresholds["improvement_task_id"])
    improvement_headroom = 8 - task_metrics[improvement]["base_successes"]
    checks = {
        "improvement_headroom": improvement_headroom
        >= thresholds["minimum_improvement_headroom_failures"],
        "improvement_paired_net_wins": task_metrics[improvement]["paired_net_wins"]
        >= thresholds["minimum_improvement_net_wins"],
        "maintenance_paired_net_wins": task_metrics[maintenance]["paired_net_wins"]
        >= thresholds["minimum_maintenance_net_wins"],
        "aggregate_paired_net_wins": aggregate_net_wins
        >= thresholds["minimum_aggregate_net_wins"],
        "each_query_reduction": all(query_checks),
        "each_selection_drift": all(drift_checks),
    }
    passed = all(checks.values())
    if not checks["improvement_headroom"]:
        status = "mature_lora_headroom_absent_source_recovery_required"
    elif passed:
        status = "mature_lora_headroom_control_passed"
    else:
        status = "mature_lora_headroom_control_failed_gate_recovery_required"
    candidate = {
        "variant": variant,
        "trainable_parameters": parameter_count,
        "task_metrics": task_metrics,
        "aggregate": {
            "paired_net_wins": aggregate_net_wins,
            "success_gain_pp": 100.0 * aggregate_net_wins / 16,
        },
        "threshold_checks": checks,
        "screening_passed": passed,
    }
    return {
        "status": status,
        "candidates": [candidate],
        "selected_variant": variant if passed else None,
        "confirmation_authorized": False,
        "rank16_authorized": False,
        "rank16_scope": None,
        "gate_zero_authorized": passed,
        "writer_authorized": passed,
        "final_writer_target_contract_sealed": passed,
    }
