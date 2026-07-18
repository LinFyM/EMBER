"""Fail-closed contract for the one bounded Gate 0 target-support audit."""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

from ember.gate_zero_contract import load_gate_zero_contract


EXPECTED_NAME = "smolvla_libero90_gate_zero_target_support_audit_v1"
EXPECTED_STATUS = "predeclared_after_locked_gate_zero_failure_before_target_support_outcomes"
EXPECTED_VARIANTS = ["last_two_qv_r8", "all_expert_qv_r8", "official_default_r8"]
EXPECTED_PRIOR_REPORT_SHA256 = (
    "b7fcfc6227ba7fd6fc2e9ad21b2e55978b54d668476c9c520e216536739e9d91"
)
EXPECTED_PRIOR_GRANT_SHA256 = (
    "313ecf738b1a69ef2934c33e0681d3cef5f83506cc28925f06b8e9e16239bfad"
)


class GateZeroTargetSupportContractError(RuntimeError):
    """Raised when target-support recovery changes its frozen scientific scope."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GateZeroTargetSupportContractError(
            f"{label} changed: {actual!r} != {expected!r}"
        )


def _require_hash(authority: dict[str, Any], key: str, path: Path, label: str) -> None:
    expected = authority.get(key)
    if not isinstance(expected, str) or _sha256(path) != expected:
        raise GateZeroTargetSupportContractError(f"{label} SHA256 changed")


def _expert_qv_targets() -> list[str]:
    return [
        f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.{projection}_proj"
        for layer in range(16)
        for projection in ("q", "v")
    ]


def _validate_authority(
    spec: dict[str, Any],
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
    prior_execution_path: Path,
) -> None:
    authority = spec.get("authority", {})
    _require_hash(authority, "gate_zero_contract_sha256", gate_zero_path, "Gate 0 contract")
    _require_hash(authority, "phase0_contract_sha256", phase0_path, "Phase 0 contract")
    _require_hash(
        authority,
        "source_competence_contract_sha256",
        competence_path,
        "source competence contract",
    )
    _require_hash(
        authority,
        "prior_oracle_execution_sha256",
        prior_execution_path,
        "prior oracle execution contract",
    )
    _require_equal(
        authority.get("prior_locked_report_sha256"),
        EXPECTED_PRIOR_REPORT_SHA256,
        "prior locked report SHA256",
    )
    _require_equal(
        authority.get("prior_selection_freeze_sha256"),
        EXPECTED_PRIOR_GRANT_SHA256,
        "prior selection grant SHA256",
    )
    _require_equal(
        authority.get("prior_locked_report_status"),
        "gate_zero_pilot_failed",
        "prior status",
    )
    _require_equal(
        authority.get("prior_locked_report_failure_class"),
        "task_local_lora_oracle_utility_not_established",
        "prior failure class",
    )
    for key in (
        "validation_numeric_access",
        "held_numeric_access",
        "locked_report_numeric_reuse_for_selection",
    ):
        _require_equal(authority.get(key), False, key)
    _require_equal(
        authority.get("old_single_target_or_rank_recovery_clause_superseded"),
        True,
        "owner support-audit override",
    )


def _validate_fit(spec: dict[str, Any], parent: dict[str, Any]) -> None:
    fit = spec.get("fit", {})
    for key, expected in (
        ("support_episode_bounds", parent["oracle"]["support_episode_bounds"]),
        ("optimizer_steps", 750),
        ("effective_batch_size", 64),
        ("micro_batch_size", 64),
        ("gradient_accumulation_steps", 1),
        ("candidate_steps", [0, 25, 50, 100, 150, 250, 500, 750]),
        ("retain_scientific_candidate_records", True),
        ("retain_selected_trainable_state", True),
        ("cleanup_recovery_state_after_completed_selection", True),
    ):
        _require_equal(fit.get(key), expected, f"fit {key}")
    _require_equal(
        fit.get("acquisition_recovery"),
        "one_lower_learning_rate_with_dense_early_query_candidates",
        "acquisition recovery",
    )

    expert = _expert_qv_targets()
    expected_targets = {
        "last_two_qv_r8": expert[-4:],
        "all_expert_qv_r8": expert,
        "official_default_r8": sorted(
            expert
            + [
                "model.state_proj",
                "model.action_in_proj",
                "model.action_out_proj",
                "model.action_time_mlp_in",
                "model.action_time_mlp_out",
            ]
        ),
    }
    expected_parameters = {
        "last_two_qv_r8": 40320,
        "all_expert_qv_r8": 322560,
        "official_default_r8": 371328,
    }
    for variant in EXPECTED_VARIANTS:
        candidate = fit.get(variant, {})
        _require_equal(candidate.get("adaptation_kind"), "lora", f"{variant} adaptation")
        _require_equal(candidate.get("rank"), 8, f"{variant} rank")
        _require_equal(candidate.get("alpha"), 8, f"{variant} alpha")
        _require_equal(candidate.get("dropout"), 0.0, f"{variant} dropout")
        _require_equal(candidate.get("init_lora_weights"), True, f"{variant} initialization")
        _require_equal(candidate.get("optimizer"), "adamw", f"{variant} optimizer")
        _require_equal(candidate.get("learning_rate"), 0.0001, f"{variant} learning rate")
        _require_equal(candidate.get("seed"), 2026071820, f"{variant} seed")
        _require_equal(
            sorted(candidate.get("target_modules", [])),
            sorted(expected_targets[variant]),
            f"{variant} exact targets",
        )
        _require_equal(
            candidate.get("expected_trainable_parameters"),
            expected_parameters[variant],
            f"{variant} parameter count",
        )
    _require_equal(
        sorted(fit["last_two_qv_r8"]["target_modules"]),
        sorted(parent["oracle"]["target_modules"]),
        "pilot target provenance",
    )


def _validate_selection_and_rollouts(spec: dict[str, Any], parent: dict[str, Any]) -> None:
    selection = spec.get("selection", {})
    parent_selection = parent["oracle"]["selection"]
    for key, expected in (
        ("query_episode_bounds", parent_selection["episode_bounds"]),
        ("candidate_rule", parent_selection["candidate_rule"]),
        ("drift_proxy_max", parent_selection["action_drift_proxy_max"]),
        ("selection_uses_locked_report", False),
        ("support_screening_requires_nonzero_selected_step", True),
    ):
        _require_equal(selection.get(key), expected, f"selection {key}")
    _require_equal(
        selection.get("support_screening_rule"),
        "max_positive_query_tasks_then_max_median_query_reduction_then_min_trainable_parameters",
        "support screening rule",
    )

    reserved = set(parent["report"]["recovery_init_state_indices"])
    screening = spec.get("screening_rollout", {})
    confirmation = spec.get("confirmation_rollout", {})
    _require_equal(
        screening.get("init_state_indices"),
        list(range(24, 32)),
        "screening init states",
    )
    _require_equal(
        confirmation.get("init_state_indices"), list(range(32, 40)), "confirmation init states"
    )
    if not set(screening["init_state_indices"]) <= reserved or not set(
        confirmation["init_state_indices"]
    ) <= reserved:
        raise GateZeroTargetSupportContractError("rollout init states left the reserved surface")
    if set(screening["init_state_indices"]) & set(confirmation["init_state_indices"]):
        raise GateZeroTargetSupportContractError("screening/confirmation init states overlap")
    _require_equal(
        screening.get("locked_report_demos_accessed"),
        False,
        "screening report access",
    )
    _require_equal(
        confirmation.get("selection_changes_after_access_forbidden"),
        True,
        "post-confirmation selection lock",
    )

    escalation = spec.get("rank_escalation", {})
    for key, expected in (
        ("conditional_only", True),
        ("rank", 16),
        ("alpha", 16),
        ("dropout", 0.0),
        ("maximum_additional_supports", 1),
        ("no_other_rank_or_support_search", True),
    ):
        _require_equal(escalation.get(key), expected, f"rank escalation {key}")


def _validate_decision_and_resources(spec: dict[str, Any], parent: dict[str, Any]) -> None:
    decision = spec.get("decision", {})
    thresholds = parent["thresholds"]
    for key, expected in (
        ("median_success_gain_pp_min", thresholds["median_success_gain_pp"]),
        (
            "median_locked_action_loss_reduction_fraction_min",
            thresholds["median_locked_action_loss_reduction_fraction"],
        ),
        ("positive_task_fraction_min", thresholds["positive_task_fraction"]),
        ("median_selection_drift_proxy_max", thresholds["median_action_kl_proxy_max"]),
        ("two_task_positive_count_required", 2),
        ("writer_authorized_by_audit_alone", False),
        ("final_target_seal_requires_confirmation", True),
    ):
        _require_equal(decision.get(key), expected, f"decision {key}")
    _require_equal(spec.get("parallel", {}).get("maximum_concurrent_gpus"), 4, "GPU ceiling")
    _require_equal(
        spec.get("resources", {}).get("minimum_free_memory_mib"),
        parent["resources"]["minimum_free_memory_mib"],
        "memory headroom",
    )


def load_target_support_audit_spec(
    path: Path,
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
    prior_execution_path: Path,
) -> dict[str, Any]:
    """Load and validate the single owner-authorized support recovery."""

    try:
        with path.open("rb") as handle:
            spec = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroTargetSupportContractError("invalid target-support TOML") from error
    _require_equal(spec.get("schema_version"), 1, "schema version")
    _require_equal(spec.get("name"), EXPECTED_NAME, "contract name")
    _require_equal(spec.get("status"), EXPECTED_STATUS, "predeclaration status")
    _require_equal(spec.get("variants"), EXPECTED_VARIANTS, "support variants")
    parent = load_gate_zero_contract(gate_zero_path, phase0_path)
    _require_equal(spec.get("task_ids"), parent["data"]["task_ids"], "audit tasks")
    _validate_authority(
        spec,
        gate_zero_path=gate_zero_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
        prior_execution_path=prior_execution_path,
    )
    _validate_fit(spec, parent)
    _validate_selection_and_rollouts(spec, parent)
    _validate_decision_and_resources(spec, parent)
    return spec
