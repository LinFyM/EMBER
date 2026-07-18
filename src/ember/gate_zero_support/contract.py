"""Fail-closed contract for the one bounded Gate 0 target-support audit."""

from __future__ import annotations

import copy
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
EXPECTED_RANK8_AUDIT_SHA256 = (
    "aca38905fd84894d7576a5c4b0e863ea6c245ec02ec1d0473a52dc8b8b027da0"
)
EXPECTED_RANK8_SCREENING_GRANT_SHA256 = (
    "fd8e28a7f0b828e14ff7cfb794a047409b6e8e96562646b38aef232b65332992"
)
EXPECTED_RANK8_SCREENING_RESULT_SHA256 = (
    "0df3acb8d3fd5f94507921298940281c7430eedc359869d3918a0f2c012c6efb"
)
EXPECTED_RANK16_NAME = "smolvla_libero90_gate_zero_target_support_rank16_v1"


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


def _load_rank16_raw(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroTargetSupportContractError("invalid rank-16 support TOML") from error
    return raw


def _validate_rank16_header(raw: dict[str, Any]) -> None:
    _require_equal(raw.get("schema_version"), 1, "rank-16 schema version")
    _require_equal(raw.get("name"), EXPECTED_RANK16_NAME, "rank-16 contract name")
    _require_equal(
        raw.get("status"),
        "predeclared_after_rank8_screen_failure_before_rank16_fit_or_outcomes",
        "rank-16 predeclaration status",
    )
    _require_equal(raw.get("screening_stage"), "rank16", "rank-16 stage")
    _require_equal(raw.get("task_ids"), [3, 4], "rank-16 tasks")
    _require_equal(
        raw.get("variants"), ["official_default_r16"], "rank-16 variants"
    )


def _validate_rank16_authority(
    authority: dict[str, Any],
    base: dict[str, Any],
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
    prior_execution_path: Path,
    rank8_audit_path: Path,
) -> None:
    _require_hash(
        authority, "gate_zero_contract_sha256", gate_zero_path, "Gate 0 contract"
    )
    _require_hash(
        authority, "phase0_contract_sha256", phase0_path, "Phase 0 contract"
    )
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
    _require_hash(
        authority,
        "rank8_audit_contract_sha256",
        rank8_audit_path,
        "rank-8 audit contract",
    )
    for key, expected in (
        ("rank8_audit_contract_sha256", EXPECTED_RANK8_AUDIT_SHA256),
        ("rank8_screening_grant_sha256", EXPECTED_RANK8_SCREENING_GRANT_SHA256),
        ("rank8_screening_result_sha256", EXPECTED_RANK8_SCREENING_RESULT_SHA256),
        ("rank8_screening_status", "rank8_support_screen_failed_rank16_authorized"),
        ("rank8_authorized_scope", "official_default_r8"),
        (
            "rank8_screening_result_relative_path",
            "gate_zero/target_support_audit/screening/"
            "rank8_recovery1_20260718T151445Z/support_screening_result.json",
        ),
        ("validation_numeric_access", False),
        ("held_numeric_access", False),
        ("locked_report_numeric_reuse_for_selection", False),
    ):
        _require_equal(authority.get(key), expected, f"rank-16 authority {key}")
    for key in (
        "source_competence_result_relative_path",
        "source_competence_result_sha256",
        "source_base_output_relative_path",
        "source_base_checkpoint_step",
        "source_base_checkpoint_role",
        "source_base_checkpoint_manifest_sha256",
    ):
        _require_equal(authority.get(key), base["authority"][key], f"rank-16 {key}")


def _validate_rank16_fit_and_resources(
    raw: dict[str, Any], base: dict[str, Any]
) -> None:
    override = raw.get("fit_override", {})
    for key, expected in (
        ("source_variant", "official_default_r8"),
        ("variant", "official_default_r16"),
        ("rank", 16),
        ("alpha", 16),
        ("dropout", 0.0),
        ("expected_trainable_parameters", 742656),
        ("optimizer_steps", 750),
        ("learning_rate", 0.0001),
        ("same_targets_optimizer_sampler_support_query_and_candidates_as_rank8", True),
    ):
        _require_equal(override.get(key), expected, f"rank-16 fit override {key}")
    escalation = raw.get("rank_escalation", {})
    for key in (
        "conditional_only",
        "no_further_support_or_rank_search",
        "failure_requires_gate_recovery_decision",
    ):
        _require_equal(escalation.get(key), True, f"rank-16 escalation {key}")
    _require_equal(
        raw.get("parallel", {}).get("fit_jobs"),
        ["official_default_r16:3", "official_default_r16:4"],
        "rank-16 fit jobs",
    )
    _require_equal(
        raw.get("parallel", {}).get("maximum_concurrent_gpus"),
        4,
        "rank-16 GPU ceiling",
    )
    _require_equal(
        raw.get("resources", {}).get("minimum_free_memory_mib"),
        base["resources"]["minimum_free_memory_mib"],
        "rank-16 memory headroom",
    )


def _validate_rank16_rollouts(raw: dict[str, Any]) -> None:
    screening = raw.get("screening_rollout", {})
    confirmation = raw.get("confirmation_rollout", {})
    for value, expected, label in (
        (
            screening.get("init_state_indices"),
            list(range(32, 40)),
            "screening states",
        ),
        (screening.get("batch_size"), 8, "screening batch"),
        (screening.get("seed_start"), 5600, "screening seeds"),
        (screening.get("warmup_seed_start"), 5592, "screening warm-up seeds"),
        (screening.get("policy_rng_seed"), 2026071822, "screening policy seed"),
        (
            screening.get("conditions"),
            ["frozen_base", "own_adapter"],
            "screening arms",
        ),
        (
            screening.get("locked_report_demos_accessed"),
            False,
            "screening report access",
        ),
        (
            confirmation.get("init_state_indices"),
            list(range(40, 48)),
            "confirmation states",
        ),
        (confirmation.get("batch_size"), 8, "confirmation batch"),
        (confirmation.get("seed_start"), 5700, "confirmation seeds"),
        (confirmation.get("warmup_seed_start"), 5692, "confirmation warm-up seeds"),
        (confirmation.get("policy_rng_seed"), 2026071823, "confirmation policy seed"),
        (
            confirmation.get("conditions"),
            ["frozen_base", "own_adapter", "swapped_adapter"],
            "confirmation arms",
        ),
        (
            confirmation.get("offline_report_episode_bounds"),
            [46, 49],
            "confirmation rows",
        ),
        (
            confirmation.get("selection_changes_after_access_forbidden"),
            True,
            "confirmation selection lock",
        ),
    ):
        _require_equal(value, expected, f"rank-16 {label}")
    if set(screening["init_state_indices"]) & set(confirmation["init_state_indices"]):
        raise GateZeroTargetSupportContractError("rank-16 rollout surfaces overlap")


def _resolve_rank16_spec(raw: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    authority = raw["authority"]
    override = raw["fit_override"]
    screening = raw["screening_rollout"]
    confirmation = raw["confirmation_rollout"]
    escalation = raw["rank_escalation"]
    resolved = copy.deepcopy(base)
    for key in (
        "schema_version",
        "name",
        "status",
        "screening_stage",
        "task_ids",
        "variants",
    ):
        resolved[key] = copy.deepcopy(raw[key])
    resolved["authority"].update(copy.deepcopy(authority))
    source = copy.deepcopy(base["fit"]["official_default_r8"])
    source.update(
        {
            "support_scope": "single_conditional_smolvla_default_like_rank16",
            "rank": override["rank"],
            "alpha": override["alpha"],
            "dropout": override["dropout"],
            "expected_trainable_parameters": override[
                "expected_trainable_parameters"
            ],
        }
    )
    resolved["fit"] = {
        key: copy.deepcopy(value)
        for key, value in base["fit"].items()
        if key not in base["variants"]
    }
    resolved["fit"]["official_default_r16"] = source
    resolved["screening_rollout"] = copy.deepcopy(screening)
    resolved["confirmation_rollout"] = copy.deepcopy(confirmation)
    resolved["rank_escalation"] = copy.deepcopy(escalation)
    resolved["parallel"] = copy.deepcopy(raw["parallel"])
    resolved["resources"].update(copy.deepcopy(raw["resources"]))
    return resolved


def load_target_support_rank16_spec(
    path: Path,
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
    prior_execution_path: Path,
    rank8_audit_path: Path,
) -> dict[str, Any]:
    """Resolve the sole rank-16 escalation from its hash-bound rank-8 parent."""

    raw = _load_rank16_raw(path)
    _validate_rank16_header(raw)
    base = load_target_support_audit_spec(
        rank8_audit_path,
        gate_zero_path=gate_zero_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
        prior_execution_path=prior_execution_path,
    )
    _validate_rank16_authority(
        raw.get("authority", {}),
        base,
        gate_zero_path=gate_zero_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
        prior_execution_path=prior_execution_path,
        rank8_audit_path=rank8_audit_path,
    )
    _validate_rank16_fit_and_resources(raw, base)
    _validate_rank16_rollouts(raw)
    return _resolve_rank16_spec(raw, base)


def load_target_support_screen_spec(
    path: Path,
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
    prior_execution_path: Path,
) -> dict[str, Any]:
    """Load either stage of the one bounded target-support recovery."""

    try:
        with path.open("rb") as handle:
            name = tomllib.load(handle).get("name")
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroTargetSupportContractError("invalid target-support TOML") from error
    if name == EXPECTED_NAME:
        return load_target_support_audit_spec(
            path,
            gate_zero_path=gate_zero_path,
            phase0_path=phase0_path,
            competence_path=competence_path,
            prior_execution_path=prior_execution_path,
        )
    if name == EXPECTED_RANK16_NAME:
        return load_target_support_rank16_spec(
            path,
            gate_zero_path=gate_zero_path,
            phase0_path=phase0_path,
            competence_path=competence_path,
            prior_execution_path=prior_execution_path,
            rank8_audit_path=path.with_name("gate_zero_target_support_audit.toml"),
        )
    if name in {
        "smolvla_libero90_gate_zero_mature_lora_positive_control_v1",
        "smolvla_libero90_gate_zero_mature_lora_all_linear_recovery_v1",
        "smolvla_libero90_gate_zero_mature_action_expert_upper_bound_v1",
        "smolvla_libero90_gate_zero_mature_action_expert_lr_recovery_v1",
    }:
        from ember.gate_zero_support.mature_contract import (
            load_mature_lora_positive_control_spec,
        )

        return load_mature_lora_positive_control_spec(
            path,
            gate_zero_path=gate_zero_path,
            phase0_path=phase0_path,
            competence_path=competence_path,
        )
    if name == "smolvla_libero90_gate_zero_mature_lora_lr_recovery_v1":
        from ember.gate_zero_support.mature_lora_lr_contract import (
            load_mature_lora_lr_recovery_spec,
        )

        return load_mature_lora_lr_recovery_spec(
            path,
            gate_zero_path=gate_zero_path,
            phase0_path=phase0_path,
            competence_path=competence_path,
        )
    if name == "smolvla_libero90_gate_zero_mature_lora_headroom_screen_v1":
        from ember.gate_zero_support.mature_headroom import (
            load_mature_lora_headroom_spec,
        )

        return load_mature_lora_headroom_spec(
            path,
            gate_zero_path=gate_zero_path,
            phase0_path=phase0_path,
            competence_path=competence_path,
        )
    raise GateZeroTargetSupportContractError("unknown target-support contract")
