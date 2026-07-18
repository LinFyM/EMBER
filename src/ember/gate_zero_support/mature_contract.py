"""Strict contract for the bounded mature-recipe task-local LoRA control."""

from __future__ import annotations

import hashlib
import math
import tomllib
from pathlib import Path
from typing import Any

from ember.gate_zero_contract import load_gate_zero_contract


EXPECTED_NAME = "smolvla_libero90_gate_zero_mature_lora_positive_control_v1"
EXPECTED_STATUS = (
    "predeclared_after_small_recipe_rank16_failure_before_mature_fit_or_rollout_outcomes"
)
RECOVERY_NAME = "smolvla_libero90_gate_zero_mature_lora_all_linear_recovery_v1"
RECOVERY_STATUS = (
    "predeclared_after_primary_stage5000_query_stop_before_conditional_fit_outcomes"
)
UPPER_BOUND_NAME = "smolvla_libero90_gate_zero_mature_action_expert_upper_bound_v1"
UPPER_BOUND_STATUS = (
    "predeclared_after_all_linear_stage2000_query_stop_before_action_expert_fit_outcomes"
)
LR_RECOVERY_NAME = "smolvla_libero90_gate_zero_mature_action_expert_lr_recovery_v1"
LR_RECOVERY_STATUS = (
    "predeclared_after_action_expert_scale_diagnostic_before_low_lr_fit_outcomes"
)
PRIMARY_VARIANT = "mature_official_default_r32"
RECOVERY_VARIANT = "all_action_expert_linear_r32_same_recipe"
UPPER_BOUND_VARIANT = "mature_action_expert_upper_bound"
LR_RECOVERY_VARIANT = "mature_action_expert_lr25e6_recovery"
EXPECTED_RANK16_RESULT_SHA256 = (
    "65b2abffcf8b2c7e8907c03f4e21cd8435da38b94afb9e8b41337a54bd323b00"
)
EXPECTED_LEROBOT_REVISION = "30da8e687a6dfc617fcd94afc367ac7071c376ce"
EXPECTED_OPENVLA_REVISION = "c8f03f48af692657d3060c19588038c7220e9af9"
EXPECTED_OPENVLA_OFT_REVISION = "e4287e94541f459edc4feabc4e181f537cd569a8"


class GateZeroMatureControlContractError(RuntimeError):
    """Raised when the mature positive-control contract changes authority."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GateZeroMatureControlContractError(
            f"{label} changed: {actual!r} != {expected!r}"
        )


def _require_hash(authority: dict[str, Any], key: str, path: Path, label: str) -> None:
    expected = authority.get(key)
    if not isinstance(expected, str) or len(expected) != 64 or _sha256(path) != expected:
        raise GateZeroMatureControlContractError(f"{label} SHA256 changed")


def mature_default_targets() -> list[str]:
    targets = [
        "model.action_in_proj",
        "model.action_out_proj",
        "model.action_time_mlp_in",
        "model.action_time_mlp_out",
        "model.state_proj",
    ]
    targets.extend(
        f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.{projection}_proj"
        for layer in range(16)
        for projection in ("q", "v")
    )
    return sorted(targets)


def mature_all_action_expert_linear_targets() -> list[str]:
    """Return the exact conditional SmolVLA all-action-expert-linear support."""

    targets = [
        "model.action_in_proj",
        "model.action_out_proj",
        "model.action_time_mlp_in",
        "model.action_time_mlp_out",
        "model.state_proj",
    ]
    targets.extend(
        f"model.vlm_with_expert.lm_expert.layers.{layer}.{module}"
        for layer in range(16)
        for module in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    )
    return sorted(targets)


def _validate_authority(
    raw: dict[str, Any],
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
    recovery: bool,
    upper_bound: bool,
    lr_recovery: bool,
) -> None:
    authority = raw.get("authority", {})
    _require_hash(authority, "gate_zero_contract_sha256", gate_zero_path, "Gate 0 contract")
    _require_hash(authority, "phase0_contract_sha256", phase0_path, "Phase 0 contract")
    _require_hash(
        authority,
        "source_competence_contract_sha256",
        competence_path,
        "source competence contract",
    )
    rank16_path = gate_zero_path.with_name("gate_zero_target_support_rank16.toml")
    _require_hash(
        authority,
        "prior_rank16_contract_sha256",
        rank16_path,
        "rank-16 contract",
    )
    _require_equal(
        authority.get("prior_rank16_screening_result_sha256"),
        EXPECTED_RANK16_RESULT_SHA256,
        "rank-16 screening result SHA256",
    )
    _require_equal(authority.get("prior_rank16_status"), "rank16_support_screen_failed", "prior rank-16 status")
    for key in (
        "validation_numeric_access",
        "held_numeric_access",
        "locked_report_numeric_reuse_for_selection",
    ):
        _require_equal(authority.get(key), False, key)
    if recovery or upper_bound or lr_recovery:
        _require_hash(
            authority,
            "primary_mature_contract_sha256",
            gate_zero_path.with_name("gate_zero_mature_lora_positive_control.toml"),
            "primary mature contract",
        )
        _require_hash(
            authority,
            "primary_stage_contract_sha256",
            gate_zero_path.with_name("gate_zero_mature_lora_stage_ladder.toml"),
            "primary stage contract",
        )
    if recovery:
        for task_id in (3, 4):
            for kind in ("candidate_manifest", "recovery_manifest", "telemetry"):
                key = f"primary_task{task_id}_{kind}_sha256"
                value = authority.get(key)
                if not isinstance(value, str) or len(value) != 64:
                    raise GateZeroMatureControlContractError(
                        f"primary task-{task_id} {kind} authority is invalid"
                    )
    if upper_bound or lr_recovery:
        _require_hash(
            authority,
            "all_linear_contract_sha256",
            gate_zero_path.with_name("gate_zero_mature_lora_all_linear_recovery.toml"),
            "all-linear recovery contract",
        )
        _require_hash(
            authority,
            "all_linear_stage_contract_sha256",
            gate_zero_path.with_name("gate_zero_mature_lora_all_linear_stage_ladder.toml"),
            "all-linear stage contract",
        )
        _require_equal(authority.get("final_closed_loop_accessed"), False, "upper-bound rollout access")
        if upper_bound:
            for task_id in (3, 4):
                for kind in ("candidate_manifest", "recovery_manifest", "telemetry"):
                    value = authority.get(f"all_linear_task{task_id}_{kind}_sha256")
                    if not isinstance(value, str) or len(value) != 64:
                        raise GateZeroMatureControlContractError(
                            f"all-linear task-{task_id} {kind} authority is invalid"
                        )
    if lr_recovery:
        _require_hash(
            authority,
            "upper_bound_contract_sha256",
            gate_zero_path.with_name("gate_zero_mature_action_expert_upper_bound.toml"),
            "action-expert upper-bound contract",
        )
        _require_hash(
            authority,
            "upper_bound_stage_contract_sha256",
            gate_zero_path.with_name("gate_zero_mature_action_expert_stage_ladder.toml"),
            "action-expert upper-bound stage contract",
        )
        for task_id in (3, 4):
            for kind in (
                "candidate_manifest",
                "recovery_manifest",
                "telemetry",
            ):
                value = authority.get(f"upper_bound_task{task_id}_{kind}_sha256")
                if not isinstance(value, str) or len(value) != 64:
                    raise GateZeroMatureControlContractError(
                        f"upper-bound task-{task_id} {kind} authority is invalid"
                    )
            for kind in ("result", "source"):
                value = authority.get(f"task{task_id}_update_scale_probe_{kind}_sha256")
                if not isinstance(value, str) or len(value) != 64:
                    raise GateZeroMatureControlContractError(
                        f"update-scale task-{task_id} {kind} authority is invalid"
                    )


def _validate_provenance(
    raw: dict[str, Any], *, recovery: bool, upper_bound: bool, lr_recovery: bool
) -> None:
    if upper_bound or lr_recovery:
        diagnostic = raw.get("diagnostic_contract", {})
        for key in (
            "matched_lora_gate_pass",
            "may_authorize_gate_zero",
            "may_authorize_writer",
            "may_seal_writer_target_contract",
        ):
            _require_equal(diagnostic.get(key), False, f"capacity diagnostic {key}")
        provenance = raw.get("recipe_provenance", {})
        _require_equal(
            provenance.get("lerobot_revision"), EXPECTED_LEROBOT_REVISION, "LeRobot revision"
        )
        _require_equal(provenance.get("lerobot_tag"), "v0.6.0", "LeRobot tag")
        if lr_recovery:
            _require_equal(
                provenance.get(
                    "same_data_sampler_noise_augmentation_optimizer_and_seed_as_upper_bound"
                ),
                True,
                "learning-rate recovery matched mechanics",
            )
            _require_equal(
                provenance.get("only_learning_rate_schedule_magnitude_differs"),
                True,
                "learning-rate recovery isolated change",
            )
            _require_equal(
                float(provenance.get("peak_and_decay_learning_rates_scaled_by")),
                0.25,
                "learning-rate recovery scale",
            )
        else:
            _require_equal(
                provenance.get("same_mature_recipe_as_lora_controls"),
                True,
                "upper-bound mature recipe",
            )
            _require_equal(
                provenance.get("only_trainable_state_class_differs"),
                True,
                "upper-bound isolated change",
            )
        return
    override = raw.get("owner_override", {})
    for key in (
        "small_recipe_final_negative_superseded",
        "support_query_is_an_isolation_principle_not_a_fixed_small_data_budget",
        "known_positive_task_local_lora_competence_required_before_final_negative_or_writer_target_seal",
        "shrink_after_success_is_not_the_mainline",
    ):
        _require_equal(override.get(key), True, f"owner override {key}")
    provenance = raw.get("recipe_provenance", {})
    _require_equal(provenance.get("lerobot_revision"), EXPECTED_LEROBOT_REVISION, "LeRobot revision")
    _require_equal(provenance.get("lerobot_tag"), "v0.6.0", "LeRobot tag")
    _require_equal(provenance.get("openvla_revision"), EXPECTED_OPENVLA_REVISION, "OpenVLA revision")
    _require_equal(
        provenance.get("openvla_oft_revision"),
        EXPECTED_OPENVLA_OFT_REVISION,
        "OpenVLA-OFT revision",
    )
    if "not validated LoRA evidence" not in provenance.get(
        "lerobot_smolvla_full_recipe_classification", ""
    ):
        raise GateZeroMatureControlContractError("SmolVLA full-recipe classification changed")
    if recovery:
        for key, expected in (
            ("conditional_all_action_expert_linear_recovery_authorized", True),
            ("maximum_additional_target_support_variants", 1),
            ("no_additional_rank_or_target_search_after_this_variant", True),
        ):
            _require_equal(override.get(key), expected, f"recovery owner override {key}")


def _validate_upper_bound_variant(
    variant: dict[str, Any], *, lr_recovery: bool = False
) -> None:
    """Keep the non-matched action-expert diagnostic unable to grant LoRA claims."""

    for key, expected in (
        ("adaptation_kind", "partial_update"),
        (
            "scope",
            "all_action_expert_and_state_projection_parameters_trainable_in_source_base_checkpoint",
        ),
        ("expected_trainable_parameters", 99_880_992),
        ("matched_baseline", False),
        ("may_authorize_gate_zero", False),
        ("may_authorize_writer", False),
        ("may_seal_writer_target_contract", False),
        ("optimizer", "adamw"),
        ("learning_rate", 0.000025 if lr_recovery else 0.0001),
        ("betas", [0.9, 0.95]),
        ("epsilon", 1e-8),
        ("weight_decay", 1e-10),
        ("gradient_clip_norm", 10.0),
        ("scheduler", "linear_warmup_cosine_decay"),
        ("warmup_steps", 1_000),
        ("decay_steps", 20_000),
        ("decay_learning_rate", 0.000000625 if lr_recovery else 0.0000025),
        ("augmentation", "random_resized_crop"),
        ("augmentation_ratio", 1.0),
        ("augmentation_seed", 2026071832),
        ("seed", 2026071830),
    ):
        _require_equal(variant.get(key), expected, f"capacity variant {key}")
    _require_equal(
        [float(variant.get("augmentation_scale_min")), float(variant.get("augmentation_scale_max"))],
        [0.9, 1.0],
        "upper-bound augmentation scale",
    )


def _validate_fit(
    raw: dict[str, Any],
    phase0: dict[str, Any],
    *,
    variant_name: str,
    expected_targets: list[str],
    expected_parameters: int,
    upper_bound: bool,
    lr_recovery: bool,
) -> None:
    fit = raw.get("fit", {})
    candidate_steps = (
        [0, 250, 500, 750, 1_000, 2_000, 5_000, 10_000, 20_000]
        if lr_recovery
        else list(range(0, 20_001, 1_000))
    )
    for key, expected in (
        ("support_episode_bounds", [0, 39]),
        ("support_episode_roles", ["writer_spec", "source_base_fit", "oracle_support"]),
        ("optimizer_steps", 20_000),
        ("effective_batch_size", 64),
        ("micro_batch_size", 64),
        ("gradient_accumulation_steps", 1),
        ("candidate_steps", candidate_steps),
        ("retain_scientific_candidate_records", True),
        ("retain_selected_trainable_state", True),
        ("cleanup_recovery_state_after_completed_selection", True),
    ):
        _require_equal(fit.get(key), expected, f"mature fit {key}")
    role_ranges = [phase0.get("episode_authority", {}).get(role) for role in fit["support_episode_roles"]]
    if any(
        not isinstance(bounds, list)
        or len(bounds) != 2
        or not all(isinstance(value, int) for value in bounds)
        for bounds in role_ranges
    ):
        raise GateZeroMatureControlContractError("mature support episode-role authority is invalid")
    role_episodes = sorted(
        episode
        for bounds in role_ranges
        for episode in range(bounds[0], bounds[1] + 1)
    )
    _require_equal(role_episodes, list(range(40)), "mature support episode-role union")
    if "writer_spec actions remain hidden from Writer inputs" not in fit.get(
        "support_action_authority", ""
    ):
        raise GateZeroMatureControlContractError("mature source-action authority changed")
    if fit.get("num_workers", -1) < 0 or fit.get("prefetch_factor", 0) <= 0:
        raise GateZeroMatureControlContractError("mature loader resources are invalid")
    variant = fit.get(variant_name, {})
    if upper_bound or lr_recovery:
        _validate_upper_bound_variant(variant, lr_recovery=lr_recovery)
        return
    for key, expected in (
        ("adaptation_kind", "lora"),
        ("rank", 32),
        ("alpha", 16),
        ("dropout", 0.0),
        ("init_lora_weights", "gaussian"),
        ("expected_trainable_parameters", expected_parameters),
        ("optimizer", "adamw"),
        ("learning_rate", 0.0001),
        ("betas", [0.9, 0.95]),
        ("epsilon", 1e-8),
        ("weight_decay", 1e-10),
        ("gradient_clip_norm", 10.0),
        ("scheduler", "linear_warmup_cosine_decay"),
        ("warmup_steps", 1_000),
        ("decay_steps", 20_000),
        ("decay_learning_rate", 0.0000025),
        ("augmentation", "random_resized_crop"),
        ("augmentation_ratio", 1.0),
    ):
        _require_equal(variant.get(key), expected, f"mature variant {key}")
    _require_equal(
        sorted(variant.get("target_modules", [])), expected_targets, "mature exact targets"
    )
    scale_min = variant.get("augmentation_scale_min")
    scale_max = variant.get("augmentation_scale_max")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (scale_min, scale_max)):
        raise GateZeroMatureControlContractError("augmentation scale is invalid")
    _require_equal([float(scale_min), float(scale_max)], [0.9, 1.0], "augmentation scale")


def _validate_selection_rollout_and_decision(
    raw: dict[str, Any], *, variant_name: str
) -> None:
    selection = raw.get("selection", {})
    for key, expected in (
        ("query_episode_bounds", [40, 45]),
        ("candidate_rule", "fixed_final_optimizer_step"),
        ("fixed_final_optimizer_step", 20_000),
        ("selection_uses_locked_report", False),
        ("step_zero_must_be_functional_zero", True),
        ("drift_role", "diagnostic_only_for_mature_task_specific_competence"),
    ):
        _require_equal(selection.get(key), expected, f"mature selection {key}")
    rollout = raw.get("screening_rollout", {})
    for key, expected in (
        ("init_state_indices", list(range(40, 48))),
        ("batch_size", 8),
        ("seed_start", 5800),
        ("warmup_seed_start", 5760),
        ("conditions", ["frozen_base", variant_name]),
        ("locked_report_demos_accessed", False),
        ("selection_changes_after_access_forbidden", True),
    ):
        _require_equal(rollout.get(key), expected, f"mature rollout {key}")
    decision = raw.get("decision", {})
    for key, expected in (
        ("median_success_gain_pp_min", 15.0),
        ("median_locked_action_loss_reduction_fraction_min", 0.20),
        ("positive_task_fraction_min", 1.0),
        ("two_task_positive_count_required", 2),
        ("selection_drift_is_diagnostic_only", True),
        ("thresholds_unchanged_from_gate_zero", True),
        ("pass_authorizes_gate_zero", True),
        ("pass_seals_final_writer_target_contract", True),
        ("pass_authorizes_direct_writer", True),
        ("failure_is_not_a_final_ember_negative", True),
    ):
        _require_equal(decision.get(key), expected, f"mature decision {key}")
    _require_equal(raw.get("screening_stage"), "mature_positive_control", "mature stage")
    _require_equal(raw.get("writer_authorized_before_closed_loop"), False, "premature Writer authority")


def _validate_recovery_boundary(raw: dict[str, Any]) -> None:
    failure = raw.get("primary_failure", {})
    for key, expected in (
        ("stage_step", 5_000),
        ("task3_query_reduction_fraction", 0.0031467544497260458),
        ("task4_query_reduction_fraction", -0.02956270294978539),
        ("median_query_reduction_fraction", -0.013207974250029672),
        ("median_regression_from_step2000_fraction", 0.07590890162321846),
        ("continuation_to_step10000", False),
        ("final_closed_loop_accessed", False),
        ("validation_numeric_access", False),
        ("held_numeric_access", False),
    ):
        _require_equal(failure.get(key), expected, f"primary failure {key}")
    bounded = raw.get("bounded_recovery", {})
    for key, expected in (
        ("maximum_compatibility_variants_after_primary_failure", 1),
        ("this_is_the_conditional_variant", True),
        ("no_further_target_or_rank_variants", True),
    ):
        _require_equal(bounded.get(key), expected, f"bounded recovery {key}")


def _validate_upper_bound_selection(raw: dict[str, Any]) -> None:
    selection = raw.get("selection", {})
    for key, expected in (
        ("query_episode_bounds", [40, 45]),
        ("candidate_rule", "fixed_final_optimizer_step"),
        ("fixed_final_optimizer_step", 20_000),
        ("selection_uses_locked_report", False),
        ("step_zero_must_be_functional_base", True),
        ("drift_role", "diagnostic_only_for_non_matched_action_expert_capacity"),
    ):
        _require_equal(selection.get(key), expected, f"upper-bound selection {key}")
    rollout = raw.get("capacity_rollout", {})
    for key, expected in (
        ("init_state_indices", list(range(40, 48))),
        ("batch_size", 8),
        ("seed_start", 5800),
        ("warmup_seed_start", 5760),
        ("conditions", ["frozen_base", UPPER_BOUND_VARIANT]),
        ("access_authorized_only_after_full_query_trajectory_pass", True),
        ("selection_changes_after_access_forbidden", True),
        ("may_authorize_gate_zero", False),
        ("may_authorize_writer", False),
    ):
        _require_equal(rollout.get(key), expected, f"upper-bound rollout {key}")
    bounded = raw.get("bounded_recovery", {})
    for key in (
        "this_is_not_a_lora_target_or_rank_variant",
        "no_further_lora_target_or_rank_variants",
        "failure_requires_gate_recovery_decision",
        "success_requires_gate_recovery_decision",
    ):
        _require_equal(bounded.get(key), True, f"upper-bound recovery {key}")
    _require_equal(raw.get("screening_stage"), "mature_capacity_upper_bound", "upper stage")
    _require_equal(raw.get("writer_authorized_before_closed_loop"), False, "premature Writer authority")


def _validate_upper_bound_boundary(raw: dict[str, Any]) -> None:
    failure = raw.get("prior_lora_failure", {})
    for key, expected in (
        ("stage_step", 2_000),
        ("task3_query_reduction_fraction", -0.03149279435261671),
        ("task4_query_reduction_fraction", -0.12093111131648279),
        ("median_query_reduction_fraction", -0.07621195283454975),
        ("median_regression_from_step1000_fraction", 0.12449740168086804),
        ("continuation_to_step5000", False),
        ("no_further_lora_target_or_rank_variant", True),
    ):
        _require_equal(failure.get(key), expected, f"upper-bound prior failure {key}")


def _validate_lr_recovery_selection(raw: dict[str, Any]) -> None:
    selection = raw.get("selection", {})
    for key, expected in (
        ("query_episode_bounds", [40, 45]),
        ("candidate_rule", "fixed_final_optimizer_step"),
        ("fixed_final_optimizer_step", 20_000),
        ("selection_uses_locked_report", False),
        ("step_zero_must_be_functional_base", True),
        ("drift_role", "diagnostic_only_for_non_matched_action_expert_capacity"),
    ):
        _require_equal(selection.get(key), expected, f"LR recovery selection {key}")
    rollout = raw.get("capacity_rollout", {})
    for key, expected in (
        ("role", "forbidden_for_this_optimization_diagnostic"),
        ("access_authorized", False),
        ("may_authorize_gate_zero", False),
        ("may_authorize_writer", False),
    ):
        _require_equal(rollout.get(key), expected, f"LR recovery rollout {key}")
    bounded = raw.get("bounded_recovery", {})
    for key in (
        "this_is_not_a_lora_target_or_rank_variant",
        "no_further_lora_target_or_rank_variants",
        "single_learning_rate_recovery_only",
        "no_learning_rate_grid",
        "upper_bound_must_pass_step1000_before_matched_lora_schedule",
        "success_authorizes_only_matched_lora_schedule",
        "failure_requires_gate_zero_data_or_acquisition_plan_revision",
    ):
        _require_equal(bounded.get(key), True, f"LR recovery boundary {key}")
    _require_equal(
        raw.get("screening_stage"),
        "mature_capacity_lr_recovery",
        "LR recovery stage",
    )
    _require_equal(
        raw.get("writer_authorized_before_closed_loop"),
        False,
        "premature Writer authority",
    )


def _validate_lr_recovery_boundary(raw: dict[str, Any]) -> None:
    failure = raw.get("prior_action_expert_failure", {})
    for key, expected in (
        ("stage_step", 1_000),
        ("task3_query_reduction_fraction", -0.05220677659648454),
        ("task4_query_reduction_fraction", -0.19788259189706617),
        ("median_query_reduction_fraction", -0.12504468424677534),
        ("continuation_to_step2000", False),
        ("support_loss_improved_on_both_tasks", True),
    ):
        _require_equal(failure.get(key), expected, f"LR recovery prior failure {key}")
    diagnostic = raw.get("update_scale_diagnostic", {})
    for key, expected in (
        ("diagnostic_only", True),
        ("scaled_final_delta_is_not_lower_lr_training", True),
        ("endpoint_identity_passed", True),
        ("task3_scale025_query_reduction_fraction", 0.027049502954258852),
        ("task4_scale025_query_reduction_fraction", 0.01311309319317943),
        ("median_scale025_query_reduction_fraction", 0.02008129807371914),
        ("task4_scale050_query_reduction_fraction", -0.012513598169400765),
        ("selected_single_recovery_scale", 0.25),
    ):
        _require_equal(diagnostic.get(key), expected, f"update-scale diagnostic {key}")


def load_mature_lora_positive_control_spec(
    path: Path,
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
) -> dict[str, Any]:
    """Load and fail closed on the one bounded mature-recipe positive control."""

    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroMatureControlContractError("invalid mature control TOML") from error
    _require_equal(raw.get("schema_version"), 1, "mature schema")
    name = raw.get("name")
    if name == EXPECTED_NAME:
        recovery = False
        upper_bound = False
        lr_recovery = False
        status = EXPECTED_STATUS
        variant_name = PRIMARY_VARIANT
        expected_targets = mature_default_targets()
        expected_parameters = 1_485_312
    elif name == RECOVERY_NAME:
        recovery = True
        upper_bound = False
        lr_recovery = False
        status = RECOVERY_STATUS
        variant_name = RECOVERY_VARIANT
        expected_targets = mature_all_action_expert_linear_targets()
        expected_parameters = 7_027_200
    elif name == UPPER_BOUND_NAME:
        recovery = False
        upper_bound = True
        lr_recovery = False
        status = UPPER_BOUND_STATUS
        variant_name = UPPER_BOUND_VARIANT
        expected_targets = []
        expected_parameters = 99_880_992
    elif name == LR_RECOVERY_NAME:
        recovery = False
        upper_bound = False
        lr_recovery = True
        status = LR_RECOVERY_STATUS
        variant_name = LR_RECOVERY_VARIANT
        expected_targets = []
        expected_parameters = 99_880_992
    else:
        raise GateZeroMatureControlContractError(f"unknown mature control name: {name!r}")
    _require_equal(raw.get("status"), status, "mature predeclaration")
    _require_equal(raw.get("task_ids"), [3, 4], "mature tasks")
    _require_equal(raw.get("variants"), [variant_name], "mature variants")
    _validate_authority(
        raw,
        gate_zero_path=gate_zero_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
        recovery=recovery,
        upper_bound=upper_bound,
        lr_recovery=lr_recovery,
    )
    parent = load_gate_zero_contract(gate_zero_path, phase0_path)
    try:
        with phase0_path.open("rb") as handle:
            phase0 = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroMatureControlContractError("invalid Phase 0 authority") from error
    _require_equal(raw["task_ids"], parent["data"]["task_ids"], "parent task authority")
    _validate_provenance(
        raw,
        recovery=recovery,
        upper_bound=upper_bound,
        lr_recovery=lr_recovery,
    )
    _validate_fit(
        raw,
        phase0,
        variant_name=variant_name,
        expected_targets=expected_targets,
        expected_parameters=expected_parameters,
        upper_bound=upper_bound,
        lr_recovery=lr_recovery,
    )
    if lr_recovery:
        _validate_lr_recovery_selection(raw)
        _validate_lr_recovery_boundary(raw)
    elif upper_bound:
        _validate_upper_bound_selection(raw)
        _validate_upper_bound_boundary(raw)
    else:
        _validate_selection_rollout_and_decision(raw, variant_name=variant_name)
    if recovery:
        _validate_recovery_boundary(raw)
    _require_equal(raw.get("parallel", {}).get("maximum_concurrent_gpus"), 4, "GPU ceiling")
    _require_equal(raw.get("resources", {}).get("minimum_free_memory_mib"), 10_240, "OOM headroom")
    return raw
