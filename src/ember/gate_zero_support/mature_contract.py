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


def _validate_authority(
    raw: dict[str, Any],
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
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


def _validate_provenance(raw: dict[str, Any]) -> None:
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


def _validate_fit(raw: dict[str, Any], phase0: dict[str, Any]) -> None:
    fit = raw.get("fit", {})
    candidate_steps = list(range(0, 20_001, 1_000))
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
    variant = fit.get("mature_official_default_r32", {})
    for key, expected in (
        ("adaptation_kind", "lora"),
        ("rank", 32),
        ("alpha", 16),
        ("dropout", 0.0),
        ("init_lora_weights", "gaussian"),
        ("expected_trainable_parameters", 1_485_312),
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
        sorted(variant.get("target_modules", [])), mature_default_targets(), "mature exact targets"
    )
    scale_min = variant.get("augmentation_scale_min")
    scale_max = variant.get("augmentation_scale_max")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (scale_min, scale_max)):
        raise GateZeroMatureControlContractError("augmentation scale is invalid")
    _require_equal([float(scale_min), float(scale_max)], [0.9, 1.0], "augmentation scale")


def _validate_selection_rollout_and_decision(raw: dict[str, Any]) -> None:
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
        ("conditions", ["frozen_base", "mature_official_default_r32"]),
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
    _require_equal(raw.get("name"), EXPECTED_NAME, "mature name")
    _require_equal(raw.get("status"), EXPECTED_STATUS, "mature predeclaration")
    _require_equal(raw.get("task_ids"), [3, 4], "mature tasks")
    _require_equal(raw.get("variants"), ["mature_official_default_r32"], "mature variants")
    _validate_authority(
        raw,
        gate_zero_path=gate_zero_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
    )
    parent = load_gate_zero_contract(gate_zero_path, phase0_path)
    try:
        with phase0_path.open("rb") as handle:
            phase0 = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroMatureControlContractError("invalid Phase 0 authority") from error
    _require_equal(raw["task_ids"], parent["data"]["task_ids"], "parent task authority")
    _validate_provenance(raw)
    _validate_fit(raw, phase0)
    _validate_selection_rollout_and_decision(raw)
    _require_equal(raw.get("parallel", {}).get("maximum_concurrent_gpus"), 4, "GPU ceiling")
    _require_equal(raw.get("resources", {}).get("minimum_free_memory_mib"), 10_240, "OOM headroom")
    return raw
