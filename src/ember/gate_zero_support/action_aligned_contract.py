"""Strict contract for the bounded generated-action Gate 0 acquisition recovery."""

from __future__ import annotations

import hashlib
import math
import tomllib
from pathlib import Path
from typing import Any

from ember.gate_zero_contract import load_gate_zero_contract
from ember.gate_zero_support.mature_contract import mature_default_targets


EXPECTED_NAME = "smolvla_libero90_gate_zero_action_aligned_lora_acquisition_v1"
EXPECTED_STATUS = (
    "predeclared_after_differentiable_action_loss_smoke_before_action_aligned_fit_outcomes"
)
EXPECTED_STAGE = "action_aligned_lora_acquisition_recovery"
EXPECTED_VARIANT = "action_aligned_official_default_r32"


class GateZeroActionAlignedContractError(RuntimeError):
    """Raised when the result-blind action-aligned recovery contract changes."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GateZeroActionAlignedContractError(
            f"{label} changed: {actual!r} != {expected!r}"
        )


def _require_hash(authority: dict[str, Any], key: str, path: Path, label: str) -> None:
    value = authority.get(key)
    if not isinstance(value, str) or len(value) != 64 or _sha256(path) != value:
        raise GateZeroActionAlignedContractError(f"{label} SHA256 changed")


def _require_declared_sha(authority: dict[str, Any], key: str, label: str) -> None:
    value = authority.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise GateZeroActionAlignedContractError(f"{label} SHA256 is invalid")


def _validate_authority(
    raw: dict[str, Any],
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
) -> None:
    authority = raw.get("authority", {})
    _require_hash(authority, "gate_zero_contract_sha256", gate_zero_path, "Gate 0")
    _require_hash(authority, "phase0_contract_sha256", phase0_path, "Phase 0")
    _require_hash(
        authority,
        "source_competence_contract_sha256",
        competence_path,
        "source competence",
    )
    _require_hash(
        authority,
        "prior_lora_fit_contract_sha256",
        gate_zero_path.with_name("gate_zero_mature_lora_lr_recovery.toml"),
        "prior LoRA fit",
    )
    _require_hash(
        authority,
        "action_loss_smoke_contract_sha256",
        gate_zero_path.with_name("gate_zero_differentiable_action_loss_smoke.toml"),
        "action-loss smoke",
    )
    for key, label in (
        ("source_competence_result_sha256", "source competence result"),
        ("source_base_checkpoint_manifest_sha256", "source-base checkpoint"),
        ("action_alignment_result_sha256", "action-alignment result"),
        ("action_loss_smoke_result_sha256", "action-loss smoke result"),
    ):
        _require_declared_sha(authority, key, label)
    for key in (
        "validation_numeric_access",
        "held_numeric_access",
        "locked_report_numeric_reuse_for_selection",
        "new_closed_loop_outcome_accessed_before_freeze",
    ):
        _require(authority.get(key), False, key)


def _validate_fit(raw: dict[str, Any]) -> None:
    fit = raw.get("fit", {})
    for key, expected in (
        ("support_episode_bounds", [0, 39]),
        ("support_episode_roles", ["writer_spec", "source_base_fit", "oracle_support"]),
        ("optimizer_steps", 200),
        ("effective_batch_size", 64),
        ("micro_batch_size", 64),
        ("gradient_accumulation_steps", 1),
        ("candidate_steps", [0, 1, 5, 10, 25, 50, 100, 200]),
        ("retain_scientific_candidate_records", True),
        ("retain_selected_trainable_state", True),
        ("cleanup_unselected_partial_trainable_states_after_selection", True),
        ("cleanup_recovery_state_after_completed_selection", True),
    ):
        _require(fit.get(key), expected, f"fit {key}")
    if fit.get("num_workers", -1) < 0 or fit.get("prefetch_factor", 0) <= 0:
        raise GateZeroActionAlignedContractError("loader resource contract is invalid")
    for key in ("persistent_workers", "pin_memory"):
        _require(fit.get(key), True, f"fit {key}")

    variant = fit.get(EXPECTED_VARIANT, {})
    for key, expected in (
        ("adaptation_kind", "lora"),
        ("rank", 32),
        ("alpha", 16),
        ("dropout", 0.0),
        ("init_lora_weights", "gaussian"),
        ("expected_trainable_parameters", 1_485_312),
        ("training_objective", "full_sampler_generated_action_mse"),
        ("action_loss_sampler_steps", 10),
        ("action_loss_noise_dimension", 32),
        ("generated_action_dimension", 7),
        ("action_loss_noise_seed", 2026072235),
        ("optimizer", "adamw"),
        ("learning_rate", 0.000025),
        ("betas", [0.9, 0.95]),
        ("epsilon", 1e-8),
        ("weight_decay", 1e-10),
        ("gradient_clip_norm", 10.0),
        ("scheduler", "linear_warmup_cosine_decay"),
        ("warmup_steps", 5),
        ("decay_steps", 200),
        ("decay_learning_rate", 0.000000625),
        ("augmentation", "random_resized_crop"),
        ("augmentation_scale_min", 0.9),
        ("augmentation_scale_max", 1.0),
        ("augmentation_ratio", 1.0),
        ("augmentation_seed", 2026071832),
        ("seed", 2026071830),
    ):
        _require(variant.get(key), expected, f"variant {key}")
    _require(
        sorted(variant.get("target_modules", [])),
        mature_default_targets(),
        "exact LoRA targets",
    )


def _validate_selection_and_ladder(raw: dict[str, Any]) -> None:
    selection = raw.get("selection", {})
    for key, expected in (
        ("query_episode_bounds", [40, 45]),
        ("candidate_rule", "minimum_mean_generated_action_mse_with_drift_cap"),
        ("selection_uses_locked_report", False),
        ("step_zero_must_be_functional_zero", True),
        ("fixed_noise_seed", 2026071833),
        ("fixed_time_seed", 2026071834),
        ("inference_noise_seed", 2026071835),
        (
            "action_error_inference_noise_seeds",
            [2026071835, 2026071935, 2026072035, 2026072135],
        ),
        ("noise_draws", 4),
        ("anchor_frames_per_demo", 8),
        ("drift_proxy_max", 0.02),
        ("evaluation_batch_size", 64),
    ):
        _require(selection.get(key), expected, f"selection {key}")
    _require(
        selection.get("flow_query_role"),
        "diagnostic_only_not_primary_selection_or_continuation_metric",
        "flow query role",
    )

    ladder = raw.get("staged_ladder", {})
    for key, expected in (
        ("stage_steps", [1, 5, 10, 25, 50, 100, 200]),
        ("first_memory_probe_step", 1),
        ("first_blind_scientific_boundary_step", 5),
        ("maximum_blind_wall_minutes", 30),
        ("maximum_resumable_segment_wall_minutes", 120),
        ("two_task_evidence_required_before_scientific_continuation", True),
        ("step1_maximum_peak_reserved_mib", 71_680),
        ("step5_to_step10_minimum_each_task_action_mse_reduction_fraction", 0.0),
        ("step10_to_step25_minimum_each_task_action_mse_reduction_fraction", 0.01),
        ("stop_on_persistent_flat_or_worse_action_mse", True),
    ):
        _require(ladder.get(key), expected, f"ladder {key}")

    opening = raw.get("closed_loop_opening", {})
    _require(opening.get("minimum_each_task_action_mse_reduction_fraction"), 0.02, "opening action reduction")
    _require(opening.get("maximum_each_task_action_drift_proxy"), 0.02, "opening drift")
    _require(opening.get("flow_query_is_diagnostic_only"), True, "opening flow role")
    _require(opening.get("writer_authorized_by_offline_metric"), False, "opening Writer authority")


def _validate_boundary_and_resources(raw: dict[str, Any]) -> None:
    rationale = raw.get("recovery_rationale", {})
    _require(
        rationale.get("bounded_intervention"),
        "replace_supervised_acquisition_loss_with_differentiable_full_sampler_generated_action_mse_and_compress_only_the_schedule_horizon_to_the_fixed_200_step_ladder_while_preserving_peak_and_decay_lr_magnitudes",
        "bounded intervention",
    )
    _require(rationale.get("no_target_rank_task_seed_or_gate_change"), True, "recovery isolation")
    boundary = raw.get("boundary", {})
    for key in (
        "same_tasks_seeds_thresholds_and_evaluator_as_failed_gate",
        "no_validation_or_held_or_locked_access",
        "no_target_rank_or_learning_rate_search",
        "no_closed_loop_before_offline_opening_rule",
        "failure_does_not_lower_gate_threshold",
    ):
        _require(boundary.get(key), True, f"boundary {key}")
    _require(boundary.get("gate_zero_authorized_before_closed_loop"), False, "Gate authority")
    _require(boundary.get("writer_authorized"), False, "Writer authority")
    parallel = raw.get("parallel", {})
    _require(
        parallel.get("fit_jobs"),
        [f"{EXPECTED_VARIANT}:3", f"{EXPECTED_VARIANT}:4"],
        "parallel jobs",
    )
    _require(parallel.get("maximum_concurrent_gpus"), 4, "GPU ceiling")
    _require(parallel.get("one_independent_process_per_gpu"), True, "process topology")
    _require(parallel.get("shared_parameters_across_jobs"), False, "shared process state")
    resources = raw.get("resources", {})
    _require(resources.get("minimum_free_memory_mib"), 10_240, "OOM headroom")
    _require(resources.get("maximum_peak_device_memory_mib"), 71_680, "peak memory")
    storage = resources.get("expected_peak_additional_storage_gib_per_task")
    if not isinstance(storage, (int, float)) or not math.isfinite(storage) or storage <= 0:
        raise GateZeroActionAlignedContractError("storage estimate is invalid")


def load_action_aligned_acquisition_spec(
    path: Path,
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
) -> dict[str, Any]:
    """Load the sole result-blind action-aligned Gate 0 acquisition recovery."""

    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroActionAlignedContractError("invalid action-aligned TOML") from error
    _require(raw.get("schema_version"), 1, "schema")
    _require(raw.get("name"), EXPECTED_NAME, "name")
    _require(raw.get("status"), EXPECTED_STATUS, "status")
    _require(raw.get("screening_stage"), EXPECTED_STAGE, "screening stage")
    _require(raw.get("task_ids"), [3, 4], "tasks")
    _require(raw.get("variants"), [EXPECTED_VARIANT], "variants")
    _require(raw.get("writer_authorized_before_closed_loop"), False, "Writer authority")
    parent = load_gate_zero_contract(gate_zero_path, phase0_path)
    _require(raw["task_ids"], parent["data"]["task_ids"], "parent task authority")
    _validate_authority(
        raw,
        gate_zero_path=gate_zero_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
    )
    _validate_fit(raw)
    _validate_selection_and_ladder(raw)
    _validate_boundary_and_resources(raw)
    return raw
