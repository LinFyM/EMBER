"""Strict contract for the matched mature-LoRA lower-LR recovery."""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

from ember.gate_zero_contract import load_gate_zero_contract
from ember.gate_zero_support.mature_contract import (
    load_mature_lora_positive_control_spec,
    mature_default_targets,
)


EXPECTED_NAME = "smolvla_libero90_gate_zero_mature_lora_lr_recovery_v1"
EXPECTED_STATUS = (
    "predeclared_after_action_expert_lr1000_pass_before_lora_lr_fit_outcomes"
)
EXPECTED_VARIANT = "mature_official_default_r32_lr25e6_recovery"
EXPECTED_STAGE = "mature_lora_lr_recovery"
EXPECTED_CANDIDATE_STEPS = [0, 250, 500, 750, 1_000, 2_000, 5_000, 10_000, 20_000]


class GateZeroMatureLoraLRContractError(RuntimeError):
    """Raised when the matched-LoRA lower-LR authority changes."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GateZeroMatureLoraLRContractError(
            f"{label} changed: {actual!r} != {expected!r}"
        )


def _require_hash(authority: dict[str, Any], key: str, path: Path, label: str) -> None:
    expected = authority.get(key)
    if not isinstance(expected, str) or len(expected) != 64 or _sha256(path) != expected:
        raise GateZeroMatureLoraLRContractError(f"{label} SHA256 changed")


def _load_toml(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroMatureLoraLRContractError(f"invalid {label} TOML") from error
    if not isinstance(value, dict):
        raise GateZeroMatureLoraLRContractError(f"invalid {label} TOML")
    return value


def _validate_authority(
    raw: dict[str, Any],
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
    action_lr: dict[str, Any],
) -> None:
    authority = raw.get("authority", {})
    config_dir = gate_zero_path.parent
    for key, path, label in (
        ("gate_zero_contract_sha256", gate_zero_path, "Gate 0 contract"),
        ("phase0_contract_sha256", phase0_path, "Phase 0 contract"),
        ("source_competence_contract_sha256", competence_path, "source competence contract"),
        (
            "primary_mature_contract_sha256",
            config_dir / "gate_zero_mature_lora_positive_control.toml",
            "primary mature-LoRA contract",
        ),
        (
            "primary_stage_contract_sha256",
            config_dir / "gate_zero_mature_lora_stage_ladder.toml",
            "primary mature-LoRA ladder",
        ),
        (
            "action_expert_lr_contract_sha256",
            config_dir / "gate_zero_mature_action_expert_lr_recovery.toml",
            "action-expert lower-LR contract",
        ),
        (
            "action_expert_lr_ladder_sha256",
            config_dir / "gate_zero_mature_action_expert_lr_recovery_ladder.toml",
            "action-expert lower-LR ladder",
        ),
    ):
        _require_hash(authority, key, path, label)
    for key in (
        "source_competence_result_relative_path",
        "source_competence_result_sha256",
        "source_base_output_relative_path",
        "source_base_checkpoint_step",
        "source_base_checkpoint_role",
        "source_base_checkpoint_manifest_sha256",
        "prior_rank16_contract_sha256",
        "prior_rank16_screening_result_relative_path",
        "prior_rank16_screening_result_sha256",
        "prior_rank16_status",
    ):
        _require_equal(
            authority.get(key), action_lr["authority"].get(key), f"authority {key}"
        )
    for key in (
        "validation_numeric_access",
        "held_numeric_access",
        "locked_report_numeric_reuse_for_selection",
        "final_closed_loop_accessed",
    ):
        _require_equal(authority.get(key), False, key)
    for task_id in (3, 4):
        for kind in ("candidate_manifest", "recovery_manifest", "telemetry"):
            value = authority.get(f"lr_action_expert_task{task_id}_{kind}_sha256")
            if not isinstance(value, str) or len(value) != 64:
                raise GateZeroMatureLoraLRContractError(
                    f"action-expert task-{task_id} {kind} authority is invalid"
                )


def _validate_fit(raw: dict[str, Any], primary: dict[str, Any], action_lr: dict[str, Any]) -> None:
    fit = raw.get("fit", {})
    action_fit = action_lr["fit"]
    for key in (
        "support_episode_bounds",
        "support_episode_roles",
        "support_action_authority",
        "optimizer_steps",
        "effective_batch_size",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
        "pin_memory",
        "retain_scientific_candidate_records",
        "retain_selected_trainable_state",
        "cleanup_unselected_partial_trainable_states_after_selection",
        "cleanup_recovery_state_after_completed_selection",
    ):
        _require_equal(fit.get(key), action_fit.get(key), f"matched fit {key}")
    _require_equal(fit.get("candidate_steps"), EXPECTED_CANDIDATE_STEPS, "candidate steps")

    variant = fit.get(EXPECTED_VARIANT, {})
    primary_variant = primary["fit"]["mature_official_default_r32"]
    action_variant = action_fit["mature_action_expert_lr25e6_recovery"]
    for key in (
        "adaptation_kind",
        "support_scope",
        "rank",
        "alpha",
        "dropout",
        "init_lora_weights",
        "expected_trainable_parameters",
        "optimizer",
        "betas",
        "epsilon",
        "weight_decay",
        "gradient_clip_norm",
        "scheduler",
        "scheduler_implementation",
        "warmup_steps",
        "decay_steps",
        "augmentation",
        "augmentation_scale_min",
        "augmentation_scale_max",
        "augmentation_ratio",
        "augmentation_seed",
        "seed",
        "target_modules",
    ):
        _require_equal(variant.get(key), primary_variant.get(key), f"LoRA variant {key}")
    for key in ("learning_rate", "decay_learning_rate"):
        _require_equal(variant.get(key), action_variant.get(key), f"matched schedule {key}")
    _require_equal(sorted(variant.get("target_modules", [])), mature_default_targets(), "exact targets")


def _validate_boundary(raw: dict[str, Any]) -> None:
    boundary = raw.get("action_expert_lr_boundary", {})
    expected = {
        "stage_step": 1_000,
        "task3_query_reduction_fraction": 0.07883101786715387,
        "task4_query_reduction_fraction": 0.03710404010162202,
        "median_query_reduction_fraction": 0.057967528984387945,
        "continuation_to_step2000": False,
        "formal_closed_loop_accessed": False,
        "authorizes_only_matched_lora_schedule": True,
        "does_not_authorize_gate_zero_or_writer": True,
    }
    for key, value in expected.items():
        _require_equal(boundary.get(key), value, f"action-expert boundary {key}")
    _require_equal(
        float(boundary.get("median_step750_query_reduction_fraction")),
        0.06974017372373628,
        "action-expert step-750 peak",
    )

    rollout = raw.get("screening_rollout", {})
    for key, expected_value in (
        ("role", "forbidden_until_headroom_safe_source_contract_is_predeclared"),
        ("access_authorized", False),
        ("requires_headroom_safe_source_contract", True),
        ("may_authorize_gate_zero", False),
        ("may_authorize_writer", False),
    ):
        _require_equal(rollout.get(key), expected_value, f"screening rollout {key}")

    bounded = raw.get("bounded_recovery", {})
    for key in (
        "same_lora_space_as_primary_mature_control",
        "same_schedule_as_successful_action_expert_lr_recovery",
        "no_second_learning_rate",
        "no_target_or_rank_search",
        "step1000_success_authorizes_only_headroom_safe_source_rollout_contract",
    ):
        _require_equal(bounded.get(key), True, f"bounded recovery {key}")

    provenance = raw.get("recipe_provenance", {})
    for key, expected_value in (
        ("lerobot_revision", "30da8e687a6dfc617fcd94afc367ac7071c376ce"),
        ("lerobot_tag", "v0.6.0"),
        (
            "same_data_sampler_noise_augmentation_optimizer_scheduler_and_seed_as_action_expert_recovery",
            True,
        ),
        ("same_37_target_rank32_alpha16_lora_space_as_primary_mature_control", True),
    ):
        _require_equal(provenance.get(key), expected_value, f"provenance {key}")


def load_mature_lora_lr_recovery_spec(
    path: Path,
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
) -> dict[str, Any]:
    """Load the one matched-LoRA test authorized by the action-expert recovery."""

    raw = _load_toml(path, "matched mature-LoRA lower-LR contract")
    _require_equal(raw.get("schema_version"), 1, "schema")
    _require_equal(raw.get("name"), EXPECTED_NAME, "name")
    _require_equal(raw.get("status"), EXPECTED_STATUS, "predeclaration")
    _require_equal(raw.get("task_ids"), [3, 4], "tasks")
    _require_equal(raw.get("variants"), [EXPECTED_VARIANT], "variants")
    _require_equal(raw.get("screening_stage"), EXPECTED_STAGE, "screening stage")
    _require_equal(raw.get("writer_authorized_before_closed_loop"), False, "Writer authority")

    primary_path = gate_zero_path.with_name("gate_zero_mature_lora_positive_control.toml")
    action_path = gate_zero_path.with_name("gate_zero_mature_action_expert_lr_recovery.toml")
    primary = load_mature_lora_positive_control_spec(
        primary_path,
        gate_zero_path=gate_zero_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
    )
    action_lr = load_mature_lora_positive_control_spec(
        action_path,
        gate_zero_path=gate_zero_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
    )
    parent = load_gate_zero_contract(gate_zero_path, phase0_path)
    _require_equal(raw["task_ids"], parent["data"]["task_ids"], "parent tasks")
    _validate_authority(
        raw,
        gate_zero_path=gate_zero_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
        action_lr=action_lr,
    )
    _validate_fit(raw, primary, action_lr)
    _require_equal(raw.get("selection"), primary.get("selection"), "query selection authority")
    _validate_boundary(raw)
    parallel = raw.get("parallel", {})
    _require_equal(
        parallel.get("fit_jobs"),
        [f"{EXPECTED_VARIANT}:3", f"{EXPECTED_VARIANT}:4"],
        "fit jobs",
    )
    _require_equal(parallel.get("maximum_concurrent_gpus"), 4, "GPU ceiling")
    _require_equal(parallel.get("one_independent_process_per_gpu"), True, "process topology")
    _require_equal(parallel.get("shared_parameters_across_jobs"), False, "shared parameters")
    _require_equal(raw.get("resources", {}).get("minimum_free_memory_mib"), 10_240, "OOM headroom")
    expected_peak = raw.get("resources", {}).get("expected_peak_device_memory_mib")
    if not isinstance(expected_peak, int) or expected_peak <= 0 or expected_peak >= 81_920 - 10_240:
        raise GateZeroMatureLoraLRContractError("expected device-memory budget is invalid")
    return raw
