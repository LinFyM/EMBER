"""Fail-closed additive execution contract for Gate 0 oracle evidence."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from ember.gate_zero_checkpoint import CHECKPOINT_MANIFEST, validate_source_base_checkpoint
from ember.gate_zero_contract import load_gate_zero_contract


class GateZeroOracleContractError(RuntimeError):
    """Raised when the additive oracle execution contract changes authority."""


MATURE_CONTROL_NAMES = frozenset(
    {
        "smolvla_libero90_gate_zero_mature_lora_positive_control_v1",
        "smolvla_libero90_gate_zero_mature_lora_all_linear_recovery_v1",
        "smolvla_libero90_gate_zero_mature_action_expert_upper_bound_v1",
        "smolvla_libero90_gate_zero_mature_action_expert_lr_recovery_v1",
        "smolvla_libero90_gate_zero_mature_lora_lr_recovery_v1",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GateZeroOracleContractError(f"{label} changed: {actual!r} != {expected!r}")


def _validate_hash_authority(
    authority: dict[str, Any], key: str, path: Path, label: str
) -> None:
    expected = authority.get(key)
    if not isinstance(expected, str) or len(expected) != 64 or _sha256(path) != expected:
        raise GateZeroOracleContractError(f"{label} SHA256 changed")


def _validate_fit(spec: dict[str, Any], parent: dict[str, Any]) -> None:
    fit = spec.get("fit", {})
    oracle = parent["oracle"]
    _require_equal(fit.get("support_episode_bounds"), oracle["support_episode_bounds"], "support bounds")
    _require_equal(fit.get("optimizer_steps"), oracle["steps"], "oracle optimizer steps")
    _require_equal(fit.get("effective_batch_size"), oracle["effective_batch_size"], "effective batch")
    _require_equal(fit.get("candidate_steps"), oracle["checkpoint_candidates"], "candidate steps")
    if (
        fit.get("micro_batch_size", 0) <= 0
        or fit.get("gradient_accumulation_steps", 0) <= 0
        or fit["micro_batch_size"] * fit["gradient_accumulation_steps"]
        != fit["effective_batch_size"]
    ):
        raise GateZeroOracleContractError("oracle batch partition changed")
    if fit.get("num_workers", -1) < 0 or fit.get("prefetch_factor", 0) <= 0:
        raise GateZeroOracleContractError("oracle loader resources are invalid")
    for required in (
        "persistent_workers",
        "pin_memory",
        "retain_scientific_candidate_records",
        "retain_selected_trainable_state",
        "cleanup_unselected_partial_trainable_states_after_selection",
        "cleanup_recovery_state_after_completed_selection",
    ):
        _require_equal(fit.get(required), True, required)

    lora = fit.get("lora", {})
    for key, parent_key in (
        ("rank", "rank"),
        ("alpha", "alpha"),
        ("dropout", "dropout"),
        ("expected_trainable_parameters", "expected_trainable_parameters"),
        ("optimizer", "optimizer"),
        ("learning_rate", "learning_rate"),
        ("betas", "betas"),
        ("epsilon", "epsilon"),
        ("weight_decay", "weight_decay"),
        ("gradient_clip_norm", "gradient_clip_norm"),
        ("seed", "seed"),
    ):
        _require_equal(lora.get(key), oracle[parent_key], f"LoRA {key}")

    upper = fit.get("partial_upper_bound", {})
    _require_equal(upper.get("matched_baseline"), False, "partial upper-bound matched status")
    _require_equal(upper.get("may_authorize_writer"), False, "partial Writer authority")
    _require_equal(upper.get("may_replace_lora_primary"), False, "partial primary authority")
    if upper.get("expected_trainable_parameters", 0) <= lora["expected_trainable_parameters"]:
        raise GateZeroOracleContractError("partial upper bound is not a larger capacity diagnostic")


def _validate_selection_and_report(spec: dict[str, Any], parent: dict[str, Any]) -> None:
    selection = spec.get("selection", {})
    parent_selection = parent["oracle"]["selection"]
    for key, parent_key in (
        ("query_episode_bounds", "episode_bounds"),
        ("fixed_noise_seed", "fixed_noise_seed"),
        ("fixed_time_seed", "fixed_time_seed"),
        ("noise_draws", "noise_draws"),
        ("candidate_rule", "candidate_rule"),
        ("anchor_frames_per_demo", "action_drift_anchor_frames_per_demo"),
        ("drift_proxy_max", "action_drift_proxy_max"),
    ):
        _require_equal(selection.get(key), parent_selection[parent_key], f"selection {key}")
    _require_equal(selection.get("selection_uses_locked_report"), False, "selection report access")
    _require_equal(selection.get("step_zero_must_be_functional_zero"), True, "step-zero mechanics")
    _require_equal(
        selection.get("anchor_algorithm"),
        "round_evenly_spaced_inclusive_first_last_then_unique",
        "anchor algorithm",
    )
    _require_equal(
        selection.get("drift_formula"),
        "mean_over_anchor_chunk_action_scalars_of_0.5_times_squared_adapter_minus_frozen_base_normalized_action",
        "drift formula",
    )

    report = spec.get("report", {})
    parent_report = parent["report"]
    for key, parent_key in (
        ("offline_episode_bounds", "offline_episode_bounds"),
        ("official_rollout_init_state_indices", "official_rollout_init_state_indices"),
        ("rollout_batch_size", "rollout_batch_size"),
        ("rollout_async_envs", "rollout_async_envs"),
        ("policy_rng_seed", "inference_noise_seed"),
        ("no_selection_after_report_access", "no_selection_after_report_access"),
    ):
        _require_equal(report.get(key), parent_report[parent_key], f"report {key}")
    _require_equal(report.get("primary_arms"), parent_report["rollout_conditions"], "primary report arms")
    _require_equal(report.get("capacity_diagnostic_arms"), ["partial_upper_bound"], "capacity report arm")
    if report.get("bootstrap_replicates", 0) < 1000:
        raise GateZeroOracleContractError("report bootstrap replicate count is too small")


def _validate_decision_and_resources(spec: dict[str, Any], parent: dict[str, Any]) -> None:
    decision = spec.get("decision", {})
    thresholds = parent["thresholds"]
    for key, parent_key in (
        ("median_success_gain_pp_min", "median_success_gain_pp"),
        (
            "median_locked_action_loss_reduction_fraction_min",
            "median_locked_action_loss_reduction_fraction",
        ),
        ("positive_task_fraction_min", "positive_task_fraction"),
        ("median_selection_drift_proxy_max", "median_action_kl_proxy_max"),
    ):
        _require_equal(decision.get(key), thresholds[parent_key], f"decision {key}")
    _require_equal(decision.get("two_task_positive_count_required"), 2, "positive task count")
    _require_equal(
        decision.get("writer_authorized_by_this_two_task_pilot"),
        parent["recovery"]["writer_authorized_by_pilot"],
        "pilot Writer authority",
    )
    _require_equal(spec.get("parallel", {}).get("maximum_concurrent_gpus"), 4, "GPU ceiling")
    _require_equal(
        spec.get("resources", {}).get("minimum_free_memory_mib"),
        parent["resources"]["minimum_free_memory_mib"],
        "memory headroom",
    )


def load_oracle_execution_spec(
    path: Path,
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
) -> dict[str, Any]:
    """Load the additive contract and bind it to every prior sealed authority."""

    try:
        with path.open("rb") as handle:
            spec = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroOracleContractError("invalid oracle execution TOML") from error
    if spec.get("schema_version") != 1:
        raise GateZeroOracleContractError("unsupported oracle execution schema")
    _require_equal(
        spec.get("status"),
        "predeclared_after_source_competence_before_oracle_fit_or_locked_report_access",
        "oracle execution predeclaration",
    )
    authority = spec.get("authority", {})
    _validate_hash_authority(authority, "gate_zero_contract_sha256", gate_zero_path, "Gate 0 contract")
    _validate_hash_authority(authority, "phase0_contract_sha256", phase0_path, "Phase 0 contract")
    _validate_hash_authority(
        authority,
        "source_competence_contract_sha256",
        competence_path,
        "source competence contract",
    )
    _require_equal(authority.get("validation_numeric_access"), False, "validation access")
    _require_equal(authority.get("held_numeric_access"), False, "held access")

    parent = load_gate_zero_contract(gate_zero_path, phase0_path)
    _require_equal(spec.get("task_ids"), parent["data"]["task_ids"], "oracle tasks")
    _require_equal(spec.get("variants"), ["lora", "partial_upper_bound"], "oracle variants")
    _validate_fit(spec, parent)
    _validate_selection_and_report(spec, parent)
    _validate_decision_and_resources(spec, parent)
    return spec


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateZeroOracleContractError(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise GateZeroOracleContractError(f"invalid {label}")
    return value


def _require_hashed_artifact(
    output_root: Path,
    authority: dict[str, Any],
    *,
    relative_key: str,
    sha_key: str,
    label: str,
) -> Path:
    artifact = output_root / Path(authority[relative_key])
    if not artifact.is_file() or _sha256(artifact) != authority[sha_key]:
        raise GateZeroOracleContractError(f"{label} authority changed")
    return artifact


def _validate_task_artifact_group(
    output_root: Path,
    authority: dict[str, Any],
    *,
    prefix: str,
    label: str,
) -> None:
    for task_id in (3, 4):
        for kind in ("candidate_manifest", "recovery_manifest", "telemetry"):
            _require_hashed_artifact(
                output_root,
                authority,
                relative_key=f"{prefix}_task{task_id}_{kind}_relative_path",
                sha_key=f"{prefix}_task{task_id}_{kind}_sha256",
                label=f"{label} task-{task_id} {kind}",
            )


def _validate_update_scale_probe(
    output_root: Path, authority: dict[str, Any], *, task_id: int
) -> None:
    paths = {}
    for kind in ("result", "source"):
        paths[kind] = _require_hashed_artifact(
            output_root,
            authority,
            relative_key=f"task{task_id}_update_scale_probe_{kind}_relative_path",
            sha_key=f"task{task_id}_update_scale_probe_{kind}_sha256",
            label=f"update-scale task-{task_id} {kind}",
        )
    probe = _load_json(paths["result"], f"update-scale task-{task_id} result")
    scope = probe.get("scope", {})
    if (
        probe.get("status") != "source_query_update_scale_diagnostic_complete"
        or probe.get("task_id") != task_id
        or not all(probe.get("endpoint_checks", {}).values())
        or scope.get("diagnostic_only") is not True
        or scope.get("formal_closed_loop_accessed") is not False
        or scope.get("validation_numeric_access") is not False
        or scope.get("held_numeric_access") is not False
    ):
        raise GateZeroOracleContractError(
            f"update-scale task-{task_id} result contract changed"
        )


def _validate_mature_prior_artifacts(
    spec: dict[str, Any],
    authority: dict[str, Any],
    competence_result_path: Path,
) -> None:
    competence_relative = Path(authority["source_competence_result_relative_path"])
    if tuple(competence_result_path.parts[-len(competence_relative.parts) :]) != competence_relative.parts:
        raise GateZeroOracleContractError("mature control output-root authority changed")
    output_root = competence_result_path
    for _ in competence_relative.parts:
        output_root = output_root.parent
    prior = _require_hashed_artifact(
        output_root,
        authority,
        relative_key="prior_rank16_screening_result_relative_path",
        sha_key="prior_rank16_screening_result_sha256",
        label="rank-16 screening result",
    )
    if _load_json(prior, "rank-16 screening result").get("status") != authority[
        "prior_rank16_status"
    ]:
        raise GateZeroOracleContractError("rank-16 screening result status changed")
    name = spec.get("name")
    if name == "smolvla_libero90_gate_zero_mature_lora_all_linear_recovery_v1":
        _validate_task_artifact_group(
            output_root, authority, prefix="primary", label="primary"
        )
    elif name == "smolvla_libero90_gate_zero_mature_action_expert_upper_bound_v1":
        _validate_task_artifact_group(
            output_root, authority, prefix="all_linear", label="all-linear"
        )
    elif name == "smolvla_libero90_gate_zero_mature_action_expert_lr_recovery_v1":
        _validate_task_artifact_group(
            output_root, authority, prefix="upper_bound", label="upper-bound"
        )
        for task_id in (3, 4):
            _validate_update_scale_probe(output_root, authority, task_id=task_id)
    elif name == "smolvla_libero90_gate_zero_mature_lora_lr_recovery_v1":
        _validate_task_artifact_group(
            output_root,
            authority,
            prefix="lr_action_expert",
            label="lower-LR action-expert",
        )


def load_oracle_fit_spec(
    path: Path,
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
) -> dict[str, Any]:
    """Dispatch frozen pilot and support-audit configs to their strict validators."""

    try:
        with path.open("rb") as handle:
            name = tomllib.load(handle).get("name")
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroOracleContractError("invalid oracle fit TOML") from error
    if name == "smolvla_libero90_gate_zero_oracle_execution_v1":
        return load_oracle_execution_spec(
            path,
            gate_zero_path=gate_zero_path,
            phase0_path=phase0_path,
            competence_path=competence_path,
        )
    if name in {
        "smolvla_libero90_gate_zero_target_support_audit_v1",
        "smolvla_libero90_gate_zero_target_support_rank16_v1",
    } | MATURE_CONTROL_NAMES:
        from ember.gate_zero_support.contract import (
            load_target_support_screen_spec,
        )

        return load_target_support_screen_spec(
            path,
            gate_zero_path=gate_zero_path,
            phase0_path=phase0_path,
            competence_path=competence_path,
            prior_execution_path=path.with_name("gate_zero_oracle_execution.toml"),
        )
    raise GateZeroOracleContractError("unknown oracle fit contract")


def validate_oracle_fit_prerequisites(
    *,
    config_path: Path,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
    competence_result_path: Path,
    source_base_checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Bind a fit job to source competence and the immutable base checkpoint."""

    spec = load_oracle_fit_spec(
        config_path,
        gate_zero_path=gate_zero_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
    )
    parent = load_gate_zero_contract(gate_zero_path, phase0_path)
    with phase0_path.open("rb") as handle:
        phase0 = tomllib.load(handle)
    authority = spec["authority"]
    if _sha256(competence_result_path) != authority["source_competence_result_sha256"]:
        raise GateZeroOracleContractError("source competence result SHA256 changed")
    if spec.get("name") in MATURE_CONTROL_NAMES:
        _validate_mature_prior_artifacts(spec, authority, competence_result_path)
    competence = _load_json(competence_result_path, "source competence result")
    decision = competence.get("decision", {})
    if (
        competence.get("status") != "source_competence_passed"
        or decision.get("task_local_oracle_fit_authorized") is not True
        or decision.get("writer_authorized") is not False
    ):
        raise GateZeroOracleContractError("source competence did not authorize task-local fitting")
    checkpoint = validate_source_base_checkpoint(
        source_base_checkpoint,
        expected={
            "step": authority["source_base_checkpoint_step"],
            "checkpoint_role": authority["source_base_checkpoint_role"],
        },
    )
    manifest_path = source_base_checkpoint / CHECKPOINT_MANIFEST
    if _sha256(manifest_path) != authority["source_base_checkpoint_manifest_sha256"]:
        raise GateZeroOracleContractError("source-base checkpoint manifest SHA256 changed")
    competence_checkpoint = competence.get("checkpoint", {})
    if (
        competence_checkpoint.get("checkpoint_manifest_sha256")
        != authority["source_base_checkpoint_manifest_sha256"]
        or competence_checkpoint.get("checkpoint_step")
        != authority["source_base_checkpoint_step"]
    ):
        raise GateZeroOracleContractError("competence/checkpoint authority changed")
    return spec, parent, phase0, checkpoint


def oracle_fit_authorities(
    *,
    config_path: Path,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
    competence_result_path: Path,
    source_base_checkpoint: Path,
    manifest_path: Path,
    spec: dict[str, Any],
    parent: dict[str, Any],
) -> dict[str, Any]:
    return {
        "execution_contract_sha256": _sha256(config_path),
        "gate_zero_contract_sha256": _sha256(gate_zero_path),
        "phase0_contract_sha256": _sha256(phase0_path),
        "source_competence_contract_sha256": _sha256(competence_path),
        "source_competence_result_sha256": _sha256(competence_result_path),
        "source_base_checkpoint_manifest_sha256": _sha256(
            source_base_checkpoint / CHECKPOINT_MANIFEST
        ),
        "canonical_manifest_sha256": _sha256(manifest_path),
        "canonical_manifest_declared_sha256": parent["authority"][
            "canonical_manifest_sha256"
        ],
        "source_normalization_sha256": parent["authority"][
            "source_normalization_sha256"
        ],
        "validation_numeric_access": spec["authority"]["validation_numeric_access"],
        "held_numeric_access": spec["authority"]["held_numeric_access"],
    }
