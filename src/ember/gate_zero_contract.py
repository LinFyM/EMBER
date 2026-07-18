"""Fail-closed loader for the predeclared Gate 0 source-only pilot."""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any


class GateZeroContractError(RuntimeError):
    """Raised when the Gate 0 pilot no longer matches its sealed authority."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inclusive(bounds: list[int], label: str) -> set[int]:
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or not all(isinstance(value, int) for value in bounds)
        or bounds[0] < 0
        or bounds[1] < bounds[0]
    ):
        raise GateZeroContractError(f"invalid {label} episode bounds")
    return set(range(bounds[0], bounds[1] + 1))


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GateZeroContractError(f"{label} changed: {actual!r} != {expected!r}")


def _validate_episode_partitions(spec: dict[str, Any], phase0: dict[str, Any]) -> None:
    declared = {
        "source_base_fit": spec["access"]["source_base_fit"],
        "oracle_support": spec["access"]["oracle_support"],
        "functional_query": spec["access"]["functional_query"],
        "locked_source_report": spec["access"]["locked_source_report"],
    }
    surfaces = {name: _inclusive(bounds, name) for name, bounds in declared.items()}
    if len(set().union(*surfaces.values())) != sum(len(values) for values in surfaces.values()):
        raise GateZeroContractError("episode partitions overlap")
    for name, bounds in declared.items():
        _require_equal(bounds, phase0["episode_authority"][name], f"episode authority {name}")
    _require_equal(spec["base_fit"]["episode_bounds"], declared["source_base_fit"], "base-fit bounds")
    _require_equal(spec["oracle"]["support_episode_bounds"], declared["oracle_support"], "support bounds")
    _require_equal(
        spec["oracle"]["selection"]["episode_bounds"], declared["functional_query"], "query bounds"
    )
    _require_equal(
        spec["report"]["offline_episode_bounds"], declared["locked_source_report"], "report bounds"
    )


def _validate_scientific_surface(spec: dict[str, Any], phase0: dict[str, Any]) -> None:
    source = set(phase0["splits"]["source"])
    validation = set(phase0["splits"]["validation"])
    held = set(phase0["splits"]["held_out"])
    task_ids = spec["data"]["task_ids"]
    _require_equal(task_ids, spec["oracle"]["task_ids"], "oracle pilot tasks")
    _require_equal(task_ids, spec["base_competence"]["task_ids"], "competence pilot tasks")
    if len(task_ids) != len(set(task_ids)) or not set(task_ids) <= source:
        raise GateZeroContractError("Gate 0 pilot tasks must be unique resealed source tasks")
    if set(task_ids) & (validation | held):
        raise GateZeroContractError("Gate 0 pilot may not use validation or held tasks")
    _require_equal(spec["access"]["forbidden_numeric_splits"], ["validation", "held_out"], "forbidden splits")
    _require_equal(spec["access"]["fully_pristine_all_fields"], [48, 49], "fully pristine rows")
    _require_equal(spec["access"]["action_reward_policy_outcome_locked"], [46, 49], "modality lock")
    _require_equal(spec["access"]["prior_rgb_only_exposure"], [40, 47], "prior RGB disclosure")
    if spec["oracle"]["selection"]["report_access_before_selection_freeze"]:
        raise GateZeroContractError("report access is forbidden before oracle selection freeze")
    if not spec["report"]["no_selection_after_report_access"]:
        raise GateZeroContractError("report access must permanently stop model selection")


def _validate_oracle(spec: dict[str, Any]) -> None:
    oracle = spec["oracle"]
    _require_equal(oracle["rank"], 8, "primary oracle rank")
    _require_equal(oracle["alpha"], 8, "primary oracle alpha")
    _require_equal(oracle["dropout"], 0.0, "primary oracle dropout")
    _require_equal(oracle["init_lora_weights"], True, "functional-zero LoRA initialization")
    _require_equal(
        oracle["initial_physical_delta_required_exact_zero"],
        True,
        "initial physical-delta requirement",
    )
    _require_equal(
        oracle["initial_fixed_loss_required_exact_base"],
        True,
        "initial fixed-loss requirement",
    )
    _require_equal(oracle["modules_to_save"], [], "modules_to_save")
    targets = oracle["target_modules"]
    expected_targets = [
        "model.vlm_with_expert.lm_expert.layers.14.self_attn.q_proj",
        "model.vlm_with_expert.lm_expert.layers.14.self_attn.v_proj",
        "model.vlm_with_expert.lm_expert.layers.15.self_attn.q_proj",
        "model.vlm_with_expert.lm_expert.layers.15.self_attn.v_proj",
    ]
    _require_equal(targets, expected_targets, "primary oracle target matrices")
    _require_equal(oracle["expected_trainable_parameters"], 40320, "oracle parameter count")
    if oracle["checkpoint_candidates"][0] != 0 or oracle["checkpoint_candidates"][-1] != oracle["steps"]:
        raise GateZeroContractError("oracle checkpoint candidates must include base and final steps")


def _validate_thresholds(spec: dict[str, Any], phase0: dict[str, Any]) -> None:
    gate = phase0["gate_zero"]["thresholds"]
    threshold = spec["thresholds"]
    _require_equal(threshold["median_success_gain_pp"], gate["median_success_gain_pp"], "success threshold")
    _require_equal(
        threshold["median_locked_action_loss_reduction_fraction"],
        gate["median_query_action_loss_reduction_fraction"],
        "action-loss threshold",
    )
    _require_equal(threshold["positive_task_fraction"], gate["positive_task_fraction"], "positive-task threshold")
    _require_equal(threshold["median_action_kl_proxy_max"], gate["median_action_kl_max"], "drift threshold")
    if spec["diagnostics"]["exact_policy_likelihood_claimed"]:
        raise GateZeroContractError("SmolVLA pilot must not claim an exact policy likelihood")


def _validate_batch_calibration(spec: dict[str, Any]) -> None:
    base_fit = spec["base_fit"]
    calibration = base_fit["batch_calibration"]
    _require_equal(base_fit["effective_batch_size"], 64, "base effective batch")
    _require_equal(
        calibration["micro_batch_candidates"], [8, 16, 32, 64], "microbatch candidates"
    )
    _require_equal(
        calibration["technical_steps_per_candidate"],
        calibration["warmup_optimizer_steps_per_candidate"]
        + calibration["measured_optimizer_steps_per_candidate"],
        "calibration step partition",
    )
    if any(64 % value for value in calibration["micro_batch_candidates"]):
        raise GateZeroContractError("each calibration microbatch must divide effective batch 64")
    _require_equal(
        calibration["candidate_order"],
        "ascending_stop_after_first_oom",
        "calibration candidate order",
    )
    _require_equal(
        calibration["reuse_single_model_process"],
        True,
        "single-model-process calibration reuse",
    )
    for key in (
        "restore_identical_trainable_state_before_each_candidate",
        "new_empty_optimizer_before_each_candidate",
        "reset_identical_global_rng_before_each_candidate",
        "matched_effective_batch_draws_across_candidates",
        "matched_fixed_flow_noise_and_time_across_candidates",
    ):
        _require_equal(calibration[key], True, f"matched calibration {key}")
    _require_equal(
        calibration["effective_batch_draw_algorithm"],
        "absolute_optimizer_step_and_effective_batch_slot_v2",
        "effective-batch draw algorithm",
    )
    _require_equal(
        calibration["include_data_loading_in_timing"], True, "calibration timing surface"
    )
    _require_equal(calibration["outcome_metrics_forbidden"], True, "outcome metric ban")
    _require_equal(calibration["stop_on_first_oom"], True, "calibration OOM stop")
    if calibration["minimum_free_memory_mib"] < 10240:
        raise GateZeroContractError("calibration memory headroom weakened")
    _validate_batch_calibration_selection(calibration)


def _validate_batch_calibration_selection(calibration: dict[str, Any]) -> None:
    selection = calibration["selection_authority"]
    _require_equal(selection["selected_micro_batch_size"], 64, "selected microbatch")
    _require_equal(
        selection["selected_gradient_accumulation_steps"], 1, "selected accumulation"
    )
    if (
        not isinstance(selection["result_sha256"], str)
        or len(selection["result_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in selection["result_sha256"])
    ):
        raise GateZeroContractError("invalid calibration result SHA256")
    _require_equal(
        selection["scientific_outcome_metrics_recorded"], False, "calibration outcome ban"
    )
    if selection["status"] not in {
        "superseded_pending_matched_recovery",
        "frozen_matched_resource_authority",
    }:
        raise GateZeroContractError("unknown calibration selection authority state")
    if selection["status"] == "superseded_pending_matched_recovery":
        _require_equal(selection["authorized_as_batch_shape"], False, "batch-shape authorization")
        _require_equal(selection["formal_base_fit_authorized"], False, "formal-fit authorization")
        _require_equal(selection["matched_initial_trainable_state"], False, "prior state match")
        _require_equal(selection["matched_effective_batch_draws"], False, "prior draw match")
    else:
        _require_equal(selection["authorized_as_batch_shape"], True, "batch-shape authorization")
        _require_equal(selection["formal_base_fit_authorized"], False, "formal-fit authorization")
        _require_equal(selection["matched_initial_trainable_state"], True, "state match")
        _require_equal(selection["matched_effective_batch_draws"], True, "draw match")
        _require_equal(selection["matched_flow_noise_and_time"], True, "flow-input match")
        _require_equal(
            selection["parameter_dtype_elements_bfloat16"], 96607440, "bf16 parameter count"
        )
        _require_equal(
            selection["parameter_dtype_elements_float32"], 3273552, "fp32 parameter count"
        )
        _require_equal(
            selection["adamw_moments_follow_parameter_dtype"], True, "AdamW state dtype"
        )
        prior = calibration["prior_diagnostic"]
        _require_equal(prior["matched_initial_trainable_state"], False, "prior state mismatch")
        _require_equal(prior["matched_effective_batch_draws"], False, "prior draw mismatch")


def _validate_base_checkpoint(spec: dict[str, Any]) -> None:
    base_fit = spec["base_fit"]
    checkpoint = base_fit["checkpoint"]
    selection = base_fit["batch_calibration"]["selection_authority"]
    _require_equal(checkpoint["world_size"], 1, "base-fit world size")
    _require_equal(
        base_fit["scheduler_implementation"],
        "lerobot.optim.schedulers.CosineDecayWithWarmupSchedulerConfig",
        "scheduler implementation",
    )
    _require_equal(
        base_fit["precision"],
        "bfloat16_autocast_mixed_parameter_native_adamw_state",
        "measured precision authority",
    )
    _require_equal(checkpoint["num_workers"], 4, "base-fit worker count")
    _require_equal(
        checkpoint["checkpoint_every_steps"], base_fit["checkpoint_every_steps"], "checkpoint cadence"
    )
    _require_equal(
        checkpoint["recoverable_checkpoints_to_keep"],
        base_fit["recoverable_checkpoints_to_keep"],
        "checkpoint retention",
    )
    _require_equal(
        checkpoint["scientific_checkpoint_step"],
        base_fit["scientific_checkpoint_step"],
        "scientific checkpoint",
    )
    for key in (
        "atomic_directory_rename",
        "save_optimizer_scheduler_rng",
        "hash_every_retained_file",
        "resume_requires_identical_batch_topology",
        "resume_requires_identical_contract_and_calibration_hashes",
    ):
        _require_equal(checkpoint[key], True, f"checkpoint {key}")
    probe = base_fit["resume_probe"]
    _require_equal(probe["uninterrupted_target_step"], 2, "resume probe target")
    _require_equal(probe["interrupted_checkpoint_step"], 1, "resume probe checkpoint")
    _require_equal(probe["micro_batch_size"], selection["selected_micro_batch_size"], "probe microbatch")
    _require_equal(
        probe["gradient_accumulation_steps"],
        selection["selected_gradient_accumulation_steps"],
        "probe accumulation",
    )
    for key in (
        "exact_model_tensor_equality",
        "exact_optimizer_scheduler_rng_equality",
        "exact_next_sampler_batch_equality",
        "cleanup_transient_full_checkpoints_after_verification",
    ):
        _require_equal(probe[key], True, f"resume probe {key}")
    _require_equal(probe["scientific_outcome_metrics_recorded"], False, "resume outcome ban")


def _validate_resources(spec: dict[str, Any], phase0: dict[str, Any]) -> None:
    resources = spec["resources"]
    _require_equal(resources["maximum_concurrent_gpus"], phase0["resources"]["max_concurrent_gpus"], "GPU ceiling")
    if resources["maximum_concurrent_gpus"] > 4 or resources["minimum_free_memory_mib"] < 10240:
        raise GateZeroContractError("Gate 0 resource headroom contract weakened")
    if resources["pilot_gpus"] > resources["maximum_concurrent_gpus"]:
        raise GateZeroContractError("Gate 0 pilot exceeds the GPU ceiling")


def load_gate_zero_contract(
    path_or_text: Path | str,
    phase0_path: Path,
    *,
    from_text: bool = False,
) -> dict[str, Any]:
    """Load and cross-check the Gate 0 contract against the permanent Phase 0 seal."""

    try:
        if from_text:
            spec = tomllib.loads(str(path_or_text))
        else:
            spec = tomllib.loads(Path(path_or_text).read_text(encoding="utf-8"))
        phase0 = tomllib.loads(phase0_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroContractError("invalid Gate 0 or Phase 0 TOML") from error
    if spec.get("schema_version") != 1:
        raise GateZeroContractError("unsupported Gate 0 contract schema")
    if spec.get("status") != "predeclared_before_source_policy_outcomes":
        raise GateZeroContractError("Gate 0 contract is not frozen before outcomes")
    _require_equal(spec["authority"]["phase0_contract_sha256"], _sha256(phase0_path), "Phase 0 SHA256")
    _validate_episode_partitions(spec, phase0)
    _validate_scientific_surface(spec, phase0)
    _validate_oracle(spec)
    _validate_batch_calibration(spec)
    _validate_base_checkpoint(spec)
    _validate_thresholds(spec, phase0)
    _validate_resources(spec, phase0)
    if spec["recovery"]["threshold_changes_forbidden"] is not True:
        raise GateZeroContractError("Gate 0 recovery may not change thresholds")
    if spec["recovery"]["held_access_forbidden"] is not True:
        raise GateZeroContractError("Gate 0 recovery may not inspect held data")
    if spec["recovery"]["writer_authorized_by_pilot"] is not False:
        raise GateZeroContractError("the two-task pilot cannot authorize Writer training")
    return spec
