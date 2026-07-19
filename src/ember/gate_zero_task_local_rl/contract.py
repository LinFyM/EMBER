"""Frozen contract and pure decisions for the Gate-0 task-local LoRA RL recovery."""

from __future__ import annotations

import hashlib
import math
import tomllib
from pathlib import Path
from typing import Any, Mapping

class GateZeroTaskLocalRLContractError(RuntimeError):
    """Raised when the task-local RL recovery differs from its sealed contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GateZeroTaskLocalRLContractError(f"task-local RL contract changed: {label}")


def _require_sha(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GateZeroTaskLocalRLContractError(f"invalid SHA256 authority: {label}")


def _load_toml(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroTaskLocalRLContractError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise GateZeroTaskLocalRLContractError(f"invalid {label}: {path}")
    return value


def _validate_upstream_hashes(
    spec: Mapping[str, Any],
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    fit_path: Path,
    headroom_path: Path,
    diagnostic_path: Path,
) -> dict[str, Any]:
    authority = spec.get("authority")
    if not isinstance(authority, dict):
        raise GateZeroTaskLocalRLContractError("task-local RL authority is missing")
    bindings = {
        "gate_zero_contract_sha256": gate_zero_path,
        "phase0_contract_sha256": phase0_path,
        "fit_contract_sha256": fit_path,
        "headroom_contract_sha256": headroom_path,
        "candidate_diagnostic_contract_sha256": diagnostic_path,
    }
    for key, path in bindings.items():
        _require_sha(authority.get(key), key)
        _require_equal(authority[key], _sha256(path), key)
    for key in (
        "headroom_result_sha256",
        "candidate_diagnostic_result_sha256",
        "previous_awr_result_sha256",
        "previous_signed_result_sha256",
        "previous_temporal_result_sha256",
        "source_base_checkpoint_manifest_sha256",
        "task3_supervised_manifest_sha256",
        "task3_supervised_state_sha256",
        "task4_supervised_manifest_sha256",
        "task4_supervised_state_sha256",
    ):
        _require_sha(authority.get(key), key)
    return authority


def _validate_lora_and_optimizer(spec: Mapping[str, Any], fit: Mapping[str, Any]) -> None:
    lora = spec.get("lora")
    algorithm = spec.get("algorithm")
    authority = spec["authority"]
    variant = fit.get("fit", {}).get(authority["fit_variant"])
    if not isinstance(lora, dict) or not isinstance(algorithm, dict) or not isinstance(variant, dict):
        raise GateZeroTaskLocalRLContractError("LoRA or algorithm authority is missing")
    expected_lora = {
        "target_count": len(variant["target_modules"]),
        "rank": variant["rank"],
        "alpha": variant["alpha"],
        "dropout": variant["dropout"],
        "trainable_parameters": variant["expected_trainable_parameters"],
    }
    for key, value in expected_lora.items():
        _require_equal(lora.get(key), value, f"lora.{key}")
    _require_equal(authority.get("supervised_step"), 1000, "supervised_step")
    _require_equal(authority.get("writer_present"), False, "writer_present")
    required_algorithm = {
        "name": "chunk_level_flow_ppo_with_task_local_critic_warmup_and_gae",
        "primary_source_fpo_plus": "https://arxiv.org/abs/2602.02481",
        "primary_source_fpo": "https://arxiv.org/abs/2507.21053",
        "primary_source_code_commit": "b80112be1e8362263c4cd176e7aef21a275ff1c6",
        "not_full_fpo_plus": True,
        "discount": 0.99,
        "gae_lambda": 0.99,
        "critic_only_rounds": 1,
        "critic": "task_local_frozen_feature_mlp",
        "critic_input_dim": 1953,
        "critic_hidden_dims": [512, 256],
        "critic_zero_output_initialization": True,
        "critic_learning_rate": 0.0001,
        "critic_betas": [0.9, 0.999],
        "critic_epsilon": 0.00001,
        "critic_weight_decay": 0.000001,
        "flow_samples_per_transition": 8,
        "flow_sample_group_size": 8,
        "ratio_clip": 0.01,
        "log_ratio_clamp": 5.0,
        "update_epochs_per_round": 10,
        "minibatch_size": 16,
        "target_kl": 0.1,
        "shared_parameter_updates": False,
        "writer_updates": False,
        "source_demonstration_actions_during_rl": False,
        "actor_optimizer": "adamw",
        "actor_learning_rate": 0.00001,
        "actor_betas": [0.9, 0.99],
        "actor_epsilon": 0.00001,
        "actor_weight_decay": 0.000001,
        "actor_gradient_clip_norm": 1.0,
        "critic_gradient_clip_norm": 1.0,
        "scheduler": "constant_over_bounded_recovery",
        "effective_replay_batch_size": 64,
        "anchors_per_episode": 8,
        "action_chunk_size": 50,
        "augmentation": "none_on_on_policy_observations",
    }
    for key, value in required_algorithm.items():
        _require_equal(algorithm.get(key), value, f"algorithm.{key}")


def _validate_recovery_provenance(spec: Mapping[str, Any]) -> None:
    provenance = spec.get("predecessor_evidence")
    if not isinstance(provenance, dict):
        raise GateZeroTaskLocalRLContractError("task-local RL recovery provenance is missing")
    expected = {
        "temporal_credit_contract_sha256": "0cfd1c74ced6b5cdc0e792d1af48555df6f2346527377cdc753ba46fc35955d2",
        "temporal_credit_result_sha256": "e13456343564880e6ef02d48119636774e0a06b783e6f4b0218692f104afa14c",
        "temporal_credit_status": "task_local_rl_early_check_not_supported",
        "temporal_credit_zero_init_paired_net_wins": [-1, 0],
        "temporal_credit_supervised_init_paired_net_wins": [0, 0],
    }
    for key, value in expected.items():
        _require_equal(provenance.get(key), value, f"predecessor_evidence.{key}")
    _require_equal(
        spec["authority"].get("previous_temporal_result_sha256"),
        expected["temporal_credit_result_sha256"],
        "authority.previous_temporal_result_sha256",
    )


def _validate_surfaces_and_decisions(spec: Mapping[str, Any]) -> None:
    _require_equal(spec.get("schema_version"), 1, "schema_version")
    _require_equal(
        spec.get("status"),
        "fpo_compatibility_critic_warmup_recovery_predeclared_after_temporal_credit_stop",
        "status",
    )
    _require_equal(spec.get("task_ids"), [3, 4], "task_ids")
    _require_equal(spec.get("initializations"), ["zero_init", "supervised_init"], "initializations")
    _require_equal(
        spec.get("reported_arms"),
        ["frozen_base", "supervised_lora", "zero_init_rl", "supervised_init_rl"],
        "reported_arms",
    )
    training = spec.get("training_interaction")
    development = spec.get("development_evaluation")
    safeguards = spec.get("offline_safeguards")
    continuation = spec.get("continuation")
    decision = spec.get("candidate_decision")
    fresh = spec.get("fresh_gate")
    parallel = spec.get("parallel")
    if not all(
        isinstance(value, dict)
        for value in (training, development, safeguards, continuation, decision, fresh, parallel)
    ):
        raise GateZeroTaskLocalRLContractError("task-local RL surface declaration is missing")
    _validate_recovery_provenance(spec)
    _require_equal(training["batch_size"], 8, "training batch_size")
    _require_equal(training["rounds_maximum"], 4, "training rounds_maximum")
    _require_equal(training["interaction_episode_nodes"], [8, 16, 24, 32], "interaction nodes")
    _require_equal(training["atomic_checkpoint_every_episodes"], 8, "checkpoint interval")
    _require_equal(
        training["train_init_state_indices_by_round"],
        [list(range(start, start + 8)) for start in (8, 16, 24, 32)],
        "training init states",
    )
    _require_equal(development["init_state_indices"], list(range(40, 48)), "development init states")
    _require_equal(development["evaluate_after_interaction_episodes"], [8, 16, 24, 32], "development nodes")
    _require_equal(development["frozen_base_successes_by_task"], [3, 3], "base J0")
    _require_equal(development["supervised_lora_successes_by_task"], [2, 4], "supervised J0")
    _require_equal(safeguards["maximum_action_drift_proxy"], 0.02, "drift safeguard")
    _require_equal(continuation["stage8_requires_exact_actor_identity_and_healthy_critic_warmup"], True, "stage8 continuation")
    _require_equal(continuation["stage16_continues_once_if_mechanics_and_temporal_credit_are_healthy"], True, "stage16 continuation")
    _require_equal(continuation["stage24_continue_requires_positive_aggregate_paired_net_gain_in_one_initialization"], True, "stage24 continuation")
    _require_equal(continuation["stage24_promising_arm_minimum_task_paired_net_win"], -1, "stage24 task floor")
    _require_equal(continuation["stage32_failure_stops_without_more_interaction"], True, "stage32 stop")
    _require_equal(continuation["stage40_and_later_are_outside_this_recovery"], True, "later boundary")
    _require_equal(
        continuation["nonterminal_statuses"],
        [
            "critic_warmup_complete_continue_to_16",
            "critic_warmup_recovery_continue_to_24",
            "critic_warmup_recovery_continue_to_32",
        ],
        "nonterminal statuses",
    )
    _require_equal(decision["minimum_each_task_success_gain_exclusive_pp"], 0.0, "task gain")
    _require_equal(decision["minimum_positive_task_count"], 2, "positive task count")
    _require_equal(decision["minimum_median_success_gain_pp"], 15.0, "median gain")
    _require_equal(decision["gate_zero_cannot_pass_on_development_evaluation"], True, "development Gate boundary")
    _require_equal(decision["writer_cannot_be_authorized_on_development_evaluation"], True, "Writer boundary")
    _require_equal(fresh["only_fresh_gate_may_authorize_gate_zero"], True, "fresh Gate boundary")
    _require_equal(parallel["world_size"], 4, "world_size")
    _require_equal(
        parallel["rank_assignments"],
        [
            "rank0:task3:zero_init",
            "rank1:task3:supervised_init",
            "rank2:task4:zero_init",
            "rank3:task4:supervised_init",
        ],
        "rank assignments",
    )
    for key in ("validation_numeric_access", "held_numeric_access", "locked_report_numeric_access"):
        _require_equal(spec["authority"].get(key), False, key)


def validate_task_local_rl_spec(
    spec: dict[str, Any],
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    fit_path: Path,
    headroom_path: Path,
    diagnostic_path: Path,
) -> dict[str, Any]:
    """Validate the result-blind four-arm source-only RL recovery."""

    _validate_surfaces_and_decisions(spec)
    _validate_upstream_hashes(
        spec,
        gate_zero_path=gate_zero_path,
        phase0_path=phase0_path,
        fit_path=fit_path,
        headroom_path=headroom_path,
        diagnostic_path=diagnostic_path,
    )
    fit = _load_toml(fit_path, "mature LoRA fit contract")
    _validate_lora_and_optimizer(spec, fit)
    exploration = spec.get("exploration", {})
    _require_equal(exploration.get("kind"), "native_stochastic_flow_sampling_with_no_external_action_noise", "exploration kind")
    _require_equal(exploration.get("standard_deviation"), [0.0] * 7, "exploration std")
    _require_equal(exploration.get("clip_low"), [-1.0] * 7, "exploration low")
    _require_equal(exploration.get("clip_high"), [1.0] * 7, "exploration high")
    _require_equal(exploration.get("external_action_noise"), False, "external action noise")
    _require_equal(exploration.get("maximum_saturation_fraction"), 0.0, "saturation safeguard")
    return spec


def load_task_local_rl_spec(
    path: Path,
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    fit_path: Path,
    headroom_path: Path,
    diagnostic_path: Path,
) -> dict[str, Any]:
    return validate_task_local_rl_spec(
        _load_toml(path, "task-local RL contract"),
        gate_zero_path=gate_zero_path,
        phase0_path=phase0_path,
        fit_path=fit_path,
        headroom_path=headroom_path,
        diagnostic_path=diagnostic_path,
    )


def assigned_task_local_rl_arm(
    *, rank: int, world_size: int, spec: Mapping[str, Any]
) -> tuple[int, str]:
    assignments = [(3, "zero_init"), (3, "supervised_init"), (4, "zero_init"), (4, "supervised_init")]
    if world_size != spec["parallel"]["world_size"] or not 0 <= rank < world_size:
        raise GateZeroTaskLocalRLContractError("task-local RL rank topology changed")
    return assignments[rank]


def _arm_passes(nets: list[int], spec: Mapping[str, Any]) -> bool:
    if len(nets) != 2 or any(not isinstance(value, int) or isinstance(value, bool) for value in nets):
        raise GateZeroTaskLocalRLContractError("invalid paired task-local RL gains")
    gains = [value * 100.0 / 8.0 for value in nets]
    median = sum(sorted(gains)) / 2.0
    threshold = spec["candidate_decision"]
    return (
        sum(value > threshold["minimum_each_task_success_gain_exclusive_pp"] for value in gains)
        >= threshold["minimum_positive_task_count"]
        and median >= threshold["minimum_median_success_gain_pp"]
    )


def decide_task_local_rl_node(
    spec: Mapping[str, Any], *, interaction_episodes: int, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the frozen four-node critic-warmup recovery selection rule."""

    if interaction_episodes not in spec["training_interaction"]["interaction_episode_nodes"]:
        raise GateZeroTaskLocalRLContractError("unknown task-local RL decision node")
    nets = metrics.get("paired_net_wins_by_arm")
    drift = metrics.get("action_drift_by_arm")
    if not isinstance(nets, dict) or set(nets) != {"zero_init_rl", "supervised_init_rl"}:
        raise GateZeroTaskLocalRLContractError("RL decision lacks both matched arms")
    if not isinstance(drift, dict) or set(drift) != set(nets):
        raise GateZeroTaskLocalRLContractError("RL decision lacks drift safeguards")
    passed = [arm for arm, values in nets.items() if _arm_passes(values, spec)]
    warmup_identity = metrics.get("critic_warmup_actor_state_unchanged") is True
    safeguards = (
        metrics.get("mechanics_valid") is True
        and metrics.get("temporal_credit_healthy") is True
        and metrics.get("nonfinite_count") == 0
        and isinstance(metrics.get("maximum_saturation_fraction"), (int, float))
        and metrics["maximum_saturation_fraction"] <= spec["exploration"]["maximum_saturation_fraction"]
        and all(
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            and value <= spec["offline_safeguards"]["maximum_action_drift_proxy"]
            for value in drift.values()
        )
        and (interaction_episodes != 8 or warmup_identity)
        and (
            interaction_episodes != 8
            or all(value == 0 for values in nets.values() for value in values)
        )
    )
    status: str
    selected: int | None = None
    if not safeguards:
        status = "task_local_rl_mechanical_or_safeguard_failure"
    elif passed:
        status = "rl_candidate_selected_for_fresh_gate"
        selected = interaction_episodes
    elif interaction_episodes == 8:
        status = "critic_warmup_complete_continue_to_16"
    elif interaction_episodes == 16:
        status = "critic_warmup_recovery_continue_to_24"
    elif interaction_episodes == 24 and any(
        sum(values) > 0
        and min(values) >= spec["continuation"]["stage24_promising_arm_minimum_task_paired_net_win"]
        for values in nets.values()
    ):
        status = "critic_warmup_recovery_continue_to_32"
    else:
        status = "task_local_rl_early_check_not_supported"
    return {
        "status": status,
        "interaction_episodes": interaction_episodes,
        "selected_interaction_episodes": selected,
        "passing_initializations": passed,
        "fresh_gate_grant_authorized": selected is not None,
        "gate_zero_authorized": False,
        "writer_authorized": False,
        "validation_numeric_access": False,
        "held_numeric_access": False,
    }
