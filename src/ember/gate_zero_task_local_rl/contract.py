"""Frozen contract and pure decisions for the Gate-0 task-local LoRA RL recovery."""

from __future__ import annotations

import hashlib
import math
import tomllib
from pathlib import Path
from typing import Any, Mapping

import torch


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
        "name": "episodic_awr_style_monte_carlo_reward_weighted_flow_regression",
        "reward": "binary simulator episode success only",
        "baseline": "mean binary success within the eight-episode task-local rollout batch",
        "temperature": 0.5,
        "maximum_exponential_weight": 20.0,
        "normalize_weights_to_mean_one": True,
        "critic": "none",
        "shared_parameter_updates": False,
        "writer_updates": False,
        "source_demonstration_actions_during_rl": False,
        "optimizer": "adamw",
        "learning_rate": variant["learning_rate"],
        "betas": variant["betas"],
        "epsilon": variant["epsilon"],
        "weight_decay": variant["weight_decay"],
        "gradient_clip_norm": variant["gradient_clip_norm"],
        "scheduler": "constant_over_bounded_recovery",
        "optimizer_steps_per_rollout_round": 8,
        "effective_replay_batch_size": 64,
        "anchors_per_episode": 8,
        "action_chunk_size": 50,
    }
    for key, value in required_algorithm.items():
        _require_equal(algorithm.get(key), value, f"algorithm.{key}")


def _validate_surfaces_and_decisions(spec: Mapping[str, Any]) -> None:
    _require_equal(spec.get("schema_version"), 1, "schema_version")
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
    _require_equal(training["batch_size"], 8, "training batch_size")
    _require_equal(training["rounds_maximum"], 4, "training rounds_maximum")
    _require_equal(training["interaction_episode_nodes"], [16, 32], "interaction nodes")
    _require_equal(training["atomic_checkpoint_every_episodes"], 8, "checkpoint interval")
    _require_equal(
        training["train_init_state_indices_by_round"],
        [list(range(start, start + 8)) for start in (8, 16, 24, 32)],
        "training init states",
    )
    _require_equal(development["init_state_indices"], list(range(40, 48)), "development init states")
    _require_equal(development["frozen_base_successes_by_task"], [3, 3], "base J0")
    _require_equal(development["supervised_lora_successes_by_task"], [2, 4], "supervised J0")
    _require_equal(safeguards["maximum_action_drift_proxy"], 0.02, "drift safeguard")
    _require_equal(continuation["stage16_minimum_aggregate_paired_net_wins_to_continue"], 1, "stage16 trend")
    _require_equal(continuation["stage16_maximum_per_task_paired_loss_to_continue"], 1, "stage16 loss cap")
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
    _require_equal(exploration.get("standard_deviation"), [0.05] * 7, "exploration std")
    _require_equal(exploration.get("clip_low"), [-1.0] * 7, "exploration low")
    _require_equal(exploration.get("clip_high"), [1.0] * 7, "exploration high")
    _require_equal(exploration.get("maximum_saturation_fraction"), 0.05, "saturation safeguard")
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


def episodic_awr_weights(
    returns: torch.Tensor, *, temperature: float, maximum_weight: float
) -> torch.Tensor:
    """Compute clipped, mean-one episodic AWR weights without a learned critic."""

    if (
        returns.ndim != 1
        or returns.numel() == 0
        or not torch.isfinite(returns).all()
        or temperature <= 0
        or maximum_weight < 1
        or not math.isfinite(temperature)
        or not math.isfinite(maximum_weight)
    ):
        raise GateZeroTaskLocalRLContractError("invalid episodic AWR inputs")
    advantages = returns.to(dtype=torch.float32) - returns.to(dtype=torch.float32).mean()
    weights = torch.exp(advantages / temperature).clamp(max=maximum_weight)
    mean = weights.mean()
    if not torch.isfinite(weights).all() or not torch.isfinite(mean) or mean <= 0:
        raise GateZeroTaskLocalRLContractError("non-finite episodic AWR weights")
    return weights / mean


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
    """Apply the frozen stage-16/stage-32 continuation and selection rule."""

    if interaction_episodes not in spec["training_interaction"]["interaction_episode_nodes"]:
        raise GateZeroTaskLocalRLContractError("unknown task-local RL decision node")
    nets = metrics.get("paired_net_wins_by_arm")
    drift = metrics.get("action_drift_by_arm")
    if not isinstance(nets, dict) or set(nets) != {"zero_init_rl", "supervised_init_rl"}:
        raise GateZeroTaskLocalRLContractError("RL decision lacks both matched arms")
    if not isinstance(drift, dict) or set(drift) != set(nets):
        raise GateZeroTaskLocalRLContractError("RL decision lacks drift safeguards")
    passed = [arm for arm, values in nets.items() if _arm_passes(values, spec)]
    safeguards = (
        metrics.get("mechanics_valid") is True
        and metrics.get("nonfinite_count") == 0
        and isinstance(metrics.get("maximum_saturation_fraction"), (int, float))
        and metrics["maximum_saturation_fraction"] <= spec["exploration"]["maximum_saturation_fraction"]
        and all(
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            and value <= spec["offline_safeguards"]["maximum_action_drift_proxy"]
            for value in drift.values()
        )
    )
    status: str
    selected: int | None = None
    if not safeguards:
        status = "task_local_rl_mechanical_or_safeguard_failure"
    elif passed:
        status = "rl_candidate_selected_for_fresh_gate"
        selected = interaction_episodes
    elif interaction_episodes == 16:
        continuation = spec["continuation"]
        trend = any(
            sum(values) >= continuation["stage16_minimum_aggregate_paired_net_wins_to_continue"]
            and min(values) >= -continuation["stage16_maximum_per_task_paired_loss_to_continue"]
            for values in nets.values()
        )
        status = (
            "continue_same_rl_trajectories_to_32_episodes"
            if trend
            else "task_local_rl_early_check_not_supported"
        )
    else:
        status = "task_local_rl_candidate_not_supported"
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
