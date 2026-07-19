"""Frozen n=32 paired evaluation for the existing task-local LoRA RL path."""

from __future__ import annotations

import json
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.gate_zero_evidence import paired_binary_summary, validate_evaluation_records
from ember.gate_zero_oracle_artifacts import atomic_json, restore_trainable_state, sha256_file
from ember.gate_zero_oracle_report_runtime import _closed_loop_metrics
from ember.gate_zero_oracle_session import capture_trainable_state
from ember.gate_zero_task_local_rl.runtime import scoped_policy_execution_horizon


class FormalEvaluationError(RuntimeError):
    """Raised when the formal source-only evaluation changes sealed authority."""


def load_formal_evaluation_spec(
    path: Path, *, repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        with path.open("rb") as handle:
            spec = tomllib.load(handle)
        training_path = repo_root / spec["authority"]["training_contract_relative_path"]
        evidence_path = repo_root / spec["authority"]["evidence_contract_relative_path"]
        with evidence_path.open("rb") as handle:
            evidence = tomllib.load(handle)
    except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
        raise FormalEvaluationError("formal evaluation authority is unreadable") from error
    expected = {
        training_path: spec["authority"]["training_contract_sha256"],
        evidence_path: spec["authority"]["evidence_contract_sha256"],
    }
    if any(sha256_file(bound) != digest for bound, digest in expected.items()):
        raise FormalEvaluationError("formal evaluation upstream hash changed")
    evaluation = spec["evaluation"]
    required = evaluation["required_training_seeds"]
    if (
        spec.get("status")
        != "predeclared_formal_development_evaluation_after_two_n8_smokes_before_n32_outcomes"
        or spec["tasks"]["development"] != [3, 4]
        or set(spec["tasks"]["development"]) & set(spec["tasks"]["confirmation"])
        or required != [2026071830, 2026072030]
        or evaluation["fixed_initialization_training_seed"] != required[0]
        or evaluation["policy_rng_seeds"] != [2026071836, 2026071837, 2026071838, 2026071839]
        or evaluation["physical_init_state_indices"] != list(range(40, 48))
        or evaluation["execution_horizons"] != [16, 50]
        or evaluation["rollouts_per_task_arm"] != 32
        or any(spec["authority"].get(f"{surface}_numeric_access") for surface in ("validation", "held", "locked"))
    ):
        raise FormalEvaluationError("formal evaluation contract changed")
    return spec, evidence


def validate_evaluation_source(
    source_root: Path, *, spec: Mapping[str, Any], training_seed: int
) -> None:
    expected = spec["authority"]["stage_result_sha256_by_training_seed"].get(
        str(training_seed)
    )
    stage = source_root / "stage_results" / f"{spec['authority']['checkpoint_interaction_episodes']:06d}.json"
    if expected is None or not stage.is_file() or sha256_file(stage) != expected:
        raise FormalEvaluationError("formal evaluation source stage changed")


def compatible_recovery_authorities(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    training_seed: int,
) -> dict[str, Any]:
    """Accept the immutable reference replicate created before seed labeling."""

    resolved = dict(expected)
    reference = spec["evaluation"]["required_training_seeds"][0]
    legacy = dict(resolved)
    legacy.pop("training_seed", None)
    if training_seed == reference and dict(actual) == legacy:
        return legacy
    if dict(actual) != resolved:
        raise FormalEvaluationError("formal recovery authority changed")
    return resolved


def rollout_rows(
    *,
    rollout: Mapping[str, Any],
    task_id: int,
    arm: str,
    training_seed: int | None,
    policy_rng_seed: int,
    execution_horizon: int,
    init_state_hashes: Mapping[int, str],
    action_drift_to_base: float,
    action_drift_to_initialization: float,
) -> list[dict[str, Any]]:
    indices = rollout["official_rollout_init_state_indices"]
    values = []
    for offset, (index, evaluator_seed, success, maximum, time_to_success) in enumerate(
        zip(
            indices,
            rollout["seeds"],
            rollout["successes"],
            rollout["max_rewards"],
            rollout["time_to_success"],
            strict=True,
        )
    ):
        values.append(
            {
                "surface": "development",
                "task_id": task_id,
                "arm": arm,
                "training_seed": training_seed,
                "policy_rng_seed": policy_rng_seed,
                "evaluator_seed": evaluator_seed,
                "physical_init_state_index": index,
                "physical_init_state_sha256": init_state_hashes[index],
                "execution_horizon": execution_horizon,
                "success": bool(success),
                "grasp": None,
                "correct_object_or_region": None,
                "drawer_closed": None,
                "time_to_success": time_to_success,
                "progress_fraction": max(0.0, min(1.0, float(maximum))),
                "action_drift_to_base": float(action_drift_to_base),
                "action_drift_to_initialization": float(
                    action_drift_to_initialization
                ),
                "episode_offset": offset,
            }
        )
    return values


def evaluate_live_arm(
    *,
    arm: Any,
    spec: Mapping[str, Any],
    output_dir: Path,
    task_id: int,
    initialization: str,
    training_seed: int,
    current_state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    evaluation = spec["evaluation"]
    model = arm.session.model
    restore_trainable_state(model, arm.initial_trainable_state)
    initial_reference = arm.session.evaluator.capture_base_reference(model)
    initial_vs_base = arm.session.evaluator.evaluate_candidate(
        model, arm.session.reference, step=0
    )
    restore_trainable_state(model, current_state)
    current_vs_base = arm.session.evaluator.evaluate_candidate(
        model, arm.session.reference, step=spec["authority"]["checkpoint_interaction_episodes"]
    )
    current_vs_initial = arm.session.evaluator.evaluate_candidate(
        model, initial_reference, step=spec["authority"]["checkpoint_interaction_episodes"]
    )
    fixed_seed = evaluation["fixed_initialization_training_seed"]
    states = arm.task_authority["init_state_indices"]
    hashes = dict(zip(states, arm.task_authority["init_state_sha256"], strict=True))
    rows: list[dict[str, Any]] = []
    for horizon in evaluation["execution_horizons"]:
        for policy_seed in evaluation["policy_rng_seeds"]:
            retain = policy_seed == evaluation["retain_video_policy_seed"]
            rollout_spec = {
                "report": {
                    "rollout_batch_size": evaluation["rollouts_per_policy_seed"],
                    "official_rollout_init_state_indices": evaluation[
                        "physical_init_state_indices"
                    ],
                    "seed_start": evaluation["evaluator_seed_start"],
                    "warmup_seed_start": evaluation["warmup_seed_start"],
                    "policy_rng_seed": policy_seed,
                },
                "resources": {
                    "retain_one_video_per_report_arm": retain,
                    "return_episode_data": True,
                },
            }

            def evaluate(condition: str) -> dict[str, Any]:
                with scoped_policy_execution_horizon(
                    arm.runtime[0],
                    execution_horizon=horizon,
                    expected_model_chunk_size=50,
                ):
                    return _closed_loop_metrics(
                        runtime=arm.runtime,
                        task_id=task_id,
                        condition=condition,
                        language=arm.language,
                        spec=rollout_spec,
                        output_dir=output_dir,
                    )

            restore_trainable_state(model, current_state)
            current_arm = f"{initialization}_rl"
            current = evaluate(
                f"seed{training_seed}_{current_arm}_h{horizon}_p{policy_seed}"
            )
            if current["mechanics_valid"] is not True:
                raise FormalEvaluationError("formal current-policy mechanics failed")
            rows.extend(
                rollout_rows(
                    rollout=current,
                    task_id=task_id,
                    arm=current_arm,
                    training_seed=training_seed,
                    policy_rng_seed=policy_seed,
                    execution_horizon=horizon,
                    init_state_hashes=hashes,
                    action_drift_to_base=current_vs_base["action_drift_proxy"],
                    action_drift_to_initialization=current_vs_initial[
                        "action_drift_proxy"
                    ],
                )
            )
            if training_seed == fixed_seed:
                restore_trainable_state(model, arm.initial_trainable_state)
                initial_arm = (
                    "frozen_base" if initialization == "zero_init" else "supervised_lora"
                )
                initial = evaluate(
                    f"fixed_{initial_arm}_h{horizon}_p{policy_seed}"
                )
                if initial["mechanics_valid"] is not True:
                    raise FormalEvaluationError("formal initialization mechanics failed")
                rows.extend(
                    rollout_rows(
                        rollout=initial,
                        task_id=task_id,
                        arm=initial_arm,
                        training_seed=None,
                        policy_rng_seed=policy_seed,
                        execution_horizon=horizon,
                        init_state_hashes=hashes,
                        action_drift_to_base=initial_vs_base["action_drift_proxy"],
                        action_drift_to_initialization=0.0,
                    )
                )
    return rows


def _episode_key(
    row: Mapping[str, Any], *, include_training_seed: bool = False
) -> tuple[Any, ...]:
    key = (
        row["policy_rng_seed"],
        row["evaluator_seed"],
        row["physical_init_state_index"],
        row["physical_init_state_sha256"],
    )
    return (row["training_seed"], *key) if include_training_seed else key


def _paired_rows(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> tuple[list[bool], list[bool]]:
    seeded = any(row["training_seed"] is not None for row in right)
    right_by_key = {
        _episode_key(row, include_training_seed=seeded): bool(row["success"])
        for row in right
    }
    ordered = sorted(
        left, key=lambda row: (row["training_seed"] or -1, _episode_key(row))
    )
    return [bool(row["success"]) for row in ordered], [
        right_by_key[_episode_key(row, include_training_seed=seeded)]
        for row in ordered
    ]


def aggregate_formal_rows(
    rows: Sequence[Mapping[str, Any]], *, spec: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    validation = validate_evaluation_records(rows, evidence)
    grouped: dict[tuple[int, str, int | None, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["task_id"], row["arm"], row["training_seed"], row["execution_horizon"])].append(row)
    seed = spec["evaluation"]["paired_bootstrap_seed"]
    reps = spec["evaluation"]["paired_bootstrap_replicates"]
    summaries: dict[str, Any] = {}
    routes = {
        "supervised_lora_vs_frozen_base": ("supervised_lora", "frozen_base"),
        "zero_init_rl_vs_frozen_base": ("zero_init_rl", "frozen_base"),
        "supervised_init_rl_vs_supervised_lora": (
            "supervised_init_rl",
            "supervised_lora",
        ),
        "supervised_init_rl_vs_zero_init_rl": (
            "supervised_init_rl",
            "zero_init_rl",
        ),
    }
    required_seeds = spec["evaluation"]["required_training_seeds"]
    fixed = {"frozen_base", "supervised_lora"}
    for horizon in spec["evaluation"]["execution_horizons"]:
        for task_id in spec["tasks"]["development"]:
            for label, (left_arm, right_arm) in routes.items():
                seeds = [None] if left_arm in fixed else required_seeds
                for training_seed in seeds:
                    left = grouped[(task_id, left_arm, training_seed, horizon)]
                    right_seed = None if right_arm in fixed else training_seed
                    right = grouped[(task_id, right_arm, right_seed, horizon)]
                    left_success, right_success = _paired_rows(left, right)
                    suffix = "fixed" if training_seed is None else f"seed{training_seed}"
                    summaries[
                        f"h{horizon}_task{task_id}_{label}_{suffix}"
                    ] = paired_binary_summary(
                        left_success,
                        right_success,
                        bootstrap_seed=seed + horizon + task_id + (training_seed or 0),
                        bootstrap_replicates=reps,
                    )
    primary = spec["evaluation"]["primary_execution_horizon"]
    route_support = {}
    for route in (
        "supervised_lora_vs_frozen_base",
        "zero_init_rl_vs_frozen_base",
        "supervised_init_rl_vs_supervised_lora",
    ):
        seeds = [None] if route == "supervised_lora_vs_frozen_base" else required_seeds
        route_support[route] = all(
            summaries[
                f"h{primary}_task{task}_{route}_{'fixed' if training_seed is None else f'seed{training_seed}'}"
            ]["paired_bootstrap_ci95_pp"][0]
            > 0.0
            for task in spec["tasks"]["development"]
            for training_seed in seeds
        )
    initialization_benefit = all(
        summaries[
            f"h{primary}_task{task}_supervised_init_rl_vs_zero_init_rl_seed{training_seed}"
        ]["paired_bootstrap_ci95_pp"][0]
        > 0.0
        for task in spec["tasks"]["development"]
        for training_seed in required_seeds
    )
    candidate = any(route_support.values())
    return {
        "schema_version": 1,
        "status": (
            "formal_development_candidate_supported"
            if candidate
            else "formal_development_ambiguous_or_negative"
        ),
        "surface": spec["surface"],
        "validation": validation,
        "summaries": summaries,
        "route_support": route_support,
        "initialization_benefit_supported": initialization_benefit,
        "development_candidate_supported": candidate,
        "gate_zero_authorized": False,
        "writer_authorized": False,
        "validation_numeric_access": False,
        "held_numeric_access": False,
        "locked_numeric_access": False,
    }


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_json(path, {"rows": list(rows)})


def read_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
        raise FormalEvaluationError("formal row packet is invalid")
    return value["rows"]
