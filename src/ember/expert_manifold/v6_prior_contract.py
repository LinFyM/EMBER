"""Canonical scientific and launch contract for the active OSG-PC Writer."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ember.expert_manifold.contract import ExpertManifoldError
from ember.writer.functional import (
    INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
    INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    TASK_LOGICAL_BATCH_POLICY_RNG_SCHEME,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
V6_PRIOR_CANONICAL_CONFIG = (
    REPO_ROOT
    / "configs/pi05_v6_on_policy_success_guarded_program_credit_v1.json"
)
V6_PRIOR_CONFIG_SCHEMA = (
    "ember_pi05_v6_on_policy_success_guarded_program_credit_v1"
)
V6_PRIOR_RUN_SCHEMA = (
    "ember_pi05_v6_on_policy_success_guarded_program_credit_launch_v1"
)
V6_PRIOR_PROFILE_SCHEMA = (
    "ember_pi05_v6_on_policy_success_guarded_program_credit_profile_v1"
)
V6_PRIOR_COMPLETION_SCHEMA = (
    "ember_pi05_v6_on_policy_success_guarded_program_credit_completion_v1"
)
V6_PRIOR_MODES = ("mechanism-profile", "formal")
_ACTIVE_AUTHORITY_REF = "origin/codex/bci-continuation"
_PICK_GC_PROVENANCE_CONFIG = (
    REPO_ROOT / "configs/pi05_v6_policy_innovation_goal_causal_key_v1.json"
)

_EXPECTED_TOP_LEVEL = {
    "schema_version",
    "status",
    "method",
    "authorities",
    "initialization",
    "information_wall",
    "writer",
    "condition_feature",
    "program_residual",
    "update",
    "data",
    "environment",
    "objective",
    "rng",
    "optimization",
    "cache_gate",
    "profile_run",
    "formal_run",
    "evaluation",
}
_UNCHANGED_PICK_GC_SECTIONS = (
    "authorities",
    "initialization",
    "writer",
    "condition_feature",
    "program_residual",
    "data",
)
_EXPECTED_METHOD = {
    "name": "frozen_v6_on_policy_success_guarded_program_credit",
    "writer_input": "exact task language plus exactly one action-hidden teacher video",
    "dynamic_value": "one_raw_teacher_video_only",
    "language_only_lora_path": False,
    "deployment_expert_bank_read": False,
    "deployment_output": "one complete 38-target rank16 public LoRA",
}
_EXPECTED_INFORMATION_WALL = {
    "writer_video_split_roles": ["train", "validation", "test"],
    "writer_forbidden_inputs": [
        "action",
        "proprio",
        "state",
        "reward",
        "terminal",
        "task_id",
        "filename",
        "object_pose",
        "hidden_normalization",
        "policy_outcome",
    ],
    "teacher_action_reads": 0,
    "source_actions_enter_only": "blind_correct_condition_functional_loss",
    "training_reward_scope": (
        "development_train_binary_success_selects_retention_constraints_only"
    ),
    "training_observation_action_scope": (
        "successful_development_train_on_policy_executed_prefixes_only"
    ),
    "failed_rollout_replay_gradient_episodes": 0,
    "negative_action_forwards": 0,
    "validation_action_or_reward_reads": 0,
    "test_action_or_reward_reads": 0,
}
_EXPECTED_UPDATE = {
    "kind": (
        "full48_on_policy_success_guarded_counterfactual_null_condition_kernel"
    ),
    "correct_conditions": 24,
    "negative_conditions": 24,
    "ordering": "correct_task_ordinal_0_to_23_then_negative_task_ordinal_0_to_23",
    "negative_schedule": (
        "task_ordinal_plus_task_visit_modulo_reversed_shuffled_wrong"
    ),
    "negative_counts_per_macro": {"reversed": 8, "shuffled": 8, "wrong": 8},
    "blind_proposal": (
        "negative_correct_condition_source_functional_program_cotangent"
    ),
    "success_guard": (
        "per_success_on_policy_executed_prefix_program_cotangent_halfspace"
    ),
    "projection": "parameter_free_exact_euclidean_k4_active_subset_kkt",
    "constraint": "retention_cotangent_inner_safe_direction_nonpositive",
    "no_success_or_raw_feasible": "elementwise_exact_blind_proposal",
    "correct_rhs": "negative_safe_program_direction_as_full48_cotangent",
    "negative_rhs": "exact_zero_incremental_program_motion",
    "step_size": 1.0,
    "relative_damping": 0.01,
    "projection_small_solve_dtype": "float64",
    "full48_small_solve_dtype": "float64",
    "large_rhs_and_memory_write_dtype": "float32",
    "optimizer": "none_manual_add",
    "persistent_precision_or_optimizer_state": False,
    "momentum": False,
    "weight_decay": False,
    "gradient_clip": False,
    "global_scale_margin_or_cap": False,
}
_EXPECTED_ENVIRONMENT = {
    "reset": "official_random_reset_without_set_init_state",
    "rollouts_per_task": 4,
    "dummy_settling_steps": 10,
    "dummy_action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    "render_resolution": 256,
    "model_resolution": 224,
    "camera_rotation_degrees": 180,
    "policy_chunk_size": 50,
    "action_execution_horizon": 5,
    "num_inference_steps": 10,
    "terminate_on_success": True,
    "retain_success_replay": True,
    "retain_failure_replay": False,
    "rollout_policy_batch_size": 4,
    "persistent_env_lanes_per_task": 4,
}
_EXPECTED_OBJECTIVE = {
    "name": (
        "blind_source_proposal_with_success_only_on_policy_retention_constraints"
    ),
    "positive_policy_randomness": {
        "scope": "one_independent_flow_noise_and_time_per_action_query",
        "seed_scheme": TASK_LOGICAL_BATCH_POLICY_RNG_SCHEME,
        "flow_time_sampling_scheme": INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
        "flow_noise_sampling_scheme": INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    },
    "retention_episode_aggregation": (
        "mean_chunks_within_each_success_episode_without_cross_episode_average"
    ),
    "retention_flow_mc_samples": 4,
    "retention_flow_time_sampling": (
        "task_keyed_independent_beta15_scaled_0999_offset_0001"
    ),
    "retention_flow_noise_sampling": "task_keyed_independent_gaussian",
    "retention_flow_panel_physical_batch_invariance": True,
    "retention_flow_panel_row_identity": (
        "historical_complete_k4_task_panel_then_success_row_select"
    ),
    "failure_episode_gradient": "exact_absent",
    "negative_policy_forwards": 0,
    "critic": False,
    "progress_reward": False,
    "spsa": False,
}
_EXPECTED_RNG = {
    "environment_seed_root": 2026081101,
    "policy_noise_seed_root": 2026081102,
    "retention_flow_seed_root": 2026081103,
    "scheme": "order_independent_splitmix64_without_rank_or_physical_microbatch",
}
_EXPECTED_OPTIMIZATION = {
    "precision": "bfloat16",
    "seed": 7,
    "functional_policy_microbatch_size": 10,
    "retention_replay_microbatch_size": 8,
    "rollout_policy_batch_size": 4,
    "optimizer": "none",
    "distributed_update": {
        "kind": (
            "all_gather_local4_features_and_safe_cotangents_then_identical_"
            "local_manual_write"
        ),
        "world_size": 6,
        "tasks_per_rank": 4,
        "memory_allreduce": False,
        "nccl_p2p_disable": "1",
        "nccl_algo": "Ring",
        "nccl_proto": "Simple",
        "deferred_process_group": True,
    },
}
_EXPECTED_CACHE_GATE = {
    "status": "inherited_passed_unchanged_pick_gc_condition_feature",
    "same_task_complete_cosine_mean_min": 0.9,
    "cross_task_complete_cosine_mean_max": 0.3,
    "correct_reversed_complete_cosine_max": 0.0,
    "shuffled_complete_cosine_abs_mean_max": 0.1,
    "correct24_rank_min": 24,
    "regularized_condition_number_max": 150.0,
}
_EXPECTED_PROFILE_BASELINE = {
    "path": (
        "runs/outputs/pi05_v6_reward_credit_program_cotangent_profile_full24_"
        "k4_nmc4_r6_b8_allmixedk4_20260810/mechanism_profile.json"
    ),
    "schema": "ember_pi05_v6_reward_credit_program_cotangent_profile_v1",
    "schedule_macro": 0,
    "task_count": 24,
    "rollouts": 96,
    "source_world_size": 6,
    "source_tasks_per_rank": 4,
    "target_world_size": 6,
    "target_tasks_per_rank": 4,
    "scaling": (
        "source_step_seconds_times_target_tasks_per_rank_over_source_tasks_per_rank"
    ),
    "source_step_seconds": 507.30541240703315,
    "step_seconds": 507.30541240703315,
}
_EXPECTED_PROFILE_GATES = {
    "task_count": 24,
    "video_count": 24,
    "source_action_query_count": 480,
    "rollout_count": 96,
    "guarded_task_count_min": 18,
    "all_success_task_count_min": 6,
    "guarded_task_per_suite_min": 2,
    "success_program_cotangent_nonzero": True,
    "failure_replay_gradient_episodes": 0,
    "projection_changed_task_count_min": 1,
    "exact_blind_fallback_required": True,
    "applied_guard_evidence_required": True,
    "feature_rank_min": 48,
    "regularized_gram_condition_number_max": 200.0,
    "correct_motion_to_cotangent_rms_min": 0.25,
    "negative_to_correct_motion_rms_max": 0.15,
    "correct_retained_task_count_min": 21,
    "negative_null_task_count_min": 18,
    "negative_null_per_kind_min": 6,
    "predicted_observed_relative_rms_max": 0.005,
    "production_wall_ratio_max": 1.25,
    "lora_a_response_rms_min": 0.0,
    "lora_b_response_rms_min": 0.0,
    "fixed_action_response_rms_min": 0.0,
    "fixed_action_probe_task_count": 4,
    "fixed_action_passing_task_count_min": 4,
    "extra_negative_policy_forwards": 0,
    "oom_count": 0,
    "nonfinite_count": 0,
}
_PROFILE_STATIC = {
    "expected_world_size": 6,
    "tasks_per_rank": 4,
    "schedule_macro": 0,
    "diagnostic_macros": 1,
    "num_workers_per_rank": 2,
    "retain_weight": False,
}
_EXPECTED_FORMAL_GATES = {
    "macro5_correct_min": 140,
    "macro5_breadth_min": 6,
    "macro5_lost_to_macro0_max": 8,
    "macro5_gained_must_exceed_lost": True,
    "macro5_non_declining_suite_count_min": 2,
    "macro5_single_task_net_gain_fraction_max": 0.5,
    "first_full_six_arm_correct_min": 144,
    "goal_full_six_arm_correct_min": 151,
    "goal_correct_strictly_exceeds_negative_controls": True,
    "goal_same_task_other_correct_ratio_min": 0.9,
    "macro10_requires_macro5_gate": True,
}
_FORMAL_STATIC = {
    "expected_world_size": 6,
    "tasks_per_rank": 4,
    "num_workers_per_rank": 2,
    "total_macros": 10,
    "checkpoint_macros": [5, 10],
    "strict400_checkpoints": [0, 5, 10],
}
_EXPECTED_EVALUATION_STATIC = {
    "throughput_policy": (
        "highest_measured_batch_throughput_with_device_memory_headroom"
    ),
    "required_writer_model_batch_sizes": [8, 16, 32],
    "minimum_smoke_writer_model_batch_size": 8,
}
_COHERENT_STATES = {
    (
        "active_cpu_ready_awaiting_live_profile",
        "awaiting_live_a40_fresh0_to1_profile",
        "blocked_until_live_profile_passes_and_is_sealed",
        "awaiting_live_osg_pc_deployment_smoke",
    ),
    (
        "active_formal_ready",
        "sealed_from_live_a40_fresh0_to1_profile",
        "ready_after_live_profile_seal",
        "sealed_from_live_osg_pc_deployment_smoke",
    ),
    (
        "formal_result_sealed",
        "sealed_from_live_a40_fresh0_to1_profile",
        "formal_result_sealed",
        "sealed_from_live_osg_pc_deployment_smoke",
    ),
    (
        "profile_result_sealed_nonpass",
        "profile_result_sealed_nonpass",
        "blocked_by_profile_nonpass",
        "not_run_after_profile_nonpass",
    ),
}


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    """Resolve one registered authority and reject missing files."""

    try:
        row = config["authorities"][name]
        path = (REPO_ROOT / str(row["path"])).resolve()
    except (KeyError, TypeError, ValueError) as error:
        raise ExpertManifoldError(f"missing OSG-PC Writer authority: {name}") from error
    if not path.is_file():
        raise ExpertManifoldError(f"OSG-PC Writer authority is missing: {name}")
    return path


def _git_is_ancestor(left: str, right: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", left, right],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def git_commit_in_active_authority_lineage(commit: str) -> bool:
    """Require a retained commit shared by this checkout and active remote."""

    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        return False
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        authority = subprocess.run(
            ["git", "rev-parse", _ACTIVE_AUTHORITY_REF],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return False
    return (
        _git_is_ancestor(commit, head)
        and _git_is_ancestor(commit, authority)
        and (_git_is_ancestor(head, authority) or _git_is_ancestor(authority, head))
    )


def _static_projection_matches(value: object, expected: Mapping[str, Any]) -> bool:
    return isinstance(value, Mapping) and {
        name: value.get(name) for name in expected
    } == dict(expected)


def _provenance_sections_match(config: Mapping[str, Any]) -> bool:
    try:
        provenance = json.loads(
            _PICK_GC_PROVENANCE_CONFIG.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    return all(
        config.get(name) == provenance.get(name)
        for name in _UNCHANGED_PICK_GC_SECTIONS
    )


def _state_tuple(config: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    return (
        config.get("status"),
        config.get("profile_run", {}).get("status"),
        config.get("formal_run", {}).get("status"),
        config.get("evaluation", {}).get("formal_status"),
    )


def _nonempty_artifact(value: object) -> bool:
    return isinstance(value, Mapping) and bool(value)


def _lifecycle_artifacts_match(config: Mapping[str, Any]) -> bool:
    status = config.get("status")
    profile = config["profile_run"].get("artifact_evidence")
    formal = config["formal_run"].get("artifact_evidence")
    smoke = config["evaluation"].get("online_smoke_evidence")
    if status == "active_cpu_ready_awaiting_live_profile":
        return profile is formal is smoke is None
    if status == "active_formal_ready":
        return (
            _nonempty_artifact(profile)
            and formal is None
            and _nonempty_artifact(smoke)
        )
    if status == "formal_result_sealed":
        return all(_nonempty_artifact(value) for value in (profile, formal, smoke))
    if status == "profile_result_sealed_nonpass":
        return _nonempty_artifact(profile) and formal is None and smoke is None
    return False


def _profile_formal_evaluation_match(config: Mapping[str, Any]) -> bool:
    profile = config.get("profile_run")
    formal = config.get("formal_run")
    evaluation = config.get("evaluation")
    if not all(isinstance(value, Mapping) for value in (profile, formal, evaluation)):
        return False
    return (
        _state_tuple(config) in _COHERENT_STATES
        and set(profile)
        == {
            "status",
            "throughput_baseline",
            "gates",
            "artifact_evidence",
            *_PROFILE_STATIC,
        }
        and set(formal)
        == {"status", "decision_gates", "artifact_evidence", *_FORMAL_STATIC}
        and set(evaluation)
        == {"formal_status", "online_smoke_evidence", *_EXPECTED_EVALUATION_STATIC}
        and _static_projection_matches(profile, _PROFILE_STATIC)
        and profile.get("throughput_baseline") == _EXPECTED_PROFILE_BASELINE
        and profile.get("gates") == _EXPECTED_PROFILE_GATES
        and _static_projection_matches(formal, _FORMAL_STATIC)
        and formal.get("decision_gates") == _EXPECTED_FORMAL_GATES
        and _static_projection_matches(evaluation, _EXPECTED_EVALUATION_STATIC)
        and _lifecycle_artifacts_match(config)
    )


def _config_matches(config: Mapping[str, Any]) -> bool:
    return (
        set(config) == _EXPECTED_TOP_LEVEL
        and config.get("schema_version") == V6_PRIOR_CONFIG_SCHEMA
        and config.get("method") == _EXPECTED_METHOD
        and config.get("information_wall") == _EXPECTED_INFORMATION_WALL
        and config.get("update") == _EXPECTED_UPDATE
        and config.get("environment") == _EXPECTED_ENVIRONMENT
        and config.get("objective") == _EXPECTED_OBJECTIVE
        and config.get("rng") == _EXPECTED_RNG
        and config.get("optimization") == _EXPECTED_OPTIMIZATION
        and config.get("cache_gate") == _EXPECTED_CACHE_GATE
        and _provenance_sections_match(config)
        and _profile_formal_evaluation_match(config)
    )


def load_v6_prior_config(path: Path = V6_PRIOR_CANONICAL_CONFIG) -> dict[str, Any]:
    """Load only the active OSG-PC config and fail closed on scientific drift."""

    path = path.resolve()
    if path != V6_PRIOR_CANONICAL_CONFIG.resolve() or not path.is_file():
        raise ExpertManifoldError("OSG-PC Writer requires its canonical config path")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExpertManifoldError("invalid OSG-PC Writer config") from error
    if not isinstance(config, dict) or not _config_matches(config):
        raise ExpertManifoldError(
            "OSG-PC Writer config violates its fail-closed contract"
        )
    for name in config["authorities"]:
        authority_path(config, name)
    initialization = (
        REPO_ROOT / str(config["initialization"]["checkpoint"])
    ).resolve()
    if not initialization.is_dir():
        raise ExpertManifoldError("OSG-PC historical v6 initialization is missing")
    return config


def runtime_for_mode(
    config: Mapping[str, Any],
    mode: str,
) -> tuple[int, tuple[int, ...], int]:
    """Return the exact active runtime segment for profile or formal training."""

    if mode == "mechanism-profile":
        if (
            config.get("status") != "active_cpu_ready_awaiting_live_profile"
            or config["profile_run"]["status"]
            != "awaiting_live_a40_fresh0_to1_profile"
        ):
            raise ExpertManifoldError("OSG-PC mechanism profile is not authorized")
        return 1, (), int(config["profile_run"]["schedule_macro"])
    if mode == "formal":
        if (
            config.get("status") != "active_formal_ready"
            or config["formal_run"]["status"] != "ready_after_live_profile_seal"
        ):
            raise ExpertManifoldError("OSG-PC formal training is blocked by live gates")
        formal = config["formal_run"]
        return (
            int(formal["total_macros"]),
            tuple(int(value) for value in formal["checkpoint_macros"]),
            0,
        )
    raise ExpertManifoldError(f"unsupported OSG-PC Writer mode: {mode}")
