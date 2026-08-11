"""Fail-closed scientific and launch contract for the active SRTP Writer."""

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
    REPO_ROOT / "configs/pi05_v6_shared_reward_tangent_projection_v1.json"
)
V6_PRIOR_CONFIG_SCHEMA = "ember_pi05_v6_shared_reward_tangent_projection_v1"
V6_PRIOR_RUN_SCHEMA = "ember_pi05_v6_shared_reward_tangent_projection_launch_v1"
V6_PRIOR_PROFILE_SCHEMA = "ember_pi05_v6_shared_reward_tangent_projection_profile_v1"
V6_PRIOR_COMPLETION_SCHEMA = (
    "ember_pi05_v6_shared_reward_tangent_projection_completion_v1"
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
    "success_key_bank",
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
    "writer",
    "condition_feature",
    "program_residual",
    "data",
    "cache_gate",
)
_EXPECTED_METHOD = {
    "name": "frozen_v6_shared_reward_tangent_projection",
    "writer_input": "exact task language plus exactly one action-hidden teacher video",
    "dynamic_value": "one_raw_teacher_video_only",
    "language_only_lora_path": False,
    "deployment_expert_bank_read": False,
    "deployment_output": "one complete 38-target rank16 public LoRA",
}
_EXPECTED_INITIALIZATION = {
    "kind": "strict_load_and_freeze_historical_v6_fast_macro400",
    "checkpoint": (
        "runs/outputs/pi05_as_writer_v6_decay400_taskcomplete_dev_r4_b20_seed7_"
        "s2400_4efa737_20260729/checkpoints/step_00000400"
    ),
    "writer_state_tensor_count": 600,
    "writer_parameter_count": 10775296,
    "residual_memory": "elementwise_zero_on_fresh_or_memory_only_exact_resume",
    "success_key_bank": "empty_on_fresh_or_bank_only_exact_resume",
    "optimizer": "not_instantiated",
    "scheduler": "not_instantiated",
    "scaler": "not_instantiated",
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
        "development_train_mixed_k4_fixed_landmark_program_tangent_halfspaces_"
        "plus_all_success_key_equalities"
    ),
    "training_observation_action_scope": (
        "online_policy_generated_constant_memory_landmarks_for_train24_reward_"
        "tangent_only"
    ),
    "maximum_landmarks_per_episode": 4,
    "landmark_payload_checkpointed": False,
    "negative_action_forwards": 0,
    "validation_action_or_reward_reads": 0,
    "test_action_or_reward_reads": 0,
}
_EXPECTED_SUCCESS_KEY_BANK = {
    "kind": "training_only_first_all_success_condition_key_bank",
    "task_slots": 24,
    "feature_width": 256,
    "feature_dtype": "float32",
    "presence_dtype": "bool_runtime_uint8_checkpoint",
    "certificate": "exact_k4_all_four_random_reset_rollouts_success",
    "current_all_success_constraint_timing": "before_same_macro_shared_write",
    "persistent_policy": (
        "first_certified_key_per_sealed_train_task_never_replace"
    ),
    "current_key_policy": (
        "every_current_all_success_key_is_temporary_constraint_even_if_task_"
        "already_present"
    ),
    "zero_to_three_success_policy": "no_add_no_remove",
    "deployment_read": False,
    "stores_lora_action_observation_or_reward": False,
    "checkpoint_tensor_count": 3,
}
_EXPECTED_UPDATE = {
    "kind": "full48_success_key_nullspace_then_shared_reward_tangent_projection",
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
    "equality_constraint": (
        "every_persisted_and_current_all_success_key_has_exact_zero_"
        "incremental_program_motion"
    ),
    "blind_parameterization": (
        "objective_features_projected_into_success_key_row_span_orthogonal_"
        "complement"
    ),
    "anchor_projector": (
        "implicit_fp64_thin_svd_row_basis_without_dense_256_square_"
        "materialization"
    ),
    "dependent_anchor_policy": (
        "machine_epsilon_matrix_size_numerical_rank_only"
    ),
    "no_anchor": "elementwise_exact_pick_gc_full48_kernel",
    "correct_rhs": "negative_correct_condition_program_cotangent",
    "negative_rhs": "exact_zero_incremental_program_motion",
    "reward_constraint": (
        "each_current_mixed_k4_program_tangent_inner_product_with_final_shared_"
        "write_nonpositive"
    ),
    "reward_constraint_timing": (
        "after_blind_full48_shared_solve_before_single_memory_add"
    ),
    "reward_projection": (
        "euclidean_nearest_anchor_null_update_via_nonnegative_small_dual"
    ),
    "reward_dual_solver": "scipy_nnls_on_fp64_machine_epsilon_eigenspace",
    "no_mixed_or_raw_feasible": "elementwise_exact_blind_shared_update",
    "step_size": 1.0,
    "relative_damping": 0.01,
    "projection_small_solve_dtype": "float64",
    "full48_small_solve_dtype": "float64",
    "reward_small_dual_dtype": "float64_cpu",
    "large_rhs_and_memory_write_dtype": "float32",
    "postsolve_roundoff_correction": "thin_success_key_row_basis_fp32",
    "optimizer": "none_manual_add",
    "persistent_precision_or_optimizer_state": False,
    "momentum": False,
    "weight_decay": False,
    "gradient_clip": False,
    "global_scale_margin_or_cap": False,
    "task_scalar_gate_or_mask": False,
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
    "retain_success_replay": False,
    "retain_failure_replay": False,
    "rollout_policy_batch_size": 4,
    "persistent_env_lanes_per_task": 4,
    "occupancy_landmarks_per_episode_max": 4,
    "occupancy_landmark_selection": (
        "first_last_plus_two_seeded_uniform_reservoir_interior"
    ),
    "occupancy_landmark_retention": (
        "ephemeral_train24_only_never_checkpointed"
    ),
}
_EXPECTED_OBJECTIVE = {
    "name": (
        "blind_source_functional_program_credit_with_shared_mixed_reward_"
        "tangent_projection"
    ),
    "positive_policy_randomness": {
        "scope": "one_independent_flow_noise_and_time_per_action_query",
        "seed_scheme": TASK_LOGICAL_BATCH_POLICY_RNG_SCHEME,
        "flow_time_sampling_scheme": INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
        "flow_noise_sampling_scheme": INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    },
    "success_certificate": "four_of_four_binary_environment_success",
    "mixed_outcome_advantage": "k4_leave_one_out_binary_(4y_minus_sumy)_over_3",
    "episode_weighting": "one_quarter_then_equal_selected_landmark_mean",
    "landmark_selection": "first_last_plus_two_seeded_uniform_reservoir_interior",
    "maximum_landmarks_per_episode": 4,
    "maximum_logical_landmark_batch": 16,
    "flow_mc_samples": 4,
    "flow_time_sampling": "task_keyed_independent_exact_beta15_time_v1",
    "flow_noise_sampling": "task_keyed_independent_gaussian_v1",
    "physical_landmark_batch_size": 16,
    "mixed_reward_credit": "executed_prefix_cfm_program_cotangent_halfspace_only",
    "homogeneous_reward_credit": "exact_zero_without_functional_forward",
    "trajectory_replay": "fixed_landmarks_only_no_full_prefix_replay",
    "reward_weight_scale_or_direction": "none_projection_halfspace_only",
    "negative_policy_forwards": 0,
    "critic": False,
    "progress_reward": False,
    "spsa": False,
}
_EXPECTED_RNG = {
    "environment_seed_root": 2026081101,
    "policy_noise_seed_root": 2026081102,
    "landmark_seed_root": 2026081201,
    "flow_credit_seed_root": 2026081103,
    "scheme": "order_independent_splitmix64_without_rank_or_physical_microbatch",
}
_EXPECTED_OPTIMIZATION = {
    "precision": "bfloat16",
    "seed": 7,
    "functional_policy_microbatch_size": 10,
    "rollout_policy_batch_size": 4,
    "optimizer": "none",
    "distributed_update": {
        "kind": (
            "all_gather_dynamic_local_task_shard_features_blind_and_reward_"
            "cotangents_success_counts_then_identical_local_projected_manual_write"
        ),
        "world_size": "fresh_live_1_to_6_then_exact_resume_locked",
        "task_assignment": "cost_balanced_long_first_dynamic_uneven_train24",
        "memory_allreduce": False,
        "nccl_p2p_disable": "1",
        "nccl_algo": "Ring",
        "nccl_proto": "Simple",
        "deferred_process_group": True,
    },
}
_EXPECTED_PROFILE_BASELINE = {
    "path": (
        "runs/outputs/pi05_sknc_success_key_nullspace_full24_profile_macro0_"
        "r3_b20_f4fdac7_20260812/mechanism_profile.json"
    ),
    "schema": "ember_pi05_v6_success_key_nullspace_consolidation_profile_v1",
    "schedule_macro": 0,
    "task_count": 24,
    "rollouts": 96,
    "source_world_size": 3,
    "source_tasks_per_rank": 8,
    "scaling": (
        "source_step_seconds_times_actual_max_tasks_per_rank_over_source_"
        "tasks_per_rank"
    ),
    "source_step_seconds": 478.62702268897556,
    "step_seconds": 478.62702268897556,
}
_EXPECTED_PROFILE_GATES = {
    "task_count": 24,
    "video_count": 24,
    "source_action_query_count": 480,
    "rollout_count": 96,
    "all_success_task_count_min": 6,
    "all_success_task_per_suite_min": 1,
    "mixed_task_count_min": 8,
    "mixed_suite_count_min": 3,
    "maximum_landmarks_per_episode": 4,
    "flow_mc_samples": 4,
    "original_feature_rank": 48,
    "projected_feature_rank_min": 24,
    "active_regularized_gram_condition_number_max": 200.0,
    "unprotected_projected_feature_energy_ratio_median_min": 0.2,
    "protected_to_unprotected_motion_ratio_max": 1e-5,
    "negative_to_unprotected_motion_rms_max": 0.15,
    "raw_reward_violation_count_min": 2,
    "final_reward_violation_count": 0,
    "projected_to_blind_energy_ratio_min": 0.25,
    "negative_null_task_count_min": 18,
    "negative_null_per_kind_min": 6,
    "predicted_observed_relative_rms_max": 0.005,
    "protected_to_unprotected_lora_response_ratio_max": 1e-5,
    "protected_fixed_action_response_rms_max": 1e-6,
    "unprotected_fixed_action_probe_task_count": 4,
    "production_wall_ratio_max": 1.25,
    "negative_policy_forwards": 0,
    "oom_count": 0,
    "nonfinite_count": 0,
}
_PROFILE_STATIC = {
    "allowed_world_sizes": [1, 2, 3, 4, 5, 6],
    "maximum_world_size": 6,
    "task_assignment": "cost_balanced_long_first_dynamic_uneven_train24",
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
    "allowed_world_sizes": [1, 2, 3, 4, 5, 6],
    "maximum_world_size": 6,
    "task_assignment": "cost_balanced_long_first_dynamic_uneven_train24",
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
    "maximum_world_size": 6,
    "device_selection": (
        "use_up_to_six_healthy_low_utilization_same_node_devices_with_"
        "sufficient_memory_headroom_without_waiting_nonzero_memory_is_not_"
        "automatic_exclusion"
    ),
}
_COHERENT_STATES = {
    (
        "active_cpu_ready_awaiting_live_profile",
        "awaiting_live_a40_fresh0_to1_profile",
        "blocked_until_live_profile_passes_and_is_sealed",
        "awaiting_live_srtp_deployment_smoke",
    ),
    (
        "active_formal_ready",
        "sealed_from_live_a40_fresh0_to1_profile",
        "ready_after_live_profile_seal",
        "sealed_from_live_srtp_deployment_smoke",
    ),
    (
        "formal_result_sealed",
        "sealed_from_live_a40_fresh0_to1_profile",
        "formal_result_sealed",
        "sealed_from_live_srtp_deployment_smoke",
    ),
    (
        "profile_result_sealed_nonpass",
        "profile_result_sealed_nonpass",
        "blocked_by_profile_nonpass",
        "not_run_after_profile_nonpass",
    ),
}


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    try:
        row = config["authorities"][name]
        path = (REPO_ROOT / str(row["path"])).resolve()
    except (KeyError, TypeError, ValueError) as error:
        raise ExpertManifoldError(f"missing SRTP Writer authority: {name}") from error
    if not path.is_file():
        raise ExpertManifoldError(f"SRTP Writer authority is missing: {name}")
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
        return _nonempty_artifact(profile) and formal is None and _nonempty_artifact(smoke)
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
        and config.get("initialization") == _EXPECTED_INITIALIZATION
        and config.get("information_wall") == _EXPECTED_INFORMATION_WALL
        and config.get("success_key_bank") == _EXPECTED_SUCCESS_KEY_BANK
        and config.get("update") == _EXPECTED_UPDATE
        and config.get("environment") == _EXPECTED_ENVIRONMENT
        and config.get("objective") == _EXPECTED_OBJECTIVE
        and config.get("rng") == _EXPECTED_RNG
        and config.get("optimization") == _EXPECTED_OPTIMIZATION
        and _provenance_sections_match(config)
        and _profile_formal_evaluation_match(config)
    )


def load_v6_prior_config(path: Path = V6_PRIOR_CANONICAL_CONFIG) -> dict[str, Any]:
    path = path.resolve()
    if path != V6_PRIOR_CANONICAL_CONFIG.resolve() or not path.is_file():
        raise ExpertManifoldError("SRTP Writer requires its canonical config path")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExpertManifoldError("invalid SRTP Writer config") from error
    if not isinstance(config, dict) or not _config_matches(config):
        raise ExpertManifoldError("SRTP Writer config violates its fail-closed contract")
    for name in config["authorities"]:
        authority_path(config, name)
    initialization = (REPO_ROOT / str(config["initialization"]["checkpoint"])).resolve()
    if not initialization.is_dir():
        raise ExpertManifoldError("SRTP historical v6 initialization is missing")
    return config


def runtime_for_mode(
    config: Mapping[str, Any], mode: str
) -> tuple[int, tuple[int, ...], int]:
    if mode == "mechanism-profile":
        if (
            config.get("status") != "active_cpu_ready_awaiting_live_profile"
            or config["profile_run"]["status"]
            != "awaiting_live_a40_fresh0_to1_profile"
        ):
            raise ExpertManifoldError("SRTP mechanism profile is not authorized")
        return 1, (), int(config["profile_run"]["schedule_macro"])
    if mode == "formal":
        if (
            config.get("status") != "active_formal_ready"
            or config["formal_run"]["status"] != "ready_after_live_profile_seal"
        ):
            raise ExpertManifoldError("SRTP formal training is blocked by live gates")
        formal = config["formal_run"]
        return (
            int(formal["total_macros"]),
            tuple(int(value) for value in formal["checkpoint_macros"]),
            0,
        )
    raise ExpertManifoldError(f"unsupported SRTP Writer mode: {mode}")
