"""Fail-closed scientific and launch contract for active PVJFC."""

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
    REPO_ROOT / "configs/pi05_v6_paired_video_joint_functional_credit_v1.json"
)
V6_PRIOR_CONFIG_SCHEMA = "ember_pi05_v6_paired_video_joint_functional_credit_v1"
V6_PRIOR_RUN_SCHEMA = "ember_pi05_v6_paired_video_joint_functional_credit_run_v1"
V6_PRIOR_PROFILE_SCHEMA = (
    "ember_pi05_v6_paired_video_joint_functional_credit_profile_v1"
)
V6_PRIOR_COMPLETION_SCHEMA = (
    "ember_pi05_v6_paired_video_joint_functional_credit_completion_v1"
)
V6_PRIOR_MODES = ("mechanism-profile", "formal")
_ACTIVE_AUTHORITY_REF = "origin/codex/bci-continuation"
_PICK_GC_PROVENANCE_CONFIG = (
    REPO_ROOT / "configs/pi05_v6_policy_innovation_goal_causal_key_v1.json"
)
_CVEG_PROVENANCE_CONFIG = (
    REPO_ROOT / "configs/pi05_v6_paired_candidate_update_guard_v1.json"
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
    "objective",
    "rng",
    "optimization",
    "profile_run",
    "formal_run",
    "evaluation",
}
_UNCHANGED_PICK_GC_SECTIONS = (
    "authorities",
    "writer",
    "condition_feature",
    "program_residual",
)
_EXPECTED_METHOD = {
    "name": "frozen_v6_paired_video_joint_functional_credit",
    "writer_input": "exact task language plus exactly one action-hidden teacher video",
    "dynamic_value": "one_raw_teacher_video_only",
    "training_views_per_task": 2,
    "deployment_views_per_task": 1,
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
    "source_actions_enter_only": (
        "two_correct_view_functional_losses_on_one_shared_cross_episode_B20_batch"
    ),
    "training_reward_reads": 0,
    "training_outcome_rollouts": 0,
    "training_companion_video_scope": (
        "one_distinct_same_task_action_hidden_ordered_video_with_complete_"
        "independent_writer_graph"
    ),
    "deployment_companion_video_count": 0,
    "negative_action_forwards": 0,
    "validation_action_or_reward_reads": 0,
    "test_action_or_reward_reads": 0,
}
_EXPECTED_UPDATE = {
    "kind": "paired_video_joint_functional_credit_full96",
    "correct_conditions": 48,
    "negative_conditions": 48,
    "views_per_task": 2,
    "view_weights": [0.5, 0.5],
    "ordering": (
        "primary_correct_0_to_23_then_companion_correct_0_to_23_then_"
        "primary_negative_0_to_23_then_companion_negative_0_to_23"
    ),
    "negative_schedule": "task_ordinal_plus_task_visit_modulo_reversed_shuffled_wrong",
    "negative_counts_per_macro": {"reversed": 16, "shuffled": 16, "wrong": 16},
    "correct_rhs": (
        "each_view_complete_program_cotangent_from_same_B20_and_same_policy_rng"
    ),
    "negative_rhs": "exact_zero_incremental_program_motion",
    "joint_solve": "sqrt_half_weighted_96_row_kernel_ridge",
    "damping_energy": "weighted_mean_unscaled_feature_row_energy",
    "view_swap_invariant": True,
    "duplicate_view_degenerates_to_single_view": True,
    "step_size": 1,
    "relative_damping": 0.01,
    "small_solve_dtype": "float64",
    "large_rhs_and_memory_write_dtype": "float32",
    "optimizer": "none_manual_add",
    "persistent_precision_or_optimizer_state": False,
    "history_replay": False,
    "momentum": False,
    "weight_decay": False,
    "gradient_clip": False,
    "global_scale_margin_or_cap": False,
    "task_scalar_gate_or_mask": False,
}
_EXPECTED_DATA = {
    "task_count": 24,
    "episodes_per_task": 50,
    "demo_indices": [0, 49],
    "action_chunk_size": 50,
    "action_queries_per_task": 20,
    "videos_per_task_per_macro": 1,
    "training_companion_videos_per_task_per_macro": 1,
    "teacher_video_schedule": (
        "deterministic_no_replacement_primary_plus_next_legal_cycle_companion"
    ),
    "teacher_action_episode_overlap": False,
    "task_aggregation": "two_half_weight_views_then_full24_equal_task_weight",
    "sampler_seed": 20260721,
    "teacher_video_seed": 20260722,
    "counterfactual_seed": 20260809,
    "wrong_video_schedule": (
        "two_distinct_deterministic_cross_suite_videos_with_current_task_language"
    ),
}
_EXPECTED_OBJECTIVE = {
    "name": "paired_video_joint_source_functional_program_credit",
    "positive_policy_randomness": {
        "scope": "same_flow_noise_and_time_for_both_views_of_each_task_query",
        "seed_scheme": TASK_LOGICAL_BATCH_POLICY_RNG_SCHEME,
        "flow_time_sampling_scheme": INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
        "flow_noise_sampling_scheme": INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    },
    "view_loss": (
        "each_ordered_video_has_its_own_complete_lora_and_B20_functional_loss"
    ),
    "task_loss_weight": "one_half_primary_plus_one_half_companion",
    "reward_use": "none",
    "policy_backward": False,
    "trajectory_replay": False,
    "negative_policy_forwards": 0,
    "critic": False,
    "progress_reward": False,
    "spsa": False,
}
_EXPECTED_RNG = {
    "view_policy_rng": (
        "same_task_logical_B20_keyed_seed_for_primary_and_companion"
    ),
    "rank_rng_checkpointed": True,
    "scheme": "order_independent_splitmix64_without_rank_or_physical_microbatch",
}
_EXPECTED_OPTIMIZATION = {
    "precision": "bfloat16",
    "seed": 7,
    "functional_policy_microbatch_size": 10,
    "optimizer": "none",
    "distributed_update": {
        "kind": (
            "host_local_completion_driven_task_claim_then_full96_paired_joint_"
            "manual_write"
        ),
        "world_size": "fresh_live_1_to_6_then_exact_resume_locked",
        "task_assignment": "host_local_atomic_completion_driven_long_first_train24",
        "retained_task_cap_per_rank": "max_8_or_ceil_train24_over_world_size",
        "memory_allreduce": False,
        "nccl_p2p_disable": "1",
        "nccl_algo": "Ring",
        "nccl_proto": "Simple",
        "deferred_process_group": True,
    },
}
_PROFILE_STATIC = {
    "allowed_world_sizes": [1, 2, 3, 4, 5, 6],
    "maximum_world_size": 6,
    "task_assignment": "host_local_atomic_completion_driven_long_first_train24",
    "schedule_macro": 0,
    "diagnostic_macros": 1,
    "num_workers_per_rank": 0,
    "retain_weight": False,
}
_EXPECTED_PROFILE_BASELINE = {
    "path": (
        "runs/outputs/pi05_cveg_cross_video_equivariant_candidate_guard_full24_"
        "reprofile_macro0_r3_b20_2eb1e8e_20260812/mechanism_profile.json"
    ),
    "schema": "ember_pi05_v6_cross_video_equivariant_candidate_guard_profile_v1",
    "source_world_size": 3,
    "step_seconds": 584.6491581050213,
    "maximum_ratio": 0.5,
}
_EXPECTED_PROFILE_GATES = {
    "positive_feature_rank_min": 24,
    "full_feature_rank_min": 48,
    "regularized_condition_max": 200,
    "both_view_descent_task_count_min": 12,
    "negative_to_correct_motion_rms_max": 0.15,
    "negative_null_per_kind_min": 12,
    "predicted_observed_relative_rms_max": 0.005,
    "retained_task_cap_max": 24,
    "queue_claim_seconds_max": 1,
    "step_seconds_max": 292.32457905251065,
    "oom_count": 0,
    "nonfinite_count": 0,
}
_FORMAL_STATIC = {
    "allowed_world_sizes": [1, 2, 3, 4, 5, 6],
    "maximum_world_size": 6,
    "task_assignment": "host_local_atomic_completion_driven_long_first_train24",
    "num_workers_per_rank": 0,
    "total_macros": 10,
    "checkpoint_macros": list(range(1, 11)),
    "launch_segments": [[0, 5], [5, 10]],
    "strict400_checkpoints": [0, 5, 10],
}
_EXPECTED_FORMAL_GATES = {
    "macro5_correct_min": 140,
    "macro5_breadth_min": 6,
    "macro5_lost_to_macro0_max": 10,
    "macro5_gained_must_exceed_lost": True,
    "macro5_non_declining_suite_count_min": 3,
    "macro5_single_task_net_gain_fraction_max": 0.5,
    "first_full_control_correct_min": 144,
    "goal_correct_min": 151,
    "goal_correct_strictly_exceeds_negative_controls": True,
    "goal_same_task_other_correct_ratio_min": 0.9,
    "macro10_requires_macro5_gate": True,
}
_EXPECTED_EVALUATION_STATIC = {
    "formal_status": "sealed_from_unchanged_v6_residual_deployment_graph",
    "maximum_world_size": 6,
    "device_selection": (
        "use_up_to_six_healthy_same_node_devices_with_sufficient_peak_memory_"
        "headroom_without_waiting_low_utilization_partial_occupancy_is_allowed"
    ),
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
    ),
    (
        "active_formal_ready",
        "sealed_from_live_a40_fresh0_to1_profile",
        "ready_after_live_profile_seal",
    ),
    (
        "formal_result_sealed",
        "sealed_from_live_a40_fresh0_to1_profile",
        "formal_result_sealed",
    ),
    (
        "profile_result_sealed_nonpass",
        "profile_result_sealed_nonpass",
        "blocked_by_profile_nonpass",
    ),
}


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    try:
        path = (REPO_ROOT / str(config["authorities"][name]["path"])).resolve()
    except (KeyError, TypeError, ValueError) as error:
        raise ExpertManifoldError(f"missing PVJFC Writer authority: {name}") from error
    if not path.is_file():
        raise ExpertManifoldError(f"PVJFC Writer authority is missing: {name}")
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
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
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


def _provenance_sections_match(config: Mapping[str, Any]) -> bool:
    try:
        pick_gc = json.loads(_PICK_GC_PROVENANCE_CONFIG.read_text(encoding="utf-8"))
        cveg = json.loads(_CVEG_PROVENANCE_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(
        config.get(name) == pick_gc.get(name)
        for name in _UNCHANGED_PICK_GC_SECTIONS
    ) and config.get("evaluation", {}).get("online_smoke_evidence") == cveg.get(
        "evaluation", {}
    ).get("online_smoke_evidence")


def _lifecycle_matches(config: Mapping[str, Any]) -> bool:
    profile = config.get("profile_run", {})
    formal = config.get("formal_run", {})
    state = (
        config.get("status"),
        profile.get("status"),
        formal.get("status"),
    )
    profile_artifact = profile.get("artifact_evidence")
    formal_artifact = formal.get("artifact_evidence")
    if state not in _COHERENT_STATES:
        return False
    if state[0] == "active_cpu_ready_awaiting_live_profile":
        return profile_artifact is formal_artifact is None
    if state[0] == "active_formal_ready":
        return isinstance(profile_artifact, Mapping) and bool(profile_artifact) and formal_artifact is None
    if state[0] == "formal_result_sealed":
        return all(isinstance(value, Mapping) and bool(value) for value in (profile_artifact, formal_artifact))
    return isinstance(profile_artifact, Mapping) and bool(profile_artifact) and formal_artifact is None


def _config_matches(config: Mapping[str, Any]) -> bool:
    profile = config.get("profile_run", {})
    formal = config.get("formal_run", {})
    evaluation = config.get("evaluation", {})
    return (
        set(config) == _EXPECTED_TOP_LEVEL
        and config.get("schema_version") == V6_PRIOR_CONFIG_SCHEMA
        and config.get("method") == _EXPECTED_METHOD
        and config.get("initialization") == _EXPECTED_INITIALIZATION
        and config.get("information_wall") == _EXPECTED_INFORMATION_WALL
        and config.get("update") == _EXPECTED_UPDATE
        and config.get("data") == _EXPECTED_DATA
        and config.get("objective") == _EXPECTED_OBJECTIVE
        and config.get("rng") == _EXPECTED_RNG
        and config.get("optimization") == _EXPECTED_OPTIMIZATION
        and set(profile) == {
            "status",
            "throughput_baseline",
            "gates",
            "artifact_evidence",
            *_PROFILE_STATIC,
        }
        and {name: profile.get(name) for name in _PROFILE_STATIC} == _PROFILE_STATIC
        and profile.get("throughput_baseline") == _EXPECTED_PROFILE_BASELINE
        and profile.get("gates") == _EXPECTED_PROFILE_GATES
        and set(formal) == {
            "status",
            "decision_gates",
            "artifact_evidence",
            *_FORMAL_STATIC,
        }
        and {name: formal.get(name) for name in _FORMAL_STATIC} == _FORMAL_STATIC
        and formal.get("decision_gates") == _EXPECTED_FORMAL_GATES
        and set(evaluation) == {
            "online_smoke_evidence",
            *_EXPECTED_EVALUATION_STATIC,
        }
        and {
            name: evaluation.get(name) for name in _EXPECTED_EVALUATION_STATIC
        }
        == _EXPECTED_EVALUATION_STATIC
        and _provenance_sections_match(config)
        and _lifecycle_matches(config)
    )


def load_v6_prior_config(path: Path = V6_PRIOR_CANONICAL_CONFIG) -> dict[str, Any]:
    path = path.resolve()
    if path != V6_PRIOR_CANONICAL_CONFIG.resolve() or not path.is_file():
        raise ExpertManifoldError("PVJFC Writer requires its canonical config path")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExpertManifoldError("invalid PVJFC Writer config") from error
    if not isinstance(config, dict) or not _config_matches(config):
        raise ExpertManifoldError("PVJFC Writer config violates its fail-closed contract")
    for name in config["authorities"]:
        authority_path(config, name)
    initialization = (REPO_ROOT / str(config["initialization"]["checkpoint"])).resolve()
    if not initialization.is_dir():
        raise ExpertManifoldError("PVJFC historical v6 initialization is missing")
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
            raise ExpertManifoldError("PVJFC mechanism profile is not authorized")
        return 1, (), int(config["profile_run"]["schedule_macro"])
    if mode == "formal":
        if (
            config.get("status") != "active_formal_ready"
            or config["formal_run"]["status"] != "ready_after_live_profile_seal"
        ):
            raise ExpertManifoldError("PVJFC formal training is blocked by live gates")
        formal = config["formal_run"]
        return (
            int(formal["total_macros"]),
            tuple(int(value) for value in formal["checkpoint_macros"]),
            0,
        )
    raise ExpertManifoldError(f"unsupported PVJFC Writer mode: {mode}")
