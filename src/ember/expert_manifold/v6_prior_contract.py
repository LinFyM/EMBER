"""Fail-closed authority for the active frozen-v6 Reward-Credit Writer."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ember.expert_manifold.contract import ExpertManifoldError


REPO_ROOT = Path(__file__).resolve().parents[3]
V6_PRIOR_CANONICAL_CONFIG = (
    REPO_ROOT / "configs/pi05_v6_reward_credit_program_cotangent_v1.json"
)
V6_PRIOR_CONFIG_SCHEMA = "ember_pi05_v6_reward_credit_program_cotangent_v1"
V6_PRIOR_RUN_SCHEMA = "ember_pi05_v6_reward_credit_program_cotangent_run_v1"
V6_PRIOR_PROFILE_SCHEMA = "ember_pi05_v6_reward_credit_program_cotangent_profile_v2"
V6_PRIOR_COMPLETION_SCHEMA = (
    "ember_pi05_v6_reward_credit_program_cotangent_completion_v1"
)
V6_PRIOR_MODES = ("mechanism-profile", "formal")

_METHOD = {
    "name": "frozen_v6_reward_credit_program_cotangent",
    "writer_input": "exact task language plus exactly one action-hidden teacher video",
    "dynamic_value": "one_raw_teacher_video_only",
    "language_only_lora_path": False,
    "deployment_expert_bank_read": False,
    "deployment_output": "one complete 38-target rank16 public LoRA",
}
_AUTHORITIES = {
    "task_expert_config": {"path": "configs/pi05_video_expert_manifold_v1.json"},
    "target_data_manifest": {"path": "configs/pi05_target_data_v1/manifest.json"},
    "evaluation_config": {"path": "configs/pi05_target_evaluation_v1.json"},
    "lora_contract": {"path": "configs/pi05_lora_v1.json"},
    "source_base_config": {"path": "configs/pi05_source_base_v1.json"},
}
_INITIALIZATION = {
    "kind": "strict_load_and_freeze_historical_v6_fast_macro400",
    "checkpoint": (
        "runs/outputs/pi05_as_writer_v6_decay400_taskcomplete_dev_r4_b20_"
        "seed7_s2400_4efa737_20260729/checkpoints/step_00000400"
    ),
    "writer_state_tensor_count": 600,
    "writer_parameter_count": 10_775_296,
    "residual_memory": (
        "fresh_elementwise_zero_with_identity_reconciliation_or_joint_exact_resume"
    ),
    "optimizer": "not_instantiated",
    "scheduler": "not_instantiated",
    "scaler": "not_instantiated",
}
_INFORMATION_WALL = {
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
    "source_action_reads": 0,
    "training_reward_scope": "development_train_on_policy_rollouts_only",
    "training_observation_action_scope": (
        "executed_prefixes_from_development_train_on_policy_rollouts_only"
    ),
    "negative_action_forwards": 0,
    "validation_action_or_reward_reads": 0,
    "test_action_or_reward_reads": 0,
}

_WRITER_CRITICAL = {
    "architecture": "frozen_pi05_v6_plus_video_keyed_program_residual_v2",
    "base_architecture": (
        "pi05_task_grounded_semantic_set_visual_transition_causal_"
        "procedure_slot_fusion_v6"
    ),
    "frame_stride": 5,
    "max_frames_per_encoder_call": 32,
    "image_width": 2048,
    "expert_width": 1024,
    "program_width": 256,
    "text_meta_lora_rank": 4,
    "vl_meta_lora_rank": 4,
    "action_meta_lora_rank": 4,
    "patch_grounding_heads": 8,
    "action_horizon": 50,
    "padded_action_dim": 32,
    "semantic_core_heads": 8,
    "semantic_core_blocks": 2,
    "procedure_heads": 8,
    "procedure_blocks": 2,
    "visual_transition_heads": 8,
    "fusion_heads": 8,
    "factor_hidden_width": 256,
    "initialization_seed": 7,
    "activation_checkpointing": True,
    "effective_runtime_activation_checkpointing": False,
    "all_600_state_tensors_frozen": True,
}
_FEATURE_CRITICAL = {
    "kind": "zero_preserving_balanced_static_causal_visual_innovation_jl_v2",
    "input": "mean_valid_task_tokens_frame_evidence_minus_text_queries",
    "static_block": "video_dc_mean_visual_innovation",
    "dynamic_block": (
        "uniform_pool_sqrt_normalized_causal_prefix_of_video_centered_innovation"
    ),
    "descriptor_width": 512,
    "descriptor_block_width": 256,
    "feature_width": 256,
    "projection_seed": 20260810,
    "projection_shape": [2, 128, 256],
    "projection": "two_fixed_no_bias_row_normalized_fp32_nonpersistent_blocks",
    "block_normalization": "independent_zero_preserving_l2",
    "fusion": "concatenate_then_zero_preserving_l2",
    "frame_content_reordered_before_dynamic_block": True,
}
_PROGRAM_CRITICAL = {
    "program_slots": 320,
    "program_width": 256,
    "feature_width": 256,
    "value_count": 20_971_520,
    "dtype": "float32",
    "initialization": "elementwise_zero",
    "fusion": "single_add_before_frozen_historical_factor_heads",
    "deployment_checkpoint_tensor_count": 1,
}
_RECONCILIATION_CRITICAL = {
    "kind": "exact_anchored_recursive_least_squares",
    "feature_width": 256,
    "precision_shape": [256, 256],
    "precision_dtype": "float64",
    "initialization": "identity",
    "assimilated_rows_initial": 0,
    "rows_per_macro": 48,
    "macro_semantics": "one_complete_full24_reward_cycle",
    "history_storage": False,
    "checkpoint_owned": True,
    "deployment_owned": False,
}
_UPDATE_CRITICAL = {
    "kind": "full48_exact_anchored_reconciliation",
    "correct_conditions": 24,
    "negative_conditions": 24,
    "ordering": ("correct_task_ordinal_0_to_23_then_negative_task_ordinal_0_to_23"),
    "negative_schedule": (
        "task_ordinal_plus_task_visit_modulo_reversed_shuffled_wrong"
    ),
    "negative_counts_per_macro": {"reversed": 8, "shuffled": 8, "wrong": 8},
    "correct_rhs": "negative_reward_program_cotangent_descent_motion",
    "negative_rhs": "exact_zero_incremental_program_motion",
    "step_size": 1.0,
    "relative_damping": 0.01,
    "small_solve_dtype": "float64",
    "large_rhs_and_memory_write_dtype": "float32",
    "precision_transition": "lambda_inverse_feature_outer_product_add",
    "optimizer": "none_joint_manual_memory_and_precision_write",
    "momentum": False,
    "weight_decay": False,
    "gradient_clip": False,
    "global_scale_or_cap": False,
}
_DATA_CRITICAL = {
    "task_count": 24,
    "episodes_per_task": 50,
    "demo_indices": [0, 49],
    "videos_per_task_per_macro": 1,
    "teacher_video_schedule": "deterministic_no_replacement_cycles",
    "teacher_video_seed": 20260722,
    "counterfactual_seed": 20260809,
    "wrong_video_schedule": (
        "deterministic_cross_suite_cycle_with_current_task_language"
    ),
    "rank_assignment": (
        "sealed_failed_profile_cost_balanced_one_task_per_suite_per_rank"
    ),
    "rank_task_ordinals": [
        [3, 8, 13, 23],
        [5, 9, 16, 22],
        [1, 11, 15, 20],
        [0, 7, 14, 18],
        [2, 6, 12, 21],
        [4, 10, 17, 19],
    ],
    "rank_assignment_evidence": (
        "runs/outputs/pi05_v6_reward_credit_program_cotangent_"
        "profile_full24_k4_nmc4_r6_b2_20260810/mechanism_profile.json"
    ),
    "rank_assignment_cost": "schedule_macro0_rollout_seconds_plus_credit_seconds",
    "teacher_action_episode_pairing": False,
}
_ENVIRONMENT_CRITICAL = {
    "reset": "official_random_reset_without_set_init_state",
    "rollouts_per_task": 4,
    "dummy_settling_steps": 10,
    "dummy_action": [0, 0, 0, 0, 0, 0, -1],
    "render_resolution": 256,
    "model_resolution": 224,
    "camera_rotation_degrees": 180,
    "policy_chunk_size": 50,
    "action_execution_horizon": 5,
    "num_inference_steps": 10,
    "terminate_on_success": True,
    "retain_success_replay": True,
    "retain_failure_replay": True,
    "policy_batch_size": 4,
    "persistent_env_lanes_per_task": 4,
}
_OBJECTIVE_CRITICAL = {
    "name": ("binary_leave_one_out_executed_prefix_flow_reward_program_cotangent"),
    "binary_leave_one_out": "A_e=(4*R_e-sum_j_R_j)/3",
    "homogeneous_task_credit": "exact_zero_without_replay_policy_forward",
    "episode_aggregation": (
        "mean_chunks_within_episode_then_equal_mean_over_four_episodes"
    ),
    "flow_mc_samples": 4,
    "flow_time_sampling": ("task_keyed_independent_beta15_scaled_0999_offset_0001"),
    "flow_noise_sampling": "task_keyed_independent_gaussian",
    "flow_panel_physical_batch_invariance": True,
    "old_policy_forwards": 0,
    "negative_policy_forwards": 0,
    "learning_epochs": 1,
    "second_replay_epoch": False,
    "critic": False,
    "progress_reward": False,
    "ratio_objective": False,
    "spsa": False,
}
_OPTIMIZATION_CRITICAL = {
    "precision": "bfloat16",
    "seed": 7,
    "reward_replay_chunk_batch_size": 8,
    "rollout_policy_batch_size": 4,
    "optimizer": "none",
}
_DISTRIBUTED_CRITICAL = {
    "kind": (
        "cuda_complete_file_rendezvous_then_two_fixed_all_gathers_and_"
        "identical_local_manual_write"
    ),
    "world_size": 6,
    "tasks_per_rank": 4,
    "memory_allreduce": False,
    "nccl_p2p_disable": "1",
    "nccl_algo": "Ring",
    "nccl_proto": "Simple",
    "deferred_process_group": True,
}

_RNG_CRITICAL = {
    "environment_seed_root": 2026081101,
    "policy_noise_seed_root": 2026081102,
    "flow_credit_seed_root": 2026081103,
    "scheme": "order_independent_splitmix64_without_rank_or_physical_microbatch",
}
_PROFILE_GATES = {
    "tasks": 24,
    "rollouts": 96,
    "rollouts_per_task": 4,
    "videos": 24,
    "videos_per_task": 1,
    "mixed_tasks_min": 6,
    "homogeneous_tasks_min": 1,
    "mixed_cotangent_nonzero": True,
    "homogeneous_cotangent_exact_zero": True,
    "program_to_lora_response_nonzero": True,
    "program_to_all_mixed_fixed_action_response_nonzero": True,
    "mixed_action_probe_scope": "all_mixed_tasks",
    "mixed_suite_count": 4,
    "fixed_action_queries_per_mixed_task": 4,
    "fixed_action_policy_forwards_per_mixed_task": 2,
    "full48_feature_rank_min": 24,
    "negative_null_motion_ratio_max": 0.25,
    "predicted_observed_relative_rms_max": 0.005,
    "extra_negative_policy_forwards": 0,
    "old_policy_forwards": 0,
    "oom_count": 0,
    "nonfinite_count": 0,
    "watchdog_count": 0,
}
_PROFILE_CORE = {
    "registered_output_root": (
        "runs/outputs/pi05_v6_reward_credit_program_cotangent_"
        "profile_full24_k4_nmc4_r6_b8_allmixedk4_20260810"
    ),
    "expected_world_size": 6,
    "tasks_per_rank": 4,
    "schedule_macro": 0,
    "diagnostic_macros": 1,
    "num_workers_per_rank": 0,
    "retain_weight": False,
    "gates": _PROFILE_GATES,
}
_DECISION_EVALUATION = {
    "macro0_reference_root": (
        "runs/outputs/pi05_v6_balanced_causal_condition_residual_correct400_"
        "noreplacement_seed7_method_macro0000_6b5f7a6_20260810"
    ),
    "macro0_reference_commit": "6b5f7a6ad6ef1a778205071f38faec9f936cf54e",
    "macro0_reference_correct": 134,
    "macro0_reference_breadth": 6,
    "macro1_registered_root": (
        "runs/outputs/pi05_v6_reward_credit_program_cotangent_"
        "correct400_cycle0001_20260810"
    ),
    "macro2_registered_root": (
        "runs/outputs/pi05_v6_reward_credit_program_cotangent_"
        "correct400_cycle0002_20260810"
    ),
    "macro1_control_registered_roots": {
        condition: (
            "runs/outputs/pi05_v6_reward_credit_program_cotangent_"
            f"{condition}400_cycle0001_20260810"
        )
        for condition in (
            "same_task_other",
            "cross_suite_wrong",
            "shuffled",
            "reversed",
            "no_video",
        )
    },
    "macro2_control_registered_roots": {
        condition: (
            "runs/outputs/pi05_v6_reward_credit_program_cotangent_"
            f"{condition}400_cycle0002_20260810"
        )
        for condition in (
            "same_task_other",
            "cross_suite_wrong",
            "shuffled",
            "reversed",
            "no_video",
        )
    },
}
_DECISION_GATES = {
    "macro1_support_correct_min": 140,
    "macro1_support_lost_to_macro0_max": 6,
    "macro1_support_breadth_min": 6,
    "macro1_support_gained_must_exceed_lost": True,
    "first_full_six_arm_correct_min": 144,
    "goal_full_six_arm_correct_min": 151,
    "goal_correct_strictly_exceeds_negative_controls": True,
    "goal_same_task_other_correct_ratio_min": 0.9,
    "macro2_requires_macro1_support_gate": True,
}
_FORMAL_CORE = {
    "registered_output_root": (
        "runs/outputs/pi05_v6_reward_credit_program_cotangent_"
        "formal_cycle0to2_r6_k4_nmc4_b8_balanced_20260810"
    ),
    "expected_world_size": 6,
    "tasks_per_rank": 4,
    "num_workers_per_rank": 0,
    "total_macros": 2,
    "checkpoint_macros": [1, 2],
    "strict400_checkpoints": [0, 1, 2],
    "decision_evaluation": _DECISION_EVALUATION,
    "decision_gates": _DECISION_GATES,
}
_EVALUATION_CORE = {
    "formal_status": "sealed_from_unchanged_v6_residual_deployment_graph",
    "throughput_policy": (
        "highest_measured_batch_throughput_with_device_memory_headroom"
    ),
    "required_writer_model_batch_sizes": [8, 16, 32],
    "minimum_smoke_writer_model_batch_size": 8,
}
_DEPLOYMENT_EVIDENCE = {
    "schema": "ember_pi05_v6_condition_program_residual_deployment_seal_v1",
    "run_commit": "2af82aa6769570786c64d3c026374150d259360c",
    "writer_model_batch_size": 8,
    "profile": {
        "path": (
            "runs/outputs/pi05_v6_balanced_causal_condition_residual_writer_"
            "throughput_profile_val8x4_correct_gpu02g0_2af82aa_20260810/"
            "writer_generation_profile.json"
        ),
        "bytes": 10_225,
    },
    "vertical": {
        "path": (
            "runs/outputs/pi05_v6_balanced_causal_condition_residual_writer_"
            "vertical_smoke_val8x1_correct_b8_gpu02g0_2af82aa_20260810/"
            "results.json"
        ),
        "bytes": 92_811,
    },
    "cache_manifest": {
        "path": (
            "runs/outputs/pi05_v6_balanced_causal_condition_residual_writer_"
            "vertical_smoke_val8x1_correct_b8_gpu02g0_2af82aa_20260810/"
            "writer_lora_cache/cache_manifest.json"
        ),
        "bytes": 52_153,
    },
}

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
    "reconciliation",
    "update",
    "data",
    "environment",
    "objective",
    "optimization",
    "rng",
    "profile_run",
    "formal_run",
    "evaluation",
}
_CRITICAL_SECTIONS = {
    "method": _METHOD,
    "authorities": _AUTHORITIES,
    "initialization": _INITIALIZATION,
    "information_wall": _INFORMATION_WALL,
    "writer": _WRITER_CRITICAL,
    "condition_feature": _FEATURE_CRITICAL,
    "program_residual": _PROGRAM_CRITICAL,
    "reconciliation": _RECONCILIATION_CRITICAL,
    "update": _UPDATE_CRITICAL,
    "data": _DATA_CRITICAL,
    "environment": _ENVIRONMENT_CRITICAL,
    "objective": _OBJECTIVE_CRITICAL,
    "rng": _RNG_CRITICAL,
}


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    try:
        relative = Path(str(config["authorities"][name]["path"]))
    except (KeyError, TypeError) as error:
        raise ExpertManifoldError(f"missing reward-credit authority: {name}") from error
    path = (REPO_ROOT / relative).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ExpertManifoldError(
            f"reward-credit authority left the repository: {name}"
        ) from error
    if not path.is_file():
        raise ExpertManifoldError(f"missing reward-credit authority: {name}")
    return path


def _registered_root(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to((REPO_ROOT / "runs/outputs").resolve())
    except ValueError:
        return False
    return True


def _profile_artifact(value: object, *, config: Mapping[str, Any]) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        path = (REPO_ROOT / str(value["path"])).resolve()
        expected_bytes = int(value["bytes"])
        expected_root = (
            REPO_ROOT / str(_PROFILE_CORE["registered_output_root"])
        ).resolve()
        path.relative_to(expected_root)
        result = json.loads(path.read_text(encoding="utf-8"))
        run = json.loads(
            (expected_root / "run_contract.json").read_text(encoding="utf-8")
        )
        from ember.expert_manifold.v6_prior_profile import profile_passes

        passed, evidence = profile_passes(config, result["macros"])
    except (
        ExpertManifoldError,
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ):
        return False
    return (
        set(value) == {"path", "bytes", "schema", "passed", "run_commit"}
        and value.get("schema") == V6_PRIOR_PROFILE_SCHEMA
        and value.get("passed") is True
        and isinstance(value.get("run_commit"), str)
        and git_commit_in_active_authority_lineage(str(value["run_commit"]))
        and path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == expected_bytes > 0
        and path == expected_root / "mechanism_profile.json"
        and isinstance(result, Mapping)
        and result.get("schema_version") == V6_PRIOR_PROFILE_SCHEMA
        and result.get("passed") is True
        and result.get("schedule_macro") == 0
        and result.get("retain_weight") is False
        and result.get("gates") == _PROFILE_GATES
        and result.get("gate_evidence") == evidence
        and passed is True
        and run.get("git", {}).get("commit") == value.get("run_commit")
    )


def _state_matches(config: Mapping[str, Any]) -> bool:
    profile = config.get("profile_run", {})
    formal = config.get("formal_run", {})
    if not isinstance(profile, Mapping) or not isinstance(formal, Mapping):
        return False
    profile_core = {name: profile.get(name) for name in _PROFILE_CORE}
    formal_core = {name: formal.get(name) for name in _FORMAL_CORE}
    roots = (
        _PROFILE_CORE["registered_output_root"],
        _FORMAL_CORE["registered_output_root"],
        *_DECISION_EVALUATION.values(),
    )
    registered = tuple(value for value in roots if str(value).startswith("runs/"))
    common = (
        set(profile) == set(_PROFILE_CORE) | {"status", "artifact_evidence"}
        and set(formal) == set(_FORMAL_CORE) | {"status", "artifact_evidence"}
        and profile_core == _PROFILE_CORE
        and formal_core == _FORMAL_CORE
        and all(_registered_root(value) for value in registered)
        and len(set(registered)) == len(registered)
    )
    if not common:
        return False
    state = (
        config.get("status"),
        profile.get("status"),
        formal.get("status"),
    )
    if state == (
        "awaiting_live_a40_reward_credit_profile",
        "awaiting_live_a40_fresh0_to1_reward_profile",
        "blocked_until_live_reward_profile_passes_and_is_sealed",
    ):
        return (
            profile.get("artifact_evidence") is None
            and formal.get("artifact_evidence") is None
        )
    if state == (
        "formal_ready",
        "sealed_from_live_a40_fresh0_to1_reward_profile",
        "ready_after_live_reward_profile_seal",
    ):
        return formal.get("artifact_evidence") is None and _profile_artifact(
            profile.get("artifact_evidence"), config=config
        )
    return False


def _deployment_evidence_matches(value: object) -> bool:
    """Bind the inherited, already-sealed deployment graph without re-profiling."""

    if value != _DEPLOYMENT_EVIDENCE:
        return False
    try:
        records = tuple(
            value[name] for name in ("profile", "vertical", "cache_manifest")
        )
        paths = tuple((REPO_ROOT / str(record["path"])).resolve() for record in records)
        outputs = (REPO_ROOT / "runs/outputs").resolve()
        for path, record in zip(paths, records, strict=True):
            path.relative_to(outputs)
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != int(record["bytes"])
            ):
                return False
        paths[2].relative_to(paths[1].parent)
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return git_commit_in_active_authority_lineage(str(value["run_commit"]))


def _critical_sections_match(config: Mapping[str, Any]) -> bool:
    return all(
        config.get(name) == expected for name, expected in _CRITICAL_SECTIONS.items()
    )


def _optimization_matches(config: Mapping[str, Any]) -> bool:
    optimization = config.get("optimization")
    return (
        isinstance(optimization, Mapping)
        and set(optimization) == set(_OPTIMIZATION_CRITICAL) | {"distributed_update"}
        and {name: optimization.get(name) for name in _OPTIMIZATION_CRITICAL}
        == _OPTIMIZATION_CRITICAL
        and optimization.get("distributed_update", {}) == _DISTRIBUTED_CRITICAL
    )


def _evaluation_matches(config: Mapping[str, Any]) -> bool:
    evaluation = config.get("evaluation", {})
    return (
        isinstance(evaluation, Mapping)
        and set(evaluation) == set(_EVALUATION_CORE) | {"online_smoke_evidence"}
        and {name: evaluation.get(name) for name in _EVALUATION_CORE}
        == _EVALUATION_CORE
        and _deployment_evidence_matches(evaluation.get("online_smoke_evidence"))
    )


def _base_matches(config: Mapping[str, Any]) -> bool:
    return (
        set(config) == _EXPECTED_TOP_LEVEL
        and config.get("schema_version") == V6_PRIOR_CONFIG_SCHEMA
        and _critical_sections_match(config)
        and _optimization_matches(config)
        and _evaluation_matches(config)
        and _state_matches(config)
    )


def load_v6_prior_config(path: Path) -> dict[str, Any]:
    """Load only the active Reward-Credit schema; retired RLS fails closed."""

    path = path.resolve()
    if path != V6_PRIOR_CANONICAL_CONFIG.resolve() or not path.is_file():
        raise ExpertManifoldError("non-canonical Reward-Credit Writer config")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExpertManifoldError("invalid Reward-Credit Writer config JSON") from error
    if not isinstance(config, dict) or not _base_matches(config):
        raise ExpertManifoldError(
            "Reward-Credit Writer config violates its fail-closed contract"
        )
    for name in _AUTHORITIES:
        authority_path(config, name)
    initialization = (REPO_ROOT / _INITIALIZATION["checkpoint"]).resolve()
    if (
        not initialization.is_dir()
        or not (initialization / "writer.safetensors").is_file()
    ):
        raise ExpertManifoldError("historical v6 initialization asset is missing")
    return config


def runtime_for_mode(
    config: Mapping[str, Any], mode: str
) -> tuple[int, tuple[int, ...], int]:
    """Return total macros, checkpoints, and schedule origin.

    One macro is exactly one full24 K4 Reward-Credit cycle.  The retained macro
    field keeps checkpoint/evaluator compatibility without changing semantics.
    """

    if mode == "mechanism-profile":
        profile = config["profile_run"]
        if profile["status"] != "awaiting_live_a40_fresh0_to1_reward_profile":
            raise ExpertManifoldError("Reward-Credit profile is not launchable")
        return int(profile["diagnostic_macros"]), (), int(profile["schedule_macro"])
    if mode == "formal":
        profile = config["profile_run"]
        formal = config["formal_run"]
        if (
            profile["status"] != "sealed_from_live_a40_fresh0_to1_reward_profile"
            or formal["status"] != "ready_after_live_reward_profile_seal"
            or not isinstance(profile.get("artifact_evidence"), Mapping)
        ):
            raise ExpertManifoldError(
                "formal Reward-Credit training is blocked by the live profile"
            )
        return (
            int(formal["total_macros"]),
            tuple(int(value) for value in formal["checkpoint_macros"]),
            0,
        )
    raise ExpertManifoldError("unsupported Reward-Credit Writer mode")


def git_commit_in_active_authority_lineage(commit: str) -> bool:
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        return False
    try:
        return (
            subprocess.run(
                [
                    "git",
                    "merge-base",
                    "--is-ancestor",
                    commit,
                    "origin/codex/bci-continuation",
                ],
                cwd=REPO_ROOT,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
    except OSError:
        return False
