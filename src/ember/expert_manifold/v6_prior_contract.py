"""Canonical scientific and launch contract for the active PICK-GC Writer."""

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
    REPO_ROOT / "configs/pi05_v6_policy_innovation_goal_causal_key_v1.json"
)
V6_PRIOR_CONFIG_SCHEMA = "ember_pi05_v6_policy_innovation_goal_causal_key_v1"
V6_PRIOR_RUN_SCHEMA = "ember_pi05_v6_policy_innovation_goal_causal_key_launch_v1"
V6_PRIOR_PROFILE_SCHEMA = "ember_pi05_v6_policy_innovation_goal_causal_key_profile_v1"
V6_PRIOR_COMPLETION_SCHEMA = (
    "ember_pi05_v6_policy_innovation_goal_causal_key_completion_v1"
)
V6_PRIOR_MODES = ("mechanism-profile", "formal")
_ACTIVE_AUTHORITY_REF = "origin/codex/bci-continuation"

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
    "optimization",
    "cache_gate",
    "profile_run",
    "formal_run",
    "evaluation",
}
_EXPECTED_AUTHORITIES = {
    "task_expert_config": {"path": "configs/pi05_video_expert_manifold_v1.json"},
    "target_data_manifest": {"path": "configs/pi05_target_data_v1/manifest.json"},
    "evaluation_config": {"path": "configs/pi05_target_evaluation_v1.json"},
    "lora_contract": {"path": "configs/pi05_lora_v1.json"},
    "source_base_config": {"path": "configs/pi05_source_base_v1.json"},
}
_EXPECTED_METHOD = {
    "name": "frozen_v6_policy_innovation_goal_causal_key",
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
    "writer_parameter_count": 10_775_296,
    "residual_memory": "elementwise_zero_on_fresh_or_memory_only_exact_resume",
    "optimizer": "not_instantiated",
    "scheduler": "not_instantiated",
    "scaler": "not_instantiated",
}
_EXPECTED_WRITER = {
    "architecture": (
        "frozen_pi05_v6_plus_policy_innovation_goal_causal_keyed_program_residual_v1"
    ),
    "base_architecture": (
        "pi05_task_grounded_semantic_set_visual_transition_causal_procedure_"
        "slot_fusion_v6"
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
_EXPECTED_CONDITION_FEATURE = {
    "kind": (
        "frozen_source_policy_zero_image_subtracted_phase_aligned_"
        "goal_causal_jl_v1"
    ),
    "input": (
        "final_task_span_language_mean_plus_final_action_suffix_mean_"
        "actual_frame_minus_matched_zero_image"
    ),
    "source_policy": (
        "same_frozen_pi05_libero_policy_used_for_functional_loss_and_rollout"
    ),
    "baseline": "same_language_same_fixed_suffix_noise_zero_image",
    "innovation_width": 3072,
    "phase_slots": 16,
    "phase_alignment": "linear_order_preserving_align_corners",
    "goal_block": (
        "terminal_quartile_mean_minus_whole_video_mean_policy_innovation"
    ),
    "causal_block": (
        "sqrt_normalized_causal_prefix_mean_of_phase_centered_policy_innovation"
    ),
    "descriptor_width": 6144,
    "feature_width": 256,
    "projection_seed": 20260810,
    "projection_shape": [2, 128, 3072],
    "projection": ("two_fixed_no_bias_row_normalized_fp32_nonpersistent_blocks"),
    "block_normalization": "independent_zero_preserving_l2",
    "fusion": "concatenate_then_zero_preserving_l2",
    "ordered_negative_hot_path": (
        "each_real_frame_encoded_once_then_hidden_reordered_before_phase_alignment"
    ),
    "formal_evaluator": "actual_reordered_frames_complete_lora_forward",
    "learned_parameters": 0,
}
_EXPECTED_PROGRAM_RESIDUAL = {
    "program_slots": 320,
    "program_width": 256,
    "feature_width": 256,
    "value_count": 20_971_520,
    "dtype": "float32",
    "initialization": "elementwise_zero",
    "fusion": "single_add_before_frozen_historical_factor_heads",
    "checkpoint_tensor_count": 1,
}
_EXPECTED_UPDATE = {
    "kind": "full48_blind_counterfactual_null_condition_kernel",
    "correct_conditions": 24,
    "negative_conditions": 24,
    "ordering": ("correct_task_ordinal_0_to_23_then_negative_task_ordinal_0_to_23"),
    "negative_schedule": (
        "task_ordinal_plus_task_visit_modulo_reversed_shuffled_wrong"
    ),
    "negative_counts_per_macro": {"reversed": 8, "shuffled": 8, "wrong": 8},
    "correct_rhs": ("negative_correct_condition_policy_functional_program_cotangent"),
    "negative_rhs": "exact_zero_incremental_program_motion",
    "step_size": 1.0,
    "relative_damping": 0.01,
    "small_solve_dtype": "float64",
    "large_rhs_and_memory_write_dtype": "float32",
    "optimizer": "none_manual_add",
    "persistent_precision_or_optimizer_state": False,
    "momentum": False,
    "weight_decay": False,
    "gradient_clip": False,
    "global_scale_or_cap": False,
}
_EXPECTED_DATA = {
    "task_count": 24,
    "episodes_per_task": 50,
    "demo_indices": [0, 49],
    "action_chunk_size": 50,
    "action_queries_per_task": 20,
    "videos_per_task_per_macro": 1,
    "teacher_video_schedule": "deterministic_no_replacement_cycles",
    "teacher_action_episode_overlap": False,
    "task_aggregation": "task_local_B20_mean_with_no_cross_task_rescale",
    "sampler_seed": 20260721,
    "teacher_video_seed": 20260722,
    "counterfactual_seed": 20260809,
    "wrong_video_schedule": (
        "deterministic_cross_suite_cycle_with_current_task_language"
    ),
}
_EXPECTED_OBJECTIVE = {
    "name": "correct_condition_policy_functional_flow_loss_only",
    "positive_policy_randomness": {
        "scope": "one_independent_flow_noise_and_time_per_action_query",
        "seed_scheme": TASK_LOGICAL_BATCH_POLICY_RNG_SCHEME,
        "flow_time_sampling_scheme": INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
        "flow_noise_sampling_scheme": INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    },
}
_EXPECTED_OPTIMIZATION = {
    "precision": "bfloat16",
    "seed": 7,
    "functional_policy_microbatch_size": 10,
    "physical_policy_forwards_per_task": 2,
    "extra_negative_policy_forwards_per_task": 0,
    "distributed_update": {
        "kind": (
            "all_gather_local6_features_and_cotangents_then_identical_"
            "local_manual_write"
        ),
        "world_size": 4,
        "tasks_per_rank": 6,
        "memory_allreduce": False,
        "nccl_p2p_disable": "1",
        "nccl_algo": "Ring",
        "nccl_proto": "Simple",
        "deferred_process_group": True,
    },
}
_EXPECTED_CACHE_GATE = {
    "status": "passed_on_canonical_train24_action_hidden_feature_cache",
    "same_task_complete_cosine_mean_min": 0.90,
    "cross_task_complete_cosine_mean_max": 0.30,
    "correct_reversed_complete_cosine_max": 0.0,
    "shuffled_complete_cosine_abs_mean_max": 0.10,
    "correct24_rank_min": 24,
    "regularized_condition_number_max": 150.0,
}
_PROFILE_STATIC = {
    "expected_world_size": 4,
    "tasks_per_rank": 6,
    "schedule_macro": 0,
    "diagnostic_macros": 1,
    "num_workers_per_rank": 2,
    "retain_weight": False,
}
_FORMAL_STATIC = {
    "expected_world_size": 4,
    "tasks_per_rank": 6,
    "num_workers_per_rank": 2,
    "total_macros": 25,
    "checkpoint_macros": [10, 25],
    "strict400_checkpoints": [0, 10, 25],
}
_EXPECTED_PROFILE_BASELINE = {
    "path": (
        "runs/outputs/pi05_pick_gc_goal_causal_full48_profile_macro0_r6_b20_"
        "717b561_20260811/mechanism_profile.json"
    ),
    "schema": "ember_pi05_v6_policy_innovation_goal_causal_key_profile_v1",
    "schedule_macro": 0,
    "task_count": 24,
    "action_queries_per_task": 20,
    "source_world_size": 6,
    "source_tasks_per_rank": 4,
    "target_world_size": 4,
    "target_tasks_per_rank": 6,
    "scaling": (
        "source_step_seconds_times_target_tasks_per_rank_over_"
        "source_tasks_per_rank"
    ),
    "source_step_seconds": 25.351229034829885,
    "step_seconds": 38.02684355224483,
}
_EXPECTED_PROFILE_GATES = {
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
_EXPECTED_FORMAL_GATES = {
    "macro10_correct_min": 144,
    "macro10_breadth_min": 6,
    "macro10_lost_to_macro0_max": 8,
    "macro10_gained_must_exceed_lost": True,
    "first_full_six_arm_correct_min": 144,
    "goal_full_six_arm_correct_min": 151,
    "goal_correct_strictly_exceeds_negative_controls": True,
    "goal_same_task_other_correct_ratio_min": 0.9,
    "macro25_requires_macro10_gate": True,
}
_EXPECTED_FORMAL_ARTIFACT_EVIDENCE = {
    "training": {
        "root": (
            "runs/outputs/"
            "pi05_pick_gc_goal_causal_formal_fresh0to10_r4_b20_c2e1ff8_20260811"
        ),
        "run_commit": "c2e1ff878b6b68cb5bc45bb5443cdbd54ab8e62a",
        "completed_macro": 10,
        "completion": {
            "path": (
                "runs/outputs/"
                "pi05_pick_gc_goal_causal_formal_fresh0to10_r4_b20_c2e1ff8_"
                "20260811/completion.json"
            ),
            "bytes": 201,
            "schema": (
                "ember_pi05_v6_policy_innovation_goal_causal_key_completion_v1"
            ),
        },
        "checkpoint_manifest": {
            "path": (
                "runs/outputs/"
                "pi05_pick_gc_goal_causal_formal_fresh0to10_r4_b20_c2e1ff8_"
                "20260811/checkpoints/macro_00000010/manifest.json"
            ),
            "bytes": 11500,
            "schema": (
                "ember_pi05_v6_policy_innovation_goal_causal_key_checkpoint_v1"
            ),
        },
        "metrics": {
            "path": (
                "runs/outputs/"
                "pi05_pick_gc_goal_causal_formal_fresh0to10_r4_b20_c2e1ff8_"
                "20260811/metrics.jsonl"
            ),
            "bytes": 203002,
            "rows": 10,
        },
    },
    "strict_correct400": {
        "root": (
            "runs/outputs/"
            "pi05_pick_gc_goal_causal_correct400_noreplacement_seed7_"
            "macro0010_retry1_398425e_20260811"
        ),
        "evaluation_commit": "398425ee018097ba4c446f91bfe04ea65f6c7c5f",
        "results": {
            "path": (
                "runs/outputs/"
                "pi05_pick_gc_goal_causal_correct400_noreplacement_seed7_"
                "macro0010_retry1_398425e_20260811/results.json"
            ),
            "bytes": 1823130,
            "schema": "ember_pi05_target_eval_results_v2",
            "episodes": 400,
            "successes": 138,
            "breadth": 6,
        },
        "launcher_completion": {
            "path": (
                "runs/outputs/"
                "pi05_pick_gc_goal_causal_correct400_noreplacement_seed7_"
                "macro0010_retry1_398425e_20260811/launcher_completion.json"
            ),
            "bytes": 2251,
            "schema": "ember_pi05_eval_launcher_completion_v1",
        },
        "decision_evidence": {
            "path": (
                "runs/outputs/"
                "pi05_pick_gc_goal_causal_correct400_noreplacement_seed7_"
                "macro0010_retry1_398425e_20260811/"
                "pick_gc_formal_decision_evidence.json"
            ),
            "bytes": 64464,
            "schema": "ember_pi05_pick_gc_formal_decision_evidence_v1",
            "passed": False,
        },
    },
    "decision": {
        "macro10_gate_passed": False,
        "resume_macro10_to25_authorized": False,
        "six_arm_controls_authorized": False,
        "retained_gained_lost_to_macro0": [118, 20, 16],
        "scope": "retire_pick_gc_plus_blind_offline_source_action_credit_only",
    },
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
        "awaiting_live_pick_gc_deployment_profile",
    ),
    (
        "active_formal_ready",
        "sealed_from_live_a40_fresh0_to1_profile",
        "ready_after_live_profile_seal",
        "sealed_from_live_pick_gc_deployment_profile",
    ),
    (
        "formal_result_sealed",
        "sealed_from_live_a40_fresh0_to1_profile",
        "formal_result_sealed",
        "sealed_from_live_pick_gc_deployment_profile",
    ),
}


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    """Resolve one registered authority and reject missing files."""

    try:
        row = config["authorities"][name]
        path = (REPO_ROOT / str(row["path"])).resolve()
    except (KeyError, TypeError, ValueError) as error:
        raise ExpertManifoldError(f"missing PICK-GC Writer authority: {name}") from error
    if not path.is_file():
        raise ExpertManifoldError(f"PICK-GC Writer authority is missing: {name}")
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


def _static_projection_matches(
    value: object,
    expected: Mapping[str, Any],
) -> bool:
    return isinstance(value, Mapping) and {
        name: value.get(name) for name in expected
    } == dict(expected)


def _state_tuple(config: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    return (
        config.get("status"),
        config.get("profile_run", {}).get("status"),
        config.get("formal_run", {}).get("status"),
        config.get("evaluation", {}).get("formal_status"),
    )


def _profile_and_formal_match(config: Mapping[str, Any]) -> bool:
    profile = config.get("profile_run")
    formal = config.get("formal_run")
    evaluation = config.get("evaluation")
    if not all(isinstance(value, Mapping) for value in (profile, formal, evaluation)):
        return False
    profile_artifact = profile.get("artifact_evidence")
    formal_artifact = formal.get("artifact_evidence")
    evaluation_artifact = evaluation.get("online_smoke_evidence")
    initial = config.get("status") == "active_cpu_ready_awaiting_live_profile"
    sealed = config.get("status") in {"active_formal_ready", "formal_result_sealed"}
    artifacts_valid = (
        initial and profile_artifact is None and evaluation_artifact is None
    ) or (
        sealed
        and isinstance(profile_artifact, Mapping)
        and bool(profile_artifact)
        and isinstance(evaluation_artifact, Mapping)
        and bool(evaluation_artifact)
    )
    formal_artifact_valid = (
        config.get("status") != "formal_result_sealed"
        and formal_artifact is None
    ) or (
        config.get("status") == "formal_result_sealed"
        and formal_artifact == _EXPECTED_FORMAL_ARTIFACT_EVIDENCE
    )
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
        and _static_projection_matches(formal, _FORMAL_STATIC)
        and profile.get("throughput_baseline") == _EXPECTED_PROFILE_BASELINE
        and profile.get("gates") == _EXPECTED_PROFILE_GATES
        and formal.get("decision_gates") == _EXPECTED_FORMAL_GATES
        and _static_projection_matches(evaluation, _EXPECTED_EVALUATION_STATIC)
        and artifacts_valid
        and formal_artifact_valid
    )


def _config_matches(config: Mapping[str, Any]) -> bool:
    information_wall = config.get("information_wall")
    return (
        set(config) == _EXPECTED_TOP_LEVEL
        and config.get("schema_version") == V6_PRIOR_CONFIG_SCHEMA
        and config.get("method") == _EXPECTED_METHOD
        and config.get("authorities") == _EXPECTED_AUTHORITIES
        and config.get("initialization") == _EXPECTED_INITIALIZATION
        and config.get("writer") == _EXPECTED_WRITER
        and config.get("condition_feature") == _EXPECTED_CONDITION_FEATURE
        and config.get("program_residual") == _EXPECTED_PROGRAM_RESIDUAL
        and config.get("update") == _EXPECTED_UPDATE
        and config.get("data") == _EXPECTED_DATA
        and config.get("objective") == _EXPECTED_OBJECTIVE
        and config.get("optimization") == _EXPECTED_OPTIMIZATION
        and config.get("cache_gate") == _EXPECTED_CACHE_GATE
        and isinstance(information_wall, Mapping)
        and information_wall.get("writer_video_split_roles")
        == ["train", "validation", "test"]
        and set(information_wall.get("writer_forbidden_inputs", ()))
        == {
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
        }
        and information_wall.get("source_actions_enter_only")
        == "correct_condition_functional_loss"
        and information_wall.get("negative_action_forwards") == 0
        and information_wall.get("validation_actions_read") == 0
        and information_wall.get("test_actions_read") == 0
        and _profile_and_formal_match(config)
    )


def load_v6_prior_config(path: Path = V6_PRIOR_CANONICAL_CONFIG) -> dict[str, Any]:
    """Load only the active PICK-GC config and fail closed on scientific drift."""

    path = path.resolve()
    if path != V6_PRIOR_CANONICAL_CONFIG.resolve() or not path.is_file():
        raise ExpertManifoldError("PICK-GC Writer requires its canonical config path")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExpertManifoldError("invalid PICK-GC Writer config") from error
    if not isinstance(config, dict) or not _config_matches(config):
        raise ExpertManifoldError(
            "PICK-GC Writer config violates its fail-closed contract"
        )
    for name in _EXPECTED_AUTHORITIES:
        authority_path(config, name)
    initialization = (REPO_ROOT / _EXPECTED_INITIALIZATION["checkpoint"]).resolve()
    if not initialization.is_dir():
        raise ExpertManifoldError("PICK-GC historical v6 initialization is missing")
    return config


def runtime_for_mode(
    config: Mapping[str, Any],
    mode: str,
) -> tuple[int, tuple[int, ...], int]:
    """Return the exact active runtime segment for profile or formal training."""

    if mode == "mechanism-profile":
        if config["profile_run"]["status"] not in {
            "awaiting_live_a40_fresh0_to1_profile",
            "sealed_from_live_a40_fresh0_to1_profile",
        }:
            raise ExpertManifoldError("PICK-GC mechanism profile is not authorized")
        return 1, (), int(config["profile_run"]["schedule_macro"])
    if mode == "formal":
        if (
            config["status"] != "active_formal_ready"
            or config["formal_run"]["status"] != "ready_after_live_profile_seal"
        ):
            raise ExpertManifoldError("PICK-GC formal training is blocked by live gates")
        formal = config["formal_run"]
        return (
            int(formal["total_macros"]),
            tuple(int(value) for value in formal["checkpoint_macros"]),
            0,
        )
    raise ExpertManifoldError(f"unsupported PICK-GC Writer mode: {mode}")
