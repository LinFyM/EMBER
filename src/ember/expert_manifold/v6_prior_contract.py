"""Scientific and runtime authority for the v6-prior Expert-Manifold Writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.expert_manifold.contract import ExpertManifoldError
from ember.pi05_source_checkpoint import read_json
from ember.writer.architecture import validate_writer_dimensions


REPO_ROOT = Path(__file__).resolve().parents[3]
V6_PRIOR_CONFIG_SCHEMA = "ember_pi05_v6_prior_policy_effective_writer_v1"
V6_PRIOR_MODES = ("gradient-profile", "profile", "formal")


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    try:
        relative = str(config["authorities"][name]["path"])
    except (KeyError, TypeError) as error:
        raise ExpertManifoldError(f"missing v6-prior authority: {name}") from error
    return REPO_ROOT / relative


def _information_wall_matches(value: Mapping[str, Any]) -> bool:
    return value == {
        "expert_action_split_roles": ["train"],
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
        "validation_experts_trained": 0,
        "test_experts_trained": 0,
        "validation_actions_read": 0,
        "test_actions_read": 0,
    }


def _method_matches(value: Mapping[str, Any]) -> bool:
    return value == {
        "name": "v6_prior_policy_effective_temporal_ranking_writer",
        "writer_input": (
            "exact task language plus exactly one action-hidden teacher video"
        ),
        "dynamic_value": "one_raw_teacher_video_only",
        "language_only_lora_path": False,
        "deployment_expert_bank_read": False,
        "deployment_output": "one complete rank16 public LoRA",
    }


def _writer_matches(value: Mapping[str, Any]) -> bool:
    dimensions = {
        name: value.get(name)
        for name in (
            "image_width",
            "expert_width",
            "program_width",
            "text_meta_lora_rank",
            "vl_meta_lora_rank",
            "action_meta_lora_rank",
            "patch_grounding_heads",
            "action_horizon",
            "padded_action_dim",
            "semantic_core_heads",
            "semantic_core_blocks",
            "procedure_heads",
            "procedure_blocks",
            "visual_transition_heads",
            "fusion_heads",
            "factor_hidden_width",
        )
    }
    try:
        validate_writer_dimensions(dimensions)
    except ValueError:
        return False
    return (
        value.get("architecture")
        == (
            "pi05_task_grounded_semantic_set_visual_transition_"
            "causal_procedure_slot_fusion_v6"
        )
        and int(value.get("frame_stride", -1)) == 5
        and int(value.get("max_frames_per_encoder_call", -1)) == 32
        and int(value.get("initialization_seed", -1)) == 7
        and value.get("activation_checkpointing") is True
        and value.get("frozen_blocks")
        == [
            "semantic_encoder",
            "semantic_core",
            "visual_transition",
            "procedure",
        ]
        and value.get("trainable_blocks") == ["compiler", "factor_heads"]
        and int(value.get("frozen_parameter_count", -1)) == 7_060_992
        and int(value.get("trainable_parameter_count", -1)) == 3_714_304
    )


def _data_matches(value: Mapping[str, Any]) -> bool:
    return (
        int(value.get("task_count", -1)) == 24
        and int(value.get("episodes_per_task", -1)) == 50
        and value.get("demo_indices") == [0, 49]
        and int(value.get("action_chunk_size", -1)) == 50
        and int(value.get("action_queries_per_task", -1)) == 20
        and int(value.get("videos_per_task_per_macro", -1)) == 1
        and value.get("teacher_video_schedule")
        == "deterministic_no_replacement_cycles"
        and value.get("teacher_action_episode_overlap") is False
        and value.get("task_aggregation")
        == "mean_within_task_then_train24_equal_mean"
        and all(
            isinstance(value.get(name), int) and int(value[name]) >= 0
            for name in (
                "sampler_seed",
                "teacher_video_seed",
                "counterfactual_seed",
            )
        )
        and value.get("negative_schedule")
        == (
            "task_ordinal_plus_task_visit_modulo_reversed_shuffled_wrong"
        )
        and value.get("wrong_video_schedule")
        == "deterministic_cross_suite_cycle_with_current_task_language"
    )


def _objective_matches(value: Mapping[str, Any]) -> bool:
    expert = value.get("expert", {})
    ranking = value.get("ranking", {})
    weights = value.get("auxiliary_weights", {})
    status = weights.get("status")
    coefficients = (weights.get("expert"), weights.get("ranking"))
    if status == "blocked_until_live_train24_gradient_profile":
        valid_weights = coefficients == (None, None)
    elif status == "sealed_from_live_train24_gradient_profile":
        valid_weights = all(
            isinstance(item, (int, float)) and 0 <= float(item) <= 1
            for item in coefficients
        )
    else:
        valid_weights = False
    return (
        float(value.get("positive_functional_weight", -1)) == 1.0
        and value.get("positive_policy_randomness")
        == "one_independent_flow_noise_and_time_per_action_query"
        and expert.get("direction")
        == "one_minus_global_effective_ba_cosine"
        and expert.get("norm")
        == "smooth_l1_global_effective_log_norm_ratio"
        and float(expert.get("norm_weight", -1)) == 0.25
        and float(expert.get("smooth_l1_beta", -1)) == 0.5
        and ranking.get("form")
        == "temperature_scaled_softplus_required_minus_observed_margin"
        and float(ranking.get("required_margin", -1)) == 0.1
        and float(ranking.get("temperature", -1)) == 0.05
        and float(
            weights.get("maximum_fraction_of_positive_gradient_per_auxiliary", -1)
        )
        == 0.25
        and valid_weights
    )


def _optimization_matches(value: Mapping[str, Any]) -> bool:
    optimizer = value.get("optimizer", {})
    scheduler = value.get("scheduler", {})
    reduction = value.get("distributed_gradient_reduction", {})
    return (
        value.get("precision") == "bfloat16"
        and int(value.get("seed", -1)) == 7
        and optimizer
        == {
            "name": "AdamW",
            "peak_lr": 0.00003,
            "betas": [0.9, 0.95],
            "eps": 1e-08,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
        }
        and scheduler
        == {
            "kind": "cosine_decay_with_warmup",
            "warmup_macros": 2,
            "total_macros": 50,
            "decay_lr": 0.000003,
        }
        and reduction
        == {
            "kind": (
                "single_flat_parameter_ordered_allreduce_mean_after_local_task_mean"
            ),
            "nccl_p2p_disable": "1",
            "nccl_algo": "Ring",
            "nccl_proto": "Simple",
            "deferred_process_group": True,
        }
    )


def _runtime_declarations_match(config: Mapping[str, Any]) -> bool:
    gradient = config.get("gradient_profile", {})
    profile = config.get("profile_run", {})
    formal = config.get("formal_run", {})
    return (
        gradient.get("status")
        in {
            "blocked_until_single_a40_warmstart_reproduction_smoke",
            "ready_after_cpu_and_single_a40_warmstart_reproduction_smoke",
            "sealed_from_live_train24_gradient_profile",
        }
        and int(gradient.get("expected_world_size", -1)) == 6
        and int(gradient.get("tasks_per_rank", -1)) == 4
        and int(gradient.get("macros", -1)) == 1
        and int(gradient.get("schedule_macro", -1)) == 49
        and int(gradient.get("physical_policy_batch", -1)) == 20
        and int(gradient.get("longest_video_sampled_frames", -1)) == 105
        and gradient.get("seal_rule")
        == (
            "each_auxiliary_at_most_one_quarter_positive_gradient_in_both_"
            "compiler_and_factor_heads"
        )
        and profile.get("status")
        in {
            "blocked_until_live_gradient_weights",
            "sealed_from_live_gradient_profile_and_a40_resume_smoke",
        }
        and int(profile.get("expected_world_size", -1)) == 6
        and int(profile.get("tasks_per_rank", -1)) == 4
        and int(profile.get("total_macros", -1)) == 3
        and profile.get("checkpoint_macros") == [1, 3]
        and profile.get("required_resume_comparison")
        == "fresh0_to1_plus_exact_resume1_to3_equals_contiguous0_to3"
        and formal.get("status")
        in {
            "blocked_until_live_a40_profile_and_macro3_online_smoke",
            "sealed_from_live_a40_profile_and_macro3_online_smoke",
        }
        and int(formal.get("expected_world_size", -1)) == 6
        and int(formal.get("tasks_per_rank", -1)) == 4
        and int(formal.get("total_macros", -1)) == 50
        and formal.get("checkpoint_macros") == [10, 25, 50]
        and formal.get("strict80_checkpoints") == [0, 10, 25, 50]
    )


def _evaluation_matches(value: Mapping[str, Any]) -> bool:
    try:
        model_batch_size = int(value.get("writer_model_batch_size", -1))
    except (TypeError, ValueError):
        return False
    if model_batch_size != 1:
        return False
    status = value.get("formal_status")
    evidence = value.get("online_smoke_evidence")
    if status == "blocked_until_live_a40_warmstart_reproduction_smoke":
        return evidence is None
    if status != "sealed" or not isinstance(evidence, Mapping):
        return False
    exact = {
        "device": "NVIDIA A40",
        "checkpoint_kind": "historical_v6_macro400_load_only",
        "video_condition": "correct",
        "video_sampling": "without_replacement",
        "writer_modules_released": True,
        "source_policy_reused_for_rollout": True,
        "source_policy_reloaded": False,
        "staged_path_matches_direct_v6_forward": True,
        "success_interpretation": "execution_smoke_only_not_performance_evidence",
    }
    integers = {
        "validation_task_count": 8,
        "state_count": 1,
        "scientific_rows": 8,
        "generated_entries": 8,
        "cache_entries": 8,
        "writer_state_tensor_count": 600,
        "writer_model_batch_size": 1,
        "retry_count": 0,
        "failure_count": 0,
        "teacher_action_reads": 0,
        "teacher_state_reads": 0,
        "reward_reads": 0,
        "terminal_reads": 0,
        "oom_count": 0,
        "nonfinite_count": 0,
    }
    try:
        return (
            all(evidence.get(name) == expected for name, expected in exact.items())
            and all(
                int(evidence.get(name, -1)) == expected
                for name, expected in integers.items()
            )
            and isinstance(evidence.get("commit"), str)
            and bool(evidence["commit"])
            and isinstance(evidence.get("root"), str)
            and bool(evidence["root"])
            and float(evidence.get("staged_path_max_abs_difference", -1))
            <= 1e-5
            and float(evidence.get("staged_path_max_abs_difference", -1)) >= 0
        )
    except (TypeError, ValueError):
        return False


def load_v6_prior_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    authorities = config.get("authorities", {})
    initialization = config.get("initialization", {})
    basis = config.get("expert_basis", {})
    valid = (
        config.get("schema_version") == V6_PRIOR_CONFIG_SCHEMA
        and set(authorities)
        == {
            "asset_config",
            "target_data_manifest",
            "evaluation_config",
            "lora_contract",
            "source_base_config",
        }
        and all(authority_path(config, name).is_file() for name in authorities)
        and _method_matches(config.get("method", {}))
        and _information_wall_matches(config.get("information_wall", {}))
        and initialization.get("kind")
        == "load_only_historical_v6_fast_macro400"
        and int(initialization.get("writer_state_tensor_count", -1)) == 600
        and initialization.get("optimizer") == "fresh"
        and initialization.get("scheduler") == "fresh"
        and initialization.get("rng") == "fresh_seed"
        and (REPO_ROOT / str(initialization.get("checkpoint", ""))).is_dir()
        and _writer_matches(config.get("writer", {}))
        and basis
        == {
            "task_count": 24,
            "expert_step": 2000,
            "checkpoint_selection": (
                "one_uniform_step_for_all_24_tasks_no_task_specific_mixing"
            ),
            "training_only": True,
            "comparison_space": "global_gauge_invariant_policy_effective_ba",
        }
        and _data_matches(config.get("data", {}))
        and _objective_matches(config.get("objective", {}))
        and _optimization_matches(config.get("optimization", {}))
        and _runtime_declarations_match(config)
        and _evaluation_matches(config.get("evaluation", {}))
        and config.get("content_hash_policy") == "disabled_by_owner"
    )
    if not valid:
        raise ExpertManifoldError("v6-prior Writer scientific boundary changed")
    return config


def runtime_for_mode(
    config: Mapping[str, Any], mode: str
) -> tuple[int, tuple[int, ...]]:
    if mode not in V6_PRIOR_MODES:
        raise ExpertManifoldError("unsupported v6-prior runtime mode")
    if mode == "gradient-profile":
        profile = config["gradient_profile"]
        if (
            profile.get("status")
            != "ready_after_cpu_and_single_a40_warmstart_reproduction_smoke"
            or config["objective"]["auxiliary_weights"]["status"]
            != "blocked_until_live_train24_gradient_profile"
        ):
            raise ExpertManifoldError("v6-prior gradient profile is not ready")
        return int(profile["macros"]), ()
    selected = config["profile_run" if mode == "profile" else "formal_run"]
    required_status = (
        "sealed_from_live_gradient_profile_and_a40_resume_smoke"
        if mode == "profile"
        else "sealed_from_live_a40_profile_and_macro3_online_smoke"
    )
    if (
        selected.get("status") != required_status
        or config["objective"]["auxiliary_weights"]["status"]
        != "sealed_from_live_train24_gradient_profile"
    ):
        raise ExpertManifoldError(f"v6-prior {mode} runtime is not sealed")
    total = int(selected["total_macros"])
    checkpoints = tuple(int(value) for value in selected["checkpoint_macros"])
    if not total > 0 or not checkpoints or checkpoints[-1] != total:
        raise ExpertManifoldError("v6-prior checkpoint schedule changed")
    return total, checkpoints
