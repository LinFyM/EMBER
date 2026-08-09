"""Scientific and runtime authority for the v6-prior Expert-Manifold Writer."""

from __future__ import annotations

import math
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
            "blocked_until_single_a40_throughput_smoke",
            "ready_after_cpu_and_single_a40_throughput_smoke",
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
            "ready_after_live_gradient_profile",
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


def _throughput_profile_matches(
    evidence: Mapping[str, Any], *, minimum_batch_size: int
) -> bool:
    try:
        sizes = [
            int(value)
            for value in evidence["profiled_writer_model_batch_sizes"]
        ]
        selected = int(evidence["writer_model_batch_size"])
        measurements = [
            dict(value) for value in evidence["writer_generation_measurements"]
        ]
        rows = {int(value["batch_size"]): value for value in measurements}
    except (KeyError, TypeError, ValueError):
        return False
    if (
        len(sizes) < 3
        or sizes != sorted(set(sizes))
        or not {8, 16, 32}.issubset(sizes)
        or sizes[0] < minimum_batch_size
        or selected not in sizes
        or set(rows) != set(sizes)
        or len(rows) != len(measurements)
        or evidence.get("throughput_comparison_panel")
        != "same_fixed_longest_first_request_panel_all_candidates"
    ):
        return False
    try:
        stable_rows = []
        reference_entry_ids: list[str] | None = None
        reference_sampled: list[int] | None = None
        panel_size = max(sizes)
        for batch_size, row in rows.items():
            repeats = [float(value) for value in row["repeat_wall_seconds"]]
            sampled = [int(value) for value in row["sampled_frame_counts"]]
            entry_ids = [str(value) for value in row["entry_ids"]]
            forward_batches = [
                int(value) for value in row["forward_batch_sizes_per_repeat"]
            ]
            expected_forward_batches = [
                min(batch_size, panel_size - offset)
                for offset in range(0, panel_size, batch_size)
            ]
            wall = float(row["wall_seconds"])
            throughput = float(row["loras_per_second"])
            allocated = int(row["peak_allocated_bytes"])
            reserved = int(row["peak_reserved_bytes"])
            total = int(row["device_total_bytes"])
            headroom = int(row["memory_headroom_bytes"])
            required_headroom = int(row["required_memory_headroom_bytes"])
            structural = (
                bool(row.get("longest_video_included"))
                and isinstance(row.get("repeat_wall_seconds"), list)
                and len(repeats) >= 2
                and all(value > 0 and math.isfinite(value) for value in repeats)
                and int(row.get("generated_entries", -1))
                == panel_size * len(repeats)
                and int(row.get("max_observed_forward_batch_size", -1))
                == max(expected_forward_batches)
                and forward_batches == expected_forward_batches
                and row.get("comparison_panel_shared_across_candidates") is True
                and int(row.get("panel_entry_count", -1)) == panel_size
                and len(entry_ids) == panel_size
                and len(set(entry_ids)) == panel_size
                and len(sampled) == panel_size
                and all(value > 0 for value in sampled)
                and int(row.get("panel_total_sampled_frames", -1)) == sum(sampled)
                and int(row.get("max_sampled_video_frames", -1)) == max(sampled)
                and wall > 0
                and math.isfinite(wall)
                and math.isclose(
                    wall,
                    sum(repeats),
                    rel_tol=1e-9,
                    abs_tol=1e-6,
                )
                and throughput > 0
                and math.isfinite(throughput)
                and math.isclose(
                    throughput,
                    int(row["generated_entries"]) / wall,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                and allocated > 0
                and reserved >= allocated
                and total > reserved
                and headroom == total - reserved
                and required_headroom > 0
            )
            expected_stable = (
                max(repeats) / min(repeats) <= 1.25
                and headroom >= required_headroom
            )
            if not structural or bool(row.get("stable")) != expected_stable:
                return False
            if reference_entry_ids is None:
                reference_entry_ids = entry_ids
                reference_sampled = sampled
            elif entry_ids != reference_entry_ids or sampled != reference_sampled:
                return False
            if expected_stable:
                stable_rows.append(row)
        if not stable_rows or rows[selected] not in stable_rows:
            return False
        best = max(
            stable_rows,
            key=lambda row: (
                float(row["loras_per_second"]),
                int(row["batch_size"]),
            ),
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return selected == int(best["batch_size"])


def _evaluation_matches(value: Mapping[str, Any]) -> bool:
    try:
        minimum_batch_size = int(
            value.get("minimum_smoke_writer_model_batch_size", -1)
        )
    except (TypeError, ValueError):
        return False
    if (
        value.get("throughput_policy")
        != "highest_measured_throughput_with_device_memory_headroom"
        or minimum_batch_size != 8
    ):
        return False
    status = value.get("formal_status")
    evidence = value.get("online_smoke_evidence")
    if status == "blocked_until_live_a40_throughput_smoke":
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
        "batch_shape_bf16_roundoff_accepted": True,
        "writer_lora_storage": "template_native_mixed_bfloat16_float32",
        "throughput_selection_rule": (
            "highest_measured_fixed_panel_loras_per_second_with_stable_"
            "longest_video_batch"
        ),
        "throughput_comparison_panel": (
            "same_fixed_longest_first_request_panel_all_candidates"
        ),
        "success_interpretation": "execution_smoke_only_not_performance_evidence",
    }
    integers = {
        "validation_task_count": 8,
        "state_count": 1,
        "scientific_rows": 8,
        "generated_entries": 8,
        "cache_entries": 8,
        "writer_state_tensor_count": 600,
        "redundant_writer_forwards": 0,
        "writer_lora_tensor_bytes_per_entry": 2641920,
        "writer_lora_bfloat16_tensor_count": 72,
        "writer_lora_float32_tensor_count": 4,
        "generator_workers": 1,
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
            and int(evidence.get("writer_model_batch_size", -1))
            >= minimum_batch_size
            and int(evidence.get("max_peak_allocated_bytes", -1)) > 0
            and int(evidence.get("max_peak_reserved_bytes", -1))
            >= int(evidence.get("max_peak_allocated_bytes", -1))
            and int(evidence.get("max_post_release_allocated_bytes", -1)) >= 0
            and int(evidence.get("max_post_release_reserved_bytes", -1)) >= 0
            and _throughput_profile_matches(
                evidence, minimum_batch_size=minimum_batch_size
            )
        )
    except (TypeError, ValueError):
        return False


def assemble_v6_prior_evaluation_smoke_evidence(
    *,
    profile_root: Path,
    vertical_root: Path,
) -> dict[str, Any]:
    """Derive the evaluation seal only from retained live artifacts."""

    from ember.writer.evaluation_cache import (
        validate_writer_cache_manifest,
        writer_cache_requests,
    )
    from ember.writer.evaluation_runtime import WRITER_GENERATION_PROFILE_SCHEMA

    profile_root = profile_root.resolve()
    vertical_root = vertical_root.resolve()
    profile_contract = read_json(profile_root / "run_contract.json")
    profile = read_json(profile_root / "writer_generation_profile.json")
    vertical_contract = read_json(vertical_root / "run_contract.json")
    results = read_json(vertical_root / "results.json")
    manifest = validate_writer_cache_manifest(
        vertical_contract,
        verify_entry_files=False,
    )
    profile_adapter = profile_contract.get("adapter", {})
    vertical_adapter = vertical_contract.get("adapter", {})
    profile_git = profile_contract.get("git", {})
    vertical_git = vertical_contract.get("git", {})
    measurements = [
        dict(row) for row in profile.get("writer_generation_measurements", ())
    ]
    sizes = [
        int(value)
        for value in profile.get("profiled_writer_model_batch_sizes", ())
    ]
    if not sizes:
        raise ExpertManifoldError("v6-prior Writer profile has no batch candidates")
    selected = int(profile.get("selected_writer_model_batch_size", -1))
    throughput_evidence = {
        "profiled_writer_model_batch_sizes": sizes,
        "writer_model_batch_size": selected,
        "writer_generation_measurements": measurements,
        "throughput_comparison_panel": profile.get(
            "throughput_comparison_panel"
        ),
    }
    profile_tasks = profile_contract.get("tasks", ())
    vertical_tasks = vertical_contract.get("tasks", ())
    writer_generation = results.get("writer_generation", {})
    attempts = results.get("launcher_attempts", {}).get("attempts", ())
    workers = results.get("workers", ())
    storage = vertical_contract.get("writer_lora_cache", {}).get(
        "lora_storage_per_entry", {}
    )
    vertical_preflight = results.get("launcher", {}).get("preflight", {})
    profile_preflight = profile.get("preflight", {})
    profile_request_ids = {
        request.entry_id for request in writer_cache_requests(profile_contract)
    }
    try:
        ordered_measurements = sorted(
            measurements, key=lambda row: int(row["batch_size"])
        )
        warmup_runs = int(profile.get("warmup_runs_per_batch", -1))
        measured_runs = int(profile.get("measured_runs_per_batch", -1))
        longest_frames = int(profile.get("longest_sampled_video_frames", -1))
        panel_entry_ids = list(ordered_measurements[0].get("entry_ids", ()))
        panel_sampled = list(
            ordered_measurements[0].get("sampled_frame_counts", ())
        )
        profile_shape_evidence = (
            warmup_runs >= 1
            and measured_runs >= 2
            and longest_frames > 0
            and len(panel_entry_ids) == max(sizes)
            and len(panel_sampled) == max(sizes)
            and all(
                len(row.get("repeat_wall_seconds", ())) == measured_runs
                and int(row.get("max_sampled_video_frames", -1))
                == longest_frames
                and set(row.get("entry_ids", ())).issubset(profile_request_ids)
                and list(row.get("entry_ids", ())) == panel_entry_ids
                and list(row.get("sampled_frame_counts", ())) == panel_sampled
                for row in ordered_measurements
            )
        )
    except (IndexError, KeyError, TypeError, ValueError):
        profile_shape_evidence = False
    valid = (
        profile.get("schema_version") == WRITER_GENERATION_PROFILE_SCHEMA
        and profile.get("contract_reference")
        == profile_contract.get("contract_reference")
        and profile.get("root") == str(profile_root)
        and profile.get("device") == "NVIDIA A40"
        and profile.get("git") == profile_git
        and not profile_git.get("dirty_paths")
        and profile_git.get("commit") == profile_git.get("upstream_commit")
        and profile_git == vertical_git
        and profile_adapter == vertical_adapter
        and profile_adapter.get("kind") == "expert_manifold_writer"
        and profile_adapter.get("video_condition") == "correct"
        and profile_adapter.get("video_schedule", {}).get("sampling_mode")
        == "without_replacement"
        and profile_contract.get("mode") == "smoke"
        and profile_contract.get("role") == "validation"
        and len(profile_tasks) == 8
        and sum(len(row.get("init_state_ids", ())) for row in profile_tasks)
        >= max(sizes)
        and int(profile_contract.get("parallel", {}).get("physical_gpu_count", -1))
        == 1
        and int(profile_contract.get("parallel", {}).get("replicas_per_gpu", -1))
        == 1
        and int(
            profile_contract.get("parallel", {}).get(
                "writer_generators_per_gpu", -1
            )
        )
        == 1
        and int(
            profile_contract.get("parallel", {}).get(
                "writer_generation_batch_size", -1
            )
        )
        == max(sizes)
        and profile.get("selection_rule")
        == (
            "highest_measured_fixed_panel_loras_per_second_with_stable_"
            "longest_video_batch"
        )
        and profile.get("throughput_comparison_panel")
        == "same_fixed_longest_first_request_panel_all_candidates"
        and profile.get("writer_modules_released") is True
        and profile.get("source_policy_reused") is True
        and int(profile.get("oom_count", -1)) == 0
        and int(profile.get("nonfinite_count", -1)) == 0
        and profile_preflight.get("compute_applications") == []
        and profile_preflight.get("device_names") == ["NVIDIA A40"]
        and profile_preflight.get("physical_gpu_ids")
        == profile_contract.get("parallel", {}).get("physical_gpu_ids")
        and profile_shape_evidence
        and _throughput_profile_matches(
            throughput_evidence,
            minimum_batch_size=8,
        )
        and vertical_contract.get("contract_reference")
        == results.get("contract_reference")
        and vertical_contract.get("mode") == "smoke"
        and vertical_contract.get("role") == "validation"
        and len(vertical_tasks) == 8
        and all(len(row.get("init_state_ids", ())) == 1 for row in vertical_tasks)
        and int(
            vertical_contract.get("parallel", {}).get(
                "writer_generation_batch_size", -1
            )
        )
        == selected
        and results.get("overall", {}).get("episodes") == 8
        and len(results.get("rows", ())) == 8
        and len(workers) == 1
        and workers[0].get("gpu_name") == "NVIDIA A40"
        and workers[0].get("source_policy_reloaded") is False
        and len(attempts) == 1
        and attempts[0].get("event") == "completed"
        and results.get("launcher", {}).get("return_codes")
        == {str(workers[0]["worker_id"]): 0}
        and vertical_preflight.get("compute_applications") == []
        and vertical_preflight.get("device_names") == ["NVIDIA A40"]
        and vertical_preflight.get("physical_gpu_ids")
        == vertical_contract.get("parallel", {}).get("physical_gpu_ids")
        and int(writer_generation.get("generator_workers", -1)) == 1
        and int(writer_generation.get("assigned_entries", -1)) == 8
        and int(writer_generation.get("generated_entries", -1)) == 8
        and int(writer_generation.get("reused_entries", -1)) == 0
        and int(writer_generation.get("max_observed_forward_batch_size", -1)) == 8
        and writer_generation.get("generation_batch_size") == [selected]
        and int(writer_generation.get("redundant_writer_forwards", -1)) == 0
        and writer_generation.get("batch_shape_bf16_roundoff_accepted") is True
        and writer_generation.get("all_source_policy_processes_reused_for_rollout")
        is True
        and writer_generation.get("all_writer_modules_released") is True
        and writer_generation.get("all_source_policies_not_reloaded") is True
        and writer_generation.get("gpu_names") == ["NVIDIA A40"]
        and len(manifest.get("entry_ids", ())) == 8
        and manifest.get("descriptor") == vertical_contract.get("writer_lora_cache")
        and int(storage.get("tensor_count", -1)) == 76
        and int(storage.get("tensor_bytes", -1)) == 2_641_920
        and storage.get("dtype_tensor_counts") == {"BF16": 72, "F32": 4}
        and len(storage.get("dtype_by_name", {})) == 76
    )
    if not valid:
        raise ExpertManifoldError("v6-prior live evaluation evidence is incomplete")
    evidence = {
        "commit": str(profile_git["commit"]),
        "root": str(vertical_root),
        "device": "NVIDIA A40",
        "checkpoint_kind": str(vertical_adapter["writer_asset"]["kind"]),
        "video_condition": "correct",
        "video_sampling": "without_replacement",
        "validation_task_count": 8,
        "state_count": 1,
        "scientific_rows": 8,
        "generated_entries": 8,
        "cache_entries": 8,
        "writer_state_tensor_count": int(
            vertical_adapter["writer_asset"]["writer_state"]["state_tensor_count"]
        ),
        "writer_model_batch_size": selected,
        "profiled_writer_model_batch_sizes": sizes,
        "writer_generation_measurements": measurements,
        "writer_modules_released": True,
        "source_policy_reused_for_rollout": True,
        "source_policy_reloaded": False,
        "batch_shape_bf16_roundoff_accepted": True,
        "redundant_writer_forwards": 0,
        "writer_lora_storage": "template_native_mixed_bfloat16_float32",
        "writer_lora_tensor_bytes_per_entry": int(storage["tensor_bytes"]),
        "writer_lora_bfloat16_tensor_count": int(
            storage["dtype_tensor_counts"]["BF16"]
        ),
        "writer_lora_float32_tensor_count": int(
            storage["dtype_tensor_counts"]["F32"]
        ),
        "generator_workers": 1,
        "max_peak_allocated_bytes": max(
            int(writer_generation["max_peak_allocated_bytes"]),
            max(int(row["peak_allocated_bytes"]) for row in measurements),
        ),
        "max_peak_reserved_bytes": max(
            int(writer_generation["max_peak_reserved_bytes"]),
            max(int(row["peak_reserved_bytes"]) for row in measurements),
        ),
        "max_post_release_allocated_bytes": int(
            writer_generation["max_post_release_allocated_bytes"]
        ),
        "max_post_release_reserved_bytes": int(
            writer_generation["max_post_release_reserved_bytes"]
        ),
        "throughput_selection_rule": (
            "highest_measured_fixed_panel_loras_per_second_with_stable_"
            "longest_video_batch"
        ),
        "throughput_comparison_panel": (
            "same_fixed_longest_first_request_panel_all_candidates"
        ),
        "retry_count": 0,
        "failure_count": 0,
        "teacher_action_reads": 0,
        "teacher_state_reads": 0,
        "reward_reads": 0,
        "terminal_reads": 0,
        "oom_count": 0,
        "nonfinite_count": 0,
        "success_interpretation": "execution_smoke_only_not_performance_evidence",
    }
    if not _evaluation_matches(
        {
            "throughput_policy": (
                "highest_measured_throughput_with_device_memory_headroom"
            ),
            "minimum_smoke_writer_model_batch_size": 8,
            "formal_status": "sealed",
            "online_smoke_evidence": evidence,
        }
    ):
        raise ExpertManifoldError("assembled v6-prior evaluation seal is invalid")
    return evidence


def _state_machine_matches(config: Mapping[str, Any]) -> bool:
    state = (
        config.get("evaluation", {}).get("formal_status"),
        config.get("gradient_profile", {}).get("status"),
        config.get("objective", {})
        .get("auxiliary_weights", {})
        .get("status"),
        config.get("profile_run", {}).get("status"),
        config.get("formal_run", {}).get("status"),
    )
    return state in {
        (
            "blocked_until_live_a40_throughput_smoke",
            "blocked_until_single_a40_throughput_smoke",
            "blocked_until_live_train24_gradient_profile",
            "blocked_until_live_gradient_weights",
            "blocked_until_live_a40_profile_and_macro3_online_smoke",
        ),
        (
            "sealed",
            "ready_after_cpu_and_single_a40_throughput_smoke",
            "blocked_until_live_train24_gradient_profile",
            "blocked_until_live_gradient_weights",
            "blocked_until_live_a40_profile_and_macro3_online_smoke",
        ),
        (
            "sealed",
            "sealed_from_live_train24_gradient_profile",
            "sealed_from_live_train24_gradient_profile",
            "ready_after_live_gradient_profile",
            "blocked_until_live_a40_profile_and_macro3_online_smoke",
        ),
        (
            "sealed",
            "sealed_from_live_train24_gradient_profile",
            "sealed_from_live_train24_gradient_profile",
            "sealed_from_live_gradient_profile_and_a40_resume_smoke",
            "sealed_from_live_a40_profile_and_macro3_online_smoke",
        ),
    }


def load_v6_prior_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    authorities = config.get("authorities", {})
    initialization = config.get("initialization", {})
    basis = config.get("expert_basis", {})
    valid = (
        config.get("schema_version") == V6_PRIOR_CONFIG_SCHEMA
        and set(authorities)
        == {
            "task_expert_config",
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
        and _state_machine_matches(config)
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
            != "ready_after_cpu_and_single_a40_throughput_smoke"
            or config["evaluation"].get("formal_status") != "sealed"
            or config["objective"]["auxiliary_weights"]["status"]
            != "blocked_until_live_train24_gradient_profile"
        ):
            raise ExpertManifoldError("v6-prior gradient profile is not ready")
        return int(profile["macros"]), ()
    selected = config["profile_run" if mode == "profile" else "formal_run"]
    if mode == "profile":
        ready = (
            selected.get("status") == "ready_after_live_gradient_profile"
            and config["gradient_profile"].get("status")
            == "sealed_from_live_train24_gradient_profile"
            and config["objective"]["auxiliary_weights"]["status"]
            == "sealed_from_live_train24_gradient_profile"
        )
    else:
        ready = (
            selected.get("status")
            == "sealed_from_live_a40_profile_and_macro3_online_smoke"
            and config["profile_run"].get("status")
            == "sealed_from_live_gradient_profile_and_a40_resume_smoke"
            and config["gradient_profile"].get("status")
            == "sealed_from_live_train24_gradient_profile"
            and config["objective"]["auxiliary_weights"]["status"]
            == "sealed_from_live_train24_gradient_profile"
        )
    if not ready:
        raise ExpertManifoldError(f"v6-prior {mode} runtime is not sealed")
    total = int(selected["total_macros"])
    checkpoints = tuple(int(value) for value in selected["checkpoint_macros"])
    if not total > 0 or not checkpoints or checkpoints[-1] != total:
        raise ExpertManifoldError("v6-prior checkpoint schedule changed")
    return total, checkpoints
