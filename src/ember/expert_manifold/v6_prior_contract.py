"""Scientific and runtime authority for the v6-prior Expert-Manifold Writer."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.expert_manifold.contract import ExpertManifoldError
from ember.pi05_source_checkpoint import read_json
from ember.writer.architecture import validate_writer_dimensions


REPO_ROOT = Path(__file__).resolve().parents[3]
V6_PRIOR_CONFIG_SCHEMA = "ember_pi05_v6_prior_policy_effective_writer_v1"
V6_PRIOR_MODES = ("gradient-profile", "profile", "formal")
V6_PRIOR_GRADIENT_PROFILE_SCHEMA = "ember_pi05_v6_prior_gradient_profile_seal_v1"
V6_PRIOR_GRADIENT_EVIDENCE_SCHEMA = (
    "ember_pi05_v6_prior_gradient_profile_artifact_evidence_v1"
)
V6_PRIOR_RESUME_EVIDENCE_SCHEMA = (
    "ember_pi05_v6_prior_resume_profile_artifact_evidence_v1"
)
V6_PRIOR_COMPLETION_SCHEMA = "ember_pi05_v6_prior_writer_completion_v1"


def suggest_auxiliary_weight(
    positive: Mapping[str, float],
    auxiliary: Mapping[str, float],
    *,
    maximum_fraction: float,
) -> float:
    """Seal one auxiliary against both trainable-group positive norms."""

    if not 0 < maximum_fraction <= 1:
        raise ExpertManifoldError("invalid v6-prior gradient fraction")
    constraints = []
    for group in ("compiler", "factor_heads"):
        left = float(positive[group])
        right = float(auxiliary[group])
        if not math.isfinite(left) or not math.isfinite(right) or right < 0:
            return 0.0
        if right == 0:
            continue
        if left <= 0:
            return 0.0
        constraints.append(maximum_fraction * left / right)
    return 0.0 if not constraints else min(1.0, *constraints)


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
        and value.get("teacher_video_schedule") == "deterministic_no_replacement_cycles"
        and value.get("teacher_action_episode_overlap") is False
        and value.get("task_aggregation") == "mean_within_task_then_train24_equal_mean"
        and all(
            isinstance(value.get(name), int) and int(value[name]) >= 0
            for name in (
                "sampler_seed",
                "teacher_video_seed",
                "counterfactual_seed",
            )
        )
        and value.get("negative_schedule")
        == ("task_ordinal_plus_task_visit_modulo_reversed_shuffled_wrong")
        and value.get("wrong_video_schedule")
        == "deterministic_cross_suite_cycle_with_current_task_language"
    )


def _objective_matches(
    value: Mapping[str, Any],
    *,
    gradient_evidence: Mapping[str, Any] | None,
) -> bool:
    expert = value.get("expert", {})
    ranking = value.get("ranking", {})
    weights = value.get("auxiliary_weights", {})
    status = weights.get("status")
    coefficients = (weights.get("expert"), weights.get("ranking"))
    if status == "blocked_until_live_train24_gradient_profile":
        valid_weights = coefficients == (None, None) and gradient_evidence is None
    elif status == "sealed_from_live_train24_gradient_profile":
        valid_weights = (
            isinstance(gradient_evidence, Mapping)
            and _gradient_profile_evidence_matches(gradient_evidence)
            and coefficients
            == (
                gradient_evidence["recommended_weights"]["expert"],
                gradient_evidence["recommended_weights"]["ranking"],
            )
        )
    else:
        valid_weights = False
    return (
        float(value.get("positive_functional_weight", -1)) == 1.0
        and value.get("positive_policy_randomness")
        == "one_independent_flow_noise_and_time_per_action_query"
        and expert.get("direction") == "one_minus_global_effective_ba_cosine"
        and expert.get("norm") == "smooth_l1_global_effective_log_norm_ratio"
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
        and (
            _gradient_profile_evidence_matches(gradient.get("artifact_evidence", {}))
            if gradient.get("status") == "sealed_from_live_train24_gradient_profile"
            else gradient.get("artifact_evidence") is None
        )
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
            "sealed_from_live_a40_resume_profile_evidence",
        }
        and int(profile.get("expected_world_size", -1)) == 6
        and int(profile.get("tasks_per_rank", -1)) == 4
        and int(profile.get("total_macros", -1)) == 3
        and profile.get("checkpoint_macros") == [1, 3]
        and profile.get("required_resume_comparison")
        == "fresh0_to1_plus_exact_resume1_to3_equals_contiguous0_to3"
        and (
            _resume_profile_evidence_matches(profile.get("artifact_evidence", {}))
            if profile.get("status") == "sealed_from_live_a40_resume_profile_evidence"
            else profile.get("artifact_evidence") is None
        )
        and formal.get("status")
        in {
            "blocked_until_live_a40_resume_profile_evidence",
            "sealed_from_live_a40_resume_profile_evidence",
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
        sizes = [int(value) for value in evidence["profiled_writer_model_batch_sizes"]]
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
                and int(row.get("generated_entries", -1)) == panel_size * len(repeats)
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
                max(repeats) / min(repeats) <= 1.25 and headroom >= required_headroom
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
        minimum_batch_size = int(value.get("minimum_smoke_writer_model_batch_size", -1))
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
            and int(evidence.get("writer_model_batch_size", -1)) >= minimum_batch_size
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
        int(value) for value in profile.get("profiled_writer_model_batch_sizes", ())
    ]
    if not sizes:
        raise ExpertManifoldError("v6-prior Writer profile has no batch candidates")
    selected = int(profile.get("selected_writer_model_batch_size", -1))
    throughput_evidence = {
        "profiled_writer_model_batch_sizes": sizes,
        "writer_model_batch_size": selected,
        "writer_generation_measurements": measurements,
        "throughput_comparison_panel": profile.get("throughput_comparison_panel"),
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
        panel_sampled = list(ordered_measurements[0].get("sampled_frame_counts", ()))
        profile_shape_evidence = (
            warmup_runs >= 1
            and measured_runs >= 2
            and longest_frames > 0
            and len(panel_entry_ids) == max(sizes)
            and len(panel_sampled) == max(sizes)
            and all(
                len(row.get("repeat_wall_seconds", ())) == measured_runs
                and int(row.get("max_sampled_video_frames", -1)) == longest_frames
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
        and int(profile_contract.get("parallel", {}).get("physical_gpu_count", -1)) == 1
        and int(profile_contract.get("parallel", {}).get("replicas_per_gpu", -1)) == 1
        and int(
            profile_contract.get("parallel", {}).get("writer_generators_per_gpu", -1)
        )
        == 1
        and int(
            profile_contract.get("parallel", {}).get("writer_generation_batch_size", -1)
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
        "writer_lora_float32_tensor_count": int(storage["dtype_tensor_counts"]["F32"]),
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ExpertManifoldError(f"missing v6-prior artifact: {path}")
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ExpertManifoldError(f"invalid v6-prior JSONL artifact: {path}") from error
    if not all(isinstance(row, dict) for row in rows):
        raise ExpertManifoldError(f"invalid v6-prior JSONL rows: {path}")
    return rows


def _clean_pushed_git(value: Mapping[str, Any]) -> bool:
    required = {
        "branch",
        "commit",
        "origin_main",
        "upstream",
        "upstream_commit",
        "dirty_paths",
    }
    return (
        set(value) == required
        and all(
            isinstance(value.get(name), str) and bool(value[name])
            for name in required - {"dirty_paths"}
        )
        and value.get("dirty_paths") == []
        and value.get("commit") == value.get("upstream_commit")
    )


def _rank_topology_matches(
    rows: Sequence[Mapping[str, Any]], *, world_size: int
) -> bool:
    if len(rows) != world_size:
        return False
    seen_devices: set[tuple[str, int]] = set()
    seen_local_ranks: set[tuple[str, int]] = set()
    affinities: dict[tuple[str, int], tuple[int, ...]] = {}
    try:
        for expected_rank, row in enumerate(
            sorted(rows, key=lambda item: int(item["rank"]))
        ):
            rank = int(row["rank"])
            local_rank = int(row["local_rank"])
            host = str(row["host"])
            physical = int(row["physical_gpu"])
            numa = int(row["numa_node"])
            visible = tuple(
                int(item.strip())
                for item in str(row["cuda_visible_devices"]).split(",")
                if item.strip()
            )
            affinity = tuple(int(item) for item in row["cpu_affinity"])
            if (
                rank != expected_rank
                or local_rank < 0
                or not host
                or not visible
                or local_rank >= len(visible)
                or visible[local_rank] != physical
                or numa < 0
                or not affinity
                or affinity != tuple(sorted(set(affinity)))
                or row.get("device_name") != "NVIDIA A40"
                or str(row.get("device", "")) != f"cuda:{local_rank}"
                or (host, physical) in seen_devices
                or (host, local_rank) in seen_local_ranks
            ):
                return False
            seen_devices.add((host, physical))
            seen_local_ranks.add((host, local_rank))
            key = (host, numa)
            if key in affinities and affinities[key] != affinity:
                return False
            affinities[key] = affinity
        for (left_host, left_numa), left in affinities.items():
            for (right_host, right_numa), right in affinities.items():
                if (
                    left_host == right_host
                    and left_numa != right_numa
                    and set(left).intersection(right)
                ):
                    return False
    except (KeyError, TypeError, ValueError):
        return False
    return len(seen_devices) == world_size


def _gradient_norms_match(value: Mapping[str, Any]) -> bool:
    if set(value) != {"positive", "expert", "ranking"}:
        return False
    try:
        for name, row in value.items():
            if set(row) != {"compiler", "factor_heads", "global"}:
                return False
            compiler = float(row["compiler"])
            factor = float(row["factor_heads"])
            global_norm = float(row["global"])
            if (
                not all(math.isfinite(item) for item in (compiler, factor, global_norm))
                or min(compiler, factor, global_norm) < 0
                or (name == "positive" and min(compiler, factor) <= 0)
                or not math.isclose(
                    global_norm * global_norm,
                    compiler * compiler + factor * factor,
                    rel_tol=1e-4,
                    abs_tol=1e-6,
                )
            ):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _recommended_gradient_weights(
    norms: Mapping[str, Mapping[str, float]], *, fraction: float
) -> dict[str, float]:
    return {
        name: suggest_auxiliary_weight(
            norms["positive"], norms[name], maximum_fraction=fraction
        )
        for name in ("expert", "ranking")
    }


def _applied_gradient_fractions(
    norms: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    return {
        name: {
            group: (
                float(weights[name])
                * float(norms[name][group])
                / float(norms["positive"][group])
            )
            for group in ("compiler", "factor_heads")
        }
        for name in ("expert", "ranking")
    }


def _gradient_profile_evidence_matches(value: Mapping[str, Any]) -> bool:
    try:
        norms = value["unweighted_gradient_norms"]
        fraction = float(value["maximum_auxiliary_fraction"])
        weights = {
            name: float(value["recommended_weights"][name])
            for name in ("expert", "ranking")
        }
        recomputed = _recommended_gradient_weights(norms, fraction=fraction)
        applied = _applied_gradient_fractions(norms, weights)
        numeric = (
            float(value["step_seconds"]),
            int(value["max_cuda_allocated_bytes"]),
            int(value["max_cuda_reserved_bytes"]),
        )
        valid = (
            value.get("schema_version") == V6_PRIOR_GRADIENT_EVIDENCE_SCHEMA
            and isinstance(value.get("root"), str)
            and bool(value["root"])
            and _clean_pushed_git(value["git"])
            and value.get("config_schema") == V6_PRIOR_CONFIG_SCHEMA
            and int(value.get("config_bytes", -1)) > 0
            and int(value.get("world_size", -1)) == 6
            and int(value.get("tasks_per_rank", -1)) == 4
            and _rank_topology_matches(value["rank_topology"], world_size=6)
            and int(value.get("schedule_start_macro", -1)) == 49
            and int(value.get("schedule_stop_macro", -1)) == 50
            and int(value.get("completed_diagnostic_macros", -1)) == 1
            and int(value.get("task_count", -1)) == 24
            and int(value.get("action_queries_per_task", -1)) == 20
            and int(value.get("total_action_queries", -1)) == 480
            and int(value.get("unique_action_queries", -1)) == 480
            and value.get("counterfactual_counts")
            == {"reversed": 8, "shuffled": 8, "wrong": 8}
            and int(value.get("longest_correct_sampled_frames", -1)) == 105
            and _gradient_norms_match(norms)
            and math.isclose(fraction, 0.25, rel_tol=0.0, abs_tol=0.0)
            and all(
                math.isclose(
                    weights[name], recomputed[name], rel_tol=1e-12, abs_tol=1e-12
                )
                for name in weights
            )
            and value.get("applied_gradient_fractions") == applied
            and all(
                0 <= float(applied[name][group]) <= fraction + 1e-9
                for name in applied
                for group in applied[name]
            )
            and value.get("seal_rule")
            == (
                "each_auxiliary_at_most_one_quarter_positive_gradient_in_both_"
                "compiler_and_factor_heads"
            )
            and value.get("initialization")
            == {
                "mode": "historical_v6_macro400_load_only",
                "optimizer": "fresh",
                "scheduler": "fresh",
                "rng": "fresh_seed",
            }
            and value.get("expert_bank") == {"step": 2000, "task_count": 24}
            and value.get("ownership")
            == {
                "frozen_parameter_count": 7_060_992,
                "trainable_parameter_count": 3_714_304,
                "trainable_tensor_count": 41,
                "source_policy_trainable_parameter_count": 0,
            }
            and value.get("method_verified") is True
            and value.get("information_wall_verified") is True
            and int(value.get("invocation_count", -1)) == 1
            and numeric[0] > 0
            and math.isfinite(numeric[0])
            and numeric[1] > 0
            and numeric[2] >= numeric[1]
            and int(value.get("oom_count", -1)) == 0
            and int(value.get("nonfinite_count", -1)) == 0
            and value.get("content_hash_policy") == "disabled_by_owner"
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return valid


def _task_records_match_contract(
    records: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    *,
    task_visit: int,
) -> bool:
    if len(records) != 24 or len(tasks) != 24:
        return False
    expected = {int(row["ordinal"]): row for row in tasks}
    metric_names = {
        "functional_loss",
        "expert_loss",
        "expert_direction",
        "expert_log_norm",
        "ranking_loss",
        "ranking_margin",
        "correct_expert_cosine",
        "counterfactual_expert_cosine",
        "correct_effective_norm",
        "counterfactual_effective_norm",
        "expert_effective_norm",
    }
    try:
        for ordinal, row in enumerate(
            sorted(records, key=lambda item: int(item["task_ordinal"]))
        ):
            task = expected[ordinal]
            if (
                int(row["task_ordinal"]) != ordinal
                or int(row["global_task_id"]) != int(task["global_task_id"])
                or row["suite"] != task["suite"]
                or int(row["task_id"]) != int(task["task_id"])
                or int(row["task_visit"]) != task_visit
                or int(row["teacher_demo"]) not in range(50)
                or row["counterfactual_kind"] not in {"reversed", "shuffled", "wrong"}
                or int(row["correct_raw_frames"]) <= 0
                or int(row["correct_sampled_frames"]) <= 0
                or int(row["counterfactual_raw_frames"]) <= 0
                or int(row["counterfactual_sampled_frames"]) <= 0
                or not all(math.isfinite(float(row[name])) for name in metric_names)
            ):
                return False
            is_wrong = row["counterfactual_kind"] == "wrong"
            if is_wrong != (row["counterfactual_global_task_id"] is not None):
                return False
            if is_wrong != (row["counterfactual_demo"] is not None):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return set(expected) == set(range(24))


def assemble_v6_prior_gradient_profile_evidence(
    profile_root: Path,
) -> dict[str, Any]:
    """Derive the gradient seal from one complete retained six-rank artifact."""

    profile_root = profile_root.resolve()
    for name in (
        "run_contract.json",
        "gradient_profile.json",
        "completion.json",
        "invocations.jsonl",
    ):
        if not (profile_root / name).is_file():
            raise ExpertManifoldError(f"missing v6-prior gradient artifact: {name}")
    contract = read_json(profile_root / "run_contract.json")
    profile = read_json(profile_root / "gradient_profile.json")
    completion = read_json(profile_root / "completion.json")
    invocations = _read_jsonl(profile_root / "invocations.jsonl")
    runtime = contract.get("runtime", {})
    data = contract.get("data", {})
    consumed = data.get("consumed_schedule", {})
    query = consumed.get("query", {})
    records = profile.get("task_records", ())
    norms = profile.get("unweighted_gradient_norms", {})
    fraction = float(profile.get("maximum_auxiliary_fraction", -1))
    weights = profile.get("recommended_weights", {})
    expected_weights = (
        _recommended_gradient_weights(norms, fraction=fraction)
        if _gradient_norms_match(norms) and 0 < fraction <= 1
        else {}
    )
    try:
        valid = (
            contract.get("schema_version") == "ember_pi05_v6_prior_writer_launch_v1"
            and contract.get("mode") == "gradient-profile"
            and _clean_pushed_git(contract.get("git", {}))
            and contract.get("config", {}).get("schema") == V6_PRIOR_CONFIG_SCHEMA
            and int(contract.get("config", {}).get("bytes", -1)) > 0
            and _method_matches(contract.get("method", {}))
            and _information_wall_matches(contract.get("information_wall", {}))
            and _writer_matches(contract.get("writer", {}))
            and _data_matches(data)
            and _objective_matches(
                contract.get("objective", {}), gradient_evidence=None
            )
            and _optimization_matches(contract.get("optimization", {}))
            and contract.get("initialization", {}).get("mode")
            == "historical_v6_macro400_load_only"
            and contract.get("initialization", {}).get("optimizer") == "fresh"
            and contract.get("initialization", {}).get("scheduler") == "fresh"
            and contract.get("initialization", {}).get("rng") == "fresh_seed"
            and int(
                contract.get("initialization", {}).get("writer_state_tensor_count", -1)
            )
            == 600
            and int(contract.get("expert_bank", {}).get("step", -1)) == 2000
            and len(contract.get("expert_bank", {}).get("tasks", ())) == 24
            and contract.get("expert_bank", {}).get("deployment_read") is False
            and contract.get("expert_basis", {}).get("task_count") == 24
            and contract.get("ownership", {}).get("frozen_parameter_count") == 7_060_992
            and contract.get("ownership", {}).get("trainable_parameter_count")
            == 3_714_304
            and contract.get("ownership", {}).get("trainable_tensor_count") == 41
            and contract.get("ownership", {}).get(
                "source_policy_trainable_parameter_count"
            )
            == 0
            and int(runtime.get("world_size", -1)) == 6
            and int(runtime.get("tasks_per_rank", -1)) == 4
            and int(runtime.get("total_macros", -1)) == 1
            and int(runtime.get("gradient_profile_schedule_macro", -1)) == 49
            and runtime.get("checkpoint_macros") == []
            and runtime.get("distributed_model_wrapper") == "none"
            and runtime.get("gradient_reduction")
            == "single_flat_parameter_ordered_allreduce_mean_after_local_task_mean"
            and runtime.get("deferred_process_group") is True
            and runtime.get("nccl_p2p_disable") == "1"
            and runtime.get("nccl_algo") == "Ring"
            and runtime.get("nccl_proto") == "Simple"
            and _rank_topology_matches(runtime.get("rank_topology", ()), world_size=6)
            and query
            == {
                "start_step": 49,
                "stop_step": 50,
                "global_examples": 480,
                "unique_query_rows": 480,
                "min_examples_per_task": 20,
                "max_examples_per_task": 20,
                "identity_evidence": "cursor_counts_and_dataset_row_coverage",
            }
            and int(consumed.get("videos_per_task_visit", -1)) == 1
            and int(consumed.get("min_video_visits_per_task", -1)) == 1
            and int(consumed.get("max_video_visits_per_task", -1)) == 1
            and int(consumed.get("min_unique_videos_per_task", -1)) == 1
            and int(consumed.get("max_unique_videos_per_task", -1)) == 1
            and profile.get("schema_version") == V6_PRIOR_GRADIENT_PROFILE_SCHEMA
            and int(profile.get("schedule_macro", -1)) == 49
            and int(profile.get("task_count", -1)) == 24
            and int(profile.get("action_queries_per_task", -1)) == 20
            and int(profile.get("total_action_queries", -1)) == 480
            and int(profile.get("unique_action_queries", -1)) == 480
            and profile.get("counterfactual_counts")
            == {"reversed": 8, "shuffled": 8, "wrong": 8}
            and _task_records_match_contract(
                records, data.get("tasks", ()), task_visit=49
            )
            and max(int(row["correct_sampled_frames"]) for row in records) == 105
            and _gradient_norms_match(norms)
            and math.isclose(fraction, 0.25, rel_tol=0.0, abs_tol=0.0)
            and set(weights) == {"expert", "ranking"}
            and all(
                math.isclose(
                    float(weights[name]),
                    expected_weights[name],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for name in expected_weights
            )
            and profile.get("seal_rule")
            == (
                "each_auxiliary_at_most_one_quarter_positive_gradient_in_both_"
                "compiler_and_factor_heads"
            )
            and float(profile.get("step_seconds", -1)) > 0
            and math.isfinite(float(profile.get("step_seconds", -1)))
            and int(profile.get("max_cuda_allocated_bytes", -1)) > 0
            and int(profile.get("max_cuda_reserved_bytes", -1))
            >= int(profile.get("max_cuda_allocated_bytes", -1))
            and int(profile.get("oom_count", -1)) == 0
            and int(profile.get("nonfinite_count", -1)) == 0
            and profile.get("content_hash_policy") == "disabled_by_owner"
            and completion
            == {
                "schema_version": V6_PRIOR_COMPLETION_SCHEMA,
                "mode": "gradient-profile",
                "completed_diagnostic_macros": 1,
                "schedule_start_macro": 49,
                "schedule_stop_macro": 50,
                "gradient_profile_complete": True,
                "oom_count": 0,
                "nonfinite_count": 0,
                "content_hash_policy": "disabled_by_owner",
            }
            and len(invocations) == 1
            and invocations[0].get("resume") is None
            and int(invocations[0].get("requested_stop_after_macro", -1)) == 1
            and isinstance(invocations[0].get("argv"), list)
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ExpertManifoldError("v6-prior gradient-profile evidence is incomplete")
    applied = _applied_gradient_fractions(norms, weights)
    evidence = {
        "schema_version": V6_PRIOR_GRADIENT_EVIDENCE_SCHEMA,
        "root": str(profile_root),
        "git": dict(contract["git"]),
        "config_schema": str(contract["config"]["schema"]),
        "config_bytes": int(contract["config"]["bytes"]),
        "world_size": 6,
        "tasks_per_rank": 4,
        "rank_topology": [dict(row) for row in runtime["rank_topology"]],
        "schedule_start_macro": 49,
        "schedule_stop_macro": 50,
        "completed_diagnostic_macros": 1,
        "task_count": 24,
        "action_queries_per_task": 20,
        "total_action_queries": 480,
        "unique_action_queries": 480,
        "counterfactual_counts": {"reversed": 8, "shuffled": 8, "wrong": 8},
        "longest_correct_sampled_frames": 105,
        "unweighted_gradient_norms": norms,
        "maximum_auxiliary_fraction": fraction,
        "recommended_weights": {
            name: float(weights[name]) for name in ("expert", "ranking")
        },
        "applied_gradient_fractions": applied,
        "seal_rule": str(profile["seal_rule"]),
        "initialization": {
            name: contract["initialization"][name]
            for name in ("mode", "optimizer", "scheduler", "rng")
        },
        "expert_bank": {"step": 2000, "task_count": 24},
        "ownership": {
            name: contract["ownership"][name]
            for name in (
                "frozen_parameter_count",
                "trainable_parameter_count",
                "trainable_tensor_count",
                "source_policy_trainable_parameter_count",
            )
        },
        "method_verified": True,
        "information_wall_verified": True,
        "invocation_count": 1,
        "step_seconds": float(profile["step_seconds"]),
        "max_cuda_allocated_bytes": int(profile["max_cuda_allocated_bytes"]),
        "max_cuda_reserved_bytes": int(profile["max_cuda_reserved_bytes"]),
        "oom_count": 0,
        "nonfinite_count": 0,
        "content_hash_policy": "disabled_by_owner",
    }
    if not _gradient_profile_evidence_matches(evidence):
        raise ExpertManifoldError("assembled v6-prior gradient seal is invalid")
    return evidence


def _metric_rows_match_contract(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    expert_weight: float,
    ranking_weight: float,
) -> bool:
    if len(rows) != 3:
        return False
    metric_names = {
        "functional_loss",
        "expert_loss",
        "expert_direction",
        "expert_log_norm",
        "ranking_loss",
        "ranking_margin",
        "correct_expert_cosine",
        "counterfactual_expert_cosine",
        "correct_effective_norm",
        "counterfactual_effective_norm",
        "expert_effective_norm",
        "gradient_norm_before_clip",
        "applied_lr",
        "next_lr",
    }
    try:
        for macro, row in enumerate(rows, start=1):
            if (
                int(row["macro"]) != macro
                or not math.isclose(
                    float(row["expert_weight"]), expert_weight, rel_tol=0.0, abs_tol=0.0
                )
                or not math.isclose(
                    float(row["ranking_weight"]),
                    ranking_weight,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                or row["counterfactual_counts"]
                != {"reversed": 8, "shuffled": 8, "wrong": 8}
                or not _task_records_match_contract(
                    row["task_records"],
                    contract["data"]["tasks"],
                    task_visit=macro - 1,
                )
                or not all(math.isfinite(float(row[name])) for name in metric_names)
                or float(row["step_seconds"]) <= 0
                or not math.isfinite(float(row["step_seconds"]))
                or float(row["elapsed_seconds"]) <= 0
                or not math.isfinite(float(row["elapsed_seconds"]))
                or int(row["max_cuda_allocated_bytes"]) <= 0
                or int(row["max_cuda_reserved_bytes"])
                < int(row["max_cuda_allocated_bytes"])
            ):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _compare_scientific_values(
    left: Any,
    right: Any,
    *,
    atol: float,
    rtol: float,
) -> tuple[float, float]:
    """Compare nested scientific values and return maximum absolute/relative drift."""

    if isinstance(left, bool) or isinstance(right, bool):
        if left is not right:
            raise ExpertManifoldError("v6-prior profile boolean evidence differs")
        return 0.0, 0.0
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise ExpertManifoldError("v6-prior profile mapping evidence differs")
        maxima = [
            _compare_scientific_values(left[name], right[name], atol=atol, rtol=rtol)
            for name in left
        ]
        return (
            max((item[0] for item in maxima), default=0.0),
            max((item[1] for item in maxima), default=0.0),
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise ExpertManifoldError("v6-prior profile sequence evidence differs")
        maxima = [
            _compare_scientific_values(a, b, atol=atol, rtol=rtol)
            for a, b in zip(left, right, strict=True)
        ]
        return (
            max((item[0] for item in maxima), default=0.0),
            max((item[1] for item in maxima), default=0.0),
        )
    if isinstance(left, int) and isinstance(right, int):
        if left != right:
            raise ExpertManifoldError("v6-prior profile integer evidence differs")
        return 0.0, 0.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        a = float(left)
        b = float(right)
        if not math.isfinite(a) or not math.isfinite(b):
            raise ExpertManifoldError("v6-prior profile metric is non-finite")
        difference = abs(a - b)
        scale = max(abs(a), abs(b), 1e-12)
        relative = difference / scale
        if difference > atol + rtol * scale:
            raise ExpertManifoldError("v6-prior profile scientific metrics differ")
        return difference, relative
    if type(left) is not type(right) or left != right:
        raise ExpertManifoldError("v6-prior profile identity evidence differs")
    return 0.0, 0.0


def _scientific_metric_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ignored = {
        "step_seconds",
        "elapsed_seconds",
        "max_cuda_allocated_bytes",
        "max_cuda_reserved_bytes",
    }
    return [
        {name: value for name, value in row.items() if name not in ignored}
        for row in rows
    ]


def _expected_checkpoint_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_schema": contract["schema_version"],
        "mode": contract["mode"],
        "git_commit": contract["git"]["commit"],
        "config": contract["config"],
        "source": contract["source"],
        "initialization": contract["initialization"],
        "expert_bank_root": contract["expert_bank"]["root"],
        "expert_step": contract["expert_bank"]["step"],
        "objective": contract["objective"],
        "ownership": contract["ownership"],
        "world_size": contract["runtime"]["world_size"],
    }


def _expected_cursor_contract(
    contract: Mapping[str, Any], macro: int
) -> dict[str, Any]:
    data = contract["data"]
    return {
        "next_macro": macro,
        "task_visits_per_task": macro,
        "sampler_seed": int(data["sampler_seed"]),
        "teacher_video_seed": int(data["teacher_video_seed"]),
        "counterfactual_seed": int(data["counterfactual_seed"]),
        "counterfactual_phase": macro % 3,
        "videos_per_task_visit": 1,
        "action_queries_per_task": 20,
    }


def _profile_run_contract_matches(
    contract: Mapping[str, Any],
    *,
    gradient_evidence: Mapping[str, Any],
) -> bool:
    runtime = contract.get("runtime", {})
    data = contract.get("data", {})
    consumed = data.get("consumed_schedule", {})
    query = consumed.get("query", {})
    try:
        return (
            contract.get("schema_version") == "ember_pi05_v6_prior_writer_launch_v1"
            and contract.get("mode") == "profile"
            and _clean_pushed_git(contract.get("git", {}))
            and contract.get("config", {}).get("schema") == V6_PRIOR_CONFIG_SCHEMA
            and int(contract.get("config", {}).get("bytes", -1)) > 0
            and _method_matches(contract.get("method", {}))
            and _information_wall_matches(contract.get("information_wall", {}))
            and _writer_matches(contract.get("writer", {}))
            and _data_matches(data)
            and _objective_matches(
                contract.get("objective", {}),
                gradient_evidence=gradient_evidence,
            )
            and _optimization_matches(contract.get("optimization", {}))
            and contract.get("initialization", {}).get("mode")
            == "historical_v6_macro400_load_only"
            and contract.get("initialization", {}).get("optimizer") == "fresh"
            and contract.get("initialization", {}).get("scheduler") == "fresh"
            and contract.get("initialization", {}).get("rng") == "fresh_seed"
            and int(
                contract.get("initialization", {}).get("writer_state_tensor_count", -1)
            )
            == 600
            and int(contract.get("expert_bank", {}).get("step", -1)) == 2000
            and len(contract.get("expert_bank", {}).get("tasks", ())) == 24
            and contract.get("expert_bank", {}).get("deployment_read") is False
            and contract.get("expert_basis", {}).get("task_count") == 24
            and contract.get("ownership", {}).get("frozen_parameter_count") == 7_060_992
            and contract.get("ownership", {}).get("trainable_parameter_count")
            == 3_714_304
            and contract.get("ownership", {}).get("trainable_tensor_count") == 41
            and contract.get("ownership", {}).get(
                "source_policy_trainable_parameter_count"
            )
            == 0
            and int(runtime.get("world_size", -1)) == 6
            and int(runtime.get("tasks_per_rank", -1)) == 4
            and int(runtime.get("total_macros", -1)) == 3
            and runtime.get("gradient_profile_schedule_macro") is None
            and runtime.get("checkpoint_macros") == [1, 3]
            and runtime.get("distributed_model_wrapper") == "none"
            and runtime.get("gradient_reduction")
            == "single_flat_parameter_ordered_allreduce_mean_after_local_task_mean"
            and runtime.get("deferred_process_group") is True
            and runtime.get("nccl_p2p_disable") == "1"
            and runtime.get("nccl_algo") == "Ring"
            and runtime.get("nccl_proto") == "Simple"
            and _rank_topology_matches(runtime.get("rank_topology", ()), world_size=6)
            and int(query.get("start_step", -1)) == 0
            and int(query.get("stop_step", -1)) == 3
            and int(query.get("global_examples", -1)) == 1440
            and 0 < int(query.get("unique_query_rows", -1)) <= 1440
            and int(query.get("min_examples_per_task", -1)) == 60
            and int(query.get("max_examples_per_task", -1)) == 60
            and query.get("identity_evidence")
            == "cursor_counts_and_dataset_row_coverage"
            and int(consumed.get("videos_per_task_visit", -1)) == 1
            and int(consumed.get("min_video_visits_per_task", -1)) == 3
            and int(consumed.get("max_video_visits_per_task", -1)) == 3
            and int(consumed.get("min_unique_videos_per_task", -1)) == 3
            and int(consumed.get("max_unique_videos_per_task", -1)) == 3
            and contract.get("content_hash_policy") == "disabled_by_owner"
        )
    except (KeyError, TypeError, ValueError):
        return False


def _profile_invocations_match(
    rows: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    resumed: bool,
) -> bool:
    expected = (
        ((None, 1), (str(root / "checkpoints/macro_00000001"), 3))
        if resumed
        else ((None, 3),)
    )
    if len(rows) != len(expected):
        return False
    try:
        return all(
            row.get("resume") == resume
            and int(row.get("requested_stop_after_macro", -1)) == stop
            and isinstance(row.get("argv"), list)
            and float(row.get("started_unix", -1)) > 0
            and math.isfinite(float(row["started_unix"]))
            for row, (resume, stop) in zip(rows, expected, strict=True)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _completion_matches_profile(value: Mapping[str, Any]) -> bool:
    return value == {
        "schema_version": V6_PRIOR_COMPLETION_SCHEMA,
        "mode": "profile",
        "completed_macro": 3,
        "metrics_rows": 3,
        "content_hash_policy": "disabled_by_owner",
    }


def _resume_profile_evidence_matches(value: Mapping[str, Any]) -> bool:
    try:
        weights = value["auxiliary_weights"]
        tolerances = value["scientific_tolerances"]
        wall = value["step_seconds"]
        throughput = value["macros_per_second"]
        valid = (
            value.get("schema_version") == V6_PRIOR_RESUME_EVIDENCE_SCHEMA
            and all(
                isinstance(value.get(name), str) and bool(value[name])
                for name in ("gradient_root", "resumed_root", "contiguous_root")
            )
            and value["resumed_root"] != value["contiguous_root"]
            and _clean_pushed_git(value["profile_git"])
            and isinstance(value.get("gradient_commit"), str)
            and bool(value["gradient_commit"])
            and value.get("config_schema") == V6_PRIOR_CONFIG_SCHEMA
            and int(value.get("config_bytes", -1)) > 0
            and set(weights) == {"expert", "ranking"}
            and all(
                isinstance(weights[name], (int, float))
                and 0 <= float(weights[name]) <= 1
                for name in weights
            )
            and int(value.get("world_size", -1)) == 6
            and int(value.get("tasks_per_rank", -1)) == 4
            and _rank_topology_matches(value["rank_topology"], world_size=6)
            and value.get("invocation_counts") == {"resumed": 2, "contiguous": 1}
            and value.get("metrics_rows") == {"resumed": 3, "contiguous": 3}
            and value.get("checkpoint_macros") == [1, 3]
            and tolerances
            == {
                "scientific_atol": 0.0002,
                "scientific_rtol": 0.002,
                "writer_max_abs": 0.0000075,
                "writer_relative_l2": 0.00001,
            }
            and value.get("run_contracts_equal") is True
            and value.get("scientific_metrics_equivalent") is True
            and value.get("checkpoint_semantics_equivalent") is True
            and len(value.get("checkpoint_comparisons", ())) == 2
            and [int(row["macro"]) for row in value["checkpoint_comparisons"]] == [1, 3]
            and all(
                _checkpoint_comparison_evidence_matches(row)
                for row in value["checkpoint_comparisons"]
            )
            and math.isfinite(float(value["metric_max_abs_difference"]))
            and float(value["metric_max_abs_difference"]) >= 0
            and math.isfinite(float(value["metric_max_relative_difference"]))
            and float(value["metric_max_relative_difference"]) >= 0
            and math.isfinite(float(value["writer_max_abs_difference"]))
            and 0
            <= float(value["writer_max_abs_difference"])
            <= tolerances["writer_max_abs"]
            and math.isfinite(float(value["writer_relative_l2_difference"]))
            and 0
            <= float(value["writer_relative_l2_difference"])
            <= tolerances["writer_relative_l2"]
            and math.isclose(
                float(value["writer_max_abs_difference"]),
                max(
                    float(row["writer"]["max_abs"])
                    for row in value["checkpoint_comparisons"]
                ),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isclose(
                float(value["writer_relative_l2_difference"]),
                max(
                    float(row["writer"]["global_relative_l2"])
                    for row in value["checkpoint_comparisons"]
                ),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and set(wall) == {"resumed", "contiguous"}
            and set(throughput) == {"resumed", "contiguous"}
            and all(
                math.isfinite(float(wall[name])) and float(wall[name]) > 0
                for name in wall
            )
            and all(
                math.isclose(
                    float(throughput[name]),
                    3.0 / float(wall[name]),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
                for name in throughput
            )
            and int(value.get("max_cuda_allocated_bytes", -1)) > 0
            and int(value.get("max_cuda_reserved_bytes", -1))
            >= int(value.get("max_cuda_allocated_bytes", -1))
            and int(value.get("oom_count", -1)) == 0
            and int(value.get("nonfinite_count", -1)) == 0
            and value.get("content_hash_policy") == "disabled_by_owner"
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return valid


def _checkpoint_comparison_evidence_matches(value: Mapping[str, Any]) -> bool:
    try:
        writer = value["writer"]
        optimizer = value["optimizer"]
        return (
            int(value.get("macro", -1)) in {1, 3}
            and value.get("cursor_semantic_equal") is True
            and value.get("checkpoint_contract_semantic_equal") is True
            and int(value.get("rng_rank_count", -1)) == 6
            and value.get("rng_semantic_equal") is True
            and value.get("scheduler_semantic_equal") is True
            and value.get("amp_semantic_equal") is True
            and writer.get("tensor_schema_equal") is True
            and writer.get("frozen_exact") is True
            and int(writer.get("tensor_count", -1)) == 600
            and int(writer.get("frozen_tensor_count", -1)) == 559
            and int(writer.get("trainable_tensor_count", -1)) == 41
            and math.isclose(
                float(writer.get("scientific_atol", -1)),
                0.0002,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isclose(
                float(writer.get("scientific_rtol", -1)),
                0.002,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isclose(
                float(writer.get("max_abs_tolerance", -1)),
                0.0000075,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isclose(
                float(writer.get("global_relative_l2_tolerance", -1)),
                0.00001,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isfinite(float(writer.get("max_abs", -1)))
            and 0 <= float(writer["max_abs"]) <= 0.0000075
            and math.isfinite(float(writer.get("global_relative_l2", -1)))
            and 0 <= float(writer["global_relative_l2"]) <= 0.00001
            and optimizer.get("param_groups_equal") is True
            and math.isclose(
                float(optimizer.get("scientific_atol", -1)),
                0.0002,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isclose(
                float(optimizer.get("scientific_rtol", -1)),
                0.002,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and int(optimizer.get("tensor_count", -1)) > 0
            and math.isfinite(float(optimizer.get("max_abs", -1)))
            and float(optimizer["max_abs"]) >= 0
            and math.isfinite(float(optimizer.get("global_relative_l2", -1)))
            and float(optimizer["global_relative_l2"]) >= 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def assemble_v6_prior_resume_profile_evidence(
    *,
    gradient_root: Path,
    resumed_root: Path,
    contiguous_root: Path,
) -> dict[str, Any]:
    """Seal fresh/resume equivalence from retained six-rank profile artifacts."""

    from ember.expert_manifold.v6_prior_checkpoint import (
        compare_v6_prior_checkpoints,
        inspect_v6_prior_checkpoint,
    )

    gradient_root = gradient_root.resolve()
    resumed_root = resumed_root.resolve()
    contiguous_root = contiguous_root.resolve()
    if len({gradient_root, resumed_root, contiguous_root}) != 3:
        raise ExpertManifoldError("v6-prior profile evidence roots must be distinct")
    gradient = assemble_v6_prior_gradient_profile_evidence(gradient_root)
    roots = {"resumed": resumed_root, "contiguous": contiguous_root}
    artifacts: dict[str, dict[str, Any]] = {}
    for name, root in roots.items():
        for filename in (
            "run_contract.json",
            "invocations.jsonl",
            "metrics.jsonl",
            "completion.json",
        ):
            if not (root / filename).is_file():
                raise ExpertManifoldError(
                    f"missing v6-prior {name} profile artifact: {filename}"
                )
        artifacts[name] = {
            "contract": read_json(root / "run_contract.json"),
            "invocations": _read_jsonl(root / "invocations.jsonl"),
            "metrics": _read_jsonl(root / "metrics.jsonl"),
            "completion": read_json(root / "completion.json"),
        }
    resumed = artifacts["resumed"]
    contiguous = artifacts["contiguous"]
    contract = resumed["contract"]
    weights = gradient["recommended_weights"]
    try:
        contracts_valid = (
            contract == contiguous["contract"]
            and _profile_run_contract_matches(contract, gradient_evidence=gradient)
            and _profile_invocations_match(
                resumed["invocations"], root=resumed_root, resumed=True
            )
            and _profile_invocations_match(
                contiguous["invocations"], root=contiguous_root, resumed=False
            )
            and _completion_matches_profile(resumed["completion"])
            and _completion_matches_profile(contiguous["completion"])
            and _metric_rows_match_contract(
                resumed["metrics"],
                contract,
                expert_weight=float(weights["expert"]),
                ranking_weight=float(weights["ranking"]),
            )
            and _metric_rows_match_contract(
                contiguous["metrics"],
                contract,
                expert_weight=float(weights["expert"]),
                ranking_weight=float(weights["ranking"]),
            )
        )
    except (KeyError, TypeError, ValueError):
        contracts_valid = False
    if not contracts_valid:
        raise ExpertManifoldError("v6-prior resume-profile contract is incomplete")
    metric_max_abs, metric_max_relative = _compare_scientific_values(
        _scientific_metric_rows(resumed["metrics"]),
        _scientific_metric_rows(contiguous["metrics"]),
        atol=0.0002,
        rtol=0.002,
    )
    checkpoint_rows = []
    expected_checkpoint = _expected_checkpoint_contract(contract)
    for macro in (1, 3):
        checkpoint_name = f"macro_{macro:08d}"
        left = resumed_root / "checkpoints" / checkpoint_name
        right = contiguous_root / "checkpoints" / checkpoint_name
        for checkpoint in (left, right):
            inspected = inspect_v6_prior_checkpoint(checkpoint)
            manifest = read_json(checkpoint / "manifest.json")
            if (
                int(inspected.get("next_macro", -1)) != macro
                or manifest.get("cursor_contract")
                != _expected_cursor_contract(contract, macro)
                or manifest.get("checkpoint_contract") != expected_checkpoint
            ):
                raise ExpertManifoldError("v6-prior profile checkpoint lineage changed")
        compared = compare_v6_prior_checkpoints(
            left,
            right,
            scientific_atol=0.0002,
            scientific_rtol=0.002,
            writer_max_abs_tolerance=0.0000075,
            writer_relative_l2_tolerance=0.00001,
        )
        trainer = compared["trainer"]
        checkpoint_row = {
            "macro": macro,
            "cursor_semantic_equal": compared["cursor"]["semantic_equal"],
            "checkpoint_contract_semantic_equal": compared["checkpoint_contract"][
                "semantic_equal"
            ],
            "rng_rank_count": int(compared["rng"]["rank_count"]),
            "rng_semantic_equal": compared["rng"]["semantic_equal"],
            "scheduler_semantic_equal": trainer["scheduler_semantic_equal"],
            "amp_semantic_equal": trainer["amp_semantic_equal"],
            "optimizer": dict(trainer["optimizer"]),
            "writer": dict(compared["writer"]),
        }
        if not _checkpoint_comparison_evidence_matches(checkpoint_row):
            raise ExpertManifoldError(
                "v6-prior checkpoint comparison summary is incomplete"
            )
        checkpoint_rows.append(checkpoint_row)
    step_seconds = {
        name: sum(float(row["step_seconds"]) for row in value["metrics"])
        for name, value in artifacts.items()
    }
    peak_allocated = max(
        int(row["max_cuda_allocated_bytes"])
        for value in artifacts.values()
        for row in value["metrics"]
    )
    peak_reserved = max(
        int(row["max_cuda_reserved_bytes"])
        for value in artifacts.values()
        for row in value["metrics"]
    )
    evidence = {
        "schema_version": V6_PRIOR_RESUME_EVIDENCE_SCHEMA,
        "gradient_root": str(gradient_root),
        "resumed_root": str(resumed_root),
        "contiguous_root": str(contiguous_root),
        "gradient_commit": str(gradient["git"]["commit"]),
        "profile_git": dict(contract["git"]),
        "config_schema": str(contract["config"]["schema"]),
        "config_bytes": int(contract["config"]["bytes"]),
        "auxiliary_weights": {
            name: float(weights[name]) for name in ("expert", "ranking")
        },
        "world_size": 6,
        "tasks_per_rank": 4,
        "rank_topology": [dict(row) for row in contract["runtime"]["rank_topology"]],
        "invocation_counts": {"resumed": 2, "contiguous": 1},
        "metrics_rows": {"resumed": 3, "contiguous": 3},
        "checkpoint_macros": [1, 3],
        "scientific_tolerances": {
            "scientific_atol": 0.0002,
            "scientific_rtol": 0.002,
            "writer_max_abs": 0.0000075,
            "writer_relative_l2": 0.00001,
        },
        "run_contracts_equal": True,
        "scientific_metrics_equivalent": True,
        "checkpoint_semantics_equivalent": True,
        "checkpoint_comparisons": checkpoint_rows,
        "metric_max_abs_difference": metric_max_abs,
        "metric_max_relative_difference": metric_max_relative,
        "writer_max_abs_difference": max(
            float(row["writer"]["max_abs"]) for row in checkpoint_rows
        ),
        "writer_relative_l2_difference": max(
            float(row["writer"]["global_relative_l2"]) for row in checkpoint_rows
        ),
        "step_seconds": step_seconds,
        "macros_per_second": {
            name: 3.0 / seconds for name, seconds in step_seconds.items()
        },
        "max_cuda_allocated_bytes": peak_allocated,
        "max_cuda_reserved_bytes": peak_reserved,
        "oom_count": 0,
        "nonfinite_count": 0,
        "content_hash_policy": "disabled_by_owner",
    }
    if not _resume_profile_evidence_matches(evidence):
        raise ExpertManifoldError("assembled v6-prior resume seal is invalid")
    return evidence


def _state_machine_matches(config: Mapping[str, Any]) -> bool:
    state = (
        config.get("evaluation", {}).get("formal_status"),
        config.get("gradient_profile", {}).get("status"),
        config.get("objective", {}).get("auxiliary_weights", {}).get("status"),
        config.get("profile_run", {}).get("status"),
        config.get("formal_run", {}).get("status"),
    )
    return state in {
        (
            "blocked_until_live_a40_throughput_smoke",
            "blocked_until_single_a40_throughput_smoke",
            "blocked_until_live_train24_gradient_profile",
            "blocked_until_live_gradient_weights",
            "blocked_until_live_a40_resume_profile_evidence",
        ),
        (
            "sealed",
            "ready_after_cpu_and_single_a40_throughput_smoke",
            "blocked_until_live_train24_gradient_profile",
            "blocked_until_live_gradient_weights",
            "blocked_until_live_a40_resume_profile_evidence",
        ),
        (
            "sealed",
            "sealed_from_live_train24_gradient_profile",
            "sealed_from_live_train24_gradient_profile",
            "ready_after_live_gradient_profile",
            "blocked_until_live_a40_resume_profile_evidence",
        ),
        (
            "sealed",
            "sealed_from_live_train24_gradient_profile",
            "sealed_from_live_train24_gradient_profile",
            "sealed_from_live_a40_resume_profile_evidence",
            "sealed_from_live_a40_resume_profile_evidence",
        ),
    }


def load_v6_prior_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    authorities = config.get("authorities", {})
    initialization = config.get("initialization", {})
    basis = config.get("expert_basis", {})
    gradient_evidence = config.get("gradient_profile", {}).get("artifact_evidence")
    resume_evidence = config.get("profile_run", {}).get("artifact_evidence")
    evidence_lineage_matches = resume_evidence is None or (
        isinstance(resume_evidence, Mapping)
        and isinstance(gradient_evidence, Mapping)
        and resume_evidence.get("gradient_root") == gradient_evidence.get("root")
        and resume_evidence.get("gradient_commit")
        == gradient_evidence.get("git", {}).get("commit")
        and resume_evidence.get("auxiliary_weights")
        == gradient_evidence.get("recommended_weights")
    )
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
        and initialization.get("kind") == "load_only_historical_v6_fast_macro400"
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
        and _objective_matches(
            config.get("objective", {}),
            gradient_evidence=gradient_evidence,
        )
        and _optimization_matches(config.get("optimization", {}))
        and _runtime_declarations_match(config)
        and evidence_lineage_matches
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
            profile.get("status") != "ready_after_cpu_and_single_a40_throughput_smoke"
            or config["evaluation"].get("formal_status") != "sealed"
            or config["objective"]["auxiliary_weights"]["status"]
            != "blocked_until_live_train24_gradient_profile"
            or profile.get("artifact_evidence") is not None
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
            and _gradient_profile_evidence_matches(
                config["gradient_profile"].get("artifact_evidence", {})
            )
            and config["objective"]["auxiliary_weights"].get("expert")
            == config["gradient_profile"]["artifact_evidence"]["recommended_weights"][
                "expert"
            ]
            and config["objective"]["auxiliary_weights"].get("ranking")
            == config["gradient_profile"]["artifact_evidence"]["recommended_weights"][
                "ranking"
            ]
        )
    else:
        ready = (
            selected.get("status") == "sealed_from_live_a40_resume_profile_evidence"
            and config["profile_run"].get("status")
            == "sealed_from_live_a40_resume_profile_evidence"
            and config["gradient_profile"].get("status")
            == "sealed_from_live_train24_gradient_profile"
            and config["objective"]["auxiliary_weights"]["status"]
            == "sealed_from_live_train24_gradient_profile"
            and _gradient_profile_evidence_matches(
                config["gradient_profile"].get("artifact_evidence", {})
            )
            and _resume_profile_evidence_matches(
                config["profile_run"].get("artifact_evidence", {})
            )
            and config["profile_run"]["artifact_evidence"].get("gradient_root")
            == config["gradient_profile"]["artifact_evidence"].get("root")
            and config["profile_run"]["artifact_evidence"].get("auxiliary_weights")
            == config["gradient_profile"]["artifact_evidence"].get(
                "recommended_weights"
            )
        )
    if not ready:
        raise ExpertManifoldError(f"v6-prior {mode} runtime is not sealed")
    total = int(selected["total_macros"])
    checkpoints = tuple(int(value) for value in selected["checkpoint_macros"])
    if not total > 0 or not checkpoints or checkpoints[-1] != total:
        raise ExpertManifoldError("v6-prior checkpoint schedule changed")
    return total, checkpoints
