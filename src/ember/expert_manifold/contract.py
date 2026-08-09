"""Hashless authority and runtime contract for task-local PI0.5 experts."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.writer.data import FunctionalQueryDataset, WriterTaskAuthority


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_SCHEMA = "ember_pi05_video_expert_manifold_v1"
BARYCENTRIC_CONFIG_SCHEMA = (
    "ember_pi05_video_expert_manifold_hard_routed_policy_effective_v2"
)
WORKER_CONTRACT_SCHEMA = "ember_pi05_task_expert_worker_launch_v1"
TOPOLOGY_ADDRESS_BINDING = "normalized_dynamic_times_normalized_chunk_plus_rank_address"


class ExpertManifoldError(RuntimeError):
    """Raised when expert-manifold training crosses its sealed boundary."""


@dataclass(frozen=True)
class ExpertTask:
    ordinal: int
    global_task_id: int
    suite: str
    task_id: int
    split_role: str
    language: str
    authority: WriterTaskAuthority


def _information_wall_matches(information: Mapping[str, Any]) -> bool:
    return (
        information.get("expert_action_split_roles") == ["train"]
        and information.get("writer_video_split_roles")
        == ["train", "validation", "test"]
        and information.get("writer_forbidden_inputs")
        == [
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
        ]
        and int(information.get("validation_experts_trained", -1)) == 0
        and int(information.get("test_experts_trained", -1)) == 0
        and int(information.get("validation_actions_read", -1)) == 0
        and int(information.get("test_actions_read", -1)) == 0
    )


def _topological_writer_matches(writer: Mapping[str, Any]) -> bool:
    return (
        int(writer.get("chunk_count", -1)) == 168
        and int(writer.get("public_rank", -1)) == 16
        and int(writer.get("valid_values", -1)) == 1_287_168
        and writer.get("video_value_path")
        == "phase_centered_projected_video_sqrt_normalized_causal_prefix_integral_only"
        and writer.get("routing_key_path")
        == "full_projected_video_innovation_plus_phase_keys"
        and writer.get("topology_address_binding") == TOPOLOGY_ADDRESS_BINDING
        and writer.get("output")
        == "zero_initialized_address_bound_chunk_values_plus_per_chunk_scale"
    )


def _exact_fields_match(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    return all(
        type(observed.get(name)) is type(value) and observed.get(name) == value
        for name, value in expected.items()
    )


def _integer_fields_match(
    observed: Mapping[str, Any], expected: Mapping[str, int]
) -> bool:
    try:
        return all(
            int(observed.get(name, -1)) == value for name, value in expected.items()
        )
    except (TypeError, ValueError):
        return False


def _meta_profile_evidence_matches(profile: Mapping[str, Any]) -> bool:
    return _exact_fields_match(
        profile,
        {
            "device": "NVIDIA A40",
            "distributed_model_wrapper": "none",
            "gradient_reduction": (
                "single_flat_parameter_ordered_allreduce_mean_after_local_task_mean"
            ),
            "nccl_p2p_disable": "1",
            "nccl_algo": "Ring",
            "nccl_proto": "Simple",
            "topology_address_binding": TOPOLOGY_ADDRESS_BINDING,
            "exact_resume_scientific_metrics_equal": True,
            "exact_resume_writer_bytes_equal": True,
            "exact_resume_rng_bytes_equal": True,
            "exact_resume_optimizer_scheduler_semantic_equal": True,
        },
    ) and _integer_fields_match(
        profile,
        {"world_size": 6, "oom_count": 0, "nonfinite_count": 0},
    )


def _meta_online_smoke_evidence_matches(smoke: Mapping[str, Any]) -> bool:
    return _exact_fields_match(
        smoke,
        {
            "device": "NVIDIA A40",
            "checkpoint_mode": "profile",
            "video_condition": "correct",
            "video_sampling": "without_replacement",
            "topology_address_binding": TOPOLOGY_ADDRESS_BINDING,
            "writer_modules_released": True,
            "source_policy_reused_for_rollout": True,
            "worker_return_codes": [0, 0, 0],
            "success_interpretation": "execution_smoke_only_not_performance_evidence",
        },
    ) and _integer_fields_match(
        smoke,
        {
            "checkpoint_macro": 3,
            "validation_task_count": 8,
            "scientific_rows": 8,
            "generated_entries": 8,
            "generated_batches": 2,
            "generation_batch_size": 4,
            "cache_entries": 8,
            "retry_count": 0,
            "failure_count": 0,
            "teacher_action_reads": 0,
            "teacher_state_reads": 0,
            "reward_reads": 0,
            "terminal_reads": 0,
            "oom_count": 0,
            "nonfinite_count": 0,
        },
    )


def _meta_formal_seal_matches(meta: Mapping[str, Any]) -> bool:
    formal = meta.get("formal_run", {})
    if formal.get("status") != "sealed":
        return formal.get("status") == (
            "blocked_until_live_a40_profile_and_online_generation_smoke"
        )
    profile = formal.get("profile_evidence", {})
    smoke = formal.get("online_smoke_evidence", {})
    return (
        int(formal.get("selected_expert_step", -1)) == 2000
        and int(formal.get("total_macros", -1)) == 800
        and int(formal.get("physical_microbatch_per_rank", -1)) == 1
        and int(formal.get("expected_world_size", -1)) == 6
        and int(formal.get("tasks_per_rank", -1)) == 4
        and formal.get("checkpoint_macros") == [50, 100, 200, 400, 600, 800]
        and _meta_profile_evidence_matches(profile)
        and _meta_online_smoke_evidence_matches(smoke)
    )


def load_expert_manifold_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ExpertManifoldError("unsupported expert-manifold config schema")
    method = config.get("method", {})
    experts = config.get("task_experts", {})
    video = config.get("video_features", {})
    writer = config.get("topological_writer", {})
    meta = config.get("meta_training", {})
    reduction = meta.get("optimization", {}).get("distributed_gradient_reduction", {})
    information = config.get("information_wall", {})
    if (
        method.get("name") != "video_conditioned_expert_manifold_topological_writer"
        or method.get("language_only_lora_path") is not False
        or not _information_wall_matches(information)
        or int(experts.get("task_count", -1)) != 24
        or int(experts.get("episodes_per_task", -1)) != 50
        or experts.get("task_parameter_sharing") != "none"
        or int(video.get("shots", -1)) != 1
        or int(video.get("phase_slots", -1)) != 16
        or int(video.get("image_hidden_width", -1)) != 2048
        or int(video.get("expert_hidden_width", -1)) != 1024
        or int(video.get("feature_width", -1)) != 3072
        or video.get("cache_contains_actions_or_state") is not False
        or not _topological_writer_matches(writer)
        or int(meta.get("task_count", -1)) != 24
        or int(meta.get("videos_per_task_per_macro", -1)) != 1
        or meta.get("task_aggregation") != "each_task_mean_then_train24_equal_mean"
        or meta.get("objective", {}).get("effective_ba_monitor_only") is not True
        or reduction.get("kind")
        != "single_flat_parameter_ordered_allreduce_mean_after_local_task_mean"
        or reduction.get("nccl_algo") != "Ring"
        or reduction.get("nccl_proto") != "Simple"
        or not _meta_formal_seal_matches(meta)
        or float(meta.get("objective", {}).get("raw_reconstruction_weight", -1)) != 1.0
        or int(experts.get("profile_defaults", {}).get("scheduler_total_steps", -1))
        != int(experts.get("formal_run", {}).get("total_steps", -2))
        or config.get("content_hash_policy") != "disabled_by_owner"
    ):
        raise ExpertManifoldError("expert-manifold scientific boundary changed")
    return config


def load_task_expert_config(path: Path) -> dict[str, Any]:
    """Load only the retained train24 task-expert authority from a legacy file."""

    config = read_json(path)
    experts = config.get("task_experts", {})
    formal = experts.get("formal_run", {})
    selection = formal.get("checkpoint_selection_evidence", {})
    profile = formal.get("profile_evidence", {})
    authorities = config.get("authorities", {})
    valid = (
        config.get("schema_version") == CONFIG_SCHEMA
        and set(authorities)
        == {
            "target_data_manifest",
            "evaluation_config",
            "lora_contract",
            "source_base_config",
        }
        and all(authority_path(config, name).is_file() for name in authorities)
        and _information_wall_matches(config.get("information_wall", {}))
        and int(experts.get("task_count", -1)) == 24
        and int(experts.get("episodes_per_task", -1)) == 50
        and experts.get("demo_indices") == [0, 49]
        and int(experts.get("action_chunk_size", -1)) == 50
        and experts.get("lora_topology")
        == "configs/pi05_lora_v1.json:38targets:rank16"
        and experts.get("task_parameter_sharing") == "none"
        and formal.get("status") == "sealed"
        and int(formal.get("total_steps", -1)) == 2000
        and int(formal.get("per_task_batch_size", -1)) == 16
        and formal.get("checkpoint_steps") == [250, 500, 1000, 1500, 2000]
        and int(formal.get("allowed_worker_count", -1)) == 6
        and int(formal.get("tasks_per_worker", -1)) == 4
        and int(formal.get("selected_stop_step", -1)) == 2000
        and formal.get("stage_stop_steps") == [1000, 2000]
        and int(selection.get("selected_step", -1)) == 2000
        and selection.get("selection")
        == "one_uniform_step_for_all_24_tasks_no_task_specific_mixing"
        and profile.get("device") == "NVIDIA A40"
        and int(profile.get("per_task_batch_size", -1)) == 16
        and profile.get("exact_resume_scientific_metrics_equal") is True
        and profile.get("exact_resume_adapter_bytes_equal") is True
        and int(profile.get("oom_count", -1)) == 0
        and int(profile.get("nonfinite_count", -1)) == 0
        and config.get("content_hash_policy") == "disabled_by_owner"
    )
    if not valid:
        raise ExpertManifoldError("task-expert scientific boundary changed")
    return config


def _barycentric_writer_matches(writer: Mapping[str, Any]) -> bool:
    return (
        _exact_fields_match(
            writer,
            {
                "causal_representation": (
                    "phase_centered_sqrt_normalized_causal_prefix_mean"
                ),
                "centroid_normalization": "unit_l2_after_train50_mean",
                "affine_score_rule": "centered_kernel_affine_barycentric",
                "deployed_coefficient_rule": (
                    "deterministic_signed_argmax_one_hot"
                ),
                "argmax_tie_break": "lowest_expert_ordinal",
                "affine_score_sum": 1.0,
                "zero_representation_coefficients": 0.0,
                "reconstruction": (
                    "per_target_unit_effective_ba_direction_plus_affine_log_frobenius"
                ),
                "scale_envelope": "per_target_train24_expert_frobenius_min_max",
                "effective_subspace": (
                    "independent_left_right_train24_energy_subspaces"
                ),
                "public_compression": "best_rank16_svd_inside_effective_subspace",
                "factor_gauge": (
                    "template_a_rowspace_procrustes_train_expert_geomean_a_rms"
                ),
                "identity": "template-A plus zero-B",
                "language_only_lora_path": False,
            },
        )
        and _integer_fields_match(
            writer,
            {
                "target_count": 38,
                "effective_basis_rank": 96,
                "public_rank": 16,
                "valid_values": 1_287_168,
                "deployed_coefficient_support": 1,
            },
        )
        and float(writer.get("ridge", -1)) == 0.3
        and float(writer.get("identity_epsilon", -1)) == 1e-12
    )


def _barycentric_smoke_evidence_matches(smoke: Mapping[str, Any]) -> bool:
    return _exact_fields_match(
        smoke,
        {
            "device": "NVIDIA A40",
            "video_condition": "correct",
            "video_sampling": "without_replacement",
            "writer_modules_released": True,
            "source_policy_reused_for_rollout": True,
            "source_policy_reloaded": False,
            "success_interpretation": "execution_smoke_only_not_performance_evidence",
        },
    ) and _integer_fields_match(
        smoke,
        {
            "validation_task_count": 8,
            "state_count": 1,
            "scientific_rows": 8,
            "generated_entries": 8,
            "cache_entries": 8,
            "retry_count": 0,
            "failure_count": 0,
            "teacher_action_reads": 0,
            "teacher_state_reads": 0,
            "reward_reads": 0,
            "terminal_reads": 0,
            "oom_count": 0,
            "nonfinite_count": 0,
        },
    )


def _barycentric_method_matches(method: Mapping[str, Any]) -> bool:
    return _exact_fields_match(
        method,
        {
            "name": (
                "video_conditioned_expert_manifold_hard_routed_"
                "policy_effective_writer"
            ),
            "writer_input": (
                "exact task language plus exactly one action-hidden teacher video"
            ),
            "dynamic_value": "one_video_phase_centered_causal_representation_only",
            "language_only_lora_path": False,
            "deployment_output": "one complete rank16 public LoRA",
            "learned_writer_parameter_count": 0,
        },
    )


def _barycentric_video_matches(video: Mapping[str, Any]) -> bool:
    return _exact_fields_match(
        video,
        {
            "shots": 1,
            "frame_stride": 5,
            "phase_slots": 16,
            "image_hidden_width": 2048,
            "expert_hidden_width": 1024,
            "feature_width": 3072,
            "source_policy_trainable": False,
            "cache_contains_actions_or_state": False,
        },
    ) and _integer_fields_match(
        video.get("extraction", {}),
        {
            "max_frames_per_encoder_call": 16,
            "initialization_seed": 20260807,
            "action_horizon": 50,
            "padded_action_dim": 32,
        },
    )


def _barycentric_basis_matches(basis: Mapping[str, Any]) -> bool:
    return _exact_fields_match(
        basis,
        {
            "checkpoint_selection": (
                "one_uniform_step_for_all_24_tasks_no_task_specific_mixing"
            )
        },
    ) and _integer_fields_match(
        basis,
        {"task_count": 24, "expert_step": 2000, "centroid_videos_per_task": 50},
    )


def _barycentric_loo_matches(loo: Mapping[str, Any]) -> bool:
    return _exact_fields_match(
        loo,
        {
            "selected_ridge": 0.3,
            "selected_reconstruction": "topological_direction_log_scale",
        },
    ) and _integer_fields_match(
        loo,
        {
            "held_out_task_count": 24,
            "available_basis_per_fold": 23,
            "full_lora_rows": 7200,
        },
    )


def _policy_effective_cpu_matches(evidence: Mapping[str, Any]) -> bool:
    return _exact_fields_match(
        evidence,
        {
            "artifact": (
                "runs/outputs/pi05_expert_manifold_causal_barycentric_"
                "correct400_noreplacement_seed7_0397be6_20260809/"
                "policy_effective_compiler_feasibility_full400_rank128_v2.json"
            ),
            "selected_reconstruction": "target_direction_log_norm",
            "pure_affine_norm_ratio_median": 0.5270653149687972,
            "selected_norm_ratio_median": 0.9864298623341197,
            "expert_captured_energy_median": 0.9967681395626802,
            "expert_captured_energy_min": 0.9933073976311083,
            "query_public_rank16_cosine_median": 0.9968194419163929,
            "query_public_rank16_cosine_min": 0.9953158395197425,
            "full_span_rank16_captured_energy_median": 0.9952348167613152,
            "cpu_only": True,
        },
    ) and _integer_fields_match(
        evidence,
        {
            "query_count": 400,
            "expert_count": 24,
            "target_count": 38,
            "selected_effective_basis_rank": 96,
            "full_span_sample_count": 8,
        },
    )


def _policy_effective_runtime_cpu_matches(evidence: Mapping[str, Any]) -> bool:
    return _exact_fields_match(
        evidence,
        {
            "artifact": (
                "runs/outputs/pi05_expert_manifold_policy_effective_"
                "cpu_real_assets_20260809/analysis.json"
            ),
            "cpu_only": True,
            "zero_identity_exact": True,
            "one_hot_expert_effective_cosine_median": 0.9983824844705533,
            "one_hot_expert_effective_cosine_min": 0.99664776353756,
            "demo0_intended_effective_target_cosine_median": 0.9983581845956992,
            "demo0_intended_effective_target_cosine_min": 0.9965721256367002,
            "ordered_reversed_coefficient_l2_min": 1.267863154411316,
            "demo0_cross_task_effective_cosine_median": 0.20255048083808294,
            "effective_lora_norm_median": 4.178999080650666,
            "stable_rank_median": 1.1254902621629606,
            "top_singular_energy_median": 0.9098611587469674,
            "public_a_rms_median": 0.01890874241400617,
            "public_b_rms_median": 0.008463979638103976,
            "q_b_column_cosine_median": 0.8145590998464618,
            "v_b_column_cosine_median": 0.8130396358999055,
            "action_b_column_cosine_median": 0.454599646424095,
        },
    ) and _integer_fields_match(
        evidence,
        {
            "expert_count": 24,
            "learned_parameter_count": 0,
            "persistent_buffer_bytes": 68_863_192,
            "active_rank_coordinates_min": 16,
        },
    )


def _soft_mixture_screen_matches(evidence: Mapping[str, Any]) -> bool:
    return _exact_fields_match(
        evidence,
        {
            "root": (
                "runs/outputs/pi05_expert_manifold_policy_effective_"
                "correct80_screen_noreplacement_seed7_ffed252_20260809"
            ),
            "artifact": (
                "runs/outputs/pi05_expert_manifold_policy_effective_"
                "correct80_screen_noreplacement_seed7_ffed252_20260809/"
                "strict_screen_and_paired_audit_v1.json"
            ),
            "exact_state_policy_rng_teacher_schedule": True,
            "interpretation": (
                "soft_policy_effective_mixture_did_not_justify_full400"
            ),
        },
    ) and _integer_fields_match(
        evidence,
        {
            "row_count": 80,
            "task_count": 8,
            "successes": 15,
            "breadth": 5,
            "raw_barycentric_reference_successes": 12,
            "raw_barycentric_gained": 6,
            "raw_barycentric_lost": 3,
        },
    )


def _hard_route_cpu_matches(evidence: Mapping[str, Any]) -> bool:
    return _exact_fields_match(
        evidence,
        {
            "artifact": (
                "runs/outputs/pi05_expert_manifold_hard_routed_"
                "cpu_real_assets_20260809/analysis.json"
            ),
            "cpu_only": True,
            "deployed_coefficient_rule": (
                "deterministic_signed_argmax_one_hot"
            ),
            "argmax_tie_break": "lowest_expert_ordinal",
            "zero_identity_exact": True,
            "all_nonzero_support_one": True,
            "all_nonzero_sum_one": True,
            "all_coefficients_finite": True,
            "all_states_finite": True,
            "train_centroid_self_route_all_tasks": True,
            "ordered_reversed_selection_changed": True,
        },
    ) and _integer_fields_match(
        evidence,
        {
            "expert_count": 24,
            "train_centroid_count": 24,
            "train_centroid_self_route_count": 24,
            "train_video_count": 1200,
            "deployed_support_min": 1,
            "deployed_support_max": 1,
        },
    )


def _hard_route_online_audit_matches(evidence: Mapping[str, Any]) -> bool:
    return _exact_fields_match(
        evidence,
        {
            "artifact": (
                "runs/outputs/pi05_expert_manifold_hard_routed_"
                "online_smoke_gpu02_14495d9_20260809/"
                "hard_route_online_smoke_route_audit_v1.json"
            ),
            "cpu_only_posthoc": True,
            "all_online_loras_match_one_hot_expert": True,
            "nearest_one_hot_effective_cosine_min": 0.9999997991655326,
            "nearest_one_hot_factor_relative_l2_max": 0.09622279383750876,
            "nearest_vs_second_relative_l2_gap_min": 0.3893612167170143,
            "rollout_outcomes_used_for_route": False,
        },
    ) and _integer_fields_match(
        evidence,
        {
            "row_count": 8,
            "unique_selected_experts": 7,
            "prior_soft_argmax_match_count": 7,
            "validation_action_reads": 0,
        },
    )


def _barycentric_evaluation_matches(evaluation: Mapping[str, Any]) -> bool:
    status = evaluation.get("formal_status")
    smoke = evaluation.get("online_smoke_evidence")
    hard_route = evaluation.get("cpu_hard_route_evidence")
    online_audit = evaluation.get("online_route_audit_evidence")
    if not _soft_mixture_screen_matches(
        evaluation.get("soft_mixture_screen_evidence", {})
    ):
        return False
    if status == "blocked_until_cpu_hard_route_evidence":
        return smoke is None and hard_route is None and online_audit is None
    if status == "blocked_until_live_a40_online_smoke":
        return (
            smoke is None
            and online_audit is None
            and isinstance(hard_route, Mapping)
            and _hard_route_cpu_matches(hard_route)
        )
    return (
        status == "sealed"
        and isinstance(smoke, Mapping)
        and isinstance(hard_route, Mapping)
        and isinstance(online_audit, Mapping)
        and _hard_route_cpu_matches(hard_route)
        and _hard_route_online_audit_matches(online_audit)
        and _barycentric_smoke_evidence_matches(smoke)
    )


def load_barycentric_writer_config(path: Path) -> dict[str, Any]:
    """Fail closed for the rejected hard-routed deployment family."""

    del path
    raise ExpertManifoldError("hard-routed Writer runtime is retired")


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    try:
        value = config["authorities"][name]["path"]
    except (KeyError, TypeError) as error:
        raise ExpertManifoldError(f"missing expert authority: {name}") from error
    return REPO_ROOT / str(value)


def load_train_tasks(
    config: Mapping[str, Any], data_root: Path
) -> tuple[ExpertTask, ...]:
    manifest = read_json(authority_path(config, "target_data_manifest"))
    selected = [
        row for row in manifest.get("tasks", []) if row.get("split_role") == "train"
    ]
    selected.sort(key=lambda row: int(row["global_task_id"]))
    if len(selected) != int(config["task_experts"]["task_count"]):
        raise ExpertManifoldError("expert target manifest did not resolve train24")
    tasks: list[ExpertTask] = []
    for ordinal, row in enumerate(selected):
        hdf5 = row["hdf5"]
        path = data_root / str(hdf5["relative_path"])
        expected_bytes = int(hdf5["bytes"])
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ExpertManifoldError(
                f"task expert HDF5 path or size changed: {int(row['global_task_id'])}"
            )
        authority = WriterTaskAuthority(
            task_id=int(row["global_task_id"]),
            language=str(row["language"]),
            path=path,
            expected_bytes=expected_bytes,
            expected_sha256=None,
        )
        tasks.append(
            ExpertTask(
                ordinal=ordinal,
                global_task_id=authority.task_id,
                suite=str(row["suite"]),
                task_id=int(row["task_id"]),
                split_role="train",
                language=authority.language,
                authority=authority,
            )
        )
    return tuple(tasks)


def parse_task_indices(value: str, task_count: int) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ExpertManifoldError(
            "task indices must be comma-separated integers"
        ) from error
    if (
        not result
        or len(set(result)) != len(result)
        or tuple(sorted(result)) != result
        or any(not 0 <= item < task_count for item in result)
    ):
        raise ExpertManifoldError(
            "task indices are duplicated, unsorted, or out of range"
        )
    return result


def _checkpoint_steps(values: Sequence[int], total_steps: int) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if (
        not result
        or tuple(sorted(set(result))) != result
        or result[-1] != total_steps
        or result[0] <= 0
    ):
        raise ExpertManifoldError("task-expert checkpoint schedule is invalid")
    return result


def resolve_runtime(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> tuple[int, int, tuple[int, ...], int]:
    experts = config["task_experts"]
    source = (
        experts["formal_run"] if args.mode == "formal" else experts["profile_defaults"]
    )
    if args.mode == "formal" and source.get("status") != "sealed":
        raise ExpertManifoldError(
            "formal task-expert config is not sealed by an A40 profile"
        )
    total_steps = int(source["total_steps"])
    batch_size = int(args.batch_size or source["per_task_batch_size"])
    checkpoints = _checkpoint_steps(source["checkpoint_steps"], total_steps)
    default_stop = int(source.get("selected_stop_step", total_steps))
    stop_step = int(args.stop_after_step or default_stop)
    allowed_stops = set(
        int(value) for value in source.get("stage_stop_steps", checkpoints)
    )
    allowed_stops.update(checkpoints)
    if batch_size <= 0 or stop_step not in allowed_stops or stop_step > total_steps:
        raise ExpertManifoldError(
            "task-expert runtime differs from an allowed stage boundary"
        )
    if args.mode == "formal":
        if batch_size != int(source["per_task_batch_size"]):
            raise ExpertManifoldError(
                "formal task-expert batch differs from its profile seal"
            )
        state = git_state(REPO_ROOT)
        if state["dirty_paths"]:
            raise ExpertManifoldError(
                "formal task-expert launch requires a clean worktree"
            )
        if args.resume is None and state["commit"] != state["upstream_commit"]:
            raise ExpertManifoldError("fresh formal task-expert launch must be pushed")
    return total_steps, batch_size, checkpoints, stop_step


def build_dataset(
    config: Mapping[str, Any], tasks: Sequence[ExpertTask]
) -> FunctionalQueryDataset:
    first, last = map(int, config["task_experts"]["demo_indices"])
    return FunctionalQueryDataset(
        [task.authority for task in tasks],
        demo_indices=range(first, last + 1),
        action_chunk_size=int(config["task_experts"]["action_chunk_size"]),
        max_open_files_per_worker=max(2, len(tasks)),
    )


def _physical_device() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return visible.split(",", 1)[0] if visible else "runtime-default"


def build_worker_contract(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    tasks: Sequence[ExpertTask],
    source: Mapping[str, Any],
    total_steps: int,
    batch_size: int,
    checkpoint_steps: Sequence[int],
) -> dict[str, Any]:
    state = git_state(REPO_ROOT)
    return {
        "schema_version": WORKER_CONTRACT_SCHEMA,
        "mode": args.mode,
        "method": "independent_task_local_rank16_policy_experts",
        "git": {key: state[key] for key in ("branch", "commit")},
        "config": {"path": str(args.config.resolve()), "schema": CONFIG_SCHEMA},
        "source": {
            "run": str(args.source_run.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "model_path": str(source["model_path"]),
        },
        "tokenizer": {
            "path": str(args.tokenizer_path.resolve()),
            "bytes": args.tokenizer_path.stat().st_size,
        },
        "tasks": [
            {
                "ordinal": task.ordinal,
                "global_task_id": task.global_task_id,
                "suite": task.suite,
                "task_id": task.task_id,
                "split_role": task.split_role,
                "language": task.language,
                "hdf5_bytes": task.authority.expected_bytes,
            }
            for task in tasks
        ],
        "information_wall": dict(config["information_wall"]),
        "runtime": {
            "host": socket.gethostname(),
            "cuda_visible_device": _physical_device(),
            "device_name": torch.cuda.get_device_name(0),
            "one_policy_per_worker": True,
            "task_parameter_sharing": "none",
            "total_steps_per_task": total_steps,
            "per_task_batch_size": batch_size,
            "checkpoint_steps": list(checkpoint_steps),
            "num_workers": 0,
        },
        "content_hash_policy": "disabled_by_owner",
    }


def publish_worker_contract(
    args: argparse.Namespace, contract: Mapping[str, Any], stop_step: int
) -> None:
    path = args.output_dir / "run_contract.json"
    if args.resume is None:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ExpertManifoldError("fresh task-expert worker output is not empty")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, dict(contract))
    else:
        if not path.is_file() or read_json(path) != contract:
            raise ExpertManifoldError("task-expert resume worker contract changed")
        resume = args.resume.resolve()
        worker_resume = resume == args.output_dir.resolve()
        checkpoint_resume = (
            len(resume.parents) >= 3 and resume.parents[2] == args.output_dir.resolve()
        )
        if not worker_resume and not checkpoint_resume:
            raise ExpertManifoldError(
                "task-expert resume checkpoint crossed worker ownership"
            )
    append_jsonl(
        args.output_dir / "invocations.jsonl",
        {
            "argv": sys.argv,
            "host": socket.gethostname(),
            "resume": str(args.resume.resolve()) if args.resume else None,
            "requested_stop_after_step": stop_step,
            "started_unix": time.time(),
        },
    )
    write_json_atomic(
        args.output_dir / "runtime_paths.json",
        {
            "source_run": str(args.source_run.resolve()),
            "source_checkpoint": str(args.checkpoint.resolve()),
            "target_data_root": str(args.data_root.resolve()),
            "tokenizer": str(args.tokenizer_path.resolve()),
        },
    )


def task_directory(output_dir: Path, task: ExpertTask) -> Path:
    return output_dir / f"task_{task.ordinal:02d}_global_{task.global_task_id:02d}"


def parse_resume_task(resume: Path | None) -> tuple[int, int] | None:
    if resume is None:
        return None
    try:
        task_dir = resume.parents[1].name
        step = int(resume.name.removeprefix("step_"))
        ordinal = int(task_dir.split("_", 2)[1])
    except (IndexError, ValueError) as error:
        raise ExpertManifoldError("task-expert resume path is malformed") from error
    if not resume.name.startswith("step_") or step <= 0:
        raise ExpertManifoldError("task-expert resume step is invalid")
    return ordinal, step


def worker_stage_resume_step(
    resume: Path | None,
    output_dir: Path,
    tasks: Sequence[ExpertTask],
) -> int | None:
    """Resolve an all-task stage resume from the existing worker root."""

    if resume is None or resume.resolve() != output_dir.resolve():
        return None
    summary = read_json(output_dir / "worker_summary.json")
    if (
        summary.get("schema_version") != "ember_pi05_task_expert_worker_summary_v1"
        or int(summary.get("completed_task_count", -1)) != len(tasks)
        or len(summary.get("tasks", [])) != len(tasks)
    ):
        raise ExpertManifoldError("task-expert worker stage summary changed")
    step = int(summary.get("selected_stop_step", -1))
    if step <= 0:
        raise ExpertManifoldError("task-expert worker stage cursor is invalid")
    rows = {int(row.get("task_ordinal", -1)): row for row in summary.get("tasks", [])}
    if set(rows) != {task.ordinal for task in tasks}:
        raise ExpertManifoldError("task-expert worker stage ownership changed")
    for task in tasks:
        row = rows[task.ordinal]
        checkpoint = (
            task_directory(output_dir, task) / "checkpoints" / f"step_{step:08d}"
        )
        if (
            int(row.get("global_task_id", -1)) != task.global_task_id
            or int(row.get("completed_steps", -1)) != step
            or not checkpoint.is_dir()
        ):
            raise ExpertManifoldError("task-expert worker stage is incomplete")
    return step
