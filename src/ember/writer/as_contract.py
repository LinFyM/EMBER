"""Authorities, data wall, and launch contracts for PI05 AS-Writer."""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.lora import canonical_contract_sha256
from ember.pi05_eval_contract import git_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import (
    DistributedContext,
    canonical_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.pi05_source_contract import append_jsonl
from ember.writer.data import FunctionalQueryDataset, WriterTaskAuthority
from ember.writer.model import CompleteLoRAWriter, WriterModelError


REPO_ROOT = Path(__file__).resolve().parents[3]
AS_WRITER_CONFIG_SCHEMA = "ember_pi05_language_axial_as_writer_v5_3"
AS_WRITER_LAUNCH_SCHEMA = "ember_pi05_language_axial_as_writer_launch_v5_3"
AS_WRITER_STAGES = ("development", "final")
_CHECKPOINT_NAME = re.compile(r"step_([0-9]{8})")


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    return REPO_ROOT / str(config["authorities"][name]["path"])


def writer_stage(config: Mapping[str, Any]) -> str:
    """Return the sealed data stage, preserving old development artifacts."""

    stage = str(config.get("sealed_stage", "development"))
    if stage not in AS_WRITER_STAGES:
        raise WriterModelError("unsupported PI05 AS-Writer stage")
    return stage


def writer_split_roles(config: Mapping[str, Any]) -> tuple[str, ...]:
    if writer_stage(config) == "development":
        return ("train",)
    return ("train", "validation")


def _validate_authorities(config: Mapping[str, Any]) -> None:
    authorities = config.get("authorities", {})
    required = {
        "target_data_manifest",
        "evaluation_config",
        "lora_contract",
        "source_base_config",
        "tokenizer_manifest",
    }
    if set(authorities) != required:
        raise WriterModelError("AS-Writer authority set changed")
    for name, authority in authorities.items():
        artifact = REPO_ROOT / str(authority.get("path", ""))
        if not artifact.is_file() or sha256_file(artifact) != authority.get("sha256"):
            raise WriterModelError(f"sealed AS-Writer authority changed: {name}")


def _expected_writer_contract(writer: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "architecture": (
            "pi05_language_axial_patch_grounded_core_visual_transition_"
            "causal_procedure_slot_fusion_v5_3"
        ),
        "generated_adapter": "complete_pi05_task_specific_rank16_lora",
        "camera_dataset": "obs/agentview_rgb",
        "camera_transform": "libero_opengl_rotate_180_chw_uint8",
        "frame_stride": writer["frame_stride"],
        "include_final_frame": True,
        "teacher_prompt": "Task: {cleaned_task};\nAction: ",
        "teacher_state_input": False,
        "task_span_extraction": (
            "authoritative_full_prompt_sentencepiece_piece_offsets"
        ),
        "text_branch_input": (
            "bos_plus_exact_authoritative_task_span_ids_without_template"
        ),
        "task_token_alignment": "text_and_multimodal_ids_identical_by_construction",
        "image_width": 2048,
        "native_image_tokens": 256,
        "multimodal_core_value": (
            "final_norm_task_span_hidden_plus_task_queried_image_position_content"
        ),
        "shared_language_projection": "bias_free_2048_to_256",
        "patch_grounding_attention": (
            "per_frame_text_only_task_queries_to_256_image_positions"
        ),
        "patch_grounding_qk": (
            "separate_pre_rmsnorm_bias_free_256_to_256"
        ),
        "patch_grounding_value": (
            "raw_shared_projected_image_position_content_no_value_projection"
        ),
        "patch_grounding_output": (
            "bias_free_256_to_256_added_to_multimodal_task_token_evidence"
        ),
        "patch_grounding_heads": 8,
        "expert_width": 1024,
        "text_meta_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "text_meta_lora_rank": 4,
        "vl_meta_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "vl_meta_lora_rank": 4,
        "action_meta_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "action_meta_lora_rank": 4,
        "max_frames_per_encoder_call": writer["max_frames_per_encoder_call"],
        "frame_batching_contract": (
            "encode_one_video_with_unpadded_memory_safety_chunks"
        ),
        "activation_checkpointing": True,
        "action_horizon": 50,
        "padded_action_dim": 32,
        "action_expert_probe": (
            "one_forward_fixed_persistent_gaussian_suffix_at_t1"
        ),
        "action_expert_action_out": False,
        "interaction_reduction": (
            "mean_50_final_suffix_hidden_then_shared_bias_free_1024_to_256"
        ),
        "program_width": 256,
        "frame_set_attention": (
            "token_aligned_frame_axis_only_video_independent_text_queries"
        ),
        "frame_attention_initial_lambda": 0.05,
        "frame_attention_order_contract": (
            "permutation_invariant_mean_anchored_no_frame_position"
        ),
        "semantic_core_heads": 8,
        "semantic_core_blocks": 2,
        "semantic_core_position_encoding": (
            "task_token_ordinal_rope_qk_only_bidirectional"
        ),
        "semantic_core_value_path": (
            "multimodal_task_token_plus_task_queried_patch_content"
        ),
        "procedure_heads": 8,
        "procedure_blocks": 2,
        "procedure_attention": "global_causal_pre_norm_with_valid_mask",
        "procedure_position_encoding": (
            "one_dimensional_rope_on_sampled_frame_ordinal_qk_only"
        ),
        "procedure_value_path": (
            "action_expert_probe_plus_task_grounded_adjacent_visual_transition"
        ),
        "visual_transition_source": (
            "adjacent_difference_of_task_queried_patch_evidence_in_actual_"
            "arm_input_order"
        ),
        "visual_transition_first_frame": "exact_zero",
        "visual_transition_padding": (
            "invalid_task_tokens_and_frames_exact_zero"
        ),
        "visual_transition_attention": (
            "action_expert_probe_queries_task_token_aligned_visual_transition"
        ),
        "visual_transition_qk": (
            "separate_pre_rmsnorm_bias_free_256_to_256"
        ),
        "visual_transition_value": (
            "raw_adjacent_patch_evidence_difference_no_value_projection"
        ),
        "visual_transition_output": (
            "bias_free_256_to_256_residual_added_to_action_expert_probe"
        ),
        "visual_transition_heads": 8,
        "procedure_initialization": "normal_nonzero",
        "query_count": 320,
        "routing_identity": "query_module_layer_rank_qk_only",
        "core_slot_reader": "routing_qk_core_content_v",
        "procedure_slot_reader": (
            "routing_plus_normalized_core_q_centered_procedure_v"
        ),
        "slot_fusion": (
            "zero_initialized_bias_free_adaln_then_one_post_fusion_block"
        ),
        "fusion_heads": 8,
        "procedure_value_centering": "parameter_free_valid_time_mean",
        "modulation_projection": "bias_free_256_to_512_zero_initialized",
        "post_fusion_blocks": 1,
        "factor_head_bias": False,
        "factor_hidden_width": 192,
        "initialization_seed": 7,
    }


def _validate_protocol(config: Mapping[str, Any]) -> None:
    target = read_json(authority_path(config, "target_data_manifest"))
    roles = target.get("summary", {}).get("roles", {})
    if (
        target.get("schema_version") != "ember_pi05_target_data_manifest_v1"
        or int(target.get("summary", {}).get("tasks", -1)) != 40
        or int(target.get("summary", {}).get("episodes", -1)) != 2000
        or {name: len(roles.get(name, [])) for name in ("train", "validation", "test")}
        != {"train": 24, "validation": 8, "test": 8}
    ):
        raise WriterModelError("AS-Writer target-data authority is not sealed 24/8/8")
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    if lora.source_base_config_sha256 != config["authorities"]["source_base_config"]["sha256"]:
        raise WriterModelError("AS-Writer LoRA and source-base authorities disagree")
    writer = config.get("writer", {})
    if (
        writer.get("frame_stride") != 5
        or int(writer.get("max_frames_per_encoder_call", 0)) <= 0
    ):
        raise WriterModelError("sealed Language-Axial Writer dimensions changed")
    expected = _expected_writer_contract(writer)
    if writer != expected:
        missing = sorted(set(expected) - set(writer))
        extra = sorted(set(writer) - set(expected))
        changed = sorted(
            key
            for key in set(writer) & set(expected)
            if writer[key] != expected[key]
        )
        raise WriterModelError(
            "Language-Axial AS-Writer architecture changed; "
            f"missing={missing}, extra={extra}, changed={changed}"
        )


def _validate_information_wall(config: Mapping[str, Any]) -> None:
    common = {
        "writer_input": (
            "task language plus exactly one raw action-hidden teacher video at inference"
        ),
        "writer_forbidden_inputs": [
            "action",
            "proprio",
            "state",
            "reward",
            "terminal",
            "task_id",
            "filename",
            "hidden_normalization",
            "policy_outcome",
        ],
        "action_owner": "frozen functional behavior loss only",
        "test_actions_read": 0,
        "test_video_values_read": 0,
    }
    if writer_stage(config) == "development":
        expected = {
            **common,
            "development_action_split_roles": ["train"],
            "development_video_split_roles": ["train"],
            "validation_actions_read_by_training_optimizer": 0,
            "validation_action_queries_per_checkpoint_monitor": 512,
            "validation_action_gradient": False,
        }
    else:
        expected = {
            **common,
            "final_action_split_roles": ["train", "validation"],
            "final_video_split_roles": ["train", "validation"],
        }
    if config.get("information_wall") != expected:
        raise WriterModelError("AS-Writer information wall changed")
    data = config.get("data", {})
    required = {
        "task_count": 24 if writer_stage(config) == "development" else 32,
        "demo_indices": [0, 49],
        "episodes_per_task": 50,
        "teacher_video_sampling": (
            "per_rank_task_visit_deterministic_single_same_task_video_in_"
            "no_replacement_cycles"
        ),
        "action_query_sampling": "task-balanced deterministic no-replacement episode cycles",
        "video_action_pairing": (
            "one task-video LoRA conditions the complete rank-local action batch"
        ),
        "writer_generation_reuse": (
            "generate one task-video LoRA once then reuse it across the complete "
            "rank-local action batch"
        ),
    }
    if any(data.get(name) != value for name, value in required.items()):
        raise WriterModelError("AS-Writer sampling contract changed")


def _validate_conditioning_training(config: Mapping[str, Any]) -> None:
    value = config.get("conditioning_training", {})
    normal = {
        "method": "single_video_multi_action_positive_functional_loss",
        "writer_language_contract": (
            "correct_task_language_state_free_teacher_action_suffix"
        ),
        "policy_language_contract": "correct_action_query_task_language",
        "action_query_batch_owner": (
            "one physical action batch per rank with no optimizer gradient accumulation"
        ),
        "task_assignment": (
            "one task per rank per optimizer step with globally balanced task rotation"
        ),
        "teacher_videos_per_task_visit": 1,
        "action_video_assignment": "all_actions_share_single_video_lora",
        "logical_pair_batch": "per_rank_action_batch",
        "policy_noise_contract": (
            "one independent policy flow noise and time draw per action query"
        ),
        "pair_loss_reduction": "mean_over_rank_local_action_batch",
        "normal_loss_weight": 1.0,
    }
    if value != normal:
        raise WriterModelError("AS-Writer conditioning contract changed")


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and value > 0


def load_writer_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != AS_WRITER_CONFIG_SCHEMA:
        raise WriterModelError("unsupported PI05 AS-Writer config schema")
    writer_stage(config)
    _validate_authorities(config)
    _validate_protocol(config)
    _validate_information_wall(config)
    _validate_conditioning_training(config)
    return config


def parse_checkpoint_steps(value: str | Sequence[int], total_steps: int) -> tuple[int, ...]:
    if isinstance(value, str) and value.startswith("every:"):
        try:
            interval = int(value.removeprefix("every:"))
        except ValueError as error:
            raise WriterModelError("invalid AS-Writer checkpoint interval") from error
        if interval <= 0 or total_steps % interval:
            raise WriterModelError(
                "AS-Writer checkpoint interval must divide total steps"
            )
        return tuple(range(interval, total_steps + 1, interval))
    raw = value.split(",") if isinstance(value, str) else value
    try:
        result = tuple(sorted({int(item) for item in raw}))
    except (TypeError, ValueError) as error:
        raise WriterModelError("invalid AS-Writer checkpoint steps") from error
    if not result or result[-1] != total_steps or any(step <= 0 for step in result):
        raise WriterModelError("AS-Writer checkpoints must end at total_steps")
    return result


def resume_step(checkpoint: Path | None) -> int:
    if checkpoint is None:
        return 0
    match = _CHECKPOINT_NAME.fullmatch(checkpoint.name)
    if match is None:
        raise WriterModelError("AS-Writer resume path is not a step checkpoint")
    return int(match.group(1))


def resolve_runtime(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> tuple[int, int, tuple[int, ...]]:
    if args.mode == "formal" and config["formal_run"].get("status") != "sealed":
        raise WriterModelError(
            "formal AS-Writer config is not sealed from the live v5.3 profile"
        )
    source = config["formal_run"] if args.mode == "formal" else config["profile_defaults"]
    total_steps = args.total_steps or int(source["total_steps"])
    batch_size = args.batch_size or int(source["per_rank_batch_size"])
    checkpoint_steps = parse_checkpoint_steps(
        args.checkpoint_steps or source["checkpoint_steps"], total_steps
    )
    default_stop = int(source.get("selected_stop_step", total_steps))
    stop_step = args.stop_after_step or default_stop
    if min(total_steps, batch_size, stop_step) <= 0 or stop_step > total_steps:
        raise WriterModelError("invalid AS-Writer runtime request")
    if int(config["conditioning_training"]["teacher_videos_per_task_visit"]) != 1:
        raise WriterModelError("AS-Writer training must use one video per rank step")
    expected_world_size = int(source.get("expected_world_size", 8))
    if context.world_size != expected_world_size:
        raise WriterModelError(
            "AS-Writer training requires exactly "
            f"{expected_world_size} symmetric ranks"
        )
    if args.mode == "formal":
        formal = config["formal_run"]
        expected = (
            "sealed",
            int(formal["expected_world_size"]),
            int(formal["total_steps"]),
            int(formal["per_rank_batch_size"]),
            parse_checkpoint_steps(formal["checkpoint_steps"], total_steps),
        )
        observed = (
            formal.get("status"),
            context.world_size,
            total_steps,
            batch_size,
            checkpoint_steps,
        )
        stage_stops = parse_checkpoint_steps(
            formal.get("stage_stop_steps", [default_stop]),
            total_steps,
        )
        if (
            observed != expected
            or not stage_stops
            or any(value not in checkpoint_steps for value in stage_stops)
            or default_stop not in stage_stops
            or stop_step not in stage_stops
        ):
            raise WriterModelError("formal AS-Writer launch differs from its sealed profile")
        state = git_state(REPO_ROOT)
        if state["dirty_paths"]:
            raise WriterModelError("formal AS-Writer launch requires a clean worktree")
        if args.resume is None and state["commit"] != state["origin_main"]:
            raise WriterModelError("fresh formal AS-Writer launch must be pushed")
        if context.numa_node is None or not context.cpu_affinity:
            raise WriterModelError("formal AS-Writer launch requires GPU-local NUMA binding")
        if args.skip_data_sha:
            raise WriterModelError("formal AS-Writer launch must verify every train HDF5")
    args.stop_after_step = stop_step
    return total_steps, batch_size, checkpoint_steps


def _broadcast_validation(
    context: DistributedContext, operation: Any
) -> dict[str, Any]:
    payload: list[Any] = [None]
    if context.is_main:
        try:
            payload[0] = operation()
        except Exception as error:
            payload[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=context.device)
    if payload[0].get("error"):
        raise WriterModelError(payload[0]["error"])
    return payload[0]


def _validate_target_files(
    tasks: Sequence[WriterTaskAuthority], verify_hashes: bool
) -> dict[str, Any]:
    for task in tasks:
        path = task.path
        if not path.is_file() or path.stat().st_size != task.expected_bytes:
            raise WriterModelError(f"AS-Writer train HDF5 size changed: {task.task_id}")
        if verify_hashes and sha256_file(path) != task.expected_sha256:
            raise WriterModelError(f"AS-Writer train HDF5 hash changed: {task.task_id}")
    return {
        "tasks_checked": len(tasks),
        "bytes_checked": sum(task.expected_bytes for task in tasks),
        "full_sha256_verified": verify_hashes,
        "hdf5_identity_sha256": canonical_hash(
            [
                [task.task_id, task.expected_bytes, task.expected_sha256]
                for task in tasks
            ]
        ),
    }


def load_training_data(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> tuple[FunctionalQueryDataset, tuple[WriterTaskAuthority, ...], dict[str, Any]]:
    target = read_json(authority_path(config, "target_data_manifest"))
    roles = set(writer_split_roles(config))
    root = args.data_root.resolve()
    rows = tuple(
        row for row in target["tasks"] if str(row["split_role"]) in roles
    )
    tasks_list = []
    for row in rows:
        path = (root / str(row["hdf5"]["relative_path"])).resolve()
        if not path.is_relative_to(root):
            raise WriterModelError("target HDF5 escaped its declared data root")
        tasks_list.append(
            WriterTaskAuthority(
                task_id=int(row["global_task_id"]),
                language=str(row["language"]),
                path=path,
                expected_bytes=int(row["hdf5"]["bytes"]),
                expected_sha256=str(row["hdf5"]["sha256"]),
            )
        )
    tasks = tuple(sorted(tasks_list, key=lambda task: task.task_id))
    suite_counts: dict[str, int] = {}
    for row in rows:
        suite_counts[str(row["suite"])] = suite_counts.get(str(row["suite"]), 0) + 1
    per_suite = 6 if writer_stage(config) == "development" else 8
    if (
        len(tasks) != int(config["data"]["task_count"])
        or sorted(suite_counts.values()) != [per_suite] * 4
    ):
        raise WriterModelError("AS-Writer action training is not its sealed source role")
    validation = _broadcast_validation(
        context, lambda: _validate_target_files(tasks, not args.skip_data_sha)
    )
    first_demo, last_demo = map(int, config["data"]["demo_indices"])
    query_authorities = tuple(
        WriterTaskAuthority(
            task_id=task.task_id,
            language=task.language,
            path=task.path,
            expected_bytes=task.expected_bytes,
        )
        for task in tasks
    )
    dataset = FunctionalQueryDataset(
        query_authorities,
        demo_indices=range(first_demo, last_demo + 1),
        action_chunk_size=int(config["data"]["action_chunk_size"]),
        max_open_files_per_worker=int(config["data"]["max_open_files_per_worker"]),
    )
    return dataset, tasks, validation


def inspect_video_data(
    root: Path,
    config: Mapping[str, Any],
    task_ids: Sequence[int],
    *,
    verify_hashes: bool,
) -> dict[str, Any]:
    root = root.resolve()
    target_path = authority_path(config, "target_data_manifest")
    target = read_json(target_path)
    by_id = {
        int(row["global_task_id"]): row for row in target.get("tasks", [])
    }
    selected_ids = tuple(sorted({int(task_id) for task_id in task_ids}))
    if not selected_ids or set(selected_ids) - set(by_id):
        raise WriterModelError("Writer video task IDs are outside target40")
    records = []
    for task_id in selected_ids:
        row = by_id[task_id]
        path = (root / str(row["hdf5"]["relative_path"])).resolve()
        expected_bytes = int(row["hdf5"]["bytes"])
        expected_sha256 = str(row["hdf5"]["sha256"])
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or path.stat().st_size != expected_bytes
            or (verify_hashes and sha256_file(path) != expected_sha256)
        ):
            raise WriterModelError(f"Writer video HDF5 changed: {task_id}")
        records.append(
            [task_id, str(row["hdf5"]["relative_path"]), expected_bytes, expected_sha256]
        )
    return {
        "root": str(root.resolve()),
        "schema_version": "ember_pi05_raw_teacher_video_data_v1",
        "target_data_manifest_file_sha256": sha256_file(target_path),
        "target_data_manifest_payload_sha256": target["canonical_payload_sha256"],
        "dataset": dict(target["dataset"]),
        "task_ids": list(selected_ids),
        "task_count": len(selected_ids),
        "episode_count": 50 * len(selected_ids),
        "hdf5_identity_sha256": canonical_hash(records),
        "full_sha256_verified": verify_hashes,
        "test_video_values_read": 0,
    }


def inspect_feature_cache(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Fail closed for retired pooled-feature callers."""

    raise WriterModelError(
        "pooled PI05 Writer feature caches are retired; Core-Causal AS-Writer "
        "requires raw teacher video data"
    )


def writer_trainable_contract(
    writer: CompleteLoRAWriter, policy: torch.nn.Module, lora: Any
) -> dict[str, Any]:
    names = sorted(name for name, value in writer.named_parameters() if value.requires_grad)
    parameter_count = sum(value.numel() for value in writer.parameters())
    if (
        not names
        or parameter_count != 10_230_536
        or any(parameter.requires_grad for parameter in policy.parameters())
    ):
        raise WriterModelError("AS-Writer freeze boundary changed")
    return {
        "object": "shared_action_supervised_writer_only",
        "parameter_count": parameter_count,
        "parameter_name_count": len(names),
        "parameter_names_sha256": canonical_hash(names),
        "generated_lora_parameter_count": lora.parameter_count,
        "generated_lora_tensor_count": lora.state_tensor_count,
        "lora_contract_sha256": canonical_contract_sha256(lora),
        "source_policy_trainable_parameter_count": 0,
    }


def _software_versions() -> dict[str, Any]:
    packages = ("lerobot", "transformers", "peft", "safetensors", "h5py")
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def _contract_stop_step(
    args: argparse.Namespace, config: Mapping[str, Any], total_steps: int
) -> int:
    source = config["formal_run"] if args.mode == "formal" else config["profile_defaults"]
    return int(source.get("selected_stop_step", total_steps))


def build_contract(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    source: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    video_data: Mapping[str, Any],
    data_validation: Mapping[str, Any],
    task_ids: Sequence[int],
    trainable: Mapping[str, Any],
    total_steps: int,
    batch_size: int,
    batch_cycle: Sequence[int],
    checkpoint_steps: Sequence[int],
    initialization: Mapping[str, Any],
) -> dict[str, Any]:
    contract_stop_step = _contract_stop_step(args, config, total_steps)
    videos_per_task_visit = int(
        config["conditioning_training"]["teacher_videos_per_task_visit"]
    )
    policy_forward_calls = 1
    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "device": str(context.device),
        "numa_node": context.numa_node,
        "cpu_affinity": list(context.cpu_affinity or ()),
    }
    topology: list[Any] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(topology, local)
    else:
        topology[0] = local
    return {
        "schema_version": AS_WRITER_LAUNCH_SCHEMA,
        "mode": args.mode,
        "stage": writer_stage(config),
        "git": {key: value for key, value in git_state(REPO_ROOT).items() if key in {"branch", "commit"}},
        "config_sha256": sha256_file(args.config.resolve()),
        "authorities": dict(config["authorities"]),
        "source": dict(source),
        "tokenizer": dict(tokenizer),
        "video_data": dict(video_data),
        "target_action_data_validation": dict(data_validation),
        "information_wall": dict(config["information_wall"]),
        "writer": dict(config["writer"]),
        "data": dict(config["data"]),
        "conditioning_training": dict(config["conditioning_training"]),
        "optimization": dict(config["optimization"]),
        **(
            {"initialization": dict(initialization)}
            if initialization.get("mode") == "writer_weight_warm_start"
            else {}
        ),
        "task_ids": list(task_ids),
        "runtime": {
            "world_size": context.world_size,
            "one_policy_cuda_process_per_rank": True,
            "extra_cuda_roles_on_any_rank": 0,
            "ddp_object": "shared_writer_only",
            "action_query_batch_size_per_rank": batch_size,
            "per_rank_unique_action_query_cycle": list(batch_cycle),
            "teacher_videos_per_task_visit": videos_per_task_visit,
            "writer_video_conditions_per_rank": videos_per_task_visit,
            "actions_per_video_condition": batch_size,
            "action_video_assignment": "all_actions_share_single_video_lora",
            "logical_pairs_per_rank": batch_size,
            "optimizer_gradient_accumulation": False,
            "global_policy_samples_per_step": (
                context.world_size
                * batch_size
            ),
            "policy_forward_calls_per_optimizer_step": policy_forward_calls,
            "writer_conditions_per_rank": videos_per_task_visit,
            "total_steps": total_steps,
            "selected_stop_step": contract_stop_step,
            "checkpoint_steps": list(checkpoint_steps),
            "num_workers_per_rank": args.num_workers,
            "rank_topology": topology,
        },
        "trainable": dict(trainable),
        "software": _software_versions(),
    }


def publish_contract(
    args: argparse.Namespace,
    context: DistributedContext,
    contract: Mapping[str, Any],
    contract_sha256: str,
) -> None:
    def operation() -> dict[str, bool]:
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
            raise WriterModelError(f"AS-Writer output directory is not empty: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        contract_path = args.output_dir / "run_contract.json"
        if args.resume is not None:
            if not contract_path.is_file() or canonical_hash(read_json(contract_path)) != contract_sha256:
                raise WriterModelError("AS-Writer resume launch contract changed")
        else:
            write_json_atomic(contract_path, dict(contract))
        append_jsonl(
            args.output_dir / "invocations.jsonl",
            {
                "argv": sys.argv,
                "contract_git": dict(contract["git"]),
                "runtime_git": {
                    key: value
                    for key, value in git_state(REPO_ROOT).items()
                    if key in {"branch", "commit"}
                },
                "contract_compatible_code_resume": bool(
                    args.resume is not None
                    and contract["git"].get("commit")
                    != git_state(REPO_ROOT).get("commit")
                ),
                "host": socket.gethostname(),
                "resume": str(args.resume) if args.resume else None,
                "started_unix": time.time(),
            },
        )
        write_json_atomic(
            args.output_dir / "runtime_paths.json",
            {
                "source_run": str(args.source_run.resolve()),
                "source_checkpoint": str(args.checkpoint.resolve()),
                "writer_initialization_checkpoint": (
                    str(args.initialize_writer_checkpoint.resolve())
                    if args.initialize_writer_checkpoint
                    else None
                ),
                "target_data_root": str(args.data_root.resolve()),
                "tokenizer": str(args.tokenizer_path.resolve()),
            },
        )
        return {"ok": True}

    _broadcast_validation(context, operation)


def reconcile_resume_contract(
    args: argparse.Namespace, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    candidate = dict(candidate)
    if args.resume is None:
        if getattr(args, "allow_contract_compatible_code_resume", False):
            raise WriterModelError(
                "contract-compatible code resume requires a checkpoint"
            )
        return candidate
    contract_path = args.output_dir / "run_contract.json"
    if not contract_path.is_file():
        return candidate
    existing = read_json(contract_path)
    if existing == candidate:
        return existing
    if not getattr(args, "allow_contract_compatible_code_resume", False):
        raise WriterModelError("AS-Writer resume launch contract changed")
    existing_git = existing.get("git", {})
    candidate_git = candidate.get("git", {})
    if (
        existing_git.get("branch") != candidate_git.get("branch")
        or existing_git.get("commit") == candidate_git.get("commit")
    ):
        raise WriterModelError(
            "AS-Writer code-compatible resume did not isolate one commit change"
        )
    normalized = dict(candidate)
    normalized["git"] = existing_git
    if normalized != existing:
        raise WriterModelError(
            "AS-Writer code-compatible resume changed the scientific contract"
        )
    return existing
