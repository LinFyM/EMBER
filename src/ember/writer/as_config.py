"""Configuration authority for the direct-family-B Dynamic-K Writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json
from ember.writer.errors import WriterModelError


REPO_ROOT = Path(__file__).resolve().parents[3]
AS_WRITER_CONFIG_SCHEMA = (
    "ember_pi05_dynamic_k_semantic_address_direct_family_b_rank8_as_writer_v1"
)
AS_WRITER_LAUNCH_SCHEMA = (
    "ember_pi05_dynamic_k_semantic_address_direct_family_b_rank8_as_writer_launch_v1"
)


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    return REPO_ROOT / str(config["authorities"][name]["path"])


def writer_stage(config: Mapping[str, Any]) -> str:
    stage = str(config.get("sealed_stage", ""))
    if stage != "development":
        raise WriterModelError("dynamic-K Writer currently supports development train24 only")
    return stage


def parse_macro_boundaries(value: str | list[int], total: int) -> tuple[int, ...]:
    if isinstance(value, str) and value.startswith("every:"):
        try:
            interval = int(value.removeprefix("every:"))
        except ValueError as error:
            raise WriterModelError("invalid Writer checkpoint interval") from error
        if interval <= 0 or total % interval:
            raise WriterModelError("Writer checkpoint interval must divide total macros")
        return tuple(range(interval, total + 1, interval))
    raw = value.split(",") if isinstance(value, str) else value
    try:
        result = tuple(sorted({int(item) for item in raw}))
    except (TypeError, ValueError) as error:
        raise WriterModelError("invalid Writer checkpoint macros") from error
    if not result or result[-1] != total or any(item <= 0 for item in result):
        raise WriterModelError("Writer checkpoints must end at total macros")
    return result


def _validate_authorities(config: Mapping[str, Any]) -> None:
    required = {
        "target_data_manifest",
        "evaluation_config",
        "lora_contract",
        "source_base_config",
        "tokenizer_manifest",
    }
    if set(config.get("authorities", {})) != required:
        raise WriterModelError("dynamic-K Writer authority set changed")
    for name in required:
        if not authority_path(config, name).is_file():
            raise WriterModelError(f"missing dynamic-K Writer authority: {name}")
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    if (
        lora.rank != 8
        or lora.alpha != 8
        or lora.parameter_count != 643_584
        or lora.state_tensor_count != 76
    ):
        raise WriterModelError("dynamic-K Writer rank-8 LoRA contract changed")


def _validate_method(config: Mapping[str, Any]) -> None:
    writer = config.get("writer", {})
    data = config.get("data", {})
    training = config.get("conditioning_training", {})
    distributed = config.get("optimization", {}).get("distributed", {})
    expected_writer = {
        "architecture": "pi05_dynamic_k_semantic_address_direct_family_b_rank8_v1",
        "generated_adapter": "complete_pi05_task_specific_rank8_lora",
        "camera_dataset": "obs/agentview_rgb",
        "camera_transform": "libero_opengl_rotate_180_chw_uint8",
        "frame_stride": 5,
        "include_final_frame": True,
        "backbone_total_frames_per_condition": 64,
        "native_action_probe_tokens": 50,
        "backbone_memory_tokens": 8,
        "backbone_layers": 18,
        "backbone_width": 1024,
        "program_width": 256,
        "mapper_width": 1024,
        "mapper_project": "shared_bias_free_256_to_1024",
        "lora_a": "fixed_template_without_dynamic_a_head",
        "lora_b_readout": (
            "four_bias_free_zero_initialized_direct_shape_family_linears_"
            "shared_across_layers_and_rank"
        ),
        "temporal_semantic_address": (
            "per_video_absolute_mean_memory_to_temporal_query_only"
        ),
        "action_meta_lora_rank": 4,
        "temporal_blocks": 2,
        "set_blocks": 2,
        "m2p_blocks": 2,
        "initialization_seed": 7,
        "activation_checkpointing": True,
    }
    expected_data = {
        "task_count": 24,
        "episodes_per_task": 50,
        "demo_indices": [0, 49],
        "action_chunk_size": 50,
        "action_queries_per_task": 20,
        "dynamic_k_max": 4,
        "dynamic_k_schedule": (
            "K(task,macro)=1+((sealed_task_permutation_position(task)+macro)%4)"
        ),
        "dynamic_k_balance": "exactly_six_tasks_at_each_K_per_macro",
    }
    expected_training = {
        "global_tasks_per_optimizer_update": 24,
        "update_topology": "one_complete_full24_equal_task_mean_per_macro",
        "task_assignment": "cost_balanced_long_first_dynamic_uneven",
        "pair_loss_reduction": "mean_within_task_then_equal_mean_over_24_tasks",
    }
    expected_distributed = {
        "fresh_world_sizes": [1, 2, 3, 4, 5, 6],
        "exact_resume_world_topology_locked": True,
        "gradient_communication": (
            "one_flat_writer_gradient_sum_all_reduce_per_macro_then_divide_by_24"
        ),
        "nccl_p2p_disable": "1",
        "deferred_process_group": True,
    }
    consistency = training.get("singleton_to_full_consistency", {})
    changed = (
        any(writer.get(key) != value for key, value in expected_writer.items())
        or int(writer.get("max_frames_per_encoder_call", 0)) <= 0
        or any(data.get(key) != value for key, value in expected_data.items())
        or any(training.get(key) != value for key, value in expected_training.items())
        or consistency.get("weight") != 0.05
        or consistency.get("kind")
        != "smooth_l1_singleton_program_to_stopgrad_full_set_program"
        or any(
            distributed.get(key) != value
            for key, value in expected_distributed.items()
        )
    )
    if changed:
        raise WriterModelError("dynamic-K Writer scientific contract changed")


def _validate_runtime(config: Mapping[str, Any]) -> None:
    for key in ("profile_defaults", "formal_run"):
        cell = config.get(key, {})
        total = int(cell.get("total_macros", 0))
        batch = int(cell.get("per_task_action_batch_size", 0))
        if (
            cell.get("allowed_world_sizes") != [1, 2, 3, 4, 5, 6]
            or total <= 0
            or batch != 20
        ):
            raise WriterModelError("dynamic-K Writer runtime contract changed")
        parse_macro_boundaries(cell["checkpoint_macros"], total)
    if config["formal_run"].get("status") not in {
        "unsealed_pending_live_profile",
        "sealed",
    }:
        raise WriterModelError("dynamic-K Writer formal status changed")
    if config["formal_run"].get("status") == "sealed":
        evidence = config["formal_run"].get("profile_evidence", {})
        if (
            evidence.get("source_commit")
            != "3866f50ab72439a4b522ead0c1a86fb7d30aa60b"
            or int(evidence.get("world_size", 0)) != 5
            or int(evidence.get("completion_macro", 0)) != 1
            or float(evidence.get("macro_seconds", 0)) <= 0
            or int(evidence.get("max_cuda_allocated_bytes", 0)) <= 0
            or evidence.get("global_k_histogram")
            != {"1": 6, "2": 6, "3": 6, "4": 6}
        ):
            raise WriterModelError("sealed dynamic-K live profile evidence changed")


def load_writer_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != AS_WRITER_CONFIG_SCHEMA:
        raise WriterModelError("unsupported dynamic-K Writer config schema")
    writer_stage(config)
    _validate_authorities(config)
    _validate_method(config)
    _validate_runtime(config)
    return config


def resolve_mode_config(config: Mapping[str, Any], mode: str) -> dict[str, Any]:
    if mode not in {"profile", "formal"}:
        raise WriterModelError("unsupported dynamic-K Writer runtime mode")
    if mode == "formal" and config["formal_run"]["status"] != "sealed":
        raise WriterModelError("formal dynamic-K training awaits a sealed live profile")
    return dict(config)
