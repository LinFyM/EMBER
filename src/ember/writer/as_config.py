"""Configuration authority for the Layer-Matched Memory Program Compiler."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json
from ember.writer.errors import WriterModelError


REPO_ROOT = Path(__file__).resolve().parents[3]
AS_WRITER_CONFIG_SCHEMA = "ember_pi05_layer_matched_memory_program_compiler_writer_v1"
AS_WRITER_LAUNCH_SCHEMA = (
    "ember_pi05_layer_matched_memory_program_compiler_writer_launch_v1"
)


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    return REPO_ROOT / str(config["authorities"][name]["path"])


def writer_stage(config: Mapping[str, Any]) -> str:
    stage = str(config.get("sealed_stage", ""))
    if stage != "development":
        raise WriterModelError("LMMPC currently supports development train24 only")
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
        raise WriterModelError("LMMPC authority set changed")
    for name in required:
        if not authority_path(config, name).is_file():
            raise WriterModelError(f"missing LMMPC authority: {name}")
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    if (
        lora.rank != 16
        or lora.alpha != 16
        or lora.parameter_count != 1_287_168
        or lora.state_tensor_count != 76
    ):
        raise WriterModelError("LMMPC rank16 LoRA contract changed")


def _validate_method(config: Mapping[str, Any]) -> None:
    writer = config.get("writer", {})
    expected_writer = {
        "architecture": "pi05_layer_matched_memory_program_compiler_v1",
        "generated_adapter": "complete_pi05_task_specific_rank16_lora",
        "frame_stride": 5,
        "include_final_frame": True,
        "backbone_total_frames_per_condition": 420,
        "maximum_stride5_frames_per_video": 105,
        "last_encoder_call": "zero_pad_to_fixed_shape_then_slice",
        "program_width": 256,
        "memory_token_count": 16,
        "memory_topology": "18_action_expert_layers_x_16_lora_rank_coordinates",
        "directed_channel": "half_natural_minus_reverse_parameter_memory",
        "video_set": "address_preserving_deepsets_consensus_k1_exact_identity",
        "m2p": "same_20x16_grid_two_block_group_rank_axial_attention",
        "factor_decoder": "eight_jointly_trained_native_rank16_factor_heads",
        "step0_contract": "fresh_native_A0_B0_source_policy_identity",
        "image_width": 2048,
        "expert_width": 1024,
        "text_meta_lora_rank": 4,
        "vl_meta_lora_rank": 4,
        "vl_meta_lora_trainable": False,
        "action_meta_lora_rank": 4,
        "patch_grounding_heads": 8,
        "action_horizon": 50,
        "padded_action_dim": 32,
        "semantic_core_heads": 8,
        "semantic_core_blocks": 2,
        "procedure_heads": 8,
        "procedure_blocks": 2,
        "visual_transition_heads": 8,
        "memory_reader_heads": 8,
        "m2p_heads": 8,
        "m2p_blocks": 2,
        "factor_hidden_width": 256,
        "matching_margin": 0.2,
        "initialization_seed": 7,
        "activation_checkpointing": True,
    }
    data = config.get("data", {})
    training = config.get("conditioning_training", {})
    distributed = config.get("optimization", {}).get("distributed", {})
    matching = training.get("program_matching", {})
    changed = (
        any(writer.get(key) != value for key, value in expected_writer.items())
        or int(writer.get("max_frames_per_encoder_call", 0)) <= 0
        or data.get("task_count") != 24
        or data.get("episodes_per_task") != 50
        or data.get("demo_indices") != [0, 49]
        or data.get("action_chunk_size") != 50
        or data.get("action_queries_per_task") != 20
        or data.get("dynamic_k_max") != 4
        or data.get("dynamic_k_balance")
        != "exactly_six_tasks_at_each_K_per_macro"
        or training.get("global_tasks_per_optimizer_update") != 24
        or training.get("update_topology")
        != "one_complete_full24_equal_task_mean_per_macro"
        or training.get("task_assignment")
        != "cost_balanced_long_first_dynamic_uneven"
        or training.get("pair_loss_reduction")
        != "mean_within_task_then_equal_mean_over_24_tasks"
        or matching.get("kind")
        != (
            "exact_language_correct_vs_reverse_shuffle_margin_plus_"
            "same_task_video_agreement"
        )
        or float(matching.get("weight", 0.0)) <= 0
        or distributed.get("fresh_world_sizes") != [1, 2, 3, 4, 5, 6]
        or distributed.get("exact_resume_world_topology_locked") is not True
        or distributed.get("nccl_p2p_disable") != "1"
        or distributed.get("deferred_process_group") is not True
    )
    if changed:
        raise WriterModelError("LMMPC scientific contract changed")


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
            raise WriterModelError("LMMPC runtime contract changed")
        parse_macro_boundaries(cell["checkpoint_macros"], total)
    formal = config["formal_run"]
    if formal.get("status") not in {"unsealed_pending_live_profile", "sealed"}:
        raise WriterModelError("LMMPC formal status changed")
    if formal.get("status") == "sealed":
        evidence = formal.get("profile_evidence", {})
        if (
            not isinstance(evidence.get("source_commit"), str)
            or len(evidence["source_commit"]) != 40
            or int(evidence.get("world_size", 0)) not in range(1, 7)
            or int(evidence.get("completion_macro", 0)) != 2
            or float(evidence.get("macro_seconds", 0)) <= 0
            or int(evidence.get("max_cuda_allocated_bytes", 0)) <= 0
            or evidence.get("global_k_histogram")
            != {"1": 6, "2": 6, "3": 6, "4": 6}
            or float(evidence.get("program_matching_weight", 0)) <= 0
            or int(evidence.get("native_context_calls", -1))
            != int(evidence.get("expected_native_context_calls", -2))
            or float(evidence.get("correct_reverse_directed_relative_l2", 0))
            <= 0
            or float(evidence.get("constant_effective_ba_max_abs", 1)) >= 1e-6
        ):
            raise WriterModelError("sealed LMMPC live profile evidence changed")


def load_writer_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != AS_WRITER_CONFIG_SCHEMA:
        raise WriterModelError("unsupported LMMPC Writer config schema")
    writer_stage(config)
    _validate_authorities(config)
    _validate_method(config)
    _validate_runtime(config)
    return config


def resolve_mode_config(config: Mapping[str, Any], mode: str) -> dict[str, Any]:
    if mode not in {"profile", "formal"}:
        raise WriterModelError("unsupported LMMPC runtime mode")
    if mode == "formal" and config["formal_run"]["status"] != "sealed":
        raise WriterModelError("formal LMMPC training awaits a sealed live profile")
    return dict(config)
