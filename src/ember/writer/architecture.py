"""Single-source declarative contract for the canonical EMBER Writer."""

from __future__ import annotations

from typing import Any, Mapping


V7_WRITER_PARAMETER_COUNT = 10_312_192
ACTION_PROBE_POSITIONS = (0, 7, 14, 21, 28, 35, 42, 49)

LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS = frozenset(
    {
        "image_width",
        "expert_width",
        "program_width",
        "vl_meta_lora_rank",
        "action_meta_lora_rank",
        "patch_grounding_heads",
        "max_frames_per_encoder_call",
        "action_horizon",
        "padded_action_dim",
        "action_probe_positions",
        "semantic_core_heads",
        "semantic_core_blocks",
        "action_effect_heads",
        "procedure_heads",
        "procedure_blocks",
        "fusion_heads",
        "factor_hidden_width",
        "initialization_seed",
        "activation_checkpointing",
    }
)

WRITER_DIMENSION_CONTRACT = {
    "image_width": 2048,
    "expert_width": 1024,
    "program_width": 256,
    "vl_meta_lora_rank": 4,
    "action_meta_lora_rank": 4,
    "patch_grounding_heads": 8,
    "action_horizon": 50,
    "padded_action_dim": 32,
    "action_probe_positions": ACTION_PROBE_POSITIONS,
    "semantic_core_heads": 8,
    "semantic_core_blocks": 2,
    "action_effect_heads": 8,
    "procedure_heads": 8,
    "procedure_blocks": 3,
    "fusion_heads": 8,
    "factor_hidden_width": 256,
}

_STATIC_WRITER_CONTRACT: dict[str, Any] = {
    "architecture": (
        "pi05_task_aligned_semantic_trajectory_action_effect_"
        "causal_program_procedure_content_compiler_v7"
    ),
    "generated_adapter": "complete_pi05_task_specific_rank16_lora",
    "camera_dataset": "obs/agentview_rgb",
    "camera_transform": "libero_opengl_rotate_180_chw_uint8",
    "include_final_frame": True,
    "teacher_prompt": "Task: {cleaned_task};\nAction: ",
    "teacher_state_input": False,
    "task_span_extraction": (
        "authoritative_full_prompt_sentencepiece_piece_offsets"
    ),
    "task_token_alignment": (
        "same_multimodal_task_span_ordinal_across_video_frames"
    ),
    "image_width": 2048,
    "native_image_tokens": 256,
    "multimodal_task_evidence": (
        "final_norm_task_span_hidden_from_same_image_language_prefix"
    ),
    "shared_semantic_projection": "bias_free_2048_to_256",
    "stable_task_query": (
        "valid_frame_mean_of_multimodal_task_evidence_per_task_token"
    ),
    "patch_grounding_attention": (
        "stable_multimodal_task_queries_to_each_frames_256_image_positions"
    ),
    "patch_grounding_qk": "separate_pre_rmsnorm_bias_free_256_to_256",
    "patch_grounding_value": (
        "raw_shared_projected_image_position_content_no_value_projection"
    ),
    "patch_grounding_output": (
        "bias_free_256_to_256_added_to_multimodal_task_token_evidence"
    ),
    "patch_grounding_heads": 8,
    "expert_width": 1024,
    "vl_meta_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "vl_meta_lora_rank": 4,
    "action_meta_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "action_meta_lora_rank": 4,
    "frame_batching_contract": (
        "encode_one_video_with_unpadded_memory_safety_chunks"
    ),
    "activation_checkpointing": True,
    "action_horizon": 50,
    "padded_action_dim": 32,
    "action_probe_positions": list(ACTION_PROBE_POSITIONS),
    "action_expert_probe": (
        "one_forward_eight_sparse_native_position_fixed_gaussian_suffix_at_t1"
    ),
    "action_expert_action_out": False,
    "action_probe_projection": (
        "each_final_suffix_hidden_shared_bias_free_1024_to_256_no_mean"
    ),
    "program_width": 256,
    "semantic_core_aggregation": (
        "parameter_free_valid_frame_mean_of_task_aligned_trajectory"
    ),
    "semantic_core_order_contract": (
        "strict_frame_set_permutation_invariance_without_frame_position"
    ),
    "semantic_core_heads": 8,
    "semantic_core_blocks": 2,
    "semantic_core_position_encoding": (
        "task_token_ordinal_rope_qk_only_bidirectional"
    ),
    "semantic_core_value_path": (
        "multimodal_task_token_plus_task_queried_patch_content"
    ),
    "action_effect_source": (
        "forward_difference_of_rms_normalized_task_aligned_trajectory_"
        "in_actual_arm_input_order"
    ),
    "action_effect_interval": "frame_f_to_frame_f_plus_one",
    "action_effect_attention": (
        "joint_softmax_over_all_eight_by_task_token_action_effect_pairs"
    ),
    "action_effect_qk": "separate_pre_rmsnorm_bias_free_256_to_256",
    "action_effect_value": (
        "raw_forward_semantic_change_modulated_by_zero_initialized_"
        "bias_free_action_feature_gate_no_value_projection"
    ),
    "action_effect_output": (
        "one_high_level_event_per_frame_interval_bias_free_256_to_256"
    ),
    "action_effect_heads": 8,
    "procedure_heads": 8,
    "procedure_blocks": 3,
    "procedure_attention": "global_causal_pre_norm_with_valid_interval_mask",
    "procedure_position_encoding": (
        "one_dimensional_rope_on_interval_start_sampled_frame_ordinal_qk_only"
    ),
    "procedure_value_path": (
        "procedure_only_action_effect_event_no_action_or_core_residual"
    ),
    "procedure_initialization": "normal_nonzero",
    "query_count": 320,
    "routing_identity": "query_module_layer_rank_qk_only",
    "core_slot_reader": "routing_qk_core_content_v",
    "procedure_slot_reader": (
        "rmsnorm_routing_plus_core_read_q_ordered_procedure_content_v"
    ),
    "core_procedure_first_interaction": "lora_compiler_only",
    "slot_fusion": (
        "procedure_readout_then_one_content_only_post_fusion_block"
    ),
    "fusion_heads": 8,
    "post_fusion_blocks": 1,
    "core_only_public_lora_delta": "exact_zero_by_structure",
    "factor_head_bias": False,
    "factor_hidden_width": 256,
    "initialization_seed": 7,
}


def expected_writer_contract(writer: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact v7 config payload, preserving profiled frame chunking."""

    return {
        **_STATIC_WRITER_CONTRACT,
        "frame_stride": writer["frame_stride"],
        "max_frames_per_encoder_call": writer["max_frames_per_encoder_call"],
    }


def validate_writer_dimensions(observed: Mapping[str, Any]) -> None:
    """Reject constructor values outside the one canonical v7 topology."""

    def normalized(name: str, value: Any) -> Any:
        if name == "action_probe_positions" and value is not None:
            return tuple(int(item) for item in value)
        return value

    changed = {
        name: (
            WRITER_DIMENSION_CONTRACT[name],
            normalized(name, observed.get(name)),
        )
        for name in WRITER_DIMENSION_CONTRACT
        if normalized(name, observed.get(name)) != WRITER_DIMENSION_CONTRACT[name]
    }
    if changed:
        raise ValueError(f"invalid EMBER Writer dimensions: {changed}")
