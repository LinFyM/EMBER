"""Single-source declarative contract for the canonical EMBER Writer."""

from __future__ import annotations

from typing import Any, Mapping


RECENTER_WRITER_PARAMETER_COUNT = 10_709_248

LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS = frozenset(
    {
        "image_width",
        "expert_width",
        "program_width",
        "text_meta_lora_rank",
        "vl_meta_lora_rank",
        "action_meta_lora_rank",
        "patch_grounding_heads",
        "max_frames_per_encoder_call",
        "action_horizon",
        "padded_action_dim",
        "semantic_core_heads",
        "semantic_core_blocks",
        "visual_transition_heads",
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
    "text_meta_lora_rank": 4,
    "vl_meta_lora_rank": 4,
    "action_meta_lora_rank": 4,
    "patch_grounding_heads": 8,
    "action_horizon": 50,
    "padded_action_dim": 32,
    "semantic_core_heads": 8,
    "semantic_core_blocks": 2,
    "visual_transition_heads": 8,
    "procedure_heads": 8,
    "procedure_blocks": 2,
    "fusion_heads": 8,
    "factor_hidden_width": 256,
}

_STATIC_WRITER_CONTRACT: dict[str, Any] = {
    "architecture": (
        "pi05_action_anchored_core_keyed_centered_procedure_writer_recenter"
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
    "text_only_task_query": (
        "frozen_gemma_plus_rank4_text_meta_lora_task_span_hidden"
    ),
    "text_meta_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "text_meta_lora_rank": 4,
    "multimodal_task_evidence": (
        "final_norm_task_span_hidden_from_same_image_language_prefix"
    ),
    "shared_semantic_projection": "bias_free_2048_to_256",
    "patch_grounding_attention": (
        "text_only_task_queries_to_each_frames_256_image_positions"
    ),
    "patch_grounding_qk": (
        "separate_pre_rmsnorm_bias_free_256_to_256"
    ),
    "patch_grounding_value": (
        "raw_shared_projected_image_position_content_no_value_projection"
    ),
    "patch_grounding_output": "bias_free_256_to_256",
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
    "action_expert_probe": (
        "one_forward_native_50_suffix_hidden_mean_fixed_gaussian_at_t1"
    ),
    "action_expert_action_out": False,
    "action_probe_projection": (
        "mean_all_50_final_suffix_hidden_then_bias_free_1024_to_256"
    ),
    "program_width": 256,
    "semantic_core_aggregation": (
        "stable_frame_mean_plus_text_selected_centered_frame_residual"
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
    "visual_transition": (
        "adjacent_task_grounded_patch_difference_recomputed_in_actual_order"
    ),
    "visual_transition_key": (
        "zero_preserving_rmsnorm_then_bias_free_projection"
    ),
    "visual_transition_value": (
        "raw_task_grounded_difference_without_value_projection"
    ),
    "visual_transition_residual": (
        "radial_cap_at_one_quarter_stopgrad_action_probe_rms"
    ),
    "visual_transition_heads": 8,
    "procedure_heads": 8,
    "procedure_blocks": 2,
    "procedure_attention": "single_global_causal_content_transformer",
    "procedure_position_encoding": "sampled_frame_ordinal_rope_qk_only",
    "procedure_value_path": (
        "native_action_probe_plus_bounded_visual_transition_residual"
    ),
    "query_count": 320,
    "routing_identity": "query_module_layer_rank_qk_only",
    "core_slot_reader": "routing_qk_core_content_v",
    "procedure_slot_reader": (
        "routing_plus_core_q_full_normalized_procedure_k_"
        "raw_time_centered_procedure_v"
    ),
    "core_procedure_first_interaction": "core_keyed_procedure_compiler_only",
    "slot_fusion": (
        "procedure_content_times_one_plus_quarter_tanh_core_modulation"
    ),
    "core_modulation_range": [0.75, 1.25],
    "fusion_heads": 8,
    "post_fusion_blocks": 1,
    "post_fusion_scale_contract": (
        "direction_mixing_then_restore_prefusion_slot_rms"
    ),
    "core_only_public_lora_delta": "exact_zero",
    "constant_procedure_public_lora_delta": "exact_zero",
    "factor_head_bias": False,
    "factor_hidden_width": 256,
    "initialization_seed": 7,
}


def expected_writer_contract(writer: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact Recenter config payload, preserving frame chunking."""

    return {
        **_STATIC_WRITER_CONTRACT,
        "frame_stride": writer["frame_stride"],
        "max_frames_per_encoder_call": writer["max_frames_per_encoder_call"],
    }


def validate_writer_dimensions(observed: Mapping[str, Any]) -> None:
    """Reject constructor values outside the one canonical Recenter topology."""

    changed = {
        name: (WRITER_DIMENSION_CONTRACT[name], observed.get(name))
        for name in WRITER_DIMENSION_CONTRACT
        if observed.get(name) != WRITER_DIMENSION_CONTRACT[name]
    }
    if changed:
        raise ValueError(f"invalid EMBER Writer dimensions: {changed}")
