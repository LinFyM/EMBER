"""Single-source declarative contract for the canonical EMBER Writer."""

from __future__ import annotations

from typing import Any, Mapping


PRIOR_INNOVATION_WRITER_PARAMETER_COUNT = 10_643_968

PRIOR_INNOVATION_WRITER_CONSTRUCTOR_KEYS = frozenset(
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
    "architecture": "pi05_semantic_prior_ordered_procedure_innovation_writer",
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
    "patch_grounding_qk": "separate_pre_rmsnorm_bias_free_256_to_256",
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
        "uncapped_bias_free_qko_attention_residual"
    ),
    "visual_transition_heads": 8,
    "procedure_heads": 8,
    "procedure_blocks": 2,
    "procedure_attention": "single_global_causal_content_transformer",
    "procedure_position_encoding": "sampled_frame_ordinal_rope_qk_only",
    "procedure_value_path": (
        "native_action_probe_plus_uncapped_task_grounded_visual_transition"
    ),
    "query_count": 320,
    "routing_identity": "query_module_layer_rank_qk_only",
    "core_slot_reader": (
        "routing_query_normalized_core_key_raw_core_value_"
        "learned_bias_free_qkvo"
    ),
    "semantic_prior": "rmsnorm_of_routed_raw_core_slot",
    "procedure_time_centering": (
        "fp32_masked_valid_frame_mean_then_cast_to_input_dtype"
    ),
    "procedure_slot_reader": (
        "semantic_prior_query_only_no_routing_rope_normalized_raw_procedure_key_"
        "learned_centered_procedure_value_bias_free_qkvo"
    ),
    "core_procedure_first_interaction": (
        "semantic_prior_queries_ordered_centered_procedure_innovation"
    ),
    "slot_fusion": "direct_semantic_prior_plus_procedure_innovation",
    "fusion_heads": 8,
    "post_fusion_blocks": 1,
    "post_fusion_slot_block": (
        "bias_free_full_qkvo_residual_attention_ffn_"
        "routing_qk_only_content_v_only_final_rmsnorm"
    ),
    "post_fusion_scale_contract": (
        "final_rmsnorm_stable_factor_head_interface_without_branch_scalar"
    ),
    "core_only_public_lora_delta": "allowed_semantic_prior_contribution",
    "procedure_only_public_lora_delta": "exact_zero",
    "zero_procedure_public_lora_delta": "semantic_prior_only",
    "constant_nonzero_procedure": (
        "zero_innovation_with_semantic_prior_preserved"
    ),
    "factor_head_bias": False,
    "factor_hidden_width": 256,
    "initialization_seed": 7,
}


def expected_writer_contract(writer: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact Prior-Innovation payload, preserving frame chunking."""

    return {
        **_STATIC_WRITER_CONTRACT,
        "frame_stride": writer["frame_stride"],
        "max_frames_per_encoder_call": writer["max_frames_per_encoder_call"],
    }


def validate_writer_dimensions(observed: Mapping[str, Any]) -> None:
    """Reject values outside the one canonical Prior-Innovation topology."""

    changed = {
        name: (WRITER_DIMENSION_CONTRACT[name], observed.get(name))
        for name in WRITER_DIMENSION_CONTRACT
        if observed.get(name) != WRITER_DIMENSION_CONTRACT[name]
    }
    if changed:
        raise ValueError(f"invalid EMBER Writer dimensions: {changed}")
