"""Declarative contract for the K4 phase-aligned Language-Axial Writer."""

from __future__ import annotations

from typing import Any, Mapping


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
        "procedure_heads",
        "procedure_blocks",
        "visual_transition_heads",
        "fusion_heads",
        "factor_hidden_width",
        "videos_per_condition",
        "phase_slots",
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
    "procedure_heads": 8,
    "procedure_blocks": 2,
    "visual_transition_heads": 8,
    "fusion_heads": 8,
    "factor_hidden_width": 256,
    "videos_per_condition": 4,
    "phase_slots": 16,
}

_STATIC_WRITER_CONTRACT: dict[str, Any] = {
    "architecture": "pi05_k4_phase_aligned_language_axial_semantic_procedure_v1",
    "generated_adapter": "one_complete_pi05_task_specific_rank16_lora",
    "camera_dataset": "obs/agentview_rgb",
    "camera_transform": "libero_opengl_rotate_180_chw_uint8",
    "include_final_frame": True,
    "teacher_prompt": "Task: {cleaned_task};\nAction: ",
    "teacher_state_input": False,
    "videos_per_condition": 4,
    "video_set_order": "permutation_invariant_without_shot_identity",
    "task_span_extraction": "authoritative_full_prompt_sentencepiece_piece_offsets",
    "text_branch_input": "bos_plus_exact_authoritative_task_span_ids_without_template",
    "task_token_alignment": "text_and_multimodal_ids_identical_by_construction",
    "image_width": 2048,
    "native_image_tokens": 256,
    "multimodal_core_value": "final_norm_task_span_hidden_plus_task_queried_image_position_content",
    "shared_language_projection": "bias_free_2048_to_256",
    "patch_grounding_attention": "per_frame_text_only_task_queries_to_256_image_positions",
    "patch_grounding_heads": 8,
    "expert_width": 1024,
    "text_meta_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "text_meta_lora_rank": 4,
    "vl_meta_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "vl_meta_lora_rank": 4,
    "action_meta_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "action_meta_lora_rank": 4,
    "frame_batching_contract": "four_videos_serial_memory_safety_chunks",
    "activation_checkpointing": True,
    "action_horizon": 50,
    "padded_action_dim": 32,
    "action_expert_probe": "one_forward_fixed_persistent_gaussian_suffix_at_t1",
    "interaction_reduction": "mean_50_final_suffix_hidden_then_shared_bias_free_1024_to_256",
    "program_width": 256,
    "phase_alignment": "each_video_differentiable_linear_resample_to_16_normalized_progress_slots",
    "phase_slots": 16,
    "semantic_set_fusion": "task_selected_union_of_four_phase_aligned_video_evidence_sets",
    "semantic_set_order_contract": "strict_video_set_and_frame_set_permutation_invariance",
    "semantic_core_heads": 8,
    "semantic_core_blocks": 2,
    "semantic_core_position_encoding": "task_token_ordinal_rope_qk_only_bidirectional",
    "procedure_heads": 8,
    "procedure_blocks": 2,
    "procedure_attention": "per_video_global_causal_then_equal_mean_across_four_aligned_videos",
    "procedure_position_encoding": "normalized_phase_slot_ordinal_rope_qk_only",
    "procedure_value_path": "action_expert_probe_plus_task_grounded_adjacent_visual_transition",
    "visual_transition_source": "within_video_adjacent_difference_after_phase_alignment_never_across_video_boundaries",
    "visual_transition_heads": 8,
    "query_count": 320,
    "routing_identity": "query_module_layer_rank_qk_only",
    "core_slot_reader": "routing_qk_core_content_v",
    "procedure_slot_reader": "routing_plus_normalized_core_q_centered_procedure_v",
    "slot_fusion": "zero_initialized_bias_free_adaln_then_one_post_fusion_block",
    "fusion_heads": 8,
    "procedure_value_centering": "parameter_free_valid_phase_mean",
    "modulation_projection": "bias_free_256_to_512_zero_initialized",
    "post_fusion_blocks": 1,
    "factor_head_bias": False,
    "factor_hidden_width": 256,
    "step0_identity": "template_A_plus_all_dynamic_zero_B",
    "initialization_seed": 7,
}


def expected_writer_contract(writer: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact K4 phase-aligned v6-derived Writer payload."""

    if writer.get("architecture") != _STATIC_WRITER_CONTRACT["architecture"]:
        raise ValueError(f"unsupported EMBER Writer architecture: {writer.get('architecture')}")
    return {
        **_STATIC_WRITER_CONTRACT,
        "frame_stride": writer["frame_stride"],
        "max_frames_per_encoder_call": writer["max_frames_per_encoder_call"],
    }


def validate_writer_dimensions(observed: Mapping[str, Any]) -> None:
    """Reject constructor values outside the canonical K4 phase-aligned topology."""

    changed = {
        name: (expected, observed.get(name))
        for name, expected in WRITER_DIMENSION_CONTRACT.items()
        if observed.get(name) != expected
    }
    if changed:
        raise ValueError(f"invalid EMBER Writer dimensions: {changed}")
