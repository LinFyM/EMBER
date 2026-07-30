"""Single-source declarative contract for the canonical EMBER Writer."""

from __future__ import annotations

from typing import Any, Mapping


LOOM_WRITER_PARAMETER_COUNT = 12_855_552
ACTION_PROBE_POSITIONS = (0, 7, 14, 21, 28, 35, 42, 49)

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
        "action_probe_positions",
        "semantic_core_heads",
        "semantic_core_blocks",
        "visual_effect_heads",
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
    "action_probe_positions": ACTION_PROBE_POSITIONS,
    "semantic_core_heads": 8,
    "semantic_core_blocks": 2,
    "visual_effect_heads": 8,
    "procedure_heads": 8,
    "procedure_blocks": 2,
    "fusion_heads": 8,
    "factor_hidden_width": 256,
}

_STATIC_WRITER_CONTRACT: dict[str, Any] = {
    "architecture": "pi05_task_grounded_teacher_policy_gap_writer_loom",
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
    "action_stream": "eight_sparse_probe_policy_imitation_value_memory",
    "teacher_semantic_relation": (
        "adjacent_task_grounded_midpoint_plus_difference_zero_at_no_change"
    ),
    "teacher_visual_relation": (
        "bidirectional_patch_correspondence_same_grid_matched_displacement"
    ),
    "teacher_relation_confidence": (
        "bounded_change_times_task_relevance_times_mutual_match_times_"
        "nonuniform_matcher_without_action_input"
    ),
    "teacher_events": (
        "three_deterministic_backbones_plus_five_learned_relation_events"
    ),
    "visual_effect_heads": 8,
    "procedure_heads": 8,
    "procedure_blocks": 2,
    "procedure_attention": (
        "shared_dual_stream_axial_local_slots_then_slotwise_causal_time"
    ),
    "procedure_position_encoding": (
        "action_at_twice_frame_ordinal_teacher_at_adjacent_ordinal_sum_qk_only"
    ),
    "procedure_value_path": (
        "strictly_separate_teacher_event_and_policy_imitation_memories"
    ),
    "procedure_initialization": "normal_nonzero",
    "query_count": 320,
    "routing_identity": "query_module_layer_rank_qk_only",
    "core_slot_reader": "routing_qk_core_content_v",
    "teacher_slot_reader": (
        "routing_plus_core_q_teacher_procedure_kv_and_same_weight_confidence"
    ),
    "policy_slot_reader": (
        "routing_plus_core_plus_teacher_q_policy_imitation_kv"
    ),
    "teacher_policy_gap": (
        "separate_full_rank_alignment_rmsnorm_teacher_minus_policy"
    ),
    "adaptation_strength": (
        "teacher_confidence_times_bounded_gap_magnitude"
    ),
    "core_procedure_first_interaction": "teacher_policy_gap_compiler_only",
    "slot_fusion": (
        "gap_content_plus_gap_gated_core_support_then_content_only_coordination"
    ),
    "fusion_heads": 8,
    "post_fusion_blocks": 1,
    "core_only_public_lora_delta": "exact_zero_when_confidence_or_gap_zero",
    "action_only_public_lora_delta": "exact_zero_when_teacher_confidence_zero",
    "final_factor_scale": (
        "per_slot_confidence_times_gap_reapplied_after_factor_head"
    ),
    "factor_head_bias": False,
    "factor_hidden_width": 256,
    "initialization_seed": 7,
}


def expected_writer_contract(writer: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact Loom config payload, preserving frame chunking."""

    return {
        **_STATIC_WRITER_CONTRACT,
        "frame_stride": writer["frame_stride"],
        "max_frames_per_encoder_call": writer["max_frames_per_encoder_call"],
    }


def validate_writer_dimensions(observed: Mapping[str, Any]) -> None:
    """Reject constructor values outside the one canonical Loom topology."""

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
