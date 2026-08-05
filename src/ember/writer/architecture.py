"""Single-source declarative contract for the canonical EMBER Writer."""

from __future__ import annotations

from typing import Any, Mapping


POLICY_WIDE_ATOM_WRITER_PARAMETER_COUNT = 13_033_728

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
        "policy_coordinate_heads",
        "policy_atom_count",
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
    "policy_coordinate_heads": 8,
    "policy_atom_count": 64,
}

_STATIC_WRITER_CONTRACT: dict[str, Any] = {
    "architecture": (
        "pi05_task_grounded_semantic_set_visual_transition_"
        "causal_procedure_slot_fusion_v6"
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
    "patch_grounding_qk": "separate_pre_rmsnorm_bias_free_256_to_256",
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
    "frame_batching_contract": (
        "encode_one_video_with_unpadded_memory_safety_chunks"
    ),
    "activation_checkpointing": True,
    "action_horizon": 50,
    "padded_action_dim": 32,
    "action_expert_probe": "one_forward_fixed_persistent_gaussian_suffix_at_t1",
    "action_expert_action_out": False,
    "interaction_reduction": (
        "mean_50_final_suffix_hidden_then_shared_bias_free_1024_to_256"
    ),
    "program_width": 256,
    "semantic_set_fusion": (
        "valid_frame_mean_backbone_plus_task_selected_centered_residual"
    ),
    "semantic_set_query": "text_only_task_query_q_only",
    "semantic_set_qk": "separate_pre_rmsnorm_bias_free_256_to_256",
    "semantic_set_mean": "bias_free_256_to_256_valid_frame_mean_backbone",
    "semantic_set_value": "raw_frame_centered_evidence_no_value_projection",
    "semantic_set_output": (
        "bias_free_256_to_256_centered_residual_added_to_mean_backbone"
    ),
    "semantic_set_order_contract": (
        "strict_frame_permutation_invariance_without_frame_position"
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
    "visual_transition_padding": "invalid_task_tokens_and_frames_exact_zero",
    "visual_transition_attention": (
        "action_expert_probe_queries_task_token_aligned_visual_transition"
    ),
    "visual_transition_qk": "separate_pre_rmsnorm_bias_free_256_to_256",
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
    "factor_hidden_width": 256,
    "initialization_seed": 7,
}

_POLICY_WIDE_ATOM_WRITER_CONTRACT = {
    **_STATIC_WRITER_CONTRACT,
    "architecture": "pi05_policy_wide_atom_dictionary_writer_v1",
    "policy_coordinate_count": 16,
    "policy_atom_count": 64,
    "policy_atom_scope": (
        "one_shared_atom_index_spans_all_38_policy_targets"
    ),
    "policy_atom_storage": (
        "target_private_a_and_b_vectors_with_shared_policy_atom_index"
    ),
    "policy_coordinate_core_read": (
        "16_queries_read_raw_semantic_core_with_independent_attention"
    ),
    "policy_coordinate_procedure_read": (
        "core_conditioned_queries_read_raw_causal_procedure_with_independent_attention"
    ),
    "policy_atom_mixing": (
        "separate_signed_rank16_by_atom64_a_and_b_dot_product_matrices"
    ),
    "public_lora_a": "sealed_template_plus_m_a_times_zero_initialized_a_atoms",
    "public_lora_b": "zero_initialized_b_atoms_times_m_b_transpose",
    "policy_coordinate_heads": 8,
    "dictionary_softmax": False,
    "dictionary_top_k": False,
    "dictionary_task_ids": False,
    "dictionary_initialization": "all_a_and_b_atoms_exact_zero",
}
for _retired_key in (
    "query_count",
    "routing_identity",
    "core_slot_reader",
    "procedure_slot_reader",
    "slot_fusion",
    "fusion_heads",
    "procedure_value_centering",
    "modulation_projection",
    "post_fusion_blocks",
    "factor_head_bias",
    "factor_hidden_width",
):
    _POLICY_WIDE_ATOM_WRITER_CONTRACT.pop(_retired_key)


def expected_writer_contract(writer: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact architecture payload, preserving profiled chunking."""

    architecture = writer.get("architecture")
    if architecture == _STATIC_WRITER_CONTRACT["architecture"]:
        static = _STATIC_WRITER_CONTRACT
    elif architecture == _POLICY_WIDE_ATOM_WRITER_CONTRACT["architecture"]:
        static = _POLICY_WIDE_ATOM_WRITER_CONTRACT
    else:
        raise ValueError(f"unsupported EMBER Writer architecture: {architecture}")

    return {
        **static,
        "frame_stride": writer["frame_stride"],
        "max_frames_per_encoder_call": writer["max_frames_per_encoder_call"],
    }


def validate_writer_dimensions(observed: Mapping[str, Any]) -> None:
    """Reject constructor values outside the canonical Writer topology."""

    changed = {
        name: (WRITER_DIMENSION_CONTRACT[name], observed.get(name))
        for name in WRITER_DIMENSION_CONTRACT
        if observed.get(name) != WRITER_DIMENSION_CONTRACT[name]
    }
    if changed:
        raise ValueError(f"invalid EMBER Writer dimensions: {changed}")
