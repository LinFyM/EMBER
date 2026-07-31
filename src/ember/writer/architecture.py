"""Single-source declarative contract for the canonical SPG Writer."""

from __future__ import annotations

from typing import Any, Mapping


SPG_WRITER_PARAMETER_COUNT = 10_633_216

SEMANTIC_PROGRAM_WRITER_CONSTRUCTOR_KEYS = frozenset(
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
        "program_heads",
        "program_blocks",
        "compiler_heads",
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
    "program_heads": 8,
    "program_blocks": 2,
    "compiler_heads": 8,
    "factor_hidden_width": 256,
}

_STATIC_WRITER_CONTRACT: dict[str, Any] = {
    "architecture": "pi05_semantic_program_grid_target_rank_compiler_v1",
    "generated_adapter": "complete_pi05_task_specific_rank16_lora",
    "camera_dataset": "obs/agentview_rgb",
    "camera_transform": "libero_opengl_rotate_180_chw_uint8",
    "include_final_frame": True,
    "teacher_prompt": "Task: {cleaned_task};\nAction: ",
    "teacher_state_input": False,
    "task_span_extraction": "authoritative_full_prompt_sentencepiece_piece_offsets",
    "text_branch_input": "bos_plus_exact_authoritative_task_span_ids_without_template",
    "task_token_alignment": "text_and_multimodal_ids_identical_by_construction",
    "image_width": 2048,
    "native_image_tokens": 256,
    "multimodal_core_value": "M_plus_task_queried_patch_G",
    "shared_language_projection": "bias_free_2048_to_256",
    "patch_grounding_attention": "per_frame_Q_text_to_256_raw_patch_values",
    "patch_grounding_qk": "separate_pre_rmsnorm_bias_free_256_to_256",
    "patch_grounding_value": "raw_shared_projected_image_position_content",
    "patch_grounding_output": "bias_free_256_to_256",
    "patch_grounding_heads": 8,
    "expert_width": 1024,
    "text_meta_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "text_meta_lora_rank": 4,
    "vl_meta_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "vl_meta_lora_rank": 4,
    "action_meta_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "action_meta_lora_rank": 4,
    "frame_batching_contract": "encode_one_video_with_unpadded_memory_safety_chunks",
    "activation_checkpointing": True,
    "action_horizon": 50,
    "padded_action_dim": 32,
    "action_expert_probe": "one_forward_fixed_persistent_gaussian_suffix_at_t1",
    "action_expert_action_out": False,
    "interaction_reduction": "mean_50_final_suffix_hidden_then_1024_to_256",
    "program_width": 256,
    "semantic_core_frame_fusion": "mean_backbone_plus_task_selected_centered_residual",
    "semantic_core_frame_order": "permutation_invariant",
    "semantic_core_heads": 8,
    "semantic_core_blocks": 2,
    "semantic_core_position_encoding": "task_token_ordinal_rope_qk_only",
    "program_grid": "interval_Action_plus_task_token_patch_change",
    "program_interval_alignment": "A_f_with_G_f_plus_1_minus_G_f",
    "program_temporal_ordinal": "interval_endpoint_sampled_frame_position",
    "program_heads": 8,
    "program_blocks": 2,
    "program_attention": "interval_local_then_column_causal_axial",
    "program_identity_path": "frame_type_and_token_ordinal_qk_only",
    "program_value_path": "raw_content_without_identity_or_terminal_norm",
    "target_count": 38,
    "public_rank": 16,
    "compiler_order": "sealed_policy_target_first_then_rank_last",
    "core_target_reader": "target_qk_core_raw_value",
    "program_coordinate_reader": "target_rank_core_qk_program_raw_value",
    "coordinate_mixer": "rank_axis_then_target_axis_raw_value_residual",
    "coordinate_identity_path": "target_and_rank_qk_only",
    "compiler_heads": 8,
    "factor_head_bias": False,
    "factor_hidden_width": 256,
    "factor_final_projection": "exact_zero_initialization",
    "initialization_seed": 7,
}


def expected_writer_contract(writer: Mapping[str, Any]) -> dict[str, Any]:
    """Return exact SPG payload while preserving profiled frame chunking."""

    return {
        **_STATIC_WRITER_CONTRACT,
        "frame_stride": writer["frame_stride"],
        "max_frames_per_encoder_call": writer["max_frames_per_encoder_call"],
    }


def validate_writer_dimensions(observed: Mapping[str, Any]) -> None:
    """Reject constructor values outside the canonical SPG topology."""

    changed = {
        name: (WRITER_DIMENSION_CONTRACT[name], observed.get(name))
        for name in WRITER_DIMENSION_CONTRACT
        if observed.get(name) != WRITER_DIMENSION_CONTRACT[name]
    }
    if changed:
        raise ValueError(f"invalid EMBER SPG Writer dimensions: {changed}")
