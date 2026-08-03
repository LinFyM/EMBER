"""Single-source contract for the semantic direction-store Writer."""

from __future__ import annotations

import math
from typing import Any, Mapping


SEMANTIC_DIRECTION_STORE_WRITER_PARAMETER_COUNT = 37_355_776

SEMANTIC_DIRECTION_STORE_WRITER_CONSTRUCTOR_KEYS = frozenset(
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
        "direction_store_count",
        "direction_store_top_k",
        "direction_store_centers",
        "direction_anchor_mean",
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
    "direction_store_count": 8,
    "direction_store_top_k": 2,
}

_STATIC_WRITER_CONTRACT: dict[str, Any] = {
    "architecture": "pi05_semantic_direction_store_program_v1",
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
    "absolute_semantic_value": "X_f_equals_M_f_plus_task_queried_patch_G_f",
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
    "semantic_core": "mean_X_plus_task_selected_centered_X_then_token_axis",
    "semantic_core_frame_order": "none_strict_frame_set_permutation_invariant",
    "semantic_core_heads": 8,
    "semantic_core_blocks": 2,
    "program_grid": "38_target_bound_outgoing_Action_Effect_Change_role_streams",
    "program_interval_alignment": "A_f_with_G_f_plus_1_and_G_f_plus_1_minus_G_f",
    "program_terminal_policy": "F_minus_1_observed_outgoing_intervals_no_terminal_token",
    "program_temporal_ordinal": "interval_endpoint_sampled_frame_position",
    "program_heads": 8,
    "program_blocks": 2,
    "program_attention": "target_action_conditioned_E_D_reads_then_private_role_causal",
    "program_identity_path": "target_role_token_and_ordinal_qk_only",
    "program_memory_path": "separate_target_bound_causal_A_E_D_raw_value_streams",
    "program_key_path": "normalized_role_content_plus_qk_only_identity",
    "program_value_path": "raw_physical_role_content_without_identity",
    "target_count": 38,
    "public_rank": 16,
    "compiler_order": "sealed_policy_target_first_then_rank_last",
    "core_reader": "38_target_raw_core_reads_before_dynamic_binding",
    "program_coordinate_reader": "38x16_private_Action_Effect_Change_history_reads",
    "reader_softmax": "independent_core_Action_Effect_Change_normalizers",
    "coordinate_mixer": "fixed_task_language_top2_complete_direction_stores",
    "coordinate_identity_path": "normalized_target_and_rank_qk_only",
    "compiler_heads": 8,
    "factor_input": "raw_target_core_Action_Effect_Change_concat_width1024",
    "factor_head_bias": False,
    "factor_hidden_width": 256,
    "direction_store_count": 8,
    "direction_store_top_k": 2,
    "direction_route_anchor": "train24_mean_centered_frozen_base_text_task_span_mean_l2",
    "direction_route_fit": "train24_language_only_seed7_spherical_kmeans",
    "direction_route_weights": "fixed_equal_top2",
    "factor_value_path": "top2_complete_independent_1024_256_output_direction_stores",
    "factor_final_projection": "exact_zero_initialization",
    "initialization_seed": 7,
}


def expected_writer_contract(writer: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact direction-store payload while preserving route data."""

    return {
        **_STATIC_WRITER_CONTRACT,
        "frame_stride": writer["frame_stride"],
        "max_frames_per_encoder_call": writer["max_frames_per_encoder_call"],
        "direction_store_centers": writer["direction_store_centers"],
        "direction_anchor_mean": writer["direction_anchor_mean"],
    }


def validate_writer_dimensions(observed: Mapping[str, Any]) -> None:
    """Reject constructor values outside the canonical direction-store topology."""

    changed = {
        name: (WRITER_DIMENSION_CONTRACT[name], observed.get(name))
        for name in WRITER_DIMENSION_CONTRACT
        if observed.get(name) != WRITER_DIMENSION_CONTRACT[name]
    }
    if changed:
        raise ValueError(f"invalid EMBER semantic direction-store dimensions: {changed}")
    centers = observed.get("direction_store_centers")
    anchor_mean = observed.get("direction_anchor_mean")
    if (
        not isinstance(centers, list)
        or len(centers) != WRITER_DIMENSION_CONTRACT["direction_store_count"]
        or any(
            not isinstance(center, list)
            or len(center) != WRITER_DIMENSION_CONTRACT["image_width"]
            or any(not math.isfinite(float(value)) for value in center)
            for center in centers
        )
    ):
        raise ValueError("invalid EMBER semantic direction-store centers")
    if (
        not isinstance(anchor_mean, list)
        or len(anchor_mean) != WRITER_DIMENSION_CONTRACT["image_width"]
        or any(not math.isfinite(float(value)) for value in anchor_mean)
    ):
        raise ValueError("invalid EMBER semantic direction-store anchor mean")
