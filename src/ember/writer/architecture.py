"""Declarative contract for the K4 policy-layer trace M2P Writer."""

from __future__ import annotations

from typing import Any, Mapping


FEWSHOT_M2P_WRITER_CONSTRUCTOR_KEYS = frozenset(
    {
        "image_width",
        "expert_width",
        "policy_groups",
        "trace_temporal_terms",
        "memory_slots",
        "m2p_width",
        "m2p_heads",
        "m2p_blocks",
        "m2p_ffn_expansion",
        "max_frames_per_encoder_call",
        "action_horizon",
        "padded_action_dim",
        "videos_per_condition",
        "initialization_seed",
    }
)

WRITER_DIMENSION_CONTRACT = {
    "image_width": 2048,
    "expert_width": 1024,
    "policy_groups": 20,
    "trace_temporal_terms": 16,
    "memory_slots": 68,
    "m2p_width": 1024,
    "m2p_heads": 8,
    "m2p_blocks": 4,
    "m2p_ffn_expansion": 2,
    "action_horizon": 50,
    "padded_action_dim": 32,
    "videos_per_condition": 4,
}

_STATIC_WRITER_CONTRACT: dict[str, Any] = {
    "architecture": "pi05_k4_policy_layer_trace_axis_m2p_v1",
    "generated_adapter": "one_complete_pi05_task_specific_rank16_lora",
    "camera_dataset": "obs/agentview_rgb",
    "camera_transform": "libero_opengl_rotate_180_chw_uint8",
    "include_final_frame": True,
    "teacher_prompt": "Task: {cleaned_task};\nAction: ",
    "teacher_state_input": False,
    "videos_per_condition": 4,
    "video_set_order": "permutation_invariant_without_shot_identity",
    "condition_descriptor": "frozen_pi05_all_action_expert_layer_video_innovation",
    "condition_descriptor_gradient": "none",
    "condition_baseline": "same_language_same_suffix_zero_image_tokens_per_policy_group",
    "language_role": "video_grounding_inside_frozen_pi05_only",
    "language_value_bypass": False,
    "policy_groups": 20,
    "trace_temporal_terms": 16,
    "trace_width": 1024,
    "trace_tokens_per_group_per_condition": 64,
    "memory_slots": 68,
    "m2p_width": 1024,
    "m2p_heads": 8,
    "m2p_blocks": 4,
    "m2p_ffn_expansion": 2,
    "m2p_topology": "alternating_policy_group_column_and_parameter_slot_row",
    "reader_value_owner": "baseline_subtracted_video_trace_only",
    "reader_group_outputs": "twenty_independent_exact_zero_initialized_matrices",
    "policy_targets": 38,
    "public_rank": 16,
    "parameterization": "direct_group_memory_slice_and_reshape_without_target_mlp",
    "decoder_freeze": "none_natural_zero_init_reachability_step1_then_step2",
    "step0_identity": "template_A_plus_direct_zero_dynamic_and_physical_zero_B",
    "video_zero_identity": True,
    "image_width": 2048,
    "expert_width": 1024,
    "frame_batching_contract": "four_videos_serial_memory_safety_chunks_plus_one_baseline_per_video",
    "action_horizon": 50,
    "padded_action_dim": 32,
    "initialization_seed": 7,
}


def expected_writer_contract(writer: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact layer-trace M2P payload with profiled frame chunking."""

    if writer.get("architecture") != _STATIC_WRITER_CONTRACT["architecture"]:
        raise ValueError(f"unsupported EMBER Writer architecture: {writer.get('architecture')}")
    return {
        **_STATIC_WRITER_CONTRACT,
        "frame_stride": writer["frame_stride"],
        "max_frames_per_encoder_call": writer["max_frames_per_encoder_call"],
    }


def validate_writer_dimensions(observed: Mapping[str, Any]) -> None:
    """Reject constructor values outside the canonical layer-trace topology."""

    changed = {
        name: (expected, observed.get(name))
        for name, expected in WRITER_DIMENSION_CONTRACT.items()
        if observed.get(name) != expected
    }
    if changed:
        raise ValueError(f"invalid EMBER Writer dimensions: {changed}")
