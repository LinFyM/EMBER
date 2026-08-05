"""Declarative contract for the K4 invariant-program M2P Writer."""

from __future__ import annotations

from typing import Any, Mapping


FEWSHOT_M2P_WRITER_CONSTRUCTOR_KEYS = frozenset(
    {
        "image_width",
        "expert_width",
        "program_width",
        "program_slots",
        "program_heads",
        "program_blocks",
        "m2p_heads",
        "m2p_blocks",
        "factor_hidden_width",
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
    "program_width": 256,
    "program_slots": 32,
    "program_heads": 8,
    "program_blocks": 2,
    "m2p_heads": 8,
    "m2p_blocks": 3,
    "factor_hidden_width": 256,
    "action_horizon": 50,
    "padded_action_dim": 32,
    "videos_per_condition": 4,
}

_STATIC_WRITER_CONTRACT: dict[str, Any] = {
    "architecture": "pi05_k4_video_value_invariant_program_policy_m2p_v1",
    "generated_adapter": "one_complete_pi05_task_specific_rank16_lora",
    "camera_dataset": "obs/agentview_rgb",
    "camera_transform": "libero_opengl_rotate_180_chw_uint8",
    "include_final_frame": True,
    "teacher_prompt": "Task: {cleaned_task};\nAction: ",
    "teacher_state_input": False,
    "videos_per_condition": 4,
    "video_set_order": "permutation_invariant_without_shot_identity",
    "per_video_temporal_values": "one_tau_cospi_sinpi_four_tokens",
    "condition_descriptor": (
        "frozen_source_text_vl_innovation_and_fixed_suffix_action_expert"
    ),
    "condition_descriptor_gradient": "none",
    "language_role": "video_grounding_and_attention_address_only",
    "language_value_bypass": False,
    "video_value_width": 128,
    "video_value_tokens_per_condition": 16,
    "program_slots": 32,
    "program_width": 256,
    "program_heads": 8,
    "program_blocks": 2,
    "program_first_read": "query_no_residual_video_values_only",
    "policy_targets": 38,
    "public_rank": 16,
    "m2p_tokens": 608,
    "m2p_heads": 8,
    "m2p_blocks": 3,
    "m2p_topology": "program_cross_attention_then_policy_token_self_attention",
    "target_heads": "target_owned_paired_complete_A_B_bias_free",
    "decoder_freeze": "none",
    "factor_hidden_width": 256,
    "step0_identity": "template_A_plus_zero_residual_and_physical_zero_B",
    "video_zero_identity": True,
    "image_width": 2048,
    "expert_width": 1024,
    "frame_batching_contract": "four_videos_serial_memory_safety_chunks",
    "action_horizon": 50,
    "padded_action_dim": 32,
    "initialization_seed": 7,
}


def expected_writer_contract(writer: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact K4 M2P payload with profiled frame chunking."""

    if writer.get("architecture") != _STATIC_WRITER_CONTRACT["architecture"]:
        raise ValueError(f"unsupported EMBER Writer architecture: {writer.get('architecture')}")
    return {
        **_STATIC_WRITER_CONTRACT,
        "frame_stride": writer["frame_stride"],
        "max_frames_per_encoder_call": writer["max_frames_per_encoder_call"],
    }


def validate_writer_dimensions(observed: Mapping[str, Any]) -> None:
    """Reject constructor values outside the canonical K4 M2P topology."""

    changed = {
        name: (expected, observed.get(name))
        for name, expected in WRITER_DIMENSION_CONTRACT.items()
        if observed.get(name) != expected
    }
    if changed:
        raise ValueError(f"invalid EMBER Writer dimensions: {changed}")
