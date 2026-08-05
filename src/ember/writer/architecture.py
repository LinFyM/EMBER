"""Single-source declarative contract for the canonical EMBER Writer."""

from __future__ import annotations

from typing import Any, Mapping


CONDITION_KERNEL_WRITER_PARAMETER_COUNT = 86_065_152

CONDITION_KERNEL_WRITER_CONSTRUCTOR_KEYS = frozenset(
    {
        "image_width",
        "expert_width",
        "program_width",
        "max_frames_per_encoder_call",
        "action_horizon",
        "padded_action_dim",
        "factor_hidden_width",
        "condition_task_rff_frequencies",
        "condition_video_rff_frequencies",
        "initialization_seed",
    }
)

WRITER_DIMENSION_CONTRACT = {
    "image_width": 2048,
    "expert_width": 1024,
    "program_width": 256,
    "action_horizon": 50,
    "padded_action_dim": 32,
    "factor_hidden_width": 256,
    "condition_task_rff_frequencies": 16,
    "condition_video_rff_frequencies": 16,
}

_STATIC_WRITER_CONTRACT: dict[str, Any] = {
    "architecture": "pi05_factorized_condition_kernel_program_memory_v1",
    "generated_adapter": "complete_pi05_task_specific_rank16_lora",
    "camera_dataset": "obs/agentview_rgb",
    "camera_transform": "libero_opengl_rotate_180_chw_uint8",
    "include_final_frame": True,
    "teacher_prompt": "Task: {cleaned_task};\nAction: ",
    "teacher_state_input": False,
    "task_span_extraction": (
        "authoritative_full_prompt_sentencepiece_piece_offsets"
    ),
    "condition_descriptor": (
        "frozen_source_text_vl_and_fixed_suffix_action_expert_innovation"
    ),
    "condition_descriptor_gradient": "none",
    "condition_task_center": "train24_text_only_mean_only",
    "condition_video_temporal_basis": "one_tau_cospi_sinpi",
    "condition_frame_projection_width": 128,
    "condition_task_rff_frequencies": 16,
    "condition_video_rff_frequencies": 16,
    "condition_feature": "l2_normalized_32x32_rff_outer_product",
    "condition_feature_width": 1024,
    "condition_router": "none",
    "program_slots": 320,
    "program_width": 256,
    "program_memory": "complete_value_per_fixed_condition_feature",
    "program_memory_parameter_count": 83_886_080,
    "program_memory_initialization": "normal_std_0.02",
    "program_memory_optimizer": "explicit_regularized_kernel_update_no_adam",
    "kernel_relative_damping": 0.01,
    "decoder": "eight_fresh_bias_free_hidden256_factor_heads",
    "decoder_freeze_after_macro": 50,
    "factor_head_bias": False,
    "factor_hidden_width": 256,
    "image_width": 2048,
    "native_image_tokens": 256,
    "expert_width": 1024,
    "frame_batching_contract": (
        "frozen_condition_descriptor_unpadded_memory_safety_chunks"
    ),
    "action_horizon": 50,
    "padded_action_dim": 32,
    "action_expert_probe": "one_frozen_forward_fixed_gaussian_suffix_at_t1",
    "action_expert_action_out": False,
    "initialization_seed": 7,
}


def expected_writer_contract(writer: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact condition-kernel payload with profiled frame chunking."""

    architecture = writer.get("architecture")
    if architecture != _STATIC_WRITER_CONTRACT["architecture"]:
        raise ValueError(f"unsupported EMBER Writer architecture: {architecture}")
    return {
        **_STATIC_WRITER_CONTRACT,
        "frame_stride": writer["frame_stride"],
        "max_frames_per_encoder_call": writer["max_frames_per_encoder_call"],
    }


def validate_writer_dimensions(observed: Mapping[str, Any]) -> None:
    """Reject constructor values outside the canonical condition-kernel topology."""

    changed = {
        name: (WRITER_DIMENSION_CONTRACT[name], observed.get(name))
        for name in WRITER_DIMENSION_CONTRACT
        if observed.get(name) != WRITER_DIMENSION_CONTRACT[name]
    }
    if changed:
        raise ValueError(f"invalid EMBER Writer dimensions: {changed}")
