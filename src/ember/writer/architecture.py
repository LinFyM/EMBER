"""Declarative contract for the canonical Dynamic-K Backbone-Memory Writer."""

from __future__ import annotations

from typing import Any, Mapping


DYNAMIC_K_WRITER_CONSTRUCTOR_KEYS = frozenset(
    {
        "image_width",
        "expert_width",
        "program_width",
        "mapper_width",
        "action_meta_lora_rank",
        "temporal_heads",
        "max_frames_per_encoder_call",
        "action_horizon",
        "padded_action_dim",
        "initialization_seed",
        "activation_checkpointing",
    }
)

WRITER_DIMENSION_CONTRACT = {
    "image_width": 2048,
    "expert_width": 1024,
    "program_width": 256,
    "mapper_width": 1024,
    "action_meta_lora_rank": 4,
    "temporal_heads": 8,
    "action_horizon": 50,
    "padded_action_dim": 32,
}

_STATIC_WRITER_CONTRACT: dict[str, Any] = {
    "architecture": "pi05_dynamic_k_backbone_memory_rank8_v1",
    "generated_adapter": "complete_pi05_task_specific_rank8_lora",
    "input": "exact_task_language_plus_one_to_four_action_hidden_videos",
    "frame_stride": 5,
    "backbone_total_frames_per_condition": 64,
    "native_action_probe_tokens": 50,
    "backbone_memory_tokens": 8,
    "backbone_layers": 18,
    "backbone_width": 1024,
    "program_width": 256,
    "mapper_width": 1024,
    "action_meta_lora_rank": 4,
    "temporal_heads": 8,
    "temporal_blocks": 2,
    "set_blocks": 2,
    "m2p_blocks": 2,
    "public_lora_rank": 8,
    "public_lora_alpha": 8,
}


def expected_writer_contract(writer: Mapping[str, Any]) -> dict[str, Any]:
    """Return the fixed topology while preserving profiled batching fields."""

    return {
        **_STATIC_WRITER_CONTRACT,
        "max_frames_per_encoder_call": writer["max_frames_per_encoder_call"],
        "initialization_seed": writer["initialization_seed"],
        "activation_checkpointing": writer["activation_checkpointing"],
    }


def validate_writer_dimensions(observed: Mapping[str, Any]) -> None:
    changed = {
        name: (expected, observed.get(name))
        for name, expected in WRITER_DIMENSION_CONTRACT.items()
        if observed.get(name) != expected
    }
    if changed:
        raise ValueError(f"invalid EMBER Writer dimensions: {changed}")
