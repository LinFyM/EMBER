"""Shared precision context and read-only identity of sealed video Writer banks."""
from __future__ import annotations
from contextlib import nullcontext
from typing import Any, Mapping
import torch

# Metadata only: historical forward/training implementations live at their frozen commits.
VIDEO_READ_MODES = frozenset({('agentview', 'repeated_full'), ('dual', 'repeated_full')})

MODEL_SCHEMA = "ember_video_teaching_writer_v1"

ARCHITECTURE = "video_core_recurrent_procedure_complete_lora"

MODEL_DEFAULTS = {
    "schema": MODEL_SCHEMA, "architecture": ARCHITECTURE,
    "image_width": 2048, "expert_width": 1024, "program_width": 256,
    "text_meta_lora_rank": 4, "vl_meta_lora_rank": 4, "action_meta_lora_rank": 4,
    "patch_grounding_heads": 8, "max_frames_per_encoder_call": 4,
    "action_horizon": 50, "padded_action_dim": 32,
    "semantic_core_heads": 8, "semantic_core_blocks": 2, "frame_attention_initial_lambda": .05,
    "procedure_heads": 8, "procedure_blocks": 2, "fusion_heads": 8,
    "factor_hidden_width": 216, "initialization_seed": 7, "activation_checkpointing": True,
    "camera_view": "agentview", "horizon_read": "repeated_full",
}

def require_architecture_identity(model: Mapping[str, Any]) -> None:
    variable = {"max_frames_per_encoder_call", "activation_checkpointing", "camera_view"}
    if (set(model) != set(MODEL_DEFAULTS)
            or any(model[key] != value for key, value in MODEL_DEFAULTS.items() if key not in variable)
            or (model["camera_view"], model["horizon_read"]) not in VIDEO_READ_MODES
            or type(model["max_frames_per_encoder_call"]) is not int
            or model["max_frames_per_encoder_call"] <= 0
            or type(model["activation_checkpointing"]) is not bool):
        raise ValueError("canonical video-teaching Writer architecture changed")

def autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
