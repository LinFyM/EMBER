"""Observation-only fixed-query and policy-forward helpers for UCP analysis."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import h5py
import numpy as np
import torch
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from ember.lora import copy_task_lora_state_
from ember.pi05_processing import Pi05LiberoProcessor
from ember.writer.data import WriterTaskAuthority, _camera


def policy_action(
    *, policy: torch.nn.Module, processor: Pi05LiberoProcessor,
    prepared: Mapping[str, torch.Tensor], state: Mapping[str, torch.Tensor],
    identity: Mapping[str, torch.Tensor], lora: Any, seed: int, device: torch.device,
) -> torch.Tensor:
    copy_task_lora_state_(policy, state, lora)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        1, int(policy.model.config.chunk_size),
        int(policy.model.config.max_action_dim), generator=generator,
        dtype=torch.float32,
    ).to(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        value = policy.predict_action_chunk(dict(prepared), noise=noise)
    copy_task_lora_state_(policy, identity, lora)
    return processor.unnormalize_action(value).detach()


def fixed_policy_query(
    authority: WriterTaskAuthority,
    processor: Pi05LiberoProcessor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Read only demo0/frame0 observations; never open the actions dataset."""

    with h5py.File(authority.path, "r") as handle:
        obs = handle["data/demo_0/obs"]
        base_array = np.asarray(obs["agentview_rgb"][0])
        wrist_array = np.asarray(obs["eye_in_hand_rgb"][0])
        state_array = np.concatenate((
            np.asarray(obs["ee_states"][0], dtype=np.float32),
            np.asarray(obs["gripper_states"][0], dtype=np.float32),
        ))
    base = torch.from_numpy(_camera(base_array))[None].to(
        device, dtype=torch.float32,
    ).div_(255)
    wrist = torch.from_numpy(_camera(wrist_array))[None].to(
        device, dtype=torch.float32,
    ).div_(255)
    states = torch.from_numpy(state_array)[None].to(device)
    tokens, masks = processor._tokenize_prompts(states, [authority.language])
    identity = {
        "demo_index": 0, "frame_index": 0, "observation_only": True,
        "actions_dataset_opened": False,
        "base_sha256": hashlib.sha256(
            np.ascontiguousarray(base_array)
        ).hexdigest(),
        "wrist_sha256": hashlib.sha256(
            np.ascontiguousarray(wrist_array)
        ).hexdigest(),
        "state_sha256": hashlib.sha256(
            np.ascontiguousarray(state_array)
        ).hexdigest(),
    }
    return {
        "observation.images.base_0_rgb": base,
        "observation.images.left_wrist_0_rgb": wrist,
        OBS_LANGUAGE_TOKENS: tokens,
        OBS_LANGUAGE_ATTENTION_MASK: masks,
    }, identity
