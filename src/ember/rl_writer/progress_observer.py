"""Frozen AS125 semantic observer for Program-Credit failure-pair ties."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from ember.lora import copy_task_lora_state_
from ember.reward.protocol import RewardProtocolError, RewardTask
from ember.reward.rollout import RewardTrajectory
from ember.rl_writer.progress_credit import (
    normalized_progress_components,
    semantic_progress_utilities,
)


def encode_progress_components(
    *,
    writer: torch.nn.Module,
    policy: torch.nn.Module,
    identity_state: Mapping[str, torch.Tensor],
    lora_contract: Any,
    tokenizer: Any,
    task: RewardTask,
    frames: torch.Tensor,
    device: torch.device,
    normalization_epsilon: float,
) -> torch.Tensor:
    """Encode one task's RGB frames without public task-LoRA self-conditioning."""

    if frames.ndim != 4 or frames.shape[1] != 3 or frames.dtype != torch.uint8:
        raise RewardProtocolError("progress observer frame batch changed")
    copy_task_lora_state_(policy, identity_state, lora_contract)
    tokens, masks, spans = tokenizer([task.language])
    values = frames.to(device, non_blocking=True)
    condition_ids = torch.zeros(values.shape[0], dtype=torch.long, device=device)
    writer.semantic_encoder.eval()
    policy.eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        _, _, grounded, interactions, valid = writer.semantic_encoder(
            policy,
            values,
            condition_ids,
            tokens,
            masks,
            spans,
        )
        packed = normalized_progress_components(
            grounded,
            interactions,
            valid,
            epsilon=normalization_epsilon,
        )
    return packed.detach().float().cpu()


def rollout_endpoint_frames(
    trajectories: Sequence[RewardTrajectory],
) -> torch.Tensor:
    frames = []
    for trajectory in trajectories:
        if (
            trajectory.progress_start_frame is None
            or trajectory.progress_terminal_frame is None
        ):
            raise RewardProtocolError("progress observer lost rollout endpoint RGB")
        frames.extend(
            (trajectory.progress_start_frame, trajectory.progress_terminal_frame)
        )
    if not frames:
        raise RewardProtocolError("progress observer received no rollout")
    return torch.stack(frames)


def observe_correct_teacher_progress(
    *,
    writer: torch.nn.Module,
    policy: torch.nn.Module,
    identity_state: Mapping[str, torch.Tensor],
    lora_contract: Any,
    tokenizer: Any,
    task: RewardTask,
    teacher_frames: torch.Tensor,
    trajectories: Sequence[RewardTrajectory],
    device: torch.device,
    normalization_epsilon: float,
    projection_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher = encode_progress_components(
        writer=writer,
        policy=policy,
        identity_state=identity_state,
        lora_contract=lora_contract,
        tokenizer=tokenizer,
        task=task,
        frames=torch.stack((teacher_frames[0], teacher_frames[-1])),
        device=device,
        normalization_epsilon=normalization_epsilon,
    )
    rollout = encode_progress_components(
        writer=writer,
        policy=policy,
        identity_state=identity_state,
        lora_contract=lora_contract,
        tokenizer=tokenizer,
        task=task,
        frames=rollout_endpoint_frames(trajectories),
        device=device,
        normalization_epsilon=normalization_epsilon,
    )
    return semantic_progress_utilities(
        teacher[0],
        teacher[1],
        rollout[0::2],
        rollout[1::2],
        epsilon=projection_epsilon,
    )
