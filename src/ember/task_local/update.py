"""Executed-prefix reward update of one physical PI05 task-local LoRA."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from ember.lora import LoRAContract, task_lora_state_dict
from ember.reward.loss import Pi05ExecutedPrefixFlowLoss, equal_episode_loss
from ember.reward.protocol import RewardProtocolError
from ember.reward.rollout import RewardTrajectory, successful_trajectory_batch


def reward_update(
    *,
    policy: torch.nn.Module,
    lora_contract: LoRAContract,
    trajectories: Sequence[RewardTrajectory],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    gradient_clip_norm: float,
    update_seed: int,
) -> tuple[float, float, Mapping[str, Any]]:
    successful = [trajectory for trajectory in trajectories if trajectory.success]
    if not successful:
        raise RewardProtocolError("task-local reward update has no successful trajectory")
    trainable = task_lora_state_dict(policy)
    names = {name for name, value in policy.named_parameters() if value.requires_grad}
    if names != set(trainable):
        raise RewardProtocolError("task-local update left state trainable outside one LoRA")
    torch.manual_seed(update_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(update_seed)
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    batch, episode_ids = successful_trajectory_batch(successful, device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        per_chunk, details = Pi05ExecutedPrefixFlowLoss(policy)(batch)
        loss, episode_details = equal_episode_loss(per_chunk, episode_ids)
    if not bool(torch.isfinite(loss).detach()):
        raise RewardProtocolError("task-local reward loss is non-finite")
    loss.backward()
    if any(
        parameter.grad is not None
        for name, parameter in policy.named_parameters()
        if name not in trainable
    ):
        raise RewardProtocolError("task-local frozen source accumulated gradients")
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable.values(), gradient_clip_norm)
    if not bool(torch.isfinite(grad_norm).detach()):
        raise RewardProtocolError("task-local reward gradient is non-finite")
    optimizer.step()
    scheduler.step()
    return float(loss.detach()), float(grad_norm), {
        **details,
        **episode_details,
        "update_seed": update_seed,
    }
