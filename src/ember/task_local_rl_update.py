"""Physical task-LoRA reward-weighted flow update."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from ember.libero_reward_rollout import (
    RewardTrajectory,
    successful_trajectory_batch,
)
from ember.lora import task_lora_state_dict
from ember.source_base_checkpoint import DistributedContext
from ember.task_local_rl_protocol import TaskArm, optimizer_seed
from ember.task_local_rl_runtime import TaskLocalRLRuntime
from ember.writer.model import WriterModelError


def equal_episode_flow_loss(
    policy: torch.nn.Module,
    trajectories: Sequence[RewardTrajectory],
    device: torch.device,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    successful = [trajectory for trajectory in trajectories if trajectory.success]
    batch, episode_ids = successful_trajectory_batch(successful, device)
    output = policy(batch, reduction="none")
    if (
        not isinstance(output, tuple)
        or len(output) != 2
        or not isinstance(output[0], torch.Tensor)
        or output[0].ndim != 1
        or not isinstance(output[1], Mapping)
        or output[0].shape != episode_ids.shape
    ):
        raise WriterModelError("task-local policy did not return per-chunk loss")
    unique = torch.unique(episode_ids, sorted=True)
    if unique.numel() != len(successful) or not torch.equal(
        unique, torch.arange(len(successful), device=device)
    ):
        raise WriterModelError("task-local successful episode IDs changed")
    episode_losses = torch.stack(
        [output[0][episode_ids == episode_id].mean() for episode_id in unique]
    )
    return episode_losses.mean(), {
        **output[1],
        "successful_episodes": len(successful),
        "successful_chunks": int(output[0].numel()),
    }


def reward_update(
    *,
    runtime: TaskLocalRLRuntime,
    context: DistributedContext,
    unit: TaskArm,
    update: int,
    trajectories: Sequence[RewardTrajectory],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[float, float]:
    seed = optimizer_seed(
        int(runtime.config["rng"]["update_seed_base"]), unit.task_id, update
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    runtime.policy.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, _ = equal_episode_flow_loss(
            runtime.policy, trajectories, context.device
        )
    if not bool(torch.isfinite(loss).detach()):
        raise WriterModelError(
            f"non-finite task-local RL loss for {unit.key} update {update}"
        )
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        task_lora_state_dict(runtime.policy).values(),
        float(runtime.config["optimization"]["grad_clip_norm"]),
    )
    optimizer.step()
    scheduler.step()
    return float(loss.detach()), float(grad_norm)
