"""Task-relative FPO++ credit for a frozen flow policy under generated LoRA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from ember.reward.protocol import RewardProtocolError


@dataclass(frozen=True)
class FlowCreditMetrics:
    objective: float
    ratio_mean: float
    ratio_min: float
    ratio_max: float
    clipped_positive_fraction: float
    positive_episodes: int
    negative_episodes: int
    zero_episodes: int


def generated_lora_gradient_norm(
    gradients: Sequence[torch.Tensor | None],
) -> float:
    if any(value is None for value in gradients):
        raise RewardProtocolError("Flow-Credit generated LoRA gradient is incomplete")
    value = math.sqrt(
        sum(float(item.detach().float().square().sum()) for item in gradients)
    )
    if not math.isfinite(value):
        raise RewardProtocolError("non-finite generated LoRA credit gradient")
    return value


def leave_one_out_binary_advantages(successes: torch.Tensor) -> torch.Tensor:
    """Return an action-independent leave-one-out baseline within one task."""

    if (
        successes.ndim != 1
        or successes.numel() < 2
        or not bool(torch.isfinite(successes).all())
        or bool(((successes != 0) & (successes != 1)).any())
    ):
        raise RewardProtocolError("task-relative rewards must be finite binary groups")
    count = successes.numel()
    return (count * successes - successes.sum()) / (count - 1)


def _episode_weights(
    episode_ids: torch.Tensor,
    episode_count: int,
    episode_chunk_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        episode_ids.ndim != 1
        or episode_count <= 0
    ):
        raise RewardProtocolError("invalid relative-flow episode aggregation")
    if (
        bool((episode_ids < 0).any())
        or bool((episode_ids >= episode_count).any())
    ):
        raise RewardProtocolError("relative-flow episode ID is outside its task")
    if episode_chunk_counts is None:
        expected = torch.arange(episode_count, device=episode_ids.device)
        if not torch.equal(torch.unique(episode_ids, sorted=True), expected):
            raise RewardProtocolError("relative-flow episode IDs must be contiguous")
        counts = torch.bincount(episode_ids, minlength=episode_count)
    else:
        counts = episode_chunk_counts.to(episode_ids.device)
        if counts.shape != (episode_count,) or bool((counts <= 0).any()):
            raise RewardProtocolError("invalid relative-flow episode chunk counts")
    return counts[episode_ids].reciprocal() / episode_count


def task_relative_aspo_loss(
    current_losses: torch.Tensor,
    old_losses: torch.Tensor,
    episode_ids: torch.Tensor,
    successes: torch.Tensor,
    *,
    task_advantages: torch.Tensor | None = None,
    clip_epsilon: float,
    loss_value_clip: float,
    log_ratio_clip: float,
    episode_chunk_counts: torch.Tensor | None = None,
    mc_samples_normalizer: int | None = None,
) -> tuple[torch.Tensor, FlowCreditMetrics]:
    """Minimize the negative per-sample ASPO objective for one task/video group.

    Loss tensors are shaped ``[N_mc, chunks]``. Chunks are first averaged within
    their episode, then the K episodes are averaged so long trajectories do not
    receive more task credit.
    """

    if (
        current_losses.shape != old_losses.shape
        or current_losses.ndim != 2
        or current_losses.numel() == 0
        or not current_losses.requires_grad
        or not bool(torch.isfinite(current_losses).all())
        or not bool(torch.isfinite(old_losses).all())
        or old_losses.requires_grad
        or not 0 < clip_epsilon < 1
        or min(loss_value_clip, log_ratio_clip) <= 0
    ):
        raise RewardProtocolError("invalid task-relative FPO loss inputs")
    binary_advantages = leave_one_out_binary_advantages(successes)
    advantages = (
        binary_advantages if task_advantages is None else task_advantages
    ).to(device=current_losses.device, dtype=current_losses.dtype)
    if (
        advantages.shape != successes.shape
        or not bool(torch.isfinite(advantages).all())
        or abs(float(advantages.sum())) > 1e-5
    ):
        raise RewardProtocolError("task-relative advantages changed contract")
    if episode_ids.device != current_losses.device:
        episode_ids = episode_ids.to(current_losses.device)
    old = old_losses.to(current_losses.device, current_losses.dtype)
    delta = old.clamp(max=loss_value_clip) - current_losses.clamp(
        max=loss_value_clip
    )
    ratio = torch.exp(delta.clamp(-log_ratio_clip, log_ratio_clip))
    advantage = advantages[episode_ids][None, :].expand_as(ratio)
    positive = advantage >= 0
    clipped = ratio.new_tensor(1.0).add(
        (ratio.new_tensor(clip_epsilon))
    )
    lower = 1.0 - clip_epsilon
    upper = 1.0 + clip_epsilon
    ppo = torch.minimum(
        ratio * advantage,
        ratio.clamp(lower, upper) * advantage,
    )
    spo = ratio * advantage - (
        advantage.abs() * (ratio - 1.0).square() / (2.0 * clip_epsilon)
    )
    objective_values = torch.where(positive, ppo, spo)
    chunk_weights = _episode_weights(
        episode_ids, advantages.numel(), episode_chunk_counts
    )
    mc_normalizer = int(mc_samples_normalizer or ratio.shape[0])
    if mc_normalizer < ratio.shape[0]:
        raise RewardProtocolError("relative-flow MC normalizer is too small")
    objective = (objective_values * chunk_weights[None, :]).sum() / mc_normalizer
    loss = -objective
    positive_mask = advantage > 0
    positive_clipped = positive_mask & (ratio > clipped)
    denominator = max(int(positive_mask.sum()), 1)
    metrics = FlowCreditMetrics(
        objective=float(objective.detach()),
        ratio_mean=float(ratio.detach().mean()),
        ratio_min=float(ratio.detach().min()),
        ratio_max=float(ratio.detach().max()),
        clipped_positive_fraction=float(positive_clipped.sum()) / denominator,
        positive_episodes=int((advantages > 0).sum()),
        negative_episodes=int((advantages < 0).sum()),
        zero_episodes=int((advantages == 0).sum()),
    )
    return loss, metrics
