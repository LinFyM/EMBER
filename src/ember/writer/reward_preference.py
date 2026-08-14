"""Task-equal binary reward credit through generated LoRA into the Writer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from lerobot.utils.constants import ACTION

from ember.lora import LoRAContract
from ember.reward.loss import functional_executed_prefix_flow_loss
from ember.reward.protocol import RewardProtocolError, reward_preference_flow_seed


@dataclass(frozen=True)
class RewardPreferenceSummary:
    objective: float
    successes: int
    replay_chunks: int
    executed_action_steps: int
    functional_policy_forwards: int
    lora_gradient_rms: float


def leave_one_out_binary_advantages(successes: torch.Tensor) -> torch.Tensor:
    if (
        successes.shape != (4,)
        or not bool(torch.isfinite(successes).all())
        or bool(((successes != 0) & (successes != 1)).any())
    ):
        raise RewardProtocolError("reward preference requires four binary outcomes")
    return (4 * successes - successes.sum()) / 3


def episode_equal_chunk_weights(
    episode_ids: torch.Tensor, successes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = leave_one_out_binary_advantages(successes).to(
        device=episode_ids.device, dtype=torch.float32
    )
    ids = episode_ids.to(dtype=torch.long)
    if (
        ids.ndim != 1
        or ids.numel() == 0
        or bool((ids < 0).any())
        or bool((ids >= 4).any())
    ):
        raise RewardProtocolError("reward replay episode IDs changed")
    counts = torch.bincount(ids, minlength=4)
    if counts.shape != (4,) or bool((counts <= 0).any()):
        raise RewardProtocolError("reward replay lost an episode")
    weights = advantages.index_select(0, ids) / (
        4.0 * counts.index_select(0, ids).to(dtype=torch.float32)
    )
    return weights, advantages


def _flow_sample_panel(
    policy: torch.nn.Module,
    *,
    count: int,
    mc_samples: int,
    seed_root: int,
    cycle: int,
    global_task_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_config = getattr(getattr(policy, "model", None), "config", None)
    if (
        count <= 0
        or mc_samples != 4
        or model_config is None
        or float(model_config.time_sampling_beta_alpha) != 1.5
        or float(model_config.time_sampling_beta_beta) != 1.0
        or float(model_config.time_sampling_scale) != 0.999
        or float(model_config.time_sampling_offset) != 0.001
    ):
        raise RewardProtocolError("reward flow sample panel changed")
    shape = (count, int(policy.config.chunk_size), int(policy.config.max_action_dim))
    noises, times = [], []
    for mc_index in range(mc_samples):
        generator = torch.Generator(device=device).manual_seed(
            reward_preference_flow_seed(
                seed_root,
                cycle=cycle,
                global_task_id=global_task_id,
                mc_index=mc_index,
            )
        )
        noises.append(torch.randn(shape, generator=generator, device=device))
        uniform = torch.rand(count, generator=generator, device=device)
        times.append(uniform.pow(2.0 / 3.0).mul_(0.999).add_(0.001))
    return torch.stack(noises), torch.stack(times)


def functional_reward_lora_gradient(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, torch.Tensor],
    episode_ids: torch.Tensor,
    successes: torch.Tensor,
    *,
    mc_samples: int,
    physical_microbatch_size: int,
    flow_seed_root: int,
    cycle: int,
    global_task_id: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], RewardPreferenceSummary]:
    """Differentiate signed task-relative CFM through LoRA leaves only."""

    action = batch.get(ACTION)
    if not isinstance(action, torch.Tensor) or action.ndim != 3:
        raise RewardProtocolError("mixed reward replay action batch changed")
    count = int(action.shape[0])
    weights, advantages = episode_equal_chunk_weights(episode_ids, successes)
    if not bool(torch.count_nonzero(advantages)):
        raise RewardProtocolError("homogeneous reward panel entered CFM")
    noises, times = _flow_sample_panel(
        policy,
        count=count,
        mc_samples=mc_samples,
        seed_root=flow_seed_root,
        cycle=cycle,
        global_task_id=global_task_id,
        device=device,
    )
    names = tuple(state)
    leaves = {
        name: value.detach().requires_grad_(True) for name, value in state.items()
    }
    gradients = {
        name: torch.zeros_like(value, dtype=torch.float32)
        for name, value in state.items()
    }
    objective = torch.zeros((), dtype=torch.float32, device=device)
    forwards = 0
    for start in range(0, count, physical_microbatch_size):
        stop = min(start + physical_microbatch_size, count)
        sliced = {
            name: value[start:stop].to(device=device, non_blocking=True)
            for name, value in batch.items()
        }
        chunk_weights = weights[start:stop].to(device=device, non_blocking=True)
        for mc_index in range(mc_samples):
            per_chunk = functional_executed_prefix_flow_loss(
                policy,
                leaves,
                contract,
                sliced,
                noise=noises[mc_index, start:stop],
                time=times[mc_index, start:stop],
            )
            scalar = (per_chunk.float() * chunk_weights).sum() / mc_samples
            per_lora = torch.autograd.grad(
                scalar, tuple(leaves[name] for name in names)
            )
            objective.add_(scalar.detach())
            for name, gradient in zip(names, per_lora, strict=True):
                gradients[name].add_(gradient.float())
            forwards += 1
    gradient_rms = (
        torch.cat([value.flatten() for value in gradients.values()])
        .square()
        .mean()
        .sqrt()
    )
    if not bool(torch.isfinite(gradient_rms)) or float(gradient_rms) <= 0:
        raise RewardProtocolError("mixed reward panel produced invalid LoRA credit")
    valid = batch["executed_action_steps"]
    return gradients, RewardPreferenceSummary(
        objective=float(objective),
        successes=int(successes.sum()),
        replay_chunks=count,
        executed_action_steps=int(valid.sum()),
        functional_policy_forwards=forwards,
        lora_gradient_rms=float(gradient_rms),
    )


def backpropagate_reward_preference(
    generated: Mapping[str, torch.Tensor],
    lora_gradients: Mapping[str, torch.Tensor],
) -> None:
    """Transport FP32 LoRA cotangents through the native compiler once."""

    active = tuple(name for name, value in generated.items() if value.requires_grad)
    if not active or set(lora_gradients) != set(generated):
        raise RewardProtocolError("reward Writer graph lost generated LoRA outputs")
    torch.autograd.backward(
        tuple(generated[name] for name in active),
        grad_tensors=tuple(lora_gradients[name] for name in active),
    )
