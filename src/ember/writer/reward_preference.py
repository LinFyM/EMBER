"""Selected-success flow credit for paired AS139/LPCP policy arms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from lerobot.utils.constants import ACTION

from ember.lora import LoRAContract
from ember.reward.loss import functional_executed_prefix_flow_loss
from ember.reward.protocol import RewardProtocolError, reward_preference_flow_seed


@dataclass(frozen=True)
class PairedSuccessCreditSummary:
    """One task's positive credit from uniquely successful paired arms."""

    objective: float
    target_trajectories: int
    replay_chunks: int
    executed_action_steps: int
    functional_policy_forwards: int
    functional_policy_backwards: int
    lora_gradient_rms: float


def selected_trajectory_chunk_weights(
    trajectory_ids: torch.Tensor,
) -> torch.Tensor:
    """Give each of one or two selected successful trajectories equal mass."""

    ids = trajectory_ids.to(dtype=torch.long)
    if ids.ndim != 1 or ids.numel() == 0:
        raise RewardProtocolError("paired success replay has no trajectory IDs")
    target_count = int(ids.max()) + 1
    counts = torch.bincount(ids, minlength=target_count)
    if (
        target_count not in {1, 2}
        or bool((ids < 0).any())
        or counts.shape != (target_count,)
        or bool((counts <= 0).any())
    ):
        raise RewardProtocolError("paired success trajectory IDs changed")
    return 1.0 / (
        target_count * counts.index_select(0, ids).to(dtype=torch.float32)
    )


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
        raise RewardProtocolError("paired success flow panel changed")
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


def _gradient_rms(gradients: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return (
        torch.cat([value.flatten() for value in gradients.values()])
        .square()
        .mean()
        .sqrt()
    )


def functional_selected_success_lora_gradient(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, torch.Tensor],
    trajectory_ids: torch.Tensor,
    *,
    mc_samples: int,
    physical_microbatch_size: int,
    flow_seed_root: int,
    cycle: int,
    global_task_id: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], PairedSuccessCreditSummary]:
    """Differentiate only the successful arm of every discordant K2 pair."""

    action = batch.get(ACTION)
    if (
        not isinstance(action, torch.Tensor)
        or action.ndim != 3
        or physical_microbatch_size <= 0
    ):
        raise RewardProtocolError("paired success replay action batch changed")
    count = int(action.shape[0])
    weights = selected_trajectory_chunk_weights(trajectory_ids)
    if weights.shape != (count,):
        raise RewardProtocolError("paired success replay weights changed")
    noises, times = _flow_sample_panel(
        policy,
        count=count,
        mc_samples=mc_samples,
        seed_root=flow_seed_root,
        cycle=cycle,
        global_task_id=global_task_id,
        device=device,
    )
    leaves = {
        name: value.detach().requires_grad_(True) for name, value in state.items()
    }
    names = tuple(leaves)
    gradients = {
        name: torch.zeros_like(value, dtype=torch.float32)
        for name, value in leaves.items()
    }
    objective = torch.zeros((), dtype=torch.float32, device=device)
    forwards = backwards = 0
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
            values = torch.autograd.grad(scalar, tuple(leaves[name] for name in names))
            objective.add_(scalar.detach())
            for name, gradient in zip(names, values, strict=True):
                gradients[name].add_(gradient.float())
            forwards += 1
            backwards += 1
    rms = _gradient_rms(gradients)
    if not bool(torch.isfinite(rms)) or float(rms) <= 0:
        raise RewardProtocolError("paired success produced invalid LoRA credit")
    valid = batch.get("executed_action_steps")
    if not isinstance(valid, torch.Tensor) or valid.shape != (count,):
        raise RewardProtocolError("paired success executed-prefix mask changed")
    return gradients, PairedSuccessCreditSummary(
        objective=float(objective),
        target_trajectories=int(trajectory_ids.max()) + 1,
        replay_chunks=count,
        executed_action_steps=int(valid.sum()),
        functional_policy_forwards=forwards,
        functional_policy_backwards=backwards,
        lora_gradient_rms=float(rms),
    )


def backpropagate_lora_cotangent(
    generated: Mapping[str, torch.Tensor],
    lora_gradients: Mapping[str, torch.Tensor],
) -> None:
    """Transport one FP32 LoRA cotangent through the frozen V6 compiler."""

    active = tuple(name for name, value in generated.items() if value.requires_grad)
    if not active or set(lora_gradients) != set(generated):
        raise RewardProtocolError("paired success Writer graph lost LoRA outputs")
    torch.autograd.backward(
        tuple(generated[name] for name in active),
        grad_tensors=tuple(lora_gradients[name] for name in active),
    )
