"""Successful-expert deployed-action credit for the shared Writer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from lerobot.utils.constants import ACTION

from ember.lora import LoRAContract
from ember.reward.loss import functional_executed_prefix_endpoint_action
from ember.reward.protocol import RewardProtocolError
from ember.reward.rollout import policy_flow_noise_cpu


@dataclass(frozen=True)
class SuccessfulExpertOccupancyCreditSummary:
    """One video view's credit on successful expert occupancy."""

    objective: float
    expert_action_distance: float
    successful_trajectories: int
    selected_credit_states: int
    replay_rows: int
    successful_action_steps: int
    matched_expert_student_action_rms: float
    functional_policy_forwards: int
    functional_policy_backwards: int
    lora_gradient_rms: float


def mean_cross_video_task_gradient(
    gradients: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Keep one task's total weight at one while averaging four view gradients."""

    if len(gradients) != 4 or any(
        value.shape != gradients[0].shape or value.dtype != torch.float32
        for value in gradients
    ):
        raise RewardProtocolError("cross-video view gradient panel changed")
    return torch.stack(tuple(gradients)).mean(dim=0)


def cross_video_gradient_geometry(
    gradients: Sequence[torch.Tensor],
) -> dict[str, Any]:
    """Measure whether the shared four-view update descends every view objective."""

    mean = mean_cross_video_task_gradient(gradients)
    rows = torch.stack(tuple(gradients))
    norms = torch.linalg.vector_norm(rows, dim=1)
    mean_norm = torch.linalg.vector_norm(mean)
    dots = rows @ mean
    view_cosines = dots / (norms * mean_norm).clamp_min(1e-30)
    units = rows / norms[:, None].clamp_min(1e-30)
    pairwise = units @ units.T
    offdiag = pairwise[~torch.eye(4, dtype=torch.bool, device=pairwise.device)]
    sample_energy = rows.square().sum(dim=1).mean()
    values = (
        torch.stack(
            (
                offdiag.mean(),
                offdiag.min(),
                offdiag.max(),
                (dots > 0).float().mean(),
                view_cosines.mean(),
                view_cosines.min(),
                mean.square().sum() / sample_energy.clamp_min(1e-30),
            )
        )
        .cpu()
        .tolist()
    )
    return {
        "pairwise_cosine_mean": values[0],
        "pairwise_cosine_minimum": values[1],
        "pairwise_cosine_maximum": values[2],
        "shared_mean_descent_coverage": values[3],
        "view_to_shared_mean_cosine_mean": values[4],
        "view_to_shared_mean_cosine_minimum": values[5],
        "shared_mean_energy_over_view_energy": values[6],
        "per_view": [
            {
                "view_index": index,
                "gradient_norm": float(norm),
                "dot_shared_mean": float(dot),
                "cosine_shared_mean": float(cosine),
            }
            for index, (norm, dot, cosine) in enumerate(
                zip(
                    norms.cpu().tolist(),
                    dots.cpu().tolist(),
                    view_cosines.cpu().tolist(),
                    strict=True,
                )
            )
        ],
    }


def stratified_occupancy_weights(trajectory_ids: torch.Tensor) -> torch.Tensor:
    """Give each successful trajectory equal mass, divided over its states."""

    ids = trajectory_ids.to(dtype=torch.long)
    if ids.ndim != 1 or ids.numel() == 0 or bool((ids < 0).any()):
        raise RewardProtocolError("successful expert replay has no trajectory IDs")
    target_count = int(ids.max()) + 1
    counts = torch.bincount(ids, minlength=target_count)
    if (
        target_count not in {1, 2}
        or counts.shape != (target_count,)
        or bool((counts <= 0).any())
    ):
        raise RewardProtocolError("successful expert trajectory IDs changed")
    return 1.0 / (target_count * counts.index_select(0, ids).to(torch.float32))


def _successful_expert_batch_contract(
    batch: Mapping[str, torch.Tensor],
    trajectory_ids: torch.Tensor,
    endpoint_action_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    action = batch.get(ACTION)
    valid = batch.get("executed_action_steps")
    noise_seed = batch.get("policy_noise_seed")
    if (
        not isinstance(action, torch.Tensor)
        or action.ndim != 3
        or not isinstance(valid, torch.Tensor)
        or valid.ndim != 1
        or valid.shape[0] != action.shape[0]
        or not isinstance(noise_seed, torch.Tensor)
        or noise_seed.shape != valid.shape
        or action.shape[0] <= 0
        or endpoint_action_batch_size != 8
    ):
        raise RewardProtocolError("successful expert endpoint batch changed")
    weights = stratified_occupancy_weights(trajectory_ids)
    if weights.shape != valid.shape:
        raise RewardProtocolError("successful expert replay weights changed")
    return action, valid, noise_seed, int(action.shape[0]), weights


def _endpoint_observation_batch(
    batch: Mapping[str, torch.Tensor],
    start: int,
    stop: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    excluded = {ACTION, "action_is_pad", "executed_action_steps", "policy_noise_seed"}
    return {
        name: value[start:stop].to(device=device, non_blocking=True)
        for name, value in batch.items()
        if name not in excluded
    }


def _endpoint_noise(
    policy: torch.nn.Module,
    noise_seeds: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return torch.cat(
        [
            policy_flow_noise_cpu(
                seed=int(seed),
                chunk_size=int(policy.config.chunk_size),
                max_action_dim=int(policy.config.max_action_dim),
            )
            for seed in noise_seeds.cpu().tolist()
        ]
    ).to(device=device, non_blocking=True)


def unit_residual_endpoint_distillation(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-state MSE and its detached-RMS-normalized objective."""

    if predicted.shape != target.shape or valid.shape != predicted.shape[:1]:
        raise RewardProtocolError("successful expert prediction shape changed")
    mask = (
        torch.arange(predicted.shape[1], device=predicted.device)[None]
        < valid[:, None]
    )
    denominator = (valid * predicted.shape[-1]).to(torch.float32)
    squared = (
        (predicted.float() - target.detach().float()).square() * mask[:, :, None]
    ).sum(dim=(1, 2)) / denominator
    rms = squared.sqrt()
    normalized = squared / rms.detach().clamp_min(1e-6)
    return rms, normalized


def _gradient_rms(gradients: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return (
        torch.cat([value.flatten() for value in gradients.values()])
        .square()
        .mean()
        .sqrt()
    )


def functional_successful_expert_occupancy_gradient(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, torch.Tensor],
    trajectory_ids: torch.Tensor,
    *,
    endpoint_action_batch_size: int,
    num_inference_steps: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], SuccessfulExpertOccupancyCreditSummary]:
    """Differentiate the deployed endpoint toward successful expert actions."""

    action, valid, noise_seed, state_count, weights = (
        _successful_expert_batch_contract(
            batch, trajectory_ids, endpoint_action_batch_size
        )
    )
    leaves = {
        name: value.detach().requires_grad_(name.endswith(".lora_B.default.weight"))
        for name, value in state.items()
    }
    names = tuple(name for name, value in leaves.items() if value.requires_grad)
    gradients = {
        name: torch.zeros_like(value, dtype=torch.float32)
        for name, value in leaves.items()
    }
    objective = torch.zeros((), dtype=torch.float32, device=device)
    distance = torch.zeros_like(objective)
    forwards = backwards = 0
    for start in range(0, state_count, endpoint_action_batch_size):
        stop = min(start + endpoint_action_batch_size, state_count)
        row_weights = weights[start:stop].to(device=device, non_blocking=True)
        prepared = _endpoint_observation_batch(batch, start, stop, device)
        noise = _endpoint_noise(policy, noise_seed[start:stop], device)
        predicted = functional_executed_prefix_endpoint_action(
            policy,
            leaves,
            contract,
            prepared,
            noise=noise,
            num_steps=num_inference_steps,
        )
        target = action[start:stop].to(device=device, non_blocking=True)
        valid_rows = valid[start:stop].to(device=device, non_blocking=True)
        rms, normalized = unit_residual_endpoint_distillation(
            predicted, target, valid_rows
        )
        scalar = (normalized * row_weights).sum()
        values = torch.autograd.grad(scalar, tuple(leaves[name] for name in names))
        objective.add_(scalar.detach())
        distance.add_((rms.detach() * row_weights).sum())
        for name, gradient in zip(names, values, strict=True):
            gradients[name].add_(gradient.float())
        forwards += 1
        backwards += 1
    gradient_rms = _gradient_rms(gradients)
    if not bool(torch.isfinite(gradient_rms)) or float(gradient_rms) <= 0:
        raise RewardProtocolError("successful expert credit produced invalid LoRA gradient")
    trajectory_count = int(trajectory_ids.max()) + 1
    return gradients, SuccessfulExpertOccupancyCreditSummary(
        objective=float(objective),
        expert_action_distance=float(distance),
        successful_trajectories=trajectory_count,
        selected_credit_states=state_count,
        replay_rows=state_count,
        successful_action_steps=int(valid.sum()),
        matched_expert_student_action_rms=float(distance),
        functional_policy_forwards=forwards,
        functional_policy_backwards=backwards,
        lora_gradient_rms=float(gradient_rms),
    )


@torch.no_grad()
def functional_successful_expert_occupancy_objective(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, torch.Tensor],
    trajectory_ids: torch.Tensor,
    *,
    endpoint_action_batch_size: int,
    num_inference_steps: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the exact deployed successful-expert objective."""

    action, valid, noise_seed, state_count, weights = (
        _successful_expert_batch_contract(
            batch, trajectory_ids, endpoint_action_batch_size
        )
    )
    objective = distance = 0.0
    for start in range(0, state_count, endpoint_action_batch_size):
        stop = min(start + endpoint_action_batch_size, state_count)
        prepared = _endpoint_observation_batch(batch, start, stop, device)
        row_weights = weights[start:stop].to(device=device, non_blocking=True)
        noise = _endpoint_noise(policy, noise_seed[start:stop], device)
        predicted = functional_executed_prefix_endpoint_action(
            policy,
            state,
            contract,
            prepared,
            noise=noise,
            num_steps=num_inference_steps,
        )
        target = action[start:stop].to(device=device, non_blocking=True)
        valid_rows = valid[start:stop].to(device=device, non_blocking=True)
        rms, normalized = unit_residual_endpoint_distillation(
            predicted, target, valid_rows
        )
        objective += float((normalized * row_weights).sum())
        distance += float((rms * row_weights).sum())
    return {
        "expert_action_distance": distance,
        "expert_distillation_objective": objective,
    }


def backpropagate_lora_cotangent(
    generated: Mapping[str, torch.Tensor],
    lora_gradients: Mapping[str, torch.Tensor],
) -> None:
    """Transport one FP32 LoRA cotangent through the frozen V6 compiler."""

    active = tuple(name for name, value in generated.items() if value.requires_grad)
    if not active or set(lora_gradients) != set(generated):
        raise RewardProtocolError("successful expert Writer graph lost LoRA outputs")
    torch.autograd.backward(
        tuple(generated[name] for name in active),
        grad_tensors=tuple(lora_gradients[name] for name in active),
    )
