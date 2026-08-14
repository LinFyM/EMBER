"""On-policy preference and successful-support credit for one shared Writer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from lerobot.utils.constants import ACTION
from scipy.optimize import nnls

from ember.lora import LoRAContract
from ember.reward.loss import functional_executed_prefix_flow_loss
from ember.reward.protocol import RewardProtocolError, reward_preference_flow_seed


@dataclass(frozen=True)
class RewardPreferenceSummary:
    objective: float
    support_objective: float
    successes: int
    replay_chunks: int
    executed_action_steps: int
    functional_policy_forwards: int
    preference_policy_backwards: int
    support_policy_backwards: int
    preference_lora_gradient_rms: float
    support_lora_gradient_rms: float


@dataclass(frozen=True)
class FinalSupportProjectionSummary:
    support_constraints: int
    constraint_rank: int
    raw_violation_count: int
    final_violation_count: int
    active_dual_count: int
    projection_changed: bool
    raw_constraint_max: float
    final_constraint_max: float
    numerical_tolerance: float
    raw_delta_rms: float
    projected_delta_rms: float
    projected_to_raw_energy_ratio: float
    raw_projected_cosine: float
    raw_preference_directional_derivative: float
    projected_preference_directional_derivative: float
    projected_preference_descent_ratio: float


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


def successful_episode_chunk_weights(
    episode_ids: torch.Tensor, successes: torch.Tensor
) -> torch.Tensor:
    leave_one_out_binary_advantages(successes)
    ids = episode_ids.to(dtype=torch.long)
    if (
        ids.ndim != 1
        or ids.numel() == 0
        or bool((ids < 0).any())
        or bool((ids >= 4).any())
    ):
        raise RewardProtocolError("success support episode IDs changed")
    counts = torch.bincount(ids, minlength=4)
    success_count = int(successes.sum())
    if counts.shape != (4,) or bool((counts <= 0).any()) or success_count <= 0:
        raise RewardProtocolError("success support requires a complete successful panel")
    selected = successes.to(device=ids.device).index_select(0, ids).bool()
    weights = torch.zeros(ids.shape, dtype=torch.float32, device=ids.device)
    weights[selected] = 1.0 / (
        success_count * counts.index_select(0, ids[selected]).to(dtype=torch.float32)
    )
    return weights


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


def _gradient_rms(gradients: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return (
        torch.cat([value.flatten() for value in gradients.values()])
        .square()
        .mean()
        .sqrt()
    )


def _credit_weights(
    action: torch.Tensor | None,
    episode_ids: torch.Tensor,
    successes: torch.Tensor,
) -> tuple[int, bool, bool, torch.Tensor, torch.Tensor]:
    if not isinstance(action, torch.Tensor) or action.ndim != 3:
        raise RewardProtocolError("reward replay action batch changed")
    weights, advantages = episode_equal_chunk_weights(episode_ids, successes)
    mixed = bool(torch.count_nonzero(advantages))
    has_success = bool(successes.sum())
    if not mixed and not has_success:
        raise RewardProtocolError("all-failure panel entered reward CFM")
    support_weights = (
        successful_episode_chunk_weights(episode_ids, successes)
        if has_success
        else torch.zeros_like(weights)
    )
    return int(action.shape[0]), mixed, has_success, weights, support_weights


def _accumulate_functional_credit(
    policy: torch.nn.Module,
    leaves: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, torch.Tensor],
    weights: torch.Tensor,
    support_weights: torch.Tensor,
    noises: torch.Tensor,
    times: torch.Tensor,
    *,
    mixed: bool,
    mc_samples: int,
    physical_microbatch_size: int,
    device: torch.device,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    int,
]:
    names = tuple(leaves)
    preference = {
        name: torch.zeros_like(value, dtype=torch.float32)
        for name, value in leaves.items()
    }
    support = {
        name: torch.zeros_like(value, dtype=torch.float32)
        for name, value in leaves.items()
    }
    objective = torch.zeros((), dtype=torch.float32, device=device)
    support_objective = torch.zeros((), dtype=torch.float32, device=device)
    forwards = preference_backwards = support_backwards = 0
    for start in range(0, len(weights), physical_microbatch_size):
        stop = min(start + physical_microbatch_size, len(weights))
        sliced = {
            name: value[start:stop].to(device=device, non_blocking=True)
            for name, value in batch.items()
        }
        chunk_weights = weights[start:stop].to(device=device, non_blocking=True)
        chunk_support = support_weights[start:stop].to(
            device=device, non_blocking=True
        )
        has_support = bool(torch.count_nonzero(chunk_support))
        for mc_index in range(mc_samples):
            per_chunk = functional_executed_prefix_flow_loss(
                policy,
                leaves,
                contract,
                sliced,
                noise=noises[mc_index, start:stop],
                time=times[mc_index, start:stop],
            )
            if mixed:
                scalar = (per_chunk.float() * chunk_weights).sum() / mc_samples
                gradients = torch.autograd.grad(
                    scalar,
                    tuple(leaves[name] for name in names),
                    retain_graph=has_support,
                )
                objective.add_(scalar.detach())
                for name, gradient in zip(names, gradients, strict=True):
                    preference[name].add_(gradient.float())
                preference_backwards += 1
            if has_support:
                scalar = (per_chunk.float() * chunk_support).sum() / mc_samples
                gradients = torch.autograd.grad(
                    scalar, tuple(leaves[name] for name in names)
                )
                support_objective.add_(scalar.detach())
                for name, gradient in zip(names, gradients, strict=True):
                    support[name].add_(gradient.float())
                support_backwards += 1
            forwards += 1
    return (
        preference,
        support,
        objective,
        support_objective,
        forwards,
        preference_backwards,
        support_backwards,
    )


def functional_reward_lora_gradients(
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
) -> tuple[
    dict[str, torch.Tensor] | None,
    dict[str, torch.Tensor] | None,
    RewardPreferenceSummary,
]:
    """Differentiate preference and successful support from one policy forward."""

    count, mixed, has_success, weights, support_weights = _credit_weights(
        batch.get(ACTION), episode_ids, successes
    )
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
    (
        preference_gradients,
        support_gradients,
        objective,
        support_objective,
        forwards,
        preference_backwards,
        support_backwards,
    ) = _accumulate_functional_credit(
        policy,
        leaves,
        contract,
        batch,
        weights,
        support_weights,
        noises,
        times,
        mixed=mixed,
        mc_samples=mc_samples,
        physical_microbatch_size=physical_microbatch_size,
        device=device,
    )
    preference_rms = _gradient_rms(preference_gradients)
    support_rms = _gradient_rms(support_gradients)
    if mixed and (
        not bool(torch.isfinite(preference_rms)) or float(preference_rms) <= 0
    ):
        raise RewardProtocolError("mixed reward panel produced invalid preference credit")
    if has_success and (
        not bool(torch.isfinite(support_rms)) or float(support_rms) <= 0
    ):
        raise RewardProtocolError("successful panel produced invalid support credit")
    valid = batch["executed_action_steps"]
    return (
        preference_gradients if mixed else None,
        support_gradients if has_success else None,
        RewardPreferenceSummary(
            objective=float(objective),
            support_objective=float(support_objective),
            successes=int(successes.sum()),
            replay_chunks=count,
            executed_action_steps=int(valid.sum()),
            functional_policy_forwards=forwards,
            preference_policy_backwards=preference_backwards,
            support_policy_backwards=support_backwards,
            preference_lora_gradient_rms=float(preference_rms),
            support_lora_gradient_rms=float(support_rms),
        ),
    )


def backpropagate_lora_cotangent(
    generated: Mapping[str, torch.Tensor],
    lora_gradients: Mapping[str, torch.Tensor],
    *,
    retain_graph: bool = False,
) -> None:
    """Transport one FP32 LoRA cotangent through the native compiler."""

    active = tuple(name for name, value in generated.items() if value.requires_grad)
    if not active or set(lora_gradients) != set(generated):
        raise RewardProtocolError("reward Writer graph lost generated LoRA outputs")
    torch.autograd.backward(
        tuple(generated[name] for name in active),
        grad_tensors=tuple(lora_gradients[name] for name in active),
        retain_graph=retain_graph,
    )


def _validate_projection_batch(
    raw_delta: torch.Tensor,
    support_gradients: torch.Tensor,
    support_mask: torch.Tensor,
    preference_gradient: torch.Tensor,
) -> None:
    same_device = len(
        {
            raw_delta.device,
            preference_gradient.device,
            support_gradients.device,
            support_mask.device,
        }
    ) == 1
    finite = all(
        bool(torch.isfinite(value).all())
        for value in (raw_delta, preference_gradient, support_gradients)
    )
    if (
        raw_delta.ndim != 1
        or preference_gradient.shape != raw_delta.shape
        or support_gradients.ndim != 2
        or support_gradients.shape[1] != raw_delta.numel()
        or support_mask.shape != (support_gradients.shape[0],)
        or support_mask.dtype != torch.bool
        or not same_device
        or not finite
    ):
        raise RewardProtocolError("invalid final support projection batch")


def _solve_support_projection(
    raw64: np.ndarray, selected: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    constraints = int(selected.shape[0])
    if not constraints:
        empty = np.empty((0,), dtype=np.float64)
        return raw64, np.empty((0, raw64.size)), empty, 0, 0, 0
    normals = selected.detach().cpu().numpy().astype(np.float64)
    norms = np.linalg.norm(normals, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise RewardProtocolError("success support contains a zero gradient")
    normals /= norms[:, None]
    gram = normals @ normals.T
    gram = (gram + gram.T) * 0.5
    raw_constraints = normals @ raw64
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    maximum = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    tolerance = constraints * np.finfo(np.float64).eps * maximum
    positive = eigenvalues > tolerance
    rank = int(np.count_nonzero(positive))
    violations = int(np.count_nonzero(raw_constraints > 0))
    if not violations:
        return raw64, normals, raw_constraints, rank, 0, 0
    if rank <= 0:
        raise RewardProtocolError("violated support constraints have zero rank")
    vectors = eigenvectors[:, positive]
    roots = np.sqrt(eigenvalues[positive])
    square_root = (vectors * roots) @ vectors.T
    inverse_root = (vectors / roots) @ vectors.T
    dual, _ = nnls(square_root, inverse_root @ raw_constraints)
    projected64 = raw64 - normals.T @ dual
    return (
        projected64,
        normals,
        raw_constraints,
        rank,
        violations,
        int(np.count_nonzero(dual > 0)),
    )


def _summarize_projection(
    raw64: np.ndarray,
    projected64: np.ndarray,
    preference64: np.ndarray,
    raw_constraints: np.ndarray,
    final_constraints: np.ndarray,
    *,
    rank: int,
    raw_violations: int,
    active_dual: int,
    changed: bool,
) -> FinalSupportProjectionSummary:
    raw_energy = float(raw64 @ raw64)
    projected_energy = float(projected64 @ projected64)
    projected_norm = math.sqrt(projected_energy)
    tolerance = (
        64
        * np.finfo(np.float32).eps
        * max(projected_norm, np.finfo(np.float32).tiny)
    )
    raw_directional = float(preference64 @ raw64)
    projected_directional = float(preference64 @ projected64)
    descent_ratio = (
        max(0.0, -projected_directional)
        / max(-raw_directional, np.finfo(np.float64).tiny)
        if raw_directional < 0
        else 0.0
    )
    denominator = max(
        math.sqrt(raw_energy * projected_energy), np.finfo(np.float64).tiny
    )
    constraints = int(raw_constraints.size)
    return FinalSupportProjectionSummary(
        support_constraints=constraints,
        constraint_rank=rank,
        raw_violation_count=raw_violations,
        final_violation_count=int(np.count_nonzero(final_constraints > tolerance)),
        active_dual_count=active_dual,
        projection_changed=changed,
        raw_constraint_max=(float(raw_constraints.max()) if constraints else 0.0),
        final_constraint_max=(float(final_constraints.max()) if constraints else 0.0),
        numerical_tolerance=float(tolerance),
        raw_delta_rms=math.sqrt(raw_energy / max(1, raw64.size)),
        projected_delta_rms=math.sqrt(projected_energy / max(1, raw64.size)),
        projected_to_raw_energy_ratio=(
            projected_energy / max(raw_energy, np.finfo(np.float64).tiny)
        ),
        raw_projected_cosine=float(raw64 @ projected64) / denominator,
        raw_preference_directional_derivative=raw_directional,
        projected_preference_directional_derivative=projected_directional,
        projected_preference_descent_ratio=float(descent_ratio),
    )


def project_final_parameter_delta(
    raw_delta: torch.Tensor,
    support_gradients: torch.Tensor,
    support_mask: torch.Tensor,
    preference_gradient: torch.Tensor,
) -> tuple[torch.Tensor, FinalSupportProjectionSummary]:
    """Project an actual shared AdamW delta into task support half-spaces."""

    _validate_projection_batch(
        raw_delta, support_gradients, support_mask, preference_gradient
    )
    selected = support_gradients[support_mask].float()
    raw64 = raw_delta.detach().float().cpu().numpy().astype(np.float64)
    preference64 = (
        preference_gradient.detach().float().cpu().numpy().astype(np.float64)
    )
    (
        solved64,
        normals64,
        raw_constraints,
        rank,
        raw_violations,
        active_dual,
    ) = _solve_support_projection(raw64, selected)
    projected = torch.from_numpy(solved64.astype(np.float32)).to(raw_delta.device)
    projected64_actual = projected.detach().cpu().numpy().astype(np.float64)
    final_constraints = normals64 @ projected64_actual
    summary = _summarize_projection(
        raw64,
        projected64_actual,
        preference64,
        raw_constraints,
        final_constraints,
        rank=rank,
        raw_violations=raw_violations,
        active_dual=active_dual,
        changed=not torch.equal(projected, raw_delta),
    )
    if summary.final_violation_count:
        raise RewardProtocolError("final support projection violated a constraint")
    if not all(
        math.isfinite(value)
        for value in summary.__dict__.values()
        if isinstance(value, float)
    ):
        raise RewardProtocolError("final support projection became non-finite")
    return projected, summary
