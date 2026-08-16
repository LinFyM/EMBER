"""Matched-batch stratified occupancy flow credit for paired AS139/LPCP arms."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from lerobot.utils.constants import ACTION

from ember.lora import LoRAContract
from ember.reward.loss import functional_executed_prefix_flow_loss
from ember.reward.protocol import RewardProtocolError, reward_preference_flow_seed


@dataclass(frozen=True)
class MatchedStratifiedOccupancyCreditSummary:
    """One view's preference credit over a stratified successful occupancy."""

    objective: float
    preference_margin: float
    winner_flow_loss: float
    loser_flow_loss: float
    discordant_trajectories: int
    selected_credit_pairs: int
    replay_rows: int
    successful_action_steps: int
    matched_winner_loser_action_rms: float
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


def maximum_margin_common_descent_gradient(
    gradients: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the symmetric max-min descent direction for four view gradients."""

    if len(gradients) != 4 or any(
        value.shape != gradients[0].shape or value.dtype != torch.float32
        for value in gradients
    ):
        raise RewardProtocolError("maximum-margin view gradient panel changed")
    rows = torch.stack(tuple(gradients))
    gram = (rows @ rows.T).to(dtype=torch.float64)
    scale = gram.diagonal().max().clamp_min(1e-30)
    tolerance = scale * 1e-6
    candidates, scores = [], []
    for size in range(1, 5):
        for support in combinations(range(4), size):
            indices = torch.tensor(support, device=rows.device)
            sub = gram.index_select(0, indices).index_select(1, indices)
            ones = torch.ones(size, dtype=torch.float64, device=rows.device)
            inverse_ones = torch.linalg.pinv(sub, hermitian=True) @ ones
            denominator = torch.dot(ones, inverse_ones)
            weights = torch.zeros(4, dtype=torch.float64, device=rows.device)
            weights.index_copy_(0, indices, inverse_ones / denominator)
            weights = weights.clamp_min(0)
            weights.div_(weights.sum())
            dots = gram @ weights
            objective = torch.dot(weights, dots)
            feasible = (dots >= objective - tolerance).all()
            candidates.append(weights)
            scores.append(
                torch.where(
                    feasible,
                    objective,
                    torch.full_like(objective, torch.inf),
                )
            )
    score = torch.stack(scores)
    if not bool(torch.isfinite(score).any()):
        raise RewardProtocolError("maximum-margin simplex solve failed")
    weights = torch.stack(candidates)[score.argmin()].to(dtype=torch.float32)
    combination = weights @ rows
    mean = rows.mean(dim=0)
    mean_norm = torch.linalg.vector_norm(mean)
    combination_norm = torch.linalg.vector_norm(combination)
    if float(mean_norm) <= 0 or float(combination_norm) <= 0:
        raise RewardProtocolError("maximum-margin common direction is empty")
    direction = combination * (mean_norm / combination_norm)
    mean_dots = rows @ (mean / mean_norm)
    direction_dots = rows @ (direction / mean_norm)
    values = torch.stack(
        (
            mean_norm,
            combination_norm,
            mean_dots.min(),
            direction_dots.min(),
            torch.dot(mean, direction) / mean_norm.square(),
        )
    ).cpu().tolist()
    return direction, {
        "kind": "four_view_maximum_margin_common_descent",
        "simplex_weights": weights.cpu().tolist(),
        "active_weight_count": int(torch.count_nonzero(weights > 1e-6)),
        "equal_mean_l2": values[0],
        "minimum_norm_combination_l2": values[1],
        "equal_mean_worst_unit_directional_derivative": values[2],
        "maximum_margin_worst_unit_directional_derivative": values[3],
        "worst_directional_derivative_gain": values[3] / values[2],
        "maximum_margin_to_equal_mean_cosine": values[4],
        "per_view_equal_mean_unit_directional_derivative": mean_dots.cpu().tolist(),
        "per_view_maximum_margin_unit_directional_derivative": (
            direction_dots.cpu().tolist()
        ),
    }


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
    offdiag = pairwise[
        ~torch.eye(4, dtype=torch.bool, device=pairwise.device)
    ]
    sample_energy = rows.square().sum(dim=1).mean()
    values = torch.stack(
        (
            offdiag.mean(),
            offdiag.min(),
            offdiag.max(),
            (dots > 0).float().mean(),
            view_cosines.mean(),
            view_cosines.min(),
            mean.square().sum() / sample_energy.clamp_min(1e-30),
        )
    ).cpu().tolist()
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


def _flow_sample_panel(
    policy: torch.nn.Module,
    *,
    pair_count: int,
    mc_samples: int,
    seed_root: int,
    cycle: int,
    global_task_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_config = getattr(getattr(policy, "model", None), "config", None)
    if (
        pair_count <= 0
        or mc_samples != 4
        or model_config is None
        or float(model_config.time_sampling_beta_alpha) != 1.5
        or float(model_config.time_sampling_beta_beta) != 1.0
        or float(model_config.time_sampling_scale) != 0.999
        or float(model_config.time_sampling_offset) != 0.001
    ):
        raise RewardProtocolError("matched occupancy flow panel changed")
    shape = (
        pair_count,
        int(policy.config.chunk_size),
        int(policy.config.max_action_dim),
    )
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
        uniform = torch.rand(pair_count, generator=generator, device=device)
        times.append(uniform.pow(2.0 / 3.0).mul_(0.999).add_(0.001))
    return torch.stack(noises), torch.stack(times)


def _gradient_rms(gradients: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return (
        torch.cat([value.flatten() for value in gradients.values()])
        .square()
        .mean()
        .sqrt()
    )


def stratified_occupancy_pair_weights(
    trajectory_ids: torch.Tensor,
) -> torch.Tensor:
    """Give each successful trajectory equal mass and divide it over replans."""

    ids = trajectory_ids.to(dtype=torch.long)
    if ids.ndim != 1 or ids.numel() == 0 or bool((ids < 0).any()):
        raise RewardProtocolError("stratified occupancy replay has no trajectory IDs")
    target_count = int(ids.max()) + 1
    counts = torch.bincount(ids, minlength=target_count)
    if (
        target_count not in {1, 2}
        or counts.shape != (target_count,)
        or bool((counts <= 0).any())
    ):
        raise RewardProtocolError("stratified occupancy trajectory IDs changed")
    return 1.0 / (
        target_count * counts.index_select(0, ids).to(dtype=torch.float32)
    )


def _matched_occupancy_batch_contract(
    batch: Mapping[str, torch.Tensor],
    trajectory_ids: torch.Tensor,
    physical_microbatch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int, float, torch.Tensor]:
    action = batch.get(ACTION)
    valid = batch.get("executed_action_steps")
    if (
        not isinstance(action, torch.Tensor)
        or action.ndim != 3
        or not isinstance(valid, torch.Tensor)
        or valid.ndim != 1
        or valid.shape[0] != action.shape[0]
        or action.shape[0] <= 0
        or action.shape[0] % 2
        or physical_microbatch_size < 2
        or not torch.equal(valid[0::2], valid[1::2])
    ):
        raise RewardProtocolError("matched occupancy preference batch changed")
    pair_count = int(action.shape[0] // 2)
    weights = stratified_occupancy_pair_weights(trajectory_ids)
    if weights.shape != (pair_count,):
        raise RewardProtocolError("stratified occupancy replay weights changed")
    steps = torch.arange(action.shape[1], device=action.device)[None]
    mask = steps < valid[0::2, None]
    difference = action[0::2].float() - action[1::2].float()
    squared = (difference.square() * mask[:, :, None]).sum()
    denominator = valid[0::2].sum() * action.shape[2]
    action_rms = float((squared / denominator).sqrt())
    if not bool(torch.isfinite(torch.tensor(action_rms))) or action_rms <= 0:
        raise RewardProtocolError("matched winner and loser actions are identical")
    return action, valid, pair_count, action_rms, weights


def functional_matched_stratified_occupancy_lora_gradient(
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
) -> tuple[dict[str, torch.Tensor], MatchedStratifiedOccupancyCreditSummary]:
    """Differentiate matched winner versus loser preference over selected strata."""

    action, valid, pair_count, action_rms, weights = (
        _matched_occupancy_batch_contract(
            batch, trajectory_ids, physical_microbatch_size
        )
    )
    noises, times = _flow_sample_panel(
        policy,
        pair_count=pair_count,
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
    winner_total = torch.zeros_like(objective)
    loser_total = torch.zeros_like(objective)
    forwards = backwards = 0
    pairs_per_batch = max(1, physical_microbatch_size // 2)
    for pair_start in range(0, pair_count, pairs_per_batch):
        pair_stop = min(pair_start + pairs_per_batch, pair_count)
        start, stop = 2 * pair_start, 2 * pair_stop
        pair_weights = weights[pair_start:pair_stop].to(
            device=device, non_blocking=True
        )
        sliced = {
            name: value[start:stop].to(device=device, non_blocking=True)
            for name, value in batch.items()
        }
        for mc_index in range(mc_samples):
            per_chunk = functional_executed_prefix_flow_loss(
                policy,
                leaves,
                contract,
                sliced,
                noise=noises[mc_index, pair_start:pair_stop].repeat_interleave(
                    2, dim=0
                ),
                time=times[mc_index, pair_start:pair_stop].repeat_interleave(2),
            )
            pair_losses = per_chunk.float().reshape(-1, 2)
            winners, losers = pair_losses[:, 0], pair_losses[:, 1]
            scalar = (F.softplus(winners - losers) * pair_weights).sum() / mc_samples
            values = torch.autograd.grad(scalar, tuple(leaves[name] for name in names))
            objective.add_(scalar.detach())
            winner_total.add_(
                (winners.detach() * pair_weights).sum() / mc_samples
            )
            loser_total.add_((losers.detach() * pair_weights).sum() / mc_samples)
            for name, gradient in zip(names, values, strict=True):
                gradients[name].add_(gradient.float())
            forwards += 1
            backwards += 1
    rms = _gradient_rms(gradients)
    if not bool(torch.isfinite(rms)) or float(rms) <= 0:
        raise RewardProtocolError(
            "matched occupancy preference produced invalid LoRA credit"
        )
    margin = winner_total - loser_total
    return gradients, MatchedStratifiedOccupancyCreditSummary(
        objective=float(objective),
        preference_margin=float(margin),
        winner_flow_loss=float(winner_total),
        loser_flow_loss=float(loser_total),
        discordant_trajectories=int(trajectory_ids.max()) + 1,
        selected_credit_pairs=pair_count,
        replay_rows=2 * pair_count,
        successful_action_steps=int(valid[0::2].sum()),
        matched_winner_loser_action_rms=action_rms,
        functional_policy_forwards=forwards,
        functional_policy_backwards=backwards,
        lora_gradient_rms=float(rms),
    )


@torch.no_grad()
def functional_matched_stratified_occupancy_margin(
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
) -> dict[str, float]:
    """Evaluate the retained occupancy panel without constructing a Writer graph."""

    _, _, pair_count, _, weights = _matched_occupancy_batch_contract(
        batch, trajectory_ids, physical_microbatch_size
    )
    noises, times = _flow_sample_panel(
        policy,
        pair_count=pair_count,
        mc_samples=mc_samples,
        seed_root=flow_seed_root,
        cycle=cycle,
        global_task_id=global_task_id,
        device=device,
    )
    winner = loser = objective = 0.0
    pairs_per_batch = max(1, physical_microbatch_size // 2)
    for pair_start in range(0, pair_count, pairs_per_batch):
        pair_stop = min(pair_start + pairs_per_batch, pair_count)
        start, stop = 2 * pair_start, 2 * pair_stop
        prepared = {
            name: value[start:stop].to(device=device, non_blocking=True)
            for name, value in batch.items()
        }
        pair_weights = weights[pair_start:pair_stop].to(
            device=device, non_blocking=True
        )
        for mc_index in range(mc_samples):
            losses = functional_executed_prefix_flow_loss(
                policy,
                state,
                contract,
                prepared,
                noise=noises[mc_index, pair_start:pair_stop].repeat_interleave(
                    2, dim=0
                ),
                time=times[mc_index, pair_start:pair_stop].repeat_interleave(2),
            ).float().reshape(-1, 2)
            winner += float((losses[:, 0] * pair_weights).sum()) / mc_samples
            loser += float((losses[:, 1] * pair_weights).sum()) / mc_samples
            objective += float(
                (
                    F.softplus(losses[:, 0] - losses[:, 1])
                    * pair_weights
                ).sum()
            ) / mc_samples
    margin = winner - loser
    return {
        "winner_flow_loss": winner,
        "loser_flow_loss": loser,
        "preference_margin": margin,
        "preference_objective": objective,
    }


def backpropagate_lora_cotangent(
    generated: Mapping[str, torch.Tensor],
    lora_gradients: Mapping[str, torch.Tensor],
) -> None:
    """Transport one FP32 LoRA cotangent through the frozen V6 compiler."""

    active = tuple(name for name, value in generated.items() if value.requires_grad)
    if not active or set(lora_gradients) != set(generated):
        raise RewardProtocolError("matched occupancy Writer graph lost LoRA outputs")
    torch.autograd.backward(
        tuple(generated[name] for name in active),
        grad_tensors=tuple(lora_gradients[name] for name in active),
    )
