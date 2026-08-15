"""Common-state winner/loser flow credit for paired AS139/LPCP arms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from lerobot.utils.constants import ACTION

from ember.lora import LoRAContract
from ember.reward.loss import functional_executed_prefix_flow_loss
from ember.reward.protocol import RewardProtocolError, reward_preference_flow_seed


@dataclass(frozen=True)
class PairedCommonStateCreditSummary:
    """One view's preference credit from exact common-state arm pairs."""

    objective: float
    preference_margin: float
    winner_flow_loss: float
    loser_flow_loss: float
    discordant_pairs: int
    replay_rows: int
    common_action_steps: int
    winner_loser_action_rms: float
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
        pair_count not in {1, 2}
        or mc_samples != 4
        or model_config is None
        or float(model_config.time_sampling_beta_alpha) != 1.5
        or float(model_config.time_sampling_beta_beta) != 1.0
        or float(model_config.time_sampling_scale) != 0.999
        or float(model_config.time_sampling_offset) != 0.001
    ):
        raise RewardProtocolError("common-state flow panel changed")
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


def _common_state_batch_contract(
    batch: Mapping[str, torch.Tensor], physical_microbatch_size: int
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    action = batch.get(ACTION)
    valid = batch.get("executed_action_steps")
    if (
        not isinstance(action, torch.Tensor)
        or action.ndim != 3
        or not isinstance(valid, torch.Tensor)
        or valid.ndim != 1
        or valid.shape[0] != action.shape[0]
        or action.shape[0] not in {2, 4}
        or physical_microbatch_size < 2
        or not torch.equal(valid[0::2], valid[1::2])
    ):
        raise RewardProtocolError("common-state preference batch changed")
    pair_count = int(action.shape[0] // 2)
    steps = torch.arange(action.shape[1], device=action.device)[None]
    mask = steps < valid[0::2, None]
    difference = action[0::2].float() - action[1::2].float()
    squared = (difference.square() * mask[:, :, None]).sum()
    denominator = valid[0::2].sum() * action.shape[2]
    action_rms = float((squared / denominator).sqrt())
    if not bool(torch.isfinite(torch.tensor(action_rms))) or action_rms <= 0:
        raise RewardProtocolError("winner and loser first actions are identical")
    return action, valid, pair_count, action_rms


def functional_paired_common_state_lora_gradient(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, torch.Tensor],
    *,
    mc_samples: int,
    physical_microbatch_size: int,
    flow_seed_root: int,
    cycle: int,
    global_task_id: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], PairedCommonStateCreditSummary]:
    """Differentiate softplus(winner loss - loser loss) through LoRA leaves."""

    action, valid, pair_count, action_rms = _common_state_batch_contract(
        batch, physical_microbatch_size
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
            scalar = F.softplus(winners - losers).sum() / (
                pair_count * mc_samples
            )
            values = torch.autograd.grad(scalar, tuple(leaves[name] for name in names))
            objective.add_(scalar.detach())
            winner_total.add_(winners.detach().sum() / (pair_count * mc_samples))
            loser_total.add_(losers.detach().sum() / (pair_count * mc_samples))
            for name, gradient in zip(names, values, strict=True):
                gradients[name].add_(gradient.float())
            forwards += 1
            backwards += 1
    rms = _gradient_rms(gradients)
    if not bool(torch.isfinite(rms)) or float(rms) <= 0:
        raise RewardProtocolError("common-state preference produced invalid LoRA credit")
    margin = winner_total - loser_total
    return gradients, PairedCommonStateCreditSummary(
        objective=float(objective),
        preference_margin=float(margin),
        winner_flow_loss=float(winner_total),
        loser_flow_loss=float(loser_total),
        discordant_pairs=pair_count,
        replay_rows=2 * pair_count,
        common_action_steps=int(valid[0::2].sum()),
        winner_loser_action_rms=action_rms,
        functional_policy_forwards=forwards,
        functional_policy_backwards=backwards,
        lora_gradient_rms=float(rms),
    )


@torch.no_grad()
def functional_paired_common_state_margin(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, torch.Tensor],
    *,
    mc_samples: int,
    flow_seed_root: int,
    cycle: int,
    global_task_id: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the exact paired panel without constructing a Writer graph."""

    _, _, pair_count, _ = _common_state_batch_contract(batch, 2)
    noises, times = _flow_sample_panel(
        policy,
        pair_count=pair_count,
        mc_samples=mc_samples,
        seed_root=flow_seed_root,
        cycle=cycle,
        global_task_id=global_task_id,
        device=device,
    )
    prepared = {
        name: value.to(device=device, non_blocking=True) for name, value in batch.items()
    }
    winner = loser = objective = 0.0
    for mc_index in range(mc_samples):
        losses = functional_executed_prefix_flow_loss(
            policy,
            state,
            contract,
            prepared,
            noise=noises[mc_index].repeat_interleave(2, dim=0),
            time=times[mc_index].repeat_interleave(2),
        ).float().reshape(-1, 2)
        winner += float(losses[:, 0].mean()) / mc_samples
        loser += float(losses[:, 1].mean()) / mc_samples
        objective += float(
            F.softplus(losses[:, 0] - losses[:, 1]).mean()
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
        raise RewardProtocolError("common-state Writer graph lost LoRA outputs")
    torch.autograd.backward(
        tuple(generated[name] for name in active),
        grad_tensors=tuple(lora_gradients[name] for name in active),
    )
