"""Binary K4 reward credit transported through the frozen-v6 Program leaf."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from lerobot.utils.constants import ACTION

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_step import (
    GeneratedConditionGraph,
    program_cotangent,
)
from ember.lora import LoRAContract
from ember.reward.loss import functional_executed_prefix_flow_loss
from ember.reward.protocol import RewardProtocolError, flow_sample_seed


@dataclass(frozen=True)
class RewardProgramCreditSummary:
    """Task-local evidence for one direct signed Program cotangent."""

    objective: float
    successes: int
    failures: int
    mixed: bool
    positive_episodes: int
    negative_episodes: int
    zero_episodes: int
    replay_chunks: int
    executed_action_steps: int
    mc_samples: int
    functional_policy_forwards: int
    program_cotangent_rms: float


def leave_one_out_binary_advantages(successes: torch.Tensor) -> torch.Tensor:
    """Return task-relative binary LOO credit for the sealed K4 panel."""

    if (
        successes.ndim != 1
        or successes.numel() != 4
        or not bool(torch.isfinite(successes).all())
        or bool(((successes != 0) & (successes != 1)).any())
    ):
        raise RewardProtocolError("reward Program credit requires four binary outcomes")
    return (4 * successes - successes.sum()) / 3


def episode_equal_chunk_weights(
    episode_ids: torch.Tensor,
    successes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map K4 LOO advantages to `A_e / (4 * chunks_e)` chunk weights."""

    advantages = leave_one_out_binary_advantages(successes).to(
        device=episode_ids.device,
        dtype=torch.float32,
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
    """Generate a complete keyed task panel before physical microbatch slicing."""

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
    shape = (
        count,
        int(policy.config.chunk_size),
        int(policy.config.max_action_dim),
    )
    noises = []
    times = []
    for mc_index in range(mc_samples):
        generator = torch.Generator(device=device).manual_seed(
            flow_sample_seed(
                seed_root,
                cycle=cycle,
                global_task_id=global_task_id,
                mc_index=mc_index,
            )
        )
        noises.append(
            torch.randn(
                shape,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
        )
        uniform = torch.rand(
            count,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        times.append(uniform.pow(2.0 / 3.0).mul_(0.999).add_(0.001))
    return torch.stack(noises), torch.stack(times)


def _batch_slice_to_device(
    batch: Mapping[str, torch.Tensor],
    start: int,
    stop: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: value[start:stop].to(device=device, non_blocking=True)
        for name, value in batch.items()
    }


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
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Differentiate one direct signed on-policy CFM objective through LoRA only."""

    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise RewardProtocolError("reward functional policy must remain frozen")
    action = batch.get(ACTION)
    if not isinstance(action, torch.Tensor) or action.ndim != 3:
        raise RewardProtocolError("reward replay action batch changed")
    count = int(action.shape[0])
    valid = batch.get("executed_action_steps")
    if (
        not isinstance(valid, torch.Tensor)
        or valid.shape != (count,)
        or bool((valid <= 0).any())
        or bool((valid > action.shape[1]).any())
    ):
        raise RewardProtocolError("reward replay executed-prefix lengths changed")
    if physical_microbatch_size <= 0:
        raise RewardProtocolError("invalid reward replay physical microbatch")
    weights, advantages = episode_equal_chunk_weights(episode_ids, successes)
    if bool((advantages == 0).all()):
        return (
            {name: torch.zeros_like(value) for name, value in state.items()},
            {
                "objective": 0.0,
                "functional_policy_forwards": 0,
                "advantages": advantages,
            },
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
    names = tuple(state)
    gradient_sum = {
        name: torch.zeros_like(value, dtype=torch.float32)
        for name, value in state.items()
    }
    objective = torch.zeros((), dtype=torch.float32, device=device)
    forwards = 0
    leaves = {
        name: value.detach().requires_grad_(True) for name, value in state.items()
    }
    for start in range(0, count, physical_microbatch_size):
        stop = min(start + physical_microbatch_size, count)
        sliced = _batch_slice_to_device(batch, start, stop, device)
        sliced_weights = weights[start:stop].to(device=device, non_blocking=True)
        for mc_index in range(mc_samples):
            per_chunk, _ = functional_executed_prefix_flow_loss(
                policy,
                leaves,
                contract,
                sliced,
                noise=noises[mc_index, start:stop],
                time=times[mc_index, start:stop],
                validate_prefix_values=False,
                collect_details=False,
            )
            scalar = (
                per_chunk.to(dtype=torch.float32) * sliced_weights
            ).sum() / mc_samples
            gradients = torch.autograd.grad(
                scalar,
                tuple(leaves[name] for name in names),
            )
            objective.add_(scalar.detach())
            for name, gradient in zip(names, gradients, strict=True):
                gradient_sum[name].add_(gradient.to(dtype=torch.float32))
            forwards += 1
    return (
        gradient_sum,
        {
            "objective": objective,
            "functional_policy_forwards": forwards,
            "advantages": advantages,
        },
    )


def reward_program_cotangent(
    graph: GeneratedConditionGraph,
    *,
    policy: torch.nn.Module,
    contract: LoRAContract,
    batch: Mapping[str, torch.Tensor],
    episode_ids: torch.Tensor,
    successes: torch.Tensor,
    mc_samples: int,
    physical_microbatch_size: int,
    flow_seed_root: int,
    cycle: int,
    global_task_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, RewardProgramCreditSummary]:
    """Return one `[320,256]` reward cotangent and its task-local evidence."""

    weights, advantages = episode_equal_chunk_weights(episode_ids, successes)
    del weights
    success_count = int(successes.sum())
    mixed = 0 < success_count < 4
    if mixed:
        gradients, details = functional_reward_lora_gradient(
            policy,
            graph.correct_lora,
            contract,
            batch,
            episode_ids,
            successes,
            mc_samples=mc_samples,
            physical_microbatch_size=physical_microbatch_size,
            flow_seed_root=flow_seed_root,
            cycle=cycle,
            global_task_id=global_task_id,
            device=device,
        )
        cotangent = program_cotangent(graph, gradients)
    else:
        details = {
            "objective": torch.zeros((), dtype=torch.float32, device=device),
            "functional_policy_forwards": 0,
        }
        cotangent = torch.zeros_like(graph.program_leaf[0], dtype=torch.float32)
    if cotangent.shape != (320, 256):
        raise ExpertManifoldError("reward Program cotangent became invalid")
    objective_value, cotangent_rms, finite = (
        torch.stack(
            (
                details["objective"].to(device=device, dtype=torch.float32),
                cotangent.square().mean().sqrt(),
                torch.isfinite(cotangent).all().to(dtype=torch.float32),
            )
        )
        .detach()
        .cpu()
        .tolist()
    )
    if finite != 1.0 or (mixed and cotangent_rms <= 0):
        raise ExpertManifoldError("mixed reward task produced zero Program credit")
    valid = batch.get("executed_action_steps")
    if not isinstance(valid, torch.Tensor):
        raise ExpertManifoldError("reward replay lost executed-prefix lengths")
    return cotangent, RewardProgramCreditSummary(
        objective=float(objective_value),
        successes=success_count,
        failures=4 - success_count,
        mixed=mixed,
        positive_episodes=int((advantages > 0).sum()),
        negative_episodes=int((advantages < 0).sum()),
        zero_episodes=int((advantages == 0).sum()),
        replay_chunks=int(episode_ids.numel()),
        executed_action_steps=int(valid.sum()),
        mc_samples=mc_samples,
        functional_policy_forwards=int(details["functional_policy_forwards"]),
        program_cotangent_rms=cotangent_rms,
    )


def reward_credit_is_finite(summary: RewardProgramCreditSummary) -> bool:
    return all(
        math.isfinite(value)
        for value in (summary.objective, summary.program_cotangent_rms)
    )
