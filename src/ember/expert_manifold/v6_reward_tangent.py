"""Fixed-landmark K4 reward tangents for the active SRTP Writer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_step import (
    GeneratedConditionGraph,
    redecoded_program_cotangent,
)
from ember.lora import LoRAContract, validate_lora_state
from ember.reward.protocol import RewardProtocolError, flow_sample_seed
from ember.reward.rollout import RewardRolloutOutcome
from ember.writer.condition_update import FrozenV6ConditionResidualWriter


@dataclass(frozen=True)
class RewardTangentSummary:
    """Scalar evidence for one fixed-landmark task-level reward tangent."""

    objective: float
    successes: int
    failures: int
    mixed: bool
    positive_episodes: int
    negative_episodes: int
    selected_landmarks: int
    maximum_landmarks_per_episode: int
    executed_action_steps: int
    mc_samples: int
    functional_policy_forwards: int
    program_cotangent_rms: float


def leave_one_out_binary_advantages(successes: torch.Tensor) -> torch.Tensor:
    """Return the sealed K4 binary leave-one-out advantages."""

    if (
        successes.ndim != 1
        or successes.numel() != 4
        or not bool(torch.isfinite(successes).all())
        or bool(((successes != 0) & (successes != 1)).any())
    ):
        raise RewardProtocolError("reward tangent requires four binary outcomes")
    return (4 * successes - successes.sum()) / 3


def landmark_credit_batch(
    outcomes: Sequence[RewardRolloutOutcome],
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate at most sixteen selected rows with episode-equal weights."""

    if len(outcomes) != 4:
        raise RewardProtocolError("reward tangent requires an exact K4 panel")
    successes = torch.tensor(
        [value.success for value in outcomes],
        dtype=torch.float32,
        device=device,
    )
    advantages = leave_one_out_binary_advantages(successes)
    if bool((advantages == 0).all()):
        return {}, successes, advantages, torch.empty(0, device=device)
    if any(not 0 < len(value.landmarks) <= 4 for value in outcomes):
        raise RewardProtocolError("reward tangent landmark budget changed")
    rows = [
        (episode, landmark)
        for episode, outcome in enumerate(outcomes)
        for landmark in outcome.landmarks
    ]
    if not 4 <= len(rows) <= 16:
        raise RewardProtocolError("reward tangent logical batch changed")
    keys = set(rows[0][1].observation)
    if any(set(value.observation) != keys for _, value in rows):
        raise RewardProtocolError("reward tangent observation keys changed")
    batch = {
        name: torch.cat([value.observation[name] for _, value in rows], dim=0).to(
            device=device, non_blocking=True
        )
        for name in sorted(keys)
    }
    actions = torch.cat(
        [value.normalized_action_chunk for _, value in rows], dim=0
    ).to(device=device, non_blocking=True)
    valid = torch.tensor(
        [value.executed_action_steps for _, value in rows],
        dtype=torch.long,
        device=device,
    )
    if actions.ndim != 3 or bool((valid <= 0).any()) or bool(
        (valid > actions.shape[1]).any()
    ):
        raise RewardProtocolError("reward tangent executed prefix changed")
    episode_ids = torch.tensor(
        [episode for episode, _ in rows], dtype=torch.long, device=device
    )
    counts = torch.bincount(episode_ids, minlength=4)
    if counts.shape != (4,) or bool((counts <= 0).any()):
        raise RewardProtocolError("reward tangent lost a K4 episode")
    weights = advantages.index_select(0, episode_ids) / (
        4.0 * counts.index_select(0, episode_ids).to(dtype=torch.float32)
    )
    batch[ACTION] = actions
    batch["executed_action_steps"] = valid
    return batch, successes, advantages, weights


def _flow_sample_panel(
    policy: torch.nn.Module,
    *,
    count: int,
    seed_root: int,
    schedule_macro: int,
    global_task_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_config = getattr(getattr(policy, "model", None), "config", None)
    if (
        not 0 < count <= 16
        or model_config is None
        or float(model_config.time_sampling_beta_alpha) != 1.5
        or float(model_config.time_sampling_beta_beta) != 1.0
        or float(model_config.time_sampling_scale) != 0.999
        or float(model_config.time_sampling_offset) != 0.001
    ):
        raise RewardProtocolError("reward tangent flow panel changed")
    shape = (
        count,
        int(policy.config.chunk_size),
        int(policy.config.max_action_dim),
    )
    noises = []
    times = []
    for mc_index in range(4):
        generator = torch.Generator(device=device).manual_seed(
            flow_sample_seed(
                seed_root,
                cycle=schedule_macro,
                global_task_id=global_task_id,
                mc_index=mc_index,
            )
        )
        noises.append(
            torch.randn(shape, dtype=torch.float32, device=device, generator=generator)
        )
        uniform = torch.rand(
            count, dtype=torch.float32, device=device, generator=generator
        )
        times.append(uniform.pow(2.0 / 3.0).mul_(0.999).add_(0.001))
    return torch.stack(noises), torch.stack(times)


def _executed_prefix_flow_loss(
    policy: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    noise: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    valid = batch["executed_action_steps"]
    images, image_masks = policy._preprocess_images(dict(batch))
    actions = policy.prepare_action(batch)
    if noise.shape != actions.shape or time.shape != (actions.shape[0],):
        raise RewardProtocolError("reward tangent noise or time shape changed")
    losses = policy.model.forward(
        images,
        image_masks,
        batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK],
        actions,
        noise,
        time,
    )
    action_dim = int(policy.config.output_features[ACTION].shape[0])
    losses = losses[:, :, :action_dim]
    mask = torch.arange(losses.shape[1], device=losses.device)[None] < valid[:, None]
    return (losses * mask[:, :, None]).sum(dim=(1, 2)) / (
        valid * action_dim
    ).to(dtype=losses.dtype)


def _functional_executed_prefix_flow_loss(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, torch.Tensor],
    *,
    noise: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    validate_lora_state(state, contract)
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise RewardProtocolError("reward tangent policy must remain frozen")

    class _Loss(torch.nn.Module):
        def __init__(self, owner: torch.nn.Module) -> None:
            super().__init__()
            self.policy = owner

        def forward(self, value: Mapping[str, torch.Tensor]) -> torch.Tensor:
            return _executed_prefix_flow_loss(
                self.policy, value, noise=noise, time=time
            )

    prefixed = {f"policy.{name}": value for name, value in state.items()}
    return torch.func.functional_call(
        _Loss(policy), prefixed, args=(batch,), strict=False
    )


def landmark_reward_program_cotangent(
    graph: GeneratedConditionGraph,
    *,
    writer: FrozenV6ConditionResidualWriter,
    policy: torch.nn.Module,
    contract: LoRAContract,
    outcomes: Sequence[RewardRolloutOutcome],
    flow_seed_root: int,
    schedule_macro: int,
    global_task_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, RewardTangentSummary]:
    """Differentiate one mixed K4 signed landmark objective to Program space."""

    batch, successes, advantages, weights = landmark_credit_batch(
        outcomes, device=device
    )
    success_count = int(successes.sum())
    mixed = 0 < success_count < 4
    if mixed:
        action = batch.get(ACTION)
        if not isinstance(action, torch.Tensor) or not 0 < action.shape[0] <= 16:
            raise RewardProtocolError("reward tangent logical action batch changed")
        noises, times = _flow_sample_panel(
            policy,
            count=int(action.shape[0]),
            seed_root=flow_seed_root,
            schedule_macro=schedule_macro,
            global_task_id=global_task_id,
            device=device,
        )
        names = tuple(graph.correct_lora)
        leaves = {
            name: value.detach().requires_grad_(True)
            for name, value in graph.correct_lora.items()
        }
        lora_gradients = {
            name: torch.zeros_like(value, dtype=torch.float32)
            for name, value in leaves.items()
        }
        objective = torch.zeros((), dtype=torch.float32, device=device)
        for mc_index in range(4):
            per_row = _functional_executed_prefix_flow_loss(
                policy,
                leaves,
                contract,
                batch,
                noise=noises[mc_index],
                time=times[mc_index],
            )
            scalar = (per_row.to(dtype=torch.float32) * weights).sum() / 4
            gradients = torch.autograd.grad(
                scalar, tuple(leaves[name] for name in names)
            )
            objective.add_(scalar.detach())
            for name, gradient in zip(names, gradients, strict=True):
                lora_gradients[name].add_(gradient.to(dtype=torch.float32))
        del batch, gradients, leaves, noises, per_row, scalar, times
        cotangent = redecoded_program_cotangent(
            writer=writer,
            program_value=graph.program_leaf,
            lora_gradients=lora_gradients,
            device=device,
        )
        forwards = 4
    else:
        objective = torch.zeros((), dtype=torch.float32, device=device)
        cotangent = torch.zeros_like(graph.program_leaf[0], dtype=torch.float32)
        forwards = 0
    if cotangent.shape != (320, 256) or not bool(torch.isfinite(cotangent).all()):
        raise ExpertManifoldError("reward Program tangent became invalid")
    cotangent_rms = float(cotangent.square().mean().sqrt())
    if mixed and cotangent_rms <= 0:
        raise ExpertManifoldError("mixed reward task produced zero Program tangent")
    landmark_counts = [len(value.landmarks) for value in outcomes]
    executed_steps = sum(
        landmark.executed_action_steps
        for value in outcomes
        for landmark in value.landmarks
    )
    summary = RewardTangentSummary(
        objective=float(objective),
        successes=success_count,
        failures=4 - success_count,
        mixed=mixed,
        positive_episodes=int((advantages > 0).sum()),
        negative_episodes=int((advantages < 0).sum()),
        selected_landmarks=sum(landmark_counts),
        maximum_landmarks_per_episode=max(landmark_counts),
        executed_action_steps=executed_steps,
        mc_samples=4,
        functional_policy_forwards=forwards,
        program_cotangent_rms=cotangent_rms,
    )
    if not all(math.isfinite(value) for value in (summary.objective, cotangent_rms)):
        raise ExpertManifoldError("reward tangent summary became non-finite")
    return cotangent, summary
