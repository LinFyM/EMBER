"""Gauge-invariant Stage 1 losses for privileged Program compilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from ember.ecp.compiler import ECPCompilerOutput
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract


@dataclass(frozen=True)
class ECPStage1Loss:
    total: torch.Tensor
    member_effective_update: torch.Tensor
    consensus_effective_update: torch.Tensor
    prior_preservation: torch.Tensor
    functional_response: torch.Tensor
    locality: torch.Tensor


def _batched(value: torch.Tensor) -> torch.Tensor:
    return value[None] if value.ndim == 2 else value


def exact_effective_update_loss(
    candidate: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    contract: LoRAContract,
) -> torch.Tensor:
    """Exact relative Frobenius BA loss using only rank-sized Gram matrices."""

    losses = []
    for owner in contract.targets:
        name_a = owner.name + LORA_A_SUFFIX
        name_b = owner.name + LORA_B_SUFFIX
        candidate_a = _batched(candidate[name_a]).float()
        candidate_b = _batched(candidate[name_b]).float()
        target_a = _batched(target[name_a]).float()
        target_b = _batched(target[name_b]).float()
        batches = max(candidate_a.shape[0], target_a.shape[0])
        candidate_a = candidate_a.expand(batches, -1, -1)
        candidate_b = candidate_b.expand(batches, -1, -1)
        target_a = target_a.expand(batches, -1, -1)
        target_b = target_b.expand(batches, -1, -1)

        candidate_left = candidate_b.transpose(1, 2) @ candidate_b
        candidate_right = candidate_a @ candidate_a.transpose(1, 2)
        target_left = target_b.transpose(1, 2) @ target_b
        target_right = target_a @ target_a.transpose(1, 2)
        candidate_energy = torch.einsum(
            "bij,bji->b", candidate_left, candidate_right
        )
        target_energy = torch.einsum("bij,bji->b", target_left, target_right)
        cross_left = candidate_b.transpose(1, 2) @ target_b
        cross_right = target_a @ candidate_a.transpose(1, 2)
        cross = torch.einsum("bij,bji->b", cross_left, cross_right)
        error = (candidate_energy + target_energy - 2.0 * cross).clamp_min(0)
        losses.append(error / target_energy.clamp_min(1e-10))
    return torch.stack(losses, dim=1).mean()


def ecp_stage1_loss(
    *,
    member: ECPCompilerOutput,
    consensus: ECPCompilerOutput,
    prior: ECPCompilerOutput,
    expert_states: Mapping[str, torch.Tensor],
    prior_target: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    functional_response: torch.Tensor,
    weights: Mapping[str, float],
) -> ECPStage1Loss:
    member_effective = exact_effective_update_loss(
        member.state, expert_states, contract
    )
    consensus_effective = exact_effective_update_loss(
        consensus.state, expert_states, contract
    )
    prior_preservation = exact_effective_update_loss(
        prior.state, prior_target, contract
    )
    locality = (
        member.locality_penalty
        + consensus.locality_penalty
        + prior.locality_penalty
    ) / 3.0
    terms = {
        "member_effective_update": member_effective,
        "consensus_effective_update": consensus_effective,
        "prior_preservation": prior_preservation,
        "functional_response": functional_response,
        "locality": locality,
    }
    missing = set(terms) - set(weights)
    if missing:
        raise ValueError(f"missing ECP Stage 1 loss weights: {sorted(missing)}")
    total = sum(float(weights[name]) * value for name, value in terms.items())
    return ECPStage1Loss(total=total, **terms)
