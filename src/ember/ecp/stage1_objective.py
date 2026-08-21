"""Gauge-invariant Stage 1 losses for privileged Program compilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from ember.ecp.compiler import ECPCompilerOutput
from ember.ecp.stage1_support import PolicySupportLoss
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract


@dataclass(frozen=True)
class ECPStage1Loss:
    total: torch.Tensor
    member_effective_update: torch.Tensor
    consensus_effective_update: torch.Tensor
    member_canonical_factor: torch.Tensor
    consensus_canonical_factor: torch.Tensor
    prior_preservation: torch.Tensor
    successful_response: torch.Tensor
    learner_response: torch.Tensor
    source_support: torch.Tensor
    shared_support: torch.Tensor
    expert_set_disagreement: torch.Tensor
    locality: torch.Tensor

    @property
    def functional_response(self) -> torch.Tensor:
        return self.successful_response + self.learner_response


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


def effective_update_cosine_matrix(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    contract: LoRAContract,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact cross-state BA cosine matrix without materializing dense updates."""

    cross: torch.Tensor | None = None
    left_energy: torch.Tensor | None = None
    right_energy: torch.Tensor | None = None
    for owner in contract.targets:
        name_a = owner.name + LORA_A_SUFFIX
        name_b = owner.name + LORA_B_SUFFIX
        left_a = _batched(left[name_a]).float()
        left_b = _batched(left[name_b]).float()
        right_a = _batched(right[name_a]).float()
        right_b = _batched(right[name_b]).float()
        b_cross = torch.einsum("nor,mos->nmrs", left_b, right_b)
        a_cross = torch.einsum("nri,msi->nmrs", left_a, right_a)
        owner_cross = (b_cross * a_cross).sum(dim=(2, 3))
        owner_left = torch.einsum(
            "nii->n",
            (left_b.transpose(1, 2) @ left_b)
            @ (left_a @ left_a.transpose(1, 2)),
        )
        owner_right = torch.einsum(
            "nii->n",
            (right_b.transpose(1, 2) @ right_b)
            @ (right_a @ right_a.transpose(1, 2)),
        )
        cross = owner_cross if cross is None else cross + owner_cross
        left_energy = owner_left if left_energy is None else left_energy + owner_left
        right_energy = (
            owner_right if right_energy is None else right_energy + owner_right
        )
    assert cross is not None and left_energy is not None and right_energy is not None
    cosine = cross / (
        left_energy.clamp_min(1e-20).sqrt()[:, None]
        * right_energy.clamp_min(1e-20).sqrt()[None]
    )
    return cosine, left_energy, right_energy


def canonical_factor_loss(
    candidate: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    contract: LoRAContract,
) -> torch.Tensor:
    """Relative A/B coordinate loss after targets enter the compact-SVD gauge."""

    losses = []
    for owner in contract.targets:
        for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX):
            name = owner.name + suffix
            candidate_value = _batched(candidate[name]).float()
            target_value = _batched(target[name]).float()
            batches = max(candidate_value.shape[0], target_value.shape[0])
            candidate_value = candidate_value.expand(
                batches, *candidate_value.shape[1:]
            )
            target_value = target_value.expand(batches, *target_value.shape[1:])
            error = (candidate_value - target_value).square().flatten(1).sum(1)
            energy = target_value.square().flatten(1).sum(1)
            losses.append(error / energy.clamp_min(1e-10))
    return torch.stack(losses, dim=1).mean()


def ecp_stage1_loss(
    *,
    member: ECPCompilerOutput,
    consensus: ECPCompilerOutput,
    prior: ECPCompilerOutput,
    expert_states: Mapping[str, torch.Tensor],
    prior_target: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    policy_support: PolicySupportLoss,
    weights: Mapping[str, float],
) -> ECPStage1Loss:
    member_effective = exact_effective_update_loss(
        member.state, expert_states, contract
    )
    consensus_effective = exact_effective_update_loss(
        consensus.state, expert_states, contract
    )
    member_canonical = canonical_factor_loss(
        member.state, expert_states, contract
    )
    consensus_canonical = canonical_factor_loss(
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
        "member_canonical_factor": member_canonical,
        "consensus_canonical_factor": consensus_canonical,
        "prior_preservation": prior_preservation,
        "successful_response": policy_support.successful_response,
        "learner_response": policy_support.learner_response,
        "source_support": policy_support.source_support,
        "shared_support": policy_support.shared_support,
        "locality": locality,
    }
    missing = set(terms) - set(weights)
    if missing:
        raise ValueError(f"missing ECP Stage 1 loss weights: {sorted(missing)}")
    total = sum(float(weights[name]) * value for name, value in terms.items())
    return ECPStage1Loss(
        total=total,
        expert_set_disagreement=policy_support.expert_set_disagreement,
        **terms,
    )
