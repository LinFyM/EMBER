"""Policy-effect realization and capacity projections for ECP Stage 1B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from ember.ecp.low_rank import canonicalize_low_rank_factors
from ember.ecp.policy_effects import PolicyEffectResponse
from ember.ecp.stage1_equivalence import Stage1EffectBank
from ember.ecp.stage1_objective import (
    ParticleObjective,
    RealizationConfig,
    RealizationSnapshot,
    build_particle_objective,
    candidate_snapshot,
    member_distances,
    reference_distances,
    response_fields,
)
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract, validate_lora_state


@dataclass(frozen=True)
class RealizationStep:
    step: int
    snapshot: RealizationSnapshot
    gradient_rms: float
    a_gradient_rms: float
    b_gradient_rms: float
    applied_step_rms: float


@dataclass(frozen=True)
class RankReservedProjectionTarget:
    """Best additive low-rank correction evidence for one LoRA target."""

    target: str
    expert_energy: float
    carrier_energy: float
    required_correction_energy: float
    projected_correction_energy: float
    projected_effective_update_energy: float
    residual_energy: float
    carrier_rank: int
    residual_rank: int


ResponseFunction = Callable[
    [Mapping[str, torch.Tensor], torch.Tensor], PolicyEffectResponse
]


def rank_reserved_state(
    carrier: Mapping[str, torch.Tensor],
    residual: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    carrier_rank: int,
) -> dict[str, torch.Tensor]:
    residual_rank = int(contract.rank) - int(carrier_rank)
    if carrier_rank <= 0 or residual_rank <= 0:
        raise ValueError("rank-reserved state requires carrier and residual ranks")
    result = {}
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        carrier_a = carrier[a_name]
        carrier_b = carrier[b_name]
        residual_a = residual[a_name]
        residual_b = residual[b_name]
        if (
            residual_a.shape != (residual_rank, target.in_features)
            or residual_b.shape != (target.out_features, residual_rank)
        ):
            raise ValueError("rank-reserved residual shapes changed")
        result[a_name] = torch.cat(
            [carrier_a[:carrier_rank], residual_a.to(carrier_a)], dim=0
        )
        result[b_name] = torch.cat(
            [carrier_b[:, :carrier_rank], residual_b.to(carrier_b)], dim=1
        )
    return result


def rank_reserved_relative_distance(
    residual: Mapping[str, torch.Tensor],
    carrier: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    carrier_rank: int,
) -> torch.Tensor:
    distances = []
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        carrier_a = carrier[a_name][:carrier_rank].float()
        carrier_b = carrier[b_name][:, :carrier_rank].float()
        residual_a = residual[a_name].float()
        residual_b = residual[b_name].float()
        residual_energy = _effective_inner_product(
            residual_b, residual_a, residual_b, residual_a
        )
        carrier_energy = _effective_inner_product(
            carrier_b, carrier_a, carrier_b, carrier_a
        )
        distances.append(residual_energy / carrier_energy.clamp_min(1e-10))
    return torch.stack(distances).mean()


def _initial_rank_reserved_residual(
    carrier: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    carrier_rank: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    residual = {}
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        carrier_a = carrier[a_name]
        carrier_b = carrier[b_name]
        if torch.count_nonzero(carrier_b[:, carrier_rank:]):
            raise ValueError("stable carrier uses ranks reserved for the residual")
        residual[a_name] = carrier_a[carrier_rank:].detach().float().to(device).clone()
        residual[b_name] = torch.zeros_like(
            carrier_b[:, carrier_rank:], dtype=torch.float32, device=device
        )
    return residual


def _balanced_rank_reserved_residual(
    residual: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    carrier_rank: int,
) -> dict[str, torch.Tensor]:
    residual_rank = int(contract.rank) - int(carrier_rank)
    balanced = {}
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        a, b = canonicalize_low_rank_factors(
            residual[a_name], residual[b_name], output_rank=residual_rank
        )
        balanced[a_name] = a.detach()
        balanced[b_name] = b.detach()
    return balanced


def _effective_inner_product(
    left_b: torch.Tensor,
    left_a: torch.Tensor,
    right_b: torch.Tensor,
    right_a: torch.Tensor,
) -> torch.Tensor:
    return torch.sum((left_b.T @ right_b) * (left_a @ right_a.T))


def project_expert_onto_rank_reserved_residual(
    *,
    carrier: Mapping[str, torch.Tensor],
    expert: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    carrier_rank: int,
) -> tuple[dict[str, torch.Tensor], tuple[RankReservedProjectionTarget, ...]]:
    """Add the best mobile residual in the ranks reserved by the carrier.

    For every target this computes the truncated-SVD solution to
    ``min_rank(X)<=r ||(W_expert - W_carrier) - X||_F`` without materializing
    either dense update.  Concatenating the frozen carrier ranks and residual
    factors yields one complete LoRA with an exact effective-update sum.
    """

    validate_lora_state(carrier, contract)
    validate_lora_state(expert, contract)
    residual_rank = int(contract.rank) - int(carrier_rank)
    if carrier_rank <= 0 or residual_rank <= 0:
        raise ValueError(
            "rank-reserved projection requires carrier and residual ranks"
        )
    projected: dict[str, torch.Tensor] = {}
    metrics = []
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        carrier_a = carrier[a_name]
        carrier_b = carrier[b_name]
        expert_a = expert[a_name]
        expert_b = expert[b_name]
        if torch.count_nonzero(carrier_b[:, carrier_rank:]):
            raise ValueError("stable carrier uses ranks reserved for the residual")

        correction_a = torch.cat(
            [expert_a.detach().float(), carrier_a.detach().float()], dim=0
        )
        correction_b = torch.cat(
            [expert_b.detach().float(), -carrier_b.detach().float()], dim=1
        )
        residual_a, residual_b = canonicalize_low_rank_factors(
            correction_a, correction_b, output_rank=residual_rank
        )
        residual_a = residual_a.to(device=carrier_a.device, dtype=carrier_a.dtype)
        residual_b = residual_b.to(device=carrier_b.device, dtype=carrier_b.dtype)
        projected[a_name] = torch.cat(
            [carrier_a[:carrier_rank].detach(), residual_a], dim=0
        )
        projected[b_name] = torch.cat(
            [carrier_b[:, :carrier_rank].detach(), residual_b], dim=1
        )

        ca = carrier_a.detach().double()
        cb = carrier_b.detach().double()
        ea = expert_a.detach().double()
        eb = expert_b.detach().double()
        ra = residual_a.detach().double()
        rb = residual_b.detach().double()
        expert_energy = _effective_inner_product(eb, ea, eb, ea)
        carrier_energy = _effective_inner_product(cb, ca, cb, ca)
        expert_carrier = _effective_inner_product(eb, ea, cb, ca)
        expert_residual = _effective_inner_product(eb, ea, rb, ra)
        carrier_residual = _effective_inner_product(cb, ca, rb, ra)
        residual_energy = _effective_inner_product(rb, ra, rb, ra)
        correction_energy = (
            expert_energy + carrier_energy - 2.0 * expert_carrier
        ).clamp_min(0.0)
        projected_energy = (
            carrier_energy + residual_energy + 2.0 * carrier_residual
        ).clamp_min(0.0)
        approximation_error = (
            expert_energy
            + projected_energy
            - 2.0 * (expert_carrier + expert_residual)
        ).clamp_min(0.0)
        metrics.append(
            RankReservedProjectionTarget(
                target=target.name,
                expert_energy=float(expert_energy),
                carrier_energy=float(carrier_energy),
                required_correction_energy=float(correction_energy),
                projected_correction_energy=float(residual_energy),
                projected_effective_update_energy=float(projected_energy),
                residual_energy=float(approximation_error),
                carrier_rank=int(carrier_rank),
                residual_rank=residual_rank,
            )
        )
    validate_lora_state(projected, contract)
    return projected, tuple(metrics)


def _indexed_response(
    value: PolicyEffectResponse, indices: torch.Tensor
) -> PolicyEffectResponse:
    return PolicyEffectResponse(
        owner=value.owner.index_select(0, indices),
        flow=value.flow.index_select(0, indices),
        action=value.action.index_select(0, indices),
    )


def _capture_all(
    response: ResponseFunction,
    state: Mapping[str, torch.Tensor],
    states: int,
    microbatch: int,
    device: torch.device,
) -> PolicyEffectResponse:
    values = []
    for start in range(0, states, microbatch):
        indices = torch.arange(start, min(start + microbatch, states), device=device)
        values.append(response(state, indices))
    return PolicyEffectResponse(
        *(
            torch.cat([getattr(value, name) for value in values])
            for name in ("owner", "flow", "action")
        )
    )


def _gradient_rows(
    *,
    state: Mapping[str, torch.Tensor],
    leaves: Mapping[str, torch.Tensor],
    names: tuple[str, ...],
    response: ResponseFunction,
    bank: Stage1EffectBank,
    objective: ParticleObjective,
    responsibilities: torch.Tensor,
    barrier_active: torch.Tensor,
    config: RealizationConfig,
) -> dict[str, torch.Tensor]:
    gradients = {name: torch.zeros_like(value) for name, value in leaves.items()}
    device = bank.suffix_noise.device
    for start in range(0, bank.state_count, config.microbatch_size):
        indices = torch.arange(
            start,
            min(start + config.microbatch_size, bank.state_count),
            device=device,
        )
        candidate = response(state, indices)
        members = PolicyEffectResponse(
            owner=bank.members.owner.index_select(1, indices),
            flow=bank.members.flow.index_select(1, indices),
            action=bank.members.action.index_select(1, indices),
        )
        scales = PolicyEffectResponse(
            owner=objective.scales.owner.index_select(1, indices),
            flow=objective.scales.flow.index_select(1, indices),
            action=objective.scales.action.index_select(1, indices),
        )
        distances = member_distances(candidate, members, scales, config)
        groups = objective.group_ids.index_select(0, indices)
        weighted = (responsibilities.index_select(0, groups).T * distances).sum(0)
        multiplier = (
            0.25
            + 0.75 * objective.confidence.index_select(0, groups)
            + float(config.carrier_barrier_weight)
            * barrier_active.index_select(0, groups)
        )
        loss = (
            objective.state_weights.index_select(0, indices) * multiplier * weighted
        ).sum()
        carrier = _indexed_response(bank.carrier, indices)
        source = _indexed_response(bank.source, indices)
        use_carrier = objective.baseline_uses_carrier.index_select(0, groups)
        reference = PolicyEffectResponse(
            *(
                torch.where(
                    use_carrier[(...,) + (None,) * (left.ndim - 1)], left, right
                )
                for left, right in zip(
                    response_fields(carrier), response_fields(source), strict=True
                )
            )
        )
        preservation = reference_distances(candidate, reference, scales, config)
        loss = (
            loss
            + float(config.preservation_weight)
            * (
                objective.state_weights.index_select(0, indices)
                * (1.0 - objective.confidence.index_select(0, groups))
                * preservation
            ).sum()
        )
        row_gradients = torch.autograd.grad(loss, tuple(leaves[name] for name in names))
        for name, gradient in zip(names, row_gradients, strict=True):
            gradients[name].add_(gradient.detach())
    return gradients


def _gradient_rms(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    energy = sum(value.square().sum() for value in values)
    count = sum(value.numel() for value in values)
    return torch.sqrt(energy / count)


def solve_rank_reserved_particle_effects(
    *,
    carrier: Mapping[str, torch.Tensor],
    bank: Stage1EffectBank,
    contract: LoRAContract,
    response: ResponseFunction,
    config: RealizationConfig,
    carrier_rank: int,
) -> tuple[dict[str, torch.Tensor], tuple[RealizationStep, ...], RealizationSnapshot]:
    """Optimize one mobile residual while keeping the effective carrier frozen."""

    validate_lora_state(carrier, contract)
    if config.steps != 12 or config.microbatch_size <= 0:
        raise ValueError("ECP Stage 1 rank-reserved solver contract changed")
    residual_rank = int(contract.rank) - int(carrier_rank)
    if carrier_rank <= 0 or residual_rank <= 0:
        raise ValueError("rank-reserved solver requires carrier and residual ranks")
    device = bank.suffix_noise.device
    objective = build_particle_objective(bank, config)
    residual = _initial_rank_reserved_residual(
        carrier, contract, carrier_rank, device
    )
    history = []
    for step in range(config.steps):
        leaves = {
            name: value.detach().requires_grad_(True)
            for name, value in residual.items()
        }
        detached = {name: value.detach() for name, value in residual.items()}
        with torch.no_grad():
            candidate = _capture_all(
                response,
                rank_reserved_state(carrier, detached, contract, carrier_rank),
                bank.state_count,
                config.microbatch_size,
                device,
            )
            trust = rank_reserved_relative_distance(
                detached, carrier, contract, carrier_rank
            )
            snapshot, responsibilities, barrier_active = candidate_snapshot(
                candidate, bank, objective, config, trust
            )
        names = tuple(leaves)
        gradients = _gradient_rows(
            state=rank_reserved_state(carrier, leaves, contract, carrier_rank),
            leaves=leaves,
            names=names,
            response=response,
            bank=bank,
            objective=objective,
            responsibilities=responsibilities,
            barrier_active=barrier_active,
            config=config,
        )
        trust = rank_reserved_relative_distance(
            leaves, carrier, contract, carrier_rank
        )
        trust_penalty = torch.relu(trust - float(config.trust_region)).square()
        if config.trust_weight and float(trust.detach()) > config.trust_region:
            trust_gradients = torch.autograd.grad(
                float(config.trust_weight) * trust_penalty,
                tuple(leaves[name] for name in names),
            )
            for name, gradient in zip(names, trust_gradients, strict=True):
                gradients[name].add_(gradient.detach())
        a_gradients = tuple(
            gradients[target.name + LORA_A_SUFFIX] for target in contract.targets
        )
        b_gradients = tuple(
            gradients[target.name + LORA_B_SUFFIX] for target in contract.targets
        )
        gradient_rms = _gradient_rms(tuple(gradients.values()))
        a_gradient_rms = _gradient_rms(a_gradients)
        b_gradient_rms = _gradient_rms(b_gradients)
        applied = float(config.step_rms) / float(step + 1) ** float(
            config.step_decay_power
        )
        updated = {}
        for target in contract.targets:
            a_name = target.name + LORA_A_SUFFIX
            b_name = target.name + LORA_B_SUFFIX
            a_gradient = gradients[a_name]
            b_gradient = gradients[b_name]
            joint_rms = torch.sqrt(
                (a_gradient.square().sum() + b_gradient.square().sum())
                / (a_gradient.numel() + b_gradient.numel())
            ).clamp_min(1e-12)
            updated[a_name] = leaves[a_name].detach() - applied * a_gradient / joint_rms
            updated[b_name] = leaves[b_name].detach() - applied * b_gradient / joint_rms
        residual = _balanced_rank_reserved_residual(
            updated, contract, carrier_rank
        )
        history.append(
            RealizationStep(
                step=step,
                snapshot=snapshot,
                gradient_rms=float(gradient_rms),
                a_gradient_rms=float(a_gradient_rms),
                b_gradient_rms=float(b_gradient_rms),
                applied_step_rms=applied,
            )
        )
    final_state = rank_reserved_state(carrier, residual, contract, carrier_rank)
    validate_lora_state(final_state, contract)
    with torch.no_grad():
        final_response = _capture_all(
            response, final_state, bank.state_count, config.microbatch_size, device
        )
        final_snapshot, _, _ = candidate_snapshot(
            final_response,
            bank,
            objective,
            config,
            rank_reserved_relative_distance(
                residual, carrier, contract, carrier_rank
            ),
        )
    return (
        {name: value.detach() for name, value in final_state.items()},
        tuple(history),
        final_snapshot,
    )
