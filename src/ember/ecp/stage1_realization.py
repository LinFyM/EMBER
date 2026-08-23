"""Fixed-A policy-effect realization solver for ECP Stage 1B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch

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
    applied_step_rms: float


ResponseFunction = Callable[
    [Mapping[str, torch.Tensor], torch.Tensor], PolicyEffectResponse
]


def fixed_a_state(
    carrier: Mapping[str, torch.Tensor],
    delta_b: Mapping[str, torch.Tensor],
    contract: LoRAContract,
) -> dict[str, torch.Tensor]:
    result = {}
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        result[a_name] = carrier[a_name]
        result[b_name] = carrier[b_name] + delta_b[b_name]
    return result


def fixed_a_relative_distance(
    delta_b: Mapping[str, torch.Tensor],
    carrier: Mapping[str, torch.Tensor],
    contract: LoRAContract,
) -> torch.Tensor:
    distances = []
    for target in contract.targets:
        a = carrier[target.name + LORA_A_SUFFIX].float()
        b = carrier[target.name + LORA_B_SUFFIX].float()
        delta = delta_b[target.name + LORA_B_SUFFIX].float()
        gram_a = a @ a.T
        delta_energy = torch.einsum("ij,ji->", delta.T @ delta, gram_a)
        carrier_energy = torch.einsum("ij,ji->", b.T @ b, gram_a)
        distances.append(delta_energy / carrier_energy.clamp_min(1e-10))
    return torch.stack(distances).mean()


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


def solve_fixed_a_particle_effects(
    *,
    carrier: Mapping[str, torch.Tensor],
    bank: Stage1EffectBank,
    contract: LoRAContract,
    response: ResponseFunction,
    config: RealizationConfig,
) -> tuple[dict[str, torch.Tensor], tuple[RealizationStep, ...], RealizationSnapshot]:
    """Use detached softmin responsibilities, then exact first-order microbatches."""

    validate_lora_state(carrier, contract)
    if config.steps != 12 or config.microbatch_size <= 0:
        raise ValueError("ECP Stage 1 fixed solver contract changed")
    device = bank.suffix_noise.device
    objective = build_particle_objective(bank, config)
    delta = {
        target.name
        + LORA_B_SUFFIX: torch.zeros_like(
            carrier[target.name + LORA_B_SUFFIX], dtype=torch.float32, device=device
        )
        for target in contract.targets
    }
    history = []
    for step in range(config.steps):
        leaves = {
            name: value.detach().requires_grad_(True) for name, value in delta.items()
        }
        detached = {name: value.detach() for name, value in delta.items()}
        with torch.no_grad():
            candidate = _capture_all(
                response,
                fixed_a_state(carrier, detached, contract),
                bank.state_count,
                config.microbatch_size,
                device,
            )
            trust = fixed_a_relative_distance(detached, carrier, contract)
            snapshot, responsibilities, barrier_active = candidate_snapshot(
                candidate, bank, objective, config, trust
            )
        names = tuple(leaves)
        gradients = _gradient_rows(
            state=fixed_a_state(carrier, leaves, contract),
            leaves=leaves,
            names=names,
            response=response,
            bank=bank,
            objective=objective,
            responsibilities=responsibilities,
            barrier_active=barrier_active,
            config=config,
        )
        trust = fixed_a_relative_distance(leaves, carrier, contract)
        trust_penalty = torch.relu(trust - float(config.trust_region)).square()
        if config.trust_weight and float(trust.detach()) > config.trust_region:
            trust_gradients = torch.autograd.grad(
                float(config.trust_weight) * trust_penalty,
                tuple(leaves[name] for name in names),
            )
            for name, gradient in zip(names, trust_gradients, strict=True):
                gradients[name].add_(gradient.detach())
        energy = sum(value.square().sum() for value in gradients.values())
        count = sum(value.numel() for value in gradients.values())
        gradient_rms = torch.sqrt(energy / count)
        applied = float(config.step_rms) / float(step + 1) ** float(
            config.step_decay_power
        )
        delta = {
            target.name
            + LORA_B_SUFFIX: leaves[target.name + LORA_B_SUFFIX].detach()
            - applied
            * gradients[target.name + LORA_B_SUFFIX]
            / gradients[target.name + LORA_B_SUFFIX]
            .square()
            .mean()
            .sqrt()
            .clamp_min(1e-12)
            for target in contract.targets
        }
        history.append(
            RealizationStep(
                step=step,
                snapshot=snapshot,
                gradient_rms=float(gradient_rms),
                applied_step_rms=applied,
            )
        )
    final_state = fixed_a_state(carrier, delta, contract)
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
            fixed_a_relative_distance(delta, carrier, contract),
        )
    return (
        {name: value.detach() for name, value in final_state.items()},
        tuple(history),
        final_snapshot,
    )
