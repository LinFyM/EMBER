"""Gauge-invariant policy-effect realization for ECP Stage 1B."""

from __future__ import annotations

import math
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
from ember.ecp.stage1_parameterization import (
    effective_inner_product,
    initial_rank_reserved_residual,
    rank_reserved_relative_distance,
    rank_reserved_state,
)
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract, validate_lora_state


@dataclass(frozen=True)
class RealizationStep:
    iteration: int
    phase: str
    before: RealizationSnapshot
    after: RealizationSnapshot
    accepted_alpha: float
    backtrack_index: int
    directional_derivative: float
    gradient_rms: float
    cumulative_vjp_evaluations: int


@dataclass(frozen=True)
class RealizationOutcome:
    state: dict[str, torch.Tensor]
    initial: RealizationSnapshot
    final: RealizationSnapshot
    history: tuple[RealizationStep, ...]
    initial_state_is_exact_carrier: bool
    initial_directional_derivative: float
    initial_gradient_rms: float
    vjp_evaluations: int
    stop_reason: str
    best_member_effect_objective: float
    objective_gap_recovery: float


ResponseFunction = Callable[
    [Mapping[str, torch.Tensor], torch.Tensor], PolicyEffectResponse
]


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


def _evaluate_residual(
    *,
    carrier: Mapping[str, torch.Tensor],
    residual: Mapping[str, torch.Tensor],
    bank: Stage1EffectBank,
    contract: LoRAContract,
    response: ResponseFunction,
    objective: ParticleObjective,
    config: RealizationConfig,
    carrier_rank: int,
) -> tuple[RealizationSnapshot, torch.Tensor, torch.Tensor]:
    device = bank.suffix_noise.device
    with torch.no_grad():
        candidate = _capture_all(
            response,
            rank_reserved_state(carrier, residual, contract, carrier_rank),
            bank.state_count,
            config.microbatch_size,
            device,
        )
        return candidate_snapshot(
            candidate,
            bank,
            objective,
            config,
            rank_reserved_relative_distance(
                residual, carrier, contract, carrier_rank
            ),
        )


def _best_member_effect_objective(
    bank: Stage1EffectBank,
    objective: ParticleObjective,
    config: RealizationConfig,
) -> float:
    values = []
    for member in range(bank.member_count):
        response = PolicyEffectResponse(
            owner=bank.members.owner[member],
            flow=bank.members.flow[member],
            action=bank.members.action[member],
        )
        snapshot, _, _ = candidate_snapshot(
            response,
            bank,
            objective,
            config,
            torch.zeros((), device=bank.suffix_noise.device),
        )
        values.append(snapshot.total)
    return min(values)


def _carrier_energy(
    carrier: Mapping[str, torch.Tensor],
    a_name: str,
    b_name: str,
    carrier_rank: int,
) -> torch.Tensor:
    a = carrier[a_name][:carrier_rank].float()
    b = carrier[b_name][:, :carrier_rank].float()
    return effective_inner_product(b, a, b, a).clamp_min(1e-12)


def _unit_effective_direction(
    *,
    direction: Mapping[str, torch.Tensor],
    carrier: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    carrier_rank: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    normalized = {}
    scales = {}
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        a = direction[a_name].float()
        b = direction[b_name].float()
        energy = effective_inner_product(b, a, b, a)
        if not torch.isfinite(energy) or float(energy) <= 0.0:
            raise ValueError(f"effective direction is degenerate for {target.name}")
        scale = torch.sqrt(
            _carrier_energy(carrier, a_name, b_name, carrier_rank) / energy
        )
        root = torch.sqrt(scale)
        normalized[a_name] = (root * a).detach()
        normalized[b_name] = (root * b).detach()
        scales[target.name] = float(scale)
    return normalized, scales


def _orthonormal_input_probe(
    *, rows: int, columns: int, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    value = torch.randn(columns, rows, generator=generator, dtype=torch.float32)
    basis, _ = torch.linalg.qr(value, mode="reduced")
    return basis.T.contiguous().to(device)


def _matrix_free_initial_direction(
    *,
    carrier: Mapping[str, torch.Tensor],
    bank: Stage1EffectBank,
    contract: LoRAContract,
    response: ResponseFunction,
    objective: ParticleObjective,
    responsibilities: torch.Tensor,
    barrier_active: torch.Tensor,
    config: RealizationConfig,
    carrier_rank: int,
) -> tuple[dict[str, torch.Tensor], float, float, int]:
    residual_rank = int(contract.rank) - int(carrier_rank)
    width = int(config.sketch_width)
    if width != 2 * residual_rank:
        raise ValueError("effective-update sketch must use two residual-rank chunks")
    device = bank.suffix_noise.device
    probes = {
        target.name: _orthonormal_input_probe(
            rows=width,
            columns=target.in_features,
            seed=int(config.probe_seed) + index,
            device=device,
        )
        for index, target in enumerate(contract.targets)
    }
    range_chunks: dict[str, list[torch.Tensor]] = {
        target.name: [] for target in contract.targets
    }
    vjp_evaluations = 0
    for start in range(0, width, residual_rank):
        residual = {}
        leaves = {}
        names = []
        for target in contract.targets:
            a_name = target.name + LORA_A_SUFFIX
            b_name = target.name + LORA_B_SUFFIX
            residual[a_name] = probes[target.name][start : start + residual_rank]
            leaves[b_name] = torch.zeros(
                target.out_features,
                residual_rank,
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            residual[b_name] = leaves[b_name]
            names.append(b_name)
        gradients = _gradient_rows(
            state=rank_reserved_state(carrier, residual, contract, carrier_rank),
            leaves=leaves,
            names=tuple(names),
            response=response,
            bank=bank,
            objective=objective,
            responsibilities=responsibilities,
            barrier_active=barrier_active,
            config=config,
        )
        for target in contract.targets:
            range_chunks[target.name].append(
                gradients[target.name + LORA_B_SUFFIX]
            )
        vjp_evaluations += 1
    output_bases = {}
    for target in contract.targets:
        value = torch.cat(range_chunks[target.name], dim=1)
        output_bases[target.name], _ = torch.linalg.qr(value, mode="reduced")

    co_range_chunks: dict[str, list[torch.Tensor]] = {
        target.name: [] for target in contract.targets
    }
    for start in range(0, width, residual_rank):
        residual = {}
        leaves = {}
        names = []
        for target in contract.targets:
            a_name = target.name + LORA_A_SUFFIX
            b_name = target.name + LORA_B_SUFFIX
            leaves[a_name] = torch.zeros(
                residual_rank,
                target.in_features,
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            residual[a_name] = leaves[a_name]
            residual[b_name] = output_bases[target.name][
                :, start : start + residual_rank
            ]
            names.append(a_name)
        gradients = _gradient_rows(
            state=rank_reserved_state(carrier, residual, contract, carrier_rank),
            leaves=leaves,
            names=tuple(names),
            response=response,
            bank=bank,
            objective=objective,
            responsibilities=responsibilities,
            barrier_active=barrier_active,
            config=config,
        )
        for target in contract.targets:
            co_range_chunks[target.name].append(
                gradients[target.name + LORA_A_SUFFIX]
            )
        vjp_evaluations += 1

    direction = {}
    unnormalized_derivative = {}
    projected_gradient_energy = 0.0
    dense_count = 0
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        z = torch.cat(co_range_chunks[target.name], dim=0)
        left, singular, right = torch.linalg.svd(z, full_matrices=False)
        singular = singular[:residual_rank]
        root = torch.sqrt(singular.clamp_min(0.0))
        direction[a_name] = root[:, None] * right[:residual_rank]
        direction[b_name] = -(
            output_bases[target.name] @ left[:, :residual_rank]
        ) * root[None, :]
        unnormalized_derivative[target.name] = -float(singular.square().sum())
        projected_gradient_energy += float(singular.square().sum())
        dense_count += target.in_features * target.out_features
    direction, scales = _unit_effective_direction(
        direction=direction,
        carrier=carrier,
        contract=contract,
        carrier_rank=carrier_rank,
    )
    derivative = sum(
        scales[target.name] * unnormalized_derivative[target.name]
        for target in contract.targets
    )
    if not math.isfinite(derivative) or derivative >= 0.0:
        raise ValueError("matrix-free sketch is not a descent direction")
    gradient_rms = math.sqrt(projected_gradient_energy / max(dense_count, 1))
    return direction, derivative, gradient_rms, vjp_evaluations


def _preconditioned_tangent_direction(
    *,
    carrier: Mapping[str, torch.Tensor],
    residual: Mapping[str, torch.Tensor],
    bank: Stage1EffectBank,
    contract: LoRAContract,
    response: ResponseFunction,
    objective: ParticleObjective,
    responsibilities: torch.Tensor,
    barrier_active: torch.Tensor,
    config: RealizationConfig,
    carrier_rank: int,
) -> tuple[dict[str, torch.Tensor], float, float]:
    leaves = {
        name: value.detach().requires_grad_(True) for name, value in residual.items()
    }
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
    direction = {}
    derivative = 0.0
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        a = residual[a_name].float()
        b = residual[b_name].float()
        grad_a = gradients[a_name].float()
        grad_b = gradients[b_name].float()
        gram_a = a @ a.T
        gram_b = b.T @ b
        damping_a = float(config.gram_damping_fraction) * torch.diagonal(
            gram_a
        ).mean().clamp_min(1e-12)
        damping_b = float(config.gram_damping_fraction) * torch.diagonal(
            gram_b
        ).mean().clamp_min(1e-12)
        identity = torch.eye(a.shape[0], device=a.device, dtype=a.dtype)
        delta_b = -torch.linalg.solve(
            gram_a + damping_a * identity, grad_b.T
        ).T
        delta_a = -torch.linalg.solve(
            gram_b + damping_b * identity, grad_a
        )
        tangent_a = torch.cat([a, delta_a], dim=0)
        tangent_b = torch.cat([delta_b, b], dim=1)
        energy = effective_inner_product(
            tangent_b, tangent_a, tangent_b, tangent_a
        )
        if not torch.isfinite(energy) or float(energy) <= 0.0:
            raise ValueError(f"preconditioned tangent is degenerate for {target.name}")
        scale = torch.sqrt(
            _carrier_energy(carrier, a_name, b_name, carrier_rank) / energy
        )
        direction[a_name] = tangent_a.detach()
        direction[b_name] = (scale * tangent_b).detach()
        derivative += float(
            scale
            * (
                torch.sum(grad_b * delta_b)
                + torch.sum(grad_a * delta_a)
            )
        )
    if not math.isfinite(derivative) or derivative >= 0.0:
        raise ValueError("preconditioned tangent is not a descent direction")
    return direction, derivative, float(_gradient_rms(tuple(gradients.values())))


def _retract_residual_sum(
    *,
    residual: Mapping[str, torch.Tensor],
    direction: Mapping[str, torch.Tensor],
    alpha: float,
    contract: LoRAContract,
    carrier_rank: int,
) -> dict[str, torch.Tensor]:
    residual_rank = int(contract.rank) - int(carrier_rank)
    result = {}
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        joined_a = torch.cat([residual[a_name], direction[a_name]], dim=0)
        joined_b = torch.cat(
            [residual[b_name], float(alpha) * direction[b_name]], dim=1
        )
        a, b = canonicalize_low_rank_factors(
            joined_a, joined_b, output_rank=residual_rank
        )
        result[a_name] = a.detach()
        result[b_name] = b.detach()
    return result


def _accept_direction(
    *,
    carrier: Mapping[str, torch.Tensor],
    residual: Mapping[str, torch.Tensor],
    direction: Mapping[str, torch.Tensor],
    current: RealizationSnapshot,
    bank: Stage1EffectBank,
    contract: LoRAContract,
    response: ResponseFunction,
    objective: ParticleObjective,
    config: RealizationConfig,
    carrier_rank: int,
) -> tuple[
    dict[str, torch.Tensor],
    RealizationSnapshot,
    torch.Tensor,
    torch.Tensor,
    float,
    int,
] | None:
    for backtrack, alpha in enumerate(config.backtrack_scales):
        candidate = _retract_residual_sum(
            residual=residual,
            direction=direction,
            alpha=float(alpha),
            contract=contract,
            carrier_rank=carrier_rank,
        )
        trust = rank_reserved_relative_distance(
            candidate, carrier, contract, carrier_rank
        )
        if not torch.isfinite(trust) or float(trust) > float(config.trust_region):
            continue
        snapshot, responsibilities, barrier_active = _evaluate_residual(
            carrier=carrier,
            residual=candidate,
            bank=bank,
            contract=contract,
            response=response,
            objective=objective,
            config=config,
            carrier_rank=carrier_rank,
        )
        if math.isfinite(snapshot.total) and snapshot.total < current.total:
            return (
                candidate,
                snapshot,
                responsibilities,
                barrier_active,
                float(alpha),
                backtrack,
            )
    return None


def solve_effective_update_particle_effects(
    *,
    carrier: Mapping[str, torch.Tensor],
    bank: Stage1EffectBank,
    contract: LoRAContract,
    response: ResponseFunction,
    config: RealizationConfig,
    carrier_rank: int,
) -> RealizationOutcome:
    """Solve in the effective-update metric while keeping the carrier frozen."""

    validate_lora_state(carrier, contract)
    residual_rank = int(contract.rank) - int(carrier_rank)
    if carrier_rank <= 0 or residual_rank <= 0:
        raise ValueError("rank-reserved solver requires carrier and residual ranks")
    if (
        int(config.max_vjp_evaluations) != 12
        or int(config.sketch_width) != 2 * residual_rank
        or tuple(config.backtrack_scales) != (1.0, 0.5, 0.25, 0.125, 0.0625)
        or config.microbatch_size <= 0
    ):
        raise ValueError("ECP effective-update solver contract changed")
    device = bank.suffix_noise.device
    objective = build_particle_objective(bank, config)
    residual = initial_rank_reserved_residual(
        carrier, contract, carrier_rank, device
    )
    initial_state = rank_reserved_state(carrier, residual, contract, carrier_rank)
    initial_state_is_exact_carrier = all(
        torch.equal(initial_state[name], carrier[name]) for name in carrier
    )
    if not initial_state_is_exact_carrier:
        raise ValueError("zero residual does not reproduce the exact carrier")
    initial, responsibilities, barrier_active = _evaluate_residual(
        carrier=carrier,
        residual=residual,
        bank=bank,
        contract=contract,
        response=response,
        objective=objective,
        config=config,
        carrier_rank=carrier_rank,
    )
    current = initial
    history = []
    direction, derivative, gradient_rms, vjp_evaluations = (
        _matrix_free_initial_direction(
            carrier=carrier,
            bank=bank,
            contract=contract,
            response=response,
            objective=objective,
            responsibilities=responsibilities,
            barrier_active=barrier_active,
            config=config,
            carrier_rank=carrier_rank,
        )
    )
    initial_derivative = derivative
    initial_gradient_rms = gradient_rms
    accepted = _accept_direction(
        carrier=carrier,
        residual=residual,
        direction=direction,
        current=current,
        bank=bank,
        contract=contract,
        response=response,
        objective=objective,
        config=config,
        carrier_rank=carrier_rank,
    )
    stop_reason = "initial_backtracking_failed"
    if accepted is not None:
        before = current
        residual, current, responsibilities, barrier_active, alpha, backtrack = accepted
        history.append(
            RealizationStep(
                iteration=0,
                phase="matrix_free_initial_sketch",
                before=before,
                after=current,
                accepted_alpha=alpha,
                backtrack_index=backtrack,
                directional_derivative=derivative,
                gradient_rms=gradient_rms,
                cumulative_vjp_evaluations=vjp_evaluations,
            )
        )
        stop_reason = "vjp_budget_exhausted"
        while vjp_evaluations < int(config.max_vjp_evaluations):
            direction, derivative, gradient_rms = _preconditioned_tangent_direction(
                carrier=carrier,
                residual=residual,
                bank=bank,
                contract=contract,
                response=response,
                objective=objective,
                responsibilities=responsibilities,
                barrier_active=barrier_active,
                config=config,
                carrier_rank=carrier_rank,
            )
            vjp_evaluations += 1
            accepted = _accept_direction(
                carrier=carrier,
                residual=residual,
                direction=direction,
                current=current,
                bank=bank,
                contract=contract,
                response=response,
                objective=objective,
                config=config,
                carrier_rank=carrier_rank,
            )
            if accepted is None:
                stop_reason = "preconditioned_backtracking_failed"
                break
            before = current
            (
                residual,
                current,
                responsibilities,
                barrier_active,
                alpha,
                backtrack,
            ) = accepted
            history.append(
                RealizationStep(
                    iteration=len(history),
                    phase="gauge_preconditioned_tangent",
                    before=before,
                    after=current,
                    accepted_alpha=alpha,
                    backtrack_index=backtrack,
                    directional_derivative=derivative,
                    gradient_rms=gradient_rms,
                    cumulative_vjp_evaluations=vjp_evaluations,
                )
            )
    final_state = rank_reserved_state(carrier, residual, contract, carrier_rank)
    validate_lora_state(final_state, contract)
    best_member = _best_member_effect_objective(bank, objective, config)
    denominator = initial.total - best_member
    gap_recovery = (
        (initial.total - current.total) / denominator if denominator > 0.0 else 0.0
    )
    return RealizationOutcome(
        state={name: value.detach() for name, value in final_state.items()},
        initial=initial,
        final=current,
        history=tuple(history),
        initial_state_is_exact_carrier=initial_state_is_exact_carrier,
        initial_directional_derivative=initial_derivative,
        initial_gradient_rms=initial_gradient_rms,
        vjp_evaluations=vjp_evaluations,
        stop_reason=stop_reason,
        best_member_effect_objective=best_member,
        objective_gap_recovery=gap_recovery,
    )
