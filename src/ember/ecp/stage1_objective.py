"""Distributional successful-policy objective for ECP Stage 1B."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ember.ecp.policy_effects import PolicyEffectResponse
from ember.ecp.stage1_equivalence import Stage1EffectBank


@dataclass(frozen=True)
class RealizationConfig:
    steps: int = 12
    step_rms: float = 0.0002
    step_decay_power: float = 0.5
    temperature: float = 0.25
    owner_weight: float = 1.0
    flow_weight: float = 1.0
    action_weight: float = 1.0
    carrier_barrier_weight: float = 0.25
    preservation_weight: float = 0.1
    signal_floor_fraction: float = 0.05
    minimum_confidence: float = 0.25
    trust_region: float = 1.5
    trust_weight: float = 0.05
    microbatch_size: int = 1


@dataclass(frozen=True)
class RealizationSnapshot:
    total: float
    equivalence: float
    carrier_barrier: float
    preservation: float
    trust_distance: float
    trust_penalty: float
    mean_confidence: float


@dataclass(frozen=True)
class ParticleObjective:
    scales: PolicyEffectResponse
    group_ids: torch.Tensor
    group_weights: torch.Tensor
    state_weights: torch.Tensor
    reliability: torch.Tensor
    confidence: torch.Tensor
    baseline: torch.Tensor
    baseline_uses_carrier: torch.Tensor


def response_fields(value: PolicyEffectResponse) -> tuple[torch.Tensor, ...]:
    return value.owner, value.flow, value.action


def _component_weights(config: RealizationConfig) -> tuple[float, ...]:
    return config.owner_weight, config.flow_weight, config.action_weight


def _member_scales(
    bank: Stage1EffectBank, config: RealizationConfig
) -> PolicyEffectResponse:
    values = []
    for members, source in zip(
        response_fields(bank.members), response_fields(bank.source), strict=True
    ):
        reduction = tuple(range(2, members.ndim))
        signal = (
            (members.float() - source.float().unsqueeze(0)).square().mean(dim=reduction)
        )
        floor = float(config.signal_floor_fraction) * signal.mean().clamp_min(1e-8)
        values.append(signal + floor)
    return PolicyEffectResponse(*values)


def member_distances(
    candidate: PolicyEffectResponse,
    members: PolicyEffectResponse,
    scales: PolicyEffectResponse,
    config: RealizationConfig,
) -> torch.Tensor:
    result = None
    for candidate_value, member_value, scale, weight in zip(
        response_fields(candidate),
        response_fields(members),
        response_fields(scales),
        _component_weights(config),
        strict=True,
    ):
        reduction = tuple(range(2, member_value.ndim))
        distance = (
            member_value.float() - candidate_value.float().unsqueeze(0)
        ).square().mean(dim=reduction) / scale
        result = (
            float(weight) * distance
            if result is None
            else result + float(weight) * distance
        )
    if result is None:
        raise RuntimeError("ECP policy-effect response is empty")
    return result


def reference_distances(
    candidate: PolicyEffectResponse,
    reference: PolicyEffectResponse,
    scales: PolicyEffectResponse,
    config: RealizationConfig,
) -> torch.Tensor:
    result = None
    for candidate_value, reference_value, scale, weight in zip(
        response_fields(candidate),
        response_fields(reference),
        response_fields(scales),
        _component_weights(config),
        strict=True,
    ):
        reduction = tuple(range(1, candidate_value.ndim))
        distance = (candidate_value.float() - reference_value.float()).square().mean(
            dim=reduction
        ) / scale.mean(0)
        result = (
            float(weight) * distance
            if result is None
            else result + float(weight) * distance
        )
    if result is None:
        raise RuntimeError("ECP policy-effect response is empty")
    return result


def _group_layout(
    category_ids: torch.Tensor, stage_ids: torch.Tensor, progress: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group_ids = torch.empty_like(category_ids)
    group_weights = []
    next_group = 0
    for category in range(4):
        category_mask = category_ids == category
        stages = torch.unique(stage_ids[category_mask], sorted=True)
        raw = []
        for stage in stages:
            mask = category_mask & (stage_ids == stage)
            group_ids[mask] = next_group
            raw.append(0.5 + progress[mask].float().mean())
            next_group += 1
        normalized = torch.stack(raw)
        normalized = 0.25 * normalized / normalized.sum()
        group_weights.extend(normalized.unbind())
    weights = torch.stack(group_weights).to(progress)
    counts = torch.bincount(group_ids, minlength=next_group).to(progress)
    return group_ids, weights, weights[group_ids] / counts[group_ids]


def _group_mean(values: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    groups = int(group_ids.max()) + 1
    return torch.stack(
        [values[:, group_ids == group].mean(1) for group in range(groups)], dim=0
    )


def _softmin(
    group_member_distance: torch.Tensor,
    reliability: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = reliability.log().unsqueeze(0) - group_member_distance / float(temperature)
    return (
        -float(temperature) * torch.logsumexp(logits, dim=1),
        torch.softmax(logits, dim=1),
    )


def build_particle_objective(
    bank: Stage1EffectBank, config: RealizationConfig
) -> ParticleObjective:
    bank.validate()
    scales = _member_scales(bank, config)
    group_ids, group_weights, state_weights = _group_layout(
        bank.category_ids, bank.stage_ids, bank.progress
    )
    reliability = bank.member_reliability.float().clamp_min(1e-4)
    reliability = reliability / reliability.sum()
    source_distance = _group_mean(
        member_distances(bank.source, bank.members, scales, config), group_ids
    )
    carrier_distance = _group_mean(
        member_distances(bank.carrier, bank.members, scales, config), group_ids
    )
    source_softmin, _ = _softmin(source_distance, reliability, config.temperature)
    carrier_softmin, _ = _softmin(carrier_distance, reliability, config.temperature)
    member_mean = PolicyEffectResponse(
        *(
            torch.einsum("m,ms...->s...", reliability, value.float())
            for value in response_fields(bank.members)
        )
    )
    disagreement = member_distances(member_mean, bank.members, scales, config)
    group_disagreement = _group_mean(disagreement, group_ids) @ reliability
    confidence = (1.0 / (1.0 + group_disagreement)).clamp(
        min=float(config.minimum_confidence), max=1.0
    )
    return ParticleObjective(
        scales=scales,
        group_ids=group_ids,
        group_weights=group_weights,
        state_weights=state_weights,
        reliability=reliability,
        confidence=confidence,
        baseline=torch.minimum(source_softmin, carrier_softmin),
        baseline_uses_carrier=carrier_softmin <= source_softmin,
    )


def candidate_snapshot(
    candidate: PolicyEffectResponse,
    bank: Stage1EffectBank,
    objective: ParticleObjective,
    config: RealizationConfig,
    trust_distance: torch.Tensor,
) -> tuple[RealizationSnapshot, torch.Tensor, torch.Tensor]:
    distances = member_distances(candidate, bank.members, objective.scales, config)
    grouped = _group_mean(distances, objective.group_ids)
    softmin, responsibilities = _softmin(
        grouped, objective.reliability, config.temperature
    )
    confidence_weight = 0.25 + 0.75 * objective.confidence
    equivalence = (objective.group_weights * confidence_weight * softmin).sum()
    barrier = (objective.group_weights * torch.relu(softmin - objective.baseline)).sum()
    reference = PolicyEffectResponse(
        *(
            torch.where(
                objective.baseline_uses_carrier[objective.group_ids][
                    (...,) + (None,) * (carrier.ndim - 1)
                ],
                carrier,
                source,
            )
            for carrier, source in zip(
                response_fields(bank.carrier), response_fields(bank.source), strict=True
            )
        )
    )
    preservation_states = reference_distances(
        candidate, reference, objective.scales, config
    )
    preservation = (
        objective.state_weights
        * (1.0 - objective.confidence[objective.group_ids])
        * preservation_states
    ).sum()
    trust_penalty = torch.relu(trust_distance - float(config.trust_region)).square()
    total = (
        equivalence
        + float(config.carrier_barrier_weight) * barrier
        + float(config.preservation_weight) * preservation
        + float(config.trust_weight) * trust_penalty
    )
    return (
        RealizationSnapshot(
            total=float(total),
            equivalence=float(equivalence),
            carrier_barrier=float(barrier),
            preservation=float(preservation),
            trust_distance=float(trust_distance),
            trust_penalty=float(trust_penalty),
            mean_confidence=float(
                (objective.group_weights * objective.confidence).sum()
            ),
        ),
        responsibilities.detach(),
        (softmin > objective.baseline).detach(),
    )
