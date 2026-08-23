"""Rank-reserved LoRA parameterization and analytic capacity projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from ember.ecp.low_rank import canonicalize_low_rank_factors
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract, validate_lora_state


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


def effective_inner_product(
    left_b: torch.Tensor,
    left_a: torch.Tensor,
    right_b: torch.Tensor,
    right_a: torch.Tensor,
) -> torch.Tensor:
    return torch.sum((left_b.T @ right_b) * (left_a @ right_a.T))


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
        residual_energy = effective_inner_product(
            residual_b, residual_a, residual_b, residual_a
        )
        carrier_energy = effective_inner_product(
            carrier_b, carrier_a, carrier_b, carrier_a
        )
        distances.append(residual_energy / carrier_energy.clamp_min(1e-10))
    return torch.stack(distances).mean()


def initial_rank_reserved_residual(
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


def interpolate_rank_reserved_endpoint(
    *,
    carrier: Mapping[str, torch.Tensor],
    endpoint: Mapping[str, torch.Tensor],
    alpha: float,
    contract: LoRAContract,
    carrier_rank: int,
) -> dict[str, torch.Tensor]:
    """Scale one canonical residual path in effective-update coordinates."""

    residual_rank = int(contract.rank) - int(carrier_rank)
    if residual_rank <= 0 or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("rank-reserved path interpolation changed")
    scale = float(alpha) ** 0.5
    result = {}
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        carrier_a = carrier[a_name]
        carrier_b = carrier[b_name]
        endpoint_a = endpoint[a_name]
        endpoint_b = endpoint[b_name]
        if (
            endpoint_a.shape != carrier_a.shape
            or endpoint_b.shape != carrier_b.shape
            or torch.count_nonzero(carrier_b[:, carrier_rank:])
            or not torch.equal(endpoint_a[:carrier_rank], carrier_a[:carrier_rank])
            or not torch.equal(endpoint_b[:, :carrier_rank], carrier_b[:, :carrier_rank])
            or endpoint_a[carrier_rank:].shape[0] != residual_rank
        ):
            raise ValueError("rank-reserved endpoint lost its carrier coordinate")
        result[a_name] = torch.cat(
            [carrier_a[:carrier_rank], endpoint_a[carrier_rank:] * scale], dim=0
        )
        result[b_name] = torch.cat(
            [carrier_b[:, :carrier_rank], endpoint_b[:, carrier_rank:] * scale],
            dim=1,
        )
    validate_lora_state(result, contract)
    return result


def project_expert_onto_rank_reserved_residual(
    *,
    carrier: Mapping[str, torch.Tensor],
    expert: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    carrier_rank: int,
) -> tuple[dict[str, torch.Tensor], tuple[RankReservedProjectionTarget, ...]]:
    """Add the best mobile residual in the ranks reserved by the carrier."""

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
        expert_energy = effective_inner_product(eb, ea, eb, ea)
        carrier_energy = effective_inner_product(cb, ca, cb, ca)
        expert_carrier = effective_inner_product(eb, ea, cb, ca)
        expert_residual = effective_inner_product(eb, ea, rb, ra)
        carrier_residual = effective_inner_product(cb, ca, rb, ra)
        residual_energy = effective_inner_product(rb, ra, rb, ra)
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
