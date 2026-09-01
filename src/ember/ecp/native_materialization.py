"""Canonical rank4 factorization and sole rank12+4 adapter materialization."""

from __future__ import annotations

from typing import Mapping

import torch

from ember.ecp.native_factors import (
    G1_RESIDUAL_RANK,
    NativeFactorError,
    NativeFactorResidual,
)
from ember.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    LoRAContract,
    validate_lora_state,
)


def small_core_balanced_svd(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonicalize B@A through a rank-four core without forming full delta-W."""

    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[0] != G1_RESIDUAL_RANK
        or b.shape[1] != G1_RESIDUAL_RANK
        or a.shape[1] < G1_RESIDUAL_RANK
        or b.shape[0] < G1_RESIDUAL_RANK
    ):
        raise NativeFactorError("G1 residual factors are not rank four")
    return low_rank_balanced_svd(a, b, output_rank=G1_RESIDUAL_RANK)


def low_rank_balanced_svd(
    a: torch.Tensor, b: torch.Tensor, *, output_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Best balanced low-rank factors for B@A through only its small core."""

    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[0] != b.shape[1]
        or not 0 < output_rank <= a.shape[0]
    ):
        raise NativeFactorError("low-rank core factorization changed shape")
    qb, rb = torch.linalg.qr(b.float(), mode="reduced")
    qa, ra = torch.linalg.qr(a.float().transpose(0, 1), mode="reduced")
    u, singular, vh = torch.linalg.svd(rb @ ra.transpose(0, 1), full_matrices=False)
    effective_rank = min(output_rank, singular.shape[0])
    u = u[:, :effective_rank]
    singular = singular[:effective_rank]
    vh = vh[:effective_rank]
    root = singular.clamp_min(0).sqrt()
    canonical_b = (qb @ u) * root[None]
    canonical_a = root[:, None] * (vh @ qa.transpose(0, 1))
    pivots = canonical_a.abs().argmax(-1)
    signs = torch.sign(
        canonical_a[
            torch.arange(effective_rank, device=canonical_a.device), pivots
        ]
    )
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    canonical_a = canonical_a * signs[:, None]
    canonical_b = canonical_b * signs[None]
    if effective_rank < output_rank:
        canonical_a = torch.cat(
            (
                canonical_a,
                canonical_a.new_zeros(output_rank - effective_rank, a.shape[1]),
            ),
            dim=0,
        )
        canonical_b = torch.cat(
            (
                canonical_b,
                canonical_b.new_zeros(b.shape[0], output_rank - effective_rank),
            ),
            dim=1,
        )
    return canonical_a.to(a), canonical_b.to(b)


def residual_lora_state(
    residual: NativeFactorResidual,
    contract: LoRAContract,
    *,
    canonicalize: bool,
) -> dict[str, torch.Tensor]:
    """Convert native factors to one complete residual state of the contract rank."""

    if len(residual.a) != len(contract.targets) or len(residual.b) != len(
        contract.targets
    ):
        raise NativeFactorError("native residual target count changed")
    residual_rank = int(contract.rank)
    if residual_rank <= 0:
        raise NativeFactorError("native residual rank changed")
    state: dict[str, torch.Tensor] = {}
    for target, a, b in zip(contract.targets, residual.a, residual.b, strict=True):
        if a.shape != (residual_rank, target.in_features) or b.shape != (
            residual_rank,
            target.out_features,
        ):
            raise NativeFactorError(f"native factor shape changed: {target.name}")
        b_columns = b.transpose(0, 1)
        if canonicalize:
            a, b_columns = low_rank_balanced_svd(
                a, b_columns, output_rank=residual_rank
            )
        state[target.name + LORA_A_SUFFIX] = a
        state[target.name + LORA_B_SUFFIX] = b_columns
    return state


def compose_rank12_plus_rank4(
    *,
    carrier_state: Mapping[str, torch.Tensor],
    residual_state: Mapping[str, torch.Tensor],
    rank16_contract: LoRAContract,
) -> dict[str, torch.Tensor]:
    """Concatenate disjoint carrier/residual slots into the sole rank16 adapter."""

    result: dict[str, torch.Tensor] = {}
    for target in rank16_contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        carrier_a = carrier_state.get(a_name)
        carrier_b = carrier_state.get(b_name)
        residual_a = residual_state.get(a_name)
        residual_b = residual_state.get(b_name)
        if (
            carrier_a is None
            or carrier_b is None
            or residual_a is None
            or residual_b is None
            or carrier_a.shape != (12, target.in_features)
            or carrier_b.shape != (target.out_features, 12)
            or residual_a.shape != (G1_RESIDUAL_RANK, target.in_features)
            or residual_b.shape != (target.out_features, G1_RESIDUAL_RANK)
        ):
            raise NativeFactorError(f"rank12+4 composition changed: {target.name}")
        result[a_name] = torch.cat((carrier_a, residual_a), dim=0)
        result[b_name] = torch.cat((carrier_b, residual_b), dim=1)
    validate_lora_state(result, rank16_contract)
    return result


def extract_rank12_carrier(
    state: Mapping[str, torch.Tensor], contract: LoRAContract
) -> dict[str, torch.Tensor]:
    """Extract the declared active carrier slots from its complete rank16 file."""

    validate_lora_state(state, contract)
    result: dict[str, torch.Tensor] = {}
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        a = state[a_name]
        b = state[b_name]
        if torch.count_nonzero(b[:, 12:]).item() != 0:
            raise NativeFactorError(
                f"carrier has an active task-residual slot: {target.name}"
            )
        result[a_name] = a[:12]
        result[b_name] = b[:, :12]
    return result


def extract_rank4_residual(
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    *,
    carrier_state: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Extract mobile slots and optionally prove their carrier prefix is unchanged."""

    validate_lora_state(state, contract)
    result: dict[str, torch.Tensor] = {}
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        a = state[a_name]
        b = state[b_name]
        if carrier_state is not None and (
            not torch.equal(a[:12].cpu(), carrier_state[a_name][:12].cpu())
            or not torch.equal(b[:, :12].cpu(), carrier_state[b_name][:, :12].cpu())
        ):
            raise NativeFactorError(
                f"mobile adapter changed the frozen carrier: {target.name}"
            )
        result[a_name] = a[12:]
        result[b_name] = b[:, 12:]
    return result
