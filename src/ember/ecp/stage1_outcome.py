"""Structured Stage 1 coordinates for task-equal closed-loop calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ember.ecp.stage1 import ECPStage1Output
from ember.reward.credit import AntitheticCredit
from ember.reward.protocol import RewardProtocolError


PROGRAM_BINDING = "program_binding"
COMPILER_BINDING = "compiler_binding"
OUTCOME_COORDINATES = (PROGRAM_BINDING, COMPILER_BINDING)


@dataclass(frozen=True)
class OutcomePerturbation:
    coordinate: str
    epsilon: torch.Tensor
    sigma: float
    plus_kwargs: dict[str, torch.Tensor]
    minus_kwargs: dict[str, torch.Tensor]


def _rademacher(shape: tuple[int, ...], *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (
        torch.randint(0, 2, shape, generator=generator, dtype=torch.float32)
        .mul_(2.0)
        .sub_(1.0)
    )


def structured_outcome_perturbation(
    output: ECPStage1Output,
    *,
    coordinate: str,
    sigma: float,
    seed: int,
) -> OutcomePerturbation:
    """Build one rank-semantic-free antithetic perturbation."""

    if coordinate not in OUTCOME_COORDINATES or sigma <= 0:
        raise RewardProtocolError("invalid ECP outcome coordinate")
    device = output.teacher.program.process.device
    if coordinate == PROGRAM_BINDING:
        shape = (1, *output.teacher.program.process.shape[1:3], 1)
        epsilon = _rademacher(shape, seed=seed)
        presence = output.teacher.program.presence[:, :, None, None].float().cpu()
        epsilon = epsilon * (presence > 0).float()
        offset = (sigma * epsilon).to(device)
        plus = {"evidence_logit_offset": offset}
        minus = {"evidence_logit_offset": -offset}
    else:
        owners = int(output.consensus_compilation.rank_angles.shape[1])
        epsilon = _rademacher((1, owners, 1), seed=seed)
        offset = (sigma * epsilon).to(device)
        plus = {"rank_angle_offset": offset}
        minus = {"rank_angle_offset": -offset}
    return OutcomePerturbation(
        coordinate=coordinate,
        epsilon=epsilon.reshape(1, -1).to(device),
        sigma=float(sigma),
        plus_kwargs=plus,
        minus_kwargs=minus,
    )


def outcome_coordinate(
    output: ECPStage1Output, coordinate: str
) -> torch.Tensor:
    """Return the differentiable base coordinate matching the shared offset."""

    if coordinate == PROGRAM_BINDING:
        logits = output.teacher.evidence_gate_logits.float()
        weights = output.teacher.member_weights.to(logits)
        value = torch.einsum("m,meoj->eoj", weights, logits)
        value = value * output.teacher.program.presence[0, :, None, None]
        return value.reshape(1, -1)
    if coordinate == COMPILER_BINDING:
        return output.consensus_compilation.rank_angles.float().mean(-1)
    raise RewardProtocolError("invalid ECP outcome coordinate")


def outcome_surrogate_loss(
    output: ECPStage1Output,
    credit: AntitheticCredit,
    *,
    coordinate: str,
    weight: float,
) -> torch.Tensor:
    value = outcome_coordinate(output, coordinate)
    if (
        weight < 0
        or value.shape != credit.gradient.shape
        or not bool(torch.isfinite(credit.gradient).all())
    ):
        raise RewardProtocolError("invalid ECP outcome surrogate")
    return -weight * (value * credit.gradient.to(value)).sum()


def perturbation_forward_kwargs(
    perturbation: OutcomePerturbation, *, plus: bool
) -> dict[str, Any]:
    return perturbation.plus_kwargs if plus else perturbation.minus_kwargs
