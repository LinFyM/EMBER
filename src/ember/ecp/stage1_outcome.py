"""Action-informed reachable Program coordinates for Stage 1 outcome credit."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.program import ECPProgram
from ember.reward.protocol import RewardProtocolError


@dataclass(frozen=True)
class ActionGuidedProgramPerturbation:
    """One paired perturbation in a fixed compiler's Program tangent space."""

    family: TargetFamily
    owner_indices: tuple[int, ...]
    epsilon: torch.Tensor
    sigma: float
    direction: torch.Tensor
    direction_norm_sq: torch.Tensor
    plus_program: ECPProgram
    minus_program: ECPProgram
    active_elements: int


def _with_process(program: ECPProgram, process: torch.Tensor) -> ECPProgram:
    return ECPProgram(
        language=program.language.detach(),
        scene=program.scene.detach(),
        process=process,
        presence=program.presence.detach(),
        uncertainty=program.uncertainty.detach(),
    )


def action_guided_program_perturbation(
    program: ECPProgram,
    action_gradient: torch.Tensor,
    owners: tuple[TargetOwner, ...],
    *,
    family: TargetFamily,
    sigma: float,
) -> ActionGuidedProgramPerturbation:
    """Normalize one family block of exact action descent in Program space.

    The block contains every visible ordered event and every native owner in the
    selected target family. Its direction has the same L2 norm as the current
    Program block, so ``sigma`` is a relative Program perturbation. Both arms
    are therefore outputs of the same permanently frozen compiler.
    """

    process = program.process.detach().float()
    if (
        process.ndim != 4
        or process.shape[0] != 1
        or action_gradient.shape != process.shape
        or process.shape[2] != len(owners)
        or sigma <= 0
        or not bool(torch.isfinite(process).all())
        or not bool(torch.isfinite(action_gradient).all())
    ):
        raise RewardProtocolError("invalid action-guided Program surface")
    owner_indices = tuple(
        owner.index for owner in owners if owner.family is family
    )
    if not owner_indices:
        raise RewardProtocolError("Program family block has no owners")
    owner_mask = torch.zeros(
        (1, 1, len(owners), 1), device=process.device, dtype=process.dtype
    )
    owner_mask[:, :, owner_indices] = 1.0
    event_mask = (program.presence.detach().float() > 0).to(process)[
        :, :, None, None
    ]
    mask = owner_mask * event_mask
    block = process * mask
    gradient = action_gradient.detach().float() * mask
    block_norm_sq = block.square().sum()
    gradient_norm_sq = gradient.square().sum()
    if (
        not bool(torch.isfinite(block_norm_sq + gradient_norm_sq))
        or float(block_norm_sq) <= 0
        or float(gradient_norm_sq) <= 0
    ):
        raise RewardProtocolError("action gradient did not reach Program block")
    direction = -gradient * (block_norm_sq / gradient_norm_sq).sqrt()
    direction_norm_sq = direction.square().sum()
    plus = (process + sigma * direction).to(program.process)
    minus = (process - sigma * direction).to(program.process)
    active_elements = int(mask.sum().item() * process.shape[-1])
    return ActionGuidedProgramPerturbation(
        family=family,
        owner_indices=owner_indices,
        epsilon=process.new_ones((1, 1)),
        sigma=float(sigma),
        direction=direction,
        direction_norm_sq=direction_norm_sq,
        plus_program=_with_process(program, plus),
        minus_program=_with_process(program, minus),
        active_elements=active_elements,
    )


def action_guided_program_leaf_gradient(
    perturbation: ActionGuidedProgramPerturbation,
    coordinate_gradient: torch.Tensor,
    *,
    weight: float,
) -> torch.Tensor:
    """Return a loss gradient whose Program-direction dot is reward ascent."""

    if (
        coordinate_gradient.shape != (1, 1)
        or weight <= 0
        or not bool(torch.isfinite(coordinate_gradient).all())
        or float(perturbation.direction_norm_sq) <= 0
    ):
        raise RewardProtocolError("invalid action-guided Program gradient")
    coefficient = (
        -weight
        * coordinate_gradient[0, 0].float()
        / perturbation.direction_norm_sq.float()
    )
    return (coefficient * perturbation.direction).to(
        perturbation.plus_program.process
    )
