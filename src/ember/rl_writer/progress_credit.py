"""Frozen task-grounded visual progress representation for pair tie-breaking."""

from __future__ import annotations

import torch

from ember.reward.protocol import RewardProtocolError


def normalized_progress_components(
    grounded_evidence: torch.Tensor,
    interactions: torch.Tensor,
    valid_task_tokens: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Pack RMS-normalized task-token evidence plus one interaction component."""

    if (
        grounded_evidence.ndim != 3
        or interactions.ndim != 2
        or grounded_evidence.shape[0] != interactions.shape[0]
        or grounded_evidence.shape[-1] != interactions.shape[-1]
        or valid_task_tokens.ndim != 2
        or valid_task_tokens.shape[0] != 1
        or valid_task_tokens.shape[1] != grounded_evidence.shape[1]
        or valid_task_tokens.dtype != torch.bool
        or not bool(valid_task_tokens.any())
        or epsilon <= 0
    ):
        raise RewardProtocolError("invalid task-grounded progress evidence")
    selected = grounded_evidence[:, valid_task_tokens[0]].float()
    packed = torch.cat((selected, interactions[:, None].float()), dim=1)
    denominator = packed.square().mean(dim=-1, keepdim=True).add(epsilon).sqrt()
    normalized = packed / denominator
    if not bool(torch.isfinite(normalized).all()):
        raise RewardProtocolError("non-finite task-grounded progress components")
    return normalized


def semantic_progress_utilities(
    teacher_start: torch.Tensor,
    teacher_goal: torch.Tensor,
    rollout_starts: torch.Tensor,
    rollout_terminals: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project start-relative rollout changes onto the teacher content change."""

    if (
        teacher_start.ndim != 2
        or teacher_goal.shape != teacher_start.shape
        or rollout_starts.ndim != 3
        or rollout_terminals.shape != rollout_starts.shape
        or rollout_starts.shape[1:] != teacher_start.shape
        or rollout_starts.shape[0] <= 0
        or epsilon <= 0
    ):
        raise RewardProtocolError("invalid semantic progress utility inputs")
    values = (teacher_start, teacher_goal, rollout_starts, rollout_terminals)
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise RewardProtocolError("semantic progress utility received non-finite input")

    direction = teacher_goal.float() - teacher_start.float()
    displacement = rollout_terminals.float() - rollout_starts.float()
    component_energy = direction.square().sum(dim=-1)
    total_energy = component_energy.sum()
    if float(total_energy) <= 0:
        return displacement.new_zeros(displacement.shape[0]), component_energy
    weights = component_energy / (total_energy + epsilon)
    direction_norm = direction.square().sum(dim=-1).sqrt()
    displacement_norm = displacement.square().sum(dim=-1).sqrt()
    alignment = (displacement * direction[None]).sum(dim=-1) / (
        displacement_norm * direction_norm[None] + epsilon
    )
    magnitude = (displacement_norm / (direction_norm[None] + epsilon)).clamp(max=1.0)
    utilities = (weights[None] * alignment * magnitude).sum(dim=-1)
    if not bool(torch.isfinite(utilities).all()) or bool((utilities.abs() > 1.00001).any()):
        raise RewardProtocolError("semantic progress utility escaped its finite bound")
    return utilities, component_energy
