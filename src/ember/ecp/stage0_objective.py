"""Task-equal Stage 0 observer calibration objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from ember.ecp.stage0 import ECPStage0Output


@dataclass(frozen=True)
class ECPStage0Loss:
    total: torch.Tensor
    action_alignment: torch.Tensor
    same_task_consistency: torch.Tensor
    uncertainty_calibration: torch.Tensor
    presence_consistency: torch.Tensor
    cross_task_contrast: torch.Tensor
    posterior_entropy: torch.Tensor
    presence_sparsity: torch.Tensor


def _event_action_targets(
    posterior: torch.Tensor,
    frame_action_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mass = posterior.sum(1).clamp_min(1e-6)
    target = torch.einsum(
        "vte,vtpa->vepa", posterior, frame_action_targets
    ) / mass[:, :, None, None]
    return target, mass


def _contrastive_pair(
    summaries: torch.Tensor,
    negative_summaries: torch.Tensor | None,
    temperature: float,
) -> torch.Tensor:
    first, second = F.normalize(summaries.float(), dim=-1).unbind(0)
    positive = (first * second).sum()[None]
    if negative_summaries is None or not negative_summaries.numel():
        return 1.0 - positive.squeeze(0)
    negatives = F.normalize(negative_summaries.float(), dim=-1)
    logits = torch.cat((positive, first @ negatives.T)) / temperature
    return F.cross_entropy(
        logits[None], torch.zeros(1, dtype=torch.long, device=logits.device)
    )


def ecp_stage0_loss(
    output: ECPStage0Output,
    frame_action_targets: torch.Tensor,
    *,
    weights: Mapping[str, float],
    negative_summaries: torch.Tensor | None = None,
    contrastive_temperature: float = 0.1,
) -> ECPStage0Loss:
    """Ground an ordered same-task video pair without exposing task ID to ECP."""

    target, mass = _event_action_targets(
        output.state_posterior, frame_action_targets
    )
    event_weights = mass / mass.sum(1, keepdim=True)
    action_error = (
        output.action_phase_predictions.float() - target.float()
    ).square().mean(dim=(2, 3))
    action_alignment = (event_weights * action_error).sum(1).mean()

    joint_presence = (output.presence[0] * output.presence[1]).detach()
    joint_presence = joint_presence / joint_presence.sum().clamp_min(1e-6)
    process_delta = F.smooth_l1_loss(
        output.process[0].float(), output.process[1].float(), reduction="none"
    ).mean(dim=(1, 2))
    same_task_consistency = (joint_presence * process_delta).sum()

    variance = (
        output.uncertainty[0].float().square()
        + output.uncertainty[1].float().square()
        + 1e-3
    )
    squared_delta = (
        output.process[0].float() - output.process[1].float()
    ).square()
    # Express the Gaussian scale term relative to the fixed variance floor.
    # This is only a constant shift from log(variance), so gradients are
    # unchanged, while every reported loss component stays non-negative.
    uncertainty_by_event = 0.5 * (
        squared_delta / variance + (variance / 1e-3).log()
    ).mean(dim=(1, 2))
    uncertainty_calibration = (joint_presence * uncertainty_by_event).sum()
    presence_consistency = F.mse_loss(
        output.presence[0].float(), output.presence[1].float()
    )
    cross_task_contrast = _contrastive_pair(
        output.program_summary,
        negative_summaries,
        contrastive_temperature,
    )

    posterior = output.state_posterior.float().clamp_min(1e-8)
    valid = output.frame_mask.sum().clamp_min(1)
    posterior_entropy = -(
        posterior * posterior.log()
    ).sum(-1).masked_fill(~output.frame_mask, 0.0).sum() / valid
    presence_sparsity = output.presence.float().mean()
    terms = {
        "action_alignment": action_alignment,
        "same_task_consistency": same_task_consistency,
        "uncertainty_calibration": uncertainty_calibration,
        "presence_consistency": presence_consistency,
        "cross_task_contrast": cross_task_contrast,
        "posterior_entropy": posterior_entropy,
        "presence_sparsity": presence_sparsity,
    }
    missing = set(terms) - set(weights)
    if missing:
        raise ValueError(f"missing ECP Stage 0 loss weights: {sorted(missing)}")
    total = sum(float(weights[name]) * value for name, value in terms.items())
    return ECPStage0Loss(total=total, **terms)
