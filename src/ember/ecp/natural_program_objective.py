"""Mechanism losses for G2 Natural Program."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ember.ecp.natural_program import NaturalProgramOutput


@dataclass(frozen=True)
class NaturalProgramLoss:
    total: torch.Tensor
    action: torch.Tensor
    action_temporal: torch.Tensor
    progress: torch.Tensor
    progress_temporal: torch.Tensor
    rising: torch.Tensor
    contact: torch.Tensor
    predicate: torch.Tensor
    scene_relation: torch.Tensor
    same_task_event: torch.Tensor
    probe_stability: torch.Tensor
    robustness: torch.Tensor
    cross_task_contrast: torch.Tensor
    event_budget: torch.Tensor
    tau_order: torch.Tensor
    uncertainty_calibration: torch.Tensor
    behavior_alignment: torch.Tensor


def program_embedding(output: NaturalProgramOutput) -> torch.Tensor:
    program = output.program
    event_weights = program.rho / program.rho.sum(-1, keepdim=True).clamp_min(1e-6)
    process = torch.einsum(
        "ce,cejd->cjd", event_weights, program.p_process
    ).mean(1)
    scene = program.p_scene.mean(1)
    language = program.p_lang.mean(1)
    return F.normalize((process + scene + language).float(), dim=-1)


def _aligned_local_process(output: NaturalProgramOutput) -> torch.Tensor:
    weights = output.alignment * output.local_presence[:, None]
    mass = weights.sum(-1).clamp_min(1e-6)
    return torch.einsum(
        "vce,vejd->vcjd", weights, output.local_process
    ) / mass[:, :, None, None]


def _cross_task_contrast(
    output: NaturalProgramOutput,
    negative_embeddings: torch.Tensor | None,
    temperature: float,
) -> torch.Tensor:
    anchor = program_embedding(output)[0]
    local = F.normalize(output.local_process.float().mean(dim=(1, 2)), dim=-1)
    positive = (local * anchor[None]).sum(-1).mean()[None]
    if negative_embeddings is None or not negative_embeddings.numel():
        return 1.0 - positive.squeeze(0)
    negatives = F.normalize(negative_embeddings.float(), dim=-1)
    logits = torch.cat((positive, anchor @ negatives.T)) / temperature
    return F.cross_entropy(
        logits[None], torch.zeros(1, dtype=torch.long, device=logits.device)
    )


def _temporal_residual_mse(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Remove each trajectory mean so a constant prediction cannot win."""

    prediction = prediction.float()
    target = target.float()
    prediction = prediction - prediction.mean(1, keepdim=True)
    target = target - target.mean(1, keepdim=True)
    return F.mse_loss(prediction, target)


def _temporal_prediction_losses(
    prediction: Any, batch: Any
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    action = F.mse_loss(
        prediction.action_phases.float(), batch.action_targets.float()
    )
    progress = F.mse_loss(
        prediction.progress.float(), batch.progress_targets.float()
    )
    return (
        action,
        _temporal_residual_mse(prediction.action_phases, batch.action_targets),
        progress,
        _temporal_residual_mse(prediction.progress, batch.progress_targets),
    )


def _event_consistency_losses(
    output: NaturalProgramOutput,
) -> tuple[torch.Tensor, torch.Tensor]:
    aligned = _aligned_local_process(output)
    targets = output.program.p_process.index_select(0, output.video_condition_ids)
    event_weight = output.program.rho.index_select(0, output.video_condition_ids)
    same_task = (
        (aligned - targets).float().square().mean(dim=(2, 3)) * event_weight.float()
    ).sum() / event_weight.sum().clamp_min(1e-6)
    probe_weight = output.probe_presence.mean(0)[..., None, None]
    probe = (
        (output.probe_process[0] - output.probe_process[1]).float().square()
        * probe_weight.float()
    ).sum() / probe_weight.sum().clamp_min(1e-6) / (
        output.local_process.shape[-2] * output.local_process.shape[-1]
    )
    return same_task, probe


def natural_program_loss(
    output: NaturalProgramOutput,
    batch: Any,
    *,
    weights: Mapping[str, float],
    robust_output: NaturalProgramOutput | None = None,
    negative_embeddings: torch.Tensor | None = None,
    contrastive_temperature: float = 0.1,
) -> NaturalProgramLoss:
    prediction = output.predictions
    action, action_temporal, progress, progress_temporal = (
        _temporal_prediction_losses(prediction, batch)
    )
    rising = F.binary_cross_entropy_with_logits(
        prediction.rising_logits.float(), batch.rising_targets.float()
    )
    contact_raw = F.binary_cross_entropy_with_logits(
        prediction.contact_logits.float(),
        batch.contact_targets.float(),
        reduction="none",
    )
    contact_mask = batch.contact_mask.to(contact_raw.dtype)
    contact = (contact_raw * contact_mask).sum() / contact_mask.sum().clamp_min(1.0)
    predicate_raw = F.binary_cross_entropy_with_logits(
        prediction.predicate_logits.float(),
        batch.predicate_targets[:, :, : prediction.predicate_logits.shape[-1]].float(),
        reduction="none",
    )
    predicate_mask = batch.predicate_mask[:, None].to(predicate_raw.dtype)
    predicate = (predicate_raw * predicate_mask).sum() / predicate_mask.sum().clamp_min(
        1.0
    ) / predicate_raw.shape[1]
    scene_targets = batch.predicate_targets[:, [0, -1], :]
    scene_raw = F.binary_cross_entropy_with_logits(
        prediction.scene_predicate_logits.float(),
        scene_targets[:, :, : prediction.scene_predicate_logits.shape[-1]].float(),
        reduction="none",
    )
    scene_mask = batch.predicate_mask[:, None].to(scene_raw.dtype)
    scene_relation = (scene_raw * scene_mask).sum() / scene_mask.sum().clamp_min(
        1.0
    ) / scene_raw.shape[1]

    same_task_event, probe_stability = _event_consistency_losses(output)

    if robust_output is None:
        robustness = output.program.p_process.new_zeros(())
    else:
        left = output.program
        right = robust_output.program
        robustness = (
            F.smooth_l1_loss(left.p_process.float(), right.p_process.float())
            + F.smooth_l1_loss(left.p_scene.float(), right.p_scene.float())
            + F.smooth_l1_loss(left.rho.float(), right.rho.float())
            + F.smooth_l1_loss(left.tau.float(), right.tau.float())
        )

    cross_task_contrast = _cross_task_contrast(
        output, negative_embeddings, contrastive_temperature
    )
    active_mass = output.program.rho.sum(-1)
    event_budget = (
        F.relu(2.0 - active_mass).square() + F.relu(active_mass - 6.0).square()
    ).mean()
    centers = output.program.tau[..., 0]
    tau_order = F.relu(centers[:, :-1] - centers[:, 1:]).mean()
    probe_delta = (
        output.probe_process[0].float() - output.probe_process[1].float()
    ).square()
    variance = output.local_sigma.float().square().clamp_min(1e-4)
    uncertainty_calibration = 0.5 * (
        probe_delta / variance + variance.log()
    ).mean()
    behavior_alignment = output.program.p_process.new_zeros(())

    terms = {
        "action": action,
        "action_temporal": action_temporal,
        "progress": progress,
        "progress_temporal": progress_temporal,
        "rising": rising,
        "contact": contact,
        "predicate": predicate,
        "scene_relation": scene_relation,
        "same_task_event": same_task_event,
        "probe_stability": probe_stability,
        "robustness": robustness,
        "cross_task_contrast": cross_task_contrast,
        "event_budget": event_budget,
        "tau_order": tau_order,
        "uncertainty_calibration": uncertainty_calibration,
        "behavior_alignment": behavior_alignment,
    }
    missing = set(terms) - set(weights) - {"behavior_alignment"}
    if missing:
        raise ValueError(f"missing Natural Program loss weights: {sorted(missing)}")
    total = sum(float(weights.get(name, 0.0)) * value for name, value in terms.items())
    return NaturalProgramLoss(total=total, **terms)
