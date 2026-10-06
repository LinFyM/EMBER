"""Once-per-condition physical supervision with clip-fixed anonymous matching.

GT names, counts, masks, and the assignment remain loss-only information. Empty
anonymous slots supervise presence alone; missing fields never become targets.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from .representation import ENTITIES, check_physical


def smooth_l1(difference: torch.Tensor) -> torch.Tensor:
    absolute = difference.abs()
    return torch.where(absolute < 1, 0.5 * difference.square(), absolute - 0.5)


def _entity_mask(gt: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    return gt[f"{name}_mask"].bool() & gt["valid"][:, None].bool() & (gt["presence"] > 0.5)


def _mean_cost(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask[:, None], error, torch.zeros_like(error)).sum(0) / (
        mask.sum(0).clamp_min(1)[None])


@torch.no_grad()
def clip_matching(pred: Mapping[str, torch.Tensor], gt: Mapping[str, torch.Tensor]
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """One Hungarian solve for the entire video, with the hand fixed at zero."""
    existing = (gt["presence"][:, 1:] > 0.5) & gt["valid"][:, None].bool()
    targets = existing.any(0).nonzero().flatten() + 1
    cost = pred["p"].new_zeros((ENTITIES - 1, len(targets)), dtype=torch.float32)
    for name in ("p", "R", "semantic"):
        mask = _entity_mask(gt, name)[:, targets]
        truth = gt[name][:, targets].float()
        expanded = mask.reshape(*mask.shape, *([1] * (truth.ndim - mask.ndim)))
        truth = torch.where(expanded, truth, torch.zeros_like(truth))
        predicted = pred[name][:, 1:].float()
        if name == "p":
            error = smooth_l1((predicted[:, :, None] - truth[:, None]) / 0.25).mean(-1)
        elif name == "R":
            error = 0.25 * (predicted[:, :, None] - truth[:, None]).square().sum((-1, -2))
        else:
            error = 1 - (F.normalize(predicted, dim=-1, eps=1e-6)[:, :, None]
                         * F.normalize(truth, dim=-1, eps=1e-6)[:, None]).sum(-1)
        cost = cost + _mean_cost(error, mask)
    rows, columns = linear_sum_assignment(cost.cpu().numpy())
    predicted = torch.as_tensor(rows, device=pred["p"].device, dtype=torch.long) + 1
    matched = targets[torch.as_tensor(columns, device=targets.device, dtype=torch.long)]
    hand = predicted.new_zeros(1)
    return torch.cat((hand, predicted)), torch.cat((hand, matched))


def _mean_if_present(value: torch.Tensor) -> torch.Tensor | None:
    return value.mean() if value.numel() else None


def _field_samples(pred, gt, pred_indices, gt_indices, name):
    mask = _entity_mask(gt, name)[:, gt_indices]
    return pred[name][:, pred_indices].float()[mask], gt[name][:, gt_indices].float()[mask]


def _presence_loss(pred, gt, pred_indices, gt_indices) -> torch.Tensor:
    target = torch.zeros_like(pred["presence"], dtype=torch.float32)
    target[:, pred_indices] = gt["presence"][:, gt_indices].float()
    mask = gt["valid"][:, None].bool().expand_as(target)
    probability = pred["presence"].float().clamp(1e-6, 1 - 1e-6)
    # Manual probability BCE is safe under CUDA autocast; no logit/hidden bypass.
    error = -(target * probability.log() + (1 - target) * torch.log1p(-probability))
    positive = _mean_if_present(error[mask & (target > 0.5)])
    negative = _mean_if_present(error[mask & (target <= 0.5)])
    available = [value for value in (positive, negative) if value is not None]
    return torch.stack(available).mean()


def _joint_loss(pred, gt, pred_indices, gt_indices) -> torch.Tensor | None:
    predicted, truth = _field_samples(pred, gt, pred_indices, gt_indices, "joint_type")
    ce = _mean_if_present(-(truth * predicted.clamp_min(1e-6).log()).sum(-1))
    scale = pred["joint_q"].new_tensor((0.25, math.pi), dtype=torch.float32)
    mask = gt["joint_q_mask"][:, gt_indices].bool()
    mask = mask & gt["valid"][:, None, None].bool()
    mask = mask & (gt["presence"][:, gt_indices, None] > 0.5)
    # Select before arithmetic, so unknown/terminal geometry cannot leak NaNs.
    predicted_q = (pred["joint_q"][:, pred_indices].float() / scale)[mask]
    truth_q = (gt["joint_q"][:, gt_indices].float() / scale)[mask]
    q = _mean_if_present(smooth_l1(predicted_q - truth_q))
    available = [value for value in (ce, q) if value is not None]
    return torch.stack(available).mean() if available else None


def relation_loss(pred: Mapping[str, torch.Tensor], gt: Mapping[str, torch.Tensor]
                  ) -> tuple[torch.Tensor, dict]:
    """Six equally weighted available groups; caller applies .1 once per clip.

    GT has valid[N], entity masks p/R/semantic/joint_type[N,33],
    joint_q_mask[N,33,2], and hand_q_mask[N,2]. joint_type is one-hot/probability
    in none/slide/hinge order. Metrics are detached JSON-compatible values.
    """
    shape = check_physical(pred)
    if len(shape) != 1 or check_physical(gt) != shape:
        raise ValueError("relation supervision requires one paired physical video")
    if gt["valid"].shape != shape or not bool(gt["valid"].any()):
        raise ValueError("relation supervision needs at least one valid geometry frame")
    pred_indices, gt_indices = clip_matching(pred, gt)
    position, position_gt = _field_samples(pred, gt, pred_indices, gt_indices, "p")
    rotation, rotation_gt = _field_samples(pred, gt, pred_indices, gt_indices, "R")
    semantic, semantic_gt = _field_samples(pred, gt, pred_indices, gt_indices, "semantic")
    position_loss = _mean_if_present(smooth_l1((position - position_gt) / 0.25).mean(-1))
    rotation_loss = _mean_if_present(0.25 * (rotation - rotation_gt).square().sum((-1, -2)))
    semantic_loss = _mean_if_present(1 - (F.normalize(semantic, dim=-1, eps=1e-6)
                                        * F.normalize(semantic_gt, dim=-1, eps=1e-6)).sum(-1)
                                    .clamp(-1, 1))
    hand_mask = gt["hand_q_mask"].bool() & gt["valid"][:, None].bool()
    hand_mask = hand_mask & (gt["presence"][:, 0, None] > 0.5)
    hand_error = (pred["hand_q"].float()[hand_mask] - gt["hand_q"].float()[hand_mask]) / 0.04
    groups = {"p": position_loss, "R": rotation_loss, "semantic": semantic_loss,
              "presence": _presence_loss(pred, gt, pred_indices, gt_indices),
              "joint": _joint_loss(pred, gt, pred_indices, gt_indices),
              "hand_q": _mean_if_present(smooth_l1(hand_error))}
    available = {name: value for name, value in groups.items() if value is not None}
    loss = torch.stack(list(available.values())).mean()
    numbers = torch.stack([loss.detach(), *(value.detach() for value in available.values())])
    numbers = numbers.cpu().tolist()
    assignment = pred_indices.new_full((ENTITIES,), -1)
    assignment[pred_indices] = gt_indices
    metrics = {name: None for name in groups}
    metrics.update(zip(available, numbers[1:]))
    metrics.update(loss=numbers[0], valid_groups=len(available),
                   assignment=assignment.cpu().tolist())
    return loss, metrics
