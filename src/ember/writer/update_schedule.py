"""Exposure-indexed optimizer/scheduler cadence for AS-Writer updates."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

from ember.writer.model import WriterModelError


def cycle_matched_weight_decay(
    logical_lr: float,
    reference_weight_decay: float,
    updates_per_cycle: int,
) -> float:
    """Match one raw AdamW decay across equal-LR cycle micro-updates."""

    if (
        logical_lr < 0.0
        or reference_weight_decay < 0.0
        or updates_per_cycle <= 0
        or logical_lr * reference_weight_decay >= 1.0
    ):
        raise WriterModelError("invalid cycle-matched AdamW decay")
    if updates_per_cycle == 1 or logical_lr == 0.0:
        return reference_weight_decay
    physical_lr = logical_lr / updates_per_cycle
    return -math.expm1(
        math.log1p(-logical_lr * reference_weight_decay)
        / updates_per_cycle
    ) / physical_lr


def prepare_optimizer_update(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the sealed logical-to-physical optimizer clock for one update."""

    normalization = config["optimization"].get("cycle_normalization")
    if normalization is None:
        applied_lrs = tuple(
            float(group["lr"]) for group in optimizer.param_groups
        )
        if not applied_lrs:
            raise WriterModelError("optimizer lost its parameter groups")
        return {
            "mode": "legacy_optimizer_clock",
            "logical_lr": applied_lrs[0],
            "applied_lr": applied_lrs[0],
            "lr_divisor": 1,
            "reference_weight_decay": float(
                optimizer.param_groups[0]["weight_decay"]
            ),
            "applied_weight_decay": float(
                optimizer.param_groups[0]["weight_decay"]
            ),
        }

    logical_lrs = tuple(float(value) for value in scheduler.get_last_lr())
    if len(logical_lrs) != len(optimizer.param_groups) or not logical_lrs:
        raise WriterModelError("optimizer and logical scheduler groups changed")

    mode = str(normalization.get("mode", ""))
    divisor = int(normalization.get("optimizer_updates_per_task_cycle", 0))
    reference_weight_decay = float(
        normalization.get("reference_weight_decay", -1.0)
    )
    if mode not in {
        "task_query_keyed_raw_reference",
        "cycle_normalized_randomized_group4",
    } or divisor != (1 if mode == "task_query_keyed_raw_reference" else 6):
        raise WriterModelError("unsupported optimizer cycle normalization")

    applied_lrs = []
    applied_weight_decays = []
    for group, logical_lr in zip(
        optimizer.param_groups, logical_lrs, strict=True
    ):
        applied_lr = logical_lr / divisor
        applied_weight_decay = cycle_matched_weight_decay(
            logical_lr,
            reference_weight_decay,
            divisor,
        )
        group["lr"] = applied_lr
        group["weight_decay"] = applied_weight_decay
        applied_lrs.append(applied_lr)
        applied_weight_decays.append(applied_weight_decay)
    if len(set(applied_lrs)) != 1 or len(set(applied_weight_decays)) != 1:
        raise WriterModelError("sealed Writer optimizer groups diverged")
    return {
        "mode": mode,
        "logical_lr": logical_lrs[0],
        "applied_lr": applied_lrs[0],
        "lr_divisor": divisor,
        "reference_weight_decay": reference_weight_decay,
        "applied_weight_decay": applied_weight_decays[0],
    }


def logical_task_cycle_steps(
    config: Mapping[str, Any], total_optimizer_updates: int
) -> int:
    """Return full-task exposures represented by an optimizer-update axis."""

    updates_per_cycle = int(
        config["conditioning_training"].get(
            "optimizer_updates_per_task_cycle", 1
        )
    )
    if (
        updates_per_cycle <= 0
        or total_optimizer_updates <= 0
        or total_optimizer_updates % updates_per_cycle
    ):
        raise WriterModelError(
            "AS-Writer optimizer updates do not form complete task cycles"
        )
    return total_optimizer_updates // updates_per_cycle


def build_exposure_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    logical_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build the unchanged cosine schedule on the task-exposure axis."""

    return CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=int(config["warmup_steps"]),
        num_decay_steps=int(config["decay_steps"]),
        peak_lr=float(config["peak_lr"]),
        decay_lr=float(config["decay_lr"]),
    ).build(optimizer, logical_steps)


def advance_scheduler_after_update(
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    completed_optimizer_updates: int,
    optimizer_updates_per_task_cycle: int,
) -> bool:
    """Advance once at a complete task-cycle boundary, never mid-cycle."""

    if completed_optimizer_updates <= 0 or optimizer_updates_per_task_cycle <= 0:
        raise WriterModelError("invalid exposure-matched scheduler cursor")
    should_advance = (
        completed_optimizer_updates % optimizer_updates_per_task_cycle == 0
    )
    if should_advance:
        scheduler.step()
    return should_advance
