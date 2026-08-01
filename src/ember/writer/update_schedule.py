"""Exposure-indexed optimizer/scheduler cadence for AS-Writer updates."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

from ember.writer.model import WriterModelError


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
