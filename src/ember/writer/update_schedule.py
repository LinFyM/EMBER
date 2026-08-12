"""One-update-per-full24 optimizer and exposure schedule."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from ember.writer.errors import WriterModelError


def build_exposure_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    total_macros: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if config.get("kind") != "cosine_decay_with_warmup" or total_macros <= 0:
        raise WriterModelError("unsupported dynamic-K Writer scheduler")
    warmup = int(config["warmup_steps"])
    decay = int(config["decay_steps"])
    peak = float(config["peak_lr"])
    floor = float(config["decay_lr"])
    if warmup < 0 or decay <= warmup or peak <= 0 or not 0 <= floor <= peak:
        raise WriterModelError("invalid dynamic-K Writer cosine schedule")
    for group in optimizer.param_groups:
        group["lr"] = peak

    def factor(macro: int) -> float:
        if macro < warmup:
            return (macro + 1) / max(warmup, 1)
        progress = min(max((macro - warmup) / (decay - warmup), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (floor + (peak - floor) * cosine) / peak

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
