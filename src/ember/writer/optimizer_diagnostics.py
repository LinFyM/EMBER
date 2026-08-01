"""Read-only Writer parameter, Adam moment, and realized-update diagnostics."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from ember.writer.model import WriterModelError

if TYPE_CHECKING:
    from ember.writer.training import WriterRuntime


def capture_optimizer_parameters(
    runtime: WriterRuntime,
) -> tuple[torch.Tensor, ...] | None:
    """Snapshot parameters only for configs that sealed realized-update metrics."""

    diagnostic = runtime.config["optimization"].get("optimizer_diagnostics")
    if diagnostic is None:
        return None
    if diagnostic != "per_owned_block_moment_parameter_and_actual_update_l2":
        raise WriterModelError("unsupported optimizer diagnostic contract")
    return tuple(
        item.parameter.detach().clone() for item in runtime.gradient_layout
    )


@torch.no_grad()
def optimizer_state_metrics(
    runtime: WriterRuntime,
    before: tuple[torch.Tensor, ...] | None,
) -> dict[str, Any] | None:
    """Measure actual Adam state and parameter displacement by owned block."""

    if before is None:
        return None
    if len(before) != len(runtime.gradient_layout):
        raise WriterModelError("optimizer diagnostic parameter layout changed")
    device = runtime.context.device
    blocks = tuple(sorted({item.block for item in runtime.gradient_layout}))
    fields = (
        "parameter_squared_l2",
        "update_squared_l2",
        "exp_avg_squared_l2",
        "sqrt_exp_avg_sq_squared_l2",
        "normalized_moment_squared_l2",
    )
    totals = {
        block: {
            field: torch.zeros((), dtype=torch.float32, device=device)
            for field in fields
        }
        for block in (*blocks, "all")
    }
    counts = {block: 0 for block in (*blocks, "all")}
    steps = {block: [] for block in (*blocks, "all")}
    eps = float(runtime.config["optimization"]["optimizer"]["eps"])
    for item, previous in zip(runtime.gradient_layout, before, strict=True):
        parameter = item.parameter.detach().float()
        update = parameter - previous.float()
        state = runtime.optimizer.state.get(item.parameter, {})
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        step = state.get("step")
        if (
            not isinstance(exp_avg, torch.Tensor)
            or not isinstance(exp_avg_sq, torch.Tensor)
            or exp_avg.shape != item.parameter.shape
            or exp_avg_sq.shape != item.parameter.shape
            or step is None
        ):
            raise WriterModelError("AdamW diagnostic state changed")
        exp_avg_float = exp_avg.detach().float()
        exp_avg_sq_float = exp_avg_sq.detach().float()
        normalized = exp_avg_float / (exp_avg_sq_float.sqrt() + eps)
        values = {
            "parameter_squared_l2": parameter.square().sum(),
            "update_squared_l2": update.square().sum(),
            "exp_avg_squared_l2": exp_avg_float.square().sum(),
            "sqrt_exp_avg_sq_squared_l2": exp_avg_sq_float.sum(),
            "normalized_moment_squared_l2": normalized.square().sum(),
        }
        step_value = int(step.item()) if isinstance(step, torch.Tensor) else int(step)
        for block in (item.block, "all"):
            for field, value in values.items():
                totals[block][field].add_(value)
            counts[block] += item.parameter.numel()
            steps[block].append(step_value)
    result = {}
    for block in (*blocks, "all"):
        if counts[block] <= 0 or not steps[block]:
            raise WriterModelError("optimizer diagnostic block is empty")
        values = {
            field.removesuffix("_squared_l2") + "_l2": math.sqrt(
                max(float(value.item()), 0.0)
            )
            for field, value in totals[block].items()
        }
        if any(not math.isfinite(value) for value in values.values()):
            raise WriterModelError("optimizer diagnostic became non-finite")
        result[block] = {
            "parameter_count": counts[block],
            "adam_step_min": min(steps[block]),
            "adam_step_max": max(steps[block]),
            **values,
            "update_to_parameter_l2": (
                values["update_l2"] / values["parameter_l2"]
                if values["parameter_l2"] > 0.0
                else 0.0
            ),
        }
    return {
        "contract": "per_owned_block_moment_parameter_and_actual_update_l2",
        "rank_local_under_exact_synchronized_optimizer": True,
        "blocks": result,
    }
