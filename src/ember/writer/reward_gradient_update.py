"""Aggregate reward gradients and measure multi-task coexistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import torch
import torch.distributed as dist

from ember.writer.as_step import assign_flat_gradient
from ember.writer.errors import WriterModelError

if TYPE_CHECKING:
    from ember.writer.reward_training import RewardRuntime


@dataclass(frozen=True)
class AppliedStep:
    active_tasks: int
    gradient_norm: float
    gradient_rms: float
    parameter_delta_rms: Mapping[str, float]
    gradient_coexistence: Mapping[str, Any]


def _trainable_named(
    runtime: RewardRuntime,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    return tuple(
        (name, value)
        for name, value in runtime.writer.named_parameters()
        if value.requires_grad
    )


def _coexistence(
    runtime: RewardRuntime,
    task_gradients: Mapping[int, torch.Tensor],
    shared_mean: torch.Tensor,
) -> dict[str, Any]:
    task_ids = tuple(task.global_task_id for task in runtime.tasks)
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    matrix = torch.zeros(
        len(task_ids), shared_mean.numel(), dtype=torch.float32, device=shared_mean.device
    )
    active = torch.zeros(len(task_ids), dtype=torch.long, device=shared_mean.device)
    for task_id, gradient in task_gradients.items():
        index = task_index[int(task_id)]
        matrix[index].copy_(gradient)
        active[index] = 1
    if runtime.context.world_size > 1:
        dist.all_reduce(matrix, op=dist.ReduceOp.SUM)
        dist.all_reduce(active, op=dist.ReduceOp.SUM)
    selected = active == 1
    rows = matrix[selected]
    selected_ids = [
        task_id
        for task_id, keep in zip(task_ids, selected.cpu().tolist(), strict=True)
        if keep
    ]
    if rows.shape[0] == 0 or bool((active > 1).any()):
        raise WriterModelError("factor-selector task gradients lost unique ownership")

    row_norms = torch.linalg.vector_norm(rows, dim=1)
    mean_norm = torch.linalg.vector_norm(shared_mean)
    dots = rows @ shared_mean
    cosines = dots / (row_norms * mean_norm).clamp_min(1e-30)
    unit = rows / row_norms[:, None].clamp_min(1e-30)
    pairwise = unit @ unit.T
    if rows.shape[0] > 1:
        offdiag = pairwise[
            ~torch.eye(rows.shape[0], dtype=torch.bool, device=rows.device)
        ]
        pairwise_values = torch.stack(
            (offdiag.mean(), offdiag.min(), offdiag.max())
        ).cpu().tolist()
        pairwise_summary = dict(
            zip(("mean", "minimum", "maximum"), pairwise_values, strict=True)
        )
    else:
        pairwise_summary = {"mean": 1.0, "minimum": 1.0, "maximum": 1.0}

    family_energy = {}
    for (name, _), item in zip(
        _trainable_named(runtime), runtime.gradient_layout, strict=True
    ):
        values = rows[:, item.start : item.stop]
        mean_values = shared_mean[item.start : item.stop]
        task_rms, shared_rms = torch.stack(
            (
                values.square().mean(dim=1).sqrt().mean(),
                mean_values.square().mean().sqrt(),
            )
        ).cpu().tolist()
        family = name.removeprefix("factor_commitment.selectors.")
        family_energy[family] = {
            "task_gradient_rms_mean": task_rms,
            "shared_mean_gradient_rms": shared_rms,
        }

    task_values = torch.stack((row_norms, dots, cosines), dim=1).cpu().tolist()
    coverage, cosine_mean, cosine_minimum = torch.stack(
        ((dots > 0).float().mean(), cosines.mean(), cosines.min())
    ).cpu().tolist()
    return {
        "active_task_ids": selected_ids,
        "shared_mean_descent_coverage": coverage,
        "task_to_shared_cosine_mean": cosine_mean,
        "task_to_shared_cosine_minimum": cosine_minimum,
        "pairwise_task_gradient_cosine": pairwise_summary,
        "per_task": [
            {
                "task_id": task_id,
                "gradient_norm": values[0],
                "dot_shared_mean": values[1],
                "cosine_shared_mean": values[2],
            }
            for task_id, values in zip(selected_ids, task_values, strict=True)
        ],
        "per_family": family_energy,
    }


def apply_reward_step(
    runtime: RewardRuntime,
    gradient_sum: torch.Tensor,
    local_active_tasks: int,
    task_gradients: Mapping[int, torch.Tensor],
) -> AppliedStep:
    active = torch.tensor(
        local_active_tasks, dtype=torch.long, device=runtime.context.device
    )
    if runtime.context.world_size > 1:
        dist.all_reduce(gradient_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(active, op=dist.ReduceOp.SUM)
    active_tasks = int(active)
    if active_tasks <= 0:
        raise WriterModelError("factor-selector cycle has no discordant success")
    gradient_sum.div_(active_tasks)
    if not bool(torch.isfinite(gradient_sum).all()) or not bool(
        torch.count_nonzero(gradient_sum)
    ):
        raise WriterModelError("factor-selector shared gradient is invalid")
    coexistence = _coexistence(runtime, task_gradients, gradient_sum)
    assign_flat_gradient(gradient_sum, runtime.gradient_layout)
    clip = float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"])
    grad_norm = torch.nn.utils.clip_grad_norm_(runtime.trainable_parameters, clip)
    named = _trainable_named(runtime)
    before = {name: value.detach().clone() for name, value in named}
    runtime.optimizer.step()
    delta = {
        name: float((value.detach() - before[name]).float().square().mean().sqrt())
        for name, value in named
    }
    return AppliedStep(
        active_tasks=active_tasks,
        gradient_norm=float(grad_norm),
        gradient_rms=float(gradient_sum.square().mean().sqrt()),
        parameter_delta_rms=delta,
        gradient_coexistence=coexistence,
    )
