"""All-task paired unit-gradient optimizer step for EBSRI S2 polish."""

from __future__ import annotations

import statistics
import time
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist


Condition = tuple[int, str]


def paired_conditions(optimizer_step: int, tasks: Sequence[int]) -> tuple[Condition, ...]:
    """Pair one alternating correct view with wrong-fit0 for every task."""

    if optimizer_step < 0 or not tasks or len(set(map(int, tasks))) != len(tasks):
        raise ValueError("S2 paired-condition schedule changed")
    correct = "correct_fit0" if optimizer_step % 2 == 0 else "correct_fit1"
    return tuple(
        condition
        for task in map(int, tasks)
        for condition in ((task, correct), (task, "wrong_fit0"))
    )


def balanced_condition_assignments(
    conditions: Sequence[Condition], costs: Mapping[Condition, int], world_size: int
) -> tuple[tuple[Condition, ...], ...]:
    """Greedily balance independent condition VJPs without changing their mass."""

    rows = tuple((int(task), str(arm)) for task, arm in conditions)
    if (
        not rows
        or len(set(rows)) != len(rows)
        or set(rows) != set(costs)
        or not 1 <= world_size <= 6
        or min(map(int, costs.values())) <= 0
    ):
        raise ValueError("S2 paired-condition assignment changed")
    assignments: list[list[Condition]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for condition in sorted(
        rows, key=lambda value: (-int(costs[value]), rows.index(value))
    ):
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        assignments[rank].append(condition)
        loads[rank] += int(costs[condition])
    return tuple(tuple(row) for row in assignments)


def _flat_gradient(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    missing = [index for index, value in enumerate(parameters) if value.grad is None]
    if missing:
        raise RuntimeError(f"S2 condition gradient is absent: {missing[:5]}")
    return torch.cat(
        tuple(value.grad.detach().float().reshape(-1) for value in parameters)
    ).contiguous()


def _install_gradient(
    parameters: Sequence[torch.nn.Parameter], gradient: torch.Tensor
) -> None:
    cursor = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = gradient[cursor : cursor + count].view_as(parameter).to(
            dtype=parameter.dtype
        ).clone()
        cursor += count
    if cursor != gradient.numel():
        raise RuntimeError("S2 flat-gradient layout changed")


def _condition_assignments(
    runtime: Any, conditions: Sequence[Condition]
) -> tuple[tuple[Condition, ...], ...]:
    from ember.ecp.joint_program_primal.bank_set_shared_training import _arm_spec

    costs = {
        condition: int(
            _arm_spec(runtime, condition[0], condition[1]).condition.sampled_frames
        )
        for condition in conditions
    }
    return balanced_condition_assignments(
        conditions, costs, int(runtime.context.world_size)
    )


def _synchronize(runtime: Any) -> None:
    if runtime.context.world_size > 1:
        dist.barrier()
    if runtime.context.device.type == "cuda":
        torch.cuda.synchronize(runtime.context.device)


def _local_condition_gradients(
    runtime: Any,
    assignments: Sequence[Condition],
    cursors: Mapping[int, int],
    epsilon: float,
) -> tuple[torch.Tensor, int, list[dict[str, Any]]]:
    from ember.ecp.joint_program_primal.bank_set_shared_training import (
        _functional_task_loss,
    )

    local_sum = None
    local_active = 0
    rows = []
    for task, arm in assignments:
        runtime.optimizer.zero_grad(set_to_none=True)
        row = _functional_task_loss(
            runtime,
            task,
            arm,
            task_cursor=cursors[task],
            task_weight=1.0,
        )
        gradient = _flat_gradient(runtime.trainable_parameters)
        norm = torch.linalg.vector_norm(gradient)
        finite = bool(torch.isfinite(norm))
        active = bool(row["gradient_active"])
        if not finite or (active and float(norm) <= epsilon):
            raise RuntimeError("S2 active condition gradient is invalid")
        if not active and float(norm) > epsilon:
            raise RuntimeError("S2 inactive wrong hinge retained a gradient")
        if active:
            unit = gradient / norm
            local_sum = unit if local_sum is None else local_sum.add(unit)
            local_active += 1
        row["raw_condition_gradient_norm"] = float(norm)
        row["unit_gradient_combiner_active"] = active
        rows.append(row)
        del gradient, norm

    if local_sum is None:
        reference = next(iter(runtime.trainable_parameters))
        local_sum = torch.zeros(
            sum(value.numel() for value in runtime.trainable_parameters),
            device=reference.device,
            dtype=torch.float32,
        )
    return local_sum, local_active, rows


def _reduce_condition_gradients(
    runtime: Any, local_sum: torch.Tensor, local_active: int, scheduled: int
) -> tuple[torch.Tensor, int]:
    active_count = torch.tensor(
        float(local_active), device=local_sum.device, dtype=torch.float32
    )
    if runtime.context.world_size > 1:
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(active_count, op=dist.ReduceOp.SUM)
    if not bool(torch.isfinite(local_sum).all()) or float(active_count) <= 0:
        raise RuntimeError("S2 unit-gradient reduction is invalid")
    return local_sum / scheduled, int(active_count.item())


def _apply_optimizer_gradient(runtime: Any, combined: torch.Tensor) -> float:
    runtime.optimizer.zero_grad(set_to_none=True)
    _install_gradient(runtime.trainable_parameters, combined)
    clip = float(
        runtime.config["optimization"]["joint"]["optimizer"]["gradient_clip_norm"]
    )
    norm = torch.nn.utils.clip_grad_norm_(runtime.trainable_parameters, clip)
    if not bool(torch.isfinite(norm)):
        raise RuntimeError("S2 combined unit gradient is non-finite")
    runtime.optimizer.step()
    runtime.scheduler.step()
    return float(norm)


def _gather_condition_rows(
    runtime: Any,
    local_rows: list[dict[str, Any]],
    conditions: Sequence[Condition],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if runtime.context.world_size > 1:
        gathered: list[Any] = [None] * runtime.context.world_size
        dist.all_gather_object(gathered, local_rows)
    else:
        gathered = [local_rows]
    records = [value for rows in gathered for value in rows]
    order = {condition: index for index, condition in enumerate(conditions)}
    records.sort(key=lambda value: order[(int(value["task"]), str(value["arm"]))])
    if [(int(row["task"]), str(row["arm"])) for row in records] != list(conditions):
        raise RuntimeError("S2 paired optimizer step lost a condition")
    active = [row for row in records if row["unit_gradient_combiner_active"]]
    for row in records:
        row["effective_unit_gradient_mass"] = (
            1.0 / len(conditions)
            if row["unit_gradient_combiner_active"]
            else 0.0
        )
    return records, active


def _step_metrics(
    runtime: Any,
    *,
    conditions: Sequence[Condition],
    records: Sequence[Mapping[str, Any]],
    active_count: int,
    cursors_before: Mapping[int, int],
    cursors_after: Mapping[int, int],
    gradient_norm: float,
    tick: float,
) -> dict[str, Any]:
    from ember.ecp.joint_program_primal.bank_set_shared_training import GRADIENT_TASKS

    active = [row for row in records if row["unit_gradient_combiner_active"]]
    if len(active) != active_count:
        raise RuntimeError("S2 ranks disagreed on active condition count")
    return {
        "optimizer_step": runtime.optimizer_steps,
        "task_group": list(GRADIENT_TASKS),
        "condition_schedule": [f"{task}:{arm}" for task, arm in conditions],
        "task_cursors_before": {
            str(task): cursors_before[task] for task in GRADIENT_TASKS
        },
        "task_cursors_after": {
            str(task): cursors_after[task] for task in GRADIENT_TASKS
        },
        "mean_training_objective": statistics.fmean(
            float(value["training_objective"]) for value in records
        ),
        "mean_generated_flow_loss": statistics.fmean(
            float(value["generated_flow_loss"]) for value in records
        ),
        "wrong_hinge_active_count": sum(
            int(value["gradient_active"])
            for value in records
            if value["arm"] == "wrong_fit0"
        ),
        "panel_a_functional_vjp_calls": len(records),
        "unit_gradient_active_condition_count": len(active),
        "raw_condition_gradient_norm": {
            "minimum": min(float(value["raw_condition_gradient_norm"]) for value in active),
            "median": statistics.median(
                float(value["raw_condition_gradient_norm"]) for value in active
            ),
            "maximum": max(float(value["raw_condition_gradient_norm"]) for value in active),
        },
        "tasks": records,
        "gradient_norm_before_clip": gradient_norm,
        "next_lr": float(runtime.scheduler.get_last_lr()[0]),
        "step_seconds": time.monotonic() - tick,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated())
        if runtime.context.device.type == "cuda"
        else 0,
        "world_size_invariant_task_weight": 1.0 / len(GRADIENT_TASKS),
        "world_size_invariant_scheduled_condition_weight": 1.0 / len(conditions),
        "role_weight": {"meta_fit": 0.5, "target_fit": 0.5},
        "gradient_combiner": (
            "scheduled_condition_unit_l2_mean_zero_for_inactive_no_mgda"
        ),
    }


def run_paired_unit_gradient_step(runtime: Any) -> dict[str, Any]:
    """Average the 16 scheduled unit-L2 Panel-A condition gradients.

    An inactive wrong hinge contributes an explicit zero at its pre-registered
    one-sixteenth mass.  This preserves task and role mass instead of giving
    tasks with two active conditions more weight than tasks with one.
    """

    from ember.ecp.joint_program_primal.bank_set_shared_training import (
        GRADIENT_TASKS,
        _advance_task_cursors,
        _validate_task_cursors,
    )

    cursors_before = _validate_task_cursors(runtime)
    conditions = paired_conditions(int(runtime.optimizer_steps), GRADIENT_TASKS)
    assignments = _condition_assignments(runtime, conditions)
    _synchronize(runtime)
    tick = time.monotonic()
    epsilon = float(
        runtime.config["optimization"]["direct_functional"][
            "condition_gradient_norm_epsilon"
        ]
    )
    local_sum, local_active, local_rows = _local_condition_gradients(
        runtime,
        assignments[runtime.context.rank],
        cursors_before,
        epsilon,
    )
    combined, active_count = _reduce_condition_gradients(
        runtime, local_sum, local_active, len(conditions)
    )
    gradient_norm = _apply_optimizer_gradient(runtime, combined)
    _advance_task_cursors(runtime, GRADIENT_TASKS)
    runtime.optimizer_steps += 1
    cursors_after = _validate_task_cursors(runtime)
    _synchronize(runtime)
    records, _ = _gather_condition_rows(runtime, local_rows, conditions)
    return _step_metrics(
        runtime,
        conditions=conditions,
        records=records,
        active_count=active_count,
        cursors_before=cursors_before,
        cursors_after=cursors_after,
        gradient_norm=gradient_norm,
        tick=tick,
    )
