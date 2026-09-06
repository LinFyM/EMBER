"""Optimizer steps for shared positive-only Policy-Response Writer training."""

from __future__ import annotations

import statistics
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist

from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.ecp.policy_response_writer.shared import (
    SHARED_RUN_SCHEMA,
    SHARED_STAGE,
    SharedEvidenceCache,
    _gather_flat,
    _materialized_state,
    functional_objective,
    shared_training_stage,
)
from ember.ecp.policy_response_writer.shared_schedule import (
    VideoSplit,
    _split_ids,
    configured_task_group,
    scheduled_task_costs,
    task_occurrence_schedule,
    training_video_demos,
)
from ember.ecp.policy_response_writer.shared_execution import (
    cost_balanced_task_assignment,
)
from ember.ecp.policy_response_writer.training import (
    PolicyResponseRuntime,
    functional_panel_batch,
)
from ember.pi05_source_checkpoint import barrier
from ember.pi05_source_contract import append_jsonl, reconcile_metrics
from ember.writer.functional import (
    ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    LATIN_BETA_TIME_SAMPLING_SCHEME,
    functional_lora_loss_gradient,
    writer_chain_rule_surrogate,
)


def _training_query_batch(
    runtime: PolicyResponseRuntime, *, task: int, occurrence: int, rows: int
) -> tuple[dict[str, Any], Any, tuple[tuple[int, int], ...]]:
    """Sample only Panel-A rows; retain the original visit's policy noise seed."""
    protocol = runtime.config["optimization"]["shared"].get(
        "query_sampling", "fixed_panel_a_visit"
    )
    pairs = None
    if protocol == "uniform_panel_a_episode_frame":
        pool = sorted({
            demo for visit in runtime.panels[task].panel_a for demo in visit.action_demos
        })
        if runtime.query_dataset is None or not 0 < rows <= len(pool):
            raise ValueError("fresh training queries need an authorized episode pool")
        rng = np.random.default_rng(
            np.random.SeedSequence([int(runtime.config["optimization"]["seed"]), task, occurrence])
        )
        episodes = runtime.query_dataset.task_episode_rows[task]
        pairs = tuple(
            (int(demo), int(rng.integers(len(episodes[int(demo)]))))
            for demo in rng.choice(pool, size=rows, replace=False)
        )
    batch, panel = functional_panel_batch(
        runtime, task_id=task, panel_name="a", visit_index=occurrence,
        rows=rows, query_pairs=pairs,
    )
    actual = pairs if pairs is not None else tuple(zip(
        panel.action_demos[:rows], panel.action_frames[:rows], strict=True
    ))
    return batch, panel, actual


def _run_training_task(
    runtime: PolicyResponseRuntime,
    cache: SharedEvidenceCache,
    *,
    task: int,
    fit: Sequence[int],
    task_occurrence: int,
    task_count: int,
) -> dict[str, Any]:
    cell = runtime.config["optimization"]["shared"]
    training_stage = shared_training_stage(runtime)
    profile_timing = runtime.args.mode == "profile"
    phase_seconds: dict[str, float] = {}

    def start_phase() -> float:
        if not profile_timing:
            return 0.0
        torch.cuda.synchronize(runtime.context.device)
        return time.monotonic()

    def finish_phase(name: str, started: float) -> None:
        if not profile_timing:
            return
        torch.cuda.synchronize(runtime.context.device)
        phase_seconds[name] = time.monotonic() - started

    input_tick = start_phase()
    cardinalities = tuple(
        map(
            int,
            runtime.config["data"].get(
                "training_K", (runtime.config["data"]["initial_K"],)
            ),
        )
    )
    demos = training_video_demos(
        fit,
        task_occurrence=task_occurrence,
        task=task,
        cardinalities=cardinalities,
        seed=int(runtime.config["optimization"]["seed"]),
    )
    videos = tuple(
        cache.videos[(task, demo)].to(runtime.context.device) for demo in demos
    )
    visit_index = task_occurrence % len(runtime.panels[task].panel_a)
    rows = int(
        cell["functional_rows"]
        if runtime.args.mode == "formal"
        else cell["profile_functional_rows"]
    )
    batch, panel, query_pairs = _training_query_batch(
        runtime, task=task, occurrence=task_occurrence, rows=rows,
    )
    functional_microbatch = min(int(cell["functional_microbatch"]), rows)
    finish_phase("input_transfer_and_batch", input_tick)
    tick = time.monotonic()
    phase_tick = start_phase()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        leaf_state = _materialized_state(runtime, videos)
    finish_phase("writer_leaf_forward", phase_tick)
    phase_tick = start_phase()
    functional_loss, details, leaf_gradients = functional_lora_loss_gradient(
        runtime.policy,
        leaf_state,
        runtime.lora_contract,
        batch=batch,
        policy_rng_seed=panel.policy_rng_seed,
        policy_rng_device=runtime.context.device,
        flow_time_sampling_scheme=LATIN_BETA_TIME_SAMPLING_SCHEME,
        flow_noise_sampling_scheme=ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
        policy_microbatch_size=functional_microbatch,
        collect_policy_details=False,
    )
    if details or not bool(torch.isfinite(functional_loss)):
        raise RuntimeError("shared Writer functional derivative changed")
    finish_phase("policy_functional_vjp", phase_tick)
    objective = functional_objective(
        generated_loss=float(functional_loss),
        normalizer=cache.functional_normalizers[task],
        task_weight=1.0 / task_count,
    )
    del leaf_state
    phase_tick = start_phase()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        generated_state = _materialized_state(runtime, videos)
        surrogate = writer_chain_rule_surrogate(
            generated_state, leaf_gradients
        ) * float(objective["gradient_mass"])
    surrogate.backward()
    del generated_state, leaf_gradients, surrogate
    finish_phase("writer_chain_backward", phase_tick)

    row = {
        "task": task,
        "task_occurrence": task_occurrence,
        "role": runtime.panels[task].role,
        "video_demo": demos[0] if len(demos) == 1 else None,
        "video_demos": list(demos),
        "K": len(demos),
        "panel": "a",
        "panel_visit": visit_index,
        "functional_rows": rows,
        "functional_microbatch": functional_microbatch,
        "functional_policy_rng_seed": int(panel.policy_rng_seed),
        "query_sampling": cell.get("query_sampling", "fixed_panel_a_visit"),
        "action_query_pairs": [list(pair) for pair in query_pairs],
        "carrier_reference_panel_loss": float(panel.flow_loss),
        "functional_loss": float(functional_loss),
        **objective,
        "training_stage": training_stage,
        "objective": "correct_cross_episode_functional_positive_only",
        "task_weight": 1.0 / task_count,
        "task_seconds": time.monotonic() - tick,
        **({"phase_seconds": phase_seconds} if profile_timing else {}),
    }
    del videos, batch
    return row


def _sum_gradients(
    runtime: PolicyResponseRuntime, parameters: Sequence[torch.nn.Parameter]
) -> None:
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        elif not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError("shared Writer produced a non-finite gradient")
        if runtime.context.world_size > 1:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def _gradient_groups(runtime: PolicyResponseRuntime) -> dict[str, float]:
    prefixes = {
        "prefix": ("evidence.prefix",),
        "patch": ("evidence.prefix.patch_projection",),
        "language": ("evidence.prefix.language_projection",),
        "response": ("evidence.response",),
        "unified": ("factor_writer.blocks",),
        "factor_heads": ("factor_writer.factor_heads",),
        "factor_writer": ("factor_writer",),
    }
    result = {}
    for label, names in prefixes.items():
        squares = [
            parameter.grad.detach().float().square().sum()
            for name, parameter in runtime.writer.named_parameters()
            if name.startswith(names) and parameter.grad is not None
        ]
        result[label] = float(torch.stack(squares).sum().sqrt()) if squares else 0.0
    return result


def resume_cursor(
    runtime: PolicyResponseRuntime,
    *,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    stop: int,
) -> tuple[int, int]:
    if runtime.args.resume is None:
        return 0, 0
    start, expected_rows = load_ecp_checkpoint(
        checkpoint=runtime.args.resume,
        stage=SHARED_STAGE,
        context=runtime.context,
        model=runtime.writer,
        optimizer=optimizer,
        scheduler=scheduler,
        run_contract_schema=SHARED_RUN_SCHEMA,
    )
    if runtime.context.is_main:
        observed = reconcile_metrics(
            runtime.args.output_dir / "metrics.jsonl",
            start,
            expected_rows,
            cursor_key="optimizer_step",
        )
        if observed != expected_rows:
            raise RuntimeError("shared Writer metrics cursor changed")
    barrier(runtime.context)
    if not 0 <= start <= stop:
        raise ValueError("shared Writer resume cursor changed")
    return start, expected_rows


def _step_row(
    runtime: PolicyResponseRuntime,
    *,
    optimizer_step: int,
    group: Sequence[int],
    records: Sequence[Mapping[str, Any]],
    gradient_norm: torch.Tensor,
    gradient_groups: Mapping[str, float],
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    performance: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    learning_rates = scheduler.get_last_lr()
    return {
        "optimizer_step": optimizer_step,
        "effective_optimizer_step": max(
            0,
            optimizer_step
            - int(runtime.config["optimization"]["shared"]["warmup_updates"]),
        ),
        "task_group": list(group),
        "training_stage": shared_training_stage(runtime),
        "objective": "correct_cross_episode_functional_positive_only",
        "records": list(records),
        "mean_functional_loss": statistics.fmean(
            float(value["functional_loss"]) for value in records
        ),
        "gradient_norm_before_clip": float(gradient_norm),
        "gradient_groups": dict(gradient_groups),
        "next_lr": learning_rates[0],
        "step_seconds": max(float(value["seconds"]) for value in performance),
        "rank_performance": sorted(performance, key=lambda value: value["rank"]),
    }


def _optimizer_step(
    runtime: PolicyResponseRuntime,
    cache: SharedEvidenceCache,
    *,
    parameters: tuple[torch.nn.Parameter, ...],
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    execution_owners: Sequence[Sequence[int]],
    video_splits: Mapping[int, VideoSplit],
    group: Sequence[int],
    task_occurrences: Mapping[int, int],
    task_costs: Mapping[int, int],
    zero_step: int,
) -> dict[str, Any]:
    selected = set(group)
    eligibility = {
        task: tuple(
            rank
            for rank, row in enumerate(execution_owners)
            if task in set(map(int, row))
        )
        for task in group
    }
    assignment = cost_balanced_task_assignment(
        group,
        task_costs,
        eligibility,
        world_size=runtime.context.world_size,
    )
    local_tasks = assignment[runtime.context.rank]
    barrier(runtime.context)
    torch.cuda.synchronize(runtime.context.device)
    torch.cuda.reset_peak_memory_stats(runtime.context.device)
    tick = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    local_records = [
        _run_training_task(
            runtime,
            cache,
            task=task,
            fit=video_splits[task][0],
            task_occurrence=int(task_occurrences[task]),
            task_count=len(group),
        )
        for task in local_tasks
    ]
    if any(value.grad is not None for value in runtime.policy.parameters()) or any(
        value.grad is not None for value in runtime.stage0.parameters()
    ):
        raise RuntimeError("shared Writer crossed a frozen authority")
    _sum_gradients(runtime, parameters)
    gradient_groups = _gradient_groups(runtime)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        float(runtime.config["optimization"]["shared"]["gradient_clip_norm"]),
    )
    if not bool(torch.isfinite(gradient_norm)) or float(gradient_norm) <= 0:
        raise RuntimeError("shared Writer gradient norm is invalid")
    optimizer.step()
    scheduler.step()
    torch.cuda.synchronize(runtime.context.device)
    allocated = int(torch.cuda.max_memory_allocated(runtime.context.device))
    reserved = int(torch.cuda.max_memory_reserved(runtime.context.device))
    records = sorted(_gather_flat(runtime, local_records), key=lambda row: row["task"])
    if len(records) != len(group) or {row["task"] for row in records} != selected:
        raise RuntimeError("shared Writer optimizer update lost a task")
    performance = _gather_flat(
        runtime,
        (
            {
                "rank": runtime.context.rank,
                "tasks": list(local_tasks),
                "predicted_task_cost": sum(task_costs[task] for task in local_tasks),
                "seconds": time.monotonic() - tick,
                "max_cuda_allocated_bytes": allocated,
                "max_cuda_reserved_bytes": reserved,
            },
        ),
    )
    return _step_row(
        runtime,
        optimizer_step=zero_step + 1,
        group=group,
        records=records,
        gradient_norm=gradient_norm,
        gradient_groups=gradient_groups,
        scheduler=scheduler,
        performance=performance,
    )


def train(
    runtime: PolicyResponseRuntime,
    cache: SharedEvidenceCache,
    *,
    parameters: tuple[torch.nn.Parameter, ...],
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    task_owners: Sequence[Sequence[int]],
    execution_owners: Sequence[Sequence[int]],
    video_splits: Mapping[int, VideoSplit],
    start_step: int,
    stop: int,
    metrics_rows: int,
    checkpoint_steps: set[int],
) -> dict[str, Any]:
    meta, target, _ = _split_ids(runtime)
    profile_tasks = (
        (int(runtime.args.task),)
        if runtime.args.mode == "profile" and runtime.args.task is not None
        else ()
    )
    if profile_tasks and not set(profile_tasks) <= set((*meta, *target)):
        raise ValueError("shared profile task is not a gradient task")
    groups = tuple(
        profile_tasks
        or configured_task_group(runtime, step, task_owners=task_owners)
        for step in range(stop)
    )
    occurrences = task_occurrence_schedule(groups)
    curve = []
    peaks = {"allocated": 0, "reserved": 0}
    started = time.monotonic()
    for zero_step in range(start_step, stop):
        global_tasks = int(
            runtime.config["optimization"]["shared"]["global_tasks_per_update"]
        )
        group = groups[zero_step]
        if len(group) != (1 if profile_tasks else global_tasks):
            raise RuntimeError("shared Writer configured task count changed")
        occurrence_indices = occurrences[zero_step]
        task_costs = scheduled_task_costs(
            runtime,
            video_splits,
            group,
            task_occurrences=occurrence_indices,
        )
        row = _optimizer_step(
            runtime,
            cache,
            parameters=parameters,
            optimizer=optimizer,
            scheduler=scheduler,
            execution_owners=execution_owners,
            video_splits=video_splits,
            group=group,
            task_occurrences=occurrence_indices,
            task_costs=task_costs,
            zero_step=zero_step,
        )
        for performance in row["rank_performance"]:
            peaks["allocated"] = max(
                peaks["allocated"], int(performance["max_cuda_allocated_bytes"])
            )
            peaks["reserved"] = max(
                peaks["reserved"], int(performance["max_cuda_reserved_bytes"])
            )
        optimizer_step = zero_step + 1
        if runtime.context.is_main:
            append_jsonl(runtime.args.output_dir / "metrics.jsonl", row)
            metrics_rows += 1
            if (
                optimizer_step == 1
                or optimizer_step % 10 == 0
                or optimizer_step == stop
            ):
                curve.append(row)
                print(row, flush=True)
        if runtime.args.mode == "formal" and optimizer_step in checkpoint_steps:
            save_ecp_checkpoint(
                output_dir=runtime.args.output_dir,
                macro=optimizer_step,
                stage=SHARED_STAGE,
                context=runtime.context,
                model=runtime.writer,
                optimizer=optimizer,
                scheduler=scheduler,
                run_contract_schema=SHARED_RUN_SCHEMA,
                metrics_rows=metrics_rows,
            )
    return {
        "curve": curve,
        "metrics_rows": metrics_rows,
        "train_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": peaks["allocated"],
        "max_cuda_reserved_bytes": peaks["reserved"],
    }
