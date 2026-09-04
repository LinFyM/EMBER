"""Optimizer steps for shared positive-only Policy-Response Writer training."""

from __future__ import annotations

import math
import statistics
import time
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.ecp.policy_response_writer.shared import (
    SHARED_RUN_SCHEMA,
    SHARED_STAGE,
    SharedEvidenceCache,
    _gather_flat,
    _materialized_state,
    causal_pair,
    functional_objective,
)
from ember.ecp.policy_response_writer.shared_schedule import (
    VideoSplit,
    _split_ids,
    configured_task_group,
    scheduled_task_costs,
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


def _run_training_task(
    runtime: PolicyResponseRuntime,
    cache: SharedEvidenceCache,
    *,
    task: int,
    fit: Sequence[int],
    optimizer_step: int,
    task_count: int,
) -> dict[str, Any]:
    cell = runtime.config["optimization"]["shared"]
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
        optimizer_step=optimizer_step,
        task=task,
        cardinalities=cardinalities,
        seed=int(runtime.config["optimization"]["seed"]),
    )
    videos = tuple(
        cache.videos[(task, demo)].to(runtime.context.device) for demo in demos
    )
    visit_index = optimizer_step % len(runtime.panels[task].panel_a)
    rows = int(
        cell["functional_rows"]
        if runtime.args.mode == "formal"
        else cell["profile_functional_rows"]
    )
    batch, panel = functional_panel_batch(
        runtime,
        task_id=task,
        panel_name="a",
        visit_index=visit_index,
        rows=rows,
    )
    functional_microbatch = min(int(cell["functional_microbatch"]), rows)
    finish_phase("input_transfer_and_batch", input_tick)
    tick = time.monotonic()
    phase_tick = start_phase()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        leaf_state = _materialized_state(runtime, videos, canonicalize=False)
    finish_phase("writer_leaf_forward", phase_tick)
    phase_tick = start_phase()
    functional_loss, details, leaf_gradients = functional_lora_loss_gradient(
        runtime.policy,
        leaf_state,
        runtime.ranks.contract,
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
        carrier_loss=float(panel.flow_loss),
        normalizer=cache.functional_normalizers[task],
        task_weight=1.0 / task_count,
        preservation_weight=float(cell["preservation_weight"]),
        preservation_epsilon=float(cell["preservation_epsilon"]),
    )
    del leaf_state
    phase_tick = start_phase()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        generated_state = _materialized_state(runtime, videos, canonicalize=False)
        surrogate = writer_chain_rule_surrogate(
            generated_state, leaf_gradients
        ) * float(objective["gradient_mass"])
    surrogate.backward()
    del generated_state, leaf_gradients, surrogate
    finish_phase("writer_chain_backward", phase_tick)

    causal_pairs = tuple(
        causal_pair(
            video.frame_count,
            int(runtime.config["model"]["event_slots"]),
            optimizer_step=optimizer_step,
            task=task,
            demo=demo,
        )
        for video, demo in zip(videos, demos, strict=True)
    )
    cutoffs = tuple((cutoff,) for cutoff, _ in causal_pairs)
    future_offsets = tuple(offset for _, offset in causal_pairs)
    phase_tick = start_phase()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        process_loss = runtime.writer.causal_prediction_loss(
            videos,
            cutoffs=cutoffs,
            future_offsets=future_offsets,
            representation=runtime.args.representation,
        )
        weighted_process = (
            process_loss.float()
            / cache.process_normalizers[task]
            * float(cell["process_weight"])
            / task_count
        )
    weighted_process.backward()
    finish_phase("causal_process_backward", phase_tick)
    row = {
        "task": task,
        "role": runtime.panels[task].role,
        "video_demo": demos[0] if len(demos) == 1 else None,
        "video_demos": list(demos),
        "K": len(demos),
        "panel": "a",
        "panel_visit": visit_index,
        "functional_rows": rows,
        "functional_microbatch": functional_microbatch,
        "functional_policy_rng_seed": int(panel.policy_rng_seed),
        "carrier_loss": float(panel.flow_loss),
        "functional_loss": float(functional_loss),
        "benefit_over_carrier": float(panel.flow_loss) - float(functional_loss),
        **objective,
        "process_loss": float(process_loss.detach()),
        "process_normalized": float(process_loss.detach())
        / cache.process_normalizers[task],
        "causal_cutoffs": [row[0] for row in cutoffs],
        "causal_future_offsets": list(future_offsets),
        "task_weight": 1.0 / task_count,
        "task_seconds": time.monotonic() - tick,
        **({"phase_seconds": phase_seconds} if profile_timing else {}),
    }
    del videos, batch, process_loss, weighted_process
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
        "frame": (
            "process.patch_projection",
            "process.relations",
            "process.frame_blocks",
        ),
        "event": ("process.events",),
        "process_prediction": (
            "process.prediction_probe",
            "process.prediction_horizon",
            "process.prediction_head",
        ),
        "composer": ("composer",),
        "composer_relation": ("composer.relation_embedding",),
        "composer_scale": ("composer.scale_head",),
    }
    result = {}
    for label, names in prefixes.items():
        squares = [
            parameter.grad.detach().float().square().sum()
            for name, parameter in runtime.writer.named_parameters()
            if name.startswith(names) and parameter.grad is not None
        ]
        result[label] = float(torch.stack(squares).sum().sqrt()) if squares else 0.0
    direction_squares = [
        parameter.grad.detach().float().square().sum()
        for name, parameter in runtime.writer.named_parameters()
        if name.startswith("composer.")
        and not name.startswith("composer.scale_head.")
        and parameter.grad is not None
    ]
    result["composer_direction"] = (
        float(torch.stack(direction_squares).sum().sqrt())
        if direction_squares
        else 0.0
    )
    return result


def _clip_scale_and_direction_gradients(
    *,
    parameters: Sequence[torch.nn.Parameter],
    scale_parameters: Sequence[torch.nn.Parameter],
    max_norm: float,
) -> torch.Tensor:
    """Clip scale and direction separately while reporting their joint norm."""

    scale_parameters = tuple(scale_parameters)
    parameter_ids = {id(parameter) for parameter in parameters}
    scale_ids = {id(parameter) for parameter in scale_parameters}
    direction_parameters = tuple(
        parameter for parameter in parameters if id(parameter) not in scale_ids
    )
    if (
        not direction_parameters
        or not scale_parameters
        or len(scale_ids) != len(scale_parameters)
        or not scale_ids <= parameter_ids
        or not math.isfinite(max_norm)
        or max_norm <= 0.0
    ):
        raise ValueError("shared Writer gradient group ownership changed")
    direction_norm = torch.nn.utils.clip_grad_norm_(direction_parameters, max_norm)
    scale_norm = torch.nn.utils.clip_grad_norm_(scale_parameters, max_norm)
    return (
        torch.stack((direction_norm.float(), scale_norm.float())).square().sum().sqrt()
    )


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
    return {
        "optimizer_step": optimizer_step,
        "effective_optimizer_step": max(
            0,
            optimizer_step
            - int(runtime.config["optimization"]["shared"]["warmup_updates"]),
        ),
        "task_group": list(group),
        "records": list(records),
        "mean_functional_loss": statistics.fmean(
            float(value["functional_loss"]) for value in records
        ),
        "mean_benefit_over_carrier": statistics.fmean(
            float(value["benefit_over_carrier"]) for value in records
        ),
        "mean_process_normalized": statistics.fmean(
            float(value["process_normalized"]) for value in records
        ),
        "preservation_active_tasks": sum(
            bool(value["preservation_active"]) for value in records
        ),
        "gradient_norm_before_clip": float(gradient_norm),
        "gradient_groups": dict(gradient_groups),
        "next_lr": scheduler.get_last_lr()[0],
        "next_process_prediction_lr": scheduler.get_last_lr()[1],
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
            optimizer_step=zero_step,
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
    gradient_norm = _clip_scale_and_direction_gradients(
        parameters=parameters,
        scale_parameters=tuple(runtime.writer.composer.scale_head.parameters()),
        max_norm=float(runtime.config["optimization"]["shared"]["gradient_clip_norm"]),
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
    curve = []
    peaks = {"allocated": 0, "reserved": 0}
    started = time.monotonic()
    for zero_step in range(start_step, stop):
        global_tasks = int(
            runtime.config["optimization"]["shared"]["global_tasks_per_update"]
        )
        group = profile_tasks or configured_task_group(
            runtime,
            zero_step,
            task_owners=task_owners,
        )
        if len(group) != (1 if profile_tasks else global_tasks):
            raise RuntimeError("shared Writer configured task count changed")
        task_costs = scheduled_task_costs(
            runtime,
            video_splits,
            group,
            optimizer_step=zero_step,
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
