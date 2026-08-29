"""Correct-functional-only training for the G3 routing-token boundary control."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch
import torch.distributed as dist

from ember.ecp.checkpoint import save_ecp_checkpoint
from ember.ecp.joint_program_primal.routing_control import (
    ROUTING_CONTROL_RUN_SCHEMA,
    ROUTING_CONTROL_STAGE,
    fixed_routing_program,
    prepare_routing_control_runtime,
)
from ember.ecp.joint_program_primal.train_step import (
    _clip_gradients,
    _gather_records,
    _rank_performance,
    _sum_gradients,
    _task_assignments,
    backward_functional_derivative,
    functional_loss_derivative,
    functional_panel_batch,
    joint_task_group,
    prepare_joint_condition,
)
from ember.ecp.native_materialization import (
    compose_rank12_plus_rank4,
    residual_lora_state,
)
from ember.pi05_source_checkpoint import write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import initialize_distributed


def _generated_rank16(runtime: Any, task_id: int, condition: Any):
    prepared, metrics = prepare_joint_condition(runtime, condition)
    program = fixed_routing_program(runtime, task_id)
    output = runtime.compiler.forward_compact(
        program, prepared.videos, s_ref=runtime.ranks.s_ref
    )
    residual = residual_lora_state(
        output.residual, runtime.rank4_contract, canonicalize=False
    )
    complete = compose_rank12_plus_rank4(
        carrier_state=runtime.ranks.carrier_rank12,
        residual_state=residual,
        rank16_contract=runtime.ranks.contract,
    )
    if output.video_weights.shape != (1,) or float(output.video_weights[0]) != 1.0:
        raise RuntimeError("routing-control adapter escaped K1 identity")
    return complete, output, metrics


def _run_task(
    runtime: Any, *, task_id: int, visit_index: int
) -> dict[str, Any]:
    tick = time.monotonic()
    batch, panel = functional_panel_batch(
        runtime, task_id=task_id, panel_name="a", visit_index=visit_index
    )
    views = []
    for condition in runtime.task_conditions[task_id].fit_views:
        complete, output, metrics = _generated_rank16(runtime, task_id, condition)
        loss, gradients = functional_loss_derivative(
            runtime,
            state=complete,
            batch=batch,
            policy_rng_seed=panel.policy_rng_seed,
        )
        backward_functional_derivative(complete, gradients, weight=1.0 / 12.0)
        views.append(
            {
                "video_demo": condition.video_demo,
                "sampled_frames": condition.sampled_frames,
                "functional_loss": loss,
                "solve_metrics": output.solve_metrics.detach().float().cpu().tolist(),
                "conditioning_metrics": output.conditioning_metrics.detach()
                .float()
                .cpu()
                .tolist(),
                "condition_metrics": metrics,
            }
        )
        del complete, output
    return {
        "authority_id": task_id,
        "role": runtime.task_conditions[task_id].fit_views[0].role,
        "panel": "a",
        "panel_visit": visit_index,
        "functional_policy_rng_seed": panel.policy_rng_seed,
        "action_demos": list(panel.action_demos),
        "action_frames": list(panel.action_frames),
        "mean_functional_loss": sum(row["functional_loss"] for row in views)
        / len(views),
        "views": views,
        "task_seconds": time.monotonic() - tick,
    }


def _scorer_gradient_probes(runtime: Any) -> dict[str, float]:
    scorer = runtime.compiler.primal_scorer
    probes = {
        "primal_input": scorer.input_primal_heads[0].weight.grad,
        "primal_output": scorer.output_primal_heads[0].weight.grad,
        "primal_program_context": scorer.program_context["q"][1].weight.grad,
        "primal_rank_context": scorer.rank_context["q"][1].weight.grad,
        "primal_event_score": scorer.event_score["q"].weight.grad,
        "owner_embedding": scorer.owner_embedding.grad,
        "rank_embedding": scorer.rank_embedding.grad,
    }
    result = {}
    for name, gradient in probes.items():
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError(
                f"routing-control {name} gradient is absent or non-finite"
            )
        result[name] = float(gradient.float().norm())
    if min(result.values()) <= 0:
        raise RuntimeError("routing-control scorer functional gradient is zero")
    return result


def run_routing_control_optimizer_step(runtime: Any) -> dict[str, Any]:
    """Run six role-balanced tasks and two correct fit videos per task."""

    group = joint_task_group(runtime, runtime.optimizer_steps)
    assignments = _task_assignments(runtime, group)
    visit_index = runtime.optimizer_steps % int(runtime.config["data"]["panel_visits"])
    if runtime.context.world_size > 1:
        dist.barrier()
    torch.cuda.synchronize(runtime.context.device)
    tick = time.monotonic()
    teacher_reads = runtime.native_teachers.tensor_reads
    runtime.optimizer.zero_grad(set_to_none=True)
    local_records = [
        _run_task(runtime, task_id=task, visit_index=visit_index)
        for task in assignments[runtime.context.rank]
    ]
    if runtime.native_teachers.tensor_reads != teacher_reads:
        raise RuntimeError("routing-control training read native teachers")
    if any(parameter.grad is not None for parameter in runtime.frozen_parameters):
        raise RuntimeError("routing-control frozen authority accumulated gradients")
    _sum_gradients(runtime)
    probes = _scorer_gradient_probes(runtime)
    gradient_norm = _clip_gradients(
        runtime.trainable_parameters,
        maximum=float(
            runtime.config["optimization"]["joint"]["optimizer"][
                "gradient_clip_norm"
            ]
        ),
    )
    runtime.optimizer.step()
    runtime.scheduler.step()
    runtime.optimizer_steps += 1
    torch.cuda.synchronize(runtime.context.device)
    local_seconds = time.monotonic() - tick
    records = sorted(
        _gather_records(local_records, runtime.context.world_size),
        key=lambda row: int(row["authority_id"]),
    )
    role_counts = {
        role: sum(row["role"] == role for row in records)
        for role in ("meta_fit", "target_fit")
    }
    if (
        len(records) != 6
        or role_counts != {"meta_fit": 3, "target_fit": 3}
        or {int(row["authority_id"]) for row in records} != set(group)
    ):
        raise RuntimeError("routing-control optimizer lost task-role weight")
    performance = _rank_performance(
        runtime,
        {
            "rank": runtime.context.rank,
            "seconds": local_seconds,
            "tasks": list(assignments[runtime.context.rank]),
            "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
                runtime.context.device
            ),
            "max_cuda_reserved_bytes": torch.cuda.max_memory_reserved(
                runtime.context.device
            ),
        },
    )
    return {
        "optimizer_step": runtime.optimizer_steps,
        "effective_optimizer_step": max(0, runtime.optimizer_steps - 10),
        "panel_visit": visit_index,
        "task_group": list(group),
        "role_counts": role_counts,
        "mean_functional_loss": sum(
            float(row["mean_functional_loss"]) for row in records
        )
        / len(records),
        "gradient_probe_norms": probes,
        "gradient_norm_before_clip": gradient_norm,
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "rank_assignments": [list(row) for row in assignments],
        "rank_performance": performance,
        "global_step_seconds": max(float(row["seconds"]) for row in performance),
        "conditions": records,
        "native_teacher_tensor_reads": runtime.native_teachers.tensor_reads,
        "fixed_routing_token_training_only": True,
        "deployment_candidate": False,
    }


def train_routing_control(args: argparse.Namespace) -> None:
    if args.phase != "joint":
        raise ValueError("routing control supports only the joint phase")
    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    runtime = None
    try:
        runtime = prepare_routing_control_runtime(args, context)
        started = time.monotonic()
        while runtime.optimizer_steps < runtime.stop_after_step:
            row = run_routing_control_optimizer_step(runtime)
            row["elapsed_seconds"] = time.monotonic() - started
            if context.is_main:
                append_jsonl(args.output_dir / "metrics.jsonl", row)
                runtime.metrics_rows += 1
                if runtime.optimizer_steps % args.log_every == 0:
                    console = {
                        name: row[name]
                        for name in (
                            "optimizer_step",
                            "effective_optimizer_step",
                            "global_step_seconds",
                            "mean_functional_loss",
                            "gradient_norm_before_clip",
                            "gradient_probe_norms",
                            "next_lr",
                            "task_group",
                            "role_counts",
                            "rank_assignments",
                            "native_teacher_tensor_reads",
                            "elapsed_seconds",
                        )
                    }
                    console["rank_performance"] = [
                        {
                            name: value[name]
                            for name in (
                                "rank",
                                "tasks",
                                "seconds",
                                "max_cuda_allocated_bytes",
                                "max_cuda_reserved_bytes",
                            )
                        }
                        for value in row["rank_performance"]
                    ]
                    print(json.dumps(console, sort_keys=True), flush=True)
            if runtime.optimizer_steps in runtime.checkpoint_steps:
                save_ecp_checkpoint(
                    output_dir=args.output_dir,
                    macro=runtime.optimizer_steps,
                    stage=ROUTING_CONTROL_STAGE,
                    context=context,
                    model=runtime.writer_state,
                    optimizer=runtime.optimizer,
                    scheduler=runtime.scheduler,
                    run_contract_schema=ROUTING_CONTROL_RUN_SCHEMA,
                    metrics_rows=runtime.metrics_rows,
                )
        if context.is_main:
            completion = {
                "stage": ROUTING_CONTROL_STAGE,
                "completed_optimizer_steps": runtime.optimizer_steps,
                "completed_effective_steps": max(0, runtime.optimizer_steps - 10),
                "deployment_candidate": False,
            }
            write_json_atomic(args.output_dir / "segment_completion.json", completion)
            if runtime.optimizer_steps == max(runtime.checkpoint_steps):
                write_json_atomic(args.output_dir / "completion.json", completion)
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
