"""One task-equal distributed macro for native ECP Stage 0."""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from ember.ecp.stage0_data import pack_stage0_pair
from ember.ecp.stage0_objective import ecp_stage0_loss

if TYPE_CHECKING:
    from ember.ecp.stage0_training import ECPStage0Runtime


def _cross_rank_negatives(summary: torch.Tensor, rank: int, world_size: int) -> torch.Tensor | None:
    if world_size == 1:
        return None
    value = summary.detach().float().mean(0)
    gathered = [torch.empty_like(value) for _ in range(world_size)]
    dist.all_gather(gathered, value)
    return torch.stack([row for index, row in enumerate(gathered) if index != rank])


def _reduce_gradients(
    parameters: tuple[torch.nn.Parameter, ...], world_size: int
) -> None:
    if world_size == 1:
        return
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def _gather_records(
    local: list[dict[str, Any]], world_size: int
) -> list[dict[str, Any]]:
    if world_size == 1:
        return local
    shards: list[Any] = [None] * world_size
    dist.all_gather_object(shards, local)
    return [row for shard in shards for row in shard]


def _action_adapter_context(runtime: "ECPStage0Runtime") -> Any:
    if runtime.action_meta_lora is None:
        return nullcontext()
    expert = runtime.policy.model.paligemma_with_expert.gemma_expert.model
    return runtime.action_meta_lora.installed(expert)


def _program_metrics(output: Any) -> dict[str, float]:
    presence = output.presence.detach().float()
    return {
        "mean_presence": float(presence.mean()),
        "mean_active_events": float((presence > 0.5).float().sum(-1).mean()),
        "mean_presence_sum": float(presence.sum(-1).mean()),
    }


def _task_record(
    *, task_id: int, task: Any, pair: Any, output: Any, loss: Any
) -> dict[str, Any]:
    return {
        "authority_id": task_id,
        "domain": task.domain,
        "domain_task_id": task.domain_task_id,
        "total": float(loss.total.detach()),
        "frame_action_grounding": float(loss.frame_action_grounding.detach()),
        "event_action_reconstruction": float(
            loss.event_action_reconstruction.detach()
        ),
        "same_task_consistency": float(loss.same_task_consistency.detach()),
        "uncertainty_calibration": float(loss.uncertainty_calibration.detach()),
        "presence_consistency": float(loss.presence_consistency.detach()),
        "cross_task_contrast": float(loss.cross_task_contrast.detach()),
        "posterior_entropy": float(loss.posterior_entropy.detach()),
        "presence_sparsity": float(loss.presence_sparsity.detach()),
        **_program_metrics(output),
        **pair.metrics,
    }


def run_stage0_macro(
    runtime: "ECPStage0Runtime", macro: int, run_started: float
) -> dict[str, Any]:
    tick = time.monotonic()
    groups = runtime.schedule.assignments(macro, runtime.context.world_size)
    task_ids = groups[runtime.context.rank][: runtime.tasks_per_rank]
    global_task_count = runtime.tasks_per_rank * runtime.context.world_size
    runtime.optimizer.zero_grad(set_to_none=True)
    records = []
    for task_id in task_ids:
        pair = pack_stage0_pair(
            store=runtime.video_store,
            action_store=runtime.action_store,
            schedule=runtime.schedule,
            authority_id=task_id,
            task_visit=macro,
            device=runtime.context.device,
        )
        language_tokens, language_mask = runtime.language_tokens[task_id]
        # Keep the hooks installed through backward: PI0.5 recomputes checkpointed
        # Action Expert layers during the adapter-only calibration arm.
        with _action_adapter_context(runtime):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = runtime.model(
                    policy=runtime.policy,
                    frames=pair.frames,
                    video_offsets=pair.video_offsets,
                    frame_condition_ids=pair.frame_condition_ids,
                    language_tokens=language_tokens,
                    language_mask=language_mask,
                    action_meta_lora=runtime.action_meta_lora,
                    install_action_meta_lora=False,
                )
                negatives = _cross_rank_negatives(
                    output.program_summary,
                    runtime.context.rank,
                    runtime.context.world_size,
                )
                loss = ecp_stage0_loss(
                    output,
                    pair.frame_action_targets,
                    weights=runtime.config["objective"]["weights"],
                    negative_summaries=negatives,
                    contrastive_temperature=float(
                        runtime.config["objective"]["contrastive_temperature"]
                    ),
                )
            if not bool(torch.isfinite(loss.total)):
                raise RuntimeError(f"non-finite ECP Stage 0 loss at macro {macro}")
            (loss.total / global_task_count).backward()
        task = runtime.task_by_id[task_id]
        records.append(
            _task_record(
                task_id=task_id,
                task=task,
                pair=pair,
                output=output,
                loss=loss,
            )
        )
    if any(parameter.grad is not None for parameter in runtime.frozen_parameters):
        raise RuntimeError("frozen ECP Stage 0 parameters accumulated gradients")
    _reduce_gradients(runtime.trainable_parameters, runtime.context.world_size)
    clip = float(
        runtime.config["optimization"]["optimizer"]["gradient_clip_norm"]
    )
    grad_norm = torch.nn.utils.clip_grad_norm_(runtime.trainable_parameters, clip)
    if not bool(torch.isfinite(grad_norm)):
        raise RuntimeError(f"non-finite ECP Stage 0 gradient at macro {macro}")
    runtime.optimizer.step()
    runtime.scheduler.step()
    global_records = sorted(
        _gather_records(records, runtime.context.world_size),
        key=lambda row: int(row["authority_id"]),
    )
    means = {
        name: sum(float(row[name]) for row in global_records) / len(global_records)
        for name in (
            "total",
            "frame_action_grounding",
            "event_action_reconstruction",
            "same_task_consistency",
            "uncertainty_calibration",
            "presence_consistency",
            "cross_task_contrast",
            "posterior_entropy",
            "presence_sparsity",
            "mean_presence",
            "mean_active_events",
            "mean_presence_sum",
        )
    }
    return {
        "macro": macro + 1,
        "rank": runtime.context.rank,
        "global_task_count": len(global_records),
        "local_task_ids": list(task_ids),
        "global_means": means,
        "gradient_norm_before_clip": float(grad_norm),
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "macro_seconds": time.monotonic() - tick,
        "elapsed_seconds": time.monotonic() - run_started,
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
            runtime.context.device
        ),
        "conditions": global_records,
    }
