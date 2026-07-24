"""One optimizer step for canonical PI05 Action-Forecast AS-Writer training."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Mapping

import torch

from ember.pi05_source_setup import reduce_max, reduce_mean
from ember.writer.functional import functional_lora_loss_gradient
from ember.writer.model import WriterModelError

if TYPE_CHECKING:
    from ember.writer.training import WriterRuntime


def _batch_task_id(batch: Mapping[str, Any]) -> int:
    values = batch.get("task_id")
    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise WriterModelError("AS-Writer action batch lost task identity")
    unique = values.unique()
    if unique.numel() != 1:
        raise WriterModelError("one AS-Writer rank received multiple tasks")
    return int(unique.item())


def _pack_raw_condition(
    runtime: WriterRuntime,
    *,
    task_id: int,
    demo_index: int,
    data_step: int,
) -> tuple[
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    dict[str, int],
]:
    video = runtime.video_store.load(task_id, demo_index)
    tokens, mask, state_positions = runtime.language_tokens[task_id]
    frames = torch.from_numpy(video.frames).to(
        runtime.context.device,
        non_blocking=True,
    )
    indices = torch.from_numpy(video.frame_indices).to(
        runtime.context.device,
        non_blocking=True,
    )
    offsets = torch.tensor(
        [0, int(video.frames.shape[0])],
        dtype=torch.long,
        device=runtime.context.device,
    )
    global_visit = data_step * runtime.context.world_size + runtime.context.rank
    flow_noise = runtime.flow_noise_schedule.noise_for_visit(
        global_visit,
        device=runtime.context.device,
    )[None]
    return (
        frames,
        indices,
        offsets,
        tokens,
        mask,
        state_positions,
        flow_noise,
    ), {
        "teacher_video_raw_frames": video.raw_frame_count,
        "teacher_video_sampled_frames": int(video.frames.shape[0]),
        "writer_flow_noise_global_visit": global_visit,
    }


def _differentiate_condition(
    runtime: WriterRuntime,
    packed: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    policy_batch: Mapping[str, Any],
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated = runtime.wrapped_writer(
            *packed,
            policy=runtime.policy,
        )
        loss, detail, gradients = functional_lora_loss_gradient(
            runtime.policy,
            generated,
            runtime.lora_contract,
            batch=policy_batch,
        )
    names = tuple(generated)
    torch.autograd.backward(
        tuple(generated[name] for name in names),
        tuple(gradients[name] for name in names),
    )
    return loss, detail


def _step_metrics(
    runtime: WriterRuntime,
    *,
    step: int,
    started: float,
    tick: float,
    task_id: int,
    demo_index: int,
    observed_batch: int,
    loss: torch.Tensor,
    detail: Mapping[str, Any],
    data_seconds: float,
    video_metrics: Mapping[str, int],
    grad_norm: torch.Tensor,
    applied_lr: float,
) -> dict[str, Any]:
    completed = step + 1
    step_seconds = reduce_max(time.monotonic() - tick, runtime.context)
    global_queries_this_step = observed_batch * runtime.context.world_size
    return {
        "optimizer_step": completed,
        "conditioning_mode": "normal",
        "writer_language_condition": "correct_task_language",
        "policy_language_condition": "correct_action_query_task_language",
        "mean_functional_action_loss": reduce_mean(float(loss), runtime.context),
        "gradient_norm_before_clip_max": reduce_max(
            float(grad_norm),
            runtime.context,
        ),
        "applied_lr": applied_lr,
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "global_unique_action_queries": (
            completed * runtime.batch_size * runtime.context.world_size
        ),
        "global_policy_samples": (
            completed * runtime.batch_size * runtime.context.world_size
        ),
        "global_writer_video_conditions": completed * runtime.context.world_size,
        "global_unique_action_queries_this_step": global_queries_this_step,
        "global_policy_samples_this_step": global_queries_this_step,
        "global_writer_video_conditions_this_step": runtime.context.world_size,
        "rank0_task_id": task_id,
        "rank0_teacher_demo_index": demo_index,
        "rank0_condition": {
            "task_id": task_id,
            "teacher_demo_index": demo_index,
            **dict(video_metrics),
        },
        "rank0_policy_loss_detail": detail.get("loss", float(loss)),
        "action_query_batch_size_per_rank": runtime.batch_size,
        "policy_forward_calls_this_step": 1,
        "data_seconds_max": reduce_max(data_seconds, runtime.context),
        "step_seconds_max": step_seconds,
        "global_action_queries_per_second": (
            global_queries_this_step / step_seconds
        ),
        "elapsed_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": int(
            reduce_max(
                torch.cuda.max_memory_allocated(runtime.context.device),
                runtime.context,
            )
        ),
        "max_cuda_reserved_bytes": int(
            reduce_max(
                torch.cuda.max_memory_reserved(runtime.context.device),
                runtime.context,
            )
        ),
    }


def run_writer_step(
    runtime: WriterRuntime,
    step: int,
    started: float,
) -> dict[str, Any]:
    """Run one task-balanced positive functional AS-Writer update."""

    tick = time.monotonic()
    runtime.optimizer.zero_grad(set_to_none=True)
    data_tick = time.monotonic()
    batch = next(runtime.iterator)
    data_seconds = time.monotonic() - data_tick
    task_id, task_visit = runtime.sampler.task_visit_for_step(step)
    if _batch_task_id(batch) != task_id:
        raise WriterModelError("AS-Writer sampler and action batch disagree")
    observed_batch = int(batch["task_id"].shape[0])
    if observed_batch != runtime.sampler.batch_size_for_step(step):
        raise WriterModelError("AS-Writer action-query batch size changed")
    demo_index = runtime.video_schedule.demo_for_task_visit(task_id, task_visit)
    packed, video_metrics = _pack_raw_condition(
        runtime,
        task_id=task_id,
        demo_index=demo_index,
        data_step=step,
    )
    policy_batch = runtime.processor.training_batch(batch)
    loss, detail = _differentiate_condition(runtime, packed, policy_batch)
    if not bool(torch.isfinite(loss)):
        raise WriterModelError(f"non-finite AS-Writer loss at step {step}")
    if any(parameter.grad is not None for parameter in runtime.policy.parameters()):
        raise WriterModelError("frozen PI05 source policy accumulated gradients")
    grad_norm = torch.nn.utils.clip_grad_norm_(
        runtime.writer.parameters(),
        float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"]),
    )
    if not bool(torch.isfinite(grad_norm).detach()):
        raise WriterModelError(f"non-finite AS-Writer gradient at step {step}")
    applied_lr = float(runtime.optimizer.param_groups[0]["lr"])
    runtime.optimizer.step()
    runtime.scheduler.step()
    return _step_metrics(
        runtime,
        step=step,
        started=started,
        tick=tick,
        task_id=task_id,
        demo_index=demo_index,
        observed_batch=observed_batch,
        loss=loss,
        detail=detail,
        data_seconds=data_seconds,
        video_metrics=video_metrics,
        grad_norm=grad_norm,
        applied_lr=applied_lr,
    )
