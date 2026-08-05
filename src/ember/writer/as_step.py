"""One full24 K4 invariant-program M2P Writer update."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, Mapping

import torch

from ember.pi05_source_setup import reduce_max, reduce_mean
from ember.rl_writer.rendezvous import rank_local_credit_ready
from ember.writer.functional import (
    TASK_QUERY_POLICY_RNG_SCHEME,
    functional_lora_loss_gradient,
    task_query_policy_rng_seed,
)
from ember.writer.model import WriterModelError
from ember.writer.optimizer_diagnostics import (
    capture_optimizer_parameters,
    optimizer_state_metrics,
)
from ember.writer.task_gradient import (
    assign_flat_gradient,
    compose_distributed_raw_mean_gradient,
    flatten_task_gradient,
)
from ember.writer.update_schedule import (
    advance_scheduler_after_update,
    prepare_optimizer_update,
)

if TYPE_CHECKING:
    from ember.writer.training import WriterRuntime


WriterCondition = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _batch_task_id(batch: Mapping[str, Any]) -> int:
    values = batch.get("task_id")
    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise WriterModelError("AS-Writer action batch lost task identity")
    unique = values.unique()
    if unique.numel() != 1:
        raise WriterModelError("one AS-Writer rank received multiple tasks")
    return int(unique.item())


def _teacher_demos(
    runtime: WriterRuntime,
    *,
    task_id: int,
    task_visit: int,
    batch: Mapping[str, Any],
) -> tuple[int, ...]:
    action_demos = batch.get("demo_index")
    if not isinstance(action_demos, torch.Tensor) or action_demos.ndim != 1:
        raise WriterModelError("action query lost episode identity")
    demos = runtime.video_schedule.demos_for_task_visit(
        task_id,
        task_visit,
        excluded=tuple(int(value) for value in action_demos.unique().cpu().tolist()),
    )
    expected = runtime.video_schedule.demos_for_task_visit(
        task_id,
        task_visit,
        excluded=runtime.sampler.action_demo_indices_for_task_visit(
            task_id, task_visit
        ),
    )
    if demos != expected:
        raise WriterModelError("action batch and K4 teacher-video exclusion differ")
    if len(demos) != runtime.videos_per_task_visit:
        raise WriterModelError("K4 teacher-video set changed size")
    return demos


def _pack_raw_conditions(
    runtime: WriterRuntime,
    *,
    task_id: int,
    teacher_demos: tuple[int, ...],
    action_batch_size: int,
) -> tuple[WriterCondition, dict[str, Any]]:
    if (
        runtime.videos_per_task_visit != 4
        or len(teacher_demos) != runtime.videos_per_task_visit
        or len(set(teacher_demos)) != len(teacher_demos)
        or action_batch_size <= 0
    ):
        raise WriterModelError("AS-Writer K4 condition changed")
    videos = [runtime.video_store.load(task_id, demo) for demo in teacher_demos]
    frames = torch.cat(
        [torch.from_numpy(video.frames) for video in videos], dim=0
    ).to(runtime.context.device, non_blocking=True)
    indices = torch.cat(
        [torch.from_numpy(video.frame_indices) for video in videos], dim=0
    ).to(runtime.context.device, non_blocking=True)
    counts = torch.tensor(
        [int(video.frames.shape[0]) for video in videos],
        dtype=torch.long,
        device=runtime.context.device,
    )
    video_offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=runtime.context.device),
            counts.cumsum(dim=0),
        )
    )
    condition_video_offsets = torch.tensor(
        [0, len(videos)], dtype=torch.long, device=runtime.context.device
    )
    tokens, mask, task_span = runtime.language_tokens[task_id]
    return (
        frames,
        indices,
        video_offsets,
        condition_video_offsets,
        tokens,
        mask,
        task_span,
    ), {
        "teacher_demo_indices": list(teacher_demos),
        "actions_per_video_set": action_batch_size,
        "teacher_video_raw_frames": [video.raw_frame_count for video in videos],
        "teacher_video_sampled_frames": [
            int(video.frames.shape[0]) for video in videos
        ],
        "teacher_video_set_sampled_frames": int(counts.sum()),
    }


def _policy_rng_seed_for_batch(
    runtime: WriterRuntime,
    batch: Mapping[str, Any],
    *,
    task_id: int,
    task_visit: int,
) -> int | None:
    scheme = runtime.config["conditioning_training"].get(
        "policy_randomness_scheme"
    )
    if scheme is None:
        return None
    if scheme != TASK_QUERY_POLICY_RNG_SCHEME:
        raise WriterModelError("unsupported policy randomness scheme")
    demos = batch.get("demo_index")
    frames = batch.get("frame_index")
    if (
        not isinstance(demos, torch.Tensor)
        or not isinstance(frames, torch.Tensor)
        or demos.ndim != 1
        or frames.shape != demos.shape
        or demos.numel() != runtime.batch_size
    ):
        raise WriterModelError("action query lost immutable randomness identity")
    return task_query_policy_rng_seed(
        optimization_seed=int(runtime.config["optimization"]["seed"]),
        task_id=task_id,
        task_visit=task_visit,
        demo_indices=demos.detach().cpu().tolist(),
        frame_indices=frames.detach().cpu().tolist(),
    )


def _differentiate_condition(
    runtime: WriterRuntime,
    packed: WriterCondition,
    policy_batch: Mapping[str, Any],
    flat_gradient: torch.Tensor,
    policy_rng_seed: int | None,
) -> tuple[torch.Tensor, Mapping[str, Any], torch.Tensor]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated = runtime.writer(*packed, policy=runtime.policy)
        loss, detail, lora_gradients = functional_lora_loss_gradient(
            runtime.policy,
            generated,
            runtime.lora_contract,
            batch=policy_batch,
            policy_rng_seed=policy_rng_seed,
            policy_rng_device=(
                runtime.context.device if policy_rng_seed is not None else None
            ),
            flow_time_sampling_scheme=runtime.config["conditioning_training"].get(
                "policy_flow_time_sampling_scheme"
            ),
            flow_noise_sampling_scheme=runtime.config["conditioning_training"].get(
                "policy_flow_noise_sampling_scheme"
            ),
            policy_microbatch_size=runtime.config["optimization"].get(
                "functional_policy_microbatch_size"
            ),
        )
    names = tuple(generated)
    gradients = torch.autograd.grad(
        tuple(generated[name] for name in names),
        tuple(item.parameter for item in runtime.gradient_layout),
        grad_outputs=tuple(lora_gradients[name] for name in names),
        allow_unused=True,
    )
    return (
        loss,
        detail,
        flatten_task_gradient(
            gradients, runtime.gradient_layout, output=flat_gradient
        ),
    )


def _collect_task_gradients(
    runtime: WriterRuntime,
    step: int,
) -> tuple[list[dict[str, Any]], float, list[int], torch.Tensor]:
    records: list[dict[str, Any]] = []
    local_ids: list[int] = []
    data_seconds = 0.0
    local_gradients = torch.empty(
        runtime.tasks_per_rank_per_update,
        runtime.gradient_layout[-1].stop,
        dtype=torch.float32,
        device=runtime.context.device,
    )
    for microtask in range(runtime.tasks_per_rank_per_update):
        data_tick = time.monotonic()
        batch = next(runtime.iterator)
        data_seconds += time.monotonic() - data_tick
        task_id, task_visit = runtime.sampler.task_visit_for_step(step, microtask)
        if _batch_task_id(batch) != task_id:
            raise WriterModelError("AS-Writer sampler and action batch disagree")
        observed_batch = int(batch["task_id"].shape[0])
        if observed_batch != runtime.sampler.batch_size_for_step(step):
            raise WriterModelError("AS-Writer action-query batch size changed")
        demos = _teacher_demos(
            runtime,
            task_id=task_id,
            task_visit=task_visit,
            batch=batch,
        )
        packed, video_metrics = _pack_raw_conditions(
            runtime,
            task_id=task_id,
            teacher_demos=demos,
            action_batch_size=observed_batch,
        )
        policy_seed = _policy_rng_seed_for_batch(
            runtime,
            batch,
            task_id=task_id,
            task_visit=task_visit,
        )
        loss, detail, gradient = _differentiate_condition(
            runtime,
            packed,
            runtime.processor.training_batch(batch),
            local_gradients[microtask],
            policy_seed,
        )
        if not bool(torch.isfinite(loss)):
            raise WriterModelError(f"non-finite AS-Writer loss at step {step}")
        if gradient.data_ptr() != local_gradients[microtask].data_ptr():
            raise WriterModelError("task gradient lost its preallocated row")
        local_ids.append(task_id)
        records.append(
            {
                "task_id": task_id,
                "task_visit": task_visit,
                "observed_batch": observed_batch,
                "loss": float(loss.detach()),
                "detail": detail,
                "video_metrics": video_metrics,
                "policy_rng_seed": policy_seed,
            }
        )
    return records, data_seconds, local_ids, local_gradients


def _global_assignments(runtime: WriterRuntime, step: int) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "microtask": microtask,
            "task_id": task_id,
            "task_visit": task_visit,
            "teacher_demo_indices": list(
                runtime.video_schedule.demos_for_task_visit(
                    task_id,
                    task_visit,
                    excluded=runtime.sampler.action_demo_indices_for_task_visit(
                        task_id, task_visit
                    ),
                )
            ),
        }
        for rank, microtask, task_id, task_visit
        in runtime.sampler.assignments_for_step(step)
    ]


def _step_metrics(
    runtime: WriterRuntime,
    *,
    step: int,
    started: float,
    tick: float,
    data_seconds: float,
    records: list[dict[str, Any]],
    grad_norm: torch.Tensor,
    gradient_metrics: Mapping[str, Any],
    optimizer_update: Mapping[str, Any],
    optimizer_diagnostics: Mapping[str, Any],
    scheduler_advanced: bool,
) -> dict[str, Any]:
    completed = step + 1
    step_seconds = reduce_max(time.monotonic() - tick, runtime.context)
    global_tasks = len(runtime.task_ids)
    global_queries = global_tasks * runtime.batch_size
    config_optimizer = runtime.config["optimization"]["optimizer"]
    return {
        "optimizer_step": completed,
        "macro_optimizer_update": completed,
        "completed_task_cycles": completed,
        "conditioning_mode": runtime.config["conditioning_training"]["method"],
        "writer_language_condition": "correct_task_language_address_only",
        "policy_language_condition": "correct_action_query_task_language",
        "mean_functional_action_loss": reduce_mean(
            sum(float(row["loss"]) for row in records) / len(records),
            runtime.context,
        ),
        "gradient_norm_before_clip_max": reduce_max(float(grad_norm), runtime.context),
        "gradient_clip_norm": float(config_optimizer["gradient_clip_norm"]),
        "gradient_clip_triggered": float(grad_norm)
        > float(config_optimizer["gradient_clip_norm"]),
        "gradient_composition": dict(gradient_metrics),
        "optimizer_update": dict(optimizer_update),
        "optimizer_diagnostics": dict(optimizer_diagnostics),
        "scheduler_advanced_this_update": scheduler_advanced,
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "global_unique_action_queries": completed * global_queries,
        "global_policy_samples": completed * global_queries,
        "global_writer_video_conditions": (
            completed * global_tasks * runtime.videos_per_task_visit
        ),
        "global_unique_action_queries_this_step": global_queries,
        "global_writer_video_conditions_this_step": (
            global_tasks * runtime.videos_per_task_visit
        ),
        "global_writer_video_sets_this_step": global_tasks,
        "functional_policy_microbatch_size": int(
            runtime.config["optimization"]["functional_policy_microbatch_size"]
        ),
        "step_seconds_max": step_seconds,
        "data_seconds_max": reduce_max(data_seconds, runtime.context),
        "macro_updates_per_hour": 3600.0 / step_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": int(
            reduce_max(torch.cuda.max_memory_allocated(runtime.context.device), runtime.context)
        ),
        "max_cuda_reserved_bytes": int(
            reduce_max(torch.cuda.max_memory_reserved(runtime.context.device), runtime.context)
        ),
        "global_task_video_assignments": _global_assignments(runtime, step),
        "rank0_task_ids": [int(row["task_id"]) for row in records],
        "rank0_conditions": [
            {"task_id": int(row["task_id"]), **dict(row["video_metrics"])}
            for row in records
        ],
        "action_query_batch_size_per_task": runtime.batch_size,
        "tasks_per_rank_per_optimizer_update": runtime.tasks_per_rank_per_update,
        "teacher_videos_per_task_visit": runtime.videos_per_task_visit,
        "source_policy_trainable_parameter_count": 0,
    }


def run_writer_step(
    runtime: WriterRuntime,
    step: int,
    started: float,
) -> dict[str, Any]:
    """Apply one equal-task full24 update to the complete K4 Writer."""

    tick = time.monotonic()
    runtime.optimizer.zero_grad(set_to_none=True)
    records, data_seconds, local_ids, local_gradients = _collect_task_gradients(
        runtime, step
    )
    if any(parameter.grad is not None for parameter in runtime.policy.parameters()):
        raise WriterModelError("frozen PI05 source policy accumulated gradients")
    rank_local_credit_ready(runtime, cycle=step, epoch=0)
    direction, gradient_metrics = compose_distributed_raw_mean_gradient(
        torch.tensor(local_ids, dtype=torch.long, device=runtime.context.device),
        local_gradients,
        runtime.gradient_layout,
        expected_task_ids=runtime.task_ids,
        world_size=runtime.context.world_size,
        rank=runtime.context.rank,
    )
    assign_flat_gradient(direction, runtime.gradient_layout)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        runtime.writer.parameters(),
        float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"]),
    )
    if not bool(torch.isfinite(grad_norm).detach()):
        raise WriterModelError(f"non-finite AS-Writer gradient at step {step}")
    optimizer_update = prepare_optimizer_update(
        runtime.optimizer, runtime.scheduler, runtime.config
    )
    before = capture_optimizer_parameters(runtime)
    runtime.optimizer.step()
    optimizer_diagnostics = optimizer_state_metrics(runtime, before)
    scheduler_advanced = advance_scheduler_after_update(
        runtime.scheduler,
        completed_optimizer_updates=step + 1,
        optimizer_updates_per_task_cycle=1,
    )
    return _step_metrics(
        runtime,
        step=step,
        started=started,
        tick=tick,
        data_seconds=data_seconds,
        records=records,
        grad_norm=grad_norm,
        gradient_metrics=gradient_metrics,
        optimizer_update=optimizer_update,
        optimizer_diagnostics=optimizer_diagnostics,
        scheduler_advanced=scheduler_advanced,
    )
