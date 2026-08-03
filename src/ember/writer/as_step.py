"""One raw task-balanced macro update for AS-Writer training."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, Mapping

import torch
import torch.distributed as dist

from ember.pi05_source_setup import reduce_max, reduce_mean
from ember.writer.task_gradient import (
    assign_flat_gradient,
    compose_distributed_raw_mean_gradient,
    flatten_task_gradient,
    gradient_direction_sketches,
)
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
]


def _batch_task_id(batch: Mapping[str, Any]) -> int:
    values = batch.get("task_id")
    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise WriterModelError("AS-Writer action batch lost task identity")
    unique = values.unique()
    if unique.numel() != 1:
        raise WriterModelError("one AS-Writer rank received multiple tasks")
    return int(unique.item())


def _optimizer_updates_per_task_cycle(runtime: WriterRuntime) -> int:
    value = int(
        getattr(runtime.sampler, "optimizer_updates_per_task_cycle", 1)
    )
    if value <= 0:
        raise WriterModelError("invalid optimizer-update task cycle")
    return value


def _pack_raw_conditions(
    runtime: WriterRuntime,
    *,
    task_id: int,
    teacher_demo: int,
    action_batch_size: int,
) -> tuple[WriterCondition, dict[str, Any]]:
    if runtime.videos_per_task_visit != 1 or action_batch_size <= 0:
        raise WriterModelError("AS-Writer one-video condition changed")
    video = runtime.video_store.load(task_id, int(teacher_demo))
    frames = torch.from_numpy(video.frames).to(
        runtime.context.device,
        non_blocking=True,
    )
    indices = torch.from_numpy(video.frame_indices).to(
        runtime.context.device,
        non_blocking=True,
    )
    video_offsets = torch.tensor(
        [0, int(video.frames.shape[0])],
        dtype=torch.long,
        device=runtime.context.device,
    )
    tokens, mask, task_span = runtime.language_tokens[task_id]
    return (
        frames,
        indices,
        video_offsets,
        tokens,
        mask,
        task_span,
    ), {
        "teacher_demo_index": int(teacher_demo),
        "action_video_assignment": "all_actions_share_single_video_lora",
        "actions_per_video": action_batch_size,
        "logical_policy_pairs": action_batch_size,
        "unique_teacher_video_conditions": 1,
        "teacher_video_raw_frames": video.raw_frame_count,
        "teacher_video_sampled_frames": int(video.frames.shape[0]),
    }


def _differentiate_conditions(
    runtime: WriterRuntime,
    packed: WriterCondition,
    policy_batch: Mapping[str, Any],
    flat_gradient: torch.Tensor,
    policy_rng_seed: int | None,
) -> tuple[torch.Tensor, Mapping[str, Any], torch.Tensor]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated = runtime.writer(
            *packed,
            policy=runtime.policy,
        )
        loss, detail, gradients = functional_lora_loss_gradient(
            runtime.policy,
            generated,
            runtime.lora_contract,
            batch=policy_batch,
            policy_rng_seed=policy_rng_seed,
            policy_rng_device=(
                runtime.context.device if policy_rng_seed is not None else None
            ),
            flow_time_sampling_scheme=runtime.config[
                "conditioning_training"
            ].get("policy_flow_time_sampling_scheme"),
            flow_noise_sampling_scheme=runtime.config[
                "conditioning_training"
            ].get("policy_flow_noise_sampling_scheme"),
            policy_microbatch_size=runtime.config["optimization"].get(
                "functional_policy_microbatch_size"
            ),
        )
    names = tuple(generated)
    parameter_gradients = torch.autograd.grad(
        tuple(generated[name] for name in names),
        tuple(item.parameter for item in runtime.gradient_layout),
        grad_outputs=tuple(gradients[name] for name in names),
        allow_unused=True,
    )
    flat_gradient = flatten_task_gradient(
        parameter_gradients,
        runtime.gradient_layout,
        output=flat_gradient,
    )
    return loss, detail, flat_gradient


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
    demo_indices = batch.get("demo_index")
    frame_indices = batch.get("frame_index")
    if (
        not isinstance(demo_indices, torch.Tensor)
        or not isinstance(frame_indices, torch.Tensor)
        or demo_indices.ndim != 1
        or frame_indices.shape != demo_indices.shape
        or demo_indices.numel() != runtime.batch_size
    ):
        raise WriterModelError("action query lost immutable randomness identity")
    return task_query_policy_rng_seed(
        optimization_seed=int(runtime.config["optimization"]["seed"]),
        task_id=task_id,
        task_visit=task_visit,
        demo_indices=demo_indices.detach().cpu().tolist(),
        frame_indices=frame_indices.detach().cpu().tolist(),
    )


def _global_task_video_assignments(
    runtime: WriterRuntime,
    step: int,
) -> list[dict[str, int]]:
    assignments = []
    for (
        rank,
        microtask,
        task_id,
        task_visit,
    ) in runtime.sampler.assignments_for_step(step):
        teacher_demo = runtime.video_schedule.demo_for_task_visit(
            task_id,
            task_visit,
        )
        assignments.append(
            {
                "rank": rank,
                "microtask": microtask,
                "task_id": task_id,
                "task_visit": task_visit,
                "teacher_demo_index": teacher_demo,
                "teacher_video_sampled_frames": (
                    runtime.sampler.task_video_costs[task_id][teacher_demo]
                ),
            }
        )
    return assignments


def _global_single_video_diagnostics(
    runtime: WriterRuntime,
    records: list[dict[str, Any]],
    assignments: list[dict[str, int]],
    gradient_composition: Mapping[str, Any],
    local_gradient_sketches: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    """Retain per-task direction evidence for cross-macro video-noise analysis."""

    block_names = tuple(sorted(local_gradient_sketches))
    if not block_names:
        raise WriterModelError("single-video diagnostics lost gradient sketches")
    dimensions = int(local_gradient_sketches[block_names[0]].shape[1])
    if dimensions <= 0 or any(
        value.shape != (len(records), dimensions)
        for value in local_gradient_sketches.values()
    ):
        raise WriterModelError("single-video gradient sketch shape changed")
    local = torch.tensor(
        [[float(record["task_id"]), float(record["loss"])] for record in records],
        dtype=torch.float32,
        device=runtime.context.device,
    )
    local = torch.cat(
        [local, *(local_gradient_sketches[name] for name in block_names)],
        dim=1,
    )
    if runtime.context.world_size == 1:
        gathered = local
    else:
        gathered = torch.empty(
            runtime.context.world_size * local.shape[0],
            local.shape[1],
            dtype=torch.float32,
            device=runtime.context.device,
        )
        dist.all_gather_into_tensor(gathered, local.contiguous())
    gathered = gathered.index_select(0, torch.argsort(gathered[:, 0]))
    task_ids = [int(value) for value in gathered[:, 0].detach().cpu().tolist()]
    expected = [int(value) for value in gradient_composition["task_ids"]]
    if task_ids != expected:
        raise WriterModelError("single-video diagnostics lost task ordering")
    losses = gathered[:, 1].detach().cpu().tolist()
    gathered_sketches = gathered[:, 2:].detach().cpu().reshape(
        len(task_ids), len(block_names), dimensions
    )
    assignment_by_task = {int(row["task_id"]): row for row in assignments}
    raw_gram = gradient_composition["raw_gradient_gram"]
    raw_dots = gradient_composition["raw_candidate_task_dots"]
    if (
        len(assignment_by_task) != len(task_ids)
        or any(len(value) != len(task_ids) for value in raw_gram)
        or len(raw_dots) != len(task_ids)
    ):
        raise WriterModelError("single-video diagnostics changed shape")
    return [
        {
            "task_id": task_id,
            "teacher_demo_index": int(assignment_by_task[task_id]["teacher_demo_index"]),
            "teacher_video_sampled_frames": int(
                assignment_by_task[task_id]["teacher_video_sampled_frames"]
            ),
            "functional_action_loss": float(losses[index]),
            "raw_task_gradient_norm": math.sqrt(
                max(float(raw_gram[index][index]), 0.0)
            ),
            "raw_task_dot_candidate_direction": float(raw_dots[index]),
            "raw_task_gradient_direction_sketch": {
                block: gathered_sketches[index, block_index].tolist()
                for block_index, block in enumerate(block_names)
            },
        }
        for index, task_id in enumerate(task_ids)
    ]


def _throughput_and_exposure_metrics(
    runtime: WriterRuntime,
    *,
    completed: int,
    global_queries_this_step: int,
    global_tasks_this_step: int,
    step_seconds: float,
    data_seconds: float,
    started: float,
) -> dict[str, Any]:
    updates_per_cycle = _optimizer_updates_per_task_cycle(runtime)
    policy_microbatch_size = int(
        runtime.config["optimization"].get(
            "functional_policy_microbatch_size", runtime.batch_size
        )
    )
    physical_forwards_per_task = math.ceil(
        runtime.batch_size / policy_microbatch_size
    )
    return {
        "global_unique_action_queries": (
            completed
            * runtime.batch_size
            * runtime.tasks_per_rank_per_update
            * runtime.context.world_size
        ),
        "global_policy_samples": (
            completed
            * runtime.batch_size
            * runtime.tasks_per_rank_per_update
            * runtime.context.world_size
        ),
        "global_writer_video_conditions": (
            completed
            * runtime.videos_per_task_visit
            * runtime.tasks_per_rank_per_update
            * runtime.context.world_size
        ),
        "global_policy_functional_forwards": completed * global_tasks_this_step,
        "functional_policy_microbatch_size": policy_microbatch_size,
        "physical_policy_forwards_per_task": physical_forwards_per_task,
        "global_physical_policy_forwards_this_step": (
            global_tasks_this_step * physical_forwards_per_task
        ),
        "global_unique_action_queries_this_step": global_queries_this_step,
        "global_policy_samples_this_step": global_queries_this_step,
        "global_writer_video_conditions_this_step": global_tasks_this_step,
        "global_unique_writer_video_conditions_this_step": global_tasks_this_step,
        "global_tasks_this_step": global_tasks_this_step,
        "data_seconds_max": reduce_max(data_seconds, runtime.context),
        "step_seconds_max": step_seconds,
        "global_action_queries_per_second": global_queries_this_step / step_seconds,
        "global_policy_pairs_per_second": global_queries_this_step / step_seconds,
        "global_task_video_conditions_per_second": (
            global_tasks_this_step / step_seconds
        ),
        "optimizer_updates_per_hour": 3600.0 / step_seconds,
        "task_cycles_per_hour": (
            3600.0
            / step_seconds
            / updates_per_cycle
        ),
        **(
            {"macro_updates_per_hour": 3600.0 / step_seconds}
            if updates_per_cycle == 1
            else {}
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


def _gradient_step_metrics(
    runtime: WriterRuntime,
    gradient_composition: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "optimizer_gradient_accumulation": False,
        "task_gradient_collection": True,
        "task_loss_scale": None,
        "gradient_composition": gradient_composition,
        "ddp_synchronizations_this_step": 0,
        "gradient_gram_chunk_allgathers_this_step": int(
            gradient_composition["gradient_gram_chunk_allgathers"]
        ),
        "gradient_gram_chunk_collective_completions_this_step": int(
            gradient_composition[
                "gradient_gram_chunk_collective_completions"
            ]
        ),
        "gradient_gram_chunk_cuda_synchronizations_this_step": int(
            gradient_composition["gradient_gram_chunk_cuda_synchronizations"]
        ),
        "gradient_task_id_allgathers_this_step": int(
            gradient_composition["gradient_task_id_allgathers"]
        ),
        "gradient_collectives_this_step": int(
            gradient_composition["gradient_collectives"]
        ),
        "diagnostic_tensor_allgathers_this_step": (
            1 if runtime.context.world_size > 1 else 0
        ),
    }


def _step_metrics(
    runtime: WriterRuntime,
    *,
    step: int,
    started: float,
    tick: float,
    records: list[dict[str, Any]],
    conditioning_mode: str,
    data_seconds: float,
    grad_norm: torch.Tensor,
    optimizer_update: Mapping[str, Any],
    optimizer_diagnostics: Mapping[str, Any] | None,
    gradient_composition: Mapping[str, Any],
    local_gradient_sketches: Mapping[str, torch.Tensor],
    scheduler_advanced: bool,
) -> dict[str, Any]:
    completed = step + 1
    updates_per_cycle = _optimizer_updates_per_task_cycle(runtime)
    task_cycle, task_cycle_phase = divmod(step, updates_per_cycle)
    step_seconds = reduce_max(time.monotonic() - tick, runtime.context)
    local_queries = sum(int(record["observed_batch"]) for record in records)
    global_queries_this_step = local_queries * runtime.context.world_size
    global_tasks_this_step = len(records) * runtime.context.world_size
    mean_loss = sum(float(record["loss"]) for record in records) / len(records)
    assignments = _global_task_video_assignments(runtime, step)
    single_video_diagnostics = _global_single_video_diagnostics(
        runtime,
        records,
        assignments,
        gradient_composition,
        local_gradient_sketches,
    )
    cursor_metrics = {
        "optimizer_update": completed,
        "task_cycle": task_cycle,
        "task_cycle_phase": task_cycle_phase,
        "task_cycle_completed": (
            task_cycle_phase
            == updates_per_cycle - 1
        ),
        "completed_task_cycles": (
            completed // updates_per_cycle
        ),
        "optimizer_updates_per_task_cycle": (
            updates_per_cycle
        ),
        "scheduler_advanced_this_update": scheduler_advanced,
        "scheduler_logical_updates": (
            completed // updates_per_cycle
        ),
    }
    if updates_per_cycle == 1:
        cursor_metrics["macro_optimizer_update"] = completed
    return {
        "optimizer_step": completed,
        **cursor_metrics,
        "conditioning_mode": conditioning_mode,
        "writer_language_condition": "correct_task_language",
        "policy_language_condition": "correct_action_query_task_language",
        "mean_functional_action_loss": reduce_mean(mean_loss, runtime.context),
        "gradient_norm_before_clip_max": reduce_max(
            float(grad_norm),
            runtime.context,
        ),
        "logical_lr": float(optimizer_update["logical_lr"]),
        "applied_lr": float(optimizer_update["applied_lr"]),
        "optimizer_lr_divisor": int(optimizer_update["lr_divisor"]),
        "cycle_lr_integral": float(optimizer_update["applied_lr"])
        * int(optimizer_update["lr_divisor"]),
        "reference_weight_decay": float(
            optimizer_update["reference_weight_decay"]
        ),
        "applied_weight_decay": float(
            optimizer_update["applied_weight_decay"]
        ),
        "optimizer_cycle_normalization": str(optimizer_update["mode"]),
        "policy_randomness_scheme": runtime.config["conditioning_training"].get(
            "policy_randomness_scheme", "ambient_rank_cuda_stream"
        ),
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "next_logical_lr": float(runtime.scheduler.get_last_lr()[0]),
        "gradient_clip_norm": float(
            runtime.config["optimization"]["optimizer"]["gradient_clip_norm"]
        ),
        "gradient_clip_triggered": float(grad_norm)
        > float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"]),
        "gradient_clip_coefficient": min(
            1.0,
            float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"])
            / (float(grad_norm) + 1e-6),
        ),
        **(
            {"optimizer_diagnostics": dict(optimizer_diagnostics)}
            if optimizer_diagnostics is not None
            else {}
        ),
        **_throughput_and_exposure_metrics(
            runtime,
            completed=completed,
            global_queries_this_step=global_queries_this_step,
            global_tasks_this_step=global_tasks_this_step,
            step_seconds=step_seconds,
            data_seconds=data_seconds,
            started=started,
        ),
        "global_task_video_assignments": assignments,
        "single_video_condition_diagnostics": single_video_diagnostics,
        "single_video_noise_observable": (
            "per-task loss, exact gradient norm, candidate alignment, and "
            "fixed 32-dimensional per-module CountSketch direction across "
            "successive one-video macro visits"
        ),
        "rank0_task_ids": [int(record["task_id"]) for record in records],
        "rank0_conditions": [
            {
                "task_id": int(record["task_id"]),
                **dict(record["video_metrics"]),
            }
            for record in records
        ],
        "rank0_policy_loss_details": [
            {
                "task_id": int(record["task_id"]),
                "detail": record["detail"],
                "policy_rng_seed": record["policy_rng_seed"],
            }
            for record in records
        ],
        "action_query_batch_size_per_task": runtime.batch_size,
        "tasks_per_rank_per_optimizer_update": (
            runtime.tasks_per_rank_per_update
        ),
        "teacher_videos_per_task_visit": runtime.videos_per_task_visit,
        "local_policy_forward_calls_this_step": len(records),
        "policy_forward_calls_this_step": global_tasks_this_step,
        **_gradient_step_metrics(runtime, gradient_composition),
    }


def _collect_task_gradients(
    runtime: WriterRuntime, step: int
) -> tuple[list[dict[str, Any]], float, list[int], torch.Tensor]:
    records: list[dict[str, Any]] = []
    data_seconds = 0.0
    local_task_ids: list[int] = []
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
        task_id, task_visit = runtime.sampler.task_visit_for_step(
            step,
            microtask,
        )
        if _batch_task_id(batch) != task_id:
            raise WriterModelError("AS-Writer sampler and action batch disagree")
        observed_batch = int(batch["task_id"].shape[0])
        if observed_batch != runtime.sampler.batch_size_for_step(step):
            raise WriterModelError("AS-Writer action-query batch size changed")
        teacher_demo = runtime.video_schedule.demo_for_task_visit(
            task_id,
            task_visit,
        )
        packed, video_metrics = _pack_raw_conditions(
            runtime,
            task_id=task_id,
            teacher_demo=teacher_demo,
            action_batch_size=observed_batch,
        )
        policy_rng_seed = _policy_rng_seed_for_batch(
            runtime,
            batch,
            task_id=task_id,
            task_visit=task_visit,
        )
        policy_batch = runtime.processor.training_batch(batch)
        loss, detail, task_gradient = _differentiate_conditions(
            runtime,
            packed,
            policy_batch,
            local_gradients[microtask],
            policy_rng_seed,
        )
        if not bool(torch.isfinite(loss)):
            raise WriterModelError(
                f"non-finite AS-Writer loss at step {step}, microtask {microtask}"
            )
        records.append(
            {
                "task_id": task_id,
                "observed_batch": observed_batch,
                "loss": float(loss.detach()),
                "detail": detail,
                "video_metrics": video_metrics,
                "policy_rng_seed": policy_rng_seed,
            }
        )
        local_task_ids.append(task_id)
        if task_gradient.data_ptr() != local_gradients[microtask].data_ptr():
            raise WriterModelError("task gradient lost its preallocated row")
    return records, data_seconds, local_task_ids, local_gradients


def _compose_selected_gradient(
    runtime: WriterRuntime,
    *,
    step: int,
    local_task_ids: list[int],
    local_gradients: torch.Tensor,
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    selected_task_ids = (
        tuple(
            sorted(
                task_id
                for _, _, task_id, _
                in runtime.sampler.assignments_for_step(step)
            )
        )
        if hasattr(runtime.sampler, "assignments_for_step")
        else runtime.task_ids
    )
    direction, gradient_metrics = compose_distributed_raw_mean_gradient(
        torch.tensor(
            local_task_ids,
            dtype=torch.long,
            device=runtime.context.device,
        ),
        local_gradients,
        runtime.gradient_layout,
        expected_task_ids=selected_task_ids,
        world_size=runtime.context.world_size,
        rank=runtime.context.rank,
    )
    gradient_metrics["selected_task_count"] = len(selected_task_ids)
    if _optimizer_updates_per_task_cycle(runtime) > 1:
        gradient_metrics[
            "orthogonal_equal_norm_mean_to_task_energy_reference"
        ] = 1.0 / len(selected_task_ids)
        gradient_metrics["energy_ratio_interpretation"] = (
            "selected4_manipulation_check_only_not_scientific_success"
        )
    local_gradient_sketches = gradient_direction_sketches(
        local_gradients,
        runtime.gradient_layout,
    )
    assign_flat_gradient(direction, runtime.gradient_layout)
    return gradient_metrics, local_gradient_sketches


def run_writer_step(
    runtime: WriterRuntime,
    step: int,
    started: float,
) -> dict[str, Any]:
    """Run one optimizer update under the configured task assignment."""

    tick = time.monotonic()
    runtime.optimizer.zero_grad(set_to_none=True)
    mode = str(runtime.config["conditioning_training"]["method"])
    supported_modes = {
        "raw_task_complete_single_video_multi_action_positive_functional_loss",
        "raw_serial4_exposure_matched_single_video_multi_action_positive_functional_loss",
        "task_query_keyed_raw_task_complete_single_video_multi_action_positive_functional_loss",
        "variance_reduced_task_query_keyed_raw_task_complete_single_video_multi_action_positive_functional_loss",
        "cycle_normalized_randomized_group4_single_video_multi_action_positive_functional_loss",
    }
    if mode not in supported_modes:
        raise WriterModelError("unsupported AS-Writer conditioning mode")
    records, data_seconds, local_task_ids, local_gradients = (
        _collect_task_gradients(runtime, step)
    )
    gradient_metrics, local_gradient_sketches = _compose_selected_gradient(
        runtime,
        step=step,
        local_task_ids=local_task_ids,
        local_gradients=local_gradients,
    )
    if any(parameter.grad is not None for parameter in runtime.policy.parameters()):
        raise WriterModelError("frozen PI05 source policy accumulated gradients")
    grad_norm = torch.nn.utils.clip_grad_norm_(
        runtime.writer.parameters(),
        float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"]),
    )
    if not bool(torch.isfinite(grad_norm).detach()):
        raise WriterModelError(f"non-finite AS-Writer gradient at step {step}")
    optimizer_update = prepare_optimizer_update(
        runtime.optimizer,
        runtime.scheduler,
        runtime.config,
    )
    before = capture_optimizer_parameters(runtime)
    runtime.optimizer.step()
    optimizer_diagnostics = optimizer_state_metrics(runtime, before)
    scheduler_advanced = advance_scheduler_after_update(
        runtime.scheduler,
        completed_optimizer_updates=step + 1,
        optimizer_updates_per_task_cycle=(
            _optimizer_updates_per_task_cycle(runtime)
        ),
    )
    return _step_metrics(
        runtime,
        step=step,
        started=started,
        tick=tick,
        records=records,
        conditioning_mode=mode,
        data_seconds=data_seconds,
        grad_norm=grad_norm,
        optimizer_update=optimizer_update,
        optimizer_diagnostics=optimizer_diagnostics,
        gradient_composition=gradient_metrics,
        local_gradient_sketches=local_gradient_sketches,
        scheduler_advanced=scheduler_advanced,
    )
