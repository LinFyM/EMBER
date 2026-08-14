"""One equal-task full24 dynamic-K Writer update."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.writer.data import pack_teacher_condition
from ember.writer.errors import WriterModelError
from ember.writer.functional import (
    TASK_LOGICAL_BATCH_POLICY_RNG_SCHEME,
    functional_lora_loss_gradient,
    task_logical_batch_policy_rng_seed,
)

if TYPE_CHECKING:
    from ember.writer.training import WriterRuntime


@dataclass(frozen=True)
class ParameterSlice:
    parameter: torch.nn.Parameter
    start: int
    stop: int


def parameter_layout(module: torch.nn.Module) -> tuple[ParameterSlice, ...]:
    layout = []
    cursor = 0
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        layout.append(ParameterSlice(parameter, cursor, cursor + parameter.numel()))
        cursor += parameter.numel()
    if not layout:
        raise WriterModelError("dynamic-K Writer has no trainable parameters")
    return tuple(layout)


def accumulate_flat_gradient(
    destination: torch.Tensor,
    gradients: Sequence[torch.Tensor | None],
    layout: Sequence[ParameterSlice],
) -> None:
    if len(gradients) != len(layout) or destination.numel() != layout[-1].stop:
        raise WriterModelError("dynamic-K flat gradient layout changed")
    for gradient, item in zip(gradients, layout, strict=True):
        if gradient is not None:
            destination[item.start : item.stop].add_(gradient.detach().reshape(-1))


def reduce_full24_gradient(
    gradient_sum: torch.Tensor,
    *,
    world_size: int,
    global_task_count: int = 24,
) -> torch.Tensor:
    """Perform the sole P-vector reduction and return the exact task mean."""

    if gradient_sum.ndim != 1 or world_size <= 0 or global_task_count != 24:
        raise WriterModelError("invalid full24 gradient reduction")
    if world_size > 1:
        dist.all_reduce(gradient_sum, op=dist.ReduceOp.SUM)
    gradient_sum.div_(global_task_count)
    return gradient_sum


def assign_flat_gradient(
    gradient: torch.Tensor, layout: Sequence[ParameterSlice]
) -> None:
    if not layout or gradient.numel() != layout[-1].stop:
        raise WriterModelError("dynamic-K optimizer gradient layout changed")
    for item in layout:
        item.parameter.grad = (
            gradient[item.start : item.stop]
            .view_as(item.parameter)
            .to(dtype=item.parameter.dtype)
        )


def gather_full24_records(
    local_records: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
    task_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Gather the small per-task evidence rows without touching tensor gradients."""

    shards: list[Any] = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(shards, list(local_records))
    else:
        shards[0] = list(local_records)
    records = [dict(row) for shard in shards for row in shard]
    expected = {int(task_id) for task_id in task_ids}
    observed = [int(row["task_id"]) for row in records]
    if len(records) != 24 or len(set(observed)) != 24 or set(observed) != expected:
        raise WriterModelError("dynamic-K per-task evidence lost full24 coverage")
    return sorted(records, key=lambda row: int(row["task_id"]))


def _batch_task_id(batch: Mapping[str, Any]) -> int:
    values = batch.get("task_id")
    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise WriterModelError("dynamic-K action batch lost task identity")
    unique = values.unique()
    if unique.numel() != 1:
        raise WriterModelError("dynamic-K action batch crossed tasks")
    return int(unique.item())


def _teacher_demos(
    runtime: WriterRuntime,
    *,
    task_id: int,
    task_visit: int,
    batch: Mapping[str, Any],
) -> tuple[int, ...]:
    demos = batch.get("demo_index")
    if not isinstance(demos, torch.Tensor) or demos.ndim != 1:
        raise WriterModelError("dynamic-K action batch lost episode identity")
    excluded = tuple(int(value) for value in demos.unique().cpu().tolist())
    scheduled_excluded = runtime.sampler.action_demo_indices_for_task_visit(
        task_id, task_visit
    )
    if tuple(sorted(excluded)) != scheduled_excluded:
        raise WriterModelError("dynamic-K sampler and action batch episodes differ")
    selected = runtime.video_schedule.demos_for_task_visit(
        task_id, task_visit, excluded=excluded
    )
    if (
        len(selected)
        != runtime.video_schedule.shot_count_for_task_visit(task_id, task_visit)
        or len(set(selected)) != len(selected)
        or set(selected) & set(excluded)
    ):
        raise WriterModelError("dynamic-K video/action complement changed")
    return selected


def _pack_condition(
    runtime: WriterRuntime,
    task_id: int,
    demos: Sequence[int],
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    return pack_teacher_condition(
        runtime.video_store,
        task_id=task_id,
        demos=demos,
        language=runtime.language_tokens[task_id],
        device=runtime.context.device,
    )


def _policy_seed(
    runtime: WriterRuntime,
    batch: Mapping[str, Any],
    task_id: int,
    task_visit: int,
) -> int:
    scheme = runtime.config["conditioning_training"]["policy_randomness_scheme"]
    if scheme != TASK_LOGICAL_BATCH_POLICY_RNG_SCHEME:
        raise WriterModelError("dynamic-K policy RNG scheme changed")
    demos = batch["demo_index"]
    frames = batch["frame_index"]
    return task_logical_batch_policy_rng_seed(
        optimization_seed=int(runtime.config["optimization"]["seed"]),
        task_id=task_id,
        task_visit=task_visit,
        demo_indices=demos.cpu().tolist(),
        frame_indices=frames.cpu().tolist(),
    )


def _task_gradient(
    runtime: WriterRuntime,
    packed: tuple[torch.Tensor, ...],
    policy_batch: Mapping[str, Any],
    policy_seed: int,
    gradient_sum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    device_type = runtime.context.device.type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=device_type == "cuda",
    ):
        generated, consistency = runtime.writer.forward_training(
            *packed,
            policy=runtime.policy,
            singleton_video_index=0,
        )
        functional_loss, detail, lora_gradients = functional_lora_loss_gradient(
            runtime.policy,
            generated,
            runtime.lora_contract,
            batch=policy_batch,
            policy_rng_seed=policy_seed,
            policy_rng_device=runtime.context.device,
            flow_time_sampling_scheme=runtime.config["conditioning_training"][
                "policy_flow_time_sampling_scheme"
            ],
            flow_noise_sampling_scheme=runtime.config["conditioning_training"][
                "policy_flow_noise_sampling_scheme"
            ],
            policy_microbatch_size=int(
                runtime.config["optimization"]["functional_policy_microbatch_size"]
            ),
            collect_policy_details=False,
        )
    if not isinstance(consistency, torch.Tensor) or consistency.ndim != 0:
        raise WriterModelError("dynamic-K Writer consistency loss is not scalar")
    names = tuple(generated)
    weight = float(
        runtime.config["conditioning_training"]["singleton_to_full_consistency"][
            "weight"
        ]
    )
    video_count = int(packed[3][-1])
    if video_count == 1 and float(consistency.detach()) != 0.0:
        raise WriterModelError("K1 dynamic-K consistency loss must be exact zero")
    consistency_kind = runtime.config["conditioning_training"][
        "singleton_to_full_consistency"
    ]["kind"]
    if (
        video_count > 1
        and consistency_kind != "exact_zero_no_auxiliary_loss"
        and not consistency.requires_grad
    ):
        raise WriterModelError("K>1 dynamic-K consistency lost its training graph")
    active_names = tuple(name for name in names if generated[name].requires_grad)
    if not active_names:
        if (
            video_count == 1
            and consistency_kind == "exact_zero_no_auxiliary_loss"
            and weight == 0.0
        ):
            return functional_loss, consistency.detach(), detail
        raise WriterModelError("dynamic-K generated LoRA lost every trainable output")
    outputs = tuple(generated[name] for name in active_names)
    grad_outputs = tuple(lora_gradients[name] for name in active_names)
    if consistency.requires_grad and weight:
        outputs = (*outputs, consistency)
        grad_outputs = (*grad_outputs, torch.ones_like(consistency) * weight)
    before = tuple(item.parameter.grad for item in runtime.gradient_layout)
    if any(value is not None for value in before):
        raise WriterModelError("dynamic-K task gradient buffer was not cleared")
    torch.autograd.backward(outputs, grad_tensors=grad_outputs)
    gradients = tuple(item.parameter.grad for item in runtime.gradient_layout)
    accumulate_flat_gradient(gradient_sum, gradients, runtime.gradient_layout)
    for item in runtime.gradient_layout:
        item.parameter.grad = None
    return functional_loss, consistency.detach(), detail


def run_writer_step(
    runtime: WriterRuntime, macro: int, started: float
) -> dict[str, Any]:
    """Apply one optimizer update after all 24 tasks contribute exactly once."""

    tick = time.monotonic()
    runtime.optimizer.zero_grad(set_to_none=True)
    flat = torch.zeros(
        runtime.gradient_layout[-1].stop,
        dtype=torch.float32,
        device=runtime.context.device,
    )
    records = []
    for task_id, task_visit in (
        runtime.sampler.task_visit_for_step(macro, index)
        for index in range(len(runtime.sampler.tasks_for_step(macro)))
    ):
        batch = next(runtime.iterator)
        if _batch_task_id(batch) != task_id:
            raise WriterModelError("dynamic-K sampler and loaded task differ")
        demos = _teacher_demos(
            runtime,
            task_id=task_id,
            task_visit=task_visit,
            batch=batch,
        )
        packed, video_metrics = _pack_condition(runtime, task_id, demos)
        policy_seed = _policy_seed(runtime, batch, task_id, task_visit)
        functional, consistency, _ = _task_gradient(
            runtime,
            packed,
            runtime.processor.training_batch(batch),
            policy_seed,
            flat,
        )
        if not bool(torch.isfinite(functional)) or not bool(
            torch.isfinite(consistency)
        ):
            raise WriterModelError(f"non-finite dynamic-K loss at macro {macro}")
        records.append(
            {
                "task_id": task_id,
                "task_visit": task_visit,
                "functional_loss": float(functional),
                "consistency_loss": float(consistency),
                **video_metrics,
            }
        )
    if any(parameter.grad is not None for parameter in runtime.policy.parameters()):
        raise WriterModelError("frozen source policy accumulated gradients")
    reduce_full24_gradient(flat, world_size=runtime.context.world_size)
    assign_flat_gradient(flat, runtime.gradient_layout)
    clip = float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"])
    grad_norm = torch.nn.utils.clip_grad_norm_(runtime.trainable_parameters, clip)
    if not bool(torch.isfinite(grad_norm)):
        raise WriterModelError(f"non-finite dynamic-K gradient at macro {macro}")
    global_k_histogram = {
        str(k): sum(
            runtime.video_schedule.shot_count_for_task_visit(task_id, visit) == k
            for _, _, task_id, visit in runtime.sampler.assignments_for_step(macro)
        )
        for k in range(1, 5)
    }
    if set(global_k_histogram.values()) != {6}:
        raise WriterModelError("dynamic-K macro lost its exact 6/6/6/6 balance")
    global_records = gather_full24_records(
        records,
        world_size=runtime.context.world_size,
        task_ids=runtime.task_ids,
    )
    runtime.optimizer.step()
    runtime.scheduler.step()
    completed = macro + 1
    return {
        "macro": completed,
        "rank": runtime.context.rank,
        "local_task_count": len(records),
        "local_task_ids": [row["task_id"] for row in records],
        "local_mean_functional_loss": sum(row["functional_loss"] for row in records)
        / len(records),
        "local_mean_consistency_loss": sum(row["consistency_loss"] for row in records)
        / len(records),
        "local_k_histogram": {
            str(k): sum(row["K"] == k for row in records) for k in range(1, 5)
        },
        "global_k_histogram": global_k_histogram,
        "global_mean_functional_loss": sum(
            row["functional_loss"] for row in global_records
        )
        / len(global_records),
        "global_mean_consistency_loss": sum(
            row["consistency_loss"] for row in global_records
        )
        / len(global_records),
        "gradient_norm_before_clip": float(grad_norm),
        "gradient_clip_norm": clip,
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "macro_seconds": time.monotonic() - tick,
        "elapsed_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
            runtime.context.device
        ),
        "max_cuda_reserved_bytes": torch.cuda.max_memory_reserved(
            runtime.context.device
        ),
        "conditions": global_records,
    }
