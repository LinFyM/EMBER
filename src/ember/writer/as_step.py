"""One optimizer step for canonical PI05 Action-Supervised Writer training."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Mapping

import torch

from ember.pi05_source_checkpoint import capture_rng, restore_rng
from ember.pi05_source_setup import reduce_max, reduce_mean
from ember.writer.conditioning import (
    adapter_state_at,
    conditioning_cycle,
    matching_objective,
    same_torch_rng,
)
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


def _policy_microbatches(
    batch: Mapping[str, Any], microbatch_size: int
) -> tuple[dict[str, torch.Tensor], ...]:
    tensors = tuple(batch.values())
    if (
        not tensors
        or microbatch_size <= 0
        or any(not isinstance(value, torch.Tensor) or value.ndim < 1 for value in tensors)
    ):
        raise WriterModelError("AS-Writer policy batch cannot be microbatched")
    sizes = {int(value.shape[0]) for value in tensors}
    if len(sizes) != 1:
        raise WriterModelError("AS-Writer policy batch dimensions disagree")
    total = sizes.pop()
    if total <= 0:
        raise WriterModelError("AS-Writer policy batch is empty")
    return tuple(
        {
            name: value[start : min(start + microbatch_size, total)]
            for name, value in batch.items()
        }
        for start in range(0, total, microbatch_size)
    )


def _functional_arm_gradient(
    runtime: WriterRuntime,
    state: Mapping[str, torch.Tensor],
    policy_batch: Mapping[str, Any],
) -> tuple[torch.Tensor, Mapping[str, Any], dict[str, torch.Tensor]]:
    microbatches = _policy_microbatches(
        policy_batch,
        int(
            runtime.config["conditioning_training"][
                "functional_policy_microbatch_size"
            ]
        ),
    )
    total = sum(int(next(iter(batch.values())).shape[0]) for batch in microbatches)
    loss: torch.Tensor | None = None
    detail_loss = 0.0
    gradients: dict[str, torch.Tensor] | None = None
    for batch in microbatches:
        count = int(next(iter(batch.values())).shape[0])
        weight = count / total
        value, detail, current = functional_lora_loss_gradient(
            runtime.policy,
            state,
            runtime.lora_contract,
            batch=batch,
        )
        loss = value * weight if loss is None else loss + value * weight
        detail_loss += float(detail.get("loss", value)) * weight
        if gradients is None:
            gradients = {name: item * weight for name, item in current.items()}
        else:
            for name, item in current.items():
                gradients[name].add_(item, alpha=weight)
    if loss is None or gradients is None:
        raise WriterModelError("AS-Writer policy microbatch produced no gradient")
    return (
        loss,
        {"loss": detail_loss, "policy_forward_calls": len(microbatches)},
        gradients,
    )


def _differentiate_condition_batch(
    runtime: WriterRuntime,
    packed: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    policy_batch: Mapping[str, Any],
    mode: str,
    gradient_scale: float = 1.0,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[Mapping[str, Any]],
    torch.Tensor | None,
]:
    if not 0.0 < gradient_scale <= 1.0:
        raise WriterModelError("AS-Writer condition gradient scale is invalid")
    count = 1 if mode == "normal" else 2
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated = runtime.wrapped_writer(
            packed[0],
            packed[1],
            packed[2],
            packed[3],
            packed[4],
            policy=runtime.policy,
        )
        values: list[torch.Tensor] = []
        gradients: list[dict[str, torch.Tensor]] = []
        details: list[Mapping[str, Any]] = []
        paired_rng = capture_rng(runtime.context) if count == 2 else None
        post_correct_rng: dict[str, Any] | None = None
        for index in range(count):
            if index == 1:
                restore_rng(paired_rng, runtime.context)  # type: ignore[arg-type]
            value, detail, gradient = _functional_arm_gradient(
                runtime,
                adapter_state_at(generated, index, count),
                policy_batch,
            )
            values.append(value)
            details.append(detail)
            gradients.append(gradient)
            if index == 0 and count == 2:
                post_correct_rng = capture_rng(runtime.context)
        if count == 1:
            coefficient = float(
                runtime.config["conditioning_training"]["normal_loss_weight"]
            )
            loss = coefficient * values[0]
            coefficients = (
                torch.as_tensor(coefficient, device=values[0].device),
            )
            probability = None
        else:
            post_wrong_rng = capture_rng(runtime.context)
            restore_rng(post_correct_rng, runtime.context)  # type: ignore[arg-type]
            if not same_torch_rng(post_correct_rng, post_wrong_rng):  # type: ignore[arg-type]
                raise WriterModelError("paired contrast policy RNG consumption diverged")
            loss, coefficients, probability = matching_objective(
                (values[0], values[1]),
                runtime.config["conditioning_training"],
            )
    names = tuple(generated)
    if count == 1:
        gradient_tensors = tuple(
            gradient_scale
            * coefficients[0].to(gradients[0][name])
            * gradients[0][name]
            for name in names
        )
    else:
        gradient_tensors = tuple(
            gradient_scale
            * torch.stack(
                [
                    coefficients[index].to(gradients[index][name])
                    * gradients[index][name]
                    for index in range(count)
                ],
                dim=0,
            )
            for name in names
        )
    torch.autograd.backward(
        tuple(generated[name] for name in names),
        gradient_tensors,
    )
    return loss, values, details, probability


def _cumulative_counts(
    runtime: WriterRuntime,
    completed: int,
) -> tuple[int, int, int]:
    conditions = runtime.conditions_per_optimizer_step
    data_steps = completed * conditions
    unique = sum(
        runtime.sampler.batch_size_for_step(step) for step in range(data_steps)
    )
    cycle = conditioning_cycle(runtime.config)
    generated = sum(
        1 if cycle[step % len(cycle)] == "normal" else 2
        for step in range(data_steps)
    )
    scale = runtime.context.world_size
    return (
        unique * scale,
        data_steps * runtime.batch_size * scale,
        generated * scale,
    )


def _pack_raw_conditions(
    runtime: WriterRuntime,
    *,
    task_id: int,
    correct_demo_index: int,
    mode: str,
    wrong_task_id: int | None,
    wrong_demo_index: int | None,
) -> tuple[
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    dict[str, int | None],
]:
    correct = runtime.video_store.load(task_id, correct_demo_index)
    videos = [correct]
    if mode != "normal":
        if wrong_task_id is None or wrong_demo_index is None:
            raise WriterModelError("contrast step lost its wrong teaching video")
        videos.append(runtime.video_store.load(wrong_task_id, wrong_demo_index))
    language, mask = (
        runtime.generic_language
        if mode == "generic_language_contrast"
        else runtime.language_tokens[task_id]
    )
    if mode == "normal":
        tokens = language
        masks = mask
    elif mode in {"full_language_contrast", "generic_language_contrast"}:
        tokens = torch.cat((language, language), dim=0)
        masks = torch.cat((mask, mask), dim=0)
    else:
        raise WriterModelError(f"unsupported AS-Writer conditioning mode: {mode}")

    frames = torch.cat(
        [torch.from_numpy(video.frames) for video in videos],
        dim=0,
    ).to(runtime.context.device, non_blocking=True)
    indices = torch.cat(
        [torch.from_numpy(video.frame_indices) for video in videos],
        dim=0,
    ).to(runtime.context.device, non_blocking=True)
    lengths = [int(video.frames.shape[0]) for video in videos]
    offsets = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.long,
        device=runtime.context.device,
    )
    return (
        frames,
        indices,
        offsets,
        tokens,
        masks,
    ), {
        "correct_video_raw_frames": correct.raw_frame_count,
        "correct_video_sampled_frames": lengths[0],
        "wrong_video_raw_frames": (
            videos[1].raw_frame_count if len(videos) == 2 else None
        ),
        "wrong_video_sampled_frames": lengths[1] if len(videos) == 2 else None,
    }


def _condition_metric_summary(
    runtime: WriterRuntime,
    conditions: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(conditions) != runtime.conditions_per_optimizer_step:
        raise WriterModelError("AS-Writer optimizer update lost an independent condition")
    modes = [str(condition["mode"]) for condition in conditions]
    positive_values = [
        float(condition["values"][0])  # type: ignore[index]
        for condition in conditions
    ]
    wrong_values = [
        float(condition["values"][1])  # type: ignore[index]
        for condition in conditions
        if len(condition["values"]) == 2  # type: ignore[arg-type]
    ]
    probabilities = [
        float(condition["matching_probability"])
        for condition in conditions
        if condition["matching_probability"] is not None
    ]
    observed_queries = sum(int(condition["observed_batch"]) for condition in conditions)
    policy_samples_this_step = (
        runtime.context.world_size
        * runtime.batch_size
        * runtime.conditions_per_optimizer_step
    )
    generated_conditions_this_step = runtime.context.world_size * sum(
        1 if mode == "normal" else 2 for mode in modes
    )
    mean_loss = sum(float(condition["loss"]) for condition in conditions) / len(
        conditions
    )
    return {
        "modes": modes,
        "mean_loss": mean_loss,
        "mean_positive": sum(positive_values) / len(positive_values),
        "mean_wrong": (
            sum(wrong_values) / len(wrong_values) if wrong_values else None
        ),
        "mean_probability": (
            sum(probabilities) / len(probabilities) if probabilities else None
        ),
        "observed_queries": observed_queries,
        "policy_samples_this_step": policy_samples_this_step,
        "generated_conditions_this_step": generated_conditions_this_step,
        "data_seconds": sum(
            float(condition["data_seconds"]) for condition in conditions
        ),
        "first": conditions[0],
        "rank0_conditions": [
            {
                "data_step": int(condition["data_step"]),
                "mode": str(condition["mode"]),
                "task_id": int(condition["task_id"]),
                "teacher_demo_index": int(condition["demo_index"]),
                "wrong_video_task_id": condition["partner_id"],
                "wrong_teacher_demo_index": condition["wrong_demo_index"],
                **dict(condition["video_metrics"]),  # type: ignore[arg-type]
            }
            for condition in conditions
        ],
        "rank0_policy_loss_detail": [
            [value.get("loss") for value in condition["details"]]  # type: ignore[union-attr]
            for condition in conditions
        ],
        "policy_forward_calls": sum(
            int(value.get("policy_forward_calls", 1))
            for condition in conditions
            for value in condition["details"]  # type: ignore[union-attr]
        ),
    }


def _step_metrics(
    runtime: WriterRuntime,
    *,
    step: int,
    started: float,
    tick: float,
    conditions: list[Mapping[str, Any]],
    grad_norm: torch.Tensor,
    applied_lr: float,
) -> dict[str, Any]:
    summary = _condition_metric_summary(runtime, conditions)
    modes = summary["modes"]
    completed = step + 1
    step_seconds = reduce_max(time.monotonic() - tick, runtime.context)
    unique_queries, policy_samples, writer_conditions = _cumulative_counts(
        runtime,
        completed,
    )
    first = summary["first"]
    return {
        "optimizer_step": completed,
        "conditioning_mode": modes[0] if len(set(modes)) == 1 else "mixed",
        "conditioning_modes": modes,
        "independent_conditions_per_optimizer_step": len(conditions),
        "writer_language_condition": (
            "generic_neutral"
            if set(modes) == {"generic_language_contrast"}
            else "mixed"
            if "generic_language_contrast" in modes
            else "task_language"
        ),
        "policy_language_condition": "correct_action_query_task_language",
        "mean_functional_action_loss": reduce_mean(
            summary["mean_loss"], runtime.context
        ),
        "mean_positive_action_loss": reduce_mean(
            summary["mean_positive"], runtime.context
        ),
        "mean_wrong_video_action_loss": (
            reduce_mean(summary["mean_wrong"], runtime.context)
            if summary["mean_wrong"] is not None
            else None
        ),
        "mean_matching_probability": (
            reduce_mean(summary["mean_probability"], runtime.context)
            if summary["mean_probability"] is not None
            else None
        ),
        "gradient_norm_before_clip_max": reduce_max(float(grad_norm), runtime.context),
        "applied_lr": applied_lr,
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "global_unique_action_queries": unique_queries,
        "global_policy_samples": policy_samples,
        "global_writer_conditions": writer_conditions,
        "global_unique_action_queries_this_step": (
            summary["observed_queries"] * runtime.context.world_size
        ),
        "global_policy_samples_this_step": summary["policy_samples_this_step"],
        "global_writer_conditions_this_step": summary[
            "generated_conditions_this_step"
        ],
        "rank0_task_id": int(first["task_id"]),
        "rank0_teacher_demo_index": int(first["demo_index"]),
        "rank0_wrong_video_task_id": first["partner_id"],
        "rank0_wrong_teacher_demo_index": first["wrong_demo_index"],
        "rank0_conditions": summary["rank0_conditions"],
        "rank0_policy_loss_detail": summary["rank0_policy_loss_detail"],
        "functional_policy_microbatch_size": int(
            runtime.config["conditioning_training"][
                "functional_policy_microbatch_size"
            ]
        ),
        "policy_forward_calls_this_step": summary["policy_forward_calls"],
        "data_seconds_max": reduce_max(summary["data_seconds"], runtime.context),
        "step_seconds_max": step_seconds,
        "global_policy_samples_per_second": (
            summary["policy_samples_this_step"] / step_seconds
        ),
        "global_unique_action_queries_per_second": (
            runtime.context.world_size
            * summary["observed_queries"]
            / step_seconds
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
    """Run one task-balanced functional AS-Writer update."""

    tick = time.monotonic()
    cycle = conditioning_cycle(runtime.config)
    count = runtime.conditions_per_optimizer_step
    runtime.optimizer.zero_grad(set_to_none=True)
    conditions: list[dict[str, Any]] = []
    for condition_index in range(count):
        data_tick = time.monotonic()
        batch = next(runtime.iterator)
        data_seconds = time.monotonic() - data_tick
        data_step = step * count + condition_index
        mode = cycle[data_step % len(cycle)]
        task_id, task_visit = runtime.sampler.task_visit_for_step(data_step)
        if _batch_task_id(batch) != task_id:
            raise WriterModelError("AS-Writer sampler and action batch disagree")
        observed_batch = int(batch["task_id"].shape[0])
        expected_batch = runtime.sampler.batch_size_for_step(data_step)
        if observed_batch != expected_batch:
            raise WriterModelError(
                "AS-Writer conditioning and sampler batch sizes disagree"
            )
        demo_index = runtime.video_schedule.demo_for_task_visit(task_id, task_visit)
        partner_id: int | None = None
        wrong_demo_index: int | None = None
        if mode != "normal":
            partner_id = runtime.video_partner[task_id]
            wrong_demo_index = runtime.video_schedule.demo_for_task_visit(
                partner_id,
                task_visit,
            )
        packed, video_metrics = _pack_raw_conditions(
            runtime,
            task_id=task_id,
            correct_demo_index=demo_index,
            mode=mode,
            wrong_task_id=partner_id,
            wrong_demo_index=wrong_demo_index,
        )
        policy_batch = runtime.processor.training_batch(batch)
        loss, values, details, matching_probability = _differentiate_condition_batch(
            runtime,
            packed,
            policy_batch,
            mode,
            gradient_scale=1.0 / count,
        )
        if not bool(torch.isfinite(loss)):
            raise WriterModelError(
                f"non-finite AS-Writer loss at step {step}, condition {condition_index}"
            )
        conditions.append(
            {
                "data_step": data_step,
                "data_seconds": data_seconds,
                "mode": mode,
                "task_id": task_id,
                "demo_index": demo_index,
                "partner_id": partner_id,
                "wrong_demo_index": wrong_demo_index,
                "observed_batch": observed_batch,
                "loss": loss,
                "values": values,
                "details": details,
                "matching_probability": matching_probability,
                "video_metrics": video_metrics,
            }
        )
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
        conditions=conditions,
        grad_norm=grad_norm,
        applied_lr=applied_lr,
    )
