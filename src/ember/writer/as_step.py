"""One full-task AS update for the condition-kernel Program Memory Writer."""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Mapping

import torch
import torch.distributed as dist

from ember.pi05_source_setup import reduce_max
from ember.rl_writer.rendezvous import rank_local_credit_ready
from ember.writer.condition_kernel import (
    apply_program_value_delta,
    kernel_corrected_value_delta,
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
]


def _batch_task_id(batch: Mapping[str, Any]) -> int:
    values = batch.get("task_id")
    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise WriterModelError("AS-Writer action batch lost task identity")
    unique = values.unique()
    if unique.numel() != 1:
        raise WriterModelError("one AS-Writer rank received multiple tasks")
    return int(unique.item())


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
        runtime.context.device, non_blocking=True
    )
    indices = torch.from_numpy(video.frame_indices).to(
        runtime.context.device, non_blocking=True
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
        "actions_per_video": action_batch_size,
        "teacher_video_raw_frames": video.raw_frame_count,
        "teacher_video_sampled_frames": int(video.frames.shape[0]),
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


def _factor_decoder_active(runtime: WriterRuntime, step: int) -> bool:
    stop = int(
        runtime.config["conditioning_training"][
            "factor_decoder_train_through_macro"
        ]
    )
    if stop <= 0:
        raise WriterModelError("invalid FactorHead training boundary")
    return step < stop


def _differentiate_condition(
    runtime: WriterRuntime,
    packed: WriterCondition,
    policy_batch: Mapping[str, Any],
    flat_factor_gradient: torch.Tensor,
    policy_rng_seed: int | None,
    *,
    factor_active: bool,
) -> tuple[
    torch.Tensor,
    Mapping[str, Any],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return loss, fixed feature, Program cotangent, and FactorHead gradient."""

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        feature, stored_program = runtime.writer.encode_condition(
            *packed, policy=runtime.policy
        )
        # Memory is explicitly updated, so Program is the differentiable leaf
        # shared by AS functional loss and later reward credit.
        program = stored_program.detach().to(torch.float32).requires_grad_(True)
        generated = runtime.writer.decode_program(program)
        loss, detail, lora_gradients = functional_lora_loss_gradient(
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
    factor_parameters = tuple(
        item.parameter for item in runtime.gradient_layout
    ) if factor_active else ()
    differentiated = torch.autograd.grad(
        tuple(generated[name] for name in names),
        (program, *factor_parameters),
        grad_outputs=tuple(lora_gradients[name] for name in names),
        allow_unused=True,
    )
    program_cotangent = differentiated[0]
    if program_cotangent is None:
        raise WriterModelError("functional loss did not reach the policy Program")
    if factor_active:
        flat_factor_gradient = flatten_task_gradient(
            differentiated[1:],
            runtime.gradient_layout,
            output=flat_factor_gradient,
        )
    else:
        flat_factor_gradient.zero_()
    feature = feature.detach().to(torch.float32)
    program_cotangent = program_cotangent.detach().to(torch.float32)
    if (
        feature.shape != (1, runtime.writer.condition_feature.feature_width)
        or program_cotangent.shape
        != (
            1,
            runtime.writer.PROGRAM_SLOTS,
            runtime.writer.program_width,
        )
        or not bool(torch.isfinite(program_cotangent).all())
    ):
        raise WriterModelError("condition Program credit changed shape")
    return (
        loss,
        detail,
        flat_factor_gradient,
        feature[0],
        program_cotangent[0],
    )


def _collect_local_credit(
    runtime: WriterRuntime,
    step: int,
    *,
    factor_active: bool,
) -> tuple[
    list[dict[str, Any]],
    float,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    records: list[dict[str, Any]] = []
    data_seconds = 0.0
    count = runtime.tasks_per_rank_per_update
    local_ids = torch.empty(count, dtype=torch.long, device=runtime.context.device)
    local_features = torch.empty(
        count,
        runtime.writer.condition_feature.feature_width,
        dtype=torch.float32,
        device=runtime.context.device,
    )
    local_cotangents = torch.empty(
        count,
        runtime.writer.PROGRAM_SLOTS,
        runtime.writer.program_width,
        dtype=torch.float32,
        device=runtime.context.device,
    )
    local_factor_gradients = torch.empty(
        count,
        runtime.gradient_layout[-1].stop,
        dtype=torch.float32,
        device=runtime.context.device,
    )
    for microtask in range(count):
        data_tick = time.monotonic()
        batch = next(runtime.iterator)
        data_seconds += time.monotonic() - data_tick
        task_id, task_visit = runtime.sampler.task_visit_for_step(step, microtask)
        if _batch_task_id(batch) != task_id:
            raise WriterModelError("AS-Writer sampler and action batch disagree")
        observed_batch = int(batch["task_id"].shape[0])
        if observed_batch != runtime.sampler.batch_size_for_step(step):
            raise WriterModelError("AS-Writer action-query batch size changed")
        teacher_demo = runtime.video_schedule.demo_for_task_visit(
            task_id, task_visit
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
        loss, detail, _, feature, cotangent = _differentiate_condition(
            runtime,
            packed,
            runtime.processor.training_batch(batch),
            local_factor_gradients[microtask],
            policy_rng_seed,
            factor_active=factor_active,
        )
        if not bool(torch.isfinite(loss)):
            raise WriterModelError(
                f"non-finite AS-Writer loss at step {step}, microtask {microtask}"
            )
        local_ids[microtask] = task_id
        local_features[microtask].copy_(feature)
        local_cotangents[microtask].copy_(cotangent)
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
    return (
        records,
        data_seconds,
        local_ids,
        local_features,
        local_cotangents,
        local_factor_gradients,
    )


def _all_gather_rows(local: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return local
    gathered = torch.empty(
        world_size * local.shape[0],
        *local.shape[1:],
        dtype=local.dtype,
        device=local.device,
    )
    dist.all_gather_into_tensor(gathered, local.contiguous())
    return gathered


def _cotangent_metrics(
    task_ids: torch.Tensor,
    features: torch.Tensor,
    cotangents: torch.Tensor,
    losses: torch.Tensor,
    update_summary: Mapping[str, Any],
) -> dict[str, Any]:
    flat = cotangents.flatten(1)
    gram = flat @ flat.transpose(0, 1)
    diagonal = gram.diagonal().clamp_min(0.0)
    mean_energy = gram.mean().clamp_min(0.0)
    average_energy = diagonal.mean()
    candidate_dots = gram.mean(dim=1)
    feature_gram = features @ features.transpose(0, 1)
    denominator = torch.sqrt(
        diagonal[:, None] * diagonal[None]
    ).clamp_min(torch.finfo(torch.float32).tiny)
    cosine = gram / denominator
    off_diagonal = ~torch.eye(
        gram.shape[0], dtype=torch.bool, device=gram.device
    )
    rows = []
    for index, task_id in enumerate(task_ids.detach().cpu().tolist()):
        rows.append(
            {
                "task_id": int(task_id),
                "functional_action_loss": float(losses[index]),
                "program_cotangent_norm": math.sqrt(float(diagonal[index])),
                "program_cotangent_dot_mean_direction": float(
                    candidate_dots[index]
                ),
                "condition_self_kernel": float(feature_gram[index, index]),
            }
        )
    return {
        "schema_version": "ember_condition_kernel_program_credit_v1",
        "task_ids": [int(value) for value in task_ids.detach().cpu().tolist()],
        "raw_gradient_gram": gram.detach().cpu().tolist(),
        "raw_candidate_task_dots": candidate_dots.detach().cpu().tolist(),
        "raw_mean_gradient_norm": math.sqrt(float(mean_energy)),
        "average_task_gradient_energy": float(average_energy),
        "raw_mean_to_average_task_energy_ratio": (
            float(mean_energy / average_energy) if float(average_energy) > 0 else 0.0
        ),
        "negative_pair_fraction": float(
            (cosine[off_diagonal] < 0).to(torch.float32).mean()
        ),
        "median_pair_cosine": float(cosine[off_diagonal].median()),
        "condition_feature_gram": feature_gram.detach().cpu().tolist(),
        "kernel_update": dict(update_summary),
        "per_task": rows,
    }


def _gather_and_update_program_memory(
    runtime: WriterRuntime,
    *,
    records: list[dict[str, Any]],
    local_task_ids: torch.Tensor,
    local_features: torch.Tensor,
    local_cotangents: torch.Tensor,
) -> dict[str, Any]:
    losses = torch.tensor(
        [float(record["loss"]) for record in records],
        dtype=torch.float32,
        device=runtime.context.device,
    )
    gathered_ids = _all_gather_rows(local_task_ids, runtime.context.world_size)
    gathered_features = _all_gather_rows(
        torch.cat((local_features, losses[:, None]), dim=1),
        runtime.context.world_size,
    )
    gathered_cotangents = _all_gather_rows(
        local_cotangents, runtime.context.world_size
    )
    order = torch.argsort(gathered_ids)
    task_ids = gathered_ids.index_select(0, order)
    expected = torch.tensor(
        sorted(int(value) for value in runtime.task_ids),
        dtype=torch.long,
        device=task_ids.device,
    )
    if not torch.equal(task_ids, expected):
        raise WriterModelError("condition-kernel full-task coverage changed")
    feature_loss = gathered_features.index_select(0, order)
    features = feature_loss[:, :-1]
    losses = feature_loss[:, -1]
    cotangents = gathered_cotangents.index_select(0, order)
    update = runtime.config["optimization"]["program_memory_update"]
    delta, summary = kernel_corrected_value_delta(
        features,
        cotangents,
        step_size=float(update["step_size"]),
        relative_damping=float(update["relative_damping"]),
        induced_update_rms_cap=float(update["induced_program_rms_cap"]),
    )
    observed_delta, application = apply_program_value_delta(
        runtime.writer.program_memory, delta, features
    )
    observed_flat = observed_delta.flatten(1)
    observed_gram = observed_flat @ observed_flat.transpose(0, 1)
    observed_energy = observed_gram.diagonal().clamp_min(0.0)
    observed_denominator = torch.sqrt(
        observed_energy[:, None] * observed_energy[None]
    ).clamp_min(torch.finfo(torch.float32).tiny)
    observed_cosine = observed_gram / observed_denominator
    off_diagonal = ~torch.eye(
        observed_gram.shape[0], dtype=torch.bool, device=observed_gram.device
    )
    mean_observed_energy = observed_gram.mean().clamp_min(0.0)
    average_observed_energy = observed_energy.mean()
    summary_payload = {
        **asdict(summary),
        **asdict(application),
        "value_delta_rms": float(delta.square().mean().sqrt()),
        "observed_task_delta_gram": observed_gram.detach().cpu().tolist(),
        "observed_task_delta_median_pair_cosine": float(
            observed_cosine[off_diagonal].median()
        ),
        "observed_task_delta_negative_pair_fraction": float(
            (observed_cosine[off_diagonal] < 0).to(torch.float32).mean()
        ),
        "observed_task_delta_mean_to_average_energy_ratio": (
            float(mean_observed_energy / average_observed_energy)
            if float(average_observed_energy) > 0
            else 0.0
        ),
        "collectives": 3 if runtime.context.world_size > 1 else 0,
    }
    return _cotangent_metrics(
        task_ids, features, cotangents, losses, summary_payload
    )


def _global_task_video_assignments(
    runtime: WriterRuntime, step: int
) -> list[dict[str, int]]:
    result = []
    for rank, microtask, task_id, task_visit in runtime.sampler.assignments_for_step(
        step
    ):
        teacher_demo = runtime.video_schedule.demo_for_task_visit(
            task_id, task_visit
        )
        result.append(
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
    return result


def _step_metrics(
    runtime: WriterRuntime,
    *,
    step: int,
    started: float,
    tick: float,
    data_seconds: float,
    records: list[dict[str, Any]],
    factor_active: bool,
    factor_grad_norm: float,
    factor_metrics: Mapping[str, Any] | None,
    optimizer_update: Mapping[str, Any],
    optimizer_diagnostics: Mapping[str, Any] | None,
    scheduler_advanced: bool,
    program_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    completed = step + 1
    step_seconds = reduce_max(time.monotonic() - tick, runtime.context)
    global_tasks = len(runtime.task_ids)
    global_queries = global_tasks * runtime.batch_size
    decoder = runtime.config["optimization"]["factor_decoder_optimizer"]
    return {
        "optimizer_step": completed,
        "macro_optimizer_update": completed,
        "completed_task_cycles": completed,
        "conditioning_mode": runtime.config["conditioning_training"]["method"],
        "writer_language_condition": "correct_task_language",
        "policy_language_condition": "correct_action_query_task_language",
        "mean_functional_action_loss": sum(
            float(row["functional_action_loss"])
            for row in program_metrics["per_task"]
        )
        / global_tasks,
        "program_memory_update": dict(program_metrics["kernel_update"]),
        "program_credit": dict(program_metrics),
        "factor_decoder_update_active": factor_active,
        "factor_decoder_frozen": not factor_active,
        "factor_gradient_norm_before_clip": factor_grad_norm,
        "factor_gradient_clip_norm": float(decoder["gradient_clip_norm"]),
        "factor_gradient_clip_triggered": (
            factor_active
            and factor_grad_norm > float(decoder["gradient_clip_norm"])
        ),
        "factor_gradient_composition": (
            dict(factor_metrics) if factor_metrics is not None else None
        ),
        "factor_optimizer_update": dict(optimizer_update),
        "factor_optimizer_diagnostics": (
            dict(optimizer_diagnostics)
            if optimizer_diagnostics is not None
            else None
        ),
        "scheduler_advanced_this_update": scheduler_advanced,
        "next_factor_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "policy_randomness_scheme": runtime.config["conditioning_training"].get(
            "policy_randomness_scheme"
        ),
        "global_unique_action_queries": completed * global_queries,
        "global_policy_samples": completed * global_queries,
        "global_writer_video_conditions": completed * global_tasks,
        "global_unique_action_queries_this_step": global_queries,
        "global_writer_video_conditions_this_step": global_tasks,
        "functional_policy_microbatch_size": int(
            runtime.config["optimization"]["functional_policy_microbatch_size"]
        ),
        "step_seconds_max": step_seconds,
        "data_seconds_max": reduce_max(data_seconds, runtime.context),
        "macro_updates_per_hour": 3600.0 / step_seconds,
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
        "global_task_video_assignments": _global_task_video_assignments(
            runtime, step
        ),
        "rank0_task_ids": [int(record["task_id"]) for record in records],
        "rank0_conditions": [
            {
                "task_id": int(record["task_id"]),
                **dict(record["video_metrics"]),
            }
            for record in records
        ],
        "action_query_batch_size_per_task": runtime.batch_size,
        "tasks_per_rank_per_optimizer_update": (
            runtime.tasks_per_rank_per_update
        ),
        "teacher_videos_per_task_visit": runtime.videos_per_task_visit,
        "source_policy_trainable_parameter_count": 0,
    }


def run_writer_step(
    runtime: WriterRuntime,
    step: int,
    started: float,
) -> dict[str, Any]:
    """Apply one full24 Program-Memory update and the optional head update."""

    tick = time.monotonic()
    factor_active = _factor_decoder_active(runtime, step)
    runtime.optimizer.zero_grad(set_to_none=True)
    (
        records,
        data_seconds,
        local_task_ids,
        local_features,
        local_cotangents,
        local_factor_gradients,
    ) = _collect_local_credit(runtime, step, factor_active=factor_active)
    if any(parameter.grad is not None for parameter in runtime.policy.parameters()):
        raise WriterModelError("frozen PI05 source policy accumulated gradients")

    # Explicit CUDA completion plus launch-unique atomic rank markers prevents
    # a short-video rank from enqueueing NCCL while another rank is still in
    # the frozen policy/functional backward path.
    rank_local_credit_ready(runtime, cycle=step, epoch=0)

    factor_metrics: Mapping[str, Any] | None = None
    factor_grad_norm = 0.0
    optimizer_diagnostics: Mapping[str, Any] | None = None
    scheduler_advanced = False
    optimizer_update: Mapping[str, Any] = {
        "mode": "factor_decoder_frozen",
        "logical_lr": float(runtime.scheduler.get_last_lr()[0]),
        "applied_lr": 0.0,
        "lr_divisor": 1,
        "reference_weight_decay": 0.0,
        "applied_weight_decay": 0.0,
    }
    if factor_active:
        direction, factor_metrics = compose_distributed_raw_mean_gradient(
            local_task_ids,
            local_factor_gradients,
            runtime.gradient_layout,
            expected_task_ids=runtime.task_ids,
            world_size=runtime.context.world_size,
            rank=runtime.context.rank,
        )
        assign_flat_gradient(direction, runtime.gradient_layout)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            runtime.writer.factor_heads.parameters(),
            float(
                runtime.config["optimization"]["factor_decoder_optimizer"][
                    "gradient_clip_norm"
                ]
            ),
        )
        if not bool(torch.isfinite(grad_norm).detach()):
            raise WriterModelError(
                f"non-finite FactorHead gradient at step {step}"
            )
        factor_grad_norm = float(grad_norm)
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

    program_metrics = _gather_and_update_program_memory(
        runtime,
        records=records,
        local_task_ids=local_task_ids,
        local_features=local_features,
        local_cotangents=local_cotangents,
    )
    return _step_metrics(
        runtime,
        step=step,
        started=started,
        tick=tick,
        data_seconds=data_seconds,
        records=records,
        factor_active=factor_active,
        factor_grad_norm=factor_grad_norm,
        factor_metrics=factor_metrics,
        optimizer_update=optimizer_update,
        optimizer_diagnostics=optimizer_diagnostics,
        scheduler_advanced=scheduler_advanced,
        program_metrics=program_metrics,
    )
