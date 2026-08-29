"""One role-balanced optimizer update for G2 Natural Program training."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import torch

from ember.ecp.behavior.kernel import (
    distributed_behavior_kernel_loss,
    program_behavior_features,
)
from ember.ecp.natural_program import NaturalProgramOutput
from ember.ecp.natural_program_data import (
    NaturalProgramTask,
    PackedNaturalProgramCondition,
    pack_natural_program_condition,
)
from ember.ecp.natural_program_objective import (
    NaturalProgramLoss,
    natural_program_loss,
)
from ember.ecp.stage0_train_step import _gather_records, _reduce_gradients

if TYPE_CHECKING:
    from ember.ecp.natural_program_training import NaturalProgramRuntime


def _forward(
    runtime: NaturalProgramRuntime,
    batch: PackedNaturalProgramCondition,
    task_id: int,
) -> NaturalProgramOutput:
    language_tokens, language_mask = runtime.language_tokens[task_id]
    return runtime.model(
        policy=runtime.policy,
        frames=batch.frames,
        frame_indices=batch.frame_indices,
        raw_frame_counts=batch.raw_frame_counts,
        video_offsets=batch.video_offsets,
        video_set_offsets=batch.video_set_offsets,
        frame_condition_ids=batch.frame_condition_ids,
        language_tokens=language_tokens,
        language_mask=language_mask,
        query_times=batch.query_times,
    )


def _negative_language_embeddings(
    runtime: NaturalProgramRuntime, *, task_id: int, macro: int
) -> torch.Tensor:
    count = int(runtime.config["objective"]["contrastive_negative_languages"])
    negative_ids = runtime.schedule.contrastive_task_ids(
        task_id, macro, count=count
    )
    tokens = torch.cat(
        [runtime.language_tokens[index][0] for index in negative_ids]
    )
    masks = torch.cat(
        [runtime.language_tokens[index][1] for index in negative_ids]
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        source = runtime.model.encoder.embed_language_conditions(
            runtime.policy, tokens
        )
        p_lang = runtime.model.language_reader(source, masks)
    return torch.nn.functional.normalize(p_lang.float().mean(1), dim=-1)


def _task_record(
    task: NaturalProgramTask,
    batch: PackedNaturalProgramCondition,
    output: NaturalProgramOutput,
    loss: NaturalProgramLoss,
) -> dict[str, Any]:
    active = (output.program.rho.detach().float() > 0.5).sum(-1)
    return {
        "authority_id": task.authority_id,
        "domain": task.domain,
        "domain_task_id": task.domain_task_id,
        "role": task.role,
        **{
            name: float(getattr(loss, name).detach())
            for name in NaturalProgramLoss.__dataclass_fields__
        },
        "mean_active_events": float(active.float().mean()),
        "one_event_fraction": float((active <= 1).float().mean()),
        "mean_presence_sum": float(output.program.rho.detach().float().sum(-1).mean()),
        "mean_cross_video_sigma": float(output.program.sigma.detach().float().mean()),
        **batch.metrics,
    }


def _run_task_step(
    runtime: NaturalProgramRuntime,
    *,
    task_id: int,
    macro: int,
    global_task_count: int,
) -> tuple[dict[str, Any], NaturalProgramLoss, torch.Tensor]:
    task = runtime.task_by_id[task_id]
    sample = runtime.schedule.sample(task_id, macro)
    second_sample = runtime.schedule.disjoint_sample(task_id, macro, sample)
    pack = {
        "task": task,
        "sample": sample,
        "video_store": runtime.video_store,
        "action_store": runtime.action_store,
        "label_store": runtime.label_store,
        "query_points": int(runtime.config["data"]["query_points"]),
        "predicate_slots": int(runtime.config["model"]["predicate_slots"]),
        "device": runtime.context.device,
    }
    batch = pack_natural_program_condition(**pack)
    second_pack = {**pack, "sample": second_sample}
    robust_batch = pack_natural_program_condition(
        **second_pack, view="full"
    )
    negatives = _negative_language_embeddings(
        runtime, task_id=task_id, macro=macro
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = _forward(runtime, batch, task_id)
        robust_output = _forward(runtime, robust_batch, task_id)
        loss = natural_program_loss(
            output,
            batch,
            weights=runtime.config["objective"]["weights"],
            robust_output=robust_output,
            negative_embeddings=negatives,
            contrastive_temperature=float(
                runtime.config["objective"]["contrastive_temperature"]
            ),
        )
    if not bool(torch.isfinite(loss.total)):
        raise RuntimeError(f"non-finite G2 loss at macro {macro}, task {task_id}")
    if runtime.behavior_codes is None:
        raise RuntimeError("G2 behavior-kernel authority is missing")
    features = torch.stack(
        (
            program_behavior_features(
                output.program, runtime.behavior_codes.selected_targets
            )[0],
            program_behavior_features(
                robust_output.program, runtime.behavior_codes.selected_targets
            )[0],
        )
    )
    return _task_record(task, batch, output, loss), loss, features


def run_natural_program_optimizer_step(
    runtime: NaturalProgramRuntime,
    *,
    macro: int,
    assignments: tuple[tuple[int, ...], ...],
) -> tuple[list[dict[str, Any]], dict[str, Any], tuple[int, ...]]:
    tick = time.monotonic()
    task_ids = assignments[runtime.context.rank]
    global_task_count = sum(map(len, assignments))
    if global_task_count <= 0:
        raise RuntimeError("G2 optimizer step lost every task")
    runtime.optimizer.zero_grad(set_to_none=True)
    task_outputs = [
        _run_task_step(
            runtime,
            task_id=task_id,
            macro=macro,
            global_task_count=global_task_count,
        )
        for task_id in task_ids
    ]
    records = [row[0] for row in task_outputs]
    base_loss = sum(row[1].total for row in task_outputs) / float(global_task_count)
    local_features = torch.stack([row[2] for row in task_outputs])
    local_task_ids = torch.tensor(
        task_ids, dtype=torch.long, device=runtime.context.device
    )
    kernel_loss, kernel_metrics = distributed_behavior_kernel_loss(
        local_features=local_features,
        local_task_ids=local_task_ids,
        authority=runtime.behavior_codes,
        world_size=runtime.context.world_size,
        cross_view_weight=float(
            runtime.config["behavior_alignment"]["cross_view_weight"]
        ),
    )
    kernel_weight = float(
        runtime.config["objective"]["weights"]["behavior_alignment"]
    )
    kernel_owner = 1.0 if runtime.context.rank == 0 else 0.0
    kernel_objective = kernel_owner * kernel_weight * kernel_loss
    kernel_parameters = (
        runtime.model.language_reader.key.weight,
        runtime.model.scene_reader.key.weight,
        runtime.model.process_fusion[0].weight,
    )
    kernel_gradients = torch.autograd.grad(
        kernel_objective,
        kernel_parameters,
        retain_graph=True,
        allow_unused=False,
    )
    kernel_gradient_norm = float(
        torch.stack([value.float().norm() for value in kernel_gradients]).norm()
    )
    (base_loss + kernel_objective).backward()
    for record in records:
        record["behavior_alignment"] = float(kernel_loss.detach())
        record["total"] += kernel_weight * float(kernel_loss.detach())
    if any(parameter.grad is not None for parameter in runtime.frozen_parameters):
        raise RuntimeError("frozen G2 source policy accumulated gradients")
    _reduce_gradients(runtime.trainable_parameters, runtime.context.world_size)
    owner_gradient = runtime.model.decoder.owner_queries.grad
    if owner_gradient is None or not bool(torch.isfinite(owner_gradient).all()):
        raise RuntimeError("G2 owner-specific temporal readout lost its gradient")
    owner_gradient_norm = float(owner_gradient.float().norm())
    if owner_gradient_norm <= 0.0:
        raise RuntimeError("G2 owner-specific temporal readout has zero gradient")
    program_gradient = runtime.model.process_fusion[0].weight.grad
    if (
        program_gradient is None
        or not bool(torch.isfinite(program_gradient).all())
        or not bool(torch.isfinite(torch.tensor(kernel_gradient_norm)))
        or kernel_gradient_norm <= 0.0
    ):
        raise RuntimeError("G2 behavior kernel lost its finite Program gradient")
    behavior_program_gradient_norm = float(program_gradient.float().norm())
    clip = float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"])
    gradient_norm = torch.nn.utils.clip_grad_norm_(runtime.trainable_parameters, clip)
    if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError(f"non-finite G2 gradient at macro {macro}")
    runtime.optimizer.step()
    runtime.scheduler.step()
    runtime.optimizer_steps += 1
    global_records = sorted(
        _gather_records(records, runtime.context.world_size),
        key=lambda row: int(row["authority_id"]),
    )
    role_counts = {
        role: sum(row["role"] == role for row in global_records)
        for role in ("meta_fit", "target_fit")
    }
    if len(global_records) != global_task_count or (
        runtime.tasks_per_rank is None
        and role_counts["meta_fit"] != role_counts["target_fit"]
    ):
        raise RuntimeError("G2 optimizer step lost task-equal role balance")
    summary = {
        "optimizer_step": runtime.optimizer_steps,
        "global_task_count": global_task_count,
        "role_counts": role_counts,
        "owner_query_gradient_norm_before_clip": owner_gradient_norm,
        "behavior_kernel_gradient_norm": kernel_gradient_norm,
        "behavior_program_gradient_norm_before_clip": behavior_program_gradient_norm,
        **kernel_metrics,
        "gradient_norm_before_clip": float(gradient_norm),
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "next_lrs": [
            float(group["lr"]) for group in runtime.optimizer.param_groups
        ],
        "step_seconds": time.monotonic() - tick,
    }
    return global_records, summary, task_ids
