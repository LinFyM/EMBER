"""One fixed six-task optimizer update for G3 mapping acquisition."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.ecp.natural_program_data import NaturalProgramSample
from ember.ecp.shared_compiler_data import (
    pack_shared_compiler_videos,
    prepare_shared_compiler_condition,
    prepare_shared_compiler_program,
)
from ember.ecp.bank_conditioning.frozen_condition_cache import (
    FrozenSharedCompilerCondition,
)
from ember.ecp.bank_conditioning.mapping import (
    MappingCondition,
    cross_video_consistency_loss,
    paired_mapping_loss,
)
from ember.ecp.stage0_train_step import _gather_records


if TYPE_CHECKING:
    from ember.ecp.bank_conditioning.mapping_training import MappingRuntime


FAMILY_NAMES = ("q", "v", "action_in", "action_out")


def _clip_gradients(
    parameters: Sequence[torch.nn.Parameter], *, maximum: float
) -> float:
    norm = torch.nn.utils.clip_grad_norm_(parameters, float(maximum))
    if not bool(torch.isfinite(norm)):
        raise RuntimeError("G3 mapping gradient norm is non-finite")
    return float(norm)


def _sum_gradients(
    parameters: Sequence[torch.nn.Parameter], *, world_size: int
) -> None:
    if world_size <= 1:
        return
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("G3 mapping rank omitted a trainable gradient")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def _pack_mapping_condition(
    runtime: MappingRuntime,
    condition: MappingCondition,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    task = runtime.task_by_id[condition.authority_id]
    sample = NaturalProgramSample(
        video_demos=(condition.video_demo,),
        action_demos=(),
        k=1,
        robustness_view="g3_mapping_k1",
    )
    packed = pack_shared_compiler_videos(
        task=task,
        sample=sample,
        video_store=runtime.video_store,
        query_points=runtime.query_points,
        device=runtime.context.device,
    )
    tokens, mask = runtime.language_tokens[condition.authority_id]
    return packed, tokens, mask


def prepare_mapping_condition(
    runtime: MappingRuntime,
    condition: MappingCondition,
    *,
    retain_cache: bool = True,
):
    def builder():
        packed, tokens, mask = _pack_mapping_condition(runtime, condition)
        return prepare_shared_compiler_condition(
            policy=runtime.policy,
            program_model=runtime.program,
            owners=runtime.owners,
            packed=packed,
            language_tokens=tokens,
            language_mask=mask,
            chunk_size=int(runtime.config["model"]["frame_chunk_size"]),
        )

    result = runtime.condition_cache.get_or_build(
        authority_id=condition.authority_id,
        video_demo=condition.video_demo,
        device=runtime.context.device,
        builder=builder,
        retain=retain_cache,
    )
    return FrozenSharedCompilerCondition(
        program=result.condition.program,
        videos=result.condition.videos,
        metrics={
            **result.condition.metrics,
            "frozen_condition_cache": "hit" if result.hit else "built",
            "frozen_condition_cache_file_bytes": result.file_bytes,
            "frozen_condition_cache_build_seconds": result.build_seconds,
            "frozen_condition_cache_load_seconds": result.load_seconds,
        },
    )


def prepare_mapping_condition_program(
    runtime: MappingRuntime, condition: MappingCondition
):
    cached = runtime.condition_cache.load_program(
        authority_id=condition.authority_id,
        video_demo=condition.video_demo,
        device=runtime.context.device,
    )
    if cached is not None:
        return cached
    packed, tokens, mask = _pack_mapping_condition(runtime, condition)
    return prepare_shared_compiler_program(
        policy=runtime.policy,
        program_model=runtime.program,
        packed=packed,
        language_tokens=tokens,
        language_mask=mask,
    )


def prepare_mapping_condition_output(
    runtime: MappingRuntime,
    condition: MappingCondition,
    *,
    retain_cache: bool = True,
) -> tuple[Any, Mapping[str, Any]]:
    prepared = prepare_mapping_condition(
        runtime, condition, retain_cache=retain_cache
    )
    return mapping_condition_output(runtime, condition, prepared)


def mapping_condition_output(
    runtime: MappingRuntime,
    condition: MappingCondition,
    prepared: Any,
    *,
    program: Any | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    selected_program = prepared.program if program is None else program
    if isinstance(prepared, FrozenSharedCompilerCondition):
        output = runtime.compiler.forward_base_compact(
            selected_program,
            prepared.videos,
            s_ref=runtime.ranks.s_ref,
        )
    else:
        output = runtime.compiler(
            selected_program, prepared.videos, s_ref=runtime.ranks.s_ref
        )
    if len(prepared.videos) != 1 or output.video_weights.shape != (1,):
        raise RuntimeError("G3 mapping condition escaped K1 identity")
    return output, {
        **prepared.metrics,
        "split_sampled_frames": condition.sampled_frames,
        "solve_metrics": output.solve_metrics.detach().float().cpu().tolist(),
        "conditioning_metrics": output.conditioning_metrics.detach()
        .float()
        .cpu()
        .tolist(),
    }


def load_mapping_condition_teachers(
    runtime: MappingRuntime, condition: MappingCondition
):
    expected = tuple(sorted(runtime.mapping_split.member_names[condition.authority_id]))
    rows = runtime.native_teachers.lookup_members(
        authority_id=condition.authority_id,
        k=1,
        video_demo=condition.video_demo,
        member_names=expected,
    )
    if rows is None or tuple(row.member_name for row in rows) != expected:
        raise RuntimeError("G3 mapping condition lost its complete member set")
    return rows


def load_mapping_consensus_teachers(
    runtime: MappingRuntime, condition: MappingCondition
):
    expected = tuple(sorted(runtime.mapping_split.member_names[condition.authority_id]))
    rows = runtime.consensus_teachers.lookup_members(
        authority_id=condition.authority_id,
        video_demo=condition.video_demo,
        member_names=expected,
    )
    if tuple(row.member_name for row in rows) != expected:
        raise RuntimeError("G3 mapping consensus lost its complete member set")
    return rows


def mapping_recovery_record(loss: Any) -> dict[str, Any]:
    return {
        "mapping_loss": float(loss.total.detach()),
        "input_subspace_loss": float(loss.input_subspace.detach()),
        "output_subspace_loss": float(loss.output_subspace.detach()),
        "update_direction_loss": float(loss.update_direction.detach()),
        "member_distances": loss.member_distances.detach().float().cpu().tolist(),
        "member_responsibilities": loss.responsibilities.detach()
        .float()
        .cpu()
        .tolist(),
        "best_member": int(loss.best_member),
        "family_recovery": {
            name: float(value)
            for name, value in zip(
                FAMILY_NAMES,
                loss.family_recovery.detach().float().cpu(),
                strict=True,
            )
        },
        "best_family_recovery": {
            name: float(value)
            for name, value in zip(
                FAMILY_NAMES,
                loss.best_family_recovery.detach().float().cpu(),
                strict=True,
            )
        },
        "mean_best_recovery": float(
            loss.best_family_recovery.detach().float().mean()
        ),
    }


def _run_task(
    runtime: MappingRuntime,
    *,
    condition: MappingCondition,
    companion: MappingCondition,
    loss_divisor: float,
) -> dict[str, Any]:
    tick = time.monotonic()
    temperature = float(runtime.config["optimization"]["mapping"]["temperature"])
    primary_teachers = load_mapping_consensus_teachers(runtime, condition)
    companion_teachers = load_mapping_consensus_teachers(runtime, companion)
    primary_output, metrics = prepare_mapping_condition_output(runtime, condition)
    primary_loss = paired_mapping_loss(
        output=primary_output,
        teachers=primary_teachers,
        owners=runtime.owners,
        temperature=temperature,
    )
    (0.5 * primary_loss.total / loss_divisor).backward()

    companion_output, companion_metrics = prepare_mapping_condition_output(
        runtime, companion
    )
    companion_loss = paired_mapping_loss(
        output=companion_output,
        teachers=companion_teachers,
        owners=runtime.owners,
        temperature=temperature,
    )
    consistency = cross_video_consistency_loss(
        primary_output=primary_output,
        companion_output=companion_output,
        primary_teachers=primary_teachers,
        companion_teachers=companion_teachers,
        owners=runtime.owners,
        responsibilities=primary_loss.responsibilities,
        margin=float(
            runtime.config["optimization"]["mapping"]["cross_video_margin"]
        ),
    )
    consistency_weight = float(
        runtime.config["optimization"]["mapping"]["cross_video_weight"]
    )
    (
        (0.5 * companion_loss.total + consistency_weight * consistency.total)
        / loss_divisor
    ).backward()
    return {
        "authority_id": condition.authority_id,
        "role": condition.role,
        "video_demo": condition.video_demo,
        "companion_video_demo": companion.video_demo,
        "teacher_target": "fit_video_rank4_truncated_mean_update",
        **mapping_recovery_record(primary_loss),
        "companion": mapping_recovery_record(companion_loss),
        "cross_video_loss": float(consistency.total.detach()),
        "predicted_family_distance": consistency.predicted_family_distance.detach()
        .float()
        .cpu()
        .tolist(),
        "allowed_family_distance": consistency.allowed_family_distance.detach()
        .float()
        .cpu()
        .tolist(),
        "condition_metrics": metrics,
        "companion_metrics": companion_metrics,
        "task_seconds": time.monotonic() - tick,
    }


def _mapping_gradient_probes(runtime: MappingRuntime) -> dict[str, Any]:
    scorer = runtime.compiler.primal_scorer
    probes = {
        "input_primal": scorer.input_primal_heads[0].weight.grad,
        "output_primal": scorer.output_primal_heads[0][0].weight.grad,
        "owner_embedding": scorer.owner_embedding.grad,
        "rank_embedding": scorer.rank_embedding.grad,
        "event_embedding": scorer.event_embedding.grad,
        "group_embedding": scorer.group_embedding.grad,
        "full_program": scorer.program_context["q"][1].weight.grad,
        "event_weight": scorer.event_score["q"].weight.grad,
    }
    return probes


def run_mapping_optimizer_step(
    runtime: MappingRuntime,
    *,
    macro: int,
    update: int,
    task_group: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(task_group) != 6:
        raise RuntimeError("G3 mapping global task group changed")
    conditions = {
        task: runtime.schedule.condition(task, macro=macro, update=update)
        for task in task_group
    }
    companions = {
        task: runtime.schedule.companion(
            conditions[task], macro=macro, update=update
        )
        for task in task_group
    }
    assignments = runtime.schedule.assignments(
        task_group,
        {
            task: MappingCondition(
                authority_id=conditions[task].authority_id,
                role=conditions[task].role,
                video_demo=conditions[task].video_demo,
                sampled_frames=(
                    conditions[task].sampled_frames
                    + companions[task].sampled_frames
                ),
            )
            for task in task_group
        },
        runtime.context.world_size,
    )
    tick = time.monotonic()
    runtime.optimizer.zero_grad(set_to_none=True)
    local_records = [
        _run_task(
            runtime,
            condition=conditions[task],
            companion=companions[task],
            loss_divisor=6.0,
        )
        for task in assignments[runtime.context.rank]
    ]
    if any(parameter.grad is not None for parameter in runtime.frozen_parameters):
        raise RuntimeError("frozen G3 mapping authority accumulated gradients")
    _sum_gradients(
        runtime.trainable_parameters, world_size=runtime.context.world_size
    )
    probes = _mapping_gradient_probes(runtime)
    probe_norms = {}
    for name, gradient in probes.items():
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError(f"G3 mapping {name} gradient is absent or non-finite")
        probe_norms[name] = float(gradient.float().norm())
    if min(probe_norms.values()) <= 0:
        raise RuntimeError("G3 mapping primal scorer gradient is zero")
    gradient_norm = _clip_gradients(
        runtime.trainable_parameters,
        maximum=float(
            runtime.config["optimization"]["optimizer"]["gradient_clip_norm"]
        ),
    )
    runtime.optimizer.step()
    runtime.scheduler_lr.step()
    runtime.optimizer_steps += 1
    records = sorted(
        _gather_records(local_records, runtime.context.world_size),
        key=lambda row: int(row["authority_id"]),
    )
    role_counts = {
        role: sum(row["role"] == role for row in records)
        for role in ("meta_fit", "target_fit")
    }
    if len(records) != 6 or role_counts != {"meta_fit": 3, "target_fit": 3}:
        raise RuntimeError("G3 mapping update lost fixed task-role weight")
    return records, {
        "optimizer_step": runtime.optimizer_steps,
        "role_counts": role_counts,
        "gradient_probe_norms": probe_norms,
        "gradient_norm_before_clip": gradient_norm,
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "rank_assignments": [list(row) for row in assignments],
        "step_seconds": time.monotonic() - tick,
    }
