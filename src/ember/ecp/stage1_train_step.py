"""One task-equal distributed update for ECP Stage 1."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from ember.ecp.compiler import select_compiled_state
from ember.ecp.stage1_objective import ECPStage1Loss
from ember.ecp.stage1_data import pack_stage1_videos
from ember.ecp.stage1_objective import ecp_stage1_loss
from ember.functional_adaptation.functional_response import (
    functional_response_distillation_loss,
)

if TYPE_CHECKING:
    from ember.ecp.stage1_training import ECPStage1Runtime


def _reduce_gradients(
    parameters: tuple[torch.nn.Parameter, ...], world_size: int
) -> None:
    if world_size == 1:
        return
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def _gather(record: dict[str, Any], world_size: int) -> list[dict[str, Any]]:
    if world_size == 1:
        return [record]
    rows: list[Any] = [None] * world_size
    dist.all_gather_object(rows, record)
    return [dict(row) for row in rows]


def _functional_loss(
    runtime: "ECPStage1Runtime",
    *,
    task_ordinal: int,
    task_visit: int,
    candidate: dict[str, torch.Tensor],
    task_visits_after_update: int,
) -> tuple[torch.Tensor, int | None, int | None]:
    if task_visits_after_update <= runtime.functional_start_task_visits:
        return next(iter(candidate.values())).new_zeros(()), None, None
    indices = runtime.evidence_bank.member_indices(task_ordinal)
    member_index = indices[task_visit % len(indices)]
    panel_index = (task_visit // len(indices)) % 4
    panel = runtime.functional_panels[member_index][panel_index]
    loss = functional_response_distillation_loss(
        runtime.policy,
        candidate,
        runtime.contract,
        panel.batch,
        panel.target,
        policy_seed=panel.policy_seed,
    )
    return loss, member_index, panel_index


def _local_record(
    runtime: "ECPStage1Runtime",
    *,
    task: Any,
    task_visit: int,
    demo_indices: tuple[int, ...],
    video_offsets: torch.Tensor,
    loss: ECPStage1Loss,
    output: Any,
    member_index: int | None,
    panel_index: int | None,
) -> dict[str, Any]:
    return {
        "rank": runtime.context.rank,
        "task_ordinal": task.ordinal,
        "global_task_id": task.global_task_id,
        "suite": task.suite,
        "task_id": task.task_id,
        "task_visit": task_visit,
        "demo_indices": list(demo_indices),
        "sampled_frames": [
            int(video_offsets[index + 1] - video_offsets[index])
            for index in range(len(demo_indices))
        ],
        "successful_members": len(runtime.evidence_bank.member_indices(task.ordinal)),
        "functional_member_index": member_index,
        "functional_panel_index": panel_index,
        "total": float(loss.total.detach()),
        "member_effective_update": float(loss.member_effective_update.detach()),
        "consensus_effective_update": float(
            loss.consensus_effective_update.detach()
        ),
        "prior_preservation": float(loss.prior_preservation.detach()),
        "functional_response": float(loss.functional_response.detach()),
        "locality": float(loss.locality.detach()),
        "consensus_exact_owner_attention": float(
            output.consensus_compilation.exact_owner_attention.detach()
        ),
        "mean_active_events": float(
            (output.teacher.program.presence.detach() > 0.5).float().sum()
        ),
    }


def run_stage1_update(
    runtime: "ECPStage1Runtime",
    *,
    cursor: int,
    run_started: float,
) -> dict[str, Any]:
    tick = time.monotonic()
    schedule_index = cursor + runtime.context.rank
    task_ordinal, task_visit = runtime.schedule[schedule_index]
    task = runtime.task_by_ordinal[task_ordinal]
    packed = pack_stage1_videos(
        store=runtime.video_store,
        ordinal=task_ordinal,
        visit=task_visit,
        seed=int(runtime.config["data"]["pair_seed"]),
        k=int(runtime.config["data"]["visible_videos_per_visit"]),
        device=runtime.context.device,
    )
    tokens, mask = runtime.language_tokens[task_ordinal]
    expert = runtime.policy.model.paligemma_with_expert.gemma_expert.model
    with torch.no_grad(), runtime.observer.action_meta.installed(expert):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = runtime.observer.model.encoder(
                policy=runtime.policy,
                frames=packed.frames,
                video_offsets=packed.video_offsets,
                frame_condition_ids=packed.frame_condition_ids,
                language_tokens=tokens,
                language_mask=mask,
            )
    evidence = runtime.evidence_bank.evidence(task_ordinal)
    runtime.optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = runtime.model(encoded, evidence, packed.video_group_ids)
        candidate = select_compiled_state(
            output.consensus_compilation.state, 0
        )
        task_visits = cursor + runtime.context.world_size
        functional, member_index, panel_index = _functional_loss(
            runtime,
            task_ordinal=task_ordinal,
            task_visit=task_visit,
            candidate=candidate,
            task_visits_after_update=task_visits,
        )
        loss = ecp_stage1_loss(
            member=output.member_compilation,
            consensus=output.consensus_compilation,
            prior=output.prior_compilation,
            expert_states=evidence.member_states,
            prior_target=runtime.prior_state,
            contract=runtime.contract,
            functional_response=functional,
            weights=runtime.config["objective"]["weights"],
        )
    if not bool(torch.isfinite(loss.total)):
        raise RuntimeError(f"non-finite ECP Stage 1 loss at task visit {task_visits}")
    loss.total.backward()
    _reduce_gradients(runtime.trainable_parameters, runtime.context.world_size)
    clip = float(
        runtime.config["optimization"]["optimizer"]["gradient_clip_norm"]
    )
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        runtime.trainable_parameters, clip
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("non-finite ECP Stage 1 gradient")
    runtime.optimizer.step()
    runtime.scheduler.step()
    local = _local_record(
        runtime,
        task=task,
        task_visit=task_visit,
        demo_indices=packed.demo_indices,
        video_offsets=packed.video_offsets,
        loss=loss,
        output=output,
        member_index=member_index,
        panel_index=panel_index,
    )
    records = _gather(local, runtime.context.world_size)
    metric_names = (
        "total",
        "member_effective_update",
        "consensus_effective_update",
        "prior_preservation",
        "functional_response",
        "locality",
        "consensus_exact_owner_attention",
        "mean_active_events",
    )
    return {
        "task_visits": task_visits,
        "optimizer_update": task_visits // runtime.context.world_size,
        "means": {
            name: sum(float(row[name]) for row in records) / len(records)
            for name in metric_names
        },
        "gradient_norm_before_clip": float(gradient_norm),
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "update_seconds": time.monotonic() - tick,
        "elapsed_seconds": time.monotonic() - run_started,
        "max_cuda_allocated_bytes": int(
            torch.cuda.max_memory_allocated(runtime.context.device)
        ),
        "records": sorted(records, key=lambda row: int(row["task_ordinal"])),
    }
