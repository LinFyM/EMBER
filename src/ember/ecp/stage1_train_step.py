"""One task-equal distributed update for ECP Stage 1."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from ember.ecp.compiler import select_compiled_state
from ember.ecp.policy_response import OwnerResolvedResponseLoss
from ember.ecp.stage1_data import pack_stage1_videos
from ember.ecp.stage1_objective import ECPStage1Loss, ecp_stage1_loss
from ember.ecp.stage1_support import (
    PolicySupportLoss,
    policy_support_owner_distillation_loss,
)

if TYPE_CHECKING:
    from ember.ecp.stage1_training import ECPStage1Runtime


def _objective_weights(
    runtime: "ECPStage1Runtime", task_visits: int
) -> tuple[str, dict[str, float]]:
    del task_visits
    return "owner_response_bootstrap", {
        name: float(value)
        for name, value in runtime.config["objective"]["weights"].items()
    }


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


def _policy_support_loss(
    runtime: "ECPStage1Runtime",
    *,
    task_ordinal: int,
    task_visit: int,
    candidate: dict[str, torch.Tensor],
    task_visits_after_update: int,
) -> tuple[PolicySupportLoss, OwnerResolvedResponseLoss, int, str]:
    del task_visits_after_update
    panel = runtime.support_bank.task(task_ordinal).panel_for_visit(task_visit)
    cached = runtime.support_panels[(task_ordinal, panel.panel_id)]
    support, owner = policy_support_owner_distillation_loss(
        policy=runtime.policy,
        candidate_state=candidate,
        contract=runtime.contract,
        cached=cached,
        projector=runtime.observer.model.encoder.observer.projector,
        horizon_basis=int(runtime.config["policy_support"]["horizon_basis"]),
        preservation=str(runtime.config["objective"]["support_preservation"]),
    )
    return support, owner, panel.panel_id, panel.kind


def _local_record(
    runtime: "ECPStage1Runtime",
    *,
    task: Any,
    task_visit: int,
    demo_indices: tuple[int, ...],
    video_offsets: torch.Tensor,
    loss: ECPStage1Loss,
    total: torch.Tensor,
    owner_response: OwnerResolvedResponseLoss,
    output: Any,
    panel_id: int,
    panel_kind: str,
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
        "support_panel_id": panel_id,
        "support_panel_kind": panel_kind,
        "total": float(total.detach()),
        "functional_total": float(loss.total.detach()),
        "owner_response": float(owner_response.loss.detach()),
        "owner_response_disagreement": float(
            owner_response.normalized_disagreement.detach()
        ),
        "owner_response_active_fraction": float(
            owner_response.active_owner_fraction.detach()
        ),
        "member_effective_update": float(loss.member_effective_update.detach()),
        "consensus_effective_update": float(
            loss.consensus_effective_update.detach()
        ),
        "member_canonical_factor": float(loss.member_canonical_factor.detach()),
        "consensus_canonical_factor": float(
            loss.consensus_canonical_factor.detach()
        ),
        "prior_preservation": float(loss.prior_preservation.detach()),
        "functional_response": float(loss.functional_response.detach()),
        "successful_response": float(loss.successful_response.detach()),
        "learner_response": float(loss.learner_response.detach()),
        "source_support": float(loss.source_support.detach()),
        "shared_support": float(loss.shared_support.detach()),
        "expert_set_disagreement": float(
            loss.expert_set_disagreement.detach()
        ),
        "locality": float(loss.locality.detach()),
        "consensus_exact_owner_attention": float(
            output.consensus_compilation.exact_owner_attention.detach()
        ),
        "consensus_rank_replacement_fraction": float(
            output.consensus_compilation.rank_replacement_fraction.detach()
        ),
        "mean_active_events": float(
            (output.teacher.program.presence.detach() > 0.5).float().sum()
        ),
        "q_pi_gate_mean": float(output.teacher.evidence_gate.detach().mean()),
        "q_pi_gate_min": float(output.teacher.evidence_gate.detach().min()),
        "q_pi_gate_max": float(output.teacher.evidence_gate.detach().max()),
        "support_attention_entropy": float(
            output.teacher.support_attention_entropy.detach().mean()
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
    evidence = runtime.evidence_bank.evidence(
        task_ordinal, runtime.support_bank.task(task_ordinal)
    )
    runtime.optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = runtime.model(encoded, evidence, packed.video_group_ids)
        candidate = select_compiled_state(
            output.consensus_compilation.state, 0
        )
        task_visits = cursor + runtime.context.world_size
        support_loss, owner_response, panel_id, panel_kind = _policy_support_loss(
            runtime,
            task_ordinal=task_ordinal,
            task_visit=task_visit,
            candidate=candidate,
            task_visits_after_update=task_visits,
        )
        objective_phase, objective_weights = _objective_weights(
            runtime, task_visits
        )
        loss = ecp_stage1_loss(
            member=output.member_compilation,
            consensus=output.consensus_compilation,
            prior=output.prior_compilation,
            expert_states=evidence.member_states,
            prior_target=runtime.prior_state,
            contract=runtime.contract,
            policy_support=support_loss,
            weights=objective_weights,
        )
        total = loss.total + float(
            runtime.config["objective"]["owner_response_distillation_weight"]
        ) * owner_response.loss
    if not bool(torch.isfinite(total)):
        raise RuntimeError(f"non-finite ECP Stage 1 loss at task visit {task_visits}")
    total.backward()
    _reduce_gradients(runtime.trainable_parameters, runtime.context.world_size)
    factor_gradient = sum(
        parameter.grad.float().square().sum()
        for heads in (
            runtime.model.compiler.factor_a,
            runtime.model.compiler.factor_b,
        )
        for parameter in heads.parameters()
        if parameter.grad is not None
    ).sqrt()
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
        total=total,
        owner_response=owner_response,
        output=output,
        panel_id=panel_id,
        panel_kind=panel_kind,
    )
    records = _gather(local, runtime.context.world_size)
    metric_names = (
        "total",
        "functional_total",
        "owner_response",
        "owner_response_disagreement",
        "owner_response_active_fraction",
        "member_effective_update",
        "consensus_effective_update",
        "member_canonical_factor",
        "consensus_canonical_factor",
        "prior_preservation",
        "functional_response",
        "successful_response",
        "learner_response",
        "source_support",
        "shared_support",
        "expert_set_disagreement",
        "locality",
        "consensus_exact_owner_attention",
        "consensus_rank_replacement_fraction",
        "mean_active_events",
        "q_pi_gate_mean",
        "q_pi_gate_min",
        "q_pi_gate_max",
        "support_attention_entropy",
    )
    return {
        "task_visits": task_visits,
        "optimizer_update": task_visits // runtime.context.world_size,
        "objective_phase": objective_phase,
        "objective_weights": objective_weights,
        "means": {
            name: sum(float(row[name]) for row in records) / len(records)
            for name in metric_names
        },
        "gradient_norm_before_clip": float(gradient_norm),
        "factor_head_gradient_norm_before_clip": float(factor_gradient),
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "update_seconds": time.monotonic() - tick,
        "elapsed_seconds": time.monotonic() - run_started,
        "max_cuda_allocated_bytes": int(
            torch.cuda.max_memory_allocated(runtime.context.device)
        ),
        "records": sorted(records, key=lambda row: int(row["task_ordinal"])),
    }
