"""One task-equal fixed-compiler free-Program update for ECP Stage 1."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from ember.ecp.compiler import select_compiled_state
from ember.ecp.policy_response import TargetActivationEffectLoss
from ember.ecp.stage1_objective import ECPStage1Loss, ecp_stage1_loss
from ember.ecp.stage1_support import (
    CachedPolicySupportPanel,
    policy_support_activation_distillation_loss,
)
from ember.writer.functional import functional_lora_loss_gradient

if TYPE_CHECKING:
    from ember.ecp.stage1_training import ECPStage1Runtime


def _objective_weights(
    runtime: "ECPStage1Runtime", task_visits: int
) -> tuple[str, dict[str, float]]:
    del task_visits
    return "fixed_compiler_free_program_reachability", {
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


def _module_gradient_norm(module: torch.nn.Module) -> torch.Tensor:
    values = [
        parameter.grad.float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not values:
        return next(module.parameters()).new_zeros((), dtype=torch.float32)
    return torch.stack(values).sum().sqrt()


def _policy_support_panel(
    runtime: "ECPStage1Runtime",
    *,
    task_ordinal: int,
    task_visit: int,
    task_visits_after_update: int,
) -> CachedPolicySupportPanel:
    del task_visits_after_update
    panel = runtime.support_bank.task(task_ordinal).panel_for_visit(task_visit)
    return runtime.support_panels[(task_ordinal, panel.panel_id)]


def _action_supervision_weight(
    runtime: "ECPStage1Runtime", cached: CachedPolicySupportPanel
) -> float:
    weights = runtime.config["objective"]["action_supervision_weights"]
    panel = cached.panel
    if panel.kind == "successful":
        return float(weights["successful"])
    if panel.learner_success is True:
        return float(weights["verified_successful_learner"])
    return float(weights["failed_learner"])


def _action_policy_seed(
    runtime: "ECPStage1Runtime",
    *,
    task_ordinal: int,
    task_visit: int,
    panel_id: int,
) -> int:
    base = int(runtime.config["objective"]["train_policy_seed"])
    return (
        base
        + (task_ordinal + 1) * 1_000_003
        + (task_visit + 1) * 10_007
        + (panel_id + 1) * 101
    ) % ((1 << 63) - 1)


def _action_policy_gradient(
    runtime: "ECPStage1Runtime",
    *,
    candidate: dict[str, torch.Tensor],
    cached: CachedPolicySupportPanel,
    task_ordinal: int,
    task_visit: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], float]:
    supervision_weight = _action_supervision_weight(runtime, cached)
    if supervision_weight == 0.0:
        return next(iter(candidate.values())).new_zeros(()), {}, 0.0
    objective = runtime.config["objective"]
    loss, _, gradients = functional_lora_loss_gradient(
        runtime.policy,
        candidate,
        runtime.contract,
        batch=cached.batch,
        policy_rng_seed=_action_policy_seed(
            runtime,
            task_ordinal=task_ordinal,
            task_visit=task_visit,
            panel_id=cached.panel.panel_id,
        ),
        policy_rng_device=runtime.context.device,
        flow_time_sampling_scheme=str(objective["policy_flow_time_sampling_scheme"]),
        flow_noise_sampling_scheme=str(objective["policy_flow_noise_sampling_scheme"]),
        policy_microbatch_size=int(
            runtime.config["optimization"]["functional_policy_microbatch_size"]
        ),
        collect_policy_details=False,
    )
    return loss, gradients, supervision_weight


def _local_record(
    runtime: "ECPStage1Runtime",
    *,
    task: Any,
    task_visit: int,
    loss: ECPStage1Loss,
    total: torch.Tensor,
    structural_total: torch.Tensor,
    action_policy_loss: torch.Tensor,
    action_supervision_weight: float,
    action_lora_gradient_norm: torch.Tensor,
    activation_effect: TargetActivationEffectLoss,
    program: Any,
    compilation: Any,
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
        "successful_members": len(runtime.evidence_bank.member_indices(task.ordinal)),
        "support_panel_id": panel_id,
        "support_panel_kind": panel_kind,
        "total": float(total.detach()),
        "structural_total": float(structural_total.detach()),
        "action_policy_loss": float(action_policy_loss.detach()),
        "action_supervision_active": float(action_supervision_weight > 0.0),
        "action_supervision_weight": float(action_supervision_weight),
        "action_lora_gradient_norm": float(action_lora_gradient_norm.detach()),
        "activation_effect": float(activation_effect.loss.detach()),
        "activation_effect_disagreement": float(
            activation_effect.normalized_disagreement.detach()
        ),
        "activation_effect_active_fraction": float(
            activation_effect.active_owner_fraction.detach()
        ),
        "member_effective_update": float(loss.member_effective_update.detach()),
        "consensus_effective_update": float(loss.consensus_effective_update.detach()),
        "member_canonical_factor": float(loss.member_canonical_factor.detach()),
        "consensus_canonical_factor": float(loss.consensus_canonical_factor.detach()),
        "prior_preservation": float(loss.prior_preservation.detach()),
        "functional_response": float(loss.functional_response.detach()),
        "successful_response": float(loss.successful_response.detach()),
        "learner_response": float(loss.learner_response.detach()),
        "source_support": float(loss.source_support.detach()),
        "shared_support": float(loss.shared_support.detach()),
        "expert_set_disagreement": float(loss.expert_set_disagreement.detach()),
        "locality": float(loss.locality.detach()),
        "consensus_exact_owner_attention": float(
            compilation.exact_owner_attention.detach()
        ),
        "consensus_rank_replacement_fraction": float(
            compilation.rank_replacement_fraction.detach()
        ),
        "mean_active_events": float((program.presence.detach() > 0.5).float().sum()),
        **{
            name: float(value.detach())
            for name, value in runtime.free_programs.row(task.ordinal)
            .diagnostics()
            .items()
        },
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
    evidence = runtime.evidence_bank.evidence(
        task_ordinal, runtime.support_bank.task(task_ordinal)
    )
    runtime.optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        program = runtime.free_programs(task_ordinal)
        compilation = runtime.model.compiler(program)
        prior_compilation = runtime.model.compiler(program.prior_only())
        candidate = select_compiled_state(compilation.state, 0)
        task_visits = cursor + runtime.context.world_size
        cached = _policy_support_panel(
            runtime,
            task_ordinal=task_ordinal,
            task_visit=task_visit,
            task_visits_after_update=task_visits,
        )
        action_policy_loss, action_gradients, action_supervision_weight = (
            _action_policy_gradient(
                runtime,
                candidate=candidate,
                cached=cached,
                task_ordinal=task_ordinal,
                task_visit=task_visit,
            )
        )
        support_loss, activation_effect = policy_support_activation_distillation_loss(
            policy=runtime.policy,
            candidate_state=candidate,
            contract=runtime.contract,
            cached=cached,
            preservation=str(runtime.config["objective"]["support_preservation"]),
        )
        objective_phase, objective_weights = _objective_weights(runtime, task_visits)
        loss = ecp_stage1_loss(
            member=compilation,
            consensus=compilation,
            prior=prior_compilation,
            expert_states=evidence.member_states,
            prior_target=runtime.prior_state,
            contract=runtime.contract,
            policy_support=support_loss,
            weights=objective_weights,
        )
        structural_total = (
            loss.total
            + float(
                runtime.config["objective"]["activation_effect_distillation_weight"]
            )
            * activation_effect.loss
        )
        action_scale = (
            float(runtime.config["objective"]["action_policy_loss_weight"])
            * action_supervision_weight
        )
        total = structural_total.detach() + action_scale * action_policy_loss
    if not bool(torch.isfinite(total)) or not bool(torch.isfinite(structural_total)):
        raise RuntimeError(f"non-finite ECP Stage 1 loss at task visit {task_visits}")
    if action_gradients:
        active_names = tuple(
            name for name, value in candidate.items() if value.requires_grad
        )
        torch.autograd.backward(
            tuple(candidate[name] for name in active_names),
            grad_tensors=tuple(
                action_scale * action_gradients[name] for name in active_names
            ),
            retain_graph=True,
        )
    structural_total.backward()
    _reduce_gradients(runtime.trainable_parameters, runtime.context.world_size)
    active_ordinals = tuple(
        runtime.schedule[cursor + rank][0] for rank in range(runtime.context.world_size)
    )
    runtime.free_programs.freeze_inactive_gradients(active_ordinals)
    action_lora_gradient_norm = (
        torch.stack(
            [gradient.float().square().sum() for gradient in action_gradients.values()]
        )
        .sum()
        .sqrt()
        if action_gradients
        else total.new_zeros(())
    )
    free_program_gradient = _module_gradient_norm(runtime.free_programs)
    compiler_gradient = _module_gradient_norm(runtime.model.compiler)
    policy_teacher_gradient = _module_gradient_norm(runtime.model.policy_teacher)
    visible_program_gradient = _module_gradient_norm(runtime.model.visible_program)
    clip = float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"])
    active_parameters = runtime.free_programs.parameters_for_ordinals(active_ordinals)
    gradient_norm = torch.nn.utils.clip_grad_norm_(active_parameters, clip)
    if (
        not bool(
            torch.isfinite(
                gradient_norm
                + free_program_gradient
                + compiler_gradient
                + policy_teacher_gradient
                + visible_program_gradient
            )
        )
        or float(policy_teacher_gradient) != 0.0
        or float(visible_program_gradient) != 0.0
        or float(compiler_gradient) != 0.0
    ):
        raise RuntimeError("invalid fixed-compiler free-Program gradient")
    runtime.optimizer.step()
    runtime.scheduler.step()
    local = _local_record(
        runtime,
        task=task,
        task_visit=task_visit,
        loss=loss,
        total=total,
        structural_total=structural_total,
        action_policy_loss=action_policy_loss,
        action_supervision_weight=action_supervision_weight,
        action_lora_gradient_norm=action_lora_gradient_norm,
        activation_effect=activation_effect,
        program=program,
        compilation=compilation,
        panel_id=cached.panel.panel_id,
        panel_kind=cached.panel.kind,
    )
    records = _gather(local, runtime.context.world_size)
    metric_names = (
        "total",
        "structural_total",
        "action_policy_loss",
        "action_supervision_active",
        "action_supervision_weight",
        "action_lora_gradient_norm",
        "activation_effect",
        "activation_effect_disagreement",
        "activation_effect_active_fraction",
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
        "process_delta_relative",
        "uncertainty_scale_mean",
        "uncertainty_scale_min",
        "uncertainty_scale_max",
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
        "free_program_gradient_norm_before_clip": float(free_program_gradient),
        "compiler_gradient_norm_before_clip": float(compiler_gradient),
        "policy_teacher_gradient_norm_before_clip": float(policy_teacher_gradient),
        "visible_program_gradient_norm_before_clip": float(visible_program_gradient),
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "update_seconds": time.monotonic() - tick,
        "elapsed_seconds": time.monotonic() - run_started,
        "max_cuda_allocated_bytes": int(
            torch.cuda.max_memory_allocated(runtime.context.device)
        ),
        "records": sorted(records, key=lambda row: int(row["task_ordinal"])),
    }
