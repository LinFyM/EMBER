"""One task-equal mapping-diverse q_pi/compiler update for ECP Stage 1."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from ember.ecp.compiler import select_compiled_state
from ember.ecp.policy_response import TargetActivationEffectLoss
from ember.ecp.stage1_objective import ECPStage1Loss, ecp_stage1_loss
from ember.ecp.stage1_data import pack_stage1_videos
from ember.ecp.stage1_support import (
    CachedPolicySupportPanel,
    policy_support_activation_distillation_loss,
    shared_prior_response_distillation_loss,
)
from ember.writer.functional import functional_lora_loss_gradient

if TYPE_CHECKING:
    from ember.ecp.stage1_training import ECPStage1Runtime


@dataclass(frozen=True)
class Stage1Forward:
    output: Any
    candidate: dict[str, torch.Tensor]
    cached: CachedPolicySupportPanel
    loss: ECPStage1Loss
    total: torch.Tensor
    structural_total: torch.Tensor
    action_policy_loss: torch.Tensor
    action_gradients: dict[str, torch.Tensor]
    action_supervision_weight: float
    activation_effect: TargetActivationEffectLoss
    prior_shared_response: torch.Tensor
    objective_phase: str
    objective_weights: dict[str, float]
    task_visits: int


def _objective_weights(
    runtime: "ECPStage1Runtime", task_visits: int
) -> tuple[str, dict[str, float]]:
    del task_visits
    return "mapping_diverse_q_pi_compiler_identification", {
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


def stage1_action_policy_seed(
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


def stage1_action_policy_gradient(
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
        policy_rng_seed=stage1_action_policy_seed(
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
    prior_shared_response: torch.Tensor,
    program: Any,
    compilation: Any,
    panel_id: int,
    panel_kind: str,
) -> dict[str, Any]:
    return {
        "rank": runtime.context.rank,
        "task_ordinal": task.ordinal,
        "task_asset_key": task.asset_key,
        "task_domain": task.domain,
        "global_task_id": task.global_task_id,
        "suite": task.suite,
        "task_id": task.task_id,
        "task_visit": task_visit,
        "successful_members": len(runtime.evidence_bank.member_indices(task.ordinal)),
        "support_panel_id": panel_id,
        "support_panel_kind": panel_kind,
        "demo_indices": list(program["demo_indices"]),
        "sampled_frames": list(program["sampled_frames"]),
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
        "prior_shared_response": float(prior_shared_response.detach()),
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
        "mean_active_events": float(
            (program["value"].presence.detach() > 0.5).float().sum()
        ),
        "q_pi_gate_mean": float(program["gate"].detach().mean()),
        "q_pi_gate_min": float(program["gate"].detach().min()),
        "q_pi_gate_max": float(program["gate"].detach().max()),
    }


def encode_stage1_visit(
    runtime: "ECPStage1Runtime", *, task_ordinal: int, task_visit: int
) -> tuple[Any, Any]:
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
    return packed, encoded


def _forward_objective(
    runtime: "ECPStage1Runtime",
    *,
    task_ordinal: int,
    task_visit: int,
    cursor: int,
    packed: Any,
    encoded: Any,
) -> Stage1Forward:
    evidence = runtime.evidence_bank.evidence(
        task_ordinal, runtime.support_bank.task(task_ordinal)
    )
    runtime.optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = runtime.model(encoded, evidence, packed.video_group_ids)
        compilation = output.consensus_compilation
        candidate = select_compiled_state(compilation.state, 0)
        prior_candidate = select_compiled_state(output.prior_compilation.state, 0)
        task_visits = cursor + runtime.context.world_size
        cached = _policy_support_panel(
            runtime,
            task_ordinal=task_ordinal,
            task_visit=task_visit,
            task_visits_after_update=task_visits,
        )
        action_loss, action_gradients, supervision_weight = stage1_action_policy_gradient(
            runtime,
            candidate=candidate,
            cached=cached,
            task_ordinal=task_ordinal,
            task_visit=task_visit,
        )
        support_loss, activation_effect = policy_support_activation_distillation_loss(
            policy=runtime.policy,
            candidate_state=candidate,
            contract=runtime.contract,
            cached=cached,
            preservation=str(runtime.config["objective"]["support_preservation"]),
        )
        prior_response = shared_prior_response_distillation_loss(
            policy=runtime.policy,
            candidate_state=prior_candidate,
            contract=runtime.contract,
            cached=cached,
        )
        objective_phase, objective_weights = _objective_weights(runtime, task_visits)
        loss = ecp_stage1_loss(
            member=output.member_compilation,
            consensus=compilation,
            prior=output.prior_compilation,
            expert_states=evidence.member_states,
            prior_target=runtime.prior_state,
            contract=runtime.contract,
            policy_support=support_loss,
            weights=objective_weights,
        )
        structural = (
            loss.total
            + float(
                runtime.config["objective"]["activation_effect_distillation_weight"]
            )
            * activation_effect.loss
            + float(runtime.config["objective"]["prior_shared_response_weight"])
            * prior_response
        )
        action_scale = (
            float(runtime.config["objective"]["action_policy_loss_weight"])
            * supervision_weight
        )
        total = structural.detach() + action_scale * action_loss
    if not bool(torch.isfinite(total)) or not bool(torch.isfinite(structural)):
        raise RuntimeError(f"non-finite ECP Stage 1 loss at task visit {task_visits}")
    return Stage1Forward(
        output=output,
        candidate=candidate,
        cached=cached,
        loss=loss,
        total=total,
        structural_total=structural,
        action_policy_loss=action_loss,
        action_gradients=action_gradients,
        action_supervision_weight=supervision_weight,
        activation_effect=activation_effect,
        prior_shared_response=prior_response,
        objective_phase=objective_phase,
        objective_weights=objective_weights,
        task_visits=task_visits,
    )


def _apply_gradients(
    runtime: "ECPStage1Runtime", forward: Stage1Forward
) -> dict[str, torch.Tensor]:
    action_scale = (
        float(runtime.config["objective"]["action_policy_loss_weight"])
        * forward.action_supervision_weight
    )
    if forward.action_gradients:
        active_names = tuple(
            name for name, value in forward.candidate.items() if value.requires_grad
        )
        torch.autograd.backward(
            tuple(forward.candidate[name] for name in active_names),
            grad_tensors=tuple(
                action_scale * forward.action_gradients[name] for name in active_names
            ),
            retain_graph=True,
        )
    forward.structural_total.backward()
    _reduce_gradients(runtime.trainable_parameters, runtime.context.world_size)
    action_norm = (
        torch.stack(
            [
                gradient.float().square().sum()
                for gradient in forward.action_gradients.values()
            ]
        )
        .sum()
        .sqrt()
        if forward.action_gradients
        else forward.total.new_zeros(())
    )
    gradients = {
        "action": action_norm,
        "compiler": _module_gradient_norm(runtime.model.compiler),
        "process": _module_gradient_norm(
            runtime.model.compiler.static_process_interaction
        ),
        "factor": sum(
            parameter.grad.float().square().sum()
            for heads in (
                runtime.model.compiler.factor_a,
                runtime.model.compiler.factor_b,
            )
            for parameter in heads.parameters()
            if parameter.grad is not None
        ).sqrt(),
        "policy_teacher": _module_gradient_norm(runtime.model.policy_teacher),
        "visible_program": _module_gradient_norm(runtime.model.visible_program),
    }
    clip = float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"])
    gradients["total"] = torch.nn.utils.clip_grad_norm_(
        runtime.trainable_parameters, clip
    )
    finite = torch.stack([value.float() for value in gradients.values()]).sum()
    if not bool(torch.isfinite(finite)) or float(gradients["visible_program"]) != 0.0:
        raise RuntimeError("invalid mapping-diverse q_pi/compiler gradient")
    runtime.optimizer.step()
    runtime.scheduler.step()
    return gradients


_METRIC_NAMES = (
    "total",
    "structural_total",
    "action_policy_loss",
    "action_supervision_active",
    "action_supervision_weight",
    "action_lora_gradient_norm",
    "activation_effect",
    "activation_effect_disagreement",
    "activation_effect_active_fraction",
    "prior_shared_response",
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
    "mean_active_events",
    "q_pi_gate_mean",
    "q_pi_gate_min",
    "q_pi_gate_max",
)


def _update_record(
    runtime: "ECPStage1Runtime",
    *,
    task: Any,
    task_visit: int,
    packed: Any,
    forward: Stage1Forward,
    gradients: dict[str, torch.Tensor],
) -> dict[str, Any]:
    return _local_record(
        runtime,
        task=task,
        task_visit=task_visit,
        loss=forward.loss,
        total=forward.total,
        structural_total=forward.structural_total,
        action_policy_loss=forward.action_policy_loss,
        action_supervision_weight=forward.action_supervision_weight,
        action_lora_gradient_norm=gradients["action"],
        activation_effect=forward.activation_effect,
        prior_shared_response=forward.prior_shared_response,
        program={
            "value": forward.output.teacher.program,
            "gate": forward.output.teacher.evidence_gate,
            "demo_indices": packed.demo_indices,
            "sampled_frames": [
                int(packed.video_offsets[index + 1] - packed.video_offsets[index])
                for index in range(len(packed.demo_indices))
            ],
        },
        compilation=forward.output.consensus_compilation,
        panel_id=forward.cached.panel.panel_id,
        panel_kind=forward.cached.panel.kind,
    )


def _update_summary(
    runtime: "ECPStage1Runtime",
    *,
    forward: Stage1Forward,
    gradients: dict[str, torch.Tensor],
    records: list[dict[str, Any]],
    tick: float,
    run_started: float,
) -> dict[str, Any]:
    return {
        "task_visits": forward.task_visits,
        "optimizer_update": forward.task_visits // runtime.context.world_size,
        "objective_phase": forward.objective_phase,
        "objective_weights": forward.objective_weights,
        "means": {
            name: sum(float(row[name]) for row in records) / len(records)
            for name in _METRIC_NAMES
        },
        "gradient_norm_before_clip": float(gradients["total"]),
        "compiler_gradient_norm_before_clip": float(gradients["compiler"]),
        "factor_head_gradient_norm_before_clip": float(gradients["factor"]),
        "process_fusion_gradient_norm_before_clip": float(gradients["process"]),
        "policy_teacher_gradient_norm_before_clip": float(
            gradients["policy_teacher"]
        ),
        "visible_program_gradient_norm_before_clip": float(
            gradients["visible_program"]
        ),
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "update_seconds": time.monotonic() - tick,
        "elapsed_seconds": time.monotonic() - run_started,
        "max_cuda_allocated_bytes": int(
            torch.cuda.max_memory_allocated(runtime.context.device)
        ),
        "records": sorted(records, key=lambda row: int(row["task_ordinal"])),
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
    packed, encoded = encode_stage1_visit(
        runtime, task_ordinal=task_ordinal, task_visit=task_visit
    )
    forward = _forward_objective(
        runtime,
        task_ordinal=task_ordinal,
        task_visit=task_visit,
        cursor=cursor,
        packed=packed,
        encoded=encoded,
    )
    gradients = _apply_gradients(runtime, forward)
    local = _update_record(
        runtime,
        task=task,
        task_visit=task_visit,
        packed=packed,
        forward=forward,
        gradients=gradients,
    )
    records = _gather(local, runtime.context.world_size)
    return _update_summary(
        runtime,
        forward=forward,
        gradients=gradients,
        records=records,
        tick=tick,
        run_started=run_started,
    )
