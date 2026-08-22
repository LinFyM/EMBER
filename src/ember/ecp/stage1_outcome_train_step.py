"""One task-equal v18 macro with action-guided paired simulator credit."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.ecp.compiler import select_compiled_state
from ember.ecp.stage1_data import pack_stage1_videos
from ember.ecp.stage1_objective import ecp_stage1_loss
from ember.ecp.stage1_outcome import (
    action_guided_factor_perturbation,
    action_guided_outcome_leaf_gradients,
)
from ember.ecp.stage1_outcome_training import successful_panel_for_visit
from ember.ecp.stage1_support import (
    CachedPolicySupportPanel,
    policy_support_activation_distillation_loss,
)
from ember.lora import copy_task_lora_state_
from ember.reward.credit import paired_antithetic_credit
from ember.reward.protocol import (
    RewardTask,
    reward_credit_environment_seed,
    reward_credit_update_seed,
)
from ember.reward.rollout import (
    RewardTrajectory,
    capture_paired_initial_states,
    collect_paired_reward_arm_trajectories,
)
from ember.writer.functional import functional_lora_loss_gradient

if TYPE_CHECKING:
    from ember.ecp.stage1_outcome_training import ECPStage1OutcomeRuntime


def _trajectory_record(value: RewardTrajectory) -> dict[str, Any]:
    return {
        "rollout_cursor": value.rollout_cursor,
        "environment_seed": value.env_seed,
        "success": value.success,
        "steps": value.steps,
        "goal_predicate_count": value.goal_predicate_count,
        "goal_predicate_peak": value.goal_predicate_peak,
    }


def _arm_rollout(
    runtime: "ECPStage1OutcomeRuntime",
    *,
    task: RewardTask,
    adapter: Mapping[str, torch.Tensor],
    rollout_cursors: Sequence[int],
    environment_seeds: Sequence[int],
    initial_states: Sequence[Any],
) -> tuple[RewardTrajectory, ...]:
    if runtime.env_pool is None:
        raise ValueError("ECP Stage 1 environment pool is unavailable")
    copy_task_lora_state_(runtime.policy, adapter, runtime.contract)
    environment = runtime.config["environment"]
    return collect_paired_reward_arm_trajectories(
        envs=tuple(runtime.env_pool.get(task, lane=lane) for lane in range(2)),
        policy=runtime.policy,
        preprocess=runtime.processor,
        postprocess=runtime.processor.unnormalize_action,
        suite=task.suite,
        task_id=task.task_id,
        global_task_id=task.global_task_id,
        language=task.language,
        adaptation_seed=int(runtime.config["optimization"]["seed"]),
        rollout_cursors=rollout_cursors,
        env_seeds=environment_seeds,
        policy_seed_root=int(
            runtime.config["outcome_binding"]["policy_noise_seed_root"]
        ),
        device=runtime.context.device,
        max_horizon=task.horizon,
        dummy_settling_steps=int(environment["dummy_settling_steps"]),
        dummy_action=environment["dummy_action"],
        action_execution_horizon=int(environment["action_execution_horizon"]),
        num_inference_steps=int(environment["num_inference_steps"]),
        initial_states=initial_states,
        capture_replay=False,
        capture_goal_progress=True,
    )


def _encode_task(
    runtime: "ECPStage1OutcomeRuntime", *, task: Any, macro: int
) -> tuple[Any, Any, Any]:
    packed = pack_stage1_videos(
        store=runtime.video_store,
        ordinal=task.ordinal,
        visit=int(runtime.config["outcome_binding"]["video_visit_root"]) + macro,
        seed=int(runtime.config["data"]["pair_seed"]),
        k=int(runtime.config["data"]["visible_videos_per_visit"]),
        device=runtime.context.device,
    )
    tokens, mask = runtime.language_tokens[task.ordinal]
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
        task.ordinal, runtime.support_bank.task(task.ordinal)
    )
    return packed, encoded, evidence


def _policy_seed(
    runtime: "ECPStage1OutcomeRuntime",
    *,
    task_ordinal: int,
    macro: int,
    panel_id: int,
) -> int:
    base = int(runtime.config["objective"]["train_policy_seed"])
    return (
        base
        + (task_ordinal + 1) * 1_000_003
        + (macro + 1) * 10_007
        + (panel_id + 1) * 101
    ) % ((1 << 63) - 1)


def _action_gradient(
    runtime: "ECPStage1OutcomeRuntime",
    *,
    candidate: Mapping[str, torch.Tensor],
    panel: CachedPolicySupportPanel,
    task_ordinal: int,
    macro: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    objective = runtime.config["objective"]
    loss, _, gradients = functional_lora_loss_gradient(
        runtime.policy,
        candidate,
        runtime.contract,
        batch=panel.batch,
        policy_rng_seed=_policy_seed(
            runtime,
            task_ordinal=task_ordinal,
            macro=macro,
            panel_id=panel.panel.panel_id,
        ),
        policy_rng_device=runtime.context.device,
        flow_time_sampling_scheme=str(objective["policy_flow_time_sampling_scheme"]),
        flow_noise_sampling_scheme=str(
            objective["policy_flow_noise_sampling_scheme"]
        ),
        policy_microbatch_size=int(
            runtime.config["optimization"]["functional_policy_microbatch_size"]
        ),
        collect_policy_details=False,
    )
    return loss, gradients


def _tensor_dict_norm(values: Mapping[str, torch.Tensor]) -> float:
    return float(
        torch.stack([value.float().square().sum() for value in values.values()])
        .sum()
        .sqrt()
    )


def _task_update(
    runtime: "ECPStage1OutcomeRuntime",
    *,
    task: Any,
    macro: int,
) -> dict[str, Any]:
    if runtime.env_pool is None:
        raise ValueError("ECP Stage 1 environment pool is unavailable")
    started = time.monotonic()
    packed, encoded, evidence = _encode_task(runtime, task=task, macro=macro)
    outcome = runtime.config["outcome_binding"]
    support_task = runtime.support_bank.task(task.ordinal)
    proposal_panel = successful_panel_for_visit(
        support_task, int(outcome["proposal_visit_root"]) + macro
    )
    proposal_cached = runtime.support_panels[
        (task.ordinal, proposal_panel.panel_id)
    ]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        baseline = runtime.model(encoded, evidence, packed.video_group_ids)
        base_adapter = {
            name: value.detach()
            for name, value in select_compiled_state(
                baseline.consensus_compilation.state, 0
            ).items()
        }
    with torch.autocast("cuda", dtype=torch.bfloat16):
        action_loss, action_gradients = _action_gradient(
            runtime,
            candidate=base_adapter,
            panel=proposal_cached,
            task_ordinal=task.ordinal,
            macro=macro,
        )
    epsilon_seed = reward_credit_update_seed(
        int(outcome["update_seed_root"]), task.global_task_id, macro
    )
    perturbation = action_guided_factor_perturbation(
        base_adapter,
        action_gradients,
        runtime.owners,
        sigma=float(outcome["relative_factor_sigma"]),
        seed=epsilon_seed,
    )
    if perturbation.active_owners < int(outcome["minimum_active_owners"]):
        raise RuntimeError("exact action proposal did not reach every target owner")
    reward_task = runtime.reward_tasks[task.ordinal]
    rollout_cursors = (macro * 2, macro * 2 + 1)
    environment_seeds = tuple(
        reward_credit_environment_seed(
            int(outcome["environment_seed_root"]),
            reward_task.suite,
            reward_task.task_id,
            int(runtime.config["optimization"]["seed"]),
            cursor,
        )
        for cursor in rollout_cursors
    )
    environment = runtime.config["environment"]
    envs = tuple(
        runtime.env_pool.get(reward_task, lane=lane) for lane in range(2)
    )
    initial_states = capture_paired_initial_states(
        envs,
        environment_seeds,
        dummy_action=environment["dummy_action"],
        dummy_settling_steps=int(environment["dummy_settling_steps"]),
    )
    try:
        plus = _arm_rollout(
            runtime,
            task=reward_task,
            adapter=perturbation.plus_state,
            rollout_cursors=rollout_cursors,
            environment_seeds=environment_seeds,
            initial_states=initial_states,
        )
        minus = _arm_rollout(
            runtime,
            task=reward_task,
            adapter=perturbation.minus_state,
            rollout_cursors=rollout_cursors,
            environment_seeds=environment_seeds,
            initial_states=initial_states,
        )
    finally:
        copy_task_lora_state_(
            runtime.policy, runtime.identity_state, runtime.contract
        )
    credit = paired_antithetic_credit(
        plus,
        minus,
        perturbation.epsilon,
        sigma=perturbation.sigma,
        success_weight=float(outcome["success_weight"]),
        progress_weight=float(outcome["progress_weight"]),
        success_efficiency_weight=float(outcome["success_efficiency_weight"]),
    )
    leaf_gradients = action_guided_outcome_leaf_gradients(
        perturbation,
        runtime.owners,
        credit.gradient,
        weight=float(outcome["leaf_gradient_weight"]),
    )
    anchor_panel = support_task.panel_for_visit(
        int(outcome["support_visit_root"]) + macro
    )
    anchor_cached = runtime.support_panels[(task.ordinal, anchor_panel.panel_id)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = runtime.model(encoded, evidence, packed.video_group_ids)
        candidate = select_compiled_state(
            output.consensus_compilation.state, 0
        )
        support_loss, activation_effect = (
            policy_support_activation_distillation_loss(
                policy=runtime.policy,
                candidate_state=candidate,
                contract=runtime.contract,
                cached=anchor_cached,
                preservation=str(outcome["support_preservation"]),
            )
        )
        functional = ecp_stage1_loss(
            member=output.member_compilation,
            consensus=output.consensus_compilation,
            prior=output.prior_compilation,
            expert_states=evidence.member_states,
            prior_target=runtime.prior_state,
            contract=runtime.contract,
            policy_support=support_loss,
            weights={
                name: float(value)
                for name, value in runtime.config["objective"]["weights"].items()
            },
        )
        structural_total = functional.total + float(
            runtime.config["objective"]["activation_effect_distillation_weight"]
        ) * activation_effect.loss
        outcome_surrogate = sum(
            (candidate[name].float() * leaf_gradients[name].float()).sum()
            for name in candidate
        )
        total = (structural_total + outcome_surrogate) / len(runtime.tasks)
    if not bool(torch.isfinite(total)):
        raise RuntimeError("non-finite action-guided Stage 1 task loss")
    total.backward()
    return {
        "rank": runtime.context.rank,
        "task_ordinal": task.ordinal,
        "global_task_id": task.global_task_id,
        "suite": task.suite,
        "task_id": task.task_id,
        "macro": macro + 1,
        "demo_indices": list(packed.demo_indices),
        "epsilon_seed": epsilon_seed,
        "active_owners": perturbation.active_owners,
        "relative_factor_sigma": perturbation.sigma,
        "action_policy_loss": float(action_loss),
        "action_lora_gradient_norm": _tensor_dict_norm(action_gradients),
        "outcome_leaf_gradient_norm": _tensor_dict_norm(leaf_gradients),
        "mean_abs_coordinate_gradient": float(
            credit.gradient.abs().mean()
        ),
        "mean_advantage": credit.mean_advantage,
        "plus_successes": credit.plus_successes,
        "minus_successes": credit.minus_successes,
        "plus_progress_mean": credit.plus_progress_mean,
        "minus_progress_mean": credit.minus_progress_mean,
        "structural_total": float(structural_total.detach()),
        "functional_response": float(functional.functional_response.detach()),
        "shared_support": float(functional.shared_support.detach()),
        "source_support": float(functional.source_support.detach()),
        "activation_effect": float(activation_effect.loss.detach()),
        "outcome_surrogate": float(outcome_surrogate.detach()),
        "proposal_panel_id": proposal_panel.panel_id,
        "proposal_panel_kind": proposal_panel.kind,
        "anchor_panel_id": anchor_panel.panel_id,
        "anchor_panel_kind": anchor_panel.kind,
        "task_seconds": time.monotonic() - started,
        "plus": [_trajectory_record(value) for value in plus],
        "minus": [_trajectory_record(value) for value in minus],
    }


def _sync_gradients(runtime: "ECPStage1OutcomeRuntime") -> tuple[float, float]:
    for parameter in runtime.trainable_parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if runtime.context.world_size > 1:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    factor_gradient = sum(
        parameter.grad.float().square().sum()
        for heads in (
            runtime.model.compiler.factor_a,
            runtime.model.compiler.factor_b,
        )
        for parameter in heads.parameters()
    ).sqrt()
    gradient = torch.nn.utils.clip_grad_norm_(
        runtime.trainable_parameters,
        float(
            runtime.config["optimization"]["optimizer"]["gradient_clip_norm"]
        ),
    )
    if not bool(torch.isfinite(gradient + factor_gradient)):
        raise RuntimeError("non-finite action-guided Stage 1 gradient")
    return float(gradient), float(factor_gradient)


def _gather_records(
    runtime: "ECPStage1OutcomeRuntime", local: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if runtime.context.world_size == 1:
        return [dict(value) for value in local]
    gathered: list[Any] | None = (
        [None] * runtime.context.world_size if runtime.context.is_main else None
    )
    dist.gather_object(list(local), gathered, dst=0)
    if not runtime.context.is_main:
        return []
    return [dict(row) for group in gathered or () for row in group]


def run_outcome_macro(
    runtime: "ECPStage1OutcomeRuntime",
    *,
    macro: int,
    run_started: float,
) -> dict[str, Any]:
    runtime.optimizer.zero_grad(set_to_none=True)
    local = [
        _task_update(runtime, task=task, macro=macro)
        for task in runtime.local_tasks
    ]
    gradient_norm, factor_gradient = _sync_gradients(runtime)
    runtime.optimizer.step()
    runtime.scheduler.step()
    records = _gather_records(runtime, local)
    completed = macro + 1
    if not runtime.context.is_main:
        return {"macro": completed}
    records.sort(key=lambda row: int(row["task_ordinal"]))
    if len(records) != len(runtime.tasks):
        raise ValueError("action-guided macro lost task-equal coverage")
    return {
        "macro": completed,
        "plus_successes": sum(int(row["plus_successes"]) for row in records),
        "minus_successes": sum(int(row["minus_successes"]) for row in records),
        "nonzero_advantage_tasks": sum(
            abs(float(row["mean_advantage"])) > 0 for row in records
        ),
        "mean_advantage": sum(float(row["mean_advantage"]) for row in records)
        / len(records),
        "mean_plus_progress": sum(
            float(row["plus_progress_mean"]) for row in records
        )
        / len(records),
        "mean_minus_progress": sum(
            float(row["minus_progress_mean"]) for row in records
        )
        / len(records),
        "mean_action_policy_loss": sum(
            float(row["action_policy_loss"]) for row in records
        )
        / len(records),
        "mean_structural_total": sum(
            float(row["structural_total"]) for row in records
        )
        / len(records),
        "mean_outcome_leaf_gradient_norm": sum(
            float(row["outcome_leaf_gradient_norm"]) for row in records
        )
        / len(records),
        "gradient_norm_before_clip": gradient_norm,
        "factor_head_gradient_norm_before_clip": factor_gradient,
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "elapsed_seconds": time.monotonic() - run_started,
        "max_cuda_allocated_bytes": int(
            torch.cuda.max_memory_allocated(runtime.context.device)
        ),
        "task_records": records,
    }
