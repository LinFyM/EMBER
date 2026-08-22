"""One task-equal fixed-compiler Program-credit macro."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.ecp.compiler import select_compiled_state
from ember.ecp.contracts import TargetFamily
from ember.ecp.program import ECPProgram
from ember.ecp.stage1_data import pack_stage1_videos
from ember.ecp.stage1_objective import ecp_stage1_loss
from ember.ecp.stage1_outcome import (
    action_guided_program_leaf_gradient,
    action_guided_program_perturbation,
)
from ember.ecp.stage1_outcome_training import successful_panel_for_visit
from ember.ecp.stage1_support import (
    CachedPolicySupportPanel,
    policy_support_activation_distillation_loss,
)
from ember.lora import copy_task_lora_state_
from ember.reward.credit import paired_antithetic_credit
from ember.reward.protocol import RewardTask, reward_credit_environment_seed
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


def _leaf_program(program: ECPProgram, process: torch.Tensor) -> ECPProgram:
    return ECPProgram(
        language=program.language.detach(),
        scene=program.scene.detach(),
        process=process,
        presence=program.presence.detach(),
        uncertainty=program.uncertainty.detach(),
    )


def _program_action_gradient(
    runtime: "ECPStage1OutcomeRuntime",
    *,
    program: ECPProgram,
    panel: CachedPolicySupportPanel,
    task_ordinal: int,
    macro: int,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    process = program.process.detach().float().requires_grad_(True)
    leaf_program = _leaf_program(program, process)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        compilation = runtime.model.compiler(leaf_program)
        candidate = select_compiled_state(compilation.state, 0)
    action_loss, action_lora_gradients = _action_gradient(
        runtime,
        candidate=candidate,
        panel=panel,
        task_ordinal=task_ordinal,
        macro=macro,
    )
    chain = sum(
        (candidate[name].float() * action_lora_gradients[name].float()).sum()
        for name in candidate
    )
    program_gradient = torch.autograd.grad(chain, process)[0].detach()
    return (
        action_loss,
        action_lora_gradients,
        program_gradient,
        {name: value.detach() for name, value in candidate.items()},
    )


def _compile_program(
    runtime: "ECPStage1OutcomeRuntime", program: ECPProgram
) -> dict[str, torch.Tensor]:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = runtime.model.compiler(program)
    return {
        name: value.detach()
        for name, value in select_compiled_state(output.state, 0).items()
    }


def _tensor_dict_norm(values: Mapping[str, torch.Tensor]) -> float:
    return float(
        torch.stack([value.float().square().sum() for value in values.values()])
        .sum()
        .sqrt()
    )


def _compiled_relative_delta(
    base: Mapping[str, torch.Tensor],
    plus: Mapping[str, torch.Tensor],
    minus: Mapping[str, torch.Tensor],
) -> float:
    base_norm_sq = sum(value.float().square().sum() for value in base.values())
    delta_norm_sq = sum(
        ((plus[name].float() - minus[name].float()) * 0.5).square().sum()
        for name in base
    )
    return float((delta_norm_sq / base_norm_sq.clamp_min(1e-20)).sqrt())


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
    family = TargetFamily(
        outcome["family_sequence"][macro % len(outcome["family_sequence"])]
    )
    (
        action_loss,
        action_lora_gradients,
        action_program_gradient,
        base_adapter,
    ) = _program_action_gradient(
        runtime,
        program=baseline.teacher.program,
        panel=proposal_cached,
        task_ordinal=task.ordinal,
        macro=macro,
    )
    perturbation = action_guided_program_perturbation(
        baseline.teacher.program,
        action_program_gradient,
        runtime.owners,
        family=family,
        sigma=float(outcome["relative_program_sigma"]),
    )
    plus_adapter = _compile_program(runtime, perturbation.plus_program)
    minus_adapter = _compile_program(runtime, perturbation.minus_program)
    compiled_delta = _compiled_relative_delta(
        base_adapter, plus_adapter, minus_adapter
    )
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
            adapter=plus_adapter,
            rollout_cursors=rollout_cursors,
            environment_seeds=environment_seeds,
            initial_states=initial_states,
        )
        minus = _arm_rollout(
            runtime,
            task=reward_task,
            adapter=minus_adapter,
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
    program_leaf = action_guided_program_leaf_gradient(
        perturbation,
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
        outcome_surrogate = (
            output.teacher.program.process.float() * program_leaf.float()
        ).sum()
        total = (structural_total + outcome_surrogate) / len(runtime.tasks)
    if not bool(torch.isfinite(total)):
        raise RuntimeError("non-finite fixed-compiler Program task loss")
    total.backward()
    return {
        "rank": runtime.context.rank,
        "task_ordinal": task.ordinal,
        "global_task_id": task.global_task_id,
        "suite": task.suite,
        "task_id": task.task_id,
        "macro": macro + 1,
        "demo_indices": list(packed.demo_indices),
        "program_family": family.value,
        "program_owner_count": len(perturbation.owner_indices),
        "active_program_elements": perturbation.active_elements,
        "relative_program_sigma": perturbation.sigma,
        "compiled_relative_delta": compiled_delta,
        "action_policy_loss": float(action_loss),
        "action_lora_gradient_norm": _tensor_dict_norm(action_lora_gradients),
        "action_program_gradient_norm": float(
            action_program_gradient.float().square().sum().sqrt()
        ),
        "outcome_program_leaf_gradient_norm": float(
            program_leaf.float().square().sum().sqrt()
        ),
        "coordinate_gradient": float(credit.gradient[0, 0]),
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


def _module_gradient_norm(module: torch.nn.Module) -> torch.Tensor:
    values = [
        parameter.grad.float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not values:
        return next(module.parameters()).new_zeros((), dtype=torch.float32)
    return torch.stack(values).sum().sqrt()


def _sync_gradients(
    runtime: "ECPStage1OutcomeRuntime",
) -> tuple[float, float, float, float]:
    for parameter in runtime.trainable_parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if runtime.context.world_size > 1:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    teacher_gradient = _module_gradient_norm(runtime.model.policy_teacher)
    compiler_gradient = _module_gradient_norm(runtime.model.compiler)
    visible_gradient = _module_gradient_norm(runtime.model.visible_program)
    gradient = torch.nn.utils.clip_grad_norm_(
        runtime.trainable_parameters,
        float(
            runtime.config["optimization"]["optimizer"]["gradient_clip_norm"]
        ),
    )
    if (
        not bool(
            torch.isfinite(
                gradient + teacher_gradient + compiler_gradient + visible_gradient
            )
        )
        or float(compiler_gradient) != 0.0
        or float(visible_gradient) != 0.0
    ):
        raise RuntimeError("fixed Stage 1 coordinate received an invalid gradient")
    return (
        float(gradient),
        float(teacher_gradient),
        float(compiler_gradient),
        float(visible_gradient),
    )


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
    (
        gradient_norm,
        teacher_gradient,
        compiler_gradient,
        visible_gradient,
    ) = _sync_gradients(runtime)
    runtime.optimizer.step()
    runtime.scheduler.step()
    records = _gather_records(runtime, local)
    completed = macro + 1
    if not runtime.context.is_main:
        return {"macro": completed}
    records.sort(key=lambda row: int(row["task_ordinal"]))
    if len(records) != len(runtime.tasks):
        raise ValueError("fixed-compiler macro lost task-equal coverage")
    return {
        "macro": completed,
        "program_family": records[0]["program_family"],
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
        "mean_action_program_gradient_norm": sum(
            float(row["action_program_gradient_norm"]) for row in records
        )
        / len(records),
        "mean_compiled_relative_delta": sum(
            float(row["compiled_relative_delta"]) for row in records
        )
        / len(records),
        "mean_structural_total": sum(
            float(row["structural_total"]) for row in records
        )
        / len(records),
        "mean_outcome_program_leaf_gradient_norm": sum(
            float(row["outcome_program_leaf_gradient_norm"]) for row in records
        )
        / len(records),
        "gradient_norm_before_clip": gradient_norm,
        "policy_teacher_gradient_norm_before_clip": teacher_gradient,
        "compiler_gradient_norm_before_clip": compiler_gradient,
        "visible_program_gradient_norm_before_clip": visible_gradient,
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "elapsed_seconds": time.monotonic() - run_started,
        "max_cuda_allocated_bytes": int(
            torch.cuda.max_memory_allocated(runtime.context.device)
        ),
        "task_records": records,
    }
