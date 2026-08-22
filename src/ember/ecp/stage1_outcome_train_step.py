"""One task-equal OCPB macro with paired simulator credit and support anchors."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.ecp.compiler import select_compiled_state
from ember.ecp.stage1_data import pack_stage1_videos
from ember.ecp.stage1_objective import ecp_stage1_loss
from ember.ecp.stage1_outcome import (
    outcome_coordinate,
    outcome_surrogate_loss,
    perturbation_forward_kwargs,
    structured_outcome_perturbation,
)
from ember.ecp.stage1_support import policy_support_distillation_loss
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
        raise ValueError("OCPB environment pool is unavailable")
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
            runtime.config["outcome_calibration"]["policy_noise_seed_root"]
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
    outcome = runtime.config["outcome_calibration"]
    packed = pack_stage1_videos(
        store=runtime.video_store,
        ordinal=task.ordinal,
        visit=int(outcome["video_visit_root"]) + macro,
        seed=int(runtime.base_config["data"]["pair_seed"]),
        k=int(runtime.base_config["data"]["visible_videos_per_visit"]),
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


def _task_update(
    runtime: "ECPStage1OutcomeRuntime",
    *,
    task: Any,
    credit_index: int,
    coordinate: str,
) -> dict[str, Any]:
    if runtime.env_pool is None:
        raise ValueError("OCPB environment pool is unavailable")
    started = time.monotonic()
    packed, encoded, evidence = _encode_task(
        runtime, task=task, macro=credit_index
    )
    outcome = runtime.config["outcome_calibration"]
    sigma = float(outcome["sigma"][coordinate])
    epsilon_seed = reward_credit_update_seed(
        int(outcome["update_seed_root"]), task.global_task_id, credit_index
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        baseline = runtime.model(encoded, evidence, packed.video_group_ids)
        perturbation = structured_outcome_perturbation(
            baseline,
            coordinate=coordinate,
            sigma=sigma,
            seed=epsilon_seed,
        )
        plus_output = runtime.model(
            encoded,
            evidence,
            packed.video_group_ids,
            **perturbation_forward_kwargs(perturbation, plus=True),
        )
        minus_output = runtime.model(
            encoded,
            evidence,
            packed.video_group_ids,
            **perturbation_forward_kwargs(perturbation, plus=False),
        )
        plus_adapter = {
            name: value.detach()
            for name, value in select_compiled_state(
                plus_output.consensus_compilation.state, 0
            ).items()
        }
        minus_adapter = {
            name: value.detach()
            for name, value in select_compiled_state(
                minus_output.consensus_compilation.state, 0
            ).items()
        }
        base_coordinate = outcome_coordinate(baseline, coordinate)
        base_fraction = float(
            baseline.consensus_compilation.rank_replacement_fraction
        )
        plus_fraction = float(
            plus_output.consensus_compilation.rank_replacement_fraction
        )
        minus_fraction = float(
            minus_output.consensus_compilation.rank_replacement_fraction
        )
    reward_task = runtime.reward_tasks[task.ordinal]
    rollout_cursors = (credit_index * 2, credit_index * 2 + 1)
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
        sigma=sigma,
        success_weight=float(outcome["success_weight"]),
        progress_weight=float(outcome["progress_weight"]),
        success_efficiency_weight=float(outcome["success_efficiency_weight"]),
    )
    support_task = runtime.support_bank.task(task.ordinal)
    panel = support_task.panel_for_visit(
        int(outcome["support_visit_root"]) + credit_index
    )
    cached = runtime.support_panels[(task.ordinal, panel.panel_id)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = runtime.model(encoded, evidence, packed.video_group_ids)
        candidate = select_compiled_state(
            output.consensus_compilation.state, 0
        )
        support_loss = policy_support_distillation_loss(
            policy=runtime.policy,
            candidate_state=candidate,
            contract=runtime.contract,
            cached=cached,
            preservation=str(
                runtime.config["outcome_calibration"]["support_preservation"]
            ),
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
                for name, value in runtime.base_config["objective"]["weights"].items()
            },
        )
        surrogate = outcome_surrogate_loss(
            output,
            credit,
            coordinate=coordinate,
            weight=float(outcome["surrogate_weight"][coordinate]),
        )
        total = (functional.total + surrogate) / len(runtime.tasks)
    if not bool(torch.isfinite(total)):
        raise RuntimeError("non-finite OCPB task loss")
    total.backward()
    return {
        "rank": runtime.context.rank,
        "task_ordinal": task.ordinal,
        "global_task_id": task.global_task_id,
        "suite": task.suite,
        "task_id": task.task_id,
        "coordinate": coordinate,
        "credit_macro": credit_index + 1,
        "demo_indices": list(packed.demo_indices),
        "epsilon_seed": epsilon_seed,
        "active_coordinate_values": int(
            (perturbation.epsilon.detach() != 0).sum()
        ),
        "base_coordinate_rms": float(
            base_coordinate.detach().float().square().mean().sqrt()
        ),
        "base_replacement_fraction": base_fraction,
        "plus_replacement_fraction": plus_fraction,
        "minus_replacement_fraction": minus_fraction,
        "mean_advantage": credit.mean_advantage,
        "plus_successes": credit.plus_successes,
        "minus_successes": credit.minus_successes,
        "plus_progress_mean": credit.plus_progress_mean,
        "minus_progress_mean": credit.minus_progress_mean,
        "functional_total": float(functional.total.detach()),
        "functional_response": float(functional.functional_response.detach()),
        "shared_support": float(functional.shared_support.detach()),
        "source_support": float(functional.source_support.detach()),
        "outcome_surrogate": float(surrogate.detach()),
        "panel_id": panel.panel_id,
        "panel_kind": panel.kind,
        "task_seconds": time.monotonic() - started,
        "plus": [_trajectory_record(value) for value in plus],
        "minus": [_trajectory_record(value) for value in minus],
    }


def _sync_gradients(runtime: "ECPStage1OutcomeRuntime") -> float:
    for parameter in runtime.trainable_parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if runtime.context.world_size > 1:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    value = torch.nn.utils.clip_grad_norm_(
        runtime.trainable_parameters,
        float(
            runtime.config["optimization"]["optimizer"]["gradient_clip_norm"]
        ),
    )
    if not bool(torch.isfinite(value)):
        raise RuntimeError("non-finite OCPB gradient")
    return float(value)


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
    coordinate = str(
        runtime.config["outcome_calibration"]["coordinate_sequence"][macro]
    )
    credit_index = macro + int(
        runtime.config["outcome_calibration"]["credit_macro_offset"]
    )
    runtime.optimizer.zero_grad(set_to_none=True)
    local = [
        _task_update(
            runtime,
            task=task,
            credit_index=credit_index,
            coordinate=coordinate,
        )
        for task in runtime.local_tasks
    ]
    gradient_norm = _sync_gradients(runtime)
    runtime.optimizer.step()
    runtime.scheduler.step()
    records = _gather_records(runtime, local)
    completed = macro + 1
    if not runtime.context.is_main:
        return {"macro": completed, "coordinate": coordinate}
    records.sort(key=lambda row: int(row["task_ordinal"]))
    if len(records) != len(runtime.tasks):
        raise ValueError("OCPB macro lost task-equal coverage")
    return {
        "macro": completed,
        "credit_macro": credit_index + 1,
        "coordinate": coordinate,
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
        "mean_functional_total": sum(
            float(row["functional_total"]) for row in records
        )
        / len(records),
        "mean_outcome_surrogate": sum(
            float(row["outcome_surrogate"]) for row in records
        )
        / len(records),
        "gradient_norm_before_clip": gradient_norm,
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "elapsed_seconds": time.monotonic() - run_started,
        "max_cuda_allocated_bytes": int(
            torch.cuda.max_memory_allocated(runtime.context.device)
        ),
        "task_records": records,
    }
