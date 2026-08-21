"""Task-equal warm-start and closed-loop updates for functional outer credit."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.functional_adaptation.code_checkpoint import (
    code_writer_rng_state,
    save_code_writer_checkpoint,
)
from ember.functional_adaptation.objectives import functional_code_inference_loss
from ember.functional_adaptation.outer_credit import (
    outer_credit_surrogate_loss,
    paired_antithetic_code_credit,
)
from ember.functional_adaptation.outer_credit_training import OuterCreditRuntime
from ember.lora import copy_task_lora_state_
from ember.pi05_source_checkpoint import write_json_atomic
from ember.pi05_source_contract import append_jsonl
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
from ember.writer.data import pack_teacher_condition


def _packed_task(
    runtime: OuterCreditRuntime, task_id: int, demos: Sequence[int]
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    return pack_teacher_condition(
        runtime.video_store,
        task_id=task_id,
        demos=demos,
        language=runtime.language[task_id],
        device=runtime.context.device,
    )


def _writer_posterior(
    runtime: OuterCreditRuntime, packed: tuple[torch.Tensor, ...]
) -> Any:
    (
        frames,
        frame_indices,
        video_offsets,
        condition_video_offsets,
        language_tokens,
        language_mask,
        task_span_mask,
    ) = packed
    features, frame_condition_ids = runtime.writer.encode_features(
        policy=runtime.policy,
        frames=frames,
        video_offsets=video_offsets,
        condition_video_offsets=condition_video_offsets,
        language_tokens=language_tokens,
        language_mask=language_mask,
        task_span_mask=task_span_mask,
    )
    return runtime.writer.infer_features(
        features=features,
        frame_condition_ids=frame_condition_ids,
        frame_indices=frame_indices,
        video_offsets=video_offsets,
        condition_video_offsets=condition_video_offsets,
    )


def _warmstart_task(
    runtime: OuterCreditRuntime,
    *,
    task_id: int,
    demos: Sequence[int],
) -> dict[str, Any]:
    packed, video = _packed_task(runtime, task_id, demos)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        posterior = _writer_posterior(runtime, packed)
        loss = functional_code_inference_loss(
            posterior,
            runtime.target_codes[task_id],
            weights=runtime.settings["warmstart_loss_weights"],
        )
    (loss.total / len(runtime.tasks)).backward()
    return {
        "global_task_id": task_id,
        "teacher_demo_indices": list(demos),
        "K": int(video["K"]),
        "loss": float(loss.total.detach()),
        "combined_code_loss": float(loss.combined_code.detach()),
    }


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
    runtime: OuterCreditRuntime,
    *,
    task: RewardTask,
    adapter: Mapping[str, torch.Tensor],
    rollout_cursors: Sequence[int],
    environment_seeds: Sequence[int],
    initial_states: Sequence[Any],
) -> tuple[RewardTrajectory, ...]:
    if runtime.env_pool is None:
        raise ValueError("outer-credit environment pool is unavailable")
    copy_task_lora_state_(runtime.policy, adapter, runtime.lora_contract)
    environment = runtime.settings["environment"]
    return collect_paired_reward_arm_trajectories(
        envs=tuple(runtime.env_pool.get(task, lane=lane) for lane in range(2)),
        policy=runtime.policy,
        preprocess=runtime.processor,
        postprocess=runtime.processor.unnormalize_action,
        suite=task.suite,
        task_id=task.task_id,
        global_task_id=task.global_task_id,
        language=task.language,
        adaptation_seed=int(runtime.settings["seed"]),
        rollout_cursors=rollout_cursors,
        env_seeds=environment_seeds,
        policy_seed_root=int(runtime.settings["policy_noise_seed_root"]),
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


def _outer_task(
    runtime: OuterCreditRuntime,
    *,
    task_id: int,
    demos: Sequence[int],
    outer_macro: int,
) -> dict[str, Any]:
    if runtime.env_pool is None:
        raise ValueError("outer-credit environment pool is unavailable")
    packed, video = _packed_task(runtime, task_id, demos)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        code_before = _writer_posterior(runtime, packed).combined_code.float()
    seed = reward_credit_update_seed(
        int(runtime.settings["update_seed_root"]), task_id, outer_macro
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    epsilon = torch.randn(code_before.shape, generator=generator).to(
        runtime.context.device
    )
    sigma = float(runtime.settings["objective"]["sigma"])
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        plus_adapter = runtime.writer.fixed_decoder((code_before + sigma * epsilon)[0])
        minus_adapter = runtime.writer.fixed_decoder((code_before - sigma * epsilon)[0])
    task = runtime.reward_tasks[task_id]
    rollout_cursors = (outer_macro * 2, outer_macro * 2 + 1)
    environment_seeds = tuple(
        reward_credit_environment_seed(
            int(runtime.settings["environment_seed_root"]),
            task.suite,
            task.task_id,
            int(runtime.settings["seed"]),
            cursor,
        )
        for cursor in rollout_cursors
    )
    environment = runtime.settings["environment"]
    envs = tuple(runtime.env_pool.get(task, lane=lane) for lane in range(2))
    initial_states = capture_paired_initial_states(
        envs,
        environment_seeds,
        dummy_action=environment["dummy_action"],
        dummy_settling_steps=int(environment["dummy_settling_steps"]),
    )
    started = time.monotonic()
    try:
        plus = _arm_rollout(
            runtime,
            task=task,
            adapter=plus_adapter,
            rollout_cursors=rollout_cursors,
            environment_seeds=environment_seeds,
            initial_states=initial_states,
        )
        minus = _arm_rollout(
            runtime,
            task=task,
            adapter=minus_adapter,
            rollout_cursors=rollout_cursors,
            environment_seeds=environment_seeds,
            initial_states=initial_states,
        )
    finally:
        copy_task_lora_state_(
            runtime.policy, runtime.identity_state, runtime.lora_contract
        )
    objective = runtime.settings["objective"]
    credit = paired_antithetic_code_credit(
        plus,
        minus,
        epsilon,
        sigma=sigma,
        success_weight=float(objective["success_weight"]),
        progress_weight=float(objective["progress_weight"]),
        success_efficiency_weight=float(objective["success_efficiency_weight"]),
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        code = _writer_posterior(runtime, packed).combined_code
        loss, anchor = outer_credit_surrogate_loss(
            code,
            credit,
            anchor_code=runtime.target_codes[task_id],
            anchor_weight=float(objective["phase_code_anchor_weight"]),
        )
    (loss / len(runtime.tasks)).backward()
    return {
        "global_task_id": task_id,
        "suite": task.suite,
        "task_id": task.task_id,
        "teacher_demo_indices": list(demos),
        "K": int(video["K"]),
        "epsilon_seed": seed,
        "code_rms": float(code_before.square().mean().sqrt()),
        "mean_advantage": credit.mean_advantage,
        "plus_successes": credit.plus_successes,
        "minus_successes": credit.minus_successes,
        "plus_progress_mean": credit.plus_progress_mean,
        "minus_progress_mean": credit.minus_progress_mean,
        "anchor_loss": float(anchor.detach()),
        "rollout_seconds": time.monotonic() - started,
        "plus": [_trajectory_record(value) for value in plus],
        "minus": [_trajectory_record(value) for value in minus],
    }


def _sync_gradients(runtime: OuterCreditRuntime) -> float:
    active = tuple(
        parameter for parameter in runtime.trainable if parameter.grad is not None
    )
    if not active:
        raise ValueError("outer-credit Writer received no gradient")
    if runtime.context.world_size > 1:
        for parameter in active:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    value = torch.nn.utils.clip_grad_norm_(
        active, float(runtime.settings["optimizer"]["gradient_clip_norm"])
    )
    return float(value)


def _gather_records(
    runtime: OuterCreditRuntime, local: Sequence[Mapping[str, Any]]
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


def _checkpoint(runtime: OuterCreditRuntime, macro: int) -> None:
    local_rng = code_writer_rng_state()
    if runtime.context.world_size > 1:
        states: list[Any] | None = (
            [None] * runtime.context.world_size if runtime.context.is_main else None
        )
        dist.gather_object(local_rng, states, dst=0)
    else:
        states = [local_rng]
    if runtime.context.is_main:
        save_code_writer_checkpoint(
            output_dir=runtime.args.output_dir,
            macro=macro,
            world_size=runtime.context.world_size,
            writer=runtime.writer,
            optimizer=runtime.optimizer,
            scheduler=runtime.scheduler,
            metrics_rows=runtime.metrics_rows,
            rank_rng_states=states or (),
        )
    if runtime.context.world_size > 1:
        dist.barrier(device_ids=[runtime.context.local_rank])


def train(runtime: OuterCreditRuntime) -> None:
    started = time.monotonic()
    warmstart_macros = int(runtime.mode["warmstart_macros"])
    for macro in range(runtime.start_macro, runtime.stop_macro):
        stage = "functional_warmstart" if macro < warmstart_macros else "outer_credit"
        stage_macro = macro if stage == "functional_warmstart" else macro - warmstart_macros
        runtime.optimizer.zero_grad(set_to_none=True)
        local_records = []
        assignments = runtime.schedule.assignments(macro)
        shot_count = 1 + (
            macro
            % int(runtime.config["code_inference"]["training"]["dynamic_k_max"])
        )
        for visit in assignments[runtime.context.rank]:
            demos = visit.demos[:shot_count]
            record = (
                _warmstart_task(
                    runtime,
                    task_id=visit.task_id,
                    demos=demos,
                )
                if stage == "functional_warmstart"
                else _outer_task(
                    runtime,
                    task_id=visit.task_id,
                    demos=demos,
                    outer_macro=stage_macro,
                )
            )
            local_records.append(record)
        grad_norm = _sync_gradients(runtime)
        runtime.optimizer.step()
        runtime.scheduler.step()
        records = _gather_records(runtime, local_records)
        completed = macro + 1
        if runtime.context.is_main:
            records.sort(key=lambda value: int(value["global_task_id"]))
            if len(records) != len(runtime.tasks):
                raise ValueError("outer-credit macro lost task-equal coverage")
            row = {
                "macro": completed,
                "stage": stage,
                "stage_macro": stage_macro + 1,
                "gradient_norm_before_clip": grad_norm,
                "learning_rate": float(runtime.scheduler.get_last_lr()[0]),
                "elapsed_seconds": time.monotonic() - started,
                "max_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "task_records": records,
            }
            if stage == "functional_warmstart":
                row["mean_loss"] = sum(
                    float(value["loss"]) for value in records
                ) / len(records)
            else:
                row.update(
                    {
                        "plus_successes": sum(
                            int(value["plus_successes"]) for value in records
                        ),
                        "minus_successes": sum(
                            int(value["minus_successes"]) for value in records
                        ),
                        "nonzero_advantage_tasks": sum(
                            abs(float(value["mean_advantage"])) > 0
                            for value in records
                        ),
                        "mean_advantage": sum(
                            float(value["mean_advantage"]) for value in records
                        )
                        / len(records),
                    }
                )
            append_jsonl(runtime.metrics_path, row)
            runtime.metrics_rows += 1
            print(json.dumps(row, sort_keys=True), flush=True)
        if runtime.context.world_size > 1:
            rows = torch.tensor(
                runtime.metrics_rows,
                device=runtime.context.device,
                dtype=torch.long,
            )
            dist.broadcast(rows, src=0)
            runtime.metrics_rows = int(rows.item())
        if completed in runtime.checkpoint_macros:
            _checkpoint(runtime, completed)
    if runtime.context.is_main:
        write_json_atomic(
            runtime.args.output_dir / "completion.json",
            {
                "schema_version": "ember_functional_outer_credit_completion_v1",
                "completed_macro": runtime.stop_macro,
                "metrics_rows": runtime.metrics_rows,
                "mode_contract_complete": runtime.stop_macro
                == int(runtime.mode["total_macros"]),
                "formal_contract_complete": runtime.args.mode == "formal"
                and runtime.stop_macro == int(runtime.mode["total_macros"]),
                "elapsed_seconds": time.monotonic() - started,
                "content_hash_policy": "disabled_by_owner",
            },
        )
    runtime.video_store.close()
    if runtime.env_pool is not None:
        runtime.env_pool.close()
    if runtime.context.world_size > 1:
        dist.barrier(device_ids=[runtime.context.local_rank])
