"""Reward collection and Writer-only optimization loop."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.lora import copy_task_lora_state_, lora_state_sha256
from ember.pi05_source_checkpoint import barrier, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import reduce_max
from ember.reward.ledger import InteractionCursors, write_rollout_once
from ember.reward.loss import functional_executed_prefix_flow_loss
from ember.reward.protocol import (
    RewardProtocolError,
    RewardTask,
    environment_seed,
    update_seed,
)
from ember.reward.rollout import (
    RewardTrajectory,
    collect_randomized_reward_trajectory,
    successful_trajectory_batch,
)
from ember.rl_writer.checkpoint import save_rl_writer_checkpoint
from ember.rl_writer.contract import task_for_update
from ember.rl_writer.runtime import RLWriterRuntime, rank_ledger_summary


def _all_reduce_int(value: int, runtime: RLWriterRuntime) -> int:
    tensor = torch.tensor(value, dtype=torch.int64, device=runtime.context.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def _episode_chunk_weights(
    episode_ids: torch.Tensor, global_successes: int
) -> torch.Tensor:
    """Weight chunks so every successful episode has global weight 1/N."""

    if episode_ids.ndim != 1 or episode_ids.numel() == 0 or global_successes <= 0:
        raise RewardProtocolError("invalid RL-Writer episode weighting")
    ids = episode_ids.to(dtype=torch.long)
    unique, counts = torch.unique(ids, sorted=True, return_counts=True)
    if (
        not torch.equal(unique, torch.arange(unique.numel(), device=unique.device))
        or unique.numel() > global_successes
    ):
        raise RewardProtocolError("RL-Writer episode IDs changed")
    return counts[ids].reciprocal() / global_successes


def _all_reduce_writer_gradients(runtime: RLWriterRuntime) -> None:
    """Synchronize branch-dependent local gradients in one fixed collective."""

    gradients = []
    for parameter in runtime.writer.parameters():
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        gradients.append(parameter.grad)
    flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    offset = 0
    for gradient in gradients:
        count = gradient.numel()
        gradient.copy_(flat[offset : offset + count].view_as(gradient))
        offset += count


def _install_and_collect(
    runtime: RLWriterRuntime,
    task: RewardTask,
    cycle: int,
    visit: int,
    update: int,
) -> tuple[list[RewardTrajectory], int, str]:
    demo = runtime.video_schedule.demo_for_task_visit(task.global_task_id, visit)
    teacher = runtime.feature_store.load_one_video(
        language_task_id=task.global_task_id,
        video_task_id=task.global_task_id,
        demo_index=demo,
    )
    runtime.writer.eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        state = runtime.writer(
            teacher.language_features.to(runtime.context.device),
            teacher.video_features.to(runtime.context.device),
            teacher.episode_offsets,
        )
    copy_task_lora_state_(runtime.policy, state, runtime.lora_contract)
    adapter_sha = lora_state_sha256(state)
    count = int(runtime.config["algorithm"]["rollouts_per_task_update"])
    trajectories = []
    for offset in range(count):
        cursor = cycle * count + offset
        env_seed = environment_seed(
            int(runtime.config["rng"]["environment_seed_root"]),
            task.suite,
            task.task_id,
            int(runtime.config["optimization"]["seed"]),
            cursor,
        )
        trajectory = collect_randomized_reward_trajectory(
            env=runtime.env_pool.get(task),
            policy=runtime.policy,
            preprocess=runtime.processor,
            postprocess=runtime.processor.unnormalize_action,
            suite=task.suite,
            task_id=task.task_id,
            global_task_id=task.global_task_id,
            language=task.language,
            adaptation_seed=int(runtime.config["optimization"]["seed"]),
            rollout_cursor=cursor,
            env_seed=env_seed,
            policy_seed_root=int(runtime.config["rng"]["policy_noise_seed_root"]),
            device=runtime.context.device,
            max_horizon=task.horizon,
            dummy_settling_steps=10,
            dummy_action=runtime.config["environment"]["dummy_action"],
            action_execution_horizon=5,
            num_inference_steps=10,
        )
        row = {
            **trajectory.ledger_row(),
            "schema_version": "ember_pi05_rl_writer_rollout_v1",
            "producer_rank": runtime.context.rank,
            "update": update,
            "task_cycle": cycle,
            "teacher_demo_index": demo,
            "writer_lora_sha256": adapter_sha,
            "branch": runtime.args.branch,
        }
        write_rollout_once(
            runtime.args.output_dir, f"task_{task.global_task_id:03d}", row
        )
        trajectories.append(trajectory)
    return trajectories, demo, adapter_sha


def _reward_update(
    runtime: RLWriterRuntime,
    *,
    task: RewardTask,
    demo: int,
    trajectories: Sequence[RewardTrajectory],
    update: int,
    global_successes: int,
) -> tuple[float, float, Mapping[str, Any]]:
    teacher = runtime.feature_store.load_one_video(
        language_task_id=task.global_task_id,
        video_task_id=task.global_task_id,
        demo_index=demo,
    )
    seed = update_seed(
        int(runtime.config["rng"]["update_seed_root"]),
        task.suite,
        task.task_id,
        int(runtime.config["optimization"]["seed"]),
        runtime.cursors.optimizer_updates,
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    runtime.writer.train()
    runtime.optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated = runtime.writer(
            teacher.language_features.to(runtime.context.device),
            teacher.video_features.to(runtime.context.device),
            teacher.episode_offsets,
        )
        successful = [value for value in trajectories if value.success]
        if successful:
            batch, episode_ids = successful_trajectory_batch(
                successful, runtime.context.device
            )
            weights = _episode_chunk_weights(episode_ids, global_successes)
            proxy = {
                name: value.detach().requires_grad_(True)
                for name, value in generated.items()
            }
            chunk_batch = int(
                runtime.config["algorithm"]["reward_replay_chunk_batch_size"]
            )
            local_sum = torch.zeros((), device=runtime.context.device)
            detail_totals = {
                "successful_chunks": 0,
                "executed_action_steps": 0,
                "masked_unexecuted_action_steps": 0,
            }
            microbatches = 0
            for start in range(0, episode_ids.numel(), chunk_batch):
                stop = min(start + chunk_batch, episode_ids.numel())
                sliced = {name: value[start:stop] for name, value in batch.items()}
                per_chunk, observed = functional_executed_prefix_flow_loss(
                    runtime.policy,
                    proxy,
                    runtime.lora_contract,
                    sliced,
                )
                weighted = (per_chunk * weights[start:stop]).sum()
                weighted.backward()
                local_sum = local_sum + (
                    per_chunk.detach()
                    * weights[start:stop]
                    * global_successes
                ).sum()
                for name in detail_totals:
                    detail_totals[name] += int(observed[name])
                microbatches += 1
            state_gradients = tuple(proxy[name].grad for name in generated)
            if any(value is None for value in state_gradients):
                raise RewardProtocolError("RL-Writer generated LoRA gradient is incomplete")
            torch.autograd.backward(tuple(generated.values()), state_gradients)
            details = {
                **detail_totals,
                "successful_episodes": len(successful),
                "replay_microbatches": microbatches,
                "reward_replay_chunk_batch_size": chunk_batch,
                "loss": float(local_sum / len(successful)),
            }
        else:
            details = {
                "successful_episodes": 0,
                "successful_chunks": 0,
                "executed_action_steps": 0,
                "masked_unexecuted_action_steps": 0,
            }
            local_sum = sum(value.sum() for value in generated.values()) * 0.0
            local_sum.backward()
    if not bool(torch.isfinite(local_sum).detach()):
        raise RewardProtocolError(f"non-finite RL-Writer loss at update {update}")
    _all_reduce_writer_gradients(runtime)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        runtime.writer.parameters(),
        float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"]),
    )
    if not bool(torch.isfinite(grad_norm).detach()):
        raise RewardProtocolError(f"non-finite RL-Writer gradient at update {update}")
    runtime.optimizer.step()
    runtime.scheduler.step()
    return float(local_sum.detach()), float(grad_norm), {**details, "update_seed": seed}


def _record_step(
    runtime: RLWriterRuntime,
    *,
    started: float,
    cycle: int,
    task: RewardTask,
    demo: int,
    adapter_sha: str,
    global_rollouts: int,
    global_actions: int,
    global_successes: int,
    loss_sum: float,
    grad_norm: float,
    details: Mapping[str, Any],
    elapsed_ns: int,
) -> None:
    row = {
        "next_update": runtime.next_update,
        "task_cycle": cycle,
        "rank0_global_task_id": task.global_task_id,
        "rank0_teacher_demo_index": demo,
        "rank0_writer_lora_sha256": adapter_sha,
        "global_rollouts": global_rollouts,
        "global_environment_actions": global_actions,
        "global_successes": global_successes,
        "optimizer_update_cursor": runtime.cursors.optimizer_updates,
        "rank0_reward_flow_loss_sum": loss_sum,
        "rank0_gradient_norm_before_clip": grad_norm,
        "rank0_loss_details": dict(details),
        "step_seconds_max": reduce_max(elapsed_ns / 1e9, runtime.context),
        "elapsed_seconds": time.monotonic() - started,
        "max_cuda_reserved_bytes": int(
            reduce_max(
                torch.cuda.max_memory_reserved(runtime.context.device),
                runtime.context,
            )
        ),
    }
    if runtime.context.is_main:
        append_jsonl(runtime.metrics_path, row)
        runtime.metrics_rows += 1
        print(json.dumps(row, sort_keys=True), flush=True)


def _checkpoint_if_due(runtime: RLWriterRuntime) -> None:
    if runtime.next_update not in runtime.checkpoint_updates:
        return
    ledger = rank_ledger_summary(runtime, runtime.next_update)
    save_rl_writer_checkpoint(
        output_dir=runtime.args.output_dir,
        next_update=runtime.next_update,
        context=runtime.context,
        writer=runtime.writer,
        optimizer=runtime.optimizer,
        scheduler=runtime.scheduler,
        tasks=runtime.tasks,
        task_schedule_seed=int(runtime.config["data"]["task_schedule_seed"]),
        rollouts_per_task_update=int(
            runtime.config["algorithm"]["rollouts_per_task_update"]
        ),
        video_schedule=runtime.video_schedule,
        contract=runtime.contract,
        cursors=runtime.cursors,
        successes=runtime.successes,
        reward_sum=runtime.reward_sum,
        wall_nanoseconds=runtime.wall_nanoseconds,
        ledger_summary=ledger,
        metrics_rows=runtime.metrics_rows,
        formal=runtime.args.mode == "formal",
    )


def _publish_summary(runtime: RLWriterRuntime) -> None:
    barrier(runtime.context)
    wall_seconds = reduce_max(runtime.wall_nanoseconds / 1e9, runtime.context)
    if runtime.context.is_main:
        write_json_atomic(
            runtime.args.output_dir / "run_summary.json",
            {
                "schema_version": "ember_pi05_rl_writer_run_summary_v1",
                "contract_sha256": runtime.contract_sha256,
                "branch": runtime.args.branch,
                "complete": runtime.next_update == runtime.total_updates,
                "next_update": runtime.next_update,
                "teacher_action_queries": 0,
                "validation_reward_reads": 0,
                "validation_action_reads": 0,
                "test_reward_reads": 0,
                "test_action_reads": 0,
                "fixed_pruned_init_reads": 0,
                "optimizer_update_cursor": runtime.cursors.optimizer_updates,
                "wall_seconds_max_rank": wall_seconds,
            },
        )
    barrier(runtime.context)


def run_updates(runtime: RLWriterRuntime) -> None:
    started = time.monotonic()
    while runtime.next_update < runtime.args.stop_after_update:
        update = runtime.next_update
        tick = time.perf_counter_ns()
        task, cycle, visit = task_for_update(
            runtime.tasks,
            world_size=runtime.context.world_size,
            rank=runtime.context.rank,
            update=update,
            seed=int(runtime.config["data"]["task_schedule_seed"]),
        )
        trajectories, demo, adapter_sha = _install_and_collect(
            runtime, task, cycle, visit, update
        )
        local_rollouts = len(trajectories)
        local_actions = sum(value.steps for value in trajectories)
        local_successes = sum(value.success for value in trajectories)
        local_reward = sum(value.reward_sum for value in trajectories)
        global_successes = _all_reduce_int(local_successes, runtime)
        global_rollouts = _all_reduce_int(local_rollouts, runtime)
        global_actions = _all_reduce_int(local_actions, runtime)
        loss_sum = 0.0
        grad_norm = 0.0
        details: Mapping[str, Any] = {}
        if global_successes:
            loss_sum, grad_norm, details = _reward_update(
                runtime,
                task=task,
                demo=demo,
                trajectories=trajectories,
                update=update,
                global_successes=global_successes,
            )
            optimizer_updates = runtime.cursors.optimizer_updates + 1
        else:
            optimizer_updates = runtime.cursors.optimizer_updates
        torch.cuda.synchronize(runtime.context.device)
        elapsed_ns = time.perf_counter_ns() - tick
        runtime.next_update += 1
        runtime.cursors = InteractionCursors(
            rollout=runtime.cursors.rollout + local_rollouts,
            environment_actions=runtime.cursors.environment_actions + local_actions,
            optimizer_updates=optimizer_updates,
        )
        runtime.successes += local_successes
        runtime.reward_sum += local_reward
        runtime.wall_nanoseconds += elapsed_ns
        _record_step(
            runtime,
            started=started,
            cycle=cycle,
            task=task,
            demo=demo,
            adapter_sha=adapter_sha,
            global_rollouts=global_rollouts,
            global_actions=global_actions,
            global_successes=global_successes,
            loss_sum=loss_sum,
            grad_norm=grad_norm,
            details=details,
            elapsed_ns=elapsed_ns,
        )
        _checkpoint_if_due(runtime)
    _publish_summary(runtime)
