"""Full-24 rollout collection and task-relative Flow-Credit updates."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from lerobot.utils.constants import ACTION

from ember.lora import copy_task_lora_state_, lora_state_sha256
from ember.pi05_source_checkpoint import barrier, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import reduce_max
from ember.reward.ledger import InteractionCursors, write_rollout_once
from ember.reward.loss import functional_executed_prefix_flow_loss
from ember.reward.protocol import RewardProtocolError, RewardTask, environment_seed
from ember.reward.rollout import (
    RewardTrajectory,
    collect_randomized_reward_trajectory,
    complete_trajectory_batch,
)
from ember.rl_writer.checkpoint import save_rl_writer_checkpoint
from ember.rl_writer.contract import cycle_assignments
from ember.rl_writer.flow_credit import (
    generated_lora_gradient_norm,
    task_relative_aspo_loss,
)
from ember.rl_writer.progress_credit import (
    binary_first_progress_advantages,
    write_task_progress_credit_once,
)
from ember.rl_writer.progress_observer import observe_correct_teacher_progress
from ember.rl_writer.rendezvous import rank_local_credit_ready
from ember.rl_writer.runtime import RLWriterRuntime, rank_ledger_summary


@dataclass
class TaskCycleReplay:
    task: RewardTask
    demo_index: int
    adapter_sha256: str
    frames: torch.Tensor
    frame_indices: torch.Tensor
    batch: dict[str, torch.Tensor]
    episode_ids: torch.Tensor
    successes: torch.Tensor
    utilities: torch.Tensor
    advantages: torch.Tensor
    credit_mode: str
    old_losses: torch.Tensor | None
    rollout_count: int
    environment_actions: int
    reward_sum: float


@dataclass
class CollectedTaskTrajectories:
    task: RewardTask
    demo_index: int
    adapter_sha256: str
    frames: torch.Tensor
    frame_indices: torch.Tensor
    state: dict[str, torch.Tensor]
    trajectories: list[RewardTrajectory]


def _all_reduce_writer_gradients(
    runtime: RLWriterRuntime, *, cycle: int, epoch: int
) -> None:
    gradients = []
    for parameter in runtime.writer.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        gradients.append(parameter.grad)
    flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
    rank_local_credit_ready(runtime, cycle=cycle, epoch=epoch)
    if runtime.context.world_size > 1:
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    offset = 0
    for gradient in gradients:
        count = gradient.numel()
        gradient.copy_(flat[offset : offset + count].view_as(gradient))
        offset += count


def _writer_state(
    runtime: RLWriterRuntime,
    *,
    task: RewardTask,
    frames: torch.Tensor,
    frame_indices: torch.Tensor,
    train: bool,
) -> dict[str, torch.Tensor]:
    copy_task_lora_state_(
        runtime.policy, runtime.identity_state, runtime.lora_contract
    )
    tokens, masks, task_spans = runtime.tokenizer([task.language])
    device = runtime.context.device
    frames = frames.to(device, non_blocking=True)
    indices = frame_indices.to(device, non_blocking=True)
    offsets = torch.tensor([0, frames.shape[0]], dtype=torch.long, device=device)
    runtime.writer.train(train)
    runtime.writer.semantic_encoder.eval()
    context = torch.enable_grad() if train else torch.inference_mode()
    with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = runtime.writer(
            frames,
            indices,
            offsets,
            tokens,
            masks,
            task_spans,
            policy=runtime.policy,
        )
    return state


def _flow_seed(
    runtime: RLWriterRuntime,
    *,
    cycle: int,
    task_id: int,
    mc_index: int,
    chunk_index: int,
) -> int:
    payload = (
        f"flow-credit-v1|{runtime.config['rng']['flow_credit_seed_root']}|"
        f"{cycle}|{task_id}|{mc_index}|{chunk_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _keyed_flow_samples(
    runtime: RLWriterRuntime,
    *,
    cycle: int,
    task_id: int,
    mc_index: int,
    start: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_size = int(runtime.policy.config.chunk_size)
    action_dim = int(runtime.policy.config.max_action_dim)
    noises = []
    times = []
    for offset in range(batch_size):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _flow_seed(
                runtime,
                cycle=cycle,
                task_id=task_id,
                mc_index=mc_index,
                chunk_index=start + offset,
            )
        )
        noises.append(
            torch.randn(
                (chunk_size, action_dim),
                dtype=torch.float32,
                generator=generator,
            )
        )
        uniform = torch.rand((), dtype=torch.float32, generator=generator)
        times.append(uniform.pow(2.0 / 3.0).mul(0.999).add(0.001))
    return (
        torch.stack(noises).to(runtime.context.device, non_blocking=True),
        torch.stack(times).to(runtime.context.device, non_blocking=True),
    )


def _batch_slice(
    batch: Mapping[str, torch.Tensor],
    start: int,
    stop: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: value[start:stop].to(device, non_blocking=True)
        for name, value in batch.items()
    }


def _old_loss_matrix(
    runtime: RLWriterRuntime,
    *,
    state: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    task: RewardTask,
    cycle: int,
) -> torch.Tensor:
    count = int(batch[ACTION].shape[0])
    n_mc = int(runtime.config["algorithm"]["flow_mc_samples"])
    microbatch = int(runtime.config["algorithm"]["reward_replay_chunk_batch_size"])
    result = torch.empty((n_mc, count), dtype=torch.float32)
    detached = {name: value.detach() for name, value in state.items()}
    with torch.no_grad():
        for mc_index in range(n_mc):
            for start in range(0, count, microbatch):
                stop = min(start + microbatch, count)
                sliced = _batch_slice(batch, start, stop, runtime.context.device)
                noise, flow_time = _keyed_flow_samples(
                    runtime,
                    cycle=cycle,
                    task_id=task.global_task_id,
                    mc_index=mc_index,
                    start=start,
                    batch_size=stop - start,
                )
                losses, _ = functional_executed_prefix_flow_loss(
                    runtime.policy,
                    detached,
                    runtime.lora_contract,
                    sliced,
                    noise=noise,
                    time=flow_time,
                )
                result[mc_index, start:stop].copy_(losses.float().cpu())
                del sliced, noise, flow_time, losses
    return result


def collect_task_trajectories(
    runtime: RLWriterRuntime,
    task: RewardTask,
    cycle: int,
) -> CollectedTaskTrajectories:
    demo = runtime.video_schedule.demo_for_task_visit(task.global_task_id, cycle)
    video = runtime.video_store.load(task.global_task_id, demo)
    frames = torch.from_numpy(video.frames)
    frame_indices = torch.from_numpy(video.frame_indices)
    state = _writer_state(
        runtime,
        task=task,
        frames=frames,
        frame_indices=frame_indices,
        train=False,
    )
    copy_task_lora_state_(runtime.policy, state, runtime.lora_contract)
    adapter_sha = lora_state_sha256(state)
    count = int(runtime.config["algorithm"]["rollouts_per_task_condition"])
    trajectories: list[RewardTrajectory] = []
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
            retain_failure_replay=True,
        )
        row = {
            **trajectory.ledger_row(),
            "schema_version": str(runtime.config["algorithm"]["rollout_schema"]),
            "failure_replay_retained": True,
            "producer_rank": runtime.context.rank,
            "outer_cycle": cycle,
            "teacher_demo_index": demo,
            "writer_lora_sha256": adapter_sha,
        }
        write_rollout_once(
            runtime.args.output_dir, f"task_{task.global_task_id:03d}", row
        )
        trajectories.append(trajectory)
    return CollectedTaskTrajectories(
        task=task,
        demo_index=demo,
        adapter_sha256=adapter_sha,
        frames=frames,
        frame_indices=frame_indices,
        state=state,
        trajectories=trajectories,
    )


def _install_and_collect(
    runtime: RLWriterRuntime,
    task: RewardTask,
    cycle: int,
) -> TaskCycleReplay:
    collected = collect_task_trajectories(runtime, task, cycle)
    trajectories = collected.trajectories
    batch, episode_ids, successes = complete_trajectory_batch(
        trajectories, torch.device("cpu")
    )
    utilities = torch.zeros_like(successes)
    if not bool(successes.any()):
        utilities, _ = observe_correct_teacher_progress(
            writer=runtime.writer,
            policy=runtime.policy,
            identity_state=runtime.identity_state,
            lora_contract=runtime.lora_contract,
            tokenizer=runtime.tokenizer,
            task=task,
            teacher_frames=collected.frames,
            trajectories=trajectories,
            device=runtime.context.device,
            normalization_epsilon=float(
                runtime.config["progress_credit"]["normalization_epsilon"]
            ),
            projection_epsilon=float(
                runtime.config["progress_credit"]["projection_epsilon"]
            ),
        )
    advantages, credit_mode = binary_first_progress_advantages(successes, utilities)
    active = bool(advantages.count_nonzero())
    old_losses = (
        _old_loss_matrix(
            runtime,
            state=collected.state,
            batch=batch,
            task=task,
            cycle=cycle,
        )
        if active
        else None
    )
    write_task_progress_credit_once(
        output_dir=runtime.args.output_dir,
        producer_rank=runtime.context.rank,
        outer_cycle=cycle,
        global_task_id=task.global_task_id,
        teacher_demo_index=collected.demo_index,
        successes=successes,
        utilities=utilities,
        advantages=advantages,
        credit_mode=credit_mode,
    )
    return TaskCycleReplay(
        task=task,
        demo_index=collected.demo_index,
        adapter_sha256=collected.adapter_sha256,
        frames=collected.frames,
        frame_indices=collected.frame_indices,
        batch=batch,
        episode_ids=episode_ids,
        successes=successes,
        utilities=utilities,
        advantages=advantages,
        credit_mode=credit_mode,
        old_losses=old_losses,
        rollout_count=len(trajectories),
        environment_actions=sum(value.steps for value in trajectories),
        reward_sum=sum(value.reward_sum for value in trajectories),
    )


def _task_epoch_backward(
    runtime: RLWriterRuntime,
    replay: TaskCycleReplay,
    *,
    cycle: int,
) -> dict[str, float]:
    if replay.old_losses is None:
        return {
            "objective": 0.0,
            "ratio_sum": 0.0,
            "ratio_count": 0.0,
            "ratio_min": math.inf,
            "ratio_max": -math.inf,
            "positive_clipped": 0.0,
            "positive_count": 0.0,
            "active_credit_tasks": 0.0,
            "all_failure_semantic_tasks": 0.0,
            "all_failure_nonzero_lora_grad_tasks": 0.0,
        }
    generated = _writer_state(
        runtime,
        task=replay.task,
        frames=replay.frames,
        frame_indices=replay.frame_indices,
        train=True,
    )
    proxy = {
        name: value.detach().requires_grad_(True)
        for name, value in generated.items()
    }
    n_mc = int(runtime.config["algorithm"]["flow_mc_samples"])
    microbatch = int(runtime.config["algorithm"]["reward_replay_chunk_batch_size"])
    count = int(replay.batch[ACTION].shape[0])
    episode_counts = torch.bincount(
        replay.episode_ids, minlength=replay.successes.numel()
    )
    algorithm = runtime.config["algorithm"]
    totals = {
        "objective": 0.0,
        "ratio_sum": 0.0,
        "ratio_count": 0.0,
        "ratio_min": math.inf,
        "ratio_max": -math.inf,
        "positive_clipped": 0.0,
        "positive_count": 0.0,
    }
    for mc_index in range(n_mc):
        for start in range(0, count, microbatch):
            stop = min(start + microbatch, count)
            sliced = _batch_slice(
                replay.batch, start, stop, runtime.context.device
            )
            noise, flow_time = _keyed_flow_samples(
                runtime,
                cycle=cycle,
                task_id=replay.task.global_task_id,
                mc_index=mc_index,
                start=start,
                batch_size=stop - start,
            )
            current, _ = functional_executed_prefix_flow_loss(
                runtime.policy,
                proxy,
                runtime.lora_contract,
                sliced,
                noise=noise,
                time=flow_time,
            )
            old = replay.old_losses[mc_index, start:stop].to(
                runtime.context.device
            )
            episode_ids = replay.episode_ids[start:stop].to(runtime.context.device)
            loss, metrics = task_relative_aspo_loss(
                current[None],
                old[None],
                episode_ids,
                replay.successes,
                task_advantages=replay.advantages,
                clip_epsilon=float(algorithm["clip_epsilon"]),
                loss_value_clip=float(algorithm["loss_value_clip"]),
                log_ratio_clip=float(algorithm["log_ratio_clip"]),
                episode_chunk_counts=episode_counts,
                mc_samples_normalizer=n_mc,
            )
            (loss / len(runtime.tasks)).backward()
            ratio = torch.exp(
                (
                    old.clamp(max=float(algorithm["loss_value_clip"]))
                    - current.detach().clamp(
                        max=float(algorithm["loss_value_clip"])
                    )
                ).clamp(
                    -float(algorithm["log_ratio_clip"]),
                    float(algorithm["log_ratio_clip"]),
                )
            )
            advantage = replay.advantages[
                replay.episode_ids[start:stop]
            ].to(ratio.device)
            totals["objective"] += metrics.objective
            totals["ratio_sum"] += float(ratio.sum())
            totals["ratio_count"] += float(ratio.numel())
            totals["ratio_min"] = min(totals["ratio_min"], float(ratio.min()))
            totals["ratio_max"] = max(totals["ratio_max"], float(ratio.max()))
            totals["positive_clipped"] += float(
                ((advantage > 0) & (ratio > 1 + float(algorithm["clip_epsilon"]))).sum()
            )
            totals["positive_count"] += float((advantage > 0).sum())
            del sliced, noise, flow_time, current, old, ratio, advantage, loss
    state_gradients = tuple(proxy[name].grad for name in generated)
    generated_lora_grad_norm = generated_lora_gradient_norm(state_gradients)
    torch.autograd.backward(tuple(generated.values()), state_gradients)
    semantic = replay.credit_mode == "all_failure_semantic"
    totals["active_credit_tasks"] = 1.0
    totals["all_failure_semantic_tasks"] = float(semantic)
    totals["all_failure_nonzero_lora_grad_tasks"] = float(
        semantic and generated_lora_grad_norm > 0
    )
    return totals


def _writer_block_gradient_norms(runtime: RLWriterRuntime) -> dict[str, float]:
    result = {}
    for name in (
        "semantic_core",
        "visual_transition",
        "procedure",
        "compiler",
        "factor_heads",
    ):
        module = getattr(runtime.writer, name)
        squared = sum(
            float(parameter.grad.detach().float().square().sum())
            for parameter in module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
        value = math.sqrt(squared)
        if not math.isfinite(value):
            raise RewardProtocolError("non-finite Writer block gradient")
        result[name] = value
    return result


def _learning_epochs(
    runtime: RLWriterRuntime,
    replays: Sequence[TaskCycleReplay],
    *,
    cycle: int,
) -> list[dict[str, Any]]:
    rows = []
    for epoch in range(runtime.learning_epochs):
        runtime.writer.train()
        runtime.writer.semantic_encoder.eval()
        runtime.optimizer.zero_grad(set_to_none=True)
        totals = {
            "objective": 0.0,
            "ratio_sum": 0.0,
            "ratio_count": 0.0,
            "ratio_min": math.inf,
            "ratio_max": -math.inf,
            "positive_clipped": 0.0,
            "positive_count": 0.0,
            "active_credit_tasks": 0.0,
            "all_failure_semantic_tasks": 0.0,
            "all_failure_nonzero_lora_grad_tasks": 0.0,
        }
        for replay in replays:
            observed = _task_epoch_backward(runtime, replay, cycle=cycle)
            for name in totals:
                if name == "ratio_min":
                    totals[name] = min(totals[name], observed[name])
                elif name == "ratio_max":
                    totals[name] = max(totals[name], observed[name])
                else:
                    totals[name] += observed[name]
        _all_reduce_writer_gradients(runtime, cycle=cycle, epoch=epoch)
        if any(
            parameter.grad is not None
            for parameter in runtime.writer.semantic_encoder.parameters()
        ):
            raise RewardProtocolError("frozen progress observer received a gradient")
        block_grad_norms = _writer_block_gradient_norms(runtime)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (
                parameter
                for parameter in runtime.writer.parameters()
                if parameter.requires_grad
            ),
            float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"]),
        )
        if not bool(torch.isfinite(grad_norm)):
            raise RewardProtocolError("non-finite Flow-Credit Writer gradient")
        runtime.optimizer.step()
        runtime.scheduler.step()
        runtime.cursors = InteractionCursors(
            rollout=runtime.cursors.rollout,
            environment_actions=runtime.cursors.environment_actions,
            optimizer_updates=runtime.cursors.optimizer_updates + 1,
        )
        sums = torch.tensor(
            [
                totals["objective"],
                totals["ratio_sum"],
                totals["ratio_count"],
                totals["positive_clipped"],
                totals["positive_count"],
                totals["active_credit_tasks"],
                totals["all_failure_semantic_tasks"],
                totals["all_failure_nonzero_lora_grad_tasks"],
            ],
            dtype=torch.float64,
            device=runtime.context.device,
        )
        limits = torch.tensor(
            [totals["ratio_min"], totals["ratio_max"]],
            dtype=torch.float64,
            device=runtime.context.device,
        )
        if runtime.context.world_size > 1:
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(limits[0], op=dist.ReduceOp.MIN)
            dist.all_reduce(limits[1], op=dist.ReduceOp.MAX)
        ratio_count = float(sums[2])
        rows.append(
            {
                "epoch": float(epoch),
                "full24_objective_mean": float(sums[0]) / len(runtime.tasks),
                "ratio_mean": float(sums[1]) / ratio_count if ratio_count else 1.0,
                "ratio_min": float(limits[0]) if ratio_count else 1.0,
                "ratio_max": float(limits[1]) if ratio_count else 1.0,
                "positive_clip_fraction": (
                    float(sums[3]) / float(sums[4]) if float(sums[4]) else 0.0
                ),
                "active_credit_tasks": int(sums[5]),
                "all_failure_semantic_tasks": int(sums[6]),
                "all_failure_nonzero_generated_lora_gradient_tasks": int(sums[7]),
                "writer_block_gradient_norms_before_clip": block_grad_norms,
                "progress_observer_gradient_tensors": 0,
                "ratio_samples": int(ratio_count),
                "grad_norm_before_clip": float(grad_norm),
            }
        )
    return rows


def _global_cycle_metrics(
    runtime: RLWriterRuntime,
    replays: Sequence[TaskCycleReplay],
    epoch_rows: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    local = torch.tensor(
        [
            sum(replay.rollout_count for replay in replays),
            sum(replay.environment_actions for replay in replays),
            sum(int(replay.successes.sum()) for replay in replays),
            sum(replay.reward_sum for replay in replays),
            sum(replay.credit_mode == "mixed_binary" for replay in replays),
            sum(
                replay.credit_mode == "all_failure_semantic"
                and replay.old_losses is not None
                for replay in replays
            ),
            sum(replay.old_losses is not None for replay in replays),
        ],
        dtype=torch.float64,
        device=runtime.context.device,
    )
    if runtime.context.world_size > 1:
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
    return {
        "global_rollouts": int(local[0]),
        "global_environment_actions": int(local[1]),
        "global_successes": int(local[2]),
        "global_reward_sum": float(local[3]),
        "global_mixed_outcome_tasks": int(local[4]),
        "global_all_failure_semantic_tasks": int(local[5]),
        "global_credit_tasks": int(local[6]),
        "learning_epochs": [dict(row) for row in epoch_rows],
    }


def _checkpoint_if_due(runtime: RLWriterRuntime) -> None:
    if runtime.next_cycle not in runtime.checkpoint_cycles:
        return
    ledger = rank_ledger_summary(runtime, runtime.next_cycle)
    save_rl_writer_checkpoint(
        output_dir=runtime.args.output_dir,
        next_cycle=runtime.next_cycle,
        context=runtime.context,
        writer=runtime.writer,
        optimizer=runtime.optimizer,
        scheduler=runtime.scheduler,
        tasks=runtime.tasks,
        task_schedule_seed=int(runtime.config["data"]["task_schedule_seed"]),
        rollouts_per_task=int(
            runtime.config["algorithm"]["rollouts_per_task_condition"]
        ),
        video_schedule=runtime.video_schedule,
        contract=runtime.contract,
        cursors=runtime.cursors,
        successes=runtime.successes,
        reward_sum=runtime.reward_sum,
        wall_nanoseconds=runtime.wall_nanoseconds,
        ledger_summary=ledger,
        metrics_rows=runtime.metrics_rows,
        learning_epochs=runtime.learning_epochs,
    )


def _publish_summary(runtime: RLWriterRuntime) -> None:
    barrier(runtime.context)
    wall_seconds = reduce_max(runtime.wall_nanoseconds / 1e9, runtime.context)
    if runtime.context.is_main:
        write_json_atomic(
            runtime.args.output_dir / "run_summary.json",
            {
                "schema_version": "ember_pi05_task_grounded_progress_credit_summary_v1",
                "contract_sha256": runtime.contract_sha256,
                "complete": runtime.next_cycle == runtime.total_cycles,
                "next_cycle": runtime.next_cycle,
                "teacher_action_reads_after_coldstart": 0,
                "validation_reward_reads": 0,
                "validation_action_reads": 0,
                "test_reward_reads": 0,
                "test_action_reads": 0,
                "optimizer_update_cursor": runtime.cursors.optimizer_updates,
                "wall_seconds_max_rank": wall_seconds,
            },
        )
    barrier(runtime.context)


def run_cycles(runtime: RLWriterRuntime) -> None:
    started = time.monotonic()
    while runtime.next_cycle < runtime.args.stop_after_cycle:
        cycle = runtime.next_cycle
        tick = time.perf_counter_ns()
        assigned = cycle_assignments(
            runtime.tasks,
            world_size=runtime.context.world_size,
            cycle=cycle,
            seed=int(runtime.config["data"]["task_schedule_seed"]),
        )[runtime.context.rank]
        replays = [_install_and_collect(runtime, task, cycle) for task in assigned]
        local_rollouts = sum(replay.rollout_count for replay in replays)
        local_actions = sum(replay.environment_actions for replay in replays)
        local_successes = sum(int(replay.successes.sum()) for replay in replays)
        local_reward = sum(replay.reward_sum for replay in replays)
        runtime.cursors = InteractionCursors(
            rollout=runtime.cursors.rollout + local_rollouts,
            environment_actions=runtime.cursors.environment_actions + local_actions,
            optimizer_updates=runtime.cursors.optimizer_updates,
        )
        epoch_rows = _learning_epochs(runtime, replays, cycle=cycle)
        torch.cuda.synchronize(runtime.context.device)
        elapsed_ns = time.perf_counter_ns() - tick
        runtime.next_cycle += 1
        runtime.successes += local_successes
        runtime.reward_sum += local_reward
        runtime.wall_nanoseconds += elapsed_ns
        row = {
            "next_cycle": runtime.next_cycle,
            "completed_cycle": cycle,
            **_global_cycle_metrics(runtime, replays, epoch_rows),
            "rank0_tasks": [task.global_task_id for task in assigned]
            if runtime.context.is_main
            else [],
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
        _checkpoint_if_due(runtime)
    _publish_summary(runtime)
