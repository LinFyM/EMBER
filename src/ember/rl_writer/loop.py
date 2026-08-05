"""Full-24 antithetic closed-loop updates on a Writer policy program."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.lora import copy_task_lora_state_, lora_state_sha256
from ember.pi05_source_checkpoint import barrier, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import reduce_max
from ember.reward.ledger import InteractionCursors, write_rollout_once
from ember.reward.protocol import RewardProtocolError, RewardTask, environment_seed
from ember.reward.rollout import RewardTrajectory, collect_randomized_reward_trajectory
from ember.rl_writer.checkpoint import save_rl_writer_checkpoint
from ember.rl_writer.contract import cycle_assignments
from ember.rl_writer.program_credit import (
    PairCredit,
    binary_first_pair_credit,
    program_cotangent,
    program_direction,
    program_direction_seed,
    write_program_credit_once,
)
from ember.rl_writer.progress_observer import observe_correct_teacher_progress
from ember.rl_writer.rendezvous import rank_local_credit_ready
from ember.rl_writer.runtime import RLWriterRuntime, rank_ledger_summary


@dataclass
class TaskProgramReplay:
    task: RewardTask
    demo_index: int
    frames: torch.Tensor
    frame_indices: torch.Tensor
    trajectories: tuple[RewardTrajectory, ...]
    utilities: torch.Tensor
    direction_seeds: tuple[int, int]
    pair_credits: tuple[PairCredit, PairCredit]
    adapter_sha256: tuple[str, str, str, str]
    rollout_count: int
    environment_actions: int
    reward_sum: float

    @property
    def successes(self) -> torch.Tensor:
        return torch.tensor(
            [trajectory.success for trajectory in self.trajectories],
            dtype=torch.bool,
        )


def _all_reduce_writer_gradients(runtime: RLWriterRuntime, *, cycle: int) -> None:
    gradients = []
    for parameter in runtime.writer.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        gradients.append(parameter.grad)
    if not gradients:
        raise RewardProtocolError("program-credit Writer has no gradients")
    flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
    rank_local_credit_ready(runtime, cycle=cycle, epoch=0)
    if runtime.context.world_size > 1:
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    offset = 0
    for gradient in gradients:
        count = gradient.numel()
        gradient.copy_(flat[offset : offset + count].view_as(gradient))
        offset += count


def _writer_inputs(
    runtime: RLWriterRuntime,
    *,
    task: RewardTask,
    frames: torch.Tensor,
    frame_indices: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    copy_task_lora_state_(runtime.policy, runtime.identity_state, runtime.lora_contract)
    tokens, masks, task_spans = runtime.tokenizer([task.language])
    device = runtime.context.device
    values = frames.to(device, non_blocking=True)
    indices = frame_indices.to(device, non_blocking=True)
    offsets = torch.tensor([0, values.shape[0]], dtype=torch.long, device=device)
    return values, indices, offsets, tokens, masks, task_spans


def _writer_program(
    runtime: RLWriterRuntime,
    *,
    task: RewardTask,
    frames: torch.Tensor,
    frame_indices: torch.Tensor,
    train: bool,
) -> torch.Tensor:
    inputs = _writer_inputs(
        runtime,
        task=task,
        frames=frames,
        frame_indices=frame_indices,
    )
    runtime.writer.train(train)
    runtime.writer.semantic_encoder.eval()
    runtime.writer.factor_heads.eval()
    context = torch.enable_grad() if train else torch.inference_mode()
    with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        program = runtime.writer.encode_program(*inputs, policy=runtime.policy)
    if program.shape != (1, 320, 256):
        raise RewardProtocolError("Writer program shape changed")
    return program


def _decode_program(
    runtime: RLWriterRuntime,
    program: torch.Tensor,
) -> dict[str, torch.Tensor]:
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        state = runtime.writer.decode_program(program)
    return state


def _pair_randomness_cursor(cycle: int, pair_index: int) -> int:
    if cycle < 0 or pair_index not in {0, 1}:
        raise RewardProtocolError("invalid program-credit pair cursor")
    return cycle * 2 + pair_index


def _collect_task(runtime: RLWriterRuntime, task: RewardTask, cycle: int) -> TaskProgramReplay:
    demo = runtime.video_schedule.demo_for_task_visit(task.global_task_id, cycle)
    video = runtime.video_store.load(task.global_task_id, demo)
    frames = torch.from_numpy(video.frames)
    frame_indices = torch.from_numpy(video.frame_indices)
    base_program = _writer_program(
        runtime,
        task=task,
        frames=frames,
        frame_indices=frame_indices,
        train=False,
    )
    sigma = float(runtime.config["algorithm"]["program_sigma"])
    seed_root = int(runtime.config["rng"]["program_direction_seed_root"])
    direction_seeds = tuple(
        program_direction_seed(
            seed_root,
            cycle=cycle,
            global_task_id=task.global_task_id,
            pair_index=pair_index,
        )
        for pair_index in range(2)
    )
    trajectories: list[RewardTrajectory] = []
    adapter_shas: list[str] = []
    for offset in range(4):
        pair_index = offset // 2
        sign = 1 if offset % 2 == 0 else -1
        direction = program_direction(
            direction_seeds[pair_index],
            base_program.shape,
            device=base_program.device,
            dtype=base_program.dtype,
        )
        state = _decode_program(runtime, base_program + sign * sigma * direction)
        copy_task_lora_state_(runtime.policy, state, runtime.lora_contract)
        adapter_sha = lora_state_sha256(state)
        artifact_cursor = cycle * 4 + offset
        randomness_cursor = _pair_randomness_cursor(cycle, pair_index)
        env_seed = environment_seed(
            int(runtime.config["rng"]["environment_seed_root"]),
            task.suite,
            task.task_id,
            int(runtime.config["optimization"]["seed"]),
            randomness_cursor,
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
            rollout_cursor=artifact_cursor,
            randomness_cursor=randomness_cursor,
            env_seed=env_seed,
            policy_seed_root=int(runtime.config["rng"]["policy_noise_seed_root"]),
            device=runtime.context.device,
            max_horizon=task.horizon,
            dummy_settling_steps=10,
            dummy_action=runtime.config["environment"]["dummy_action"],
            action_execution_horizon=5,
            num_inference_steps=10,
            retain_failure_replay=False,
        )
        row = {
            **trajectory.ledger_row(),
            "schema_version": str(runtime.config["algorithm"]["rollout_schema"]),
            "failure_replay_retained": False,
            "producer_rank": runtime.context.rank,
            "outer_cycle": cycle,
            "teacher_demo_index": demo,
            "program_pair_index": pair_index,
            "program_sign": sign,
            "program_sigma": sigma,
            "program_direction_seed": direction_seeds[pair_index],
            "randomness_cursor": randomness_cursor,
            "writer_lora_sha256": adapter_sha,
        }
        write_rollout_once(
            runtime.args.output_dir,
            f"task_{task.global_task_id:03d}",
            row,
        )
        trajectories.append(trajectory)
        adapter_shas.append(adapter_sha)

    utilities, _ = observe_correct_teacher_progress(
        writer=runtime.writer,
        policy=runtime.policy,
        identity_state=runtime.identity_state,
        lora_contract=runtime.lora_contract,
        tokenizer=runtime.tokenizer,
        task=task,
        teacher_frames=frames,
        trajectories=trajectories,
        device=runtime.context.device,
        normalization_epsilon=float(
            runtime.config["progress_credit"]["normalization_epsilon"]
        ),
        projection_epsilon=float(
            runtime.config["progress_credit"]["projection_epsilon"]
        ),
    )
    if utilities.shape != (4,):
        raise RewardProtocolError("program progress utility shape changed")
    credits = tuple(
        binary_first_pair_credit(
            success_plus=trajectories[2 * pair_index].success,
            success_minus=trajectories[2 * pair_index + 1].success,
            progress_plus=float(utilities[2 * pair_index]),
            progress_minus=float(utilities[2 * pair_index + 1]),
        )
        for pair_index in range(2)
    )
    ledger_directions = tuple(
        program_direction(seed, (320, 256)) for seed in direction_seeds
    )
    cotangent = program_cotangent(
        ledger_directions,
        [credit.value for credit in credits],
    )
    pair_rows = []
    for pair_index, credit in enumerate(credits):
        plus = trajectories[2 * pair_index]
        minus = trajectories[2 * pair_index + 1]
        prefix = min(len(plus.policy_noise_seeds), len(minus.policy_noise_seeds))
        if (
            plus.env_seed != minus.env_seed
            or plus.initial_observation_sha256 != minus.initial_observation_sha256
            or not prefix
            or plus.policy_noise_seeds[:prefix] != minus.policy_noise_seeds[:prefix]
        ):
            raise RewardProtocolError("antithetic pair lost common random numbers")
        pair_rows.append(
            {
                "pair_index": pair_index,
                "direction_seed": direction_seeds[pair_index],
                "randomness_cursor": _pair_randomness_cursor(cycle, pair_index),
                "plus_rollout_cursor": plus.rollout_cursor,
                "minus_rollout_cursor": minus.rollout_cursor,
                "plus_lora_sha256": adapter_shas[2 * pair_index],
                "minus_lora_sha256": adapter_shas[2 * pair_index + 1],
                "plus_success": plus.success,
                "minus_success": minus.success,
                "plus_progress": float(utilities[2 * pair_index]),
                "minus_progress": float(utilities[2 * pair_index + 1]),
                "credit": credit.value,
                "credit_mode": credit.mode,
            }
        )
    write_program_credit_once(
        output_dir=runtime.args.output_dir,
        producer_rank=runtime.context.rank,
        cycle=cycle,
        global_task_id=task.global_task_id,
        teacher_demo_index=demo,
        sigma=sigma,
        pairs=pair_rows,
        cotangent_norm=float(cotangent.norm()),
    )
    return TaskProgramReplay(
        task=task,
        demo_index=demo,
        frames=frames,
        frame_indices=frame_indices,
        trajectories=tuple(trajectories),
        utilities=utilities,
        direction_seeds=direction_seeds,
        pair_credits=credits,
        adapter_sha256=tuple(adapter_shas),
        rollout_count=4,
        environment_actions=sum(value.steps for value in trajectories),
        reward_sum=sum(value.reward_sum for value in trajectories),
    )


def _task_program_backward(
    runtime: RLWriterRuntime,
    replay: TaskProgramReplay,
) -> float:
    program = _writer_program(
        runtime,
        task=replay.task,
        frames=replay.frames,
        frame_indices=replay.frame_indices,
        train=True,
    )
    directions = tuple(
        program_direction(
            seed,
            program.shape,
            device=program.device,
            dtype=program.dtype,
        )
        for seed in replay.direction_seeds
    )
    cotangent = program_cotangent(
        directions,
        [credit.value for credit in replay.pair_credits],
    )
    torch.autograd.backward(program, cotangent / len(runtime.tasks))
    return float(cotangent.detach().float().norm())


def _block_gradient_norms(runtime: RLWriterRuntime) -> dict[str, float]:
    result = {}
    for name in ("semantic_core", "visual_transition", "procedure", "compiler"):
        module = getattr(runtime.writer, name)
        squared = sum(
            float(parameter.grad.detach().float().square().sum())
            for parameter in module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
        result[name] = math.sqrt(squared)
    if not all(math.isfinite(value) for value in result.values()):
        raise RewardProtocolError("non-finite program-credit block gradient")
    return result


def _frozen_gradient_tensors(runtime: RLWriterRuntime) -> dict[str, int]:
    return {
        "semantic_encoder": sum(
            parameter.grad is not None
            for parameter in runtime.writer.semantic_encoder.parameters()
        ),
        "factor_heads": sum(
            parameter.grad is not None
            for parameter in runtime.writer.factor_heads.parameters()
        ),
        "source_policy": sum(
            parameter.grad is not None for parameter in runtime.policy.parameters()
        ),
    }


def _program_update(
    runtime: RLWriterRuntime,
    replays: Sequence[TaskProgramReplay],
    *,
    cycle: int,
) -> dict[str, Any]:
    runtime.writer.train()
    runtime.writer.semantic_encoder.eval()
    runtime.writer.factor_heads.eval()
    runtime.optimizer.zero_grad(set_to_none=True)
    cotangent_norms = [_task_program_backward(runtime, replay) for replay in replays]
    _all_reduce_writer_gradients(runtime, cycle=cycle)
    frozen = _frozen_gradient_tensors(runtime)
    if any(frozen.values()):
        raise RewardProtocolError("frozen program-credit owner received a gradient")
    block_norms = _block_gradient_norms(runtime)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        (
            parameter
            for parameter in runtime.writer.parameters()
            if parameter.requires_grad
        ),
        float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"]),
    )
    if not bool(torch.isfinite(grad_norm)):
        raise RewardProtocolError("non-finite program-credit Writer gradient")
    runtime.optimizer.step()
    runtime.scheduler.step()
    runtime.cursors = InteractionCursors(
        rollout=runtime.cursors.rollout,
        environment_actions=runtime.cursors.environment_actions,
        optimizer_updates=runtime.cursors.optimizer_updates + 1,
    )
    local = torch.tensor(
        [sum(cotangent_norms), len(cotangent_norms)],
        dtype=torch.float64,
        device=runtime.context.device,
    )
    if runtime.context.world_size > 1:
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
    return {
        "updates": 1,
        "full24_cotangent_norm_mean": float(local[0] / local[1]),
        "writer_block_gradient_norms_before_clip": block_norms,
        "frozen_gradient_tensors": frozen,
        "grad_norm_before_clip": float(grad_norm),
    }


def _global_cycle_metrics(
    runtime: RLWriterRuntime,
    replays: Sequence[TaskProgramReplay],
    update: Mapping[str, Any],
) -> dict[str, Any]:
    modes = [credit.mode for replay in replays for credit in replay.pair_credits]
    local = torch.tensor(
        [
            sum(replay.rollout_count for replay in replays),
            sum(replay.environment_actions for replay in replays),
            sum(int(replay.successes.sum()) for replay in replays),
            sum(replay.reward_sum for replay in replays),
            sum(mode == "binary_discordant" for mode in modes),
            sum(mode == "paired_failure_semantic" for mode in modes),
            sum(mode == "paired_success_zero" for mode in modes),
            sum(
                credit.value != 0
                for replay in replays
                for credit in replay.pair_credits
            ),
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
        "global_binary_discordant_pairs": int(local[4]),
        "global_paired_failure_semantic_pairs": int(local[5]),
        "global_paired_success_zero_pairs": int(local[6]),
        "global_nonzero_credit_pairs": int(local[7]),
        "program_update": dict(update),
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
        rollouts_per_task=4,
        video_schedule=runtime.video_schedule,
        contract=runtime.contract,
        cursors=runtime.cursors,
        successes=runtime.successes,
        reward_sum=runtime.reward_sum,
        wall_nanoseconds=runtime.wall_nanoseconds,
        ledger_summary=ledger,
        metrics_rows=runtime.metrics_rows,
        learning_epochs=1,
    )


def _publish_summary(runtime: RLWriterRuntime) -> None:
    barrier(runtime.context)
    wall_seconds = reduce_max(runtime.wall_nanoseconds / 1e9, runtime.context)
    if runtime.context.is_main:
        write_json_atomic(
            runtime.args.output_dir / "run_summary.json",
            {
                "schema_version": "ember_pi05_antithetic_program_credit_summary_v1",
                "contract_sha256": runtime.contract_sha256,
                "complete": runtime.next_cycle == runtime.total_cycles,
                "next_cycle": runtime.next_cycle,
                "teacher_action_reads_after_coldstart": 0,
                "validation_reward_reads": 0,
                "validation_action_reads": 0,
                "test_reward_reads": 0,
                "test_action_reads": 0,
                "source_policy_backward_calls": 0,
                "functional_action_loss_calls": 0,
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
        replays = [_collect_task(runtime, task, cycle) for task in assigned]
        local_rollouts = sum(replay.rollout_count for replay in replays)
        local_actions = sum(replay.environment_actions for replay in replays)
        local_successes = sum(int(replay.successes.sum()) for replay in replays)
        local_reward = sum(replay.reward_sum for replay in replays)
        runtime.cursors = InteractionCursors(
            rollout=runtime.cursors.rollout + local_rollouts,
            environment_actions=runtime.cursors.environment_actions + local_actions,
            optimizer_updates=runtime.cursors.optimizer_updates,
        )
        update = _program_update(runtime, replays, cycle=cycle)
        torch.cuda.synchronize(runtime.context.device)
        elapsed_ns = time.perf_counter_ns() - tick
        runtime.next_cycle += 1
        runtime.successes += local_successes
        runtime.reward_sum += local_reward
        runtime.wall_nanoseconds += elapsed_ns
        row = {
            "next_cycle": runtime.next_cycle,
            "completed_cycle": cycle,
            **_global_cycle_metrics(runtime, replays, update),
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
