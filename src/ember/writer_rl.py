"""Eight-rank source-only reward training that updates only the shared Writer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
from lerobot.envs import close_envs, make_env
from lerobot.envs.configs import LiberoEnv

from ember.libero_reward_rollout import (
    RewardTrajectory,
    collect_randomized_reward_trajectory,
    successful_trajectory_batch,
)
from ember.lora import (
    copy_task_lora_state_,
    lora_state_sha256,
    validate_lora_state,
)
from ember.source_base_checkpoint import (
    DistributedContext,
    barrier,
    read_json,
    write_json_atomic,
)
from ember.writer.functional import writer_success_weighted_flow_loss
from ember.writer.model import WriterModelError
from ember.writer_rl_checkpoint import (
    save_writer_rl_checkpoint,
    write_update_ledger_once,
)
from ember.writer_rl_protocol import (
    environment_seed,
    policy_seed,
    schedule_summary,
    task_for_update,
    update_seed,
    updates_per_cycle,
)
from ember.writer_rl_runtime import (
    REPO_ROOT,
    WriterRLRuntime,
    initialize_distributed,
    prepare_runtime,
    writer_inputs,
)


@torch.no_grad()
def _install_writer_lora(
    runtime: WriterRLRuntime, task_id: int, context: DistributedContext
) -> str:
    _, language, video, offsets = writer_inputs(runtime, task_id, context.device)
    runtime.writer.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = runtime.writer(language, video, offsets)
    validate_lora_state(state, runtime.lora_contract)
    copy_task_lora_state_(runtime.policy, state, runtime.lora_contract)
    return lora_state_sha256(state)


def _collect_task_rollouts(
    runtime: WriterRLRuntime,
    context: DistributedContext,
    task_id: int,
    cycle: int,
) -> tuple[list[RewardTrajectory], str]:
    adapter_sha256 = _install_writer_lora(runtime, task_id, context)
    environment = runtime.config["environment"]
    env_config = LiberoEnv(
        task=environment["suite"],
        task_ids=[task_id],
        episode_length=int(environment["max_horizon"]),
        obs_type="pixels_agent_pos",
        camera_name=environment["camera_name"],
        init_states=False,
        observation_height=int(environment["observation_height"]),
        observation_width=int(environment["observation_width"]),
        control_mode=environment["control_mode"],
    )
    if env_config.init_states:
        raise WriterModelError("Writer-only RL attempted to enable fixed init states")
    envs = make_env(env_config, n_envs=1, use_async_envs=False)
    env = envs[environment["suite"]][task_id]
    trajectories: list[RewardTrajectory] = []
    try:
        for rollout in range(
            int(runtime.config["algorithm"]["rollouts_per_task_cycle"])
        ):
            trajectories.append(
                collect_randomized_reward_trajectory(
                    env=env,
                    policy=runtime.policy,
                    env_preprocessor=runtime.env_preprocessor,
                    env_postprocessor=runtime.env_postprocessor,
                    preprocessor=runtime.preprocessor,
                    postprocessor=runtime.postprocessor,
                    task_id=task_id,
                    language=runtime.languages[task_id],
                    env_seed=environment_seed(
                        int(runtime.config["rng"]["environment_seed_base"]),
                        cycle,
                        task_id,
                        rollout,
                    ),
                    policy_seed=policy_seed(
                        int(runtime.config["rng"]["policy_seed_base"]),
                        cycle,
                        task_id,
                        rollout,
                    ),
                    device=context.device,
                    max_horizon=int(environment["max_horizon"]),
                    action_execution_horizon=int(
                        runtime.config["policy"]["action_execution_horizon"]
                    ),
                    use_bfloat16=runtime.config["policy"]["precision"]
                    == "bfloat16",
                )
            )
    finally:
        close_envs(envs)
    return trajectories, adapter_sha256


def _all_reduce_int(value: int, context: DistributedContext) -> int:
    tensor = torch.tensor(value, dtype=torch.int64, device=context.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def _all_reduce_float(
    value: float, context: DistributedContext, operation: dist.ReduceOp
) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=context.device)
    dist.all_reduce(tensor, op=operation)
    return float(tensor.item())


def _reward_update(
    *,
    runtime: WriterRLRuntime,
    context: DistributedContext,
    task_id: int | None,
    trajectories: list[RewardTrajectory],
    update: int,
    global_successes: int,
) -> tuple[float, float]:
    runtime.optimizer.zero_grad(set_to_none=True)
    local_loss_sum = 0.0
    if task_id is not None:
        _, language, video, offsets = writer_inputs(
            runtime, task_id, context.device
        )
    else:
        padding_task = runtime.task_ids[context.rank]
        _, language, video, offsets = writer_inputs(
            runtime, padding_task, context.device
        )
    seed = update_seed(
        int(runtime.config["rng"]["update_seed_base"]), update, context.rank
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    runtime.writer.train()
    successful = [trajectory for trajectory in trajectories if trajectory.success]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if successful:
            batch, episode_ids = successful_trajectory_batch(
                successful, context.device
            )
            loss_sum, _ = writer_success_weighted_flow_loss(
                runtime.wrapped_writer,  # type: ignore[arg-type]
                runtime.policy,
                runtime.lora_contract,
                language_features=language,
                video_features=video,
                episode_offsets=offsets,
                batch=batch,
                rollout_episode_ids=episode_ids,
            )
            local_loss_sum = float(loss_sum.detach())
            loss = loss_sum * context.world_size / global_successes
        else:
            generated = runtime.wrapped_writer(language, video, offsets)
            loss = sum(value.sum() for value in generated.values()) * 0.0
    if not bool(torch.isfinite(loss).detach()):
        raise WriterModelError(f"non-finite Writer-only RL loss at update {update}")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        runtime.writer.parameters(),
        float(runtime.config["optimization"]["grad_clip_norm"]),
    )
    runtime.optimizer.step()
    runtime.scheduler.step()
    runtime.optimizer_updates += 1
    return local_loss_sum, float(grad_norm)


def run_updates(
    args: argparse.Namespace,
    context: DistributedContext,
    runtime: WriterRLRuntime,
) -> None:
    update = runtime.next_update
    cycle_updates = updates_per_cycle(runtime.task_ids, context.world_size)
    while update < args.stop_after_update:
        tick = time.perf_counter_ns()
        scheduled = task_for_update(
            runtime.task_ids, context.world_size, context.rank, update
        )
        trajectories: list[RewardTrajectory] = []
        adapter_sha256 = None
        task_id = None if scheduled is None else scheduled[0]
        cycle, slot = divmod(update, cycle_updates)
        if scheduled is not None:
            task_id, scheduled_cycle, scheduled_slot = scheduled
            if (scheduled_cycle, scheduled_slot) != (cycle, slot):
                raise WriterModelError("Writer-only RL schedule cursor changed")
            trajectories, adapter_sha256 = _collect_task_rollouts(
                runtime, context, task_id, cycle
            )
        local_successes = sum(trajectory.success for trajectory in trajectories)
        local_rollouts = len(trajectories)
        local_env_steps = sum(trajectory.steps for trajectory in trajectories)
        runtime.local_counters["rollouts"] += local_rollouts
        runtime.local_counters["successes"] += local_successes
        runtime.local_counters["env_steps"] += local_env_steps
        ledger = {
            "schema_version": "ember_writer_only_rl_rollout_update_v1",
            "rank": context.rank,
            "update": update,
            "cycle": cycle,
            "cycle_slot": slot,
            "task_id": task_id,
            "official_random_reset": True,
            "fixed_init_state_id": None,
            "adapter_sha256": adapter_sha256,
            "trajectories": [trajectory.ledger_row() for trajectory in trajectories],
        }
        write_update_ledger_once(args.output_dir, context.rank, update, ledger)
        global_successes = _all_reduce_int(local_successes, context)
        global_rollouts = _all_reduce_int(local_rollouts, context)
        global_env_steps = _all_reduce_int(local_env_steps, context)
        local_loss_sum = 0.0
        grad_norm = 0.0
        if global_successes > 0:
            local_loss_sum, grad_norm = _reward_update(
                runtime=runtime,
                context=context,
                task_id=task_id,
                trajectories=trajectories,
                update=update,
                global_successes=global_successes,
            )
        torch.cuda.synchronize(context.device)
        elapsed_ns = time.perf_counter_ns() - tick
        runtime.local_counters["wall_nanoseconds"] += elapsed_ns
        update += 1
        runtime.next_update = update
        reward_loss = _all_reduce_float(
            local_loss_sum, context, dist.ReduceOp.SUM
        )
        if global_successes:
            reward_loss /= global_successes
        step_seconds = _all_reduce_float(
            elapsed_ns / 1e9, context, dist.ReduceOp.MAX
        )
        peak_reserved = _all_reduce_float(
            torch.cuda.max_memory_reserved(context.device) / 2**30,
            context,
            dist.ReduceOp.MAX,
        )
        if context.is_main:
            print(
                json.dumps(
                    {
                        "event": "reward_update",
                        "next_update": update,
                        "cycle": cycle,
                        "cycle_slot": slot,
                        "global_rollouts": global_rollouts,
                        "global_env_steps": global_env_steps,
                        "global_successes": global_successes,
                        "reward_weighted_flow_loss": reward_loss,
                        "grad_norm_rank0": grad_norm,
                        "optimizer_updates": runtime.optimizer_updates,
                        "step_seconds_max": step_seconds,
                        "peak_reserved_gib_max": peak_reserved,
                    }
                ),
                flush=True,
            )
        if update in runtime.checkpoint_updates:
            checkpoint = save_writer_rl_checkpoint(
                output_dir=args.output_dir,
                next_update=update,
                optimizer_updates=runtime.optimizer_updates,
                context=context,
                writer=runtime.writer,
                optimizer=runtime.optimizer,
                scheduler=runtime.scheduler,
                task_ids=runtime.task_ids,
                rollouts_per_task=int(
                    runtime.config["algorithm"]["rollouts_per_task_cycle"]
                ),
                contract=runtime.contract,
                local_counters=runtime.local_counters,
                formal=args.mode == "formal",
            )
            if context.is_main:
                print(
                    json.dumps(
                        {
                            "event": "checkpoint",
                            "path": str(checkpoint),
                            **schedule_summary(
                                runtime.task_ids,
                                context.world_size,
                                update,
                                int(
                                    runtime.config["algorithm"][
                                        "rollouts_per_task_cycle"
                                    ]
                                ),
                            ),
                        }
                    ),
                    flush=True,
                )
    _write_summary(args, context, runtime)


def _write_summary(
    args: argparse.Namespace,
    context: DistributedContext,
    runtime: WriterRLRuntime,
) -> None:
    rank_summary = {
        "rank": context.rank,
        "next_update": runtime.next_update,
        "optimizer_updates": runtime.optimizer_updates,
        "local_counters": runtime.local_counters,
    }
    write_json_atomic(
        args.output_dir / f"rank_{context.rank:02d}_summary.json", rank_summary
    )
    barrier(context)
    if context.is_main:
        summaries = [
            read_json(args.output_dir / f"rank_{rank:02d}_summary.json")
            for rank in range(context.world_size)
        ]
        counters = [summary["local_counters"] for summary in summaries]
        result = {
            "schema_version": "ember_writer_only_rl_summary_v1",
            "complete": runtime.next_update == args.total_updates,
            "next_update": runtime.next_update,
            "optimizer_updates": runtime.optimizer_updates,
            "rollouts": sum(int(value["rollouts"]) for value in counters),
            "successes": sum(int(value["successes"]) for value in counters),
            "env_steps": sum(int(value["env_steps"]) for value in counters),
            "wall_seconds_max_rank": max(
                int(value["wall_nanoseconds"]) for value in counters
            )
            / 1e9,
            "schedule": schedule_summary(
                runtime.task_ids,
                context.world_size,
                runtime.next_update,
                int(runtime.config["algorithm"]["rollouts_per_task_cycle"]),
            ),
        }
        write_json_atomic(args.output_dir / "run_summary.json", result)
        print(json.dumps({"event": "complete", **result}), flush=True)
    barrier(context)


def train(args: argparse.Namespace) -> None:
    context = initialize_distributed()
    try:
        runtime = prepare_runtime(args, context)
        if context.is_main:
            print(
                json.dumps(
                    {
                        "event": "start",
                        "mode": args.mode,
                        "contract_sha256": runtime.contract_sha256,
                        "next_update": runtime.next_update,
                        "stop_after_update": args.stop_after_update,
                        "task_count": len(runtime.task_ids),
                        "trainable": runtime.contract["trainable"],
                    }
                ),
                flush=True,
            )
        run_updates(args, context, runtime)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/writer_only_rl_v1.json"
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--writer-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--total-updates", type=int, required=True)
    parser.add_argument("--stop-after-update", type=int)
    parser.add_argument("--checkpoint-updates", type=str, required=True)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.stop_after_update is None:
        args.stop_after_update = args.total_updates
    args.config = args.config.resolve()
    args.policy_path = args.policy_path.resolve()
    args.writer_checkpoint = args.writer_checkpoint.resolve()
    args.feature_cache = args.feature_cache.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.resume is not None:
        args.resume = args.resume.resolve()
    return args
