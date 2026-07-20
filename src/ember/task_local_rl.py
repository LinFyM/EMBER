"""Matched ordinary task-local LoRA reward adaptation on validation tasks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from lerobot.envs import close_envs, make_env
from lerobot.envs.configs import LiberoEnv

from ember.libero_reward_rollout import (
    RewardTrajectory,
    collect_randomized_reward_trajectory,
)
from ember.lora import (
    initialize_identity_lora_,
    lora_state_sha256,
    task_lora_state_dict,
)
from ember.source_base_checkpoint import (
    DistributedContext,
    canonical_hash,
    read_json,
    write_json_atomic,
)
from ember.task_local_rl_checkpoint import (
    load_task_local_checkpoint,
    restore_rng,
    save_task_local_checkpoint,
    verify_task_local_checkpoint,
    write_unit_ledger_once,
)
from ember.task_local_rl_protocol import (
    TaskArm,
    rollout_seed,
)
from ember.task_local_rl_reporting import (
    matched_schedule_summary,
    select_unit_checkpoint,
)
from ember.task_local_rl_runtime import (
    REPO_ROOT,
    TaskLocalRLRuntime,
    barrier,
    initialize_distributed,
    prepare_runtime,
)
from ember.task_local_rl_update import reward_update
from ember.writer.model import WriterModelError


def _latest_checkpoint(unit_dir: Path) -> Path | None:
    latest_path = unit_dir / "latest_checkpoint.json"
    if not latest_path.is_file():
        return None
    latest = read_json(latest_path)
    value = latest.get("path")
    if not isinstance(value, str) or not value:
        raise WriterModelError(f"invalid task-local latest checkpoint: {unit_dir}")
    checkpoint = Path(value).resolve(strict=True)
    expected_parent = (unit_dir / "checkpoints").resolve(strict=True)
    if checkpoint.parent != expected_parent:
        raise WriterModelError("task-local checkpoint escaped its unit directory")
    verify_task_local_checkpoint(checkpoint)
    return checkpoint


@torch.no_grad()
def _initialize_arm(
    runtime: TaskLocalRLRuntime, unit: TaskArm
) -> str:
    if unit.arm == "identity":
        initialize_identity_lora_(runtime.policy, runtime.lora_contract)
        return lora_state_sha256(task_lora_state_dict(runtime.policy))
    if unit.arm == "writer":
        return runtime.adapter.apply(unit.task_id)
    raise WriterModelError(f"unsupported task-local RL arm: {unit.arm}")


def _make_task_env(runtime: TaskLocalRLRuntime, task_id: int) -> tuple[Any, Any]:
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
        raise WriterModelError("task-local RL attempted fixed init-state sampling")
    envs = make_env(env_config, n_envs=1, use_async_envs=False)
    return envs, envs[environment["suite"]][task_id]


def _collect_rollouts(
    *,
    runtime: TaskLocalRLRuntime,
    context: DistributedContext,
    env: Any,
    unit: TaskArm,
    update: int,
) -> list[RewardTrajectory]:
    config = runtime.config
    environment = config["environment"]
    trajectories = []
    for rollout in range(int(config["algorithm"]["rollouts_per_update"])):
        trajectories.append(
            collect_randomized_reward_trajectory(
                env=env,
                policy=runtime.policy,
                env_preprocessor=runtime.env_preprocessor,
                env_postprocessor=runtime.env_postprocessor,
                preprocessor=runtime.preprocessor,
                postprocessor=runtime.postprocessor,
                task_id=unit.task_id,
                language=runtime.languages[unit.task_id],
                env_seed=rollout_seed(
                    int(config["rng"]["environment_seed_base"]),
                    unit.task_id,
                    update,
                    rollout,
                ),
                policy_seed=rollout_seed(
                    int(config["rng"]["policy_seed_base"]),
                    unit.task_id,
                    update,
                    rollout,
                ),
                device=context.device,
                max_horizon=int(environment["max_horizon"]),
                action_execution_horizon=int(
                    config["policy"]["action_execution_horizon"]
                ),
                use_bfloat16=config["policy"]["precision"] == "bfloat16",
            )
        )
    return trajectories


def _unit_contract(
    runtime: TaskLocalRLRuntime, unit: TaskArm, initialization_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": "ember_task_local_lora_rl_unit_v1",
        "run_contract_sha256": runtime.contract_sha256,
        "task_id": unit.task_id,
        "language": runtime.languages[unit.task_id],
        "arm": unit.arm,
        "initialization_sha256": initialization_sha256,
        "matched_seed_schedule_excludes_arm": True,
    }


def _train_unit(
    *,
    args: argparse.Namespace,
    runtime: TaskLocalRLRuntime,
    context: DistributedContext,
    unit: TaskArm,
) -> dict[str, Any]:
    started = time.monotonic()
    unit_dir = args.output_dir / "units" / unit.key
    initialization_sha256 = _initialize_arm(runtime, unit)
    contract = _unit_contract(runtime, unit, initialization_sha256)
    unit_contract_sha256 = canonical_hash(contract)
    contract_path = unit_dir / "unit_contract.json"
    if args.resume:
        if not contract_path.is_file() or canonical_hash(
            read_json(contract_path)
        ) != unit_contract_sha256:
            raise WriterModelError(f"task-local resume unit changed: {unit.key}")
    else:
        if unit_dir.exists() and any(unit_dir.iterdir()):
            raise WriterModelError(f"task-local unit is not empty: {unit_dir}")
        write_json_atomic(contract_path, contract)

    lora_parameters = tuple(task_lora_state_dict(runtime.policy).values())
    optimizer = torch.optim.AdamW(
        lora_parameters,
        lr=float(runtime.config["optimization"]["learning_rate"]),
        betas=tuple(runtime.config["optimization"]["betas"]),
        eps=float(runtime.config["optimization"]["eps"]),
        weight_decay=float(runtime.config["optimization"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    next_update = 0
    counters = {
        "rollouts": 0,
        "successes": 0,
        "env_steps": 0,
        "optimizer_updates": 0,
        "wall_nanoseconds": 0,
    }
    latest = _latest_checkpoint(unit_dir) if args.resume else None
    if latest is not None:
        next_update, counters, rng = load_task_local_checkpoint(
            checkpoint=latest,
            policy=runtime.policy,
            contract=runtime.lora_contract,
            optimizer=optimizer,
            scheduler=scheduler,
            unit_contract_sha256=unit_contract_sha256,
            rollouts_per_update=int(
                runtime.config["algorithm"]["rollouts_per_update"]
            ),
            device=context.device,
        )
        restore_rng(rng, context.device)
    if not 0 <= next_update <= args.stop_after_update:
        raise WriterModelError(f"task-local resume cursor is invalid: {unit.key}")

    segment_successes = 0
    segment_rollouts = 0
    if next_update < args.stop_after_update:
        envs, env = _make_task_env(runtime, unit.task_id)
        try:
            while next_update < args.stop_after_update:
                tick = time.perf_counter_ns()
                runtime.policy.eval()
                adapter_sha256 = lora_state_sha256(
                    task_lora_state_dict(runtime.policy)
                )
                trajectories = _collect_rollouts(
                    runtime=runtime,
                    context=context,
                    env=env,
                    unit=unit,
                    update=next_update,
                )
                successes = sum(trajectory.success for trajectory in trajectories)
                rollouts = len(trajectories)
                env_steps = sum(trajectory.steps for trajectory in trajectories)
                ledger = {
                    "schema_version": "ember_task_local_lora_rl_rollout_update_v1",
                    "rank": context.rank,
                    "unit": unit.key,
                    "task_id": unit.task_id,
                    "arm": unit.arm,
                    "update": next_update,
                    "interaction_cursor_before": counters["rollouts"],
                    "interaction_cursor_after": counters["rollouts"] + rollouts,
                    "official_random_reset": True,
                    "fixed_init_state_id": None,
                    "adapter_sha256": adapter_sha256,
                    "trajectories": [
                        trajectory.ledger_row() for trajectory in trajectories
                    ],
                }
                write_unit_ledger_once(unit_dir, next_update, ledger)
                counters["rollouts"] += rollouts
                counters["successes"] += successes
                counters["env_steps"] += env_steps
                segment_successes += successes
                segment_rollouts += rollouts
                loss = None
                grad_norm = None
                if successes:
                    loss, grad_norm = reward_update(
                        runtime=runtime,
                        context=context,
                        unit=unit,
                        update=next_update,
                        trajectories=trajectories,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )
                    counters["optimizer_updates"] += 1
                torch.cuda.synchronize(context.device)
                elapsed_ns = time.perf_counter_ns() - tick
                counters["wall_nanoseconds"] += elapsed_ns
                next_update += 1
                print(
                    json.dumps(
                        {
                            "event": "task_local_reward_update",
                            "rank": context.rank,
                            "unit": unit.key,
                            "next_update": next_update,
                            "interaction_cursor": counters["rollouts"],
                            "rollouts": rollouts,
                            "successes": successes,
                            "env_steps": env_steps,
                            "reward_weighted_flow_loss": loss,
                            "grad_norm": grad_norm,
                            "optimizer_updates": counters["optimizer_updates"],
                            "step_seconds": elapsed_ns / 1e9,
                            "peak_reserved_gib": torch.cuda.max_memory_reserved(
                                context.device
                            )
                            / 2**30,
                        }
                    ),
                    flush=True,
                )
                if next_update in runtime.checkpoint_updates:
                    checkpoint = save_task_local_checkpoint(
                        unit_dir=unit_dir,
                        next_update=next_update,
                        policy=runtime.policy,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        unit_contract_sha256=unit_contract_sha256,
                        counters=counters,
                        rollouts_per_update=int(
                            runtime.config["algorithm"]["rollouts_per_update"]
                        ),
                        segment_successes=segment_successes,
                        segment_rollouts=segment_rollouts,
                        device=context.device,
                    )
                    print(
                        json.dumps(
                            {
                                "event": "checkpoint",
                                "rank": context.rank,
                                "unit": unit.key,
                                "path": str(checkpoint),
                                "interaction_cursor": counters["rollouts"],
                                "segment_successes": segment_successes,
                                "segment_rollouts": segment_rollouts,
                            }
                        ),
                        flush=True,
                    )
                    segment_successes = 0
                    segment_rollouts = 0
        finally:
            close_envs(envs)

    selected = None
    if next_update == args.total_updates:
        selected = select_unit_checkpoint(
            unit_dir, runtime.checkpoint_updates
        )
    result = {
        "unit": unit.key,
        "task_id": unit.task_id,
        "arm": unit.arm,
        "rank": context.rank,
        "next_update": next_update,
        "interaction_cursor": counters["rollouts"],
        "successes": counters["successes"],
        "env_steps": counters["env_steps"],
        "optimizer_updates": counters["optimizer_updates"],
        "status": (
            "complete" if next_update == args.total_updates else "segment_complete"
        ),
        "selected": selected,
        "wall_seconds": time.monotonic() - started,
    }
    write_json_atomic(unit_dir / "unit_result.json", result)
    return result


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
                        "task_count": len(runtime.task_ids),
                        "unit_count": len(runtime.units),
                        "total_updates_per_arm": args.total_updates,
                        "stop_after_update": args.stop_after_update,
                        "trainable": runtime.contract["trainable"],
                    }
                ),
                flush=True,
            )
        results = [
            _train_unit(
                args=args, runtime=runtime, context=context, unit=unit
            )
            for unit in runtime.assignments[context.rank]
        ]
        write_json_atomic(
            args.output_dir / f"rank_{context.rank:02d}.json",
            {
                "rank": context.rank,
                "assigned_units": [
                    unit.key for unit in runtime.assignments[context.rank]
                ],
                "results": results,
            },
        )
        barrier(context)
        if context.is_main:
            rank_rows = [
                read_json(args.output_dir / f"rank_{rank:02d}.json")
                for rank in range(context.world_size)
            ]
            unit_results = [
                result for row in rank_rows for result in row["results"]
            ]
            matched = matched_schedule_summary(
                args.output_dir, runtime.task_ids, args.stop_after_update
            )
            summary = {
                "schema_version": "ember_task_local_lora_rl_summary_v1",
                "mode": args.mode,
                "run_contract_sha256": runtime.contract_sha256,
                "complete": all(
                    result["status"] == "complete" for result in unit_results
                ),
                "task_ids": list(runtime.task_ids),
                "stop_after_update": args.stop_after_update,
                "matched_schedule": matched,
                "results": sorted(
                    unit_results,
                    key=lambda row: (int(row["task_id"]), str(row["arm"])),
                ),
            }
            write_json_atomic(args.output_dir / "run_summary.json", summary)
            print(
                json.dumps(
                    {
                        "event": "complete",
                        "complete": summary["complete"],
                        "units": len(unit_results),
                        **matched,
                    }
                ),
                flush=True,
            )
        barrier(context)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/task_local_lora_rl_v1.json",
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--writer-checkpoint", type=Path, required=True)
    parser.add_argument("--writer-rl-config", type=Path)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--total-updates", type=int, required=True)
    parser.add_argument("--stop-after-update", type=int)
    parser.add_argument("--checkpoint-updates", type=str, required=True)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.stop_after_update is None:
        args.stop_after_update = args.total_updates
    for name in (
        "config",
        "policy_path",
        "writer_checkpoint",
        "feature_cache",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.writer_rl_config is not None:
        args.writer_rl_config = args.writer_rl_config.resolve()
    return args


def main() -> None:
    train(finalize_args(build_parser().parse_args()))
