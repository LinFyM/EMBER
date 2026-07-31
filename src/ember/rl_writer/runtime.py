"""Construction and exact-resume state for the PI05 RL-Writer runtime."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist

from ember.pi05_assets import prepare_libero_config
from ember.pi05_eval_contract import (
    inspect_source_checkpoint,
    inspect_tokenizer,
    load_evaluation_authorities,
)
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import (
    DistributedContext,
    barrier,
    canonical_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.pi05_source_contract import reconcile_metrics
from ember.pi05_source_setup import load_policy, load_stats, seed_everything
from ember.reward.ledger import InteractionCursors
from ember.reward.protocol import RewardProtocolError, RewardTask
from ember.reward.rollout import RandomResetEnvironmentPool
from ember.rl_writer.checkpoint import (
    load_rl_writer_checkpoint,
    restore_rng,
)
from ember.rl_writer.contract import (
    REPO_ROOT,
    authority_path,
    build_contract,
    load_rl_writer_config,
    publish_contract,
    resolve_runtime,
    reward_tasks,
    task_for_update,
)
from ember.writer.as_contract import inspect_feature_cache, load_writer_config
from ember.writer.as_sampling import TeacherVideoSchedule
from ember.writer.feature_cache import WriterFeatureStore
from ember.writer.model import CompleteLoRAWriter


_CHECKPOINT_NAME = re.compile(r"update_([0-9]{8})")
_TARGET_SPECTRAL_RETRAIN_REQUIRED = (
    "RL-Writer runtime is unavailable until it is rebuilt and retrained for "
    "the canonical raw-video Target-Spectral Writer"
)


@dataclass
class RLWriterRuntime:
    args: argparse.Namespace
    context: DistributedContext
    config: dict[str, Any]
    tasks: tuple[RewardTask, ...]
    writer: CompleteLoRAWriter
    policy: torch.nn.Module
    processor: Pi05LiberoProcessor
    feature_store: WriterFeatureStore
    video_schedule: TeacherVideoSchedule
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    lora_contract: Any
    contract: dict[str, Any]
    contract_sha256: str
    total_updates: int
    checkpoint_updates: tuple[int, ...]
    next_update: int
    cursors: InteractionCursors
    successes: int
    reward_sum: float
    wall_nanoseconds: int
    metrics_path: Path
    metrics_rows: int
    env_pool: RandomResetEnvironmentPool


def _resume_update(path: Path | None) -> int:
    if path is None:
        return 0
    match = _CHECKPOINT_NAME.fullmatch(path.name)
    if match is None:
        raise RewardProtocolError("RL-Writer resume path is not an update checkpoint")
    return int(match.group(1))


def _fresh_writer(
    policy: torch.nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[CompleteLoRAWriter, Any, dict[str, Any]]:
    del policy, config, device
    raise RewardProtocolError(_TARGET_SPECTRAL_RETRAIN_REQUIRED)


def _optimizer(
    writer: CompleteLoRAWriter, config: Mapping[str, Any]
) -> torch.optim.Optimizer:
    values = config["optimization"]["optimizer"]
    return torch.optim.AdamW(
        writer.parameters(),
        lr=float(config["optimization"]["profile_learning_rate"]),
        betas=tuple(values["betas"]),
        eps=float(values["eps"]),
        weight_decay=float(values["weight_decay"]),
    )


def rank_ledger_summary(runtime: RLWriterRuntime, next_update: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    actions = 0
    successes = 0
    reward_sum = 0.0
    rollouts = int(runtime.config["algorithm"]["rollouts_per_task_update"])
    for update in range(next_update):
        task, cycle, _ = task_for_update(
            runtime.tasks,
            world_size=runtime.context.world_size,
            rank=runtime.context.rank,
            update=update,
            seed=int(runtime.config["data"]["task_schedule_seed"]),
        )
        for offset in range(rollouts):
            cursor = cycle * rollouts + offset
            path = (
                runtime.args.output_dir
                / "rollouts"
                / f"task_{task.global_task_id:03d}"
                / f"rollout_{cursor:08d}.json"
            )
            if not path.is_file():
                raise RewardProtocolError(
                    f"RL-Writer ledger prefix has a gap: {task.global_task_id}/{cursor}"
                )
            row = read_json(path)
            expected = (runtime.context.rank, update, task.global_task_id, cursor)
            observed = (
                int(row.get("producer_rank", -1)),
                int(row.get("update", -1)),
                int(row.get("global_task_id", -1)),
                int(row.get("rollout_cursor", -1)),
            )
            if observed != expected:
                raise RewardProtocolError("RL-Writer ledger schedule changed")
            digest.update(bytes.fromhex(sha256_file(path)))
            actions += int(row["steps"])
            successes += int(bool(row["success"]))
            reward_sum += float(row["reward_sum"])
    return {
        "rollout_cursor": next_update * rollouts,
        "environment_action_cursor": actions,
        "successes": successes,
        "reward_sum": reward_sum,
        "ledger_prefix_sha256": digest.hexdigest(),
    }


def _prepare_libero_paths(
    args: argparse.Namespace, context: DistributedContext
) -> dict[str, Any]:
    if context.is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths = prepare_libero_config(args.output_dir / "libero_config")
    else:
        paths = {}
    values: list[Any] = [paths]
    if context.world_size > 1:
        dist.broadcast_object_list(values, src=0, device=context.device)
    os.environ["LIBERO_CONFIG_PATH"] = str(
        (args.output_dir / "libero_config").resolve()
    )
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(context.local_rank)
    return values[0]


def _restore_runtime(runtime: RLWriterRuntime, initial: int) -> RLWriterRuntime:
    expected_rows = 0
    if runtime.args.resume is not None:
        ledger = rank_ledger_summary(runtime, initial)
        update, cursors, rng, expected_rows, counters = load_rl_writer_checkpoint(
            checkpoint=runtime.args.resume,
            context=runtime.context,
            writer=runtime.writer,
            optimizer=runtime.optimizer,
            scheduler=runtime.scheduler,
            contract_sha256=runtime.contract_sha256,
            tasks=runtime.tasks,
            task_schedule_seed=int(runtime.config["data"]["task_schedule_seed"]),
            rollouts_per_task_update=int(
                runtime.config["algorithm"]["rollouts_per_task_update"]
            ),
            video_schedule=runtime.video_schedule,
            ledger_summary=ledger,
        )
        runtime.next_update = update
        runtime.cursors = cursors
        runtime.successes = int(counters["successes"])
        runtime.reward_sum = float(counters["reward_sum"])
        runtime.wall_nanoseconds = int(counters["wall_nanoseconds"])
        restore_rng(rng, runtime.context)
    reconciliation: list[Any] = [None]
    if runtime.context.is_main:
        try:
            reconciliation[0] = {
                "rows": reconcile_metrics(
                    runtime.metrics_path,
                    initial,
                    expected_rows,
                    cursor_key="next_update",
                )
            }
        except Exception as error:
            reconciliation[0] = {"error": repr(error)}
    if runtime.context.world_size > 1:
        dist.broadcast_object_list(
            reconciliation, src=0, device=runtime.context.device
        )
    if reconciliation[0].get("error"):
        raise RewardProtocolError(reconciliation[0]["error"])
    runtime.metrics_rows = int(reconciliation[0]["rows"])
    runtime.writer.train()
    runtime.policy.eval()
    torch.cuda.reset_peak_memory_stats(runtime.context.device)
    return runtime


def build_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> RLWriterRuntime:
    config = load_rl_writer_config(args.config.resolve())
    if args.stage != config["sealed_stage"]:
        raise RewardProtocolError("RL-Writer stage requires its own immutable config")
    raise RewardProtocolError(_TARGET_SPECTRAL_RETRAIN_REQUIRED)
    total, checkpoints = resolve_runtime(args, config, context)
    initial = _resume_update(args.resume)
    if not 0 <= initial < args.stop_after_update:
        raise RewardProtocolError("RL-Writer resume cursor is outside this segment")
    seed_everything(int(config["optimization"]["seed"]), context)

    paths = _prepare_libero_paths(args, context)

    authorities = load_evaluation_authorities(
        authority_path(config, "evaluation_config"), REPO_ROOT
    )
    source = inspect_source_checkpoint(
        authorities, args.source_run, args.checkpoint, evaluation_mode="formal"
    )
    tokenizer = inspect_tokenizer(authorities, args.tokenizer_path)
    tasks = reward_tasks(config, stage=args.stage)
    task_ids = tuple(task.global_task_id for task in tasks)
    as_config = load_writer_config(authority_path(config, "as_writer_config"))
    cache = inspect_feature_cache(args.feature_cache, as_config, source, task_ids)
    policy = load_policy(
        Path(source["model_path"]), authorities.source_base_config, context.device
    )
    writer, lora, trainable = _fresh_writer(policy, config, context.device)
    optimizer = _optimizer(writer, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    contract = build_contract(
        args=args,
        config=config,
        context=context,
        source=source,
        tokenizer=tokenizer,
        feature_cache=cache,
        tasks=tasks,
        trainable=trainable,
        total_updates=total,
        checkpoint_updates=checkpoints,
        libero_paths=paths,
    )
    contract_sha = publish_contract(
        output_dir=args.output_dir,
        contract=contract,
        resume=args.resume,
        context=context,
    )
    if context.is_main:
        write_json_atomic(
            args.output_dir / "runtime_paths.json",
            {
                "source_run": str(args.source_run),
                "source_checkpoint": str(args.checkpoint),
                "tokenizer": str(args.tokenizer_path),
                "feature_cache": str(args.feature_cache),
            },
        )
    barrier(context)

    video_schedule = TeacherVideoSchedule(
        task_ids=task_ids,
        demo_indices=range(50),
        seed=int(config["data"]["teacher_video_seed"]),
    )
    feature_store = WriterFeatureStore(
        args.feature_cache,
        task_ids=task_ids,
        expected_extraction_sha256=str(cache["extraction_sha256"]),
        max_cached_tasks=4,
        expected_dim=int(as_config["writer"]["vision_feature_dim"]),
        expected_spatial_tokens=int(as_config["writer"]["vision_spatial_tokens"]),
        expected_run_contract_file_sha256=str(cache["run_contract_file_sha256"]),
        expected_manifest_file_sha256=str(cache["cache_manifest_file_sha256"]),
    )
    processor = Pi05LiberoProcessor(
        load_stats(
            authorities.source_base_config,
            authorities.source_base_config["data"]["active_task_ids"],
        ),
        args.tokenizer_path,
        int(authorities.source_base_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    runtime = RLWriterRuntime(
        args=args,
        context=context,
        config=config,
        tasks=tasks,
        writer=writer,
        policy=policy,
        processor=processor,
        feature_store=feature_store,
        video_schedule=video_schedule,
        optimizer=optimizer,
        scheduler=scheduler,
        lora_contract=lora,
        contract=contract,
        contract_sha256=contract_sha,
        total_updates=total,
        checkpoint_updates=checkpoints,
        next_update=initial,
        cursors=InteractionCursors(initial, initial, 0),
        successes=0,
        reward_sum=0.0,
        wall_nanoseconds=0,
        metrics_path=args.output_dir / "metrics.jsonl",
        metrics_rows=0,
        env_pool=RandomResetEnvironmentPool(
            bddl_root=Path(paths["bddl_files"]), render_resolution=256
        ),
    )
    return _restore_runtime(runtime, initial)
