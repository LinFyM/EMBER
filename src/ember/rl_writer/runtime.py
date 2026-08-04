"""Construction and exact-resume state for task-relative Flow-Credit Writer."""

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
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import (
    DistributedContext,
    barrier,
    canonical_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.pi05_source_contract import reconcile_metrics
from ember.pi05_source_setup import (
    initialize_deferred_process_group,
    load_policy,
    load_stats,
    seed_everything,
)
from ember.reward.ledger import InteractionCursors
from ember.reward.protocol import RewardProtocolError, RewardTask
from ember.reward.rollout import RandomResetEnvironmentPool
from ember.rl_writer.checkpoint import load_rl_writer_checkpoint, restore_rng
from ember.rl_writer.contract import (
    REPO_ROOT,
    authority_path,
    build_contract,
    cycle_assignments,
    load_rl_writer_config,
    publish_contract,
    resolve_runtime,
    reward_tasks,
)
from ember.writer.as_config import writer_stage
from ember.writer.as_contract import inspect_video_data, load_writer_config
from ember.writer.as_sampling import TeacherVideoSchedule
from ember.writer.checkpoint import initialize_writer_phase
from ember.writer.data import RawTeacherVideoStore, WriterTaskAuthority
from ember.writer.model import CompleteLoRAWriter
from ember.writer.topology import visible_physical_cuda_index
from ember.writer.training import build_writer


_CHECKPOINT_NAME = re.compile(r"cycle_([0-9]{8})")


@dataclass
class RLWriterRuntime:
    args: argparse.Namespace
    context: DistributedContext
    config: dict[str, Any]
    as_config: dict[str, Any]
    tasks: tuple[RewardTask, ...]
    task_authorities: dict[int, WriterTaskAuthority]
    writer: CompleteLoRAWriter
    policy: torch.nn.Module
    processor: Pi05LiberoProcessor
    tokenizer: Pi05TeacherPrefixTokenizer
    video_store: RawTeacherVideoStore
    video_schedule: TeacherVideoSchedule
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    lora_contract: Any
    identity_state: dict[str, torch.Tensor]
    contract: dict[str, Any]
    contract_sha256: str
    total_cycles: int
    checkpoint_cycles: tuple[int, ...]
    learning_epochs: int
    next_cycle: int
    cursors: InteractionCursors
    successes: int
    reward_sum: float
    wall_nanoseconds: int
    metrics_path: Path
    metrics_rows: int
    env_pool: RandomResetEnvironmentPool


@dataclass(frozen=True)
class _LocalWriterModels:
    authorities: Any
    source: dict[str, Any]
    tokenizer_record: dict[str, Any]
    as_config: dict[str, Any]
    policy: torch.nn.Module
    writer: CompleteLoRAWriter
    lora_contract: Any
    trainable: dict[str, Any]
    identity_state: dict[str, torch.Tensor]


def _resume_cycle(path: Path | None) -> int:
    if path is None:
        return 0
    match = _CHECKPOINT_NAME.fullmatch(path.name)
    if match is None:
        raise RewardProtocolError("Flow-Credit resume path is not a cycle checkpoint")
    return int(match.group(1))


def _optimizer(
    writer: CompleteLoRAWriter, config: Mapping[str, Any]
) -> torch.optim.Optimizer:
    values = config["optimization"]["optimizer"]
    parameters = [value for value in writer.parameters() if value.requires_grad]
    if not parameters:
        raise RewardProtocolError("progress-credit Writer has no trainable downstream")
    return torch.optim.AdamW(
        parameters,
        lr=float(config["optimization"]["learning_rate"]),
        betas=tuple(values["betas"]),
        eps=float(values["eps"]),
        weight_decay=float(values["weight_decay"]),
    )


def _prepare_progress_optimizer(
    writer: CompleteLoRAWriter,
    coldstart: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    if coldstart.get("mode") != "writer_weight_warm_start":
        raise RewardProtocolError("Flow-Credit requires an independent AS cold start")
    for parameter in writer.semantic_encoder.parameters():
        parameter.requires_grad_(False)
    optimizer = _optimizer(writer, config)
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


def rank_ledger_summary(runtime: RLWriterRuntime, next_cycle: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    actions = 0
    successes = 0
    reward_sum = 0.0
    rollouts = int(runtime.config["algorithm"]["rollouts_per_task_condition"])
    local_rollouts = 0
    for cycle in range(next_cycle):
        assigned = cycle_assignments(
            runtime.tasks,
            world_size=runtime.context.world_size,
            cycle=cycle,
            seed=int(runtime.config["data"]["task_schedule_seed"]),
        )[runtime.context.rank]
        for task in assigned:
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
                        f"Flow-Credit ledger prefix gap: {task.global_task_id}/{cursor}"
                    )
                row = read_json(path)
                expected = (
                    runtime.context.rank,
                    cycle,
                    task.global_task_id,
                    cursor,
                )
                observed = (
                    int(row.get("producer_rank", -1)),
                    int(row.get("outer_cycle", -1)),
                    int(row.get("global_task_id", -1)),
                    int(row.get("rollout_cursor", -1)),
                )
                if observed != expected:
                    raise RewardProtocolError("Flow-Credit ledger schedule changed")
                digest.update(bytes.fromhex(sha256_file(path)))
                actions += int(row["steps"])
                successes += int(bool(row["success"]))
                reward_sum += float(row["reward_sum"])
                local_rollouts += 1
    return {
        "rollout_cursor": local_rollouts,
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
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(
        visible_physical_cuda_index(context.local_rank)
    )
    return values[0]


def _broadcast_main(
    context: DistributedContext, callback: Any
) -> dict[str, Any]:
    payload: list[Any] = [None]
    if context.is_main:
        try:
            payload[0] = {"value": callback()}
        except Exception as error:
            payload[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=context.device)
    if payload[0].get("error"):
        raise RewardProtocolError(payload[0]["error"])
    return dict(payload[0]["value"])


def _task_authorities(
    config: Mapping[str, Any], data_root: Path
) -> dict[int, WriterTaskAuthority]:
    target = read_json(authority_path(config, "target_data_manifest"))
    result = {}
    for row in target.get("tasks", []):
        if row.get("split_role") != "train":
            continue
        task_id = int(row["global_task_id"])
        path = (data_root / str(row["hdf5"]["relative_path"])).resolve()
        if not path.is_relative_to(data_root):
            raise RewardProtocolError("Flow-Credit video path escaped data root")
        result[task_id] = WriterTaskAuthority(
            task_id=task_id,
            language=str(row["language"]),
            path=path,
            expected_bytes=int(row["hdf5"]["bytes"]),
        )
    if len(result) != 24:
        raise RewardProtocolError("Flow-Credit video authorities changed")
    return result


def _restore_runtime(runtime: RLWriterRuntime, initial: int) -> RLWriterRuntime:
    expected_rows = 0
    if runtime.args.resume is not None:
        ledger = rank_ledger_summary(runtime, initial)
        cycle, cursors, rng, expected_rows, counters = load_rl_writer_checkpoint(
            checkpoint=runtime.args.resume,
            context=runtime.context,
            writer=runtime.writer,
            optimizer=runtime.optimizer,
            scheduler=runtime.scheduler,
            contract_sha256=runtime.contract_sha256,
            tasks=runtime.tasks,
            task_schedule_seed=int(runtime.config["data"]["task_schedule_seed"]),
            rollouts_per_task=int(
                runtime.config["algorithm"]["rollouts_per_task_condition"]
            ),
            video_schedule=runtime.video_schedule,
            ledger_summary=ledger,
            learning_epochs=runtime.learning_epochs,
        )
        runtime.next_cycle = cycle
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
                    cursor_key="next_cycle",
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


def _load_local_writer_models(
    args: argparse.Namespace,
    context: DistributedContext,
    config: Mapping[str, Any],
) -> _LocalWriterModels:
    """Load policy and Writer before NCCL claims the selected devices."""

    authorities = load_evaluation_authorities(
        authority_path(config, "evaluation_config"), REPO_ROOT
    )
    source = inspect_source_checkpoint(
        authorities, args.source_run, args.checkpoint, evaluation_mode="formal"
    )
    tokenizer_record = inspect_tokenizer(authorities, args.tokenizer_path)
    as_config = load_writer_config(authority_path(config, "as_writer_config"))
    if writer_stage(as_config) != "development":
        raise RewardProtocolError("Flow-Credit cold start is not development AS")
    policy = load_policy(
        Path(source["model_path"]), authorities.source_base_config, context.device
    )
    writer, lora, trainable, identity = build_writer(as_config, policy)
    writer.to(context.device)
    return _LocalWriterModels(
        authorities=authorities,
        source=source,
        tokenizer_record=tokenizer_record,
        as_config=as_config,
        policy=policy,
        writer=writer,
        lora_contract=lora,
        trainable=trainable,
        identity_state=identity,
    )


def _publish_runtime_contract(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    config: Mapping[str, Any],
    models: _LocalWriterModels,
    coldstart: Mapping[str, Any],
    video_data: Mapping[str, Any],
    tasks: tuple[RewardTask, ...],
    total_cycles: int,
    checkpoint_cycles: tuple[int, ...],
    learning_epochs: int,
    libero_paths: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    trainable_names = sorted(
        name for name, value in models.writer.named_parameters() if value.requires_grad
    )
    frozen_observer_names = sorted(
        f"semantic_encoder.{name}"
        for name, value in models.writer.semantic_encoder.named_parameters()
        if not value.requires_grad
    )
    trainable = {
        **models.trainable,
        "object": "shared_task_grounded_progress_credit_writer_downstream_only",
        "coldstart_teacher_action_phase_closed": True,
        "semantic_encoder_frozen": True,
        "rl_trainable_parameter_count": sum(
            value.numel() for value in models.writer.parameters() if value.requires_grad
        ),
        "rl_trainable_parameter_name_count": len(trainable_names),
        "rl_trainable_parameter_names_sha256": canonical_hash(trainable_names),
        "progress_observer_parameter_count": sum(
            value.numel() for value in models.writer.semantic_encoder.parameters()
        ),
        "progress_observer_parameter_names_sha256": canonical_hash(
            frozen_observer_names
        ),
    }
    contract = build_contract(
        args=args,
        config=config,
        context=context,
        source=models.source,
        tokenizer=models.tokenizer_record,
        coldstart=coldstart,
        video_data=video_data,
        tasks=tasks,
        trainable=trainable,
        total_cycles=total_cycles,
        checkpoint_cycles=checkpoint_cycles,
        learning_epochs=learning_epochs,
        libero_paths=libero_paths,
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
                "video_data_root": str(args.data_root),
                "coldstart_checkpoint": str(args.coldstart_checkpoint),
            },
        )
    barrier(context)
    return contract, contract_sha


def build_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> RLWriterRuntime:
    config = load_rl_writer_config(args.config.resolve())
    total, checkpoints, learning_epochs = resolve_runtime(args, config, context)
    initial = _resume_cycle(args.resume)
    if not 0 <= initial < args.stop_after_cycle:
        raise RewardProtocolError("Flow-Credit resume cursor is outside this segment")
    seed_everything(int(config["optimization"]["seed"]), context)
    models = _load_local_writer_models(args, context, config)
    initialize_deferred_process_group(
        context, rendezvous_root=args.output_dir.parent
    )
    coldstart = initialize_writer_phase(
        args.coldstart_checkpoint,
        context,
        "development",
        models.source,
        models.as_config["authorities"],
        models.as_config["writer"],
        models.writer,
        str(models.trainable["lora_contract_sha256"]),
    )
    optimizer, scheduler = _prepare_progress_optimizer(
        models.writer, coldstart, config
    )
    paths = _prepare_libero_paths(args, context)

    tasks = reward_tasks(config)
    task_ids = tuple(task.global_task_id for task in tasks)
    video_data = _broadcast_main(
        context,
        lambda: inspect_video_data(
            args.data_root,
            models.as_config,
            task_ids,
            verify_hashes=False,
        ),
    )
    task_authorities = _task_authorities(config, args.data_root)
    contract, contract_sha = _publish_runtime_contract(
        args=args,
        context=context,
        config=config,
        models=models,
        coldstart=coldstart,
        video_data=video_data,
        tasks=tasks,
        total_cycles=total,
        checkpoint_cycles=checkpoints,
        learning_epochs=learning_epochs,
        libero_paths=paths,
    )

    video_schedule = TeacherVideoSchedule(
        task_ids=task_ids,
        demo_indices=range(50),
        seed=int(config["data"]["teacher_video_seed"]),
    )
    processor = Pi05LiberoProcessor(
        load_stats(
            models.authorities.source_base_config,
            models.authorities.source_base_config["data"]["active_task_ids"],
        ),
        args.tokenizer_path,
        int(models.authorities.source_base_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    tokenizer = Pi05TeacherPrefixTokenizer(
        args.tokenizer_path,
        int(models.authorities.source_base_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    tasks_per_rank = len(tasks) // context.world_size
    runtime = RLWriterRuntime(
        args=args,
        context=context,
        config=config,
        as_config=models.as_config,
        tasks=tasks,
        task_authorities=task_authorities,
        writer=models.writer,
        policy=models.policy,
        processor=processor,
        tokenizer=tokenizer,
        video_store=RawTeacherVideoStore(
            tuple(task_authorities.values()),
            frame_stride=int(models.as_config["writer"]["frame_stride"]),
            max_open_files=max(2, tasks_per_rank),
        ),
        video_schedule=video_schedule,
        optimizer=optimizer,
        scheduler=scheduler,
        lora_contract=models.lora_contract,
        identity_state=models.identity_state,
        contract=contract,
        contract_sha256=contract_sha,
        total_cycles=total,
        checkpoint_cycles=checkpoints,
        learning_epochs=learning_epochs,
        next_cycle=initial,
        cursors=InteractionCursors(
            initial
            * tasks_per_rank
            * int(config["algorithm"]["rollouts_per_task_condition"]),
            0,
            initial * learning_epochs,
        ),
        successes=0,
        reward_sum=0.0,
        wall_nanoseconds=0,
        metrics_path=args.output_dir / "metrics.jsonl",
        metrics_rows=0,
        env_pool=RandomResetEnvironmentPool(
            bddl_root=Path(paths["bddl_files"]),
            assets_root=Path(paths["assets"]),
            render_resolution=256,
        ),
    )
    return _restore_runtime(runtime, initial)
