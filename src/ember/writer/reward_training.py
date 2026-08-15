"""Train direct native-factor heads with paired common-state preference."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from ember.pi05_assets import prepare_libero_config
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import (
    DistributedContext,
    barrier,
    read_json,
    write_json_atomic,
)
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import (
    initialize_deferred_process_group,
    initialize_distributed,
    load_policy,
    load_stats,
    seed_everything,
)
from ember.reward.protocol import SUITE_HORIZONS, RewardTask
from ember.reward.rollout import RandomResetEnvironmentPool
from ember.writer.as_config import authority_path
from ember.writer.as_contract import load_run_authorities, writer_trainable_contract
from ember.writer.as_step import ParameterSlice, parameter_layout
from ember.writer.data import RawTeacherVideoStore, WriterTaskAuthority
from ember.writer.errors import WriterModelError
from ember.writer.reward_checkpoint import (
    checkpoint_cycle,
    load_reward_checkpoint,
    save_reward_checkpoint,
)
from ember.writer.reward_config import (
    REWARD_CONFIG,
    REWARD_LAUNCH_SCHEMA,
    load_reward_config,
    require_reward_mode,
)
from ember.writer.reward_cycle import run_cycle
from ember.writer.teacher_video_schedule import TeacherVideoSchedule
from ember.writer.training import build_writer


@dataclass
class RewardRuntime:
    args: argparse.Namespace
    context: DistributedContext
    config: dict[str, Any]
    base_config: dict[str, Any]
    source_config: dict[str, Any]
    tasks: tuple[RewardTask, ...]
    writer_tasks: tuple[WriterTaskAuthority, ...]
    video_store: RawTeacherVideoStore
    video_schedule: TeacherVideoSchedule
    language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    processor: Pi05LiberoProcessor
    env_pool: RandomResetEnvironmentPool
    policy: torch.nn.Module
    writer: torch.nn.Module
    identity_state: Mapping[str, torch.Tensor]
    lora_contract: Any
    optimizer: torch.optim.Optimizer
    trainable_parameters: tuple[torch.nn.Parameter, ...]
    gradient_layout: tuple[ParameterSlice, ...]
    contract: dict[str, Any]
    start_cycle: int
    stop_cycle: int
    metrics_path: Path


def _load_tasks(
    *, data_root: Path, base_config: Mapping[str, Any]
) -> tuple[tuple[RewardTask, ...], tuple[WriterTaskAuthority, ...]]:
    manifest = read_json(authority_path(base_config, "target_data_manifest"))
    reward_tasks, writer_tasks = [], []
    for row in manifest["tasks"]:
        if row["split_role"] != "train":
            continue
        hdf5 = row["hdf5"]
        path = (data_root / str(hdf5["relative_path"])).resolve()
        writer_tasks.append(
            WriterTaskAuthority(
                task_id=int(row["global_task_id"]),
                language=str(row["language"]),
                path=path,
                expected_bytes=int(hdf5["bytes"]),
            )
        )
        bddl = row["bddl"]
        reward_tasks.append(
            RewardTask(
                suite=str(row["suite"]),
                task_id=int(row["task_id"]),
                global_task_id=int(row["global_task_id"]),
                split_role="train",
                language=str(row["language"]),
                problem_folder=str(row["problem_folder"]),
                bddl_file=str(bddl["filename"]),
                bddl_bytes=int(bddl["bytes"]),
                bddl_sha256=None,
                horizon=SUITE_HORIZONS[str(row["suite"])],
            )
        )
    reward_tasks.sort(key=lambda task: task.global_task_id)
    writer_tasks.sort(key=lambda task: task.task_id)
    if len(reward_tasks) != 24 or [task.global_task_id for task in reward_tasks] != [
        task.task_id for task in writer_tasks
    ]:
        raise WriterModelError("direct-factor run lost train24 task authority")
    return tuple(reward_tasks), tuple(writer_tasks)


def _optimizer(writer: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.AdamW:
    cell = config["optimization"]["optimizer"]
    return torch.optim.AdamW(
        (value for value in writer.parameters() if value.requires_grad),
        lr=float(cell["lr"]),
        betas=tuple(cell["betas"]),
        eps=float(cell["eps"]),
        weight_decay=float(cell["weight_decay"]),
    )


def _publish_contract(
    runtime_args: argparse.Namespace,
    context: DistributedContext,
    contract: Mapping[str, Any],
) -> None:
    if context.is_main:
        path = runtime_args.output_dir / "run_contract.json"
        if runtime_args.resume is None:
            if runtime_args.output_dir.exists() and any(
                runtime_args.output_dir.iterdir()
            ):
                raise WriterModelError("fresh direct-factor output is not empty")
            runtime_args.output_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(path, dict(contract))
        elif not path.is_file() or read_json(path) != dict(contract):
            raise WriterModelError(
                "direct-factor exact-resume launch contract changed"
            )
        append_jsonl(
            runtime_args.output_dir / "invocations.jsonl",
            {
                "argv": list(os.sys.argv),
                "host": socket.gethostname(),
                "resume": str(runtime_args.resume) if runtime_args.resume else None,
                "stop_after_cycle": runtime_args.stop_after_cycle,
                "started_unix": time.time(),
            },
        )
    barrier(context)


def _contract(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    config: Mapping[str, Any],
    base_config: Mapping[str, Any],
    source: Mapping[str, Any],
    trainable: Mapping[str, Any],
    tasks: tuple[RewardTask, ...],
) -> dict[str, Any]:
    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "device": str(context.device),
        "numa_node": context.numa_node,
        "cpu_affinity": list(context.cpu_affinity or ()),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    topology: list[Any] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(topology, local)
    else:
        topology[0] = local
    state = git_state(Path(__file__).resolve().parents[3])
    return {
        "schema_version": REWARD_LAUNCH_SCHEMA,
        "mode": args.mode,
        "git": {"branch": state["branch"], "commit": state["commit"]},
        "config_path": str(args.config),
        "base_as_config_path": config["resolved_base_as_config"],
        "base_as_schema": base_config["schema_version"],
        "initialization": {
            **dict(config["initialization"]),
            "checkpoint": config["resolved_cold_start"],
        },
        "source": dict(source),
        "information_wall": dict(config["information_wall"]),
        "writer": dict(base_config["writer"]),
        "data": dict(config["data"]),
        "environment": dict(config["environment"]),
        "objective": dict(config["objective"]),
        "rng": dict(config["rng"]),
        "optimization": dict(config["optimization"]),
        "formal_run": dict(config["formal_run"]),
        "task_ids": [task.global_task_id for task in tasks],
        "runtime": {
            "world_size": context.world_size,
            "rank_topology": topology,
            "total_cycles": int(config["formal_run"]["total_cycles"]),
            "task_assignment": config["data"]["task_queue"],
        },
        "trainable": dict(trainable),
    }


def _load_direct_factor_models(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    config: dict[str, Any],
    base_config: Mapping[str, Any],
    source: Mapping[str, Any],
    source_base_config: Mapping[str, Any],
) -> tuple[torch.nn.Module, torch.nn.Module, Any, dict[str, Any], torch.optim.Optimizer]:
    cold_start = (
        args.source_run.resolve().parents[2] / config["cold_start_relative"]
    ).resolve()
    if not (cold_start / "writer.safetensors").is_file():
        raise WriterModelError("direct-factor LPCP cold start is missing")
    config["resolved_cold_start"] = str(cold_start)
    policy = load_policy(
        Path(source["model_path"]), source_base_config, context.device
    )
    writer, lora = build_writer(
        base_config,
        policy,
        asset_root=args.source_run.resolve().parents[2],
    )
    writer.to(context.device)
    writer.load_lpcp_state_(
        load_file(str(cold_start / "writer.safetensors"), device=str(context.device))
    )
    writer.requires_grad_(False)
    writer.factor_commitment.requires_grad_(True)
    writer.eval()
    trainable_names = tuple(
        name for name, value in writer.named_parameters() if value.requires_grad
    )
    if (
        not trainable_names
        or any(
            not name.startswith("factor_commitment.") for name in trainable_names
        )
        or sum(
            value.numel()
            for value in writer.factor_commitment.parameters()
            if value.requires_grad
        )
        != 1_654_784
    ):
        raise WriterModelError(
            "direct native-factor commitment must train only 1654784 parameters"
        )
    trainable = writer_trainable_contract(writer, policy, lora)
    trainable["object"] = (
        "v6_lpcp_direct_factor_paired_common_state_preference_only"
    )
    trainable["writer_trainable_parameter_names"] = list(trainable_names)
    return policy, writer, lora, trainable, _optimizer(writer, config)


def _condition_inputs(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    config: Mapping[str, Any],
    base_config: Mapping[str, Any],
    source_config: Mapping[str, Any],
    tasks: tuple[RewardTask, ...],
    writer_tasks: tuple[WriterTaskAuthority, ...],
) -> tuple[
    Pi05LiberoProcessor,
    RawTeacherVideoStore,
    dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    TeacherVideoSchedule,
    RandomResetEnvironmentPool,
]:
    length = int(source_config["features"]["tokenizer_max_length"])
    processor = Pi05LiberoProcessor(
        load_stats(source_config, source_config["data"]["active_task_ids"]),
        args.tokenizer_path,
        length,
        str(context.device),
    )
    tokenizer = Pi05TeacherPrefixTokenizer(
        args.tokenizer_path, length, str(context.device)
    )
    language = {task.task_id: tokenizer([task.language]) for task in writer_tasks}
    store = RawTeacherVideoStore(
        writer_tasks,
        frame_stride=int(base_config["writer"]["frame_stride"]),
        max_open_files=int(base_config["data"]["video_open_files_per_rank"]),
    )
    first, last = config["data"]["demo_indices"]
    schedule = TeacherVideoSchedule(
        task_ids=[task.global_task_id for task in tasks],
        demo_indices=range(first, last + 1),
        seed=int(config["data"]["teacher_video_seed"]),
        videos_per_visit=int(config["data"]["videos_per_task"]),
    )
    libero_paths = prepare_libero_config(
        args.output_dir / f".libero_config_rank_{context.rank:02d}"
    )
    env_pool = RandomResetEnvironmentPool(
        bddl_root=Path(libero_paths["bddl_files"]),
        assets_root=Path(libero_paths["assets"]),
        render_resolution=int(config["environment"]["render_resolution"]),
    )
    return processor, store, language, schedule, env_pool


def prepare_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> RewardRuntime:
    config, base_config = load_reward_config(args.config)
    require_reward_mode(config, args.mode)
    if args.mode == "smoke" and context.world_size != 1:
        raise WriterModelError("direct-factor smoke uses one GPU")
    allowed = config["formal_run"]["allowed_world_sizes"]
    if context.world_size not in allowed:
        raise WriterModelError("direct-factor world size is outside 1--6")
    if args.mode == "formal":
        state = git_state(Path(__file__).resolve().parents[3])
        if not git_state_is_clean_pushed_or_frozen_authority(state):
            raise WriterModelError(
                "formal direct-factor training requires clean pushed Git"
            )
    seed_everything(int(config["rng"]["optimizer_seed"]), context)
    authorities, source, _ = load_run_authorities(args, base_config)
    tasks, writer_tasks = _load_tasks(
        data_root=args.data_root, base_config=base_config
    )
    policy, writer, lora, trainable, optimizer = _load_direct_factor_models(
        args=args,
        context=context,
        config=config,
        base_config=base_config,
        source=source,
        source_base_config=authorities.source_base_config,
    )
    initialize_deferred_process_group(
        context,
        rendezvous_root=args.output_dir.parent,
        collective_timeout=timedelta(
            minutes=int(
                config["optimization"]["distributed"][
                    "collective_timeout_minutes"
                ]
            )
        ),
    )
    contract = _contract(
        args=args,
        context=context,
        config=config,
        base_config=base_config,
        source=source,
        trainable=trainable,
        tasks=tasks,
    )
    _publish_contract(args, context, contract)
    start_cycle = checkpoint_cycle(args.resume)
    if args.resume is not None:
        loaded, _ = load_reward_checkpoint(
            checkpoint=args.resume,
            context=context,
            writer=writer,
            optimizer=optimizer,
            contract=contract,
        )
        if loaded != start_cycle:
            raise WriterModelError("direct-factor resume cursor changed")
    stop_cycle = (
        1
        if args.mode == "smoke"
        else int(args.stop_after_cycle or config["formal_run"]["stage_stop_cycles"][0])
    )
    if args.mode == "formal" and (
        stop_cycle not in config["formal_run"]["stage_stop_cycles"]
        or not start_cycle < stop_cycle
    ):
        raise WriterModelError("direct-factor formal stop boundary changed")
    source_config = authorities.source_base_config
    processor, store, language, schedule, env_pool = _condition_inputs(
        args=args,
        context=context,
        config=config,
        base_config=base_config,
        source_config=source_config,
        tasks=tasks,
        writer_tasks=writer_tasks,
    )
    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)
    return RewardRuntime(
        args=args,
        context=context,
        config=config,
        base_config=base_config,
        source_config=source_config,
        tasks=tasks,
        writer_tasks=writer_tasks,
        video_store=store,
        video_schedule=schedule,
        language_tokens=language,
        processor=processor,
        env_pool=env_pool,
        policy=policy,
        writer=writer,
        identity_state=writer.template_state(),
        lora_contract=lora,
        optimizer=optimizer,
        trainable_parameters=tuple(
            value for value in writer.parameters() if value.requires_grad
        ),
        gradient_layout=parameter_layout(writer),
        contract=contract,
        start_cycle=start_cycle,
        stop_cycle=stop_cycle,
        metrics_path=args.output_dir / "metrics.jsonl",
    )


def train(args: argparse.Namespace) -> None:
    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    runtime: RewardRuntime | None = None
    try:
        runtime = prepare_runtime(args, context)
        if context.is_main:
            print(
                json.dumps(
                    {
                        "event": "start",
                        "mode": args.mode,
                        "world_size": context.world_size,
                        "start_cycle": runtime.start_cycle,
                        "stop_cycle": runtime.stop_cycle,
                        "trainable_parameters": runtime.gradient_layout[-1].stop,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        for cycle in range(runtime.start_cycle + 1, runtime.stop_cycle + 1):
            row = run_cycle(runtime, cycle)
            if context.is_main:
                append_jsonl(runtime.metrics_path, row)
                print(json.dumps(row, sort_keys=True), flush=True)
            if args.mode == "formal":
                save_reward_checkpoint(
                    output_dir=args.output_dir,
                    cycle=cycle,
                    context=context,
                    writer=runtime.writer,
                    optimizer=runtime.optimizer,
                    contract=runtime.contract,
                    metrics_rows=cycle,
                )
            elif context.is_main:
                save_file(
                    {
                        name: value.detach().cpu().contiguous()
                        for name, value in (
                            runtime.writer.factor_commitment.state_dict().items()
                        )
                    },
                    str(args.output_dir / "factor_commitment.safetensors"),
                )
        if context.is_main:
            write_json_atomic(
                args.output_dir / "completion.json",
                {
                    "schema_version": (
                        "ember_pi05_v6_lpcp_direct_factor_paired_common_state_"
                        "preference_"
                        "completion_v1"
                    ),
                    "mode": args.mode,
                    "completed_cycle": runtime.stop_cycle,
                    "strict400_required": args.mode == "formal",
                },
            )
    finally:
        if runtime is not None:
            runtime.env_pool.close()
            runtime.video_store.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REWARD_CONFIG)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-cycle", type=int)
    parser.add_argument("--smoke-task-id", type=int)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
        "source_run",
        "checkpoint",
        "tokenizer_path",
        "data_root",
        "output_dir",
        "resume",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    for name in ("config", "source_run", "checkpoint", "tokenizer_path", "data_root"):
        if not getattr(args, name).exists():
            raise WriterModelError(
                f"missing reward training path: {getattr(args, name)}"
            )
    return args
