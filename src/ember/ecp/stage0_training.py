"""Canonical native-observer training entrypoint for EMBER-ECP Stage 0A."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ember.ecp.checkpoint import (
    checkpoint_macro,
    load_ecp_checkpoint,
    save_ecp_checkpoint,
)
from ember.ecp.contracts import build_target_owners
from ember.ecp.stage0 import ECPStage0Model
from ember.ecp.stage0_data import ECPStage0Schedule, ECPStage0Task, load_stage0_tasks
from ember.ecp.stage0_train_step import run_stage0_macro
from ember.functional_adaptation.action_alignment import PrivilegedMetaActionStore
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import (
    DistributedContext,
    read_json,
    write_json_atomic,
)
from ember.pi05_source_contract import append_jsonl, reconcile_metrics
from ember.pi05_source_setup import (
    initialize_deferred_process_group,
    initialize_distributed,
    load_config,
    load_policy,
    load_stats,
    seed_everything,
)
from ember.writer.data import RawTeacherVideoStore


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_SCHEMA = "ember_ecp_stage0_native_run_v1"
STAGE = "stage0_native"


@dataclass
class ECPStage0Runtime:
    args: argparse.Namespace
    config: dict[str, Any]
    context: DistributedContext
    tasks: tuple[ECPStage0Task, ...]
    task_by_id: dict[int, ECPStage0Task]
    schedule: ECPStage0Schedule
    video_store: RawTeacherVideoStore
    action_store: PrivilegedMetaActionStore
    language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor]]
    policy: torch.nn.Module
    model: ECPStage0Model
    action_meta_lora: torch.nn.Module | None
    trainable_parameters: tuple[torch.nn.Parameter, ...]
    frozen_parameters: tuple[torch.nn.Parameter, ...]
    checkpoint_module: torch.nn.Module
    checkpoint_stage: str
    run_schema: str
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    tasks_per_rank: int
    total_macros: int
    stop_after_macro: int
    checkpoint_macros: tuple[int, ...]
    start_macro: int
    metrics_rows: int
    run_contract: dict[str, Any]


def stage0_authority_path(config: dict[str, Any], name: str) -> Path:
    return REPO_ROOT / str(config["authorities"][name])


def load_stage0_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "ember_ecp_stage0_native_v1":
        raise ValueError("unsupported ECP Stage 0 config")
    return config


def _resolve_runtime(
    args: argparse.Namespace,
    config: dict[str, Any],
    context: DistributedContext,
) -> tuple[int, int, tuple[int, ...], int]:
    cell = config["formal_run" if args.mode == "formal" else "profile_defaults"]
    if context.world_size not in cell["allowed_world_sizes"]:
        raise ValueError("ECP Stage 0 world size is outside its launch contract")
    total = int(cell["total_macros"])
    stop = int(args.stop_after_macro or cell.get("stop_after_macro", total))
    checkpoints = tuple(map(int, cell["checkpoint_macros"]))
    tasks_per_rank = (
        90 // context.world_size
        if args.mode == "formal"
        else int(cell["tasks_per_rank_per_macro"])
    )
    if args.mode == "formal":
        if stop not in set(map(int, cell["stage_stop_macros"])):
            raise ValueError("formal ECP Stage 0 stop is not pre-registered")
        state = git_state(REPO_ROOT)
        if not git_state_is_clean_pushed_or_frozen_authority(state):
            raise ValueError("formal ECP Stage 0 requires a clean pushed commit")
    return total, stop, checkpoints, tasks_per_rank


def build_stage0_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    total_macros: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    cell = config["optimization"]["scheduler"]
    warmup = int(cell["warmup_macros"])
    floor = float(cell["decay_lr"]) / float(cell["peak_lr"])

    def scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_macros - warmup, 1)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def build_stage0_optimizer(
    parameters: Any, optimization: dict[str, Any]
) -> torch.optim.AdamW:
    cell = optimization["optimizer"]
    return torch.optim.AdamW(
        parameters,
        lr=float(optimization["scheduler"]["peak_lr"]),
        betas=tuple(cell["betas"]),
        eps=float(cell["eps"]),
        weight_decay=float(cell["weight_decay"]),
    )


def build_stage0_tasks_and_schedule(
    config: dict[str, Any], data_root: Path
) -> tuple[tuple[ECPStage0Task, ...], ECPStage0Schedule]:
    tasks = load_stage0_tasks(
        source_manifest=stage0_authority_path(config, "source_manifest"),
        target_manifest=stage0_authority_path(config, "target_manifest"),
        data_root=data_root,
        held_target_ids=config["task_roles"]["target_held_task_ids"],
    )
    return tasks, ECPStage0Schedule(
        tasks,
        seed=int(config["data"]["pair_seed"]),
        frame_stride=int(config["data"]["frame_stride"]),
    )


def stage0_source_authority(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint
    if (
        checkpoint.parent.parent != args.source_run
        or checkpoint.parent.name != "checkpoints"
    ):
        raise ValueError("ECP source checkpoint escaped its retained source run")
    model_path = checkpoint / "policy"
    trainer = read_json(checkpoint / "trainer_state.json")
    if int(trainer.get("optimizer_step", -1)) != 1000:
        raise ValueError("ECP requires the retained source step1000 policy")
    return {
        "source_run": str(args.source_run),
        "checkpoint": str(checkpoint),
        "model_path": str(model_path),
        "optimizer_step": 1000,
        "model_files": {
            name: (model_path / name).stat().st_size
            for name in ("config.json", "model.safetensors")
        },
    }


def publish_stage0_contract(
    runtime_args: argparse.Namespace,
    context: DistributedContext,
    contract: dict[str, Any],
) -> None:
    payload: list[Any] = [None]
    if context.is_main:
        try:
            path = runtime_args.output_dir / "run_contract.json"
            if runtime_args.resume is None:
                if runtime_args.output_dir.exists() and any(
                    runtime_args.output_dir.iterdir()
                ):
                    raise ValueError("fresh ECP Stage 0 output directory is not empty")
                runtime_args.output_dir.mkdir(parents=True, exist_ok=True)
                write_json_atomic(path, contract)
            elif not path.is_file() or read_json(path) != contract:
                raise ValueError("exact-resume ECP Stage 0 contract changed")
            append_jsonl(
                runtime_args.output_dir / "invocations.jsonl",
                {
                    "argv": sys.argv,
                    "host": socket.gethostname(),
                    "resume": str(runtime_args.resume) if runtime_args.resume else None,
                    "started_unix": time.time(),
                },
            )
            payload[0] = {"ok": True}
        except Exception as error:
            payload[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=context.device)
    if payload[0].get("error"):
        raise ValueError(payload[0]["error"])


def _build_contract(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    context: DistributedContext,
    tasks: tuple[ECPStage0Task, ...],
    source: dict[str, Any],
    total_macros: int,
    checkpoint_macros: tuple[int, ...],
    tasks_per_rank: int,
    model: torch.nn.Module,
) -> dict[str, Any]:
    topology: list[Any] = [None] * context.world_size
    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "device": str(context.device),
        "numa_node": context.numa_node,
        "cpu_affinity": list(context.cpu_affinity or ()),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if context.world_size > 1:
        dist.all_gather_object(topology, local)
    else:
        topology[0] = local
    state = git_state(REPO_ROOT)
    return {
        "schema_version": RUN_SCHEMA,
        "stage": STAGE,
        "mode": args.mode,
        "git": {"branch": state["branch"], "commit": state["commit"]},
        "config_path": str(args.config),
        "authorities": dict(config["authorities"]),
        "source": source,
        "tokenizer": {
            "path": str(args.tokenizer_path),
            "bytes": args.tokenizer_path.stat().st_size,
        },
        "data_root": str(args.data_root),
        "information_wall": dict(config["information_wall"]),
        "task_roles": dict(config["task_roles"]),
        "tasks": [
            {
                "authority_id": task.authority_id,
                "domain": task.domain,
                "domain_task_id": task.domain_task_id,
                "language": task.language,
                "path": str(task.path),
                "bytes": task.expected_bytes,
            }
            for task in tasks
        ],
        "model": dict(config["model"]),
        "data": dict(config["data"]),
        "objective": dict(config["objective"]),
        "optimization": dict(config["optimization"]),
        "runtime": {
            "world_size": context.world_size,
            "topology": topology,
            "total_macros": total_macros,
            "checkpoint_macros": list(checkpoint_macros),
            "tasks_per_rank_per_macro": tasks_per_rank,
            "task_assignment": "dynamic_cost_balanced_long_first",
        },
        "trainable": {
            "model_parameters": sum(value.numel() for value in model.parameters()),
            "parameter_tensors": sum(1 for _ in model.parameters()),
            "source_policy_parameters": 0,
        },
    }


def tokenize_stage0_languages(
    tasks: tuple[ECPStage0Task, ...],
    *,
    tokenizer_path: Path,
    max_length: int,
    device: torch.device,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    tokenizer = Pi05TeacherPrefixTokenizer(
        tokenizer_path, max_length, str(device)
    )
    tokens, masks, _ = tokenizer([task.language for task in tasks])
    return {
        task.authority_id: (tokens[index : index + 1], masks[index : index + 1])
        for index, task in enumerate(tasks)
    }


def build_stage0_training_stores(
    tasks: tuple[ECPStage0Task, ...],
    *,
    config: dict[str, Any],
    source_config: dict[str, Any],
) -> tuple[RawTeacherVideoStore, PrivilegedMetaActionStore]:
    authorities = tuple(task.writer_authority() for task in tasks)
    video_store = RawTeacherVideoStore(
        authorities, frame_stride=int(config["data"]["frame_stride"])
    )
    stats = load_stats(source_config, source_config["data"]["active_task_ids"])
    action_store = PrivilegedMetaActionStore(
        authorities,
        action_q01=stats["action"]["q01"],
        action_q99=stats["action"]["q99"],
        phase_count=int(config["model"]["action_phases"]),
    )
    return video_store, action_store


def prepare_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> ECPStage0Runtime:
    config = load_stage0_config(args.config)
    total, stop, checkpoints, tasks_per_rank = _resolve_runtime(args, config, context)
    seed_everything(int(config["optimization"]["seed"]), context)
    tasks, schedule = build_stage0_tasks_and_schedule(config, args.data_root)
    source = stage0_source_authority(args)
    source_config = load_config(stage0_authority_path(config, "source_base_config"))
    policy = load_policy(
        Path(source["model_path"]), source_config, context.device
    ).requires_grad_(False).eval()
    owners = build_target_owners(
        load_pi05_lora_contract(stage0_authority_path(config, "lora_contract"))
    )
    model_cell = config["model"]
    model = ECPStage0Model(
        owners,
        prefix_width=int(model_cell["prefix_width"]),
        expert_width=int(model_cell["expert_width"]),
        program_width=int(model_cell["program_width"]),
        event_slots=int(model_cell["event_slots"]),
        action_phases=int(model_cell["action_phases"]),
        max_frames_per_call=int(model_cell["max_frames_per_call"]),
        fixed_probe_seed=int(model_cell["fixed_probe_seed"]),
    ).to(context.device)
    initialize_deferred_process_group(context, rendezvous_root=args.output_dir.parent)
    if context.world_size > 1:
        for value in model.state_dict().values():
            dist.broadcast(value, src=0)
    optimizer = build_stage0_optimizer(model.parameters(), config["optimization"])
    scheduler = build_stage0_scheduler(optimizer, config, total)
    video_store, action_store = build_stage0_training_stores(
        tasks, config=config, source_config=source_config
    )
    language = tokenize_stage0_languages(
        tasks,
        tokenizer_path=args.tokenizer_path,
        max_length=int(source_config["features"]["tokenizer_max_length"]),
        device=context.device,
    )
    contract = _build_contract(
        args=args,
        config=config,
        context=context,
        tasks=tasks,
        source=source,
        total_macros=total,
        checkpoint_macros=checkpoints,
        tasks_per_rank=tasks_per_rank,
        model=model,
    )
    publish_stage0_contract(args, context, contract)
    start_macro = 0
    expected_metrics = 0
    if args.resume is not None:
        start_macro, expected_metrics = load_ecp_checkpoint(
            checkpoint=args.resume,
            stage=STAGE,
            context=context,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            run_contract_schema=RUN_SCHEMA,
        )
    if not 0 <= start_macro < stop:
        raise ValueError("ECP Stage 0 resume cursor is outside this segment")
    metrics_rows = (
        reconcile_metrics(
            args.output_dir / "metrics.jsonl",
            start_macro,
            expected_metrics,
            cursor_key="macro",
        )
        if context.is_main
        else 0
    )
    model.train()
    torch.cuda.reset_peak_memory_stats(context.device)
    return ECPStage0Runtime(
        args=args,
        config=config,
        context=context,
        tasks=tasks,
        task_by_id={task.authority_id: task for task in tasks},
        schedule=schedule,
        video_store=video_store,
        action_store=action_store,
        language_tokens=language,
        policy=policy,
        model=model,
        action_meta_lora=None,
        trainable_parameters=tuple(model.parameters()),
        frozen_parameters=tuple(policy.parameters()),
        checkpoint_module=model,
        checkpoint_stage=STAGE,
        run_schema=RUN_SCHEMA,
        optimizer=optimizer,
        scheduler=scheduler,
        tasks_per_rank=tasks_per_rank,
        total_macros=total,
        stop_after_macro=stop,
        checkpoint_macros=checkpoints,
        start_macro=start_macro,
        metrics_rows=metrics_rows,
        run_contract=contract,
    )


def run_stage0_training(
    args: argparse.Namespace,
    prepare: Any,
) -> None:
    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    runtime: ECPStage0Runtime | None = None
    try:
        runtime = prepare(args, context)
        started = time.monotonic()
        for macro in range(runtime.start_macro, runtime.stop_after_macro):
            row = run_stage0_macro(runtime, macro, started)
            if context.is_main:
                append_jsonl(args.output_dir / "metrics.jsonl", row)
                runtime.metrics_rows += 1
                if (macro + 1) % args.log_every == 0:
                    print(json.dumps(row, sort_keys=True), flush=True)
            if macro + 1 in runtime.checkpoint_macros:
                save_ecp_checkpoint(
                    output_dir=args.output_dir,
                    macro=macro + 1,
                    stage=runtime.checkpoint_stage,
                    context=context,
                    model=runtime.checkpoint_module,
                    optimizer=runtime.optimizer,
                    scheduler=runtime.scheduler,
                    run_contract_schema=runtime.run_schema,
                    metrics_rows=runtime.metrics_rows,
                )
        if context.is_main:
            segment = {
                "stage": runtime.checkpoint_stage,
                "completed_macros": runtime.stop_after_macro,
                "total_macros": runtime.total_macros,
            }
            write_json_atomic(args.output_dir / "segment_completion.json", segment)
            if runtime.stop_after_macro == runtime.total_macros:
                write_json_atomic(args.output_dir / "completion.json", segment)
    finally:
        if runtime is not None:
            runtime.video_store.close()
            runtime.action_store.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def train(args: argparse.Namespace) -> None:
    run_stage0_training(args, prepare_runtime)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_stage0_native_v1.json",
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-macro", type=int)
    parser.add_argument("--log-every", type=int, default=1)
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
    if args.log_every <= 0:
        raise ValueError("ECP Stage 0 log interval must be positive")
    return args
