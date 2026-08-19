"""Train language prior and ordered-video posterior against fixed meta-task codes."""

from __future__ import annotations

import argparse
import math
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist

from ember.expert_manifold.contract import (
    authority_path as expert_authority_path,
    load_task_expert_config,
    load_train_tasks,
)
from ember.functional_adaptation.action_alignment import PrivilegedMetaActionStore
from ember.functional_adaptation.code_checkpoint import (
    RUN_SCHEMA,
    load_code_writer_checkpoint,
)
from ember.functional_adaptation.code_schedule import MetaCodeTrainingSchedule
from ember.functional_adaptation.code_writer import FunctionalCodeWriter
from ember.functional_adaptation.decoder_training import (
    authority_path,
    load_functional_adapter_config,
    load_meta_decoder_code_targets,
)
from ember.pi05_eval_contract import (
    git_state,
    inspect_source_checkpoint,
    load_evaluation_authorities,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import (
    DistributedContext,
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
from ember.writer.data import (
    RawTeacherVideoStore,
    WriterTaskAuthority,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class CodeTrainingRuntime:
    args: argparse.Namespace
    context: DistributedContext
    config: dict[str, Any]
    settings: Mapping[str, Any]
    schedule: MetaCodeTrainingSchedule
    tasks: Mapping[int, WriterTaskAuthority]
    video_store: RawTeacherVideoStore
    action_store: PrivilegedMetaActionStore
    language: Mapping[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    target_codes: Mapping[int, torch.Tensor]
    policy: torch.nn.Module
    writer: FunctionalCodeWriter
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    trainable: tuple[torch.nn.Parameter, ...]
    checkpoint_macros: tuple[int, ...]
    start_macro: int
    stop_macro: int
    metrics_rows: int
    metrics_path: Path


@dataclass(frozen=True)
class CodeTrainingSetup:
    context: DistributedContext
    config: dict[str, Any]
    settings: Mapping[str, Any]
    total_macros: int
    checkpoint_macros: tuple[int, ...]
    start_macro: int
    stop_macro: int


def _sampled_frame_count(raw_count: int, stride: int) -> int:
    count = (raw_count - 1) // stride + 1
    return count + int((raw_count - 1) % stride != 0)


def _frame_costs(
    expert_config: Mapping[str, Any],
    task_ids: tuple[int, ...],
    demos: tuple[int, ...],
    stride: int,
) -> dict[int, dict[int, int]]:
    manifest = read_json(expert_authority_path(expert_config, "source_manifest"))
    rows = {int(row["task_index"]): row for row in manifest["tasks"]}
    return {
        task_id: {
            demo: _sampled_frame_count(
                int(rows[task_id]["demonstrations"]["episode_lengths"][demo]), stride
            )
            for demo in demos
        }
        for task_id in task_ids
    }


def _optimizer(
    writer: torch.nn.Module,
    settings: Mapping[str, Any],
    total_macros: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    values = settings["optimizer"]
    learning_rate = float(values["learning_rate"])
    optimizer = torch.optim.AdamW(
        (parameter for parameter in writer.parameters() if parameter.requires_grad),
        lr=learning_rate,
        betas=tuple(float(value) for value in values["betas"]),
        eps=float(values["eps"]),
        weight_decay=float(values["weight_decay"]),
    )

    def factor(step: int) -> float:
        warmup = min(2, max(1, total_macros // 10))
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_macros - warmup))
        return 0.01 + 0.495 * (1.0 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _run_contract(
    runtime: CodeTrainingRuntime,
    *,
    source: Mapping[str, Any],
    task_ids: tuple[int, ...],
) -> dict[str, Any]:
    state = git_state(REPO_ROOT)
    return {
        "schema_version": RUN_SCHEMA,
        "mode": runtime.args.mode,
        "method": "fixed_decoder_language_prior_ordered_video_posterior",
        "repository": {key: state[key] for key in ("branch", "commit")},
        "source": {
            "run": str(runtime.args.source_run),
            "checkpoint": str(runtime.args.checkpoint),
            "model_path": str(source["model_path"]),
        },
        "decoder_profile": str(runtime.args.decoder_profile_root),
        "config": str(runtime.args.config),
        "data_root": str(runtime.args.data_root),
        "tokenizer": str(runtime.args.tokenizer_path),
        "tasks": {
            "role": "meta_train",
            "count": len(task_ids),
            "global_task_ids": list(task_ids),
            "equal_weight_per_macro": True,
        },
        "runtime": {
            "host": socket.gethostname(),
            "world_size": runtime.context.world_size,
            "rank": runtime.context.rank,
            "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
            "total_macros": int(runtime.settings[runtime.args.mode]["total_macros"]),
        },
        "trainable": {
            "writer_parameter_count": sum(
                int(parameter.numel()) for parameter in runtime.trainable
            ),
            "fixed_decoder_parameter_count": sum(
                int(parameter.numel())
                for parameter in runtime.writer.fixed_decoder.parameters()
            ),
            "fixed_decoder_trainable_parameters": 0,
        },
        "deployment": {
            "inputs": ["exact language", "action-hidden ordered teacher videos"],
            "writer_runs_once_before_rollout": True,
            "output": "one complete 38-target rank16 LoRA",
            "task_id_route": False,
            "decoder_frozen": True,
        },
        "training_privileged": {
            "meta_train_teacher_actions": "phase-alignment auxiliary only",
            "video_action_episode_pairing": "same_task_disjoint_episode",
            "meta_validation_actions": 0,
            "target40_actions_or_rewards": 0,
            "available_at_deployment": False,
        },
        "content_hash_policy": "disabled_by_owner",
    }


def _publish_contract(
    runtime: CodeTrainingRuntime,
    contract: Mapping[str, Any],
) -> None:
    path = runtime.args.output_dir / "run_contract.json"
    if runtime.context.is_main:
        if runtime.args.resume is None:
            if runtime.args.output_dir.exists() and any(
                runtime.args.output_dir.iterdir()
            ):
                raise ValueError("fresh functional-code output directory is not empty")
            runtime.args.output_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(path, dict(contract))
        elif not path.is_file() or read_json(path) != dict(contract):
            raise ValueError("functional-code resume contract changed")
        append_jsonl(
            runtime.args.output_dir / "invocations.jsonl",
            {
                "argv": sys.argv,
                "started_unix": time.time(),
                "resume": str(runtime.args.resume) if runtime.args.resume else None,
            },
        )
    if runtime.context.world_size > 1:
        dist.barrier(device_ids=[runtime.context.local_rank])


def _synchronize_writer(runtime: CodeTrainingRuntime) -> None:
    if runtime.context.world_size <= 1:
        return
    for value in runtime.writer.state_dict().values():
        dist.broadcast(value, src=0)


def _resume(runtime: CodeTrainingRuntime) -> None:
    if runtime.args.resume is None:
        _synchronize_writer(runtime)
        return
    macro, metrics_rows = load_code_writer_checkpoint(
        checkpoint=runtime.args.resume,
        rank=runtime.context.rank,
        world_size=runtime.context.world_size,
        writer=runtime.writer,
        optimizer=runtime.optimizer,
        scheduler=runtime.scheduler,
    )
    if macro != runtime.start_macro or metrics_rows != macro:
        raise ValueError("functional-code resume cursor changed")
    runtime.metrics_rows = metrics_rows


def _resolve_training_setup(args: argparse.Namespace) -> CodeTrainingSetup:
    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    config = load_functional_adapter_config(args.config, REPO_ROOT)
    settings = config["code_inference"]["training"]
    mode = settings[args.mode]
    if args.mode == "formal" and context.world_size != int(mode["world_size"]):
        raise ValueError("formal functional-code training requires six workers")
    state = git_state(REPO_ROOT)
    if args.mode == "formal" and (
        state["dirty_paths"]
        or (args.resume is None and state["commit"] != state["upstream_commit"])
    ):
        raise ValueError(
            "formal functional-code training requires a clean pushed commit"
        )
    total_macros = int(mode["total_macros"])
    checkpoint_macros = tuple(int(value) for value in mode["checkpoint_macros"])
    stop_macro = int(
        args.stop_after_macro or mode.get("default_stop_macro", total_macros)
    )
    if stop_macro not in checkpoint_macros:
        raise ValueError("functional-code stop must be a declared checkpoint")
    start_macro = int(args.resume.name.removeprefix("macro_")) if args.resume else 0
    seed_everything(int(config["code_inference"]["initialization_seed"]), context)
    return CodeTrainingSetup(
        context=context,
        config=config,
        settings=settings,
        total_macros=total_macros,
        checkpoint_macros=checkpoint_macros,
        start_macro=start_macro,
        stop_macro=stop_macro,
    )


def prepare_runtime(args: argparse.Namespace) -> CodeTrainingRuntime:
    setup = _resolve_training_setup(args)
    context = setup.context
    config = setup.config
    settings = setup.settings
    mode = settings[args.mode]

    expert_config = load_task_expert_config(
        authority_path(config, "meta_experts", REPO_ROOT)
    )
    all_tasks = tuple(
        task
        for task in load_train_tasks(expert_config, args.data_root)
        if task.split_role == "meta_train"
    )
    active_count = int(mode["task_count"])
    active_tasks = all_tasks[:active_count]
    task_ids = tuple(task.global_task_id for task in active_tasks)
    targets = load_meta_decoder_code_targets(
        args.decoder_profile_root, device=context.device
    )
    if any(task_id not in targets.train_codes for task_id in task_ids):
        raise ValueError("functional-code targets do not cover the training tasks")
    authorities = load_evaluation_authorities(
        authority_path(config, "evaluation_config", REPO_ROOT), REPO_ROOT
    )
    source = inspect_source_checkpoint(
        authorities,
        args.source_run,
        args.checkpoint,
        evaluation_mode="formal",
    )
    policy = load_policy(
        Path(str(source["model_path"])), authorities.source_base_config, context.device
    )
    policy.requires_grad_(False)
    policy.eval()
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract", REPO_ROOT))
    writer = FunctionalCodeWriter.from_policy(
        policy=policy,
        config=config,
        contract=lora,
        decoder_checkpoint=targets.decoder_checkpoint,
        device=context.device,
    )
    optimizer, scheduler = _optimizer(writer, settings, setup.total_macros)
    trainable = tuple(
        parameter for parameter in writer.parameters() if parameter.requires_grad
    )
    initialize_deferred_process_group(context, rendezvous_root=args.output_dir.parent)
    first, last = map(int, settings["train_demo_indices"])
    demos = tuple(range(first, last + 1))
    stride = int(settings["frame_stride"])
    costs = _frame_costs(expert_config, task_ids, demos, stride)
    schedule = MetaCodeTrainingSchedule(
        task_ids=task_ids,
        demo_indices=demos,
        sampled_frame_counts=costs,
        world_size=context.world_size,
        seed=int(settings["task_macro_seed"]),
        dynamic_k_max=int(settings["dynamic_k_max"]),
        temporal_controls=settings["temporal_controls"],
    )
    task_authorities = {task.global_task_id: task.authority for task in active_tasks}
    video_store = RawTeacherVideoStore(
        tuple(task_authorities.values()),
        frame_stride=stride,
        max_open_files=int(settings["video_open_files_per_rank"]),
    )
    stats = load_stats(
        authorities.source_base_config,
        authorities.source_base_config["data"]["active_task_ids"],
    )
    action_store = PrivilegedMetaActionStore(
        tuple(task_authorities.values()),
        action_q01=stats["action"]["q01"],
        action_q99=stats["action"]["q99"],
        phase_count=int(config["code_inference"]["phase_queries"]),
        max_open_files=int(settings["video_open_files_per_rank"]),
    )
    tokenizer = Pi05TeacherPrefixTokenizer(
        args.tokenizer_path,
        int(authorities.source_base_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    language = {
        task.global_task_id: tokenizer([task.language]) for task in active_tasks
    }
    runtime = CodeTrainingRuntime(
        args=args,
        context=context,
        config=config,
        settings=settings,
        schedule=schedule,
        tasks=task_authorities,
        video_store=video_store,
        action_store=action_store,
        language=language,
        target_codes={task_id: targets.train_codes[task_id] for task_id in task_ids},
        policy=policy,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        trainable=trainable,
        checkpoint_macros=setup.checkpoint_macros,
        start_macro=setup.start_macro,
        stop_macro=setup.stop_macro,
        metrics_rows=setup.start_macro,
        metrics_path=args.output_dir / "metrics.jsonl",
    )
    contract = _run_contract(runtime, source=source, task_ids=task_ids)
    _publish_contract(runtime, contract)
    _resume(runtime)
    writer.train()
    writer.fixed_decoder.eval()
    torch.cuda.reset_peak_memory_stats(context.device)
    return runtime


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_functional_adapter_v1.json",
    )
    result.add_argument("--mode", choices=("smoke", "profile", "formal"), required=True)
    result.add_argument("--source-run", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--decoder-profile-root", type=Path, required=True)
    result.add_argument("--tokenizer-path", type=Path, required=True)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--stop-after-macro", type=int)
    result.add_argument("--resume", type=Path)
    return result


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
        "source_run",
        "checkpoint",
        "decoder_profile_root",
        "tokenizer_path",
        "data_root",
    ):
        value = getattr(args, name).resolve()
        if not value.exists():
            raise ValueError(f"missing functional-code path: {value}")
        setattr(args, name, value)
    args.output_dir = args.output_dir.resolve()
    args.resume = args.resume.resolve() if args.resume else None
    return args
