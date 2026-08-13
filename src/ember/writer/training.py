"""Canonical dynamic-K Backbone-Memory AS-Writer training entrypoint."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import DistributedContext, barrier
from ember.pi05_source_contract import append_jsonl, reconcile_metrics
from ember.pi05_source_setup import (
    initialize_deferred_process_group,
    initialize_distributed,
    load_policy,
    load_stats,
    seed_everything,
)
from ember.writer.as_config import (
    REPO_ROOT,
    authority_path,
    load_writer_config,
    resolve_mode_config,
)
from ember.writer.as_contract import (
    build_contract,
    inspect_video_data,
    load_run_authorities,
    load_training_data,
    publish_contract,
    resolve_runtime,
    writer_trainable_contract,
)
from ember.writer.as_sampling import MixedTaskBatchSampler, TeacherVideoSchedule
from ember.writer.as_step import ParameterSlice, parameter_layout, run_writer_step
from ember.writer.checkpoint import (
    checkpoint_macro,
    load_writer_checkpoint,
    save_writer_checkpoint,
)
from ember.writer.data import (
    FunctionalQueryDataset,
    RawTeacherVideoStore,
    WriterTaskAuthority,
)
from ember.writer.errors import WriterModelError
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.update_schedule import build_exposure_scheduler


@dataclass
class WriterRuntime:
    args: argparse.Namespace
    context: DistributedContext
    config: dict[str, Any]
    dataset: FunctionalQueryDataset
    task_authorities: tuple[WriterTaskAuthority, ...]
    task_ids: tuple[int, ...]
    sampler: MixedTaskBatchSampler
    video_schedule: TeacherVideoSchedule
    iterator: Iterator[dict[str, Any]]
    video_store: RawTeacherVideoStore
    language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    processor: Pi05LiberoProcessor
    policy: torch.nn.Module
    writer: torch.nn.Module
    trainable_parameters: tuple[torch.nn.Parameter, ...]
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    lora_contract: Any
    contract: dict[str, Any]
    total_macros: int
    checkpoint_macros: tuple[int, ...]
    resume_macro: int
    metrics_path: Path
    metrics_rows: int
    gradient_layout: tuple[ParameterSlice, ...]


def build_writer(
    config: Mapping[str, Any], policy: torch.nn.Module
) -> tuple[torch.nn.Module, Any]:
    """Construct through the dynamic-K model's stable policy-owned factory."""

    from ember.writer.model import CompleteLoRAWriter

    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    template = prepare_frozen_writer_policy(policy, lora)
    factory = getattr(CompleteLoRAWriter, "from_policy", None)
    if not callable(factory):
        raise WriterModelError(
            "dynamic-K model integration requires CompleteLoRAWriter.from_policy"
        )
    writer = factory(
        policy=policy,
        template_state=template,
        writer_config=config["writer"],
    )
    if not callable(getattr(writer, "forward_training", None)):
        raise WriterModelError(
            "dynamic-K model integration requires CompleteLoRAWriter.forward_training"
        )
    return writer, lora


def _synchronize_writer(writer: torch.nn.Module, context: DistributedContext) -> None:
    if context.world_size <= 1:
        return
    for value in writer.state_dict().values():
        dist.broadcast(value, src=0)


def _build_sampler(
    *,
    dataset: FunctionalQueryDataset,
    task_ids: tuple[int, ...],
    config: Mapping[str, Any],
    video_data: Mapping[str, Any],
    context: DistributedContext,
    start_macro: int,
    stop_macro: int,
    batch_size: int,
    num_workers: int,
) -> tuple[MixedTaskBatchSampler, TeacherVideoSchedule, DataLoader[Any]]:
    first, last = map(int, config["data"]["demo_indices"])
    schedule = TeacherVideoSchedule(
        task_ids=task_ids,
        demo_indices=range(first, last + 1),
        seed=int(config["data"]["teacher_video_seed"]),
        videos_per_visit=int(config["data"]["dynamic_k_max"]),
        dynamic_k_max=int(config["data"]["dynamic_k_max"]),
    )
    sampler = MixedTaskBatchSampler(
        dataset,
        task_ids=task_ids,
        per_rank_batch_size=batch_size,
        start_step=start_macro,
        stop_step=stop_macro,
        rank=context.rank,
        world_size=context.world_size,
        seed=int(config["data"]["sampler_seed"]),
        tasks_per_rank_per_update=(len(task_ids) + context.world_size - 1)
        // context.world_size,
        video_schedule=schedule,
        task_video_costs={
            int(task_id): {
                int(demo): int(cost)
                for demo, cost in per_demo.items()
            }
            for task_id, per_demo in video_data["sampled_frame_counts_by_task"].items()
        },
        condition_frame_budget=int(
            config["writer"]["backbone_total_frames_per_condition"]
        ),
        assignment_strategy="cost_balanced_long_first_dynamic_uneven",
    )
    workers = num_workers
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=int(config["loader"]["prefetch_factor"])
        if workers
        else None,
        generator=torch.Generator().manual_seed(
            int(config["optimization"]["seed"]) + context.rank + 0xA55A
        ),
    )
    return sampler, schedule, loader


def _condition_inputs(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    source_config: dict[str, Any],
    context: DistributedContext,
    tasks: tuple[WriterTaskAuthority, ...],
) -> tuple[
    RawTeacherVideoStore,
    Pi05LiberoProcessor,
    dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    store = RawTeacherVideoStore(
        tasks,
        frame_stride=int(config["writer"]["frame_stride"]),
        max_open_files=int(config["data"]["video_open_files_per_rank"]),
    )
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
    language = {task.task_id: tokenizer([task.language]) for task in tasks}
    return store, processor, language


def _build_optimizer(
    writer: torch.nn.Module,
    config: Mapping[str, Any],
    total_macros: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer_config = config["optimization"]["optimizer"]
    optimizer = torch.optim.AdamW(
        (value for value in writer.parameters() if value.requires_grad),
        lr=float(config["optimization"]["scheduler"]["peak_lr"]),
        betas=tuple(optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler = build_exposure_scheduler(
        optimizer, config["optimization"]["scheduler"], total_macros
    )
    return optimizer, scheduler


def _resume_if_requested(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    writer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    contract: Mapping[str, Any],
    initial_macro: int,
) -> int:
    if args.resume is None:
        return 0
    loaded, metrics_rows = load_writer_checkpoint(
        checkpoint=args.resume,
        context=context,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        contract=contract,
    )
    if loaded != initial_macro:
        raise WriterModelError("dynamic-K resume cursor disagrees with its state")
    return metrics_rows


def prepare_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> WriterRuntime:
    config = resolve_mode_config(load_writer_config(args.config), args.mode)
    total, batch_size, checkpoint_macros, stop_macro = resolve_runtime(
        args, config, context
    )
    args.stop_after_macro = stop_macro
    initial_macro = checkpoint_macro(args.resume)
    if args.resume is not None and args.resume.parent.parent != args.output_dir:
        raise WriterModelError("dynamic-K resume checkpoint belongs to another run")
    if not 0 <= initial_macro < stop_macro:
        raise WriterModelError("dynamic-K resume cursor is outside this segment")
    seed_everything(int(config["optimization"]["seed"]), context)
    dataset, tasks, data_validation = load_training_data(args, config)
    task_ids = tuple(task.task_id for task in tasks)
    authorities, source, tokenizer = load_run_authorities(args, config)
    video_data = inspect_video_data(args.data_root, config, task_ids)
    policy = load_policy(
        Path(source["model_path"]),
        authorities.source_base_config,
        context.device,
    )
    writer, lora = build_writer(config, policy)
    writer.to(context.device)
    trainable = writer_trainable_contract(writer, policy, lora)
    optimizer, scheduler = _build_optimizer(writer, config, total)
    initialize_deferred_process_group(context, rendezvous_root=args.output_dir.parent)
    contract = build_contract(
        args=args,
        config=config,
        context=context,
        source=source,
        tokenizer=tokenizer,
        video_data=video_data,
        data_validation=data_validation,
        task_ids=task_ids,
        trainable=trainable,
        total_macros=total,
        batch_size=batch_size,
        checkpoint_macros=checkpoint_macros,
    )
    publish_contract(args, context, contract)
    _synchronize_writer(writer, context)
    expected_metrics_rows = _resume_if_requested(
        args=args,
        context=context,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        contract=contract,
        initial_macro=initial_macro,
    )
    sampler, schedule, loader = _build_sampler(
        dataset=dataset,
        task_ids=task_ids,
        config=config,
        video_data=video_data,
        context=context,
        start_macro=initial_macro,
        stop_macro=stop_macro,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )
    store, processor, language = _condition_inputs(
        args=args,
        config=config,
        source_config=authorities.source_base_config,
        context=context,
        tasks=tasks,
    )
    metrics_path = args.output_dir / "metrics.jsonl"
    metrics_rows = (
        reconcile_metrics(
            metrics_path,
            initial_macro,
            expected_metrics_rows,
            cursor_key="macro",
        )
        if context.is_main
        else 0
    )
    rows = torch.tensor(metrics_rows, dtype=torch.long, device=context.device)
    if context.world_size > 1:
        dist.broadcast(rows, src=0)
    writer.train()
    torch.cuda.reset_peak_memory_stats(context.device)
    return WriterRuntime(
        args=args,
        context=context,
        config=config,
        dataset=dataset,
        task_authorities=tasks,
        task_ids=task_ids,
        sampler=sampler,
        video_schedule=schedule,
        iterator=iter(loader),
        video_store=store,
        language_tokens=language,
        processor=processor,
        policy=policy,
        writer=writer,
        trainable_parameters=tuple(
            value for value in writer.parameters() if value.requires_grad
        ),
        optimizer=optimizer,
        scheduler=scheduler,
        lora_contract=lora,
        contract=contract,
        total_macros=total,
        checkpoint_macros=checkpoint_macros,
        resume_macro=initial_macro,
        metrics_path=metrics_path,
        metrics_rows=int(rows.item()),
        gradient_layout=parameter_layout(writer),
    )


def run_macros(runtime: WriterRuntime) -> None:
    started = time.monotonic()
    for macro in range(runtime.resume_macro, runtime.args.stop_after_macro):
        row = run_writer_step(runtime, macro, started)
        completed = int(row["macro"])
        if runtime.context.is_main:
            append_jsonl(runtime.metrics_path, row)
            runtime.metrics_rows += 1
            if completed == 1 or completed % runtime.args.log_every == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
        if completed in runtime.checkpoint_macros:
            save_writer_checkpoint(
                output_dir=runtime.args.output_dir,
                macro=completed,
                context=runtime.context,
                writer=runtime.writer,
                optimizer=runtime.optimizer,
                scheduler=runtime.scheduler,
                contract=runtime.contract,
                metrics_rows=runtime.metrics_rows,
            )
    barrier(runtime.context)
    if runtime.context.is_main:
        from ember.pi05_source_checkpoint import write_json_atomic

        write_json_atomic(
            runtime.args.output_dir / "completion.json",
            {
                "schema_version": "ember_pi05_dynamic_k_writer_completion_v1",
                "completed_macro": runtime.args.stop_after_macro,
                "elapsed_seconds": time.monotonic() - started,
            },
        )


def train(args: argparse.Namespace) -> None:
    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    runtime: WriterRuntime | None = None
    try:
        runtime = prepare_runtime(args, context)
        if context.is_main:
            print(
                json.dumps(
                    {
                        "event": "start",
                        "mode": args.mode,
                        "resume_macro": runtime.resume_macro,
                        "stop_after_macro": args.stop_after_macro,
                        "world_size": context.world_size,
                        "tasks": len(runtime.task_ids),
                        "trainable": runtime.contract["trainable"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        run_macros(runtime)
    finally:
        if runtime is not None:
            runtime.dataset.close()
            runtime.video_store.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/pi05_as_writer_dynamic_k_semantic_address_rank8_v1.json",
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--total-macros", type=int)
    parser.add_argument("--stop-after-macro", type=int)
    parser.add_argument("--checkpoint-macros", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    config = load_writer_config(args.config.resolve())
    if args.num_workers is None:
        args.num_workers = int(config["loader"]["num_workers_per_rank"])
    if args.num_workers < 0 or args.log_every <= 0:
        raise WriterModelError("invalid dynamic-K loader or logging request")
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
    return args
