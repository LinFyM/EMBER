"""Canonical symmetric-rank PI05 Action-Supervised Writer training.

Only the shared Writer is trainable.  It receives pure task language plus one
action-hidden teacher video and generates the complete sealed PI05 task LoRA;
an independently sampled action query from the same development-train task is
used only by the frozen policy's functional behavior loss.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.distributed as dist
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ember.pi05_eval_contract import (
    inspect_source_checkpoint,
    inspect_tokenizer,
    load_evaluation_authorities,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05ForecastPrefixTokenizer, Pi05LiberoProcessor
from ember.pi05_source_checkpoint import (
    DistributedContext,
    barrier,
    canonical_hash,
    restore_rng,
    write_json_atomic,
)
from ember.pi05_source_contract import append_jsonl, reconcile_metrics
from ember.pi05_source_setup import (
    initialize_distributed,
    load_policy,
    load_stats,
    seed_everything,
)
from ember.writer.as_step import run_writer_step
from ember.writer.checkpoint import (
    initialize_writer_phase,
    load_writer_checkpoint,
    save_writer_checkpoint,
)
from ember.writer.as_contract import (
    REPO_ROOT,
    authority_path,
    build_contract,
    inspect_video_data,
    load_training_data,
    load_writer_config,
    publish_contract,
    reconcile_resume_contract,
    resume_step,
    resolve_runtime,
    writer_trainable_contract,
    writer_stage,
)
from ember.writer.data import (
    FunctionalQueryDataset,
    MixedTaskBatchSampler,
    RawTeacherVideoStore,
    TeacherVideoSchedule,
    WriterFlowNoiseSchedule,
    WriterTaskAuthority,
)
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.model import (
    ACTION_FORECAST_WRITER_CONSTRUCTOR_KEYS,
    CompleteLoRAWriter,
    WriterModelError,
    build_lora_tensor_specs,
)
from ember.writer.online_validation import (
    OnlineWriterValidation,
    evaluate_online_writer_checkpoint,
    prepare_online_writer_validation,
)


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
    flow_noise_schedule: WriterFlowNoiseSchedule
    iterator: Iterator[dict[str, Any]]
    video_store: RawTeacherVideoStore
    language_tokens: dict[
        int,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]
    processor: Pi05LiberoProcessor
    policy: torch.nn.Module
    identity_state: dict[str, torch.Tensor]
    writer: CompleteLoRAWriter
    wrapped_writer: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    lora_contract: Any
    contract: dict[str, Any]
    contract_sha256: str
    total_steps: int
    batch_size: int
    conditions_per_optimizer_step: int
    checkpoint_steps: tuple[int, ...]
    resume_step: int
    metrics_path: Path
    metrics_rows: int
    checkpoint_validation: OnlineWriterValidation | None


@dataclass
class WriterSetup:
    dataset: FunctionalQueryDataset
    tasks: tuple[WriterTaskAuthority, ...]
    task_ids: tuple[int, ...]
    authorities: Any
    policy: torch.nn.Module
    writer: CompleteLoRAWriter
    lora_contract: Any
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    trainable: dict[str, Any]
    identity_state: dict[str, torch.Tensor]
    contract: dict[str, Any]
    contract_sha256: str


def _build_writer(
    config: Mapping[str, Any], policy: torch.nn.Module
) -> tuple[
    CompleteLoRAWriter,
    Any,
    dict[str, Any],
    dict[str, torch.Tensor],
]:
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    if hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
    if hasattr(policy, "config"):
        policy.config.gradient_checkpointing = False
    template = prepare_frozen_writer_policy(policy, lora)
    writer_config = {
        key: value
        for key, value in config["writer"].items()
        if key in ACTION_FORECAST_WRITER_CONSTRUCTOR_KEYS
    }
    bridge = policy.model.paligemma_with_expert
    writer = CompleteLoRAWriter(
        build_lora_tensor_specs(template),
        template_state=template,
        paligemma_model=bridge.paligemma.model.language_model,
        expert_model=bridge.gemma_expert.model,
        **writer_config,
    )
    return (
        writer,
        lora,
        writer_trainable_contract(writer, policy, lora),
        {name: value.detach().clone() for name, value in template.items()},
    )


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    return CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=int(config["warmup_steps"]),
        num_decay_steps=int(config["decay_steps"]),
        peak_lr=float(config["peak_lr"]),
        decay_lr=float(config["decay_lr"]),
    ).build(optimizer, total_steps)


def _build_trainable_models(
    *,
    config: Mapping[str, Any],
    context: DistributedContext,
    source: Mapping[str, Any],
    source_config: Mapping[str, Any],
    total_steps: int,
) -> tuple[
    torch.nn.Module,
    CompleteLoRAWriter,
    Any,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    dict[str, Any],
    dict[str, torch.Tensor],
]:
    policy = load_policy(Path(source["model_path"]), source_config, context.device)
    writer, lora, trainable, identity = _build_writer(config, policy)
    writer.to(context.device)
    optimizer_config = config["optimization"]["optimizer"]
    optimizer = torch.optim.AdamW(
        writer.parameters(),
        lr=float(config["optimization"]["scheduler"]["peak_lr"]),
        betas=tuple(optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler = _make_scheduler(
        optimizer, config["optimization"]["scheduler"], total_steps
    )
    return policy, writer, lora, optimizer, scheduler, trainable, identity


def _restore_training_state(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    config: Mapping[str, Any],
    writer: CompleteLoRAWriter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    batch_size: int,
    batch_cycle: tuple[int, ...],
    conditions_per_optimizer_step: int,
    contract_sha256: str,
    initial_step: int,
) -> tuple[dict[str, Any] | None, int]:
    if args.resume is None:
        return None, 0
    loaded, rng, metrics_rows = load_writer_checkpoint(
        checkpoint=args.resume.resolve(),
        context=context,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler_seed=int(config["data"]["sampler_seed"]),
        teacher_video_seed=int(config["data"]["teacher_video_seed"]),
        writer_flow_noise_seed=int(config["data"]["writer_flow_noise_seed"]),
        per_rank_batch_size=batch_size,
        per_rank_batch_cycle=batch_cycle,
        conditions_per_optimizer_step=conditions_per_optimizer_step,
        contract_sha256=contract_sha256,
    )
    if loaded != initial_step:
        raise WriterModelError("AS-Writer resume path and state disagree")
    return rng, metrics_rows


def _build_sampler_and_loader(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    config: Mapping[str, Any],
    dataset: FunctionalQueryDataset,
    task_ids: tuple[int, ...],
    batch_size: int,
    batch_cycle: tuple[int, ...],
    conditions_per_optimizer_step: int,
    initial_step: int,
) -> tuple[MixedTaskBatchSampler, TeacherVideoSchedule, DataLoader[Any]]:
    if conditions_per_optimizer_step != 1:
        raise WriterModelError("Action-Forecast AS uses one video condition per rank")
    initial_data_step = initial_step
    stop_data_step = args.stop_after_step
    sampler = MixedTaskBatchSampler(
        dataset,
        task_ids=task_ids,
        per_rank_batch_size=batch_size,
        per_rank_batch_cycle=batch_cycle,
        start_step=initial_data_step,
        stop_step=stop_data_step,
        rank=context.rank,
        world_size=context.world_size,
        seed=int(config["data"]["sampler_seed"]),
    )
    first_demo, last_demo = map(int, config["data"]["demo_indices"])
    schedule = TeacherVideoSchedule(
        task_ids=task_ids,
        demo_indices=range(first_demo, last_demo + 1),
        seed=int(config["data"]["teacher_video_seed"]),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=int(config["loader"]["prefetch_factor"]) if args.num_workers else None,
        generator=torch.Generator().manual_seed(
            int(config["optimization"]["seed"]) + context.rank + 0xA55A
        ),
    )
    return sampler, schedule, loader


def _wrap_writer(writer: CompleteLoRAWriter, context: DistributedContext) -> torch.nn.Module:
    if context.world_size == 1:
        return writer
    return DistributedDataParallel(
        writer,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
        static_graph=False,
    )


def _metrics_cursor(
    path: Path,
    *,
    context: DistributedContext,
    initial_step: int,
    expected_rows: int,
) -> int:
    count = reconcile_metrics(path, initial_step, expected_rows) if context.is_main else 0
    rows = torch.tensor(count, dtype=torch.int64, device=context.device)
    if context.world_size > 1:
        dist.broadcast(rows, src=0)
    return int(rows.item())


def _build_condition_inputs(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    authorities: Any,
    context: DistributedContext,
    task_ids: tuple[int, ...],
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
    processor = Pi05LiberoProcessor(
        load_stats(
            authorities.source_base_config,
            authorities.source_base_config["data"]["active_task_ids"],
        ),
        args.tokenizer_path,
        int(authorities.source_base_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    tokenizer = Pi05ForecastPrefixTokenizer(
        args.tokenizer_path,
        int(authorities.source_base_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    language_tokens = {
        task.task_id: tokenizer([task.language]) for task in tasks
    }
    if set(language_tokens) != set(task_ids):
        raise WriterModelError("Writer language authorities changed")
    return store, processor, language_tokens


def _load_run_authorities(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    task_ids: tuple[int, ...],
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    authorities = load_evaluation_authorities(
        authority_path(config, "evaluation_config"),
        REPO_ROOT,
    )
    source = inspect_source_checkpoint(
        authorities,
        args.source_run,
        args.checkpoint,
        evaluation_mode="formal",
    )
    tokenizer = inspect_tokenizer(authorities, args.tokenizer_path)
    # load_training_data already performs and broadcasts the full rank-0 SHA
    # validation. This second inspection only records the raw-video identity.
    video_data = inspect_video_data(
        args.data_root.resolve(),
        config,
        task_ids,
        verify_hashes=False,
    )
    return authorities, source, tokenizer, video_data


def _prepare_setup(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    config: Mapping[str, Any],
    total_steps: int,
    batch_size: int,
    batch_cycle: tuple[int, ...],
    checkpoint_steps: tuple[int, ...],
) -> WriterSetup:
    dataset, tasks, data_validation = load_training_data(args, config, context)
    task_ids = tuple(task.task_id for task in tasks)
    authorities, source, tokenizer, video_data = _load_run_authorities(
        args,
        config,
        task_ids,
    )
    (
        policy,
        writer,
        lora_contract,
        optimizer,
        scheduler,
        trainable,
        identity_state,
    ) = _build_trainable_models(
        config=config,
        context=context,
        source=source,
        source_config=authorities.source_base_config,
        total_steps=total_steps,
    )
    initialization = initialize_writer_phase(
        args.initialize_writer_checkpoint, context, writer_stage(config), source,
        config["authorities"], config["writer"], writer,
        str(trainable["lora_contract_sha256"]),
    )
    candidate = build_contract(
        args=args,
        config=config,
        context=context,
        source=source,
        tokenizer=tokenizer,
        video_data=video_data,
        data_validation=data_validation,
        task_ids=task_ids,
        trainable=trainable,
        total_steps=total_steps,
        batch_size=batch_size,
        batch_cycle=batch_cycle,
        checkpoint_steps=checkpoint_steps,
        initialization=initialization,
    )
    contract = reconcile_resume_contract(args, candidate)
    contract_sha256 = canonical_hash(contract)
    publish_contract(args, context, contract, contract_sha256)
    return WriterSetup(
        dataset=dataset,
        tasks=tasks,
        task_ids=task_ids,
        authorities=authorities,
        policy=policy,
        writer=writer,
        lora_contract=lora_contract,
        optimizer=optimizer,
        scheduler=scheduler,
        trainable=trainable,
        identity_state=identity_state,
        contract=contract,
        contract_sha256=contract_sha256,
    )


def prepare_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> WriterRuntime:
    config = load_writer_config(args.config.resolve())
    total_steps, batch_size, checkpoint_steps = resolve_runtime(args, config, context)
    batch_cycle = (batch_size,)
    conditions_per_optimizer_step = int(
        config["conditioning_training"]["independent_conditions_per_optimizer_step"]
    )
    initial_step = resume_step(args.resume)
    if not 0 <= initial_step < args.stop_after_step:
        raise WriterModelError("AS-Writer resume cursor is outside this segment")
    seed_everything(int(config["optimization"]["seed"]), context)

    setup = _prepare_setup(
        args=args,
        context=context,
        config=config,
        total_steps=total_steps,
        batch_size=batch_size,
        batch_cycle=batch_cycle,
        checkpoint_steps=checkpoint_steps,
    )
    resume_rng, expected_metrics_rows = _restore_training_state(
        args=args,
        context=context,
        config=config,
        writer=setup.writer,
        optimizer=setup.optimizer,
        scheduler=setup.scheduler,
        batch_size=batch_size,
        batch_cycle=batch_cycle,
        conditions_per_optimizer_step=conditions_per_optimizer_step,
        contract_sha256=setup.contract_sha256,
        initial_step=initial_step,
    )
    sampler, video_schedule, loader = _build_sampler_and_loader(
        args=args,
        context=context,
        config=config,
        dataset=setup.dataset,
        task_ids=setup.task_ids,
        batch_size=batch_size,
        batch_cycle=batch_cycle,
        conditions_per_optimizer_step=conditions_per_optimizer_step,
        initial_step=initial_step,
    )
    wrapped = _wrap_writer(setup.writer, context)
    setup.writer.train()
    video_store, processor, language_tokens = _build_condition_inputs(
        args=args,
        config=config,
        authorities=setup.authorities,
        context=context,
        task_ids=setup.task_ids,
        tasks=setup.tasks,
    )
    metrics_path = args.output_dir / "metrics.jsonl"
    metrics_rows = _metrics_cursor(
        metrics_path,
        context=context,
        initial_step=initial_step,
        expected_rows=expected_metrics_rows,
    )
    checkpoint_validation = (
        prepare_online_writer_validation(
            training=setup.contract,
            data_root=args.data_root,
            tokenizer_path=args.tokenizer_path,
            context=context,
            output_dir=args.output_dir,
        )
        if args.mode == "formal" and writer_stage(config) == "development"
        else None
    )
    torch.cuda.reset_peak_memory_stats(context.device)
    barrier(context)
    if resume_rng is not None:
        restore_rng(resume_rng, context)
    return WriterRuntime(
        args=args,
        context=context,
        config=config,
        dataset=setup.dataset,
        task_authorities=setup.tasks,
        task_ids=setup.task_ids,
        sampler=sampler,
        video_schedule=video_schedule,
        flow_noise_schedule=WriterFlowNoiseSchedule(
            seed=int(config["data"]["writer_flow_noise_seed"]),
        ),
        iterator=iter(loader),
        video_store=video_store,
        language_tokens=language_tokens,
        processor=processor,
        policy=setup.policy,
        identity_state=setup.identity_state,
        writer=setup.writer,
        wrapped_writer=wrapped,
        optimizer=setup.optimizer,
        scheduler=setup.scheduler,
        lora_contract=setup.lora_contract,
        contract=setup.contract,
        contract_sha256=setup.contract_sha256,
        total_steps=total_steps,
        batch_size=batch_size,
        conditions_per_optimizer_step=conditions_per_optimizer_step,
        checkpoint_steps=checkpoint_steps,
        resume_step=initial_step,
        metrics_path=metrics_path,
        metrics_rows=metrics_rows,
        checkpoint_validation=checkpoint_validation,
    )


def _run_checkpoint_validation(
    runtime: WriterRuntime,
    checkpoint_cursor: int,
) -> None:
    if runtime.checkpoint_validation is None:
        return
    checkpoint_dir = (
        runtime.args.output_dir
        / "checkpoints"
        / f"step_{checkpoint_cursor:08d}"
    )
    summary = evaluate_online_writer_checkpoint(
        validation=runtime.checkpoint_validation,
        context=runtime.context,
        checkpoint_cursor=checkpoint_cursor,
        checkpoint_dir=checkpoint_dir,
        policy=runtime.policy,
        writer=runtime.writer,
        identity=runtime.identity_state,
        lora=runtime.lora_contract,
        processor=runtime.processor,
    )
    if runtime.context.is_main:
        print(
            json.dumps(
                {
                    "event": "validation_functional_loss",
                    **summary,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def run_steps(runtime: WriterRuntime) -> None:
    started = time.monotonic()
    if runtime.resume_step in runtime.checkpoint_steps:
        _run_checkpoint_validation(runtime, runtime.resume_step)
    for step in range(runtime.resume_step, runtime.args.stop_after_step):
        row = run_writer_step(runtime, step, started)
        completed = int(row["optimizer_step"])
        if runtime.context.is_main:
            append_jsonl(runtime.metrics_path, row)
            runtime.metrics_rows += 1
            if completed == 1 or completed % runtime.args.log_every == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
        if completed in runtime.checkpoint_steps:
            save_writer_checkpoint(
                output_dir=runtime.args.output_dir,
                step=completed,
                context=runtime.context,
                writer=runtime.writer,
                optimizer=runtime.optimizer,
                scheduler=runtime.scheduler,
                sampler=runtime.sampler,
                video_schedule=runtime.video_schedule,
                flow_noise_schedule=runtime.flow_noise_schedule,
                contract=runtime.contract,
                mode=runtime.args.mode,
                metrics_rows=runtime.metrics_rows,
            )
            _run_checkpoint_validation(runtime, completed)
    barrier(runtime.context)
    if runtime.context.is_main:
        stop = runtime.args.stop_after_step
        summary = {
            "schema_version": "ember_pi05_as_writer_run_summary_v1",
            "contract_sha256": runtime.contract_sha256,
            "completed_optimizer_steps": stop,
            "requested_optimizer_steps": runtime.total_steps,
            "stopped_early_for_profile": (
                runtime.args.mode == "profile" and stop < runtime.total_steps
            ),
            "selected_stage_stop": (
                runtime.args.mode == "formal" and stop < runtime.total_steps
            ),
            "metrics_rows": runtime.metrics_rows,
            "wall_seconds": time.monotonic() - started,
            "final_checkpoint": str(
                runtime.args.output_dir / "checkpoints" / f"step_{stop:08d}"
            )
            if stop in runtime.checkpoint_steps
            else None,
            "train_tasks": len(runtime.task_ids),
            "teacher_action_episodes_available": len(runtime.task_ids) * 50,
            "global_policy_samples": (
                stop * runtime.context.world_size * runtime.batch_size
            ),
            "global_independent_task_video_conditions": (
                stop * runtime.context.world_size
            ),
            "test_action_reads": 0,
            "test_video_value_reads": 0,
        }
        if writer_stage(runtime.config) == "final":
            summary["validation_action_episodes_available"] = 400
            summary["validation_video_episodes_available"] = 400
        else:
            summary["validation_action_reads"] = 0
        write_json_atomic(runtime.args.output_dir / "run_summary.json", summary)


def train(args: argparse.Namespace) -> None:
    context = initialize_distributed(require_numa=args.mode == "formal")
    runtime: WriterRuntime | None = None
    try:
        runtime = prepare_runtime(args, context)
        if context.is_main:
            print(
                json.dumps(
                    {
                        "event": "start",
                        "mode": args.mode,
                        "contract_sha256": runtime.contract_sha256,
                        "resume_step": runtime.resume_step,
                        "stop_after_step": args.stop_after_step,
                        "independent_conditions_per_optimizer_step": (
                            runtime.conditions_per_optimizer_step
                        ),
                        "tasks": len(runtime.task_ids),
                        "trainable": runtime.contract["trainable"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        run_steps(runtime)
    finally:
        if runtime is not None:
            runtime.dataset.close()
            runtime.video_store.close()
            if runtime.checkpoint_validation is not None:
                runtime.checkpoint_validation.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_as_writer_action_forecast_v1.json",
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initialize-writer-checkpoint",
        type=Path,
        help="Warm-start Writer weights in a new optimizer/scheduler/RNG phase.",
    )
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--checkpoint-steps", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--skip-data-sha", action="store_true")
    parser.add_argument(
        "--allow-contract-compatible-code-resume",
        action="store_true",
        help=(
            "Allow an explicit exact resume when every run-contract field except "
            "the recorded code commit is unchanged."
        ),
    )
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    config = load_writer_config(args.config.resolve())
    if args.num_workers is None:
        args.num_workers = int(config["loader"]["num_workers_per_rank"])
    if args.num_workers < 0 or args.log_every <= 0:
        raise WriterModelError("invalid AS-Writer loader or logging request")
    for name in (
        "config",
        "source_run",
        "checkpoint",
        "tokenizer_path",
        "data_root",
        "output_dir",
        "resume",
        "initialize_writer_checkpoint",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    return args
