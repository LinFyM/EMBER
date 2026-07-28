"""Symmetric-rank shared PI05 Source-SFT LoRA training."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.distributed as dist
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ember.lora import inject_task_lora, task_lora_state_dict
from ember.pi05_eval_contract import (
    inspect_source_checkpoint,
    inspect_tokenizer,
    load_evaluation_authorities,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor
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
    reduce_max,
    reduce_mean,
    seed_everything,
)
from ember.source_sft.checkpoint import (
    load_source_sft_checkpoint,
    save_source_sft_checkpoint,
)
from ember.source_sft.contract import (
    REPO_ROOT,
    Pi05SourceSFTError,
    SourceSFTTask,
    authority_path,
    build_contract,
    load_source_sft_config,
    load_training_data,
    publish_contract,
    reconcile_resume_contract,
    resolve_runtime,
    trainable_contract,
)
from ember.source_sft.online_validation import (
    OnlineSourceSFTValidation,
    evaluate_online_source_sft_checkpoint,
    prepare_online_source_sft_validation,
)
from ember.source_sft.sampler import HierarchicalMixedBatchSampler
from ember.writer.data import FunctionalQueryDataset


_CHECKPOINT_NAME = re.compile(r"step_([0-9]{8})")
_ACTIVE_TRAINING_RECIPE = "hierarchical_task_episode_chunk_mixed_v1"


@dataclass
class SourceSFTRuntime:
    args: argparse.Namespace
    context: DistributedContext
    config: dict[str, Any]
    dataset: FunctionalQueryDataset
    tasks: tuple[SourceSFTTask, ...]
    task_ids: tuple[int, ...]
    sampler: HierarchicalMixedBatchSampler
    iterator: Iterator[dict[str, Any]]
    processor: Pi05LiberoProcessor
    policy: torch.nn.Module
    wrapped: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    lora_contract: Any
    contract: dict[str, Any]
    contract_sha256: str
    total_steps: int
    batch_size: int
    checkpoint_steps: tuple[int, ...]
    resume_step: int
    metrics_path: Path
    metrics_rows: int
    checkpoint_validation: OnlineSourceSFTValidation | None


def _resume_step(checkpoint: Path | None) -> int:
    if checkpoint is None:
        return 0
    match = _CHECKPOINT_NAME.fullmatch(checkpoint.name)
    if match is None:
        raise Pi05SourceSFTError("Source-SFT resume path is not a step checkpoint")
    return int(match.group(1))


def _validate_active_training_recipe(config: Mapping[str, Any]) -> None:
    recipe = config.get("training_recipe", {})
    if (
        not isinstance(recipe, Mapping)
        or recipe.get("kind") != _ACTIVE_TRAINING_RECIPE
        or recipe.get("rank_task_binding") != "none"
        or recipe.get("loss_reduction")
        != "equal samples per task then ordinary batch mean"
    ):
        raise Pi05SourceSFTError(
            "legacy rank-pure Source-SFT training is retired; "
            "use the canonical hierarchical mixed-task recipe"
        )


def _scheduler(
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


def _build_trainable_policy(
    *,
    source: Mapping[str, Any],
    source_config: Mapping[str, Any],
    config: Mapping[str, Any],
    context: DistributedContext,
    total_steps: int,
) -> tuple[
    torch.nn.Module,
    Any,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    dict[str, Any],
]:
    policy = load_policy(Path(source["model_path"]), dict(source_config), context.device)
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    inject_task_lora(policy, lora)
    policy.train()
    trainable = trainable_contract(policy, lora)
    optimizer_config = config["optimization"]["optimizer"]
    optimizer = torch.optim.AdamW(
        task_lora_state_dict(policy).values(),
        lr=float(config["optimization"]["scheduler"]["peak_lr"]),
        betas=tuple(optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler = _scheduler(
        optimizer, config["optimization"]["scheduler"], total_steps
    )
    return policy, lora, optimizer, scheduler, trainable


def _loader(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    config: Mapping[str, Any],
    dataset: FunctionalQueryDataset,
    task_ids: tuple[int, ...],
    batch_size: int,
    initial_step: int,
) -> tuple[HierarchicalMixedBatchSampler, DataLoader[Any]]:
    sampler = HierarchicalMixedBatchSampler(
        dataset,
        task_ids=task_ids,
        per_rank_batch_size=batch_size,
        start_step=initial_step,
        stop_step=args.stop_after_step,
        rank=context.rank,
        world_size=context.world_size,
        seed=int(config["data"]["sampler_seed"]),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=int(config["loader"]["prefetch_factor"])
        if args.num_workers
        else None,
        generator=torch.Generator().manual_seed(
            int(config["optimization"]["seed"]) + context.rank + 0x5F7
        ),
    )
    return sampler, loader


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


def _wrap(policy: torch.nn.Module, context: DistributedContext) -> torch.nn.Module:
    if context.world_size == 1:
        return policy
    return DistributedDataParallel(
        policy,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
        static_graph=True,
    )


def _build_processor(
    args: argparse.Namespace,
    context: DistributedContext,
    authorities: Any,
) -> Pi05LiberoProcessor:
    source = authorities.source_base_config
    return Pi05LiberoProcessor(
        load_stats(source, source["data"]["active_task_ids"]),
        args.tokenizer_path,
        int(source["features"]["tokenizer_max_length"]),
        str(context.device),
    )


def _prepare_validation_monitor(
    args: argparse.Namespace,
    context: DistributedContext,
    contract: Mapping[str, Any],
) -> OnlineSourceSFTValidation | None:
    if args.mode != "formal" or args.stage != "development":
        return None
    return prepare_online_source_sft_validation(
        training=contract,
        data_root=args.data_root,
        context=context,
        output_dir=args.output_dir,
    )


def prepare_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> SourceSFTRuntime:
    config = load_source_sft_config(args.config.resolve())
    _validate_active_training_recipe(config)
    total_steps, batch_size, checkpoint_steps = resolve_runtime(args, config, context)
    initial_step = _resume_step(args.resume)
    if not 0 <= initial_step < args.stop_after_step:
        raise Pi05SourceSFTError("Source-SFT resume cursor is outside this segment")
    seed_everything(int(config["optimization"]["seed"]), context)
    dataset, tasks, data_validation = load_training_data(args, config, context)
    task_ids = tuple(task.global_task_id for task in tasks)
    authorities = load_evaluation_authorities(
        authority_path(config, "evaluation_config"), REPO_ROOT
    )
    source = inspect_source_checkpoint(
        authorities,
        args.source_run,
        args.checkpoint,
        evaluation_mode="formal",
    )
    tokenizer = inspect_tokenizer(authorities, args.tokenizer_path)
    policy, lora, optimizer, scheduler, trainable = _build_trainable_policy(
        source=source,
        source_config=authorities.source_base_config,
        config=config,
        context=context,
        total_steps=total_steps,
    )
    candidate_contract = build_contract(
        args=args,
        config=config,
        context=context,
        source=source,
        tokenizer=tokenizer,
        data_validation=data_validation,
        tasks=tasks,
        trainable=trainable,
        total_steps=total_steps,
        batch_size=batch_size,
        checkpoint_steps=checkpoint_steps,
    )
    contract = reconcile_resume_contract(args, candidate_contract)
    contract_sha256 = canonical_hash(contract)
    publish_contract(args, context, contract, contract_sha256)

    resume_rng = None
    expected_metrics_rows = 0
    if args.resume is not None:
        loaded, resume_rng, expected_metrics_rows = load_source_sft_checkpoint(
            checkpoint=args.resume,
            context=context,
            policy=policy,
            lora_contract=lora,
            optimizer=optimizer,
            scheduler=scheduler,
            per_rank_batch_size=batch_size,
            samples_per_task_per_rank=batch_size // len(task_ids),
            sampler_seed=int(config["data"]["sampler_seed"]),
            dataloader_generator_seed=int(config["optimization"]["seed"])
            + context.rank
            + 0x5F7,
            contract_sha256=contract_sha256,
        )
        if loaded != initial_step:
            raise Pi05SourceSFTError("Source-SFT resume path and state disagree")
    sampler, loader = _loader(
        args=args,
        context=context,
        config=config,
        dataset=dataset,
        task_ids=task_ids,
        batch_size=batch_size,
        initial_step=initial_step,
    )
    wrapped = _wrap(policy, context)
    processor = _build_processor(args, context, authorities)
    metrics_path = args.output_dir / "metrics.jsonl"
    metrics_rows = _metrics_cursor(
        metrics_path,
        context=context,
        initial_step=initial_step,
        expected_rows=expected_metrics_rows,
    )
    iterator = iter(loader)
    checkpoint_validation = _prepare_validation_monitor(
        args,
        context,
        contract,
    )
    torch.cuda.reset_peak_memory_stats(context.device)
    barrier(context)
    if resume_rng is not None:
        restore_rng(resume_rng, context)
    return SourceSFTRuntime(
        args=args,
        context=context,
        config=config,
        dataset=dataset,
        tasks=tasks,
        task_ids=task_ids,
        sampler=sampler,
        iterator=iterator,
        processor=processor,
        policy=policy,
        wrapped=wrapped,
        optimizer=optimizer,
        scheduler=scheduler,
        lora_contract=lora,
        contract=contract,
        contract_sha256=contract_sha256,
        total_steps=total_steps,
        batch_size=batch_size,
        checkpoint_steps=checkpoint_steps,
        resume_step=initial_step,
        metrics_path=metrics_path,
        metrics_rows=metrics_rows,
        checkpoint_validation=checkpoint_validation,
    )


def _batch_task_counts(batch: Mapping[str, Any]) -> dict[int, int]:
    values = batch.get("task_id")
    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise Pi05SourceSFTError("Source-SFT batch lost task identity")
    unique, counts = values.unique(return_counts=True)
    return {
        int(task_id): int(count)
        for task_id, count in zip(unique.tolist(), counts.tolist(), strict=True)
    }


def _one_step(runtime: SourceSFTRuntime, step: int, started: float) -> dict[str, Any]:
    tick = time.monotonic()
    batch = next(runtime.iterator)
    data_seconds = time.monotonic() - tick
    task_counts = _batch_task_counts(batch)
    expected_per_task = runtime.sampler.samples_per_task_per_rank
    if (
        set(task_counts) != set(runtime.task_ids)
        or set(task_counts.values()) != {expected_per_task}
    ):
        raise Pi05SourceSFTError(
            "Source-SFT physical batch is not exactly task-balanced"
        )
    policy_batch = runtime.processor.training_batch(batch)
    runtime.optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, _ = runtime.wrapped(policy_batch)
    if not bool(torch.isfinite(loss).detach()):
        raise Pi05SourceSFTError(f"non-finite Source-SFT loss at step {step}")
    loss.backward()
    if any(
        parameter.grad is not None
        for parameter in runtime.policy.parameters()
        if not parameter.requires_grad
    ):
        raise Pi05SourceSFTError("frozen source policy accumulated gradients")
    trainable = tuple(task_lora_state_dict(runtime.policy).values())
    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable,
        float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"]),
    )
    if not bool(torch.isfinite(grad_norm).detach()):
        raise Pi05SourceSFTError(f"non-finite Source-SFT gradient at step {step}")
    applied_lr = float(runtime.optimizer.param_groups[0]["lr"])
    runtime.optimizer.step()
    runtime.scheduler.step()
    completed = step + 1
    step_seconds = reduce_max(time.monotonic() - tick, runtime.context)
    examples = runtime.context.world_size * runtime.batch_size
    return {
        "optimizer_step": completed,
        "mean_action_loss": reduce_mean(float(loss.detach()), runtime.context),
        "gradient_norm_before_clip_max": reduce_max(float(grad_norm), runtime.context),
        "applied_lr": applied_lr,
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "global_action_queries": completed * examples,
        "rank0_task_ids": list(runtime.task_ids),
        "rank0_samples_per_task": expected_per_task,
        "global_samples_per_task_this_step": (
            expected_per_task * runtime.context.world_size
        ),
        "physical_batch_task_count": len(task_counts),
        "loss_reduction": "equal_samples_per_task_then_batch_mean",
        "data_seconds_max": reduce_max(data_seconds, runtime.context),
        "step_seconds_max": step_seconds,
        "global_action_queries_per_second": examples / step_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": int(
            reduce_max(
                torch.cuda.max_memory_allocated(runtime.context.device),
                runtime.context,
            )
        ),
        "max_cuda_reserved_bytes": int(
            reduce_max(
                torch.cuda.max_memory_reserved(runtime.context.device),
                runtime.context,
            )
        ),
    }


def _run_checkpoint_validation(
    runtime: SourceSFTRuntime,
    checkpoint_cursor: int,
) -> None:
    if runtime.checkpoint_validation is None:
        return
    checkpoint_dir = (
        runtime.args.output_dir
        / "checkpoints"
        / f"step_{checkpoint_cursor:08d}"
    )
    summary = evaluate_online_source_sft_checkpoint(
        validation=runtime.checkpoint_validation,
        context=runtime.context,
        checkpoint_cursor=checkpoint_cursor,
        checkpoint_dir=checkpoint_dir,
        policy=runtime.policy,
        processor=runtime.processor,
    )
    if runtime.context.is_main:
        print(
            json.dumps(
                {"event": "validation_functional_loss", **summary},
                sort_keys=True,
            ),
            flush=True,
        )


def run_steps(runtime: SourceSFTRuntime) -> None:
    started = time.monotonic()
    if runtime.resume_step in runtime.checkpoint_steps:
        _run_checkpoint_validation(runtime, runtime.resume_step)
    for step in range(runtime.resume_step, runtime.args.stop_after_step):
        row = _one_step(runtime, step, started)
        completed = int(row["optimizer_step"])
        if runtime.context.is_main:
            append_jsonl(runtime.metrics_path, row)
            runtime.metrics_rows += 1
            if completed == 1 or completed % runtime.args.log_every == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
        if completed in runtime.checkpoint_steps:
            save_source_sft_checkpoint(
                output_dir=runtime.args.output_dir,
                step=completed,
                context=runtime.context,
                policy=runtime.policy,
                optimizer=runtime.optimizer,
                scheduler=runtime.scheduler,
                sampler=runtime.sampler,
                contract=runtime.contract,
                mode=runtime.args.mode,
                metrics_rows=runtime.metrics_rows,
            )
            _run_checkpoint_validation(runtime, completed)
    barrier(runtime.context)
    if runtime.context.is_main:
        stop = runtime.args.stop_after_step
        validation_episodes = 400 if runtime.args.stage == "final" else 0
        validation_summaries = (
            [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(
                    runtime.checkpoint_validation.output_dir.glob(
                        "step_*/summary.json"
                    )
                )
            ]
            if runtime.checkpoint_validation is not None
            else []
        )
        write_json_atomic(
            runtime.args.output_dir / "run_summary.json",
            {
                "schema_version": "ember_pi05_source_sft_run_summary_v1",
                "contract_sha256": runtime.contract_sha256,
                "stage": runtime.args.stage,
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
                "validation_action_episodes_available": validation_episodes,
                "validation_action_queries_read_by_checkpoint_monitor": sum(
                    int(summary["row_count"]) for summary in validation_summaries
                ),
                "validation_checkpoint_monitor_count": len(validation_summaries),
                "validation_checkpoint_monitor_optimizer_updates": 0,
                "test_action_reads": 0,
                "teacher_video_value_reads": 0,
                "trainable_parameter_count": runtime.contract["trainable"][
                    "parameter_count"
                ],
                "global_action_queries": stop
                * runtime.context.world_size
                * runtime.batch_size,
            },
        )


def train(args: argparse.Namespace) -> None:
    context = initialize_distributed(require_numa=args.mode == "formal")
    runtime: SourceSFTRuntime | None = None
    try:
        runtime = prepare_runtime(args, context)
        if context.is_main:
            print(
                json.dumps(
                    {
                        "event": "start",
                        "mode": args.mode,
                        "stage": args.stage,
                        "contract_sha256": runtime.contract_sha256,
                        "resume_step": runtime.resume_step,
                        "stop_after_step": args.stop_after_step,
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
            if runtime.checkpoint_validation is not None:
                runtime.checkpoint_validation.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_source_sft_rank128_mixed_v2.json",
    )
    parser.add_argument("--stage", choices=("development", "final"), required=True)
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
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
    config = load_source_sft_config(args.config.resolve())
    if args.num_workers is None:
        args.num_workers = int(config["loader"]["num_workers_per_rank"])
    if args.num_workers < 0 or args.log_every <= 0:
        raise Pi05SourceSFTError("invalid Source-SFT loader or logging request")
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
