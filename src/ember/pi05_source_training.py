"""Canonical multi-GPU PI05 source-base action-SFT on the filtered LIBERO-90 corpus."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_contract import (
    append_jsonl,
    build_contract,
    load_resume,
    reconcile_metrics,
    resolve_runtime,
)
from ember.pi05_source_checkpoint import (
    DistributedContext,
    Pi05SourceTrainingError,
    barrier,
    canonical_hash,
    git_state,
    read_json,
    restore_rng,
    save_checkpoint,
    write_json_atomic,
)
from ember.pi05_source_setup import (
    initialize_distributed,
    load_authorities,
    load_config,
    load_policy,
    load_stats,
    make_scheduler,
    reduce_max,
    reduce_mean,
    seed_everything,
    trainable_contract,
    update_ema,
    validate_runtime_assets,
    validate_source_files,
)
from ember.writer.data import (
    FunctionalQueryDataset,
    MixedTaskBatchSampler,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainingRuntime:
    args: argparse.Namespace
    context: DistributedContext
    config: dict[str, Any]
    manifest: dict[str, Any]
    dataset: FunctionalQueryDataset
    task_ids: tuple[int, ...]
    policy: torch.nn.Module
    ema_policy: torch.nn.Module | None
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    wrapped: torch.nn.Module
    processor: Pi05LiberoProcessor
    iterator: Iterator[dict[str, Any]]
    contract_sha256: str
    optimizer_steps: int
    micro_batch: int
    accumulation: int
    checkpoint_interval: int
    resume_optimizer_step: int
    metrics_path: Path
    metrics_rows: int


def _build_dataset(
    args: argparse.Namespace,
    config: dict[str, Any],
    context: DistributedContext,
) -> tuple[FunctionalQueryDataset, dict[str, Any], tuple[int, ...], dict[str, Any]]:
    asset_validation = validate_runtime_assets(
        config=config,
        foundation_path=args.foundation_path,
        tokenizer_path=args.tokenizer_path,
        context=context,
        verify_weight_hash=args.mode == "formal",
    )
    authorities, manifest = load_authorities(
        config, args.data_root, task_limit=args.task_limit
    )
    asset_validation["source_corpus"] = validate_source_files(
        authorities=authorities,
        manifest=manifest,
        context=context,
        verify_hashes=not args.skip_data_sha,
    )
    first_demo, last_demo = config["data"]["demo_indices"]
    dataset = FunctionalQueryDataset(
        authorities,
        demo_indices=range(int(first_demo), int(last_demo) + 1),
        action_chunk_size=int(config["features"]["chunk_size"]),
        max_open_files_per_worker=int(config["data"]["max_open_files_per_worker"]),
    )
    return dataset, manifest, tuple(item.task_id for item in authorities), asset_validation


def _build_models(
    args: argparse.Namespace,
    config: dict[str, Any],
    context: DistributedContext,
    ema_enabled: bool,
) -> tuple[torch.nn.Module, torch.nn.Module | None, Any, Any, dict[str, Any]]:
    model_path = args.foundation_path if args.resume is None else args.resume / "policy"
    policy = load_policy(model_path, config, context.device)
    trainable = trainable_contract(policy)
    ema_policy = None
    if ema_enabled:
        ema_path = None if args.resume is None else args.resume / "ema_policy"
        ema_policy = (
            copy.deepcopy(policy).eval()
            if ema_path is None
            else load_policy(ema_path, config, context.device).eval()
        )
        for parameter in ema_policy.parameters():
            parameter.requires_grad_(False)
    optimizer_config = config["optimization"]["optimizer"]
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(config["optimization"]["schedule"]["peak_lr"]),
        betas=tuple(optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler = make_scheduler(
        optimizer,
        warmup_steps=int(config["optimization"]["schedule"]["warmup_steps"]),
        peak_lr=float(config["optimization"]["schedule"]["peak_lr"]),
    )
    return policy, ema_policy, optimizer, scheduler, trainable


def _publish_contract(
    args: argparse.Namespace,
    context: DistributedContext,
    contract: dict[str, Any],
    contract_sha256: str,
) -> None:
    payload: list[Any] = [None]
    if context.is_main:
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            contract_path = args.output_dir / "run_contract.json"
            if contract_path.exists():
                if canonical_hash(read_json(contract_path)) != contract_sha256:
                    raise Pi05SourceTrainingError("output directory has another launch contract")
            else:
                if any(args.output_dir.iterdir()):
                    raise Pi05SourceTrainingError("fresh output directory is not empty")
                write_json_atomic(contract_path, contract)
            append_jsonl(
                args.output_dir / "invocations.jsonl",
                {
                    "argv": sys.argv,
                    "git_observed": git_state(),
                    "resume": str(args.resume) if args.resume else None,
                    "started_unix": time.time(),
                },
            )
            payload[0] = {"ok": True}
        except Exception as error:
            payload[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=context.device)
    if payload[0].get("error"):
        raise Pi05SourceTrainingError(payload[0]["error"])


def _build_loader(
    runtime: TrainingRuntime,
    resume_micro_step: int,
    expected_metrics_rows: int,
) -> None:
    seed = int(runtime.config["optimization"]["seed"])
    sampler = MixedTaskBatchSampler(
        runtime.dataset,
        task_ids=runtime.task_ids,
        per_rank_batch_size=runtime.micro_batch,
        start_step=resume_micro_step,
        stop_step=runtime.optimizer_steps * runtime.accumulation,
        rank=runtime.context.rank,
        world_size=runtime.context.world_size,
        seed=int(runtime.config["data"]["sampler_seed"]),
    )
    loader = DataLoader(
        runtime.dataset,
        batch_sampler=sampler,
        num_workers=runtime.args.num_workers,
        pin_memory=True,
        persistent_workers=runtime.args.num_workers > 0,
        prefetch_factor=2 if runtime.args.num_workers > 0 else None,
        generator=torch.Generator().manual_seed(seed + runtime.context.rank + 0xDADA),
    )
    runtime.iterator = iter(loader)
    if runtime.args.resume is None:
        seed_everything(seed, runtime.context)
    else:
        rank_state = torch.load(
            runtime.args.resume / f"rank_{runtime.context.rank:02d}_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        restore_rng(rank_state["rng"], runtime.context)
    runtime.metrics_rows = 0
    if runtime.context.is_main:
        runtime.metrics_rows = reconcile_metrics(
            runtime.metrics_path,
            runtime.resume_optimizer_step,
            expected_metrics_rows,
        )
    if runtime.context.world_size > 1:
        rows = torch.tensor(runtime.metrics_rows, device=runtime.context.device)
        dist.broadcast(rows, src=0)
        runtime.metrics_rows = int(rows.item())


def _prepare_training(args: argparse.Namespace, context: DistributedContext) -> TrainingRuntime:
    config_path = args.config.resolve()
    config = load_config(config_path)
    optimizer_steps, micro_batch, accumulation, checkpoint_interval, ema_enabled = (
        resolve_runtime(args, config, context)
    )
    seed_everything(int(config["optimization"]["seed"]), context)
    dataset, manifest, task_ids, asset_validation = _build_dataset(args, config, context)
    policy, ema_policy, optimizer, scheduler, trainable = _build_models(
        args, config, context, ema_enabled
    )
    contract = build_contract(
        args=args,
        config_path=config_path,
        config=config,
        context=context,
        trainable=trainable,
        task_ids=task_ids,
        ema_enabled=ema_enabled,
        optimizer_steps=optimizer_steps,
        micro_batch_size=micro_batch,
        gradient_accumulation=accumulation,
        checkpoint_interval=checkpoint_interval,
        asset_validation=asset_validation,
    )
    contract_sha256 = canonical_hash(contract)
    _publish_contract(args, context, contract, contract_sha256)
    wrapped: torch.nn.Module = policy
    if context.world_size > 1:
        wrapped = DistributedDataParallel(
            policy,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            static_graph=True,
        )
    resume_step, resume_micro, expected_rows = load_resume(
        args=args,
        context=context,
        contract_sha256=contract_sha256,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_accumulation=accumulation,
    )
    if resume_step > optimizer_steps:
        raise Pi05SourceTrainingError("resume step exceeds requested training horizon")
    runtime = TrainingRuntime(
        args=args,
        context=context,
        config=config,
        manifest=manifest,
        dataset=dataset,
        task_ids=task_ids,
        policy=policy,
        ema_policy=ema_policy,
        optimizer=optimizer,
        scheduler=scheduler,
        wrapped=wrapped,
        processor=Pi05LiberoProcessor(
            load_stats(config, task_ids),
            args.tokenizer_path,
            int(config["features"]["tokenizer_max_length"]),
            str(context.device),
        ),
        iterator=iter(()),
        contract_sha256=contract_sha256,
        optimizer_steps=optimizer_steps,
        micro_batch=micro_batch,
        accumulation=accumulation,
        checkpoint_interval=checkpoint_interval,
        resume_optimizer_step=resume_step,
        metrics_path=args.output_dir / "metrics.jsonl",
        metrics_rows=0,
    )
    _build_loader(runtime, resume_micro, expected_rows)
    return runtime


def _optimizer_step(
    runtime: TrainingRuntime, optimizer_step: int, started: float
) -> dict[str, Any]:
    raw_loss_sum = 0.0
    step_started = time.monotonic()
    for accumulation_index in range(runtime.accumulation):
        batch = runtime.processor.training_batch(next(runtime.iterator))
        synchronize = accumulation_index == runtime.accumulation - 1
        sync_context = (
            contextlib.nullcontext()
            if synchronize or not isinstance(runtime.wrapped, DistributedDataParallel)
            else runtime.wrapped.no_sync()
        )
        with sync_context:
            loss, _ = runtime.wrapped(batch)
            if not torch.isfinite(loss):
                raise Pi05SourceTrainingError(
                    f"non-finite loss at optimizer step {optimizer_step}"
                )
            raw_loss_sum += float(loss.detach())
            (loss / runtime.accumulation).backward()
    optimizer_config = runtime.config["optimization"]["optimizer"]
    grad_norm = torch.nn.utils.clip_grad_norm_(
        runtime.policy.parameters(), float(optimizer_config["gradient_clip_norm"])
    )
    if not torch.isfinite(grad_norm):
        raise Pi05SourceTrainingError(f"non-finite gradient at optimizer step {optimizer_step}")
    applied_lr = float(runtime.optimizer.param_groups[0]["lr"])
    runtime.optimizer.step()
    runtime.scheduler.step()
    runtime.optimizer.zero_grad(set_to_none=True)
    if runtime.ema_policy is not None:
        update_ema(
            runtime.ema_policy,
            runtime.policy,
            float(runtime.config["optimization"]["ema_decay"]),
        )
    completed = optimizer_step + 1
    seconds = reduce_max(time.monotonic() - step_started, runtime.context)
    examples = runtime.context.world_size * runtime.micro_batch * runtime.accumulation
    return {
        "optimizer_step": completed,
        "micro_step": completed * runtime.accumulation,
        "mean_loss": reduce_mean(raw_loss_sum / runtime.accumulation, runtime.context),
        "gradient_norm_before_clip_max": reduce_max(float(grad_norm), runtime.context),
        "applied_lr": applied_lr,
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "global_examples": completed * examples,
        "step_seconds_max": seconds,
        "examples_per_second": examples / seconds,
        "elapsed_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": int(
            reduce_max(torch.cuda.max_memory_allocated(runtime.context.device), runtime.context)
        ),
        "max_cuda_reserved_bytes": int(
            reduce_max(torch.cuda.max_memory_reserved(runtime.context.device), runtime.context)
        ),
    }


def _train(runtime: TrainingRuntime) -> None:
    runtime.optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(runtime.context.device)
    started = time.monotonic()
    stop_step = runtime.args.stop_after_optimizer_step or runtime.optimizer_steps
    if not runtime.resume_optimizer_step < stop_step <= runtime.optimizer_steps:
        raise Pi05SourceTrainingError("smoke stop step must be after resume and within horizon")
    for optimizer_step in range(runtime.resume_optimizer_step, stop_step):
        row = _optimizer_step(runtime, optimizer_step, started)
        completed = int(row["optimizer_step"])
        if runtime.context.is_main:
            append_jsonl(runtime.metrics_path, row)
            runtime.metrics_rows += 1
            interval = int(runtime.config["logging"]["interval_steps"])
            if completed == 1 or completed % interval == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
        should_checkpoint = runtime.checkpoint_interval > 0 and (
            completed % runtime.checkpoint_interval == 0
            or completed == stop_step
        )
        if should_checkpoint:
            save_checkpoint(
                output_dir=runtime.args.output_dir,
                optimizer_step=completed,
                micro_step=completed * runtime.accumulation,
                context=runtime.context,
                policy=runtime.policy,
                ema_policy=runtime.ema_policy,
                optimizer=runtime.optimizer,
                scheduler=runtime.scheduler,
                contract_sha256=runtime.contract_sha256,
                metrics_rows=runtime.metrics_rows,
                keep_latest=int(runtime.config["checkpointing"]["keep_latest"]),
            )
    barrier(runtime.context)
    if runtime.context.is_main:
        checkpoint = (
            runtime.args.output_dir
            / "checkpoints"
            / f"step_{stop_step:08d}"
            if runtime.checkpoint_interval > 0
            else None
        )
        write_json_atomic(
            runtime.args.output_dir / "run_summary.json",
            {
                "schema_version": "ember_pi05_source_run_summary_v1",
                "contract_sha256": runtime.contract_sha256,
                "completed_optimizer_steps": stop_step,
                "completed_micro_steps": stop_step * runtime.accumulation,
                "requested_optimizer_steps": runtime.optimizer_steps,
                "stopped_early_for_resume_smoke": stop_step < runtime.optimizer_steps,
                "metrics_rows": runtime.metrics_rows,
                "wall_seconds": time.monotonic() - started,
                "final_checkpoint": str(checkpoint) if checkpoint else None,
                "frozen_policy_subdir": (
                    "ema_policy" if runtime.ema_policy is not None else "policy"
                ),
                "source_tasks": len(runtime.task_ids),
                "episodes": len(runtime.task_ids) * 50,
                "source_manifest_summary": runtime.manifest["summary"],
            },
        )


def run(args: argparse.Namespace) -> None:
    context = initialize_distributed()
    runtime: TrainingRuntime | None = None
    try:
        runtime = _prepare_training(args, context)
        _train(runtime)
    finally:
        if runtime is not None:
            runtime.dataset.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--foundation-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--optimizer-steps", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ema", choices=("config", "on", "off"), default="config")
    parser.add_argument("--task-limit", type=int)
    parser.add_argument("--skip-data-sha", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-optimizer-step", type=int)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
