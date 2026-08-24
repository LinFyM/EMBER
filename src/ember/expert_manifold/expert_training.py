"""Train independent policy-effective PI0.5 rank-16 task experts."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import default_collate

from ember.expert_manifold.checkpoint import (
    load_task_expert_checkpoint,
    save_task_expert_checkpoint,
)
from ember.expert_manifold.composite_contract import (
    COMPOSITE_DISTILLATION_CONFIG_SCHEMA,
)
from ember.expert_manifold.recovery_contract import RECOVERY_EXPERT_CONFIG_SCHEMA
from ember.expert_manifold.contract import (
    REPO_ROOT,
    ExpertManifoldError,
    ExpertTask,
    authority_path,
    build_dataset,
    build_worker_contract,
    load_task_expert_config,
    load_train_tasks,
    parse_resume_task,
    parse_task_indices,
    publish_worker_contract,
    resolve_runtime,
    task_directory,
    validate_formal_task_assignment,
    worker_stage_resume_step,
)
from ember.expert_manifold.sampler import (
    BalancedTwoDomainSampler,
    TaskLocalEpochSampler,
)
from ember.lora import (
    copy_task_lora_state_,
    inject_task_lora,
    task_lora_state_dict,
    validate_lora_state,
)
from ember.pi05_eval_contract import (
    inspect_source_checkpoint,
    load_evaluation_authorities,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.functional import pi05_mean_flow_loss
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import load_policy, load_stats


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    decay_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    if not 0 < warmup_steps < total_steps or not 0 < decay_lr <= peak_lr:
        raise ExpertManifoldError("task-expert scheduler values are invalid")

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
        value = decay_lr + 0.5 * (peak_lr - decay_lr) * (
            1.0 + math.cos(math.pi * progress)
        )
        return value / peak_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _optimizer_and_scheduler(
    policy: torch.nn.Module,
    config: Mapping[str, Any],
    total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    optimization = config["task_experts"]["optimization"]
    optimizer_config = optimization["optimizer"]
    scheduler_config = optimization["scheduler"]
    optimizer = torch.optim.AdamW(
        task_lora_state_dict(policy).values(),
        lr=float(scheduler_config["peak_lr"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler = _scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=int(scheduler_config["warmup_steps"]),
        peak_lr=float(scheduler_config["peak_lr"]),
        decay_lr=float(scheduler_config["decay_lr"]),
    )
    return optimizer, scheduler


def _metric_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _task_summary(
    *, task: ExpertTask, task_dir: Path, step: int, metrics_rows: int
) -> None:
    write_json_atomic(
        task_dir / "completion.json",
        {
            "schema_version": "ember_pi05_task_expert_completion_v1",
            "task_ordinal": task.ordinal,
            "global_task_id": task.global_task_id,
            "completed_steps": step,
            "metrics_rows": metrics_rows,
            "final_checkpoint": str(
                (task_dir / "checkpoints" / f"step_{step:08d}").resolve()
            ),
            "content_hash_policy": "disabled_by_owner",
        },
    )


def _task_expert_metric_values(
    loss: torch.Tensor,
    grad_norm: torch.Tensor,
    *,
    task: ExpertTask,
    step: int,
) -> tuple[float, float]:
    values = (
        torch.stack(
            (loss.detach().to(torch.float32), grad_norm.detach().to(torch.float32))
        )
        .to(device="cpu")
        .tolist()
    )
    if not math.isfinite(values[0]):
        raise ExpertManifoldError(
            f"non-finite task-expert loss for task {task.ordinal} at step {step}"
        )
    if not math.isfinite(values[1]):
        raise ExpertManifoldError("task-expert gradient is non-finite")
    return values[0], values[1]


def _task_expert_metric_row(
    *,
    task: ExpertTask,
    completed: int,
    batch_rows: int,
    action_queries: int,
    loss: float,
    grad_norm: float,
    applied_lr: float,
    next_lr: float,
    data_seconds: float,
    tick: float,
    started: float,
) -> dict[str, Any]:
    return {
        "optimizer_step": completed,
        "task_ordinal": task.ordinal,
        "global_task_id": task.global_task_id,
        "mean_action_loss": loss,
        "gradient_norm_before_clip": grad_norm,
        "applied_lr": applied_lr,
        "next_lr": next_lr,
        "batch_rows": batch_rows,
        "action_queries": action_queries,
        "data_seconds": data_seconds,
        "step_seconds": time.monotonic() - tick,
        "elapsed_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _optimize_task_batch(
    *,
    policy: torch.nn.Module,
    processor: Pi05LiberoProcessor,
    batch: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    task: ExpertTask,
    step: int,
    clip: float,
    action_queries: int,
    data_seconds: float,
    tick: float,
    started: float,
    mask_action_padding: bool,
) -> dict[str, Any]:
    policy_batch = processor.training_batch(dict(batch))
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = pi05_mean_flow_loss(
            policy,
            policy_batch,
            action_is_pad=(batch["action_is_pad"] if mask_action_padding else None),
        )
    loss.backward()
    if any(
        parameter.grad is not None
        for name, parameter in policy.named_parameters()
        if ".lora_" not in name
    ):
        raise ExpertManifoldError("frozen source policy accumulated expert gradients")
    trainable = tuple(task_lora_state_dict(policy).values())
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, clip)
    loss_value, grad_norm_value = _task_expert_metric_values(
        loss, grad_norm, task=task, step=step
    )
    applied_lr = float(optimizer.param_groups[0]["lr"])
    optimizer.step()
    scheduler.step()
    return _task_expert_metric_row(
        task=task,
        completed=step + 1,
        batch_rows=int(batch["action"].shape[0]),
        action_queries=action_queries,
        loss=loss_value,
        grad_norm=grad_norm_value,
        applied_lr=applied_lr,
        next_lr=float(optimizer.param_groups[0]["lr"]),
        data_seconds=data_seconds,
        tick=tick,
        started=started,
    )


def _train_one_task(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    task: ExpertTask,
    dataset: Any,
    policy: torch.nn.Module,
    processor: Pi05LiberoProcessor,
    lora_contract: Any,
    initial_state: Mapping[str, torch.Tensor],
    total_steps: int,
    batch_size: int,
    checkpoint_steps: Sequence[int],
    stop_step: int,
    resume: Path | None,
) -> dict[str, Any]:
    task_dir = task_directory(args.output_dir, task)
    metrics_path = task_dir / "metrics.jsonl"
    if resume is None:
        if task_dir.exists() and any(task_dir.iterdir()):
            raise ExpertManifoldError("fresh task-expert directory is not empty")
        task_dir.mkdir(parents=True, exist_ok=True)
        copy_task_lora_state_(policy, initial_state, lora_contract)
        initial_step = 0
        metrics_rows = 0
    else:
        initial_step = -1
        metrics_rows = -1
    schedule_total_steps = int(
        config["task_experts"]["profile_defaults"].get(
            "scheduler_total_steps", total_steps
        )
    )
    optimizer, scheduler = _optimizer_and_scheduler(
        policy, config, schedule_total_steps
    )
    if resume is not None:
        initial_step, metrics_rows = load_task_expert_checkpoint(
            checkpoint=resume,
            task=task,
            policy=policy,
            lora_contract=lora_contract,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if _metric_rows(metrics_path) != metrics_rows:
            raise ExpertManifoldError(
                "task-expert metrics cursor changed during resume"
            )
    if not 0 <= initial_step < stop_step:
        raise ExpertManifoldError("task-expert resume cursor is outside this segment")
    rows = dataset.task_rows[task.global_task_id]
    sampler = (
        BalancedTwoDomainSampler(
            dataset.domain_rows,
            task_id=task.global_task_id,
            batch_size=batch_size,
            seed=int(config["task_experts"]["sampler"]["seed"]),
        )
        if config.get("schema_version") == RECOVERY_EXPERT_CONFIG_SCHEMA
        else TaskLocalEpochSampler(
            rows,
            task_id=task.global_task_id,
            batch_size=batch_size,
            seed=int(config["task_experts"]["sampler"]["seed"]),
        )
    )
    clip = float(
        config["task_experts"]["optimization"]["optimizer"]["gradient_clip_norm"]
    )
    query_limit = None
    if config.get("schema_version") == COMPOSITE_DISTILLATION_CONFIG_SCHEMA:
        query_limit = int(
            config["task_experts"]["distillation"]["training_epochs"]
        ) * len(rows)
        if total_steps != math.ceil(query_limit / batch_size):
            raise ExpertManifoldError("distillation epoch and step contracts disagree")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step in range(initial_step, stop_step):
        tick = time.monotonic()
        selected = sampler.batch_for_step(step)
        if query_limit is not None:
            remaining = query_limit - step * batch_size
            selected = selected[: min(batch_size, remaining)]
            if not selected:
                raise ExpertManifoldError("distillation query stream ended early")
        samples = [dataset[index] for index in selected]
        batch = (
            dataset.collate(samples)
            if config.get("schema_version") == RECOVERY_EXPERT_CONFIG_SCHEMA
            else default_collate(samples)
        )
        data_seconds = time.monotonic() - tick
        completed = step + 1
        row = _optimize_task_batch(
            policy=policy,
            processor=processor,
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            task=task,
            step=step,
            clip=clip,
            action_queries=(
                min(completed * batch_size, query_limit)
                if query_limit is not None
                else completed * batch_size
            ),
            data_seconds=data_seconds,
            tick=tick,
            started=started,
            mask_action_padding=(
                config.get("schema_version") == RECOVERY_EXPERT_CONFIG_SCHEMA
            ),
        )
        append_jsonl(metrics_path, row)
        metrics_rows += 1
        if completed == 1 or completed % args.log_every == 0:
            print(json.dumps(row, sort_keys=True), flush=True)
        if completed in checkpoint_steps:
            save_task_expert_checkpoint(
                task_dir=task_dir,
                task=task,
                step=completed,
                policy=policy,
                lora_contract=lora_contract,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics_rows=metrics_rows,
            )
    if stop_step not in checkpoint_steps:
        raise ExpertManifoldError("task-expert segment ended outside a checkpoint")
    _task_summary(
        task=task, task_dir=task_dir, step=stop_step, metrics_rows=metrics_rows
    )
    result = {
        "task_ordinal": task.ordinal,
        "global_task_id": task.global_task_id,
        "completed_steps": stop_step,
        "metrics_rows": metrics_rows,
        "max_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    del optimizer, scheduler
    gc.collect()
    return result


def _initial_lora_state(
    config: Mapping[str, Any],
    identity_state: Mapping[str, torch.Tensor],
    lora_contract: Any,
) -> Mapping[str, torch.Tensor]:
    initialization = config["task_experts"].get("initialization")
    if initialization is None:
        return identity_state
    path = REPO_ROOT / str(initialization["adapter"])
    if (
        initialization.get("kind")
        not in {
            "fixed_step1000_composite_adapter_no_optimizer_reuse",
            "fixed_step1000_primitive_adapter_no_optimizer_reuse",
        }
        or not path.is_file()
        or path.stat().st_size != int(initialization["adapter_bytes"])
    ):
        raise ExpertManifoldError("task-expert warm-start adapter changed")
    state = load_file(str(path), device="cpu")
    validate_lora_state(state, lora_contract)
    return state


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ExpertManifoldError("task-expert worker requires one visible CUDA device")
    torch.cuda.set_device(0)
    config = load_task_expert_config(args.config.resolve())
    total_steps, batch_size, checkpoint_steps, stop_step = resolve_runtime(args, config)
    all_tasks = load_train_tasks(config, args.data_root.resolve())
    indices = parse_task_indices(args.task_indices, len(all_tasks))
    tasks = tuple(all_tasks[index] for index in indices)
    if args.mode == "formal":
        validate_formal_task_assignment(config, indices)
    worker_resume_step = worker_stage_resume_step(args.resume, args.output_dir, tasks)
    resume_identity = (
        None if worker_resume_step is not None else parse_resume_task(args.resume)
    )
    if worker_resume_step is not None and not worker_resume_step < stop_step:
        raise ExpertManifoldError("worker stage resume must advance every task")
    if resume_identity is not None and resume_identity[0] not in indices:
        raise ExpertManifoldError("resume task is outside this worker assignment")
    authorities = load_evaluation_authorities(
        authority_path(config, "evaluation_config"), REPO_ROOT
    )
    source = inspect_source_checkpoint(
        authorities,
        args.source_run.resolve(),
        args.checkpoint.resolve(),
        evaluation_mode="formal",
    )
    contract = build_worker_contract(
        args=args,
        config=config,
        tasks=tasks,
        source=source,
        total_steps=total_steps,
        batch_size=batch_size,
        checkpoint_steps=checkpoint_steps,
    )
    publish_worker_contract(args, contract, stop_step)
    seed = int(config["task_experts"]["optimization"]["seed"])
    _seed(seed)
    policy = load_policy(
        Path(source["model_path"]),
        authorities.source_base_config,
        torch.device("cuda:0"),
    )
    lora_contract = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    inject_task_lora(policy, lora_contract)
    policy.train()
    identity_state = {
        name: value.detach().cpu().clone()
        for name, value in task_lora_state_dict(policy).items()
    }
    initial_state = _initial_lora_state(config, identity_state, lora_contract)
    stats = load_stats(
        authorities.source_base_config,
        authorities.source_base_config["data"]["active_task_ids"],
    )
    processor = Pi05LiberoProcessor(
        stats,
        args.tokenizer_path.resolve(),
        int(authorities.source_base_config["features"]["tokenizer_max_length"]),
        "cuda:0",
    )
    dataset = build_dataset(config, tasks, data_root=args.data_root.resolve())
    results = []
    resume_consumed = False
    for task in tasks:
        task_resume = None
        if worker_resume_step is not None:
            task_resume = (
                task_directory(args.output_dir, task)
                / "checkpoints"
                / f"step_{worker_resume_step:08d}"
            )
            resume_consumed = True
        elif resume_identity is not None:
            if task.ordinal < resume_identity[0]:
                completion = task_directory(args.output_dir, task) / "completion.json"
                if not completion.is_file():
                    raise ExpertManifoldError(
                        "resume worker lacks an earlier completed task"
                    )
                results.append(read_json(completion))
                continue
            if task.ordinal == resume_identity[0]:
                task_resume = args.resume.resolve()
                resume_consumed = True
        _seed(seed + task.global_task_id)
        results.append(
            _train_one_task(
                args=args,
                config=config,
                task=task,
                dataset=dataset,
                policy=policy,
                processor=processor,
                lora_contract=lora_contract,
                initial_state=initial_state,
                total_steps=total_steps,
                batch_size=batch_size,
                checkpoint_steps=checkpoint_steps,
                stop_step=stop_step,
                resume=task_resume,
            )
        )
    if args.resume is not None and not resume_consumed:
        raise ExpertManifoldError("task-expert resume checkpoint was not consumed")
    dataset.close()
    write_json_atomic(
        args.output_dir / "worker_summary.json",
        {
            "schema_version": "ember_pi05_task_expert_worker_summary_v1",
            "mode": args.mode,
            "tasks": results,
            "completed_task_count": len(results),
            "selected_stop_step": stop_step,
            "content_hash_policy": "disabled_by_owner",
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train independent PI0.5 rank-16 task experts."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_video_expert_manifold_v1.json",
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-indices", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in ("source_run", "checkpoint", "tokenizer_path", "data_root"):
        path = getattr(args, name).resolve()
        if not path.exists():
            raise ExpertManifoldError(f"missing task-expert runtime path: {path}")
        setattr(args, name, path)
    args.config = args.config.resolve()
    args.output_dir = args.output_dir.resolve()
    args.resume = args.resume.resolve() if args.resume else None
    if args.log_every <= 0:
        raise ExpertManifoldError("task-expert log interval must be positive")
    return args
