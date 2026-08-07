"""Distributed task-complete training for the video-conditioned topological Writer."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel

from ember.expert_manifold.contract import (
    REPO_ROOT,
    ExpertManifoldError,
    authority_path,
    load_expert_manifold_config,
)
from ember.expert_manifold.evaluation import inspect_task_expert_bank
from ember.expert_manifold.feature_cache import inspect_feature_cache
from ember.expert_manifold.model import (
    VideoConditionedTopologicalWriter,
    topological_reconstruction_loss,
)
from ember.expert_manifold.sampler import TaskLocalEpochSampler
from ember.expert_manifold.writer_checkpoint import (
    load_writer_checkpoint,
    save_writer_checkpoint,
)
from ember.lora import identity_lora_state
from ember.pi05_eval_contract import (
    git_state,
    inspect_source_checkpoint,
    load_evaluation_authorities,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import DistributedContext, read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import (
    initialize_deferred_process_group,
    initialize_distributed,
    seed_everything,
)


WRITER_RUN_SCHEMA = "ember_pi05_expert_manifold_writer_launch_v1"


@dataclass(frozen=True)
class LocalWriterData:
    ordinals: tuple[int, ...]
    global_task_ids: tuple[int, ...]
    features: torch.Tensor
    targets: torch.Tensor
    samplers: tuple[TaskLocalEpochSampler, ...]


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_macros: int,
    warmup_macros: int,
    peak_lr: float,
    decay_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    if not 0 < warmup_macros < total_macros or not 0 < decay_lr <= peak_lr:
        raise ExpertManifoldError("topological Writer scheduler changed")

    def factor(macro: int) -> float:
        if macro < warmup_macros:
            return (macro + 1) / warmup_macros
        progress = min(
            1.0, (macro - warmup_macros) / (total_macros - warmup_macros)
        )
        value = decay_lr + 0.5 * (peak_lr - decay_lr) * (
            1.0 + math.cos(math.pi * progress)
        )
        return value / peak_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _runtime(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> tuple[int, int, tuple[int, ...], int]:
    meta = config["meta_training"]
    formal = meta["formal_run"]
    selected = meta["profile_defaults"] if args.mode == "profile" else formal
    total_macros = int(selected["total_macros"])
    scheduler_total = int(meta["profile_defaults"]["scheduler_total_macros"])
    microbatch = int(args.microbatch or selected["physical_microbatch_per_rank"])
    checkpoints = tuple(int(value) for value in selected["checkpoint_macros"])
    stop_macro = int(args.stop_after_macro or total_macros)
    valid = (
        int(meta.get("task_count", -1)) == 24
        and int(meta.get("videos_per_task_per_macro", -1)) == 1
        and scheduler_total == int(formal["total_macros"])
        and tuple(sorted(set(checkpoints))) == checkpoints
        and checkpoints[-1] == total_macros
        and stop_macro in checkpoints
        and 24 % context.world_size == 0
        and microbatch > 0
        and (24 // context.world_size) % microbatch == 0
    )
    if args.mode == "formal":
        valid = (
            valid
            and formal.get("status") == "sealed"
            and context.world_size == int(formal["expected_world_size"])
            and 24 // context.world_size == int(formal["tasks_per_rank"])
            and microbatch == int(formal["physical_microbatch_per_rank"])
            and int(formal.get("selected_expert_step", -1)) == args.expert_step
        )
        state = git_state(REPO_ROOT)
        valid = valid and not state["dirty_paths"] and state["commit"] == state["upstream_commit"]
    if not valid:
        raise ExpertManifoldError("topological Writer runtime differs from its sealed contract")
    return scheduler_total, microbatch, checkpoints, stop_macro


def _source(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    authorities = load_evaluation_authorities(
        authority_path(config, "evaluation_config"), REPO_ROOT
    )
    return inspect_source_checkpoint(
        authorities,
        args.source_run,
        args.checkpoint,
        evaluation_mode="formal",
    )


def _task_keys(config: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    manifest = read_json(authority_path(config, "target_data_manifest"))
    rows = [row for row in manifest["tasks"] if row["split_role"] == "train"]
    rows.sort(key=lambda row: int(row["global_task_id"]))
    if len(rows) != 24:
        raise ExpertManifoldError("topological Writer did not resolve train24")
    return tuple((str(row["suite"]), int(row["task_id"])) for row in rows)


def _build_model_and_data(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    source: Mapping[str, Any],
) -> tuple[
    VideoConditionedTopologicalWriter,
    LocalWriterData,
    dict[str, Any],
    dict[str, Any],
]:
    expert = inspect_task_expert_bank(
        config_path=args.config,
        bank_root=args.expert_bank_root,
        step=args.expert_step,
        source=source,
        task_keys=_task_keys(config),
        evaluation_role="development_train",
        require_formal=True,
    )
    cache = inspect_feature_cache(args.config, args.feature_cache_root, source=source)
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    template = identity_lora_state(lora)
    writer_config = config["topological_writer"]
    writer = VideoConditionedTopologicalWriter(
        contract=lora,
        template_state=template,
        phase_slots=int(config["video_features"]["phase_slots"]),
        feature_width=int(config["video_features"]["feature_width"]),
        memory_width=int(writer_config["memory_width"]),
        attention_heads=int(writer_config["attention_heads"]),
        axial_blocks=int(writer_config["axial_blocks"]),
        chunk_width=int(writer_config["chunk_width"]),
    ).to(context.device)
    ownership = tuple(range(context.rank, 24, context.world_size))
    expert_by_ordinal = {int(row["ordinal"]): row for row in expert["tasks"]}
    cache_by_ordinal = {int(row["task_ordinal"]): row for row in cache["tasks"]}
    targets = []
    features = []
    global_task_ids = []
    seed = int(config["meta_training"]["optimization"]["seed"])
    samplers = []
    for ordinal in ownership:
        expert_row = expert_by_ordinal[ordinal]
        cache_row = cache_by_ordinal[ordinal]
        if int(expert_row["global_task_id"]) != int(cache_row["global_task_id"]):
            raise ExpertManifoldError("expert and video feature task identity changed")
        state = load_file(
            str(Path(expert_row["checkpoint"]) / "adapter.safetensors"),
            device="cpu",
        )
        targets.append(writer.layout.tokenize(state, template))
        feature_state = load_file(str(cache_row["features"]["path"]), device="cpu")
        value = feature_state["video_innovation"]
        if value.shape != (50, 16, 2048) or value.dtype != torch.bfloat16:
            raise ExpertManifoldError("cached video innovation changed")
        features.append(value)
        global_task_id = int(expert_row["global_task_id"])
        global_task_ids.append(global_task_id)
        samplers.append(
            TaskLocalEpochSampler(
                range(50), task_id=global_task_id, batch_size=1, seed=seed
            )
        )
    local = LocalWriterData(
        ordinals=ownership,
        global_task_ids=tuple(global_task_ids),
        features=torch.stack(features).to(context.device),
        targets=torch.stack(targets).to(context.device, dtype=torch.float32),
        samplers=tuple(samplers),
    )
    return writer, local, expert, cache


def _initialize_scale_prior(
    writer: VideoConditionedTopologicalWriter,
    targets: torch.Tensor,
    context: DistributedContext,
) -> None:
    mask = writer.valid_value_mask[None, :, None, :].to(
        device=targets.device, dtype=targets.dtype
    )
    count = mask.sum(dim=(-2, -1)) * writer.layout.rank
    log_scale = torch.sqrt(
        (targets.square() * mask).sum(dim=(-2, -1)) / count + 1e-24
    ).log()
    active = log_scale > math.log(1e-12)
    total = log_scale.masked_fill(~active, 0.0).sum(dim=0)
    active_count = active.sum(dim=0).to(total.dtype)
    if context.world_size > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(active_count, op=dist.ReduceOp.SUM)
    total = torch.where(active_count > 0, total / active_count.clamp_min(1), -12.0)
    total.clamp_(-12.0, 8.0)
    with torch.no_grad():
        writer.chunk_log_scale_offset.copy_(total)


def _optimizer_and_scheduler(
    writer: torch.nn.Module,
    config: Mapping[str, Any],
    total_macros: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    optimization = config["meta_training"]["optimization"]
    optimizer_config = optimization["optimizer"]
    schedule = optimization["scheduler"]
    optimizer = torch.optim.AdamW(
        writer.parameters(),
        lr=float(optimizer_config["peak_lr"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler = _scheduler(
        optimizer,
        total_macros=total_macros,
        warmup_macros=int(schedule["warmup_macros"]),
        peak_lr=float(optimizer_config["peak_lr"]),
        decay_lr=float(schedule["decay_lr"]),
    )
    return optimizer, scheduler


def _contract(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    source: Mapping[str, Any],
    expert: Mapping[str, Any],
    cache: Mapping[str, Any],
    scheduler_total: int,
    microbatch: int,
    checkpoints: Sequence[int],
) -> dict[str, Any]:
    state = git_state(REPO_ROOT)
    return {
        "schema_version": WRITER_RUN_SCHEMA,
        "mode": args.mode,
        "git": {key: state[key] for key in ("branch", "commit")},
        "config": {
            "path": str(args.config),
            "schema": config["schema_version"],
            "bytes": args.config.stat().st_size,
        },
        "source": dict(source),
        "expert_bank": dict(expert),
        "feature_cache": {
            "path": str((args.feature_cache_root / "cache_manifest.json").resolve()),
            "bytes": (args.feature_cache_root / "cache_manifest.json").stat().st_size,
            "schema": cache["schema_version"],
            "training_commit": cache["training_commit"],
            "task_count": cache["task_count"],
            "demo_count": cache["demo_count"],
            "source": cache["source"],
        },
        "method": dict(config["method"]),
        "information_wall": dict(config["information_wall"]),
        "topological_writer": dict(config["topological_writer"]),
        "meta_training": dict(config["meta_training"]),
        "runtime": {
            "host": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "device": torch.cuda.get_device_name(context.device),
            "world_size": context.world_size,
            "rank_task_ownership": [
                list(range(rank, 24, context.world_size))
                for rank in range(context.world_size)
            ],
            "scheduler_total_macros": scheduler_total,
            "physical_microbatch_per_rank": microbatch,
            "checkpoint_macros": list(checkpoints),
            "logical_tasks_per_macro": 24,
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
            "deferred_process_group": True,
        },
        "content_hash_policy": "disabled_by_owner",
    }


def _publish_or_reconcile_contract(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    context: DistributedContext,
) -> None:
    path = args.output_dir / "run_contract.json"
    if context.is_main:
        if args.resume is None:
            if args.output_dir.exists() and any(args.output_dir.iterdir()):
                raise ExpertManifoldError("fresh topological Writer output is not empty")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(path, dict(contract))
        elif not path.is_file() or read_json(path) != contract:
            raise ExpertManifoldError("topological Writer resume contract changed")
        append_jsonl(
            args.output_dir / "invocations.jsonl",
            {
                "argv": os.sys.argv,
                "started_unix": time.time(),
                "resume": str(args.resume) if args.resume else None,
                "requested_stop_after_macro": args.stop_after_macro,
            },
        )
    if context.world_size > 1:
        dist.barrier(device_ids=[context.local_rank])


def _metric_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def train(args: argparse.Namespace) -> None:
    context = initialize_distributed(require_numa=False, defer_process_group=True)
    config = load_expert_manifold_config(args.config)
    scheduler_total, microbatch, checkpoints, stop_macro = _runtime(
        args, config, context
    )
    seed_everything(int(config["meta_training"]["optimization"]["seed"]), context)
    source = _source(args, config)
    writer, local, expert, cache = _build_model_and_data(
        args=args,
        config=config,
        context=context,
        source=source,
    )
    initialize_deferred_process_group(context, rendezvous_root=args.output_dir.parent)
    _initialize_scale_prior(writer, local.targets, context)
    writer = DistributedDataParallel(
        writer,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=True,
    )
    optimizer, scheduler = _optimizer_and_scheduler(writer, config, scheduler_total)
    contract = _contract(
        args=args,
        config=config,
        context=context,
        source=source,
        expert=expert,
        cache=cache,
        scheduler_total=scheduler_total,
        microbatch=microbatch,
        checkpoints=checkpoints,
    )
    _publish_or_reconcile_contract(args, contract, context)
    start_macro = 0
    metrics_rows = 0
    if args.resume is not None:
        start_macro, metrics_rows = load_writer_checkpoint(
            checkpoint=args.resume,
            writer=writer,
            optimizer=optimizer,
            scheduler=scheduler,
            context=context,
        )
    metrics_path = args.output_dir / "metrics.jsonl"
    if context.is_main and _metric_rows(metrics_path) != metrics_rows:
        raise ExpertManifoldError("topological Writer metrics differ from resume cursor")
    if context.world_size > 1:
        dist.barrier(device_ids=[context.local_rank])
    objective = config["meta_training"]["objective"]
    optimizer_config = config["meta_training"]["optimization"]["optimizer"]
    local_count = len(local.ordinals)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(context.device)
    for macro_index in range(start_macro, stop_macro):
        optimizer.zero_grad(set_to_none=True)
        demo_indices = [
            sampler.batch_for_step(macro_index)[0] for sampler in local.samplers
        ]
        metric_sum = torch.zeros(4, dtype=torch.float64, device=context.device)
        step_started = time.monotonic()
        for left in range(0, local_count, microbatch):
            right = left + microbatch
            selected = torch.arange(left, right, device=context.device)
            demos = torch.tensor(
                demo_indices[left:right], dtype=torch.long, device=context.device
            )
            features = local.features[selected, demos]
            targets = local.targets[selected]
            final_microbatch = right == local_count
            synchronization = nullcontext() if final_microbatch else writer.no_sync()
            with synchronization, torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                predicted, predicted_log_scale = writer(
                    features, return_values_with_scale=True
                )
                loss, loss_metrics = topological_reconstruction_loss(
                    predicted.float(),
                    targets,
                    writer.module.valid_value_mask,
                    cosine_weight=float(
                        objective["chunk_rank_direction_cosine_weight"]
                    ),
                    log_scale_weight=float(objective["chunk_log_scale_weight"]),
                    predicted_log_scale=predicted_log_scale.float(),
                )
                scaled = loss * ((right - left) / local_count)
            scaled.backward()
            batch_weight = right - left
            metric_sum += torch.stack(
                (
                    loss.detach().double(),
                    loss_metrics["raw_reconstruction"].double(),
                    loss_metrics["direction"].double(),
                    loss_metrics["log_scale"].double(),
                )
            ) * batch_weight
        grad_norm = torch.nn.utils.clip_grad_norm_(
            writer.parameters(), float(optimizer_config["gradient_clip_norm"])
        )
        applied_lr = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        cursor = macro_index + 1
        if context.world_size > 1:
            dist.all_reduce(metric_sum, op=dist.ReduceOp.SUM)
        metric_sum.div_(24)
        torch.cuda.synchronize(context.device)
        row = {
            "macro": cursor,
            "loss": float(metric_sum[0]),
            "raw_reconstruction": float(metric_sum[1]),
            "direction": float(metric_sum[2]),
            "log_scale": float(metric_sum[3]),
            "gradient_norm_before_clip": float(grad_norm),
            "applied_lr": applied_lr,
            "next_lr": float(optimizer.param_groups[0]["lr"]),
            "step_seconds": time.monotonic() - step_started,
            "elapsed_seconds": time.monotonic() - started,
            "max_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(context.device)),
            "max_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(context.device)),
            "logical_tasks": 24,
            "videos": 24,
        }
        if not all(
            math.isfinite(float(row[name]))
            for name in (
                "loss",
                "raw_reconstruction",
                "direction",
                "log_scale",
                "gradient_norm_before_clip",
            )
        ):
            raise ExpertManifoldError("topological Writer produced nonfinite metrics")
        if context.is_main:
            append_jsonl(metrics_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)
        metrics_rows = cursor
        if cursor in checkpoints:
            save_writer_checkpoint(
                output_dir=args.output_dir,
                macro=cursor,
                writer=writer,
                optimizer=optimizer,
                scheduler=scheduler,
                context=context,
                metrics_rows=metrics_rows,
            )
    if context.is_main:
        write_json_atomic(
            args.output_dir / "completion.json",
            {
                "schema_version": "ember_pi05_expert_manifold_writer_completion_v1",
                "completed_macro": stop_macro,
                "metrics_rows": metrics_rows,
                "expert_step": args.expert_step,
                "world_size": context.world_size,
                "max_cuda_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(context.device)
                ),
                "max_cuda_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(context.device)
                ),
                "content_hash_policy": "disabled_by_owner",
            },
        )
    if context.world_size > 1:
        dist.barrier(device_ids=[context.local_rank])
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_video_expert_manifold_v1.json",
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expert-bank-root", type=Path, required=True)
    parser.add_argument("--expert-step", type=int, required=True)
    parser.add_argument("--feature-cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--stop-after-macro", type=int)
    parser.add_argument("--resume", type=Path)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
        "source_run",
        "checkpoint",
        "expert_bank_root",
        "feature_cache_root",
    ):
        path = getattr(args, name).resolve()
        if not path.exists():
            raise ExpertManifoldError(f"missing topological Writer path: {path}")
        setattr(args, name, path)
    args.output_dir = args.output_dir.resolve()
    args.resume = args.resume.resolve() if args.resume else None
    if args.expert_step <= 0:
        raise ExpertManifoldError("topological Writer expert step must be positive")
    return args
