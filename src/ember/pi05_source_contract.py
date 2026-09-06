"""Launch contract, metrics cursor, and resume validation for PI05 source training."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist

from ember.pi05_source_checkpoint import (
    DistributedContext,
    Pi05SourceTrainingError,
    canonical_hash,
    git_state,
    restore_rng,
    sha256_file,
    verify_checkpoint,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def reconcile_metrics(
    path: Path,
    optimizer_step: int,
    expected_rows: int,
    *,
    cursor_key: str = "optimizer_step",
    packet_label: str = "",
) -> int:
    if not path.exists():
        if optimizer_step:
            raise Pi05SourceTrainingError("resume checkpoint has no retained metrics")
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    retained = [row for row in parsed if int(row[cursor_key]) <= optimizer_step]
    orphaned = [row for row in parsed if int(row[cursor_key]) > optimizer_step]
    if len(retained) != expected_rows:
        raise Pi05SourceTrainingError("metrics row count differs from checkpoint cursor")
    if orphaned:
        prefix = f"{packet_label}_" if packet_label else ""
        packet = path.parent / "failure_packets" / f"{prefix}orphaned_after_step_{optimizer_step:08d}.jsonl"
        packet.parent.mkdir(parents=True, exist_ok=True)
        packet.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in orphaned),
            encoding="utf-8",
        )
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in retained),
            encoding="utf-8",
        )
    return len(retained)


def build_contract(
    *,
    args: argparse.Namespace,
    config_path: Path,
    config: dict[str, Any],
    context: DistributedContext,
    trainable: dict[str, Any],
    task_ids: Sequence[int],
    ema_enabled: bool,
    optimizer_steps: int,
    micro_batch_size: int,
    gradient_accumulation: int,
    checkpoint_interval: int,
    asset_validation: dict[str, Any],
) -> dict[str, Any]:
    git = git_state()
    local_topology = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "cuda_device": str(context.device),
        "numa_node": context.numa_node,
        "cpu_affinity": list(context.cpu_affinity or ()),
    }
    rank_topology: list[Any] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(rank_topology, local_topology)
    else:
        rank_topology[0] = local_topology
    return {
        "schema_version": "ember_pi05_source_launch_v1",
        "mode": args.mode,
        "git": {"branch": git["branch"], "commit": git["commit"]},
        "host": socket.gethostname(),
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": sha256_file(config_path),
        "authorities": config["authorities"],
        "models": config["models"],
        "asset_validation": asset_validation,
        "features": config["features"],
        "optimization": config["optimization"],
        "runtime": {
            "world_size": context.world_size,
            "one_policy_cuda_process_per_rank": True,
            "micro_batch_size_per_rank": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation,
            "effective_global_batch_size": context.world_size
            * micro_batch_size
            * gradient_accumulation,
            "optimizer_steps": optimizer_steps,
            "micro_steps": optimizer_steps * gradient_accumulation,
            "checkpoint_interval": checkpoint_interval,
            "num_workers_per_rank": args.num_workers,
            "ema_enabled": ema_enabled,
            "task_limit": args.task_limit,
            "data_sha256_verified": not args.skip_data_sha,
            "rank_topology": rank_topology,
        },
        "task_ids": list(task_ids),
        "trainable": trainable,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "lerobot": importlib.metadata.version("lerobot"),
        },
    }


def validate_formal(
    args: argparse.Namespace,
    config: dict[str, Any],
    context: DistributedContext,
    *,
    ema_enabled: bool,
    optimizer_steps: int,
    micro_batch_size: int,
    gradient_accumulation: int,
    checkpoint_interval: int,
) -> None:
    if args.mode != "formal":
        return
    formal = config["formal_run"]
    git = git_state()
    failures = []
    if not formal.get("locked"):
        failures.append("formal config is not locked after profiling")
    if context.numa_node is None or not context.cpu_affinity:
        failures.append("formal launch requires GPU-local NUMA affinity")
    if git["dirty_paths"]:
        failures.append("formal launch requires a clean worktree")
    if args.resume is None and git["commit"] != git["origin_main"]:
        failures.append("formal launch commit must already be pushed to origin/main")
    expected = {
        "world_size": context.world_size,
        "optimizer_steps": optimizer_steps,
        "micro_batch_size_per_rank": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "checkpoint_interval": checkpoint_interval,
        "ema_enabled": ema_enabled,
    }
    for key, observed in expected.items():
        if formal.get(key) != observed:
            failures.append(f"formal {key} differs from sealed config")
    if args.task_limit is not None or args.skip_data_sha:
        failures.append("formal launch must use all tasks and verify every HDF5 hash")
    if args.stop_after_optimizer_step is not None:
        failures.append("formal launch cannot stop before its sealed horizon")
    if context.world_size * micro_batch_size * gradient_accumulation != 256:
        failures.append("formal effective global batch must equal the official 256")
    if failures:
        raise Pi05SourceTrainingError("; ".join(failures))


def resolve_runtime(
    args: argparse.Namespace, config: dict[str, Any], context: DistributedContext
) -> tuple[int, int, int, int, bool]:
    formal = config["formal_run"]
    profile = config["profile_defaults"]
    source = formal if args.mode == "formal" else profile
    optimizer_steps = args.optimizer_steps or int(source["optimizer_steps"])
    micro_batch = args.micro_batch_size or int(source["micro_batch_size_per_rank"])
    accumulation = args.gradient_accumulation or int(source["gradient_accumulation_steps"])
    checkpoint_interval = (
        args.checkpoint_interval
        if args.checkpoint_interval is not None
        else int(source["checkpoint_interval"])
    )
    ema_enabled = bool(source["ema_enabled"])
    if args.ema == "on":
        ema_enabled = True
    elif args.ema == "off":
        ema_enabled = False
    if min(optimizer_steps, micro_batch, accumulation) <= 0 or checkpoint_interval < 0:
        raise Pi05SourceTrainingError("invalid optimizer, batch, accumulation, or checkpoint request")
    validate_formal(
        args,
        config,
        context,
        ema_enabled=ema_enabled,
        optimizer_steps=optimizer_steps,
        micro_batch_size=micro_batch,
        gradient_accumulation=accumulation,
        checkpoint_interval=checkpoint_interval,
    )
    return optimizer_steps, micro_batch, accumulation, checkpoint_interval, ema_enabled


def load_resume(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    contract_sha256: str,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    gradient_accumulation: int,
) -> tuple[int, int, int]:
    if args.resume is None:
        return 0, 0, 0
    payload: list[Any] = [None]
    if context.is_main:
        try:
            payload[0] = {"state": verify_checkpoint(args.resume, contract_sha256)}
        except Exception as error:  # propagated coherently to every rank
            payload[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=context.device)
    if payload[0].get("error"):
        raise Pi05SourceTrainingError(payload[0]["error"])
    state = payload[0]["state"]
    optimizer.load_state_dict(
        torch.load(args.resume / "optimizer.pt", map_location=context.device, weights_only=False)
    )
    scheduler.load_state_dict(
        torch.load(args.resume / "scheduler.pt", map_location="cpu", weights_only=False)
    )
    rank_state = torch.load(
        args.resume / f"rank_{context.rank:02d}_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    optimizer_step = int(state["optimizer_step"])
    micro_step = int(state["micro_step"])
    if (
        rank_state["rank"] != context.rank
        or rank_state["world_size"] != context.world_size
        or int(rank_state["optimizer_step"]) != optimizer_step
        or int(rank_state["micro_step"]) != micro_step
        or micro_step != optimizer_step * gradient_accumulation
    ):
        raise Pi05SourceTrainingError("resume rank, world size, or sampler cursor changed")
    restore_rng(rank_state["rng"], context)
    return optimizer_step, micro_step, int(state["metrics_rows"])
