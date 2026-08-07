"""Atomic exact-resume checkpoints for the distributed topological Writer."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from ember.expert_manifold.contract import ExpertManifoldError
from ember.pi05_source_checkpoint import DistributedContext, read_json, write_json_atomic


WRITER_CHECKPOINT_SCHEMA = "ember_pi05_expert_manifold_writer_checkpoint_v1"
WRITER_TRAINER_SCHEMA = "ember_pi05_expert_manifold_writer_trainer_v1"
WRITER_RNG_SCHEMA = "ember_pi05_expert_manifold_writer_rng_v1"


def _barrier(context: DistributedContext) -> None:
    if context.world_size > 1:
        dist.barrier(device_ids=[context.local_rank])


def _rng_state(context: DistributedContext) -> dict[str, Any]:
    return {
        "schema_version": WRITER_RNG_SCHEMA,
        "rank": context.rank,
        "world_size": context.world_size,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(context.device),
    }


def _restore_rng(value: Mapping[str, Any], context: DistributedContext) -> None:
    if (
        value.get("schema_version") != WRITER_RNG_SCHEMA
        or int(value.get("rank", -1)) != context.rank
        or int(value.get("world_size", -1)) != context.world_size
    ):
        raise ExpertManifoldError("topological Writer rank RNG state changed")
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"].cpu())
    torch.cuda.set_rng_state(value["torch_cuda"].cpu(), context.device)


def _writer_module(writer: torch.nn.Module) -> torch.nn.Module:
    return writer.module if hasattr(writer, "module") else writer


def save_writer_checkpoint(
    *,
    output_dir: Path,
    macro: int,
    writer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    context: DistributedContext,
    metrics_rows: int,
) -> Path:
    if macro <= 0 or metrics_rows != macro:
        raise ExpertManifoldError("topological Writer checkpoint cursor changed")
    checkpoints = output_dir / "checkpoints"
    final = checkpoints / f"macro_{macro:08d}"
    temporary = checkpoints / f".macro_{macro:08d}.tmp"
    if context.is_main:
        checkpoints.mkdir(parents=True, exist_ok=True)
        if final.exists() or temporary.exists():
            raise ExpertManifoldError("topological Writer checkpoint already exists")
        temporary.mkdir()
    _barrier(context)
    rng_name = f"rng_rank_{context.rank:03d}.pt"
    torch.save(_rng_state(context), temporary / rng_name)
    _barrier(context)
    if context.is_main:
        writer_path = temporary / "writer.safetensors"
        trainer_path = temporary / "trainer.pt"
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in _writer_module(writer).state_dict().items()
            },
            str(writer_path),
        )
        torch.save(
            {
                "schema_version": WRITER_TRAINER_SCHEMA,
                "next_macro": macro,
                "metrics_rows": metrics_rows,
                "world_size": context.world_size,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "amp_scaler": {"enabled": False, "state": {}},
            },
            trainer_path,
        )
        files = {
            "writer.safetensors": writer_path.stat().st_size,
            "trainer.pt": trainer_path.stat().st_size,
        }
        for rank in range(context.world_size):
            name = f"rng_rank_{rank:03d}.pt"
            files[name] = (temporary / name).stat().st_size
        write_json_atomic(
            temporary / "manifest.json",
            {
                "schema_version": WRITER_CHECKPOINT_SCHEMA,
                "next_macro": macro,
                "metrics_rows": metrics_rows,
                "world_size": context.world_size,
                "files": files,
                "content_hash_policy": "disabled_by_owner",
            },
        )
        os.replace(temporary, final)
    _barrier(context)
    return final


def load_writer_checkpoint(
    *,
    checkpoint: Path,
    writer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    context: DistributedContext,
) -> tuple[int, int]:
    checkpoint = checkpoint.resolve()
    manifest = read_json(checkpoint / "manifest.json")
    if (
        manifest.get("schema_version") != WRITER_CHECKPOINT_SCHEMA
        or int(manifest.get("world_size", -1)) != context.world_size
        or manifest.get("content_hash_policy") != "disabled_by_owner"
    ):
        raise ExpertManifoldError("topological Writer checkpoint manifest changed")
    for name, expected_bytes in manifest.get("files", {}).items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(expected_bytes):
            raise ExpertManifoldError("topological Writer checkpoint file changed")
    state = load_file(str(checkpoint / "writer.safetensors"), device="cpu")
    _writer_module(writer).load_state_dict(state, strict=True)
    trainer = torch.load(checkpoint / "trainer.pt", map_location="cpu", weights_only=False)
    macro = int(manifest.get("next_macro", -1))
    if (
        trainer.get("schema_version") != WRITER_TRAINER_SCHEMA
        or int(trainer.get("next_macro", -2)) != macro
        or int(trainer.get("metrics_rows", -2)) != int(manifest.get("metrics_rows", -1))
        or int(trainer.get("world_size", -1)) != context.world_size
    ):
        raise ExpertManifoldError("topological Writer trainer state changed")
    optimizer.load_state_dict(trainer["optimizer"])
    for state_values in optimizer.state.values():
        for name, value in state_values.items():
            if isinstance(value, torch.Tensor):
                state_values[name] = value.to(context.device)
    scheduler.load_state_dict(trainer["scheduler"])
    rng = torch.load(
        checkpoint / f"rng_rank_{context.rank:03d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    _restore_rng(rng, context)
    return macro, int(trainer["metrics_rows"])
