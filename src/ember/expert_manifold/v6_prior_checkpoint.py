"""Hashless atomic exact-resume checkpoints for the v6-prior Writer."""

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
from ember.expert_manifold.v6_prior import (
    V6_WRITER_STATE_TENSOR_COUNT,
    configure_v6_prior_trainability,
)
from ember.pi05_source_checkpoint import DistributedContext, read_json, write_json_atomic
from ember.writer.model import CompleteLoRAWriter


V6_PRIOR_CHECKPOINT_SCHEMA = "ember_pi05_v6_prior_writer_checkpoint_v1"
V6_PRIOR_TRAINER_SCHEMA = "ember_pi05_v6_prior_writer_trainer_v1"
V6_PRIOR_RNG_SCHEMA = "ember_pi05_v6_prior_writer_rank_rng_v1"


def _barrier(context: DistributedContext) -> None:
    if context.world_size > 1:
        dist.barrier(device_ids=[context.local_rank])


def _rng_state(context: DistributedContext) -> dict[str, Any]:
    cuda = (
        torch.cuda.get_rng_state(context.device)
        if context.device.type == "cuda"
        else None
    )
    return {
        "schema_version": V6_PRIOR_RNG_SCHEMA,
        "rank": context.rank,
        "world_size": context.world_size,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda,
    }


def _restore_rng(value: Mapping[str, Any], context: DistributedContext) -> None:
    if (
        value.get("schema_version") != V6_PRIOR_RNG_SCHEMA
        or int(value.get("rank", -1)) != context.rank
        or int(value.get("world_size", -1)) != context.world_size
        or (context.device.type == "cuda") != (value.get("torch_cuda") is not None)
    ):
        raise ExpertManifoldError("v6-prior rank RNG state changed")
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"].cpu())
    if context.device.type == "cuda":
        torch.cuda.set_rng_state(value["torch_cuda"].cpu(), context.device)


def _move_optimizer_state_(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for values in optimizer.state.values():
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                values[name] = value.to(device)


def save_v6_prior_checkpoint(
    *,
    output_dir: Path,
    macro: int,
    writer: CompleteLoRAWriter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    context: DistributedContext,
    metrics_rows: int,
    cursor_contract: Mapping[str, Any],
    checkpoint_contract: Mapping[str, Any],
) -> Path:
    """Publish one complete checkpoint only at a macro boundary."""

    if (
        macro <= 0
        or metrics_rows != macro
        or int(cursor_contract.get("next_macro", -1)) != macro
    ):
        raise ExpertManifoldError("v6-prior checkpoint cursor changed")
    checkpoints = output_dir / "checkpoints"
    final = checkpoints / f"macro_{macro:08d}"
    temporary = checkpoints / f".macro_{macro:08d}.tmp"
    if context.is_main:
        checkpoints.mkdir(parents=True, exist_ok=True)
        if final.exists() or temporary.exists():
            raise ExpertManifoldError("v6-prior checkpoint already exists")
        temporary.mkdir()
    _barrier(context)
    saved_rng = _rng_state(context)
    rng_name = f"rng_rank_{context.rank:03d}.pt"
    torch.save(saved_rng, temporary / rng_name)
    _barrier(context)
    if context.is_main:
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in writer.state_dict().items()
        }
        if len(state) != V6_WRITER_STATE_TENSOR_COUNT:
            raise ExpertManifoldError("v6-prior checkpoint Writer state changed")
        writer_path = temporary / "writer.safetensors"
        trainer_path = temporary / "trainer.pt"
        save_file(state, str(writer_path))
        torch.save(
            {
                "schema_version": V6_PRIOR_TRAINER_SCHEMA,
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
            **{
                f"rng_rank_{rank:03d}.pt": (
                    temporary / f"rng_rank_{rank:03d}.pt"
                ).stat().st_size
                for rank in range(context.world_size)
            },
        }
        write_json_atomic(
            temporary / "manifest.json",
            {
                "schema_version": V6_PRIOR_CHECKPOINT_SCHEMA,
                "next_macro": macro,
                "metrics_rows": metrics_rows,
                "world_size": context.world_size,
                "cursor_contract": dict(cursor_contract),
                "checkpoint_contract": dict(checkpoint_contract),
                "files": files,
                "content_hash_policy": "disabled_by_owner",
            },
        )
        os.replace(temporary, final)
    _barrier(context)
    _restore_rng(saved_rng, context)
    return final


def load_v6_prior_checkpoint(
    *,
    checkpoint: Path,
    writer: CompleteLoRAWriter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    context: DistributedContext,
    expected_cursor_contract: Mapping[str, Any],
    expected_checkpoint_contract: Mapping[str, Any],
) -> tuple[int, int]:
    """Restore only this method's exact-resume schema, never historical v6 state."""

    checkpoint = checkpoint.resolve()
    manifest = read_json(checkpoint / "manifest.json")
    macro = int(manifest.get("next_macro", -1))
    expected_files = {
        "writer.safetensors",
        "trainer.pt",
        *(f"rng_rank_{rank:03d}.pt" for rank in range(context.world_size)),
    }
    files = manifest.get("files", {})
    if (
        manifest.get("schema_version") != V6_PRIOR_CHECKPOINT_SCHEMA
        or macro <= 0
        or checkpoint.name != f"macro_{macro:08d}"
        or int(manifest.get("world_size", -1)) != context.world_size
        or int(manifest.get("metrics_rows", -1)) != macro
        or manifest.get("cursor_contract") != dict(expected_cursor_contract)
        or manifest.get("checkpoint_contract")
        != dict(expected_checkpoint_contract)
        or set(files) != expected_files
        or manifest.get("content_hash_policy") != "disabled_by_owner"
    ):
        raise ExpertManifoldError("v6-prior checkpoint manifest changed")
    for name, expected_bytes in files.items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(expected_bytes):
            raise ExpertManifoldError("v6-prior checkpoint file changed")
    state = load_file(str(checkpoint / "writer.safetensors"), device="cpu")
    if len(state) != V6_WRITER_STATE_TENSOR_COUNT:
        raise ExpertManifoldError("v6-prior resume Writer state changed")
    writer.load_state_dict(state, strict=True)
    configure_v6_prior_trainability(writer)
    trainer = torch.load(
        checkpoint / "trainer.pt",
        map_location="cpu",
        weights_only=False,
    )
    if (
        trainer.get("schema_version") != V6_PRIOR_TRAINER_SCHEMA
        or int(trainer.get("next_macro", -1)) != macro
        or int(trainer.get("metrics_rows", -1)) != macro
        or int(trainer.get("world_size", -1)) != context.world_size
        or trainer.get("amp_scaler") != {"enabled": False, "state": {}}
    ):
        raise ExpertManifoldError("v6-prior trainer state changed")
    optimizer.load_state_dict(trainer["optimizer"])
    _move_optimizer_state_(optimizer, context.device)
    scheduler.load_state_dict(trainer["scheduler"])
    rng = torch.load(
        checkpoint / f"rng_rank_{context.rank:03d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    _restore_rng(rng, context)
    return macro, int(trainer["metrics_rows"])
