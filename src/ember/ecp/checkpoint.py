"""Hashless exact-resume checkpoints shared by ECP training stages."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from ember.pi05_source_checkpoint import (
    DistributedContext,
    barrier,
    capture_rng,
    read_json,
    restore_rng,
    write_json_atomic,
)


ECP_CHECKPOINT_SCHEMA = "ember_ecp_checkpoint_v1"
_CHECKPOINT_NAME = re.compile(r"macro_([0-9]{8})")


def checkpoint_macro(path: Path | None) -> int:
    if path is None:
        return 0
    match = _CHECKPOINT_NAME.fullmatch(path.name)
    if match is None or path.parent.name != "checkpoints":
        raise ValueError("ECP resume path is not a macro checkpoint")
    return int(match.group(1))


def save_ecp_checkpoint(
    *,
    output_dir: Path,
    macro: int,
    stage: str,
    context: DistributedContext,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    run_contract_schema: str,
    metrics_rows: int,
    sampler_state: Mapping[str, Any] | None = None,
) -> Path:
    checkpoints = output_dir / "checkpoints"
    partial = checkpoints / f".macro_{macro:08d}.partial"
    final = checkpoints / f"macro_{macro:08d}"
    if context.is_main:
        checkpoints.mkdir(parents=True, exist_ok=True)
        if partial.exists() or final.exists():
            raise ValueError(f"ECP checkpoint already exists: {final}")
        partial.mkdir()
    barrier(context)
    torch.save(
        {
            "schema_version": ECP_CHECKPOINT_SCHEMA,
            "stage": stage,
            "rank": context.rank,
            "world_size": context.world_size,
            "next_macro": macro,
            "rng": capture_rng(context),
        },
        partial / f"rank_{context.rank:02d}_state.pt",
    )
    barrier(context)
    if context.is_main:
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in model.state_dict().items()
            },
            str(partial / "ecp.safetensors"),
        )
        torch.save(
            {
                "schema_version": ECP_CHECKPOINT_SCHEMA,
                "stage": stage,
                "next_macro": macro,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics_rows": metrics_rows,
                "sampler_state": dict(sampler_state) if sampler_state is not None else None,
                "scaler": None,  # BF16 does not use FP16 gradient scaling.
            },
            partial / "trainer_state.pt",
        )
        files = {
            path.name: {"bytes": path.stat().st_size}
            for path in sorted(partial.iterdir())
            if path.is_file()
        }
        write_json_atomic(
            partial / "checkpoint_manifest.json",
            {
                "schema_version": ECP_CHECKPOINT_SCHEMA,
                "stage": stage,
                "next_macro": macro,
                "world_size": context.world_size,
                "run_contract_schema": run_contract_schema,
                "files": files,
            },
        )
        os.replace(partial, final)
        write_json_atomic(
            output_dir / "latest_checkpoint.json",
            {"path": str(final), "macro": macro, "stage": stage},
        )
    barrier(context)
    return final


def load_ecp_checkpoint(
    *,
    checkpoint: Path,
    stage: str,
    context: DistributedContext,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    run_contract_schema: str,
    expected_sampler_state: Mapping[str, Any] | None = None,
) -> tuple[int, int]:
    macro = checkpoint_macro(checkpoint)
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    expected_files = {
        "ecp.safetensors",
        "trainer_state.pt",
        *(f"rank_{rank:02d}_state.pt" for rank in range(context.world_size)),
    }
    if (
        manifest.get("schema_version") != ECP_CHECKPOINT_SCHEMA
        or manifest.get("stage") != stage
        or int(manifest.get("next_macro", -1)) != macro
        or int(manifest.get("world_size", -1)) != context.world_size
        or manifest.get("run_contract_schema") != run_contract_schema
        or set(manifest.get("files", {})) != expected_files
    ):
        raise ValueError("ECP checkpoint authority changed")
    for name, record in manifest["files"].items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"ECP checkpoint file changed: {name}")
    model.load_state_dict(
        load_file(str(checkpoint / "ecp.safetensors"), device=str(context.device)),
        strict=True,
    )
    trainer = torch.load(
        checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False
    )
    rank_state = torch.load(
        checkpoint / f"rank_{context.rank:02d}_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    if (
        trainer.get("schema_version") != ECP_CHECKPOINT_SCHEMA
        or trainer.get("stage") != stage
        or rank_state.get("schema_version") != ECP_CHECKPOINT_SCHEMA
        or rank_state.get("stage") != stage
        or int(rank_state.get("rank", -1)) != context.rank
        or int(rank_state.get("world_size", -1)) != context.world_size
        or int(trainer.get("next_macro", -1)) != macro
        or int(rank_state.get("next_macro", -1)) != macro
        or (expected_sampler_state is not None and trainer.get("sampler_state") != dict(expected_sampler_state))
    ):
        raise ValueError("ECP checkpoint cursor changed")
    optimizer.load_state_dict(trainer["optimizer"])
    scheduler.load_state_dict(trainer["scheduler"])
    restore_rng(rank_state["rng"], context)
    return macro, int(trainer["metrics_rows"])
