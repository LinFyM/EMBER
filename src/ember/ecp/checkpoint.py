"""Hashless full-state checkpoints with exact restore by default."""

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
    training_state: Mapping[str, Any] | None = None,
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
                "training_state": dict(training_state) if training_state is not None else None,
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


def _checkpoint_world_size(
    manifest: Mapping[str, Any],
    *,
    stage: str,
    macro: int,
    context: DistributedContext,
    run_contract_schema: str,
    allow_world_size_change: bool,
) -> int:
    world_size = int(manifest.get("world_size", -1))
    expected_files = {
        "ecp.safetensors",
        "trainer_state.pt",
        *(f"rank_{rank:02d}_state.pt" for rank in range(world_size)),
    }
    if (
        manifest.get("schema_version") != ECP_CHECKPOINT_SCHEMA
        or manifest.get("stage") != stage
        or int(manifest.get("next_macro", -1)) != macro
        or world_size <= 0
        or (not allow_world_size_change and world_size != context.world_size)
        or manifest.get("run_contract_schema") != run_contract_schema
        or set(manifest.get("files", {})) != expected_files
    ):
        raise ValueError("ECP checkpoint authority changed")
    return world_size


def _load_rank_states(checkpoint: Path, *, stage: str, macro: int, world_size: int) -> list[dict[str, Any]]:
    states = [
        torch.load(checkpoint / f"rank_{rank:02d}_state.pt", map_location="cpu", weights_only=False)
        for rank in range(world_size)
    ]
    for rank, state in enumerate(states):
        if (
            state.get("schema_version") != ECP_CHECKPOINT_SCHEMA
            or state.get("stage") != stage
            or int(state.get("rank", -1)) != rank
            or int(state.get("world_size", -1)) != world_size
            or int(state.get("next_macro", -1)) != macro
        ):
            raise ValueError("ECP checkpoint cursor changed")
    return states


def _validate_trainer_state(
    trainer: Mapping[str, Any], *, stage: str, macro: int, expected_sampler_state: Mapping[str, Any] | None
) -> None:
    if (
        trainer.get("schema_version") != ECP_CHECKPOINT_SCHEMA
        or trainer.get("stage") != stage
        or int(trainer.get("next_macro", -1)) != macro
        or (expected_sampler_state is not None and trainer.get("sampler_state") != dict(expected_sampler_state))
    ):
        raise ValueError("ECP checkpoint cursor changed")


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
    restored_state: dict[str, Any] | None = None,
    allow_world_size_change: bool = False,
) -> tuple[int, int]:
    macro = checkpoint_macro(checkpoint)
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    checkpoint_world_size = _checkpoint_world_size(
        manifest, stage=stage, macro=macro, context=context, run_contract_schema=run_contract_schema,
        allow_world_size_change=allow_world_size_change,
    )
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
    _validate_trainer_state(trainer, stage=stage, macro=macro, expected_sampler_state=expected_sampler_state)
    rank_states = _load_rank_states(
        checkpoint, stage=stage, macro=macro, world_size=checkpoint_world_size
    )
    optimizer.load_state_dict(trainer["optimizer"])
    scheduler.load_state_dict(trainer["scheduler"])
    if context.rank < checkpoint_world_size:
        restore_rng(rank_states[context.rank]["rng"], context)
    if restored_state is not None:
        restored_state.update(sampler_state=trainer.get("sampler_state"), training_state=trainer.get("training_state"))
        if checkpoint_world_size != context.world_size:
            restored_state["topology_resume"] = {
                "checkpoint_world_size": checkpoint_world_size,
                "current_world_size": context.world_size,
                "checkpoint_rng_ranks": list(range(min(checkpoint_world_size, context.world_size))),
                "fresh_seeded_ranks": list(range(checkpoint_world_size, context.world_size)),
            }
    return macro, int(trainer["metrics_rows"])
