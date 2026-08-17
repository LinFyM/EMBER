"""Hashless macro-boundary exact-resume state for dynamic-K training."""

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
from ember.writer.errors import WriterModelError
CHECKPOINT_SCHEMA = "ember_pi05_layer_matched_memory_program_compiler_checkpoint_v2"
DEPLOYMENT_CHECKPOINT_KIND = "layer_matched_memory_program_compiler_v2_macro_checkpoint"
_CHECKPOINT_NAME = re.compile(r"macro_([0-9]{8})")


def checkpoint_macro(path: Path | None) -> int:
    if path is None:
        return 0
    match = _CHECKPOINT_NAME.fullmatch(path.name)
    if match is None or path.parent.name != "checkpoints":
        raise WriterModelError("dynamic-K resume path is not a macro checkpoint")
    return int(match.group(1))


def save_writer_checkpoint(
    *,
    output_dir: Path,
    macro: int,
    context: DistributedContext,
    writer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    contract: Mapping[str, Any],
    metrics_rows: int,
) -> Path:
    if macro <= 0:
        raise WriterModelError("dynamic-K checkpoint macro must be positive")
    checkpoints = output_dir / "checkpoints"
    partial = checkpoints / f".macro_{macro:08d}.partial"
    final = checkpoints / f"macro_{macro:08d}"
    if context.is_main:
        checkpoints.mkdir(parents=True, exist_ok=True)
        if partial.exists() or final.exists():
            raise WriterModelError(f"dynamic-K checkpoint already exists: {final}")
        partial.mkdir()
    barrier(context)
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA,
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
                for name, value in writer.state_dict().items()
            },
            str(partial / "writer.safetensors"),
        )
        torch.save(
            {
                "schema_version": CHECKPOINT_SCHEMA,
                "next_macro": macro,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics_rows": metrics_rows,
            },
            partial / "trainer_state.pt",
        )
        files = {
            path.name: {"bytes": path.stat().st_size}
            for path in sorted(partial.iterdir())
            if path.is_file()
        }
        expected = {
            "writer.safetensors",
            "trainer_state.pt",
            *(f"rank_{rank:02d}_state.pt" for rank in range(context.world_size)),
        }
        if set(files) != expected:
            raise WriterModelError("dynamic-K checkpoint rank state is incomplete")
        write_json_atomic(
            partial / "checkpoint_manifest.json",
            {
                "schema_version": CHECKPOINT_SCHEMA,
                "next_macro": macro,
                "world_size": context.world_size,
                "run_contract_schema": contract["schema_version"],
                "files": files,
            },
        )
        os.replace(partial, final)
        write_json_atomic(
            output_dir / "latest_checkpoint.json",
            {"path": str(final), "macro": macro},
        )
    barrier(context)
    return final


def load_writer_checkpoint(
    *,
    checkpoint: Path,
    context: DistributedContext,
    writer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    contract: Mapping[str, Any],
) -> tuple[int, int]:
    expected_macro = checkpoint_macro(checkpoint)
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    expected_files = {
        "writer.safetensors",
        "trainer_state.pt",
        *(f"rank_{rank:02d}_state.pt" for rank in range(context.world_size)),
    }
    if (
        manifest.get("schema_version") != CHECKPOINT_SCHEMA
        or int(manifest.get("next_macro", -1)) != expected_macro
        or int(manifest.get("world_size", -1)) != context.world_size
        or manifest.get("run_contract_schema") != contract["schema_version"]
        or set(manifest.get("files", {})) != expected_files
    ):
        raise WriterModelError("dynamic-K checkpoint manifest changed")
    for name, record in manifest["files"].items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise WriterModelError(f"dynamic-K checkpoint file changed: {name}")
    writer.load_state_dict(
        load_file(
            str(checkpoint / "writer.safetensors"), device=str(context.device)
        ),
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
        trainer.get("schema_version") != CHECKPOINT_SCHEMA
        or rank_state.get("schema_version") != CHECKPOINT_SCHEMA
        or int(trainer.get("next_macro", -1)) != expected_macro
        or int(rank_state.get("next_macro", -1)) != expected_macro
        or int(rank_state.get("rank", -1)) != context.rank
        or int(rank_state.get("world_size", -1)) != context.world_size
    ):
        raise WriterModelError("dynamic-K checkpoint cursor changed")
    optimizer.load_state_dict(trainer["optimizer"])
    scheduler.load_state_dict(trainer["scheduler"])
    restore_rng(rank_state["rng"], context)
    return expected_macro, int(trainer["metrics_rows"])


def load_writer_deployment_state_(
    *,
    writer: torch.nn.Module,
    writer_asset: Mapping[str, Any],
    device: torch.device,
) -> None:
    """Load only the Writer weights needed for deployment.

    Deployment deliberately does not deserialize ``trainer_state.pt`` or any
    per-rank RNG state.  The immutable asset record is produced by the
    evaluation inspector before the policy process is constructed.
    """

    state_record = writer_asset.get("writer_state", {})
    path = Path(str(state_record.get("path", "")))
    if (
        writer_asset.get("kind") != DEPLOYMENT_CHECKPOINT_KIND
        or not path.is_file()
        or path.name != "writer.safetensors"
        or path.stat().st_size != int(state_record.get("bytes", -1))
    ):
        raise WriterModelError("dynamic-K deployment Writer state changed")
    state = load_file(str(path), device=str(device))
    try:
        writer.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise WriterModelError(
            "dynamic-K deployment Writer topology changed"
        ) from error
    finally:
        del state
