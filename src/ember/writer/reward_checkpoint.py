"""Cycle checkpoints for causal coefficient transport."""

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


REWARD_CHECKPOINT_SCHEMA = (
    "ember_pi05_v6_lpcp_causal_coefficient_transport_checkpoint_v1"
)
REWARD_DEPLOYMENT_KIND = (
    "v6_lpcp_causal_coefficient_transport_cycle_checkpoint"
)
_CYCLE_NAME = re.compile(r"cycle_([0-9]{8})")


def checkpoint_cycle(path: Path | None) -> int:
    if path is None:
        return 0
    match = _CYCLE_NAME.fullmatch(path.name)
    if match is None or path.parent.name != "checkpoints":
        raise WriterModelError("causal coefficient resume path is not a cycle checkpoint")
    return int(match.group(1))


def save_reward_checkpoint(
    *,
    output_dir: Path,
    cycle: int,
    context: DistributedContext,
    writer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    contract: Mapping[str, Any],
    metrics_rows: int,
) -> Path:
    checkpoints = output_dir / "checkpoints"
    partial = checkpoints / f".cycle_{cycle:08d}.partial"
    final = checkpoints / f"cycle_{cycle:08d}"
    if context.is_main:
        checkpoints.mkdir(parents=True, exist_ok=True)
        if partial.exists() or final.exists():
            raise WriterModelError("causal coefficient checkpoint already exists")
        partial.mkdir()
    barrier(context)
    torch.save(
        {
            "schema_version": REWARD_CHECKPOINT_SCHEMA,
            "rank": context.rank,
            "world_size": context.world_size,
            "next_cycle": cycle,
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
                "schema_version": REWARD_CHECKPOINT_SCHEMA,
                "next_cycle": cycle,
                "optimizer": optimizer.state_dict(),
                "metrics_rows": metrics_rows,
            },
            partial / "trainer_state.pt",
        )
        files = {
            path.name: {"bytes": path.stat().st_size}
            for path in partial.iterdir()
            if path.is_file()
        }
        write_json_atomic(
            partial / "checkpoint_manifest.json",
            {
                "schema_version": REWARD_CHECKPOINT_SCHEMA,
                "next_cycle": cycle,
                "world_size": context.world_size,
                "run_contract_schema": contract["schema_version"],
                "files": files,
            },
        )
        os.replace(partial, final)
        write_json_atomic(
            output_dir / "latest_checkpoint.json",
            {"path": str(final), "cycle": cycle},
        )
    barrier(context)
    return final


def load_reward_checkpoint(
    *,
    checkpoint: Path,
    context: DistributedContext,
    writer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    contract: Mapping[str, Any],
) -> tuple[int, int]:
    cycle = checkpoint_cycle(checkpoint)
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    if (
        manifest.get("schema_version") != REWARD_CHECKPOINT_SCHEMA
        or int(manifest.get("next_cycle", -1)) != cycle
        or int(manifest.get("world_size", -1)) != context.world_size
        or manifest.get("run_contract_schema") != contract["schema_version"]
    ):
        raise WriterModelError("causal coefficient checkpoint manifest changed")
    writer.load_state_dict(
        load_file(str(checkpoint / "writer.safetensors"), device=str(context.device)),
        strict=True,
    )
    trainer = torch.load(
        checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False
    )
    rank = torch.load(
        checkpoint / f"rank_{context.rank:02d}_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    if (
        trainer.get("schema_version") != REWARD_CHECKPOINT_SCHEMA
        or rank.get("schema_version") != REWARD_CHECKPOINT_SCHEMA
        or int(trainer.get("next_cycle", -1)) != cycle
        or int(rank.get("next_cycle", -1)) != cycle
        or int(rank.get("world_size", -1)) != context.world_size
        or int(rank.get("rank", -1)) != context.rank
    ):
        raise WriterModelError("causal coefficient checkpoint cursor changed")
    optimizer.load_state_dict(trainer["optimizer"])
    restore_rng(rank["rng"], context)
    return cycle, int(trainer["metrics_rows"])
