"""Atomic exact-resume state for source-only Writer reward training."""

from __future__ import annotations

import os
import random
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from ember.source_base_checkpoint import (
    DistributedContext,
    barrier,
    canonical_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.writer.model import WriterModelError
from ember.writer_rl_protocol import schedule_summary


def capture_rng(context: DistributedContext) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state(context.device)
            if context.device.type == "cuda"
            else None
        ),
    }


def restore_rng(state: Mapping[str, Any], context: DistributedContext) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if context.device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], context.device)


def _checkpoint_nonce(context: DistributedContext) -> str:
    nonce = uuid.uuid4().hex
    if context.world_size == 1:
        return nonce
    encoded = torch.zeros(16, dtype=torch.uint8, device=context.device)
    if context.is_main:
        encoded.copy_(
            torch.tensor(
                list(bytes.fromhex(nonce)), dtype=torch.uint8, device=context.device
            )
        )
    dist.broadcast(encoded, src=0)
    return bytes(encoded.cpu().tolist()).hex()


def write_update_ledger_once(
    output_dir: Path, rank: int, update: int, payload: dict[str, Any]
) -> Path:
    path = output_dir / "rollouts" / f"rank_{rank:02d}" / f"update_{update:08d}.json"
    if path.is_file():
        if canonical_hash(read_json(path)) != canonical_hash(payload):
            raise WriterModelError(
                f"replayed Writer-only RL interaction changed: rank={rank} update={update}"
            )
        return path
    write_json_atomic(path, payload)
    return path


def verify_writer_rl_checkpoint(checkpoint: Path) -> dict[str, Any]:
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    files = manifest.get("files", {})
    required = {"writer.safetensors", "trainer_state.pt"}
    if not isinstance(files, dict) or not required.issubset(files):
        raise WriterModelError("Writer-only RL checkpoint manifest is incomplete")
    for name, record in files.items():
        path = checkpoint / name
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise WriterModelError(f"Writer-only RL checkpoint file changed: {name}")
    return manifest


def save_writer_rl_checkpoint(
    *,
    output_dir: Path,
    next_update: int,
    optimizer_updates: int,
    context: DistributedContext,
    writer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    task_ids: Sequence[int],
    rollouts_per_task: int,
    contract: dict[str, Any],
    local_counters: Mapping[str, int],
    formal: bool,
) -> Path:
    nonce = _checkpoint_nonce(context)
    temporary = (
        output_dir
        / "checkpoints"
        / f".update_{next_update:08d}.{nonce}.partial"
    )
    final = output_dir / "checkpoints" / f"update_{next_update:08d}"
    if context.is_main:
        if final.exists():
            raise WriterModelError(f"Writer-only RL checkpoint exists: {final}")
        temporary.mkdir(parents=True)
    barrier(context)

    rng = capture_rng(context)
    torch.save(
        {
            "next_update": next_update,
            "optimizer_updates": optimizer_updates,
            "rank": context.rank,
            "world_size": context.world_size,
            "local_counters": dict(local_counters),
            "rng": rng,
        },
        temporary / f"rank_{context.rank:02d}_state.pt",
    )
    barrier(context)

    if context.is_main:
        consumed = schedule_summary(
            task_ids,
            context.world_size,
            next_update,
            rollouts_per_task,
        )
        if formal and consumed["cycle_slot_cursor"] != 0:
            raise WriterModelError(
                "formal Writer-only RL checkpoint is not a full-task-cycle boundary"
            )
        save_file(
            {
                name: value.detach().to(device="cpu").contiguous()
                for name, value in writer.state_dict().items()
            },
            str(temporary / "writer.safetensors"),
        )
        torch.save(
            {
                "next_update": next_update,
                "optimizer_updates": optimizer_updates,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "amp_scaler": {"enabled": False, "state": {}},
                "contract_sha256": canonical_hash(contract),
            },
            temporary / "trainer_state.pt",
        )
        files = {}
        for path in sorted(value for value in temporary.iterdir() if value.is_file()):
            files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        write_json_atomic(
            temporary / "checkpoint_manifest.json",
            {
                "schema_version": "ember_writer_only_rl_checkpoint_v1",
                "contract_sha256": canonical_hash(contract),
                "consumed": consumed,
                "optimizer_updates": optimizer_updates,
                "files": files,
            },
        )
        os.replace(temporary, final)
        write_json_atomic(
            output_dir / "latest_checkpoint.json",
            {"path": str(final), "next_update": next_update},
        )
    barrier(context)
    restore_rng(rng, context)
    return final


def load_writer_rl_checkpoint(
    *,
    checkpoint: Path,
    context: DistributedContext,
    writer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    contract_sha256: str,
) -> tuple[int, int, dict[str, int], Mapping[str, Any]]:
    manifest = verify_writer_rl_checkpoint(checkpoint)
    trainer = torch.load(
        checkpoint / "trainer_state.pt",
        map_location=context.device,
        weights_only=False,
    )
    if (
        manifest.get("contract_sha256") != contract_sha256
        or trainer.get("contract_sha256") != contract_sha256
    ):
        raise WriterModelError("Writer-only RL resume contract changed")
    writer.load_state_dict(
        load_file(str(checkpoint / "writer.safetensors"), device=str(context.device))
    )
    optimizer.load_state_dict(trainer["optimizer"])
    scheduler.load_state_dict(trainer["scheduler"])
    rank_state = torch.load(
        checkpoint / f"rank_{context.rank:02d}_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    expected = (
        int(trainer["next_update"]),
        int(trainer["optimizer_updates"]),
        context.rank,
        context.world_size,
    )
    actual = (
        int(rank_state.get("next_update", -1)),
        int(rank_state.get("optimizer_updates", -1)),
        int(rank_state.get("rank", -1)),
        int(rank_state.get("world_size", -1)),
    )
    if actual != expected:
        raise WriterModelError("Writer-only RL rank resume state changed")
    counters = {key: int(value) for key, value in rank_state["local_counters"].items()}
    return expected[0], expected[1], counters, rank_state["rng"]
