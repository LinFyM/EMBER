"""Atomic exact-resume checkpoints for Writer cold-start training."""

from __future__ import annotations

import json
import os
import random
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from ember.source_base_checkpoint import (
    DistributedContext,
    barrier,
    canonical_hash,
    restore_rng,
    sha256_file,
    write_json_atomic,
)
from ember.writer.data import MixedTaskBatchSampler
from ember.writer.model import CompleteLoRAWriter, WriterModelError


def _rng_state(context: DistributedContext) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(context.device),
    }


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


def save_writer_checkpoint(
    *,
    output_dir: Path,
    step: int,
    context: DistributedContext,
    writer: CompleteLoRAWriter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler: MixedTaskBatchSampler,
    contract: dict[str, Any],
    mode: str,
) -> Path:
    nonce = _checkpoint_nonce(context)
    temporary = output_dir / "checkpoints" / f".step_{step:08d}.{nonce}.partial"
    final = output_dir / "checkpoints" / f"step_{step:08d}"
    if context.is_main:
        if final.exists():
            raise WriterModelError(f"Writer checkpoint already exists: {final}")
        temporary.mkdir(parents=True)
    barrier(context)

    saved_rng = _rng_state(context)
    torch.save(
        {
            "next_step": step,
            "rank": context.rank,
            "world_size": context.world_size,
            "per_rank_batch_size": sampler.per_rank_batch_size,
            "sampler_seed": sampler.seed,
            "rng": saved_rng,
        },
        temporary / f"rank_{context.rank:02d}_state.pt",
    )
    barrier(context)

    if context.is_main:
        save_file(
            {
                name: value.detach().to(device="cpu").contiguous()
                for name, value in writer.state_dict().items()
            },
            str(temporary / "writer.safetensors"),
        )
        torch.save(
            {
                "next_step": step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "amp_scaler": {"enabled": False, "state": {}},
                "contract_sha256": canonical_hash(contract),
            },
            temporary / "trainer_state.pt",
        )
        coverage = sampler.coverage_for_steps(0, step)
        if mode == "formal" and any(
            len(episodes) != sampler.episodes_per_task
            for episodes in coverage.values()
        ):
            raise WriterModelError(
                "formal Writer checkpoint lacks full declared episode coverage"
            )
        consumed = {
            **sampler.consumed_identity_summary(0, step),
            "declared_task_count": len(coverage),
            "tasks_with_signal": sum(bool(episodes) for episodes in coverage.values()),
            "min_episodes_per_task": min(map(len, coverage.values())),
            "max_episodes_per_task": max(map(len, coverage.values())),
            "next_step": step,
        }
        files = {}
        for path in sorted(value for value in temporary.rglob("*") if value.is_file()):
            relative = str(path.relative_to(temporary))
            files[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        write_json_atomic(
            temporary / "checkpoint_manifest.json",
            {
                "contract_sha256": canonical_hash(contract),
                "consumed": consumed,
                "files": files,
            },
        )
        os.replace(temporary, final)
        write_json_atomic(
            output_dir / "latest_checkpoint.json",
            {"path": str(final), "step": step},
        )
        print(
            json.dumps({"event": "checkpoint", "path": str(final), **consumed}),
            flush=True,
        )
    barrier(context)
    restore_rng(saved_rng, context)
    return final


def load_writer_checkpoint(
    *,
    checkpoint: Path,
    context: DistributedContext,
    writer: CompleteLoRAWriter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler_seed: int,
    per_rank_batch_size: int,
    contract_sha256: str,
) -> tuple[int, dict[str, Any]]:
    trainer = torch.load(
        checkpoint / "trainer_state.pt",
        map_location=context.device,
        weights_only=False,
    )
    if trainer.get("contract_sha256") != contract_sha256:
        raise WriterModelError("Writer resume contract changed")
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
        int(trainer["next_step"]),
        context.rank,
        context.world_size,
        per_rank_batch_size,
        sampler_seed,
    )
    actual = (
        int(rank_state["next_step"]),
        int(rank_state["rank"]),
        int(rank_state["world_size"]),
        int(rank_state["per_rank_batch_size"]),
        int(rank_state["sampler_seed"]),
    )
    if actual != expected:
        raise WriterModelError("Writer rank resume state changed")
    return int(trainer["next_step"]), rank_state["rng"]
