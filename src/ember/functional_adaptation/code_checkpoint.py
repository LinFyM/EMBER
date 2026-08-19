"""Exact-resume checkpoints for language/video functional-code inference."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic


CHECKPOINT_SCHEMA = "ember_functional_code_writer_checkpoint_v1"
TRAINER_SCHEMA = "ember_functional_code_writer_trainer_v1"
RUN_SCHEMA = "ember_functional_code_writer_run_v1"


def code_writer_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
    }


def save_code_writer_checkpoint(
    *,
    output_dir: Path,
    macro: int,
    world_size: int,
    writer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metrics_rows: int,
    rank_rng_states: Sequence[Mapping[str, Any]],
) -> Path:
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    final = checkpoints / f"macro_{macro:08d}"
    temporary = checkpoints / f".macro_{macro:08d}.tmp-{os.getpid()}"
    if (
        final.exists()
        or temporary.exists()
        or macro <= 0
        or metrics_rows != macro
        or len(rank_rng_states) != world_size
    ):
        raise ValueError("invalid functional-code checkpoint target")
    temporary.mkdir()
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in writer.state_dict().items()
    }
    weights = temporary / "writer.safetensors"
    trainer_path = temporary / "trainer.pt"
    save_file(state, str(weights))
    torch.save(
        {
            "schema_version": TRAINER_SCHEMA,
            "macro": macro,
            "world_size": world_size,
            "metrics_rows": metrics_rows,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rank_rng": list(rank_rng_states),
        },
        trainer_path,
    )
    write_json_atomic(
        temporary / "manifest.json",
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "macro": macro,
            "world_size": world_size,
            "metrics_rows": metrics_rows,
            "files": {
                "writer.safetensors": weights.stat().st_size,
                "trainer.pt": trainer_path.stat().st_size,
            },
            "content_hash_policy": "disabled_by_owner",
        },
    )
    os.replace(temporary, final)
    return final


def load_code_writer_checkpoint(
    *,
    checkpoint: Path,
    rank: int,
    world_size: int,
    writer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, int]:
    manifest = read_json(checkpoint / "manifest.json")
    if (
        manifest.get("schema_version") != CHECKPOINT_SCHEMA
        or int(manifest.get("world_size", -1)) != world_size
        or not 0 <= rank < world_size
        or manifest.get("content_hash_policy") != "disabled_by_owner"
    ):
        raise ValueError("functional-code checkpoint topology changed")
    for name, expected_bytes in manifest["files"].items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(expected_bytes):
            raise ValueError("functional-code checkpoint file changed")
    writer.load_state_dict(
        load_file(str(checkpoint / "writer.safetensors"), device="cpu"), strict=True
    )
    trainer = torch.load(
        checkpoint / "trainer.pt", map_location="cpu", weights_only=False
    )
    if (
        trainer.get("schema_version") != TRAINER_SCHEMA
        or int(trainer.get("world_size", -1)) != world_size
        or int(trainer.get("macro", -1)) != int(manifest["macro"])
        or len(trainer.get("rank_rng", ())) != world_size
    ):
        raise ValueError("functional-code trainer state changed")
    optimizer.load_state_dict(trainer["optimizer"])
    scheduler.load_state_dict(trainer["scheduler"])
    rng = trainer["rank_rng"][rank]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    torch.cuda.set_rng_state(rng["torch_cuda"])
    return int(trainer["macro"]), int(trainer["metrics_rows"])
