"""Atomic task-local checkpoints for direct LoRA supervised training."""

from __future__ import annotations

import os
import random
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ember.lora import (
    SmolVLALoRAContract,
    copy_task_lora_state_,
    task_lora_state_dict,
)
from ember.source_base_checkpoint import sha256_file, write_json_atomic
from ember.writer.data import MixedTaskBatchSampler


class DirectLoRACheckpointError(RuntimeError):
    """Raised when a direct-LoRA checkpoint is incomplete or incompatible."""


def capture_rng(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        ),
    }


def restore_task_rng(state: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def verify_checkpoint_files(checkpoint: Path) -> dict[str, Any]:
    from ember.source_base_checkpoint import read_json

    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    files = manifest.get("files", {})
    required = {"lora.safetensors", "trainer_state.pt", "rng_state.pt"}
    if (
        manifest.get("schema_version") != "ember_direct_lora_checkpoint_v1"
        or not isinstance(files, dict)
        or not required.issubset(files)
    ):
        raise DirectLoRACheckpointError("direct-LoRA checkpoint manifest is incomplete")
    for name, record in files.items():
        path = checkpoint / name
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise DirectLoRACheckpointError(
                f"direct-LoRA checkpoint file changed: {name}"
            )
    return manifest


def save_direct_lora_checkpoint(
    *,
    task_dir: Path,
    task_id: int,
    step: int,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler: MixedTaskBatchSampler,
    task_contract_sha256: str,
    device: torch.device,
    formal: bool,
) -> Path:
    temporary = (
        task_dir
        / "checkpoints"
        / f".step_{step:08d}.{uuid.uuid4().hex}.partial"
    )
    final = task_dir / "checkpoints" / f"step_{step:08d}"
    if final.exists():
        raise DirectLoRACheckpointError(f"checkpoint already exists: {final}")
    temporary.mkdir(parents=True)

    state = task_lora_state_dict(policy, clone=True)
    save_file(
        {name: value.to(device="cpu").contiguous() for name, value in state.items()},
        str(temporary / "lora.safetensors"),
    )
    torch.save(
        {
            "next_step": step,
            "task_id": task_id,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "amp_scaler": {"enabled": False, "state": {}},
            "task_contract_sha256": task_contract_sha256,
        },
        temporary / "trainer_state.pt",
    )
    torch.save(
        {
            "next_step": step,
            "task_id": task_id,
            "per_rank_batch_size": sampler.per_rank_batch_size,
            "sampler_seed": sampler.seed,
            "rng": capture_rng(device),
        },
        temporary / "rng_state.pt",
    )
    coverage = sampler.coverage_for_steps(0, step)[task_id]
    if formal and len(coverage) != sampler.episodes_per_task:
        raise DirectLoRACheckpointError(
            "formal direct-LoRA checkpoint lacks all declared episodes"
        )
    consumed = {
        **sampler.consumed_identity_summary(0, step),
        "task_id": task_id,
        "episodes_with_signal": len(coverage),
        "declared_episodes": sampler.episodes_per_task,
        "next_step": step,
    }
    files = {}
    for path in sorted(value for value in temporary.iterdir() if value.is_file()):
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    write_json_atomic(
        temporary / "checkpoint_manifest.json",
        {
            "schema_version": "ember_direct_lora_checkpoint_v1",
            "task_contract_sha256": task_contract_sha256,
            "consumed": consumed,
            "files": files,
        },
    )
    os.replace(temporary, final)
    write_json_atomic(
        task_dir / "latest_checkpoint.json", {"path": str(final), "step": step}
    )
    return final


def load_direct_lora_checkpoint(
    *,
    checkpoint: Path,
    task_id: int,
    policy: torch.nn.Module,
    contract: SmolVLALoRAContract,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    task_contract_sha256: str,
    per_rank_batch_size: int,
    sampler_seed: int,
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    manifest = verify_checkpoint_files(checkpoint)
    trainer = torch.load(
        checkpoint / "trainer_state.pt", map_location=device, weights_only=False
    )
    rank_state = torch.load(
        checkpoint / "rng_state.pt", map_location="cpu", weights_only=False
    )
    expected = (
        task_id,
        task_contract_sha256,
        per_rank_batch_size,
        sampler_seed,
    )
    actual = (
        int(trainer.get("task_id", -1)),
        trainer.get("task_contract_sha256"),
        int(rank_state.get("per_rank_batch_size", -1)),
        int(rank_state.get("sampler_seed", -1)),
    )
    if (
        actual != expected
        or int(rank_state.get("task_id", -1)) != task_id
        or int(rank_state.get("next_step", -1))
        != int(trainer.get("next_step", -2))
        or manifest.get("task_contract_sha256") != task_contract_sha256
        or int(manifest.get("consumed", {}).get("next_step", -1))
        != int(trainer.get("next_step", -2))
    ):
        raise DirectLoRACheckpointError("direct-LoRA resume state changed")
    copy_task_lora_state_(
        policy,
        load_file(str(checkpoint / "lora.safetensors"), device=str(device)),
        contract,
    )
    optimizer.load_state_dict(trainer["optimizer"])
    scheduler.load_state_dict(trainer["scheduler"])
    return int(trainer["next_step"]), rank_state["rng"]
