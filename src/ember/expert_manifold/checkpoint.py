"""Atomic exact-resume checkpoints for independent task experts."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ember.expert_manifold.contract import ExpertManifoldError, ExpertTask
from ember.lora import (
    LoRAContract,
    copy_task_lora_state_,
    task_lora_state_dict,
    validate_lora_state,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic


CHECKPOINT_SCHEMA = "ember_pi05_task_expert_checkpoint_v1"
TRAINER_SCHEMA = "ember_pi05_task_expert_trainer_v1"


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])


def save_task_expert_checkpoint(
    *,
    task_dir: Path,
    task: ExpertTask,
    step: int,
    policy: torch.nn.Module,
    lora_contract: LoRAContract,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metrics_rows: int,
) -> Path:
    if step <= 0 or metrics_rows != step:
        raise ExpertManifoldError("task-expert checkpoint cursor differs from metrics")
    checkpoints = task_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    final = checkpoints / f"step_{step:08d}"
    temporary = checkpoints / f".step_{step:08d}.tmp-{os.getpid()}"
    if final.exists() or temporary.exists():
        raise ExpertManifoldError("task-expert checkpoint already exists")
    temporary.mkdir()
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in task_lora_state_dict(policy).items()
    }
    validate_lora_state(state, lora_contract)
    adapter_path = temporary / "adapter.safetensors"
    trainer_path = temporary / "trainer.pt"
    save_file(state, str(adapter_path))
    torch.save(
        {
            "schema_version": TRAINER_SCHEMA,
            "step": step,
            "task_ordinal": task.ordinal,
            "global_task_id": task.global_task_id,
            "metrics_rows": metrics_rows,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": _rng_state(),
        },
        trainer_path,
    )
    write_json_atomic(
        temporary / "manifest.json",
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "step": step,
            "task_ordinal": task.ordinal,
            "global_task_id": task.global_task_id,
            "state_tensor_count": len(state),
            "state_parameter_count": sum(value.numel() for value in state.values()),
            "files": {
                "adapter.safetensors": adapter_path.stat().st_size,
                "trainer.pt": trainer_path.stat().st_size,
            },
            "content_hash_policy": "disabled_by_owner",
        },
    )
    os.replace(temporary, final)
    return final


def load_task_expert_checkpoint(
    *,
    checkpoint: Path,
    task: ExpertTask,
    policy: torch.nn.Module,
    lora_contract: LoRAContract,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, int]:
    manifest = read_json(checkpoint / "manifest.json")
    if (
        manifest.get("schema_version") != CHECKPOINT_SCHEMA
        or int(manifest.get("task_ordinal", -1)) != task.ordinal
        or int(manifest.get("global_task_id", -1)) != task.global_task_id
        or manifest.get("content_hash_policy") != "disabled_by_owner"
    ):
        raise ExpertManifoldError("task-expert checkpoint manifest changed")
    for name, expected_bytes in manifest.get("files", {}).items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(expected_bytes):
            raise ExpertManifoldError("task-expert checkpoint file path or size changed")
    state = load_file(str(checkpoint / "adapter.safetensors"), device="cpu")
    validate_lora_state(state, lora_contract)
    copy_task_lora_state_(policy, state, lora_contract)
    trainer = torch.load(checkpoint / "trainer.pt", map_location=device, weights_only=False)
    if (
        trainer.get("schema_version") != TRAINER_SCHEMA
        or int(trainer.get("task_ordinal", -1)) != task.ordinal
        or int(trainer.get("global_task_id", -1)) != task.global_task_id
        or int(trainer.get("step", -1)) != int(manifest["step"])
    ):
        raise ExpertManifoldError("task-expert trainer state changed")
    optimizer.load_state_dict(trainer["optimizer"])
    scheduler.load_state_dict(trainer["scheduler"])
    _restore_rng(trainer["rng"])
    return int(trainer["step"]), int(trainer["metrics_rows"])
