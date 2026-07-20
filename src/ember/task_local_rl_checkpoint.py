"""Exact-resume task-local LoRA reward checkpoints and interaction ledgers."""

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
from ember.source_base_checkpoint import (
    canonical_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.writer.model import WriterModelError


def capture_rng(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        ),
    }


def restore_rng(state: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def write_unit_ledger_once(
    unit_dir: Path, update: int, payload: dict[str, Any]
) -> Path:
    path = unit_dir / "rollouts" / f"update_{update:08d}.json"
    if path.is_file():
        if canonical_hash(read_json(path)) != canonical_hash(payload):
            raise WriterModelError(
                f"replayed task-local interaction changed: {unit_dir.name} {update}"
            )
        return path
    write_json_atomic(path, payload)
    return path


def verify_task_local_checkpoint(checkpoint: Path) -> dict[str, Any]:
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    files = manifest.get("files", {})
    required = {"lora.safetensors", "trainer_state.pt", "rng_state.pt"}
    if (
        manifest.get("schema_version") != "ember_task_local_lora_rl_checkpoint_v1"
        or not isinstance(files, dict)
        or not required.issubset(files)
    ):
        raise WriterModelError("task-local RL checkpoint manifest is incomplete")
    for name, record in files.items():
        path = checkpoint / name
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise WriterModelError(f"task-local RL checkpoint file changed: {name}")
    return manifest


def save_task_local_checkpoint(
    *,
    unit_dir: Path,
    next_update: int,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    unit_contract_sha256: str,
    counters: Mapping[str, int],
    rollouts_per_update: int,
    segment_successes: int,
    segment_rollouts: int,
    device: torch.device,
) -> Path:
    if (
        rollouts_per_update <= 0
        or int(counters.get("rollouts", -1))
        != next_update * rollouts_per_update
        or not 0 <= int(counters.get("successes", -1))
        <= int(counters["rollouts"])
        or not 0 <= int(counters.get("optimizer_updates", -1)) <= next_update
        or not 0 <= segment_successes <= segment_rollouts
        or not 0 < segment_rollouts <= int(counters["rollouts"])
    ):
        raise WriterModelError("task-local RL checkpoint counters are inconsistent")
    temporary = (
        unit_dir
        / "checkpoints"
        / f".update_{next_update:08d}.{uuid.uuid4().hex}.partial"
    )
    final = unit_dir / "checkpoints" / f"update_{next_update:08d}"
    if final.exists():
        raise WriterModelError(f"task-local RL checkpoint exists: {final}")
    temporary.mkdir(parents=True)
    state = task_lora_state_dict(policy, clone=True)
    save_file(
        {name: value.to(device="cpu").contiguous() for name, value in state.items()},
        str(temporary / "lora.safetensors"),
    )
    torch.save(
        {
            "next_update": next_update,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "amp_scaler": {"enabled": False, "state": {}},
            "unit_contract_sha256": unit_contract_sha256,
            "rollouts_per_update": rollouts_per_update,
            "counters": dict(counters),
        },
        temporary / "trainer_state.pt",
    )
    rng = capture_rng(device)
    torch.save(
        {
            "next_update": next_update,
            "unit_contract_sha256": unit_contract_sha256,
            "rng": rng,
        },
        temporary / "rng_state.pt",
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
            "schema_version": "ember_task_local_lora_rl_checkpoint_v1",
            "unit_contract_sha256": unit_contract_sha256,
            "interaction_cursor": int(counters["rollouts"]),
            "next_update": next_update,
            "optimizer_updates": int(counters["optimizer_updates"]),
            "segment_successes": segment_successes,
            "segment_rollouts": segment_rollouts,
            "files": files,
        },
    )
    os.replace(temporary, final)
    write_json_atomic(
        unit_dir / "latest_checkpoint.json",
        {"path": str(final), "next_update": next_update},
    )
    restore_rng(rng, device)
    return final


def load_task_local_checkpoint(
    *,
    checkpoint: Path,
    policy: torch.nn.Module,
    contract: SmolVLALoRAContract,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    unit_contract_sha256: str,
    rollouts_per_update: int,
    device: torch.device,
) -> tuple[int, dict[str, int], Mapping[str, Any]]:
    manifest = verify_task_local_checkpoint(checkpoint)
    trainer = torch.load(
        checkpoint / "trainer_state.pt", map_location=device, weights_only=False
    )
    rng_state = torch.load(
        checkpoint / "rng_state.pt", map_location="cpu", weights_only=False
    )
    expected = (
        unit_contract_sha256,
        int(trainer.get("next_update", -1)),
    )
    if (
        (trainer.get("unit_contract_sha256"), int(rng_state.get("next_update", -2)))
        != expected
        or rng_state.get("unit_contract_sha256") != unit_contract_sha256
        or manifest.get("unit_contract_sha256") != unit_contract_sha256
        or int(manifest.get("next_update", -3)) != expected[1]
        or int(trainer.get("rollouts_per_update", -1)) != rollouts_per_update
    ):
        raise WriterModelError("task-local RL resume authority changed")
    copy_task_lora_state_(
        policy,
        load_file(str(checkpoint / "lora.safetensors"), device=str(device)),
        contract,
    )
    optimizer.load_state_dict(trainer["optimizer"])
    scheduler.load_state_dict(trainer["scheduler"])
    counters = {key: int(value) for key, value in trainer["counters"].items()}
    if (
        int(manifest.get("interaction_cursor", -1)) != counters.get("rollouts")
        or counters.get("rollouts") != expected[1] * rollouts_per_update
        or not 0 <= counters.get("successes", -1) <= counters["rollouts"]
        or not 0 <= counters.get("optimizer_updates", -1) <= expected[1]
    ):
        raise WriterModelError("task-local RL resume counters changed")
    return expected[1], counters, rng_state["rng"]
