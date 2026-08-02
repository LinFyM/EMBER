"""Atomic exact-resume state for the canonical PI05 source-base trainer."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_NAME = re.compile(r"step_[0-9]{8}")


class Pi05SourceTrainingError(RuntimeError):
    """Raised when a PI05 source-base launch or checkpoint violates authority."""


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    numa_node: int | None = None
    cpu_affinity: tuple[int, ...] | None = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def barrier(context: DistributedContext) -> None:
    if context.world_size > 1:
        dist.barrier(device_ids=[context.local_rank])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_reference_matches(
    recorded: object,
    current: Mapping[str, Any],
) -> bool:
    """Compare a source policy without binding it to one host or hash ledger."""

    if not isinstance(recorded, Mapping):
        return False
    for key in ("optimizer_step", "frozen_policy_subdir", "source_training_commit"):
        if recorded.get(key) != current.get(key):
            return False

    def model_layout(reference: Mapping[str, Any]) -> list[tuple[str, int]] | None:
        files = reference.get("model_files")
        if not isinstance(files, list):
            return None
        layout: list[tuple[str, int]] = []
        for row in files:
            if not isinstance(row, Mapping):
                return None
            path = row.get("path")
            size = row.get("bytes")
            if not isinstance(path, str) or not isinstance(size, int):
                return None
            layout.append((path, size))
        return sorted(layout)

    recorded_layout = model_layout(recorded)
    current_layout = model_layout(current)
    if recorded_layout is None or current_layout is None:
        return dict(recorded) == dict(current)
    return recorded_layout == current_layout


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Pi05SourceTrainingError(f"invalid JSON authority: {path}") from error
    if not isinstance(value, dict):
        raise Pi05SourceTrainingError(f"JSON authority must be an object: {path}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def git_state() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "origin_main": run("rev-parse", "origin/main"),
        "dirty_paths": run("status", "--porcelain").splitlines(),
    }


def capture_rng(context: DistributedContext) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(context.device),
    }


def restore_rng(state: dict[str, Any], context: DistributedContext) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], context.device)


def checkpoint_files(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _retire_old_checkpoints(checkpoints: Path, keep_latest: int) -> list[str]:
    if keep_latest <= 0:
        raise Pi05SourceTrainingError("checkpoint retention must keep at least one state")
    candidates = sorted(
        path
        for path in checkpoints.iterdir()
        if path.is_dir() and not path.is_symlink() and _CHECKPOINT_NAME.fullmatch(path.name)
    )
    retired = candidates[:-keep_latest]
    if not retired:
        return []
    records = []
    for path in retired:
        if path.parent.resolve() != checkpoints.resolve():
            raise Pi05SourceTrainingError("checkpoint retention escaped its owned directory")
        manifest = path / "checkpoint_manifest.json"
        records.append(
            {
                "checkpoint": path.name,
                "manifest_sha256": sha256_file(manifest),
                "retired_unix": time.time(),
            }
        )
    retention_log = checkpoints / "retention.jsonl"
    with retention_log.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    for path in retired:
        shutil.rmtree(path)
    return [path.name for path in retired]


def save_checkpoint(
    *,
    output_dir: Path,
    optimizer_step: int,
    micro_step: int,
    context: DistributedContext,
    policy: Any,
    ema_policy: Any | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    contract_sha256: str,
    metrics_rows: int,
    keep_latest: int,
) -> Path:
    """Write one complete checkpoint, publish atomically, then retire superseded states."""

    checkpoints = output_dir / "checkpoints"
    nonce = uuid.uuid4().hex
    if context.world_size > 1:
        encoded = torch.zeros(16, dtype=torch.uint8, device=context.device)
        if context.is_main:
            encoded.copy_(torch.tensor(list(bytes.fromhex(nonce)), device=context.device))
        dist.broadcast(encoded, src=0)
        nonce = bytes(encoded.cpu().tolist()).hex()
    temporary = checkpoints / f".step_{optimizer_step:08d}.{nonce}.partial"
    final = checkpoints / f"step_{optimizer_step:08d}"
    if context.is_main:
        checkpoints.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise Pi05SourceTrainingError(f"checkpoint already exists: {final}")
        temporary.mkdir()
    barrier(context)

    torch.save(
        {
            "rank": context.rank,
            "world_size": context.world_size,
            "optimizer_step": optimizer_step,
            "micro_step": micro_step,
            "rng": capture_rng(context),
        },
        temporary / f"rank_{context.rank:02d}_state.pt",
    )
    barrier(context)
    if context.is_main:
        policy.save_pretrained(temporary / "policy")
        if ema_policy is not None:
            ema_policy.save_pretrained(temporary / "ema_policy")
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
        write_json_atomic(
            temporary / "trainer_state.json",
            {
                "schema_version": "ember_pi05_source_trainer_state_v1",
                "contract_sha256": contract_sha256,
                "optimizer_step": optimizer_step,
                "micro_step": micro_step,
                "gradient_accumulation_offset": 0,
                "metrics_rows": metrics_rows,
                "ema_enabled": ema_policy is not None,
            },
        )
        files = checkpoint_files(temporary)
        write_json_atomic(
            temporary / "checkpoint_manifest.json",
            {
                "schema_version": "ember_pi05_source_checkpoint_v1",
                "contract_sha256": contract_sha256,
                "optimizer_step": optimizer_step,
                "micro_step": micro_step,
                "files": files,
                "aggregate_sha256": canonical_hash(files),
            },
        )
        os.replace(temporary, final)
        _retire_old_checkpoints(checkpoints, keep_latest)
    barrier(context)
    return final


def verify_checkpoint(path: Path, contract_sha256: str) -> dict[str, Any]:
    manifest = read_json(path / "checkpoint_manifest.json")
    state = read_json(path / "trainer_state.json")
    if (
        manifest.get("schema_version") != "ember_pi05_source_checkpoint_v1"
        or state.get("schema_version") != "ember_pi05_source_trainer_state_v1"
        or manifest.get("contract_sha256") != contract_sha256
        or state.get("contract_sha256") != contract_sha256
    ):
        raise Pi05SourceTrainingError("resume checkpoint belongs to another launch contract")
    expected = manifest.get("files", [])
    actual = checkpoint_files(path)
    actual = [row for row in actual if row["path"] != "checkpoint_manifest.json"]
    if actual != expected or canonical_hash(expected) != manifest.get("aggregate_sha256"):
        raise Pi05SourceTrainingError("resume checkpoint file hashes changed")
    return state
