"""Launch provenance and exact-resume checkpoints for the source-base stage."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from ember.writer.data import MixedTaskBatchSampler


REPO_ROOT = Path(__file__).resolve().parents[2]


class SourceBaseError(RuntimeError):
    """Raised when a source-base launch violates its sealed contract."""


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourceBaseError(f"invalid JSON authority: {path}") from error
    if not isinstance(value, dict):
        raise SourceBaseError(f"JSON authority must be an object: {path}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def barrier(context: DistributedContext) -> None:
    if context.world_size > 1:
        dist.barrier(device_ids=[context.local_rank])


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
        "dirty_paths": run("status", "--porcelain").splitlines(),
    }


def restore_rng(state: dict[str, Any], context: DistributedContext) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], context.device)


def _rng_state(context: DistributedContext) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(context.device),
    }


def build_contract(
    *,
    config_path: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    context: DistributedContext,
    checkpoint_steps: tuple[int, ...],
    trainable: dict[str, Any],
    git: dict[str, Any],
) -> dict[str, Any]:
    segment = (
        resolve_formal_segment(config, resuming=args.continuation)
        if args.mode == "formal"
        else {
            "start_step": 0,
            "scheduler_horizon_steps": args.total_steps,
            "parent_contract_sha256": None,
            "parent_checkpoint_manifest_sha256": None,
        }
    )
    return {
        "schema_version": "ember_source_base_launch_v1",
        "mode": args.mode,
        "git": git,
        "config_sha256": sha256_file(config_path),
        "protocol": config["protocol"],
        "models": config["models"],
        "model_files": {
            "foundation_config_sha256": sha256_file(args.foundation_path / "config.json"),
            "foundation_model_bytes": (args.foundation_path / "model.safetensors").stat().st_size,
            "vlm_config_sha256": sha256_file(args.vlm_path / "config.json"),
            "vlm_model_bytes": (args.vlm_path / "model.safetensors").stat().st_size,
        },
        "features": config["features"],
        "data": config["data"],
        "optimization": config["optimization"],
        "runtime": {
            "world_size": context.world_size,
            "per_rank_batch_size": args.batch_size,
            "effective_batch_size": context.world_size * args.batch_size,
            "total_steps": args.total_steps,
            "checkpoint_steps": list(checkpoint_steps),
            "continuation": bool(args.continuation),
            "segment_start_step": segment["start_step"],
            "scheduler_horizon_steps": segment["scheduler_horizon_steps"],
            "parent_contract_sha256": segment["parent_contract_sha256"],
            "parent_checkpoint_manifest_sha256": segment[
                "parent_checkpoint_manifest_sha256"
            ],
            "num_workers_per_rank": args.num_workers,
            "distributed_backend": "nccl" if context.world_size > 1 else None,
            "one_policy_cuda_process_per_rank": True,
            "ddp_broadcast_buffers": False,
            "ddp_static_graph": context.world_size > 1,
        },
        "trainable": trainable,
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "lerobot": importlib.metadata.version("lerobot"),
        },
    }


def parse_checkpoint_steps(raw: str, total_steps: int) -> tuple[int, ...]:
    steps = tuple(sorted({int(value) for value in raw.split(",") if value}))
    if (
        not steps
        or steps[-1] != total_steps
        or any(not 0 < value <= total_steps for value in steps)
    ):
        raise SourceBaseError(
            "checkpoint steps must be positive and include total steps"
        )
    return steps


def resolve_formal_segment(
    config: dict[str, Any], *, resuming: bool
) -> dict[str, Any]:
    """Resolve either the original trajectory or its one sealed source-only extension."""

    formal = config["formal_run"]
    if not resuming:
        total_steps = int(formal["total_steps"])
        return {
            "start_step": 0,
            "total_steps": total_steps,
            "checkpoint_steps": (
                total_steps // 3,
                2 * total_steps // 3,
                total_steps,
            ),
            "scheduler_horizon_steps": total_steps,
            "parent_contract_sha256": None,
            "parent_checkpoint_manifest_sha256": None,
        }
    continuation = formal.get("continuation")
    if not isinstance(continuation, dict):
        raise SourceBaseError("formal source-base continuation is not sealed")
    start_step = int(continuation["parent_step"])
    total_steps = int(continuation["total_steps"])
    checkpoint_steps = tuple(int(value) for value in continuation["checkpoint_steps"])
    if (
        start_step != int(formal["total_steps"])
        or int(continuation["additional_steps"]) != total_steps - start_step
        or len(checkpoint_steps) != 3
        or checkpoint_steps[-1] != total_steps
        or tuple(value - start_step for value in checkpoint_steps)
        != tuple((index + 1) * (total_steps - start_step) // 3 for index in range(3))
        or int(continuation["scheduler_horizon_steps"]) != start_step
    ):
        raise SourceBaseError("formal continuation thirds or scheduler boundary changed")
    return {
        "start_step": start_step,
        "total_steps": total_steps,
        "checkpoint_steps": checkpoint_steps,
        "scheduler_horizon_steps": int(continuation["scheduler_horizon_steps"]),
        "parent_contract_sha256": str(continuation["parent_contract_sha256"]),
        "parent_checkpoint_manifest_sha256": str(
            continuation["parent_checkpoint_manifest_sha256"]
        ),
    }


def validate_launch(
    config: dict[str, Any],
    args: argparse.Namespace,
    context: DistributedContext,
    checkpoint_steps: tuple[int, ...],
) -> None:
    if args.continuation and args.mode != "formal":
        raise SourceBaseError("source-base continuation is only available in formal mode")
    if args.foundation_path.name != config["models"]["foundation_revision"]:
        raise SourceBaseError("foundation snapshot does not match its locked revision")
    if args.vlm_path.name != config["models"]["vlm_revision"]:
        raise SourceBaseError("VLM snapshot does not match its locked revision")
    for root in (args.foundation_path, args.vlm_path):
        for filename in ("config.json", "model.safetensors"):
            if not (root / filename).is_file():
                raise SourceBaseError(f"missing locked model file: {root / filename}")
    if args.mode == "formal":
        formal = config["formal_run"]
        if args.continuation and args.resume is None:
            raise SourceBaseError("formal continuation requires its sealed parent checkpoint")
        segment = resolve_formal_segment(config, resuming=args.continuation)
        if context.world_size != formal["expected_world_size"]:
            raise SourceBaseError("formal source-base launch requires exactly eight ranks")
        if formal["per_rank_batch_size"] != args.batch_size:
            raise SourceBaseError("formal batch size is not locked in the active config")
        if segment["total_steps"] != args.total_steps:
            raise SourceBaseError("formal total steps are not locked in the active config")
        if (segment["total_steps"] - segment["start_step"]) % 105 != 0:
            raise SourceBaseError(
                "formal total steps must align all thirds to 35-step task cycles"
            )
        if checkpoint_steps != segment["checkpoint_steps"]:
            raise SourceBaseError("formal checkpoints must be exact thirds")
        if args.stop_after_step != args.total_steps:
            raise SourceBaseError("formal runs cannot stop before the full contract")


def save_checkpoint(
    *,
    output_dir: Path,
    step: int,
    context: DistributedContext,
    policy: SmolVLAPolicy,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    preprocessor: Any,
    postprocessor: Any,
    sampler: MixedTaskBatchSampler,
    contract: dict[str, Any],
    mode: str,
) -> Path:
    nonce = uuid.uuid4().hex
    if context.world_size > 1:
        encoded = torch.zeros(16, dtype=torch.uint8, device=context.device)
        if context.is_main:
            encoded.copy_(
                torch.tensor(
                    list(bytes.fromhex(nonce)), dtype=torch.uint8, device=context.device
                )
            )
        dist.broadcast(encoded, src=0)
        nonce = bytes(encoded.cpu().tolist()).hex()
    temporary = output_dir / "checkpoints" / f".step_{step:08d}.{nonce}.partial"
    final = output_dir / "checkpoints" / f"step_{step:08d}"
    if context.is_main:
        if final.exists():
            raise SourceBaseError(f"checkpoint already exists: {final}")
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
        policy_dir = temporary / "policy"
        policy.save_pretrained(policy_dir)
        preprocessor.save_pretrained(
            policy_dir, config_filename="policy_preprocessor.json"
        )
        postprocessor.save_pretrained(
            policy_dir, config_filename="policy_postprocessor.json"
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
            raise SourceBaseError("formal checkpoint lacks full declared episode coverage")
        consumed = {
            "global_examples": step * context.world_size * sampler.per_rank_batch_size,
            "global_task_slots": step * context.world_size,
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
