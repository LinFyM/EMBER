"""Static split, matching, topology, and launch authority for direct LoRA SFT."""

from __future__ import annotations

import argparse
import importlib.metadata
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist

from ember.source_base_checkpoint import (
    canonical_hash,
    git_state,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.writer.feature_cache import (
    FeatureCacheTask,
    load_feature_cache_config,
    load_feature_tasks,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_FILE_NAMES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


class DirectLoRAError(RuntimeError):
    """Raised when direct-LoRA training crosses a sealed contract."""


@dataclass(frozen=True)
class DirectContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


def load_direct_lora_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "ember_direct_lora_sft_v1":
        raise DirectLoRAError("unsupported direct-LoRA config")
    protocol = config.get("protocol", {})
    for key in (
        "manifest",
        "validation_task_config",
        "source_selection",
        "lora_contract",
    ):
        authority = REPO_ROOT / str(protocol.get(key, ""))
        if (
            not authority.is_file()
            or sha256_file(authority) != protocol.get(f"{key}_sha256")
        ):
            raise DirectLoRAError(f"sealed direct-LoRA authority changed: {key}")
    matching = config.get("matching", {})
    numerator = (
        int(matching.get("writer_total_steps", -1))
        * int(matching.get("writer_world_size", -1))
        * int(matching.get("writer_per_rank_batch_size", -1))
    )
    task_count = int(matching.get("writer_source_task_count", -1))
    if (
        task_count <= 0
        or numerator % task_count
        or numerator // task_count
        != int(matching.get("consumed_queries_per_target_task", -1))
        or config.get("data", {}).get("required_split") != "validation"
        or int(config.get("data", {}).get("task_count", -1)) != 10
    ):
        raise DirectLoRAError("direct-LoRA matching formula changed")
    return config


def policy_files(
    policy_path: Path, selection: dict[str, Any]
) -> dict[str, str]:
    expected = selection.get("selected", {}).get("policy_files", {})
    result = {}
    for name in POLICY_FILE_NAMES:
        path = policy_path / name
        record = expected.get(name, {})
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise DirectLoRAError(f"selected source policy changed: {name}")
        result[name] = str(record["sha256"])
    return result


def load_tasks(
    config: dict[str, Any], data_root: Path
) -> tuple[FeatureCacheTask, ...]:
    task_config_path = REPO_ROOT / config["protocol"]["validation_task_config"]
    task_config = load_feature_cache_config(task_config_path, REPO_ROOT)
    tasks = load_feature_tasks(task_config, REPO_ROOT, data_root)
    if (
        task_config["protocol"]["required_split"] != "validation"
        or len(tasks) != int(config["data"]["task_count"])
    ):
        raise DirectLoRAError("direct-LoRA tasks are not the sealed validation pool")
    return tasks


def task_assignments(
    task_ids: Sequence[int], world_size: int
) -> tuple[tuple[int, ...], ...]:
    if world_size <= 0 or not task_ids or len(set(task_ids)) != len(task_ids):
        raise DirectLoRAError("invalid direct-LoRA task assignment")
    return tuple(tuple(task_ids[rank::world_size]) for rank in range(world_size))


def validate_launch(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    context: DirectContext,
    task_ids: tuple[int, ...],
    checkpoint_steps: tuple[int, ...],
) -> None:
    if context.world_size != 8:
        raise DirectLoRAError("direct-LoRA training requires eight symmetric ranks")
    matched_queries = int(config["matching"]["consumed_queries_per_target_task"])
    if args.batch_size * args.total_steps != matched_queries:
        raise DirectLoRAError(
            "direct-LoRA steps and batch must match Writer queries per task"
        )
    if not 0 < args.stop_after_step <= args.total_steps:
        raise DirectLoRAError("invalid direct-LoRA stop step")
    if args.stop_after_step not in checkpoint_steps:
        raise DirectLoRAError(
            "direct-LoRA segment must end at an exact-resume checkpoint"
        )
    if args.mode == "formal":
        formal = config["formal_run"]
        expected = (
            formal.get("status"),
            int(formal.get("expected_world_size", -1)),
            int(formal.get("task_count", -1)),
            int(formal.get("per_rank_batch_size", -1)),
            int(formal.get("total_steps", -1)),
            tuple(int(value) for value in formal.get("checkpoint_steps", [])),
        )
        actual = (
            "sealed",
            context.world_size,
            len(task_ids),
            args.batch_size,
            args.total_steps,
            checkpoint_steps,
        )
        if expected != actual or args.stop_after_step != args.total_steps:
            raise DirectLoRAError("formal direct-LoRA launch differs from profile seal")
        if git_state()["dirty_paths"]:
            raise DirectLoRAError("formal direct-LoRA launch requires a clean worktree")


def build_run_contract(
    *,
    args: argparse.Namespace,
    config_path: Path,
    config: dict[str, Any],
    context: DirectContext,
    tasks: Sequence[FeatureCacheTask],
    assignments: tuple[tuple[int, ...], ...],
    checkpoint_steps: tuple[int, ...],
    source_policy_files: dict[str, str],
    trainable: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "ember_direct_lora_sft_launch_v1",
        "mode": args.mode,
        "git": git_state(),
        "config_sha256": sha256_file(config_path),
        "protocol": config["protocol"],
        "source_policy_files": source_policy_files,
        "task_ids": [task.task_id for task in tasks],
        "expected_hdf5_sha256": {
            str(task.task_id): task.expected_hdf5_sha256 for task in tasks
        },
        "assignments": [list(values) for values in assignments],
        "data": config["data"],
        "matching": config["matching"],
        "optimization": config["optimization"],
        "runtime": {
            "world_size": context.world_size,
            "one_policy_cuda_process_per_gpu": True,
            "independent_task_local_optimizers": True,
            "per_task_batch_size": args.batch_size,
            "per_task_total_steps": args.total_steps,
            "per_task_checkpoint_steps": list(checkpoint_steps),
            "per_task_consumed_queries": args.batch_size * args.total_steps,
            "num_workers_per_rank": args.num_workers,
        },
        "trainable": trainable,
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "lerobot": importlib.metadata.version("lerobot"),
        },
    }


def persist_run_contract(
    *,
    args: argparse.Namespace,
    context: DirectContext,
    contract: dict[str, Any],
) -> str:
    contract_sha256 = canonical_hash(contract)
    if context.rank == 0:
        contract_path = args.output_dir / "run_contract.json"
        if args.resume:
            if (
                not contract_path.is_file()
                or canonical_hash(read_json(contract_path)) != contract_sha256
            ):
                raise DirectLoRAError("direct-LoRA resume launch contract changed")
        else:
            if args.output_dir.exists() and any(args.output_dir.iterdir()):
                raise DirectLoRAError(
                    f"direct-LoRA output is not empty: {args.output_dir}"
                )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(contract_path, contract)
            write_json_atomic(
                args.output_dir / "runtime_paths.json",
                {
                    "policy_path": str(args.policy_path.resolve()),
                    "data_root": str(args.data_root.resolve()),
                    "host": socket.gethostname(),
                },
            )
    dist.barrier()
    return contract_sha256
