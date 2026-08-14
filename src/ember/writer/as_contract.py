"""Data wall and launch contract for dynamic-K AS-Writer training."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_source_checkpoint import DistributedContext, read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.writer.as_config import (
    AS_WRITER_LAUNCH_SCHEMA,
    REPO_ROOT,
    authority_path,
    parse_macro_boundaries,
)
from ember.writer.data import FunctionalQueryDataset, WriterTaskAuthority
from ember.writer.errors import WriterModelError


def resolve_runtime(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> tuple[int, int, tuple[int, ...], int]:
    cell = config["formal_run" if args.mode == "formal" else "profile_defaults"]
    if context.world_size not in cell["allowed_world_sizes"]:
        raise WriterModelError("dynamic-K Writer world size is outside 1--6")
    total = int(args.total_macros or cell["total_macros"])
    batch = int(args.batch_size or cell["per_task_action_batch_size"])
    checkpoints = parse_macro_boundaries(
        args.checkpoint_macros or cell["checkpoint_macros"], total
    )
    stop = int(args.stop_after_macro or cell.get("stop_after_macro", total))
    if total <= 0 or batch != 20 or not 0 < stop <= total:
        raise WriterModelError("invalid dynamic-K Writer runtime request")
    if args.mode == "formal":
        allowed_stops = {int(value) for value in cell["stage_stop_macros"]}
        if (
            total != int(cell["total_macros"])
            or checkpoints
            != parse_macro_boundaries(cell["checkpoint_macros"], total)
            or stop not in allowed_stops
        ):
            raise WriterModelError("formal dynamic-K runtime differs from its authority")
        state = git_state(REPO_ROOT)
        if not git_state_is_clean_pushed_or_frozen_authority(state):
            raise WriterModelError(
                "formal dynamic-K training requires a clean pushed or frozen commit"
            )
        if context.numa_node is None or not context.cpu_affinity:
            raise WriterModelError("formal dynamic-K training requires GPU-local NUMA binding")
    return total, batch, checkpoints, stop


def _broadcast(
    context: DistributedContext, operation: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    payload: list[Any] = [None]
    if context.is_main:
        try:
            payload[0] = operation()
        except Exception as error:
            payload[0] = {"error": repr(error)}
    if context.world_size > 1:
        if not dist.is_initialized():
            raise WriterModelError("dynamic-K authority broadcast requires initialized NCCL")
        dist.broadcast_object_list(payload, src=0, device=context.device)
    elif payload[0] is None:
        payload[0] = operation()
    if payload[0].get("error"):
        raise WriterModelError(payload[0]["error"])
    return payload[0]


def load_training_data(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> tuple[FunctionalQueryDataset, tuple[WriterTaskAuthority, ...], dict[str, Any]]:
    manifest = read_json(authority_path(config, "target_data_manifest"))
    rows = tuple(row for row in manifest["tasks"] if row["split_role"] == "train")
    root = args.data_root.resolve()
    tasks = []
    suites: dict[str, int] = {}
    for row in rows:
        path = (root / str(row["hdf5"]["relative_path"])).resolve()
        if not path.is_relative_to(root):
            raise WriterModelError("target HDF5 escaped its declared root")
        authority = WriterTaskAuthority(
            task_id=int(row["global_task_id"]),
            language=str(row["language"]),
            path=path,
            expected_bytes=int(row["hdf5"]["bytes"]),
        )
        if not path.is_file() or path.stat().st_size != authority.expected_bytes:
            raise WriterModelError(f"target HDF5 size changed: {authority.task_id}")
        tasks.append(authority)
        suite = str(row["suite"])
        suites[suite] = suites.get(suite, 0) + 1
    tasks = sorted(tasks, key=lambda task: task.task_id)
    if len(tasks) != 24 or sorted(suites.values()) != [6, 6, 6, 6]:
        raise WriterModelError("dynamic-K action data is not the sealed train24 split")
    first, last = map(int, config["data"]["demo_indices"])
    dataset = FunctionalQueryDataset(
        tasks,
        demo_indices=range(first, last + 1),
        action_chunk_size=int(config["data"]["action_chunk_size"]),
        max_open_files_per_worker=int(config["data"]["max_open_files_per_worker"]),
    )
    return dataset, tuple(tasks), {
        "tasks_checked": len(tasks),
        "bytes_checked": sum(task.expected_bytes for task in tasks),
        "identity_evidence": "manifest_path_size_and_dataset_schema",
    }


def inspect_video_data(
    root: Path,
    config: Mapping[str, Any],
    task_ids: Sequence[int],
) -> dict[str, Any]:
    manifest = read_json(authority_path(config, "target_data_manifest"))
    by_id = {int(row["global_task_id"]): row for row in manifest["tasks"]}
    first, last = map(int, config["data"]["demo_indices"])
    stride = int(config["writer"]["frame_stride"])
    costs: dict[str, dict[str, int]] = {}
    for task_id in task_ids:
        row = by_id[int(task_id)]
        path = (root / str(row["hdf5"]["relative_path"])).resolve()
        if not path.is_relative_to(root) or path.stat().st_size != int(row["hdf5"]["bytes"]):
            raise WriterModelError(f"teacher-video HDF5 changed: {task_id}")
        episode_lengths = row.get("demonstrations", {}).get("episode_lengths", [])
        if len(episode_lengths) != last - first + 1:
            raise WriterModelError("teacher-video length metadata changed")
        per_demo = {}
        for demo in range(first, last + 1):
            raw = int(episode_lengths[demo - first])
            sampled = (raw - 1) // stride + 1
            if (raw - 1) % stride:
                sampled += 1
            per_demo[str(demo)] = sampled
        costs[str(task_id)] = per_demo
    return {
        "root": str(root),
        "task_ids": list(task_ids),
        "sampled_frame_counts_by_task": costs,
        "max_sampled_frames": max(value for row in costs.values() for value in row.values()),
        "identity_evidence": "manifest_path_size_and_episode_frame_counts",
    }


def load_run_authorities(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    source_run = args.source_run.resolve()
    checkpoint = args.checkpoint.resolve()
    if checkpoint.parent.parent != source_run or checkpoint.parent.name != "checkpoints":
        raise WriterModelError("source checkpoint is outside its owning run")
    model_path = checkpoint / "policy"
    files = {
        name: (model_path / name).stat().st_size
        for name in ("config.json", "model.safetensors")
        if (model_path / name).is_file()
    }
    trainer = read_json(checkpoint / "trainer_state.json")
    if set(files) != {"config.json", "model.safetensors"} or int(
        trainer.get("optimizer_step", -1)
    ) != 1000:
        raise WriterModelError("source policy is not the retained step1000 checkpoint")
    tokenizer = args.tokenizer_path.resolve()
    tokenizer_manifest = read_json(authority_path(config, "tokenizer_manifest"))
    expected_tokenizer_bytes = int(tokenizer_manifest.get("bytes", -1))
    if (
        not tokenizer.is_file()
        or expected_tokenizer_bytes <= 0
        or tokenizer.stat().st_size != expected_tokenizer_bytes
    ):
        raise WriterModelError("OpenPI tokenizer is missing")
    source_config = read_json(authority_path(config, "source_base_config"))
    source = {
        "source_run": str(source_run),
        "checkpoint": str(checkpoint),
        "optimizer_step": 1000,
        "model_path": str(model_path),
        "model_files": files,
    }
    return SimpleNamespace(source_base_config=source_config), source, {
        "path": str(tokenizer),
        "bytes": expected_tokenizer_bytes,
    }


def writer_trainable_contract(
    writer: torch.nn.Module, policy: torch.nn.Module, lora: Any
) -> dict[str, Any]:
    trainable = [(name, value) for name, value in writer.named_parameters() if value.requires_grad]
    if (
        not trainable
        or any(value.requires_grad for value in policy.parameters())
    ):
        raise WriterModelError("dynamic-K Writer freeze boundary changed")
    return {
        "object": "v6_semantic_core_common_value_set_bridge_writer_only",
        "writer_parameter_count": sum(value.numel() for value in writer.parameters()),
        "writer_trainable_parameter_count": sum(value.numel() for _, value in trainable),
        "writer_trainable_parameter_tensors": len(trainable),
        "writer_frozen_parameter_count": sum(
            value.numel() for value in writer.parameters() if not value.requires_grad
        ),
        "generated_lora_parameter_count": lora.parameter_count,
        "generated_lora_tensor_count": lora.state_tensor_count,
        "source_policy_trainable_parameter_count": 0,
    }


def build_contract(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    source: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    video_data: Mapping[str, Any],
    data_validation: Mapping[str, Any],
    task_ids: Sequence[int],
    trainable: Mapping[str, Any],
    total_macros: int,
    batch_size: int,
    checkpoint_macros: Sequence[int],
) -> dict[str, Any]:
    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "device": str(context.device),
        "numa_node": context.numa_node,
        "cpu_affinity": list(context.cpu_affinity or ()),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    topology: list[Any] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(topology, local)
    else:
        topology[0] = local
    state = git_state(REPO_ROOT)
    return {
        "schema_version": AS_WRITER_LAUNCH_SCHEMA,
        "mode": args.mode,
        "git": {"branch": state["branch"], "commit": state["commit"]},
        "config_path": str(args.config.resolve()),
        "authorities": dict(config["authorities"]),
        "source": dict(source),
        "tokenizer": dict(tokenizer),
        "video_data": dict(video_data),
        "target_action_data_validation": dict(data_validation),
        "information_wall": dict(config["information_wall"]),
        "writer": dict(config["writer"]),
        "data": dict(config["data"]),
        "conditioning_training": dict(config["conditioning_training"]),
        "optimization": dict(config["optimization"]),
        "task_ids": list(task_ids),
        "runtime": {
            "world_size": context.world_size,
            "rank_topology": topology,
            "total_macros": total_macros,
            "per_task_action_batch_size": batch_size,
            "checkpoint_macros": list(checkpoint_macros),
            "num_workers_per_rank": args.num_workers,
            "task_assignment": "cost_balanced_long_first_dynamic_uneven",
        },
        "trainable": dict(trainable),
    }


def publish_contract(
    args: argparse.Namespace,
    context: DistributedContext,
    contract: Mapping[str, Any],
) -> None:
    def operation() -> dict[str, Any]:
        path = args.output_dir / "run_contract.json"
        if args.resume is None:
            if args.output_dir.exists() and any(args.output_dir.iterdir()):
                raise WriterModelError("fresh dynamic-K output directory is not empty")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(path, dict(contract))
        elif not path.is_file() or read_json(path) != dict(contract):
            raise WriterModelError("exact-resume dynamic-K launch contract changed")
        append_jsonl(
            args.output_dir / "invocations.jsonl",
            {
                "argv": sys.argv,
                "host": socket.gethostname(),
                "resume": str(args.resume) if args.resume else None,
                "requested_stop_after_macro": args.stop_after_macro,
                "started_unix": time.time(),
            },
        )
        return {"published": True}

    _broadcast(context, operation)
