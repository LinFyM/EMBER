"""Hashless authority and runtime contract for task-local PI0.5 experts."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.writer.data import FunctionalQueryDataset, WriterTaskAuthority


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_SCHEMA = "ember_pi05_video_expert_manifold_v1"
WORKER_CONTRACT_SCHEMA = "ember_pi05_task_expert_worker_launch_v1"


class ExpertManifoldError(RuntimeError):
    """Raised when expert-manifold training crosses its sealed boundary."""


@dataclass(frozen=True)
class ExpertTask:
    ordinal: int
    global_task_id: int
    suite: str
    task_id: int
    split_role: str
    language: str
    authority: WriterTaskAuthority


def _information_wall_matches(information: Mapping[str, Any]) -> bool:
    return (
        information.get("expert_action_split_roles") == ["train"]
        and information.get("writer_video_split_roles")
        == ["train", "validation", "test"]
        and int(information.get("validation_experts_trained", -1)) == 0
        and int(information.get("test_experts_trained", -1)) == 0
        and int(information.get("validation_actions_read", -1)) == 0
        and int(information.get("test_actions_read", -1)) == 0
    )


def _topological_writer_matches(writer: Mapping[str, Any]) -> bool:
    return (
        int(writer.get("chunk_count", -1)) == 168
        and int(writer.get("public_rank", -1)) == 16
        and int(writer.get("valid_values", -1)) == 1_287_168
        and writer.get("video_value_path")
        == "phase_centered_projected_video_sqrt_normalized_causal_prefix_integral_only"
        and writer.get("routing_key_path")
        == "full_projected_video_innovation_plus_phase_keys"
    )


def load_expert_manifold_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ExpertManifoldError("unsupported expert-manifold config schema")
    method = config.get("method", {})
    experts = config.get("task_experts", {})
    video = config.get("video_features", {})
    writer = config.get("topological_writer", {})
    meta = config.get("meta_training", {})
    information = config.get("information_wall", {})
    if (
        method.get("name")
        != "video_conditioned_expert_manifold_topological_writer"
        or method.get("language_only_lora_path") is not False
        or not _information_wall_matches(information)
        or int(experts.get("task_count", -1)) != 24
        or int(experts.get("episodes_per_task", -1)) != 50
        or experts.get("task_parameter_sharing") != "none"
        or int(video.get("shots", -1)) != 1
        or int(video.get("phase_slots", -1)) != 16
        or int(video.get("image_hidden_width", -1)) != 2048
        or int(video.get("expert_hidden_width", -1)) != 1024
        or int(video.get("feature_width", -1)) != 3072
        or video.get("cache_contains_actions_or_state") is not False
        or not _topological_writer_matches(writer)
        or int(meta.get("task_count", -1)) != 24
        or int(meta.get("videos_per_task_per_macro", -1)) != 1
        or meta.get("task_aggregation")
        != "each_task_mean_then_train24_equal_mean"
        or meta.get("objective", {}).get("effective_ba_monitor_only") is not True
        or float(meta.get("objective", {}).get("raw_reconstruction_weight", -1))
        != 1.0
        or int(experts.get("profile_defaults", {}).get("scheduler_total_steps", -1))
        != int(experts.get("formal_run", {}).get("total_steps", -2))
        or config.get("content_hash_policy") != "disabled_by_owner"
    ):
        raise ExpertManifoldError("expert-manifold scientific boundary changed")
    return config


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    try:
        value = config["authorities"][name]["path"]
    except (KeyError, TypeError) as error:
        raise ExpertManifoldError(f"missing expert authority: {name}") from error
    return REPO_ROOT / str(value)


def load_train_tasks(
    config: Mapping[str, Any], data_root: Path
) -> tuple[ExpertTask, ...]:
    manifest = read_json(authority_path(config, "target_data_manifest"))
    selected = [row for row in manifest.get("tasks", []) if row.get("split_role") == "train"]
    selected.sort(key=lambda row: int(row["global_task_id"]))
    if len(selected) != int(config["task_experts"]["task_count"]):
        raise ExpertManifoldError("expert target manifest did not resolve train24")
    tasks: list[ExpertTask] = []
    for ordinal, row in enumerate(selected):
        hdf5 = row["hdf5"]
        path = data_root / str(hdf5["relative_path"])
        expected_bytes = int(hdf5["bytes"])
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ExpertManifoldError(
                f"task expert HDF5 path or size changed: {int(row['global_task_id'])}"
            )
        authority = WriterTaskAuthority(
            task_id=int(row["global_task_id"]),
            language=str(row["language"]),
            path=path,
            expected_bytes=expected_bytes,
            expected_sha256=None,
        )
        tasks.append(
            ExpertTask(
                ordinal=ordinal,
                global_task_id=authority.task_id,
                suite=str(row["suite"]),
                task_id=int(row["task_id"]),
                split_role="train",
                language=authority.language,
                authority=authority,
            )
        )
    return tuple(tasks)


def parse_task_indices(value: str, task_count: int) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ExpertManifoldError("task indices must be comma-separated integers") from error
    if (
        not result
        or len(set(result)) != len(result)
        or tuple(sorted(result)) != result
        or any(not 0 <= item < task_count for item in result)
    ):
        raise ExpertManifoldError("task indices are duplicated, unsorted, or out of range")
    return result


def _checkpoint_steps(values: Sequence[int], total_steps: int) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if (
        not result
        or tuple(sorted(set(result))) != result
        or result[-1] != total_steps
        or result[0] <= 0
    ):
        raise ExpertManifoldError("task-expert checkpoint schedule is invalid")
    return result


def resolve_runtime(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> tuple[int, int, tuple[int, ...], int]:
    experts = config["task_experts"]
    source = experts["formal_run"] if args.mode == "formal" else experts["profile_defaults"]
    if args.mode == "formal" and source.get("status") != "sealed":
        raise ExpertManifoldError("formal task-expert config is not sealed by an A40 profile")
    total_steps = int(source["total_steps"])
    batch_size = int(args.batch_size or source["per_task_batch_size"])
    checkpoints = _checkpoint_steps(source["checkpoint_steps"], total_steps)
    default_stop = int(source.get("selected_stop_step", total_steps))
    stop_step = int(args.stop_after_step or default_stop)
    allowed_stops = set(int(value) for value in source.get("stage_stop_steps", checkpoints))
    allowed_stops.update(checkpoints)
    if batch_size <= 0 or stop_step not in allowed_stops or stop_step > total_steps:
        raise ExpertManifoldError("task-expert runtime differs from an allowed stage boundary")
    if args.mode == "formal":
        if batch_size != int(source["per_task_batch_size"]):
            raise ExpertManifoldError("formal task-expert batch differs from its profile seal")
        state = git_state(REPO_ROOT)
        if state["dirty_paths"]:
            raise ExpertManifoldError("formal task-expert launch requires a clean worktree")
        if args.resume is None and state["commit"] != state["upstream_commit"]:
            raise ExpertManifoldError("fresh formal task-expert launch must be pushed")
    return total_steps, batch_size, checkpoints, stop_step


def build_dataset(
    config: Mapping[str, Any], tasks: Sequence[ExpertTask]
) -> FunctionalQueryDataset:
    first, last = map(int, config["task_experts"]["demo_indices"])
    return FunctionalQueryDataset(
        [task.authority for task in tasks],
        demo_indices=range(first, last + 1),
        action_chunk_size=int(config["task_experts"]["action_chunk_size"]),
        max_open_files_per_worker=max(2, len(tasks)),
    )


def _physical_device() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return visible.split(",", 1)[0] if visible else "runtime-default"


def build_worker_contract(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    tasks: Sequence[ExpertTask],
    source: Mapping[str, Any],
    total_steps: int,
    batch_size: int,
    checkpoint_steps: Sequence[int],
) -> dict[str, Any]:
    state = git_state(REPO_ROOT)
    return {
        "schema_version": WORKER_CONTRACT_SCHEMA,
        "mode": args.mode,
        "method": "independent_task_local_rank16_policy_experts",
        "git": {key: state[key] for key in ("branch", "commit")},
        "config": {"path": str(args.config.resolve()), "schema": CONFIG_SCHEMA},
        "source": {
            "run": str(args.source_run.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "model_path": str(source["model_path"]),
        },
        "tokenizer": {
            "path": str(args.tokenizer_path.resolve()),
            "bytes": args.tokenizer_path.stat().st_size,
        },
        "tasks": [
            {
                "ordinal": task.ordinal,
                "global_task_id": task.global_task_id,
                "suite": task.suite,
                "task_id": task.task_id,
                "split_role": task.split_role,
                "language": task.language,
                "hdf5_bytes": task.authority.expected_bytes,
            }
            for task in tasks
        ],
        "information_wall": dict(config["information_wall"]),
        "runtime": {
            "host": socket.gethostname(),
            "cuda_visible_device": _physical_device(),
            "device_name": torch.cuda.get_device_name(0),
            "one_policy_per_worker": True,
            "task_parameter_sharing": "none",
            "total_steps_per_task": total_steps,
            "per_task_batch_size": batch_size,
            "checkpoint_steps": list(checkpoint_steps),
            "num_workers": 0,
        },
        "content_hash_policy": "disabled_by_owner",
    }


def publish_worker_contract(
    args: argparse.Namespace, contract: Mapping[str, Any], stop_step: int
) -> None:
    path = args.output_dir / "run_contract.json"
    if args.resume is None:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ExpertManifoldError("fresh task-expert worker output is not empty")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, dict(contract))
    else:
        if not path.is_file() or read_json(path) != contract:
            raise ExpertManifoldError("task-expert resume worker contract changed")
        resume = args.resume.resolve()
        worker_resume = resume == args.output_dir.resolve()
        checkpoint_resume = len(resume.parents) >= 3 and resume.parents[2] == args.output_dir.resolve()
        if not worker_resume and not checkpoint_resume:
            raise ExpertManifoldError("task-expert resume checkpoint crossed worker ownership")
    append_jsonl(
        args.output_dir / "invocations.jsonl",
        {
            "argv": sys.argv,
            "host": socket.gethostname(),
            "resume": str(args.resume.resolve()) if args.resume else None,
            "requested_stop_after_step": stop_step,
            "started_unix": time.time(),
        },
    )
    write_json_atomic(
        args.output_dir / "runtime_paths.json",
        {
            "source_run": str(args.source_run.resolve()),
            "source_checkpoint": str(args.checkpoint.resolve()),
            "target_data_root": str(args.data_root.resolve()),
            "tokenizer": str(args.tokenizer_path.resolve()),
        },
    )


def task_directory(output_dir: Path, task: ExpertTask) -> Path:
    return output_dir / f"task_{task.ordinal:02d}_global_{task.global_task_id:02d}"


def parse_resume_task(resume: Path | None) -> tuple[int, int] | None:
    if resume is None:
        return None
    try:
        task_dir = resume.parents[1].name
        step = int(resume.name.removeprefix("step_"))
        ordinal = int(task_dir.split("_", 2)[1])
    except (IndexError, ValueError) as error:
        raise ExpertManifoldError("task-expert resume path is malformed") from error
    if not resume.name.startswith("step_") or step <= 0:
        raise ExpertManifoldError("task-expert resume step is invalid")
    return ordinal, step


def worker_stage_resume_step(
    resume: Path | None,
    output_dir: Path,
    tasks: Sequence[ExpertTask],
) -> int | None:
    """Resolve an all-task stage resume from the existing worker root."""

    if resume is None or resume.resolve() != output_dir.resolve():
        return None
    summary = read_json(output_dir / "worker_summary.json")
    if (
        summary.get("schema_version") != "ember_pi05_task_expert_worker_summary_v1"
        or int(summary.get("completed_task_count", -1)) != len(tasks)
        or len(summary.get("tasks", [])) != len(tasks)
    ):
        raise ExpertManifoldError("task-expert worker stage summary changed")
    step = int(summary.get("selected_stop_step", -1))
    if step <= 0:
        raise ExpertManifoldError("task-expert worker stage cursor is invalid")
    rows = {
        int(row.get("task_ordinal", -1)): row for row in summary.get("tasks", [])
    }
    if set(rows) != {task.ordinal for task in tasks}:
        raise ExpertManifoldError("task-expert worker stage ownership changed")
    for task in tasks:
        row = rows[task.ordinal]
        checkpoint = task_directory(output_dir, task) / "checkpoints" / f"step_{step:08d}"
        if (
            int(row.get("global_task_id", -1)) != task.global_task_id
            or int(row.get("completed_steps", -1)) != step
            or not checkpoint.is_dir()
        ):
            raise ExpertManifoldError("task-expert worker stage is incomplete")
    return step
