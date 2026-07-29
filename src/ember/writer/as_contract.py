"""Authorities, data wall, and launch contracts for PI05 AS-Writer."""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import torch
import torch.distributed as dist

from ember.lora import canonical_contract_sha256
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import (
    DistributedContext,
    canonical_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.pi05_source_contract import append_jsonl
from ember.writer.architecture import V8_WRITER_PARAMETER_COUNT
from ember.writer.as_config import (
    REPO_ROOT,
    authority_path,
    load_writer_config,
    writer_split_roles,
    writer_stage,
)
from ember.writer.data import FunctionalQueryDataset, WriterTaskAuthority
from ember.writer.model import CompleteLoRAWriter, WriterModelError
from ember.writer.topology import validate_task_complete_topology


AS_WRITER_LAUNCH_SCHEMA = "ember_pi05_language_axial_as_writer_launch_v8"
_CHECKPOINT_NAME = re.compile(r"step_([0-9]{8})")


def parse_checkpoint_steps(value: str | Sequence[int], total_steps: int) -> tuple[int, ...]:
    if isinstance(value, str) and value.startswith("every:"):
        try:
            interval = int(value.removeprefix("every:"))
        except ValueError as error:
            raise WriterModelError("invalid AS-Writer checkpoint interval") from error
        if interval <= 0 or total_steps % interval:
            raise WriterModelError(
                "AS-Writer checkpoint interval must divide total steps"
            )
        return tuple(range(interval, total_steps + 1, interval))
    raw = value.split(",") if isinstance(value, str) else value
    try:
        result = tuple(sorted({int(item) for item in raw}))
    except (TypeError, ValueError) as error:
        raise WriterModelError("invalid AS-Writer checkpoint steps") from error
    if not result or result[-1] != total_steps or any(step <= 0 for step in result):
        raise WriterModelError("AS-Writer checkpoints must end at total_steps")
    return result


def resume_step(checkpoint: Path | None) -> int:
    if checkpoint is None:
        return 0
    match = _CHECKPOINT_NAME.fullmatch(checkpoint.name)
    if match is None:
        raise WriterModelError("AS-Writer resume path is not a step checkpoint")
    return int(match.group(1))


def _runtime_values(
    args: argparse.Namespace,
    source: Mapping[str, Any],
) -> tuple[int, int, tuple[int, ...], int, int]:
    total_steps = args.total_steps or int(source["total_steps"])
    batch_size = args.batch_size or int(source["per_rank_batch_size"])
    checkpoint_steps = parse_checkpoint_steps(
        args.checkpoint_steps or source["checkpoint_steps"],
        total_steps,
    )
    default_stop = int(source.get("selected_stop_step", total_steps))
    stop_step = args.stop_after_step or default_stop
    if min(total_steps, batch_size, stop_step) <= 0 or stop_step > total_steps:
        raise WriterModelError("invalid AS-Writer runtime request")
    return total_steps, batch_size, checkpoint_steps, default_stop, stop_step


def _validate_formal_runtime(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    *,
    total_steps: int,
    batch_size: int,
    checkpoint_steps: tuple[int, ...],
    default_stop: int,
    stop_step: int,
) -> None:
    formal = config["formal_run"]
    expected = (
        "sealed",
        int(formal["expected_world_size"]),
        int(formal["total_steps"]),
        int(formal["per_rank_batch_size"]),
        parse_checkpoint_steps(formal["checkpoint_steps"], total_steps),
    )
    observed = (
        formal.get("status"),
        context.world_size,
        total_steps,
        batch_size,
        checkpoint_steps,
    )
    stage_stops = parse_checkpoint_steps(
        formal.get("stage_stop_steps", [default_stop]),
        total_steps,
    )
    invalid_schedule = (
        observed != expected
        or any(value not in checkpoint_steps for value in stage_stops)
        or default_stop not in stage_stops
        or stop_step not in stage_stops
    )
    if invalid_schedule:
        raise WriterModelError(
            "formal AS-Writer launch differs from its sealed profile"
        )
    state = git_state(REPO_ROOT)
    if state["dirty_paths"]:
        raise WriterModelError("formal AS-Writer launch requires a clean worktree")
    if args.resume is None and state["commit"] != state["origin_main"]:
        raise WriterModelError("fresh formal AS-Writer launch must be pushed")
    if context.numa_node is None or not context.cpu_affinity:
        raise WriterModelError(
            "formal AS-Writer launch requires GPU-local NUMA binding"
        )
def resolve_runtime(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> tuple[int, int, tuple[int, ...]]:
    if args.mode == "formal" and config["formal_run"].get("status") != "sealed":
        raise WriterModelError(
            "formal AS-Writer config is not sealed from the live v8 profile"
        )
    source = config["formal_run"] if args.mode == "formal" else config["profile_defaults"]
    (
        total_steps,
        batch_size,
        checkpoint_steps,
        default_stop,
        stop_step,
    ) = _runtime_values(
        args,
        source,
    )
    expected_world_size = int(source.get("expected_world_size", 8))
    validate_task_complete_topology(
        config,
        context,
        expected_world_size=expected_world_size,
        batch_size=batch_size,
        mode=args.mode,
    )
    if args.mode == "formal":
        _validate_formal_runtime(
            args,
            config,
            context,
            total_steps=total_steps,
            batch_size=batch_size,
            checkpoint_steps=checkpoint_steps,
            default_stop=default_stop,
            stop_step=stop_step,
        )
    args.stop_after_step = stop_step
    return total_steps, batch_size, checkpoint_steps


def _broadcast_validation(
    context: DistributedContext, operation: Any
) -> dict[str, Any]:
    payload: list[Any] = [None]
    if context.is_main:
        try:
            payload[0] = operation()
        except Exception as error:
            payload[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=context.device)
    if payload[0].get("error"):
        raise WriterModelError(payload[0]["error"])
    return payload[0]


def _validate_target_files(
    tasks: Sequence[WriterTaskAuthority], verify_hashes: bool
) -> dict[str, Any]:
    for task in tasks:
        path = task.path
        if not path.is_file() or path.stat().st_size != task.expected_bytes:
            raise WriterModelError(f"AS-Writer train HDF5 size changed: {task.task_id}")
        if verify_hashes and sha256_file(path) != task.expected_sha256:
            raise WriterModelError(f"AS-Writer train HDF5 hash changed: {task.task_id}")
    return {
        "tasks_checked": len(tasks),
        "bytes_checked": sum(task.expected_bytes for task in tasks),
        "full_sha256_verified": verify_hashes,
        "hdf5_identity_sha256": canonical_hash(
            [
                [task.task_id, task.expected_bytes, task.expected_sha256]
                for task in tasks
            ]
        ),
    }


def load_training_data(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> tuple[FunctionalQueryDataset, tuple[WriterTaskAuthority, ...], dict[str, Any]]:
    target = read_json(authority_path(config, "target_data_manifest"))
    roles = set(writer_split_roles(config))
    root = args.data_root.resolve()
    rows = tuple(
        row for row in target["tasks"] if str(row["split_role"]) in roles
    )
    tasks_list = []
    for row in rows:
        path = (root / str(row["hdf5"]["relative_path"])).resolve()
        if not path.is_relative_to(root):
            raise WriterModelError("target HDF5 escaped its declared data root")
        tasks_list.append(
            WriterTaskAuthority(
                task_id=int(row["global_task_id"]),
                language=str(row["language"]),
                path=path,
                expected_bytes=int(row["hdf5"]["bytes"]),
                expected_sha256=str(row["hdf5"]["sha256"]),
            )
        )
    tasks = tuple(sorted(tasks_list, key=lambda task: task.task_id))
    suite_counts: dict[str, int] = {}
    for row in rows:
        suite_counts[str(row["suite"])] = suite_counts.get(str(row["suite"]), 0) + 1
    per_suite = 6 if writer_stage(config) == "development" else 8
    if (
        len(tasks) != int(config["data"]["task_count"])
        or sorted(suite_counts.values()) != [per_suite] * 4
    ):
        raise WriterModelError("AS-Writer action training is not its sealed source role")
    validation = _broadcast_validation(
        context, lambda: _validate_target_files(tasks, not args.skip_data_sha)
    )
    first_demo, last_demo = map(int, config["data"]["demo_indices"])
    query_authorities = tuple(
        WriterTaskAuthority(
            task_id=task.task_id,
            language=task.language,
            path=task.path,
            expected_bytes=task.expected_bytes,
        )
        for task in tasks
    )
    dataset = FunctionalQueryDataset(
        query_authorities,
        demo_indices=range(first_demo, last_demo + 1),
        action_chunk_size=int(config["data"]["action_chunk_size"]),
        max_open_files_per_worker=int(config["data"]["max_open_files_per_worker"]),
    )
    return dataset, tasks, validation


def inspect_video_data(
    root: Path,
    config: Mapping[str, Any],
    task_ids: Sequence[int],
    *,
    verify_hashes: bool,
) -> dict[str, Any]:
    root = root.resolve()
    target_path = authority_path(config, "target_data_manifest")
    target = read_json(target_path)
    by_id = {
        int(row["global_task_id"]): row for row in target.get("tasks", [])
    }
    selected_ids = tuple(sorted({int(task_id) for task_id in task_ids}))
    if not selected_ids or set(selected_ids) - set(by_id):
        raise WriterModelError("Writer video task IDs are outside target40")
    records = []
    first_demo, last_demo = map(int, config["data"]["demo_indices"])
    frame_stride = int(config["writer"]["frame_stride"])
    sampled_frame_counts: dict[str, dict[str, int]] = {}
    for task_id in selected_ids:
        row = by_id[task_id]
        path = (root / str(row["hdf5"]["relative_path"])).resolve()
        expected_bytes = int(row["hdf5"]["bytes"])
        expected_sha256 = str(row["hdf5"]["sha256"])
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or path.stat().st_size != expected_bytes
            or (verify_hashes and sha256_file(path) != expected_sha256)
        ):
            raise WriterModelError(f"Writer video HDF5 changed: {task_id}")
        records.append(
            [task_id, str(row["hdf5"]["relative_path"]), expected_bytes, expected_sha256]
        )
        demo_counts: dict[str, int] = {}
        with h5py.File(path, "r") as handle:
            for demo_index in range(first_demo, last_demo + 1):
                pixels = handle.get(
                    f"data/demo_{demo_index}/obs/agentview_rgb"
                )
                if (
                    not isinstance(pixels, h5py.Dataset)
                    or pixels.ndim != 4
                    or pixels.shape[0] <= 0
                    or pixels.shape[-1] != 3
                ):
                    raise WriterModelError(
                        "Writer video-cost metadata changed"
                    )
                raw_frames = int(pixels.shape[0])
                sampled = (raw_frames + frame_stride - 1) // frame_stride
                if (raw_frames - 1) % frame_stride:
                    sampled += 1
                demo_counts[str(demo_index)] = sampled
        sampled_frame_counts[str(task_id)] = demo_counts
    return {
        "root": str(root.resolve()),
        "schema_version": "ember_pi05_raw_teacher_video_data_v1",
        "target_data_manifest_file_sha256": sha256_file(target_path),
        "target_data_manifest_payload_sha256": target["canonical_payload_sha256"],
        "dataset": dict(target["dataset"]),
        "task_ids": list(selected_ids),
        "task_count": len(selected_ids),
        "episode_count": 50 * len(selected_ids),
        "hdf5_identity_sha256": canonical_hash(records),
        "sampled_frame_counts_by_task": sampled_frame_counts,
        "sampled_frame_cost_sha256": canonical_hash(sampled_frame_counts),
        "max_sampled_frames": max(
            value
            for task in sampled_frame_counts.values()
            for value in task.values()
        ),
        "full_sha256_verified": verify_hashes,
        "test_video_values_read": 0,
    }


def inspect_feature_cache(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Fail closed for retired pooled-feature callers."""

    raise WriterModelError(
        "pooled PI05 Writer feature caches are retired; Core-Causal AS-Writer "
        "requires raw teacher video data"
    )


def writer_trainable_contract(
    writer: CompleteLoRAWriter, policy: torch.nn.Module, lora: Any
) -> dict[str, Any]:
    names = sorted(name for name, value in writer.named_parameters() if value.requires_grad)
    parameter_count = sum(value.numel() for value in writer.parameters())
    if (
        not names
        or parameter_count != V8_WRITER_PARAMETER_COUNT
        or any(parameter.requires_grad for parameter in policy.parameters())
    ):
        raise WriterModelError("AS-Writer freeze boundary changed")
    return {
        "object": "shared_action_supervised_writer_only",
        "parameter_count": parameter_count,
        "parameter_name_count": len(names),
        "parameter_names_sha256": canonical_hash(names),
        "generated_lora_parameter_count": lora.parameter_count,
        "generated_lora_tensor_count": lora.state_tensor_count,
        "lora_contract_sha256": canonical_contract_sha256(lora),
        "source_policy_trainable_parameter_count": 0,
    }


def _software_versions() -> dict[str, Any]:
    packages = ("lerobot", "transformers", "peft", "safetensors", "h5py")
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def _contract_stop_step(
    args: argparse.Namespace, config: Mapping[str, Any], total_steps: int
) -> int:
    source = config["formal_run"] if args.mode == "formal" else config["profile_defaults"]
    return int(source.get("selected_stop_step", total_steps))


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
    total_steps: int,
    batch_size: int,
    batch_cycle: Sequence[int],
    checkpoint_steps: Sequence[int],
    initialization: Mapping[str, Any],
) -> dict[str, Any]:
    contract_stop_step = _contract_stop_step(args, config, total_steps)
    videos_per_task_visit = int(
        config["conditioning_training"]["teacher_videos_per_task_visit"]
    )
    tasks_per_rank = int(
        config["conditioning_training"]["tasks_per_rank_per_optimizer_update"]
    )
    update_topology = str(
        config["conditioning_training"].get(
            "update_topology",
            "task_complete_all_tasks",
        )
    )
    task_complete = update_topology == "task_complete_all_tasks"
    global_tasks = context.world_size * tasks_per_rank
    global_queries = global_tasks * batch_size
    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "device": str(context.device),
        "numa_node": context.numa_node,
        "cpu_affinity": list(context.cpu_affinity or ()),
    }
    topology: list[Any] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(topology, local)
    else:
        topology[0] = local
    return {
        "schema_version": AS_WRITER_LAUNCH_SCHEMA,
        "mode": args.mode,
        "stage": writer_stage(config),
        "git": {key: value for key, value in git_state(REPO_ROOT).items() if key in {"branch", "commit"}},
        "config_sha256": sha256_file(args.config.resolve()),
        **(
            {"config_derivation": dict(config["_config_derivation"])}
            if "_config_derivation" in config
            else {}
        ),
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
        **(
            {"initialization": dict(initialization)}
            if initialization.get("mode") == "writer_weight_warm_start"
            else {}
        ),
        "task_ids": list(task_ids),
        "runtime": {
            "world_size": context.world_size,
            "one_policy_cuda_process_per_rank": True,
            "extra_cuda_roles_on_any_rank": 0,
            "ddp_object": "shared_writer_only",
            "macro_step_axis": (
                "complete_task_balanced_optimizer_update"
                if task_complete
                else "rank_rotating_task_balanced_optimizer_update"
            ),
            "tasks_per_rank_per_optimizer_update": tasks_per_rank,
            "global_tasks_per_optimizer_update": global_tasks,
            "task_assignment": (
                "selected_video_frame_cost_balanced_groups_rotated_across_"
                "physical_ranks_longest_task_first_within_each_rank"
                if task_complete
                else "seeded_global_task_permutations_across_rank_step_slots"
            ),
            "task_video_cost_sha256": video_data[
                "sampled_frame_cost_sha256"
            ],
            "action_query_batch_size_per_task": batch_size,
            "action_query_batch_size_per_rank_per_macro": (
                tasks_per_rank * batch_size
            ),
            "per_rank_unique_action_query_cycle": list(batch_cycle),
            "teacher_videos_per_task_visit": videos_per_task_visit,
            "writer_video_conditions_per_rank_per_macro": (
                tasks_per_rank * videos_per_task_visit
            ),
            "actions_per_video_condition": batch_size,
            "action_video_assignment": "all_actions_share_single_video_lora",
            "logical_pairs_per_rank_per_macro": tasks_per_rank * batch_size,
            "optimizer_gradient_accumulation": tasks_per_rank > 1,
            "loss_reduction": (
                "mean_within_each_task_then_equal_mean_across_all_tasks"
                if task_complete
                else "mean_rank_local_task_loss_then_ddp_mean_across_four_tasks"
            ),
            "ddp_no_sync_microtasks_per_macro": tasks_per_rank - 1,
            "ddp_gradient_synchronizations_per_macro": (
                1 if context.world_size > 1 else 0
            ),
            "adamw_updates_per_macro": 1,
            "global_policy_samples_per_macro": global_queries,
            "local_policy_functional_forwards_per_macro": tasks_per_rank,
            "global_policy_functional_forwards_per_macro": global_tasks,
            "writer_conditions_per_rank_per_macro": (
                tasks_per_rank * videos_per_task_visit
            ),
            "total_steps": total_steps,
            "selected_stop_step": contract_stop_step,
            "checkpoint_steps": list(checkpoint_steps),
            "num_workers_per_rank": args.num_workers,
            "rank_topology": topology,
        },
        "trainable": dict(trainable),
        "software": _software_versions(),
    }


def publish_contract(
    args: argparse.Namespace,
    context: DistributedContext,
    contract: Mapping[str, Any],
    contract_sha256: str,
) -> None:
    def operation() -> dict[str, bool]:
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
            raise WriterModelError(f"AS-Writer output directory is not empty: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        contract_path = args.output_dir / "run_contract.json"
        if args.resume is not None:
            if not contract_path.is_file() or canonical_hash(read_json(contract_path)) != contract_sha256:
                raise WriterModelError("AS-Writer resume launch contract changed")
        else:
            write_json_atomic(contract_path, dict(contract))
        append_jsonl(
            args.output_dir / "invocations.jsonl",
            {
                "argv": sys.argv,
                "contract_git": dict(contract["git"]),
                "runtime_git": {
                    key: value
                    for key, value in git_state(REPO_ROOT).items()
                    if key in {"branch", "commit"}
                },
                "contract_compatible_code_resume": bool(
                    args.resume is not None
                    and contract["git"].get("commit")
                    != git_state(REPO_ROOT).get("commit")
                ),
                "contract_selected_stop_step": int(
                    contract["runtime"]["selected_stop_step"]
                ),
                "host": socket.gethostname(),
                "monotonic_stage_extension": bool(
                    args.resume is not None
                    and int(args.stop_after_step)
                    > int(contract["runtime"]["selected_stop_step"])
                ),
                "requested_stop_after_step": int(args.stop_after_step),
                "resume": str(args.resume) if args.resume else None,
                "started_unix": time.time(),
            },
        )
        write_json_atomic(
            args.output_dir / "runtime_paths.json",
            {
                "source_run": str(args.source_run.resolve()),
                "source_checkpoint": str(args.checkpoint.resolve()),
                "writer_initialization_checkpoint": (
                    str(args.initialize_writer_checkpoint.resolve())
                    if args.initialize_writer_checkpoint
                    else None
                ),
                "target_data_root": str(args.data_root.resolve()),
                "tokenizer": str(args.tokenizer_path.resolve()),
            },
        )
        return {"ok": True}

    _broadcast_validation(context, operation)


def reconcile_resume_contract(
    args: argparse.Namespace, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    candidate = dict(candidate)
    if args.resume is None:
        if getattr(args, "allow_contract_compatible_code_resume", False):
            raise WriterModelError(
                "contract-compatible code resume requires a checkpoint"
            )
        return candidate
    contract_path = args.output_dir / "run_contract.json"
    if not contract_path.is_file():
        return candidate
    existing = read_json(contract_path)
    if existing == candidate:
        return existing

    existing_runtime = dict(existing.get("runtime", {}))
    candidate_runtime = dict(candidate.get("runtime", {}))
    existing_stop = int(existing_runtime.get("selected_stop_step", -1))
    candidate_stop = int(candidate_runtime.get("selected_stop_step", -1))
    if (
        existing_stop <= 0
        or candidate_stop < existing_stop
        or candidate_stop > int(existing_runtime.get("total_steps", -1))
    ):
        raise WriterModelError(
            "AS-Writer resume cannot shorten or exceed its sealed stage axis"
        )

    normalized = dict(candidate)
    normalized["runtime"] = {
        **candidate_runtime,
        "selected_stop_step": existing_stop,
    }
    existing_git = existing.get("git", {})
    candidate_git = candidate.get("git", {})
    if existing_git != candidate_git:
        if not getattr(args, "allow_contract_compatible_code_resume", False):
            raise WriterModelError("AS-Writer resume launch contract changed")
        if (
            existing_git.get("branch") != candidate_git.get("branch")
            or existing_git.get("commit") == candidate_git.get("commit")
        ):
            raise WriterModelError(
                "AS-Writer code-compatible resume did not isolate one commit change"
            )
        normalized["git"] = existing_git
    if normalized != existing:
        raise WriterModelError(
            "AS-Writer code-compatible resume changed the scientific contract"
        )
    return existing
