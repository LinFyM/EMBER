"""Launch-contract construction and publication for the canonical v6 runtime."""

from __future__ import annotations

import argparse
import os
import socket
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.expert_manifold.contract import ExpertManifoldError, ExpertTask
from ember.expert_manifold.v6_prior import (
    V6PriorDynamicAnchor,
    V6PriorOwnership,
    V6PriorWarmStart,
)
from ember.expert_manifold.v6_prior_checkpoint import V6_PRIOR_CHECKPOINT_SCHEMA
from ember.expert_manifold.v6_prior_contract import (
    REPO_ROOT,
    V6_PRIOR_CONFIG_SCHEMA,
    V6_PRIOR_RUN_SCHEMA,
    _formal_result_matches,
)
from ember.expert_manifold.v6_prior_policy_batch import policy_runtime_fields
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import DistributedContext, read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.writer.as_sampling import MixedTaskBatchSampler, TeacherVideoSchedule
from ember.writer.topology import visible_physical_cuda_index


V6_PRIOR_TEACHER_AUDIT_RUN_SCHEMA = "ember_pi05_expert_flow_teacher_audit_launch_v1"
V6_PRIOR_TEACHER_AUDIT_SCHEMA = "ember_pi05_expert_flow_teacher_audit_v1"
V6_PRIOR_TEACHER_AUDIT_COMPLETION_SCHEMA = (
    "ember_pi05_expert_flow_teacher_audit_completion_v1"
)

_COMPARISON_CHECKPOINT = (
    "runs/outputs/pi05_v6_tangent_tube_formal_r6_lb20_mb10_"
    "b308941_20260810/checkpoints/macro_00000010"
)
_COMPARISON_COMMIT = "b3089417b6b3df6f22cf3dc296015a80ff228b7a"


def teacher_audit_matches(value: Mapping[str, Any]) -> bool:
    """Validate the exact preregistered audit declaration without touching assets."""

    return (
        value.get("status")
        in {
            "ready_after_tangent_strict_nonpass",
            "sealed_after_matched_flow_teacher_audit",
        }
        and int(value.get("expected_world_size", -1)) == 6
        and int(value.get("tasks_per_rank", -1)) == 4
        and int(value.get("num_workers_per_rank", -1)) == 2
        and int(value.get("diagnostic_macros", -1)) == 1
        and int(value.get("schedule_macro", -1)) == 49
        and int(value.get("logical_policy_batch_size", -1)) == 20
        and int(value.get("functional_policy_microbatch_size", -1)) == 10
        and int(value.get("physical_policy_forwards_per_task", -1)) == 6
        and int(value.get("real_action_dimensions", -1)) == 7
        and value.get("comparison_checkpoint") == _COMPARISON_CHECKPOINT
        and value.get("comparison_commit") == _COMPARISON_COMMIT
        and int(value.get("comparison_macro", -1)) == 10
        and int(value.get("parameter_updates", -1)) == 0
        and int(value.get("rollouts", -1)) == 0
        and value.get("gradient_span_components")
        == ["positive", "completion", "ranking"]
        and float(value.get("gradient_span_pinv_rtol", -1)) == 1e-5
        and float(value.get("gradient_residual_ratio_min", -1)) == 0.25
        and int(value.get("teacher_quality_min_tasks", -1)) == 18
        and int(value.get("teacher_quality_min_suites", -1)) == 3
        and value.get("result") is None
        and value.get("content_hash_policy") == "disabled_by_owner"
    )


def teacher_audit_runtime(config: Mapping[str, Any]) -> tuple[int, tuple[int, ...]]:
    """Return the sole fresh diagnostic segment or fail closed."""

    audit = config.get("teacher_audit", {})
    formal = config.get("formal_run", {})
    if (
        not teacher_audit_matches(audit)
        or audit.get("status") != "ready_after_tangent_strict_nonpass"
        or formal.get("status")
        != "retired_after_macro10_strict_closed_loop_nonpass"
        or not _formal_result_matches(formal.get("formal_result", {}))
    ):
        raise ExpertManifoldError("v6-prior teacher audit is not ready")
    return int(audit["diagnostic_macros"]), ()


def comparison_checkpoint(config: Mapping[str, Any]) -> Path:
    """Resolve and validate the sealed tangent10 comparison decoder asset."""

    audit = config["teacher_audit"]
    checkpoint = (REPO_ROOT / str(audit["comparison_checkpoint"])).resolve()
    manifest = read_json(checkpoint / "manifest.json")
    files = manifest.get("files", {})
    writer_path = checkpoint / "writer.safetensors"
    formal = config["formal_run"]["formal_result"]
    expected = (
        Path(str(formal["training_root"]))
        / "checkpoints"
        / f"macro_{int(audit['comparison_macro']):08d}"
    ).resolve()
    if (
        checkpoint != expected
        or checkpoint.is_symlink()
        or manifest.get("schema_version") != V6_PRIOR_CHECKPOINT_SCHEMA
        or int(manifest.get("next_macro", -1)) != int(audit["comparison_macro"])
        or int(manifest.get("world_size", -1)) != int(audit["expected_world_size"])
        or manifest.get("content_hash_policy") != "disabled_by_owner"
        or manifest.get("checkpoint_contract", {}).get("git_commit")
        != audit["comparison_commit"]
        or not writer_path.is_file()
        or writer_path.is_symlink()
        or writer_path.stat().st_size != int(files.get("writer.safetensors", -1))
    ):
        raise ExpertManifoldError("flow-teacher comparison checkpoint changed")
    return checkpoint


def rank_topology(
    context: DistributedContext,
    physical_index: Any = visible_physical_cuda_index,
) -> list[dict[str, Any]]:
    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_name": torch.cuda.get_device_name(context.device),
        "physical_gpu": physical_index(context.local_rank),
        "device": str(context.device),
        "numa_node": context.numa_node,
        "cpu_affinity": list(context.cpu_affinity or ()),
    }
    rows: list[Any] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(rows, local)
    else:
        rows[0] = local
    return rows


def _initialization_contract(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    warm_start: V6PriorWarmStart,
) -> dict[str, Any]:
    audit = args.mode == "teacher-audit"
    return {
        "mode": "historical_v6_macro400_load_only",
        "checkpoint": str(warm_start.checkpoint),
        "writer_state_tensor_count": warm_start.state_tensor_count,
        "writer_state_value_count": warm_start.state_value_count,
        "dynamic_anchor": str(config["initialization"]["dynamic_anchor"]),
        "resume_writer_load_scope": str(
            config["initialization"]["resume_writer_load_scope"]
        ),
        "optimizer": "not_instantiated_no_update" if audit else "fresh",
        "scheduler": "not_instantiated_no_update" if audit else "fresh",
        "rng": "fresh_seed",
        "comparison_checkpoint": (
            str(comparison_checkpoint(config)) if audit else None
        ),
    }


def _data_contract(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    tasks: Sequence[ExpertTask],
    sampler: MixedTaskBatchSampler,
    video_schedule: TeacherVideoSchedule,
    schedule_start: int,
    schedule_stop: int,
) -> dict[str, Any]:
    return {
        "root": str(args.data_root),
        "tasks": [
            {
                "ordinal": task.ordinal,
                "global_task_id": task.global_task_id,
                "suite": task.suite,
                "task_id": task.task_id,
                "language": task.language,
                "path": str(task.authority.path),
                "bytes": task.authority.expected_bytes,
            }
            for task in tasks
        ],
        **dict(config["data"]),
        "consumed_schedule": video_schedule.consumed_identity_summary(
            sampler, schedule_start, schedule_stop
        ),
    }


def _ownership_contract(
    ownership: V6PriorOwnership,
    dynamic_anchor: V6PriorDynamicAnchor,
    comparison_decoder: V6PriorDynamicAnchor | None,
    trainable_names: Sequence[str],
) -> dict[str, Any]:
    reference = lambda value: {
        "parameter_count": value.parameter_count,
        "tensor_count": value.tensor_count,
        "optimizer_owned": False,
        "checkpoint_owned": False,
        "deployment_owned": False,
    }
    return {
        "frozen_parameter_count": ownership.frozen_parameter_count,
        "trainable_parameter_count": ownership.trainable_parameter_count,
        "frozen_tensor_count": ownership.frozen_tensor_count,
        "trainable_tensor_count": ownership.trainable_tensor_count,
        "trainable_tensor_names": list(trainable_names),
        "source_policy_trainable_parameter_count": 0,
        "dynamic_anchor": reference(dynamic_anchor),
        "comparison_decoder": (
            reference(comparison_decoder) if comparison_decoder is not None else None
        ),
    }


def _runtime_contract(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    segment: Any,
    rank_topology_fn: Any = rank_topology,
) -> dict[str, Any]:
    audit = args.mode == "teacher-audit"
    return {
        "host": socket.gethostname(),
        "device": torch.cuda.get_device_name(context.device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "world_size": context.world_size,
        "tasks_per_rank": 24 // context.world_size,
        "rank_topology": rank_topology_fn(context),
        "total_macros": segment.total_macros,
        "gradient_profile_schedule_macro": (
            segment.schedule_start_macro
            if args.mode in {"gradient-profile", "teacher-audit"}
            else None
        ),
        "checkpoint_macros": list(segment.checkpoint_macros),
        "num_workers_per_rank": args.num_workers,
        "action_loader_prefetch_factor": 2 if args.num_workers else None,
        "action_loader_persistent_workers": args.num_workers > 0,
        **policy_runtime_fields(config),
        "distributed_model_wrapper": "none",
        "gradient_reduction": (
            "four_flat_component_vectors_allreduce_mean_after_local_task_mean"
            if audit
            else "single_flat_parameter_ordered_allreduce_mean_after_local_task_mean"
        ),
        "parameter_updates": 0 if audit else None,
        "rollouts": 0 if audit else None,
        "deferred_process_group": True,
        "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
        "nccl_algo": os.environ.get("NCCL_ALGO"),
        "nccl_proto": os.environ.get("NCCL_PROTO"),
        "cuda_allocator_conf_observed": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }


def build_run_contract(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    segment: Any,
    source: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    tasks: Sequence[ExpertTask],
    sampler: MixedTaskBatchSampler,
    video_schedule: TeacherVideoSchedule,
    expert: Mapping[str, Any],
    warm_start: V6PriorWarmStart,
    ownership: V6PriorOwnership,
    dynamic_anchor: V6PriorDynamicAnchor,
    comparison_decoder: V6PriorDynamicAnchor | None = None,
    trainable_names: Sequence[str],
    git_state_fn: Any = git_state,
    rank_topology_fn: Any = rank_topology,
) -> dict[str, Any]:
    audit = args.mode == "teacher-audit"
    schedule_start = (
        segment.schedule_start_macro
        if args.mode in {"gradient-profile", "teacher-audit"}
        else 0
    )
    schedule_stop = schedule_start + segment.total_macros
    return {
        "schema_version": V6_PRIOR_TEACHER_AUDIT_RUN_SCHEMA if audit else V6_PRIOR_RUN_SCHEMA,
        "mode": args.mode,
        "git": dict(git_state_fn(REPO_ROOT)),
        "config": {
            "path": str(args.config),
            "schema": V6_PRIOR_CONFIG_SCHEMA,
            "bytes": args.config.stat().st_size,
        },
        "source": dict(source),
        "tokenizer": dict(tokenizer),
        "initialization": _initialization_contract(args, config, warm_start),
        "expert_bank": {
            "root": str(args.expert_bank_root),
            "step": int(config["expert_basis"]["expert_step"]),
            "training_commit": expert["training_commit"],
            "tasks": [dict(row) for row in expert["tasks"]],
            "deployment_read": False,
        },
        "data": _data_contract(
            args, config, tasks, sampler, video_schedule, schedule_start, schedule_stop
        ),
        "method": dict(config["method"]),
        "information_wall": dict(config["information_wall"]),
        "writer": dict(config["writer"]),
        "expert_basis": dict(config["expert_basis"]),
        "objective": dict(config["objective"]),
        "teacher_audit": dict(config["teacher_audit"]) if audit else None,
        "optimization": dict(config["optimization"]),
        "ownership": _ownership_contract(
            ownership, dynamic_anchor, comparison_decoder, trainable_names
        ),
        "runtime": _runtime_contract(
            args, config, context, segment, rank_topology_fn=rank_topology_fn
        ),
        "content_hash_policy": "disabled_by_owner",
    }


def checkpoint_contract(run_contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_schema": run_contract["schema_version"],
        "mode": run_contract["mode"],
        "git_commit": run_contract["git"]["commit"],
        "config": run_contract["config"],
        "source": run_contract["source"],
        "initialization": run_contract["initialization"],
        "expert_bank_root": run_contract["expert_bank"]["root"],
        "expert_step": run_contract["expert_bank"]["step"],
        "objective": run_contract["objective"],
        "ownership": run_contract["ownership"],
        "world_size": run_contract["runtime"]["world_size"],
    }


def cursor_contract(config: Mapping[str, Any], macro: int) -> dict[str, Any]:
    data = config["data"]
    return {
        "next_macro": macro,
        "task_visits_per_task": macro,
        "sampler_seed": int(data["sampler_seed"]),
        "teacher_video_seed": int(data["teacher_video_seed"]),
        "counterfactual_seed": int(data["counterfactual_seed"]),
        "counterfactual_phase": macro % 3,
        "videos_per_task_visit": 1,
        "action_queries_per_task": int(data["action_queries_per_task"]),
    }


def publish_contract(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    context: DistributedContext,
) -> None:
    path = args.output_dir / "run_contract.json"
    if context.is_main:
        if args.resume is None:
            if args.output_dir.exists() and any(args.output_dir.iterdir()):
                raise ExpertManifoldError("fresh v6-prior output is not empty")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(path, dict(contract))
        elif (
            args.resume.parent.parent.resolve() != args.output_dir
            or not path.is_file()
            or read_json(path) != dict(contract)
        ):
            raise ExpertManifoldError("v6-prior resume run contract changed")
        append_jsonl(
            args.output_dir / "invocations.jsonl",
            {
                "argv": os.sys.argv,
                "started_unix": time.time(),
                "resume": str(args.resume) if args.resume else None,
                "requested_stop_after_macro": args.stop_after_macro,
            },
        )
    if context.world_size > 1:
        dist.barrier(device_ids=[context.local_rank])
