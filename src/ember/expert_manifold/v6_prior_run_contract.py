"""Launch, cursor, and ownership records for the residual Writer."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import time
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.expert_manifold.contract import ExpertManifoldError, ExpertTask
from ember.expert_manifold.v6_prior import V6PriorOwnership, V6PriorWarmStart
from ember.expert_manifold.v6_prior_contract import (
    V6_PRIOR_CONFIG_SCHEMA,
    V6_PRIOR_RUN_SCHEMA,
)
from ember.pi05_source_checkpoint import DistributedContext, read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.writer.as_sampling import MixedTaskBatchSampler, TeacherVideoSchedule
from ember.writer.condition_update import (
    FrozenV6ConditionResidualWriter,
    ProgramReconciliationState,
)
from ember.writer.topology import visible_physical_cuda_index


def residual_git_state(repo_root: Any) -> dict[str, Any]:
    """Seal either the tracked main branch or its clean detached worktree."""

    def run(*arguments: str, check: bool = True) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=check,
            text=True,
            capture_output=True,
        ).stdout.strip()

    branch = run("branch", "--show-current")
    upstream = run(
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
        check=False,
    )
    authority_ref = "origin/codex/bci-continuation"
    commit = run("rev-parse", "HEAD")
    authority_commit = run("rev-parse", authority_ref)
    authority_contains_commit = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, authority_commit],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    ).returncode == 0
    return {
        "branch": branch,
        "commit": commit,
        "origin_main": run("rev-parse", "origin/main"),
        "upstream": upstream or None,
        "upstream_commit": run("rev-parse", upstream) if upstream else None,
        "authority_ref": authority_ref,
        "authority_commit": authority_commit,
        "authority_contains_commit": authority_contains_commit,
        "dirty_paths": run("status", "--porcelain").splitlines(),
    }


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
    return [dict(row) for row in rows]


def _data_contract(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    tasks: Sequence[ExpertTask],
    sampler: MixedTaskBatchSampler,
    video_schedule: TeacherVideoSchedule,
    *,
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
    writer: FrozenV6ConditionResidualWriter,
    reconciliation: ProgramReconciliationState,
) -> dict[str, Any]:
    memory = writer.program_memory.value
    return {
        "historical_v6_base": {
            "state_tensor_count": ownership.state_tensor_count,
            "parameter_tensor_count": ownership.frozen_parameter_tensor_count,
            "parameter_count": ownership.frozen_parameter_count,
            "trainable_parameter_count": 0,
            "checkpoint_owned": False,
            "deployment_owned": True,
        },
        "fixed_projection": {
            "shape": list(writer.condition_feature.projection.shape),
            "dtype": str(writer.condition_feature.projection.dtype),
            "trainable": False,
            "persistent": False,
            "checkpoint_owned": False,
        },
        "program_residual_memory": {
            "shape": list(memory.shape),
            "dtype": str(memory.dtype),
            "value_count": memory.numel(),
            "trainable": False,
            "manual_update": True,
            "checkpoint_owned": True,
            "deployment_owned": True,
        },
        "reconciliation_precision": {
            "shape": list(reconciliation.precision.shape),
            "dtype": str(reconciliation.precision.dtype),
            "value_count": reconciliation.precision.numel(),
            "trainable": False,
            "checkpoint_owned": True,
            "deployment_owned": False,
        },
        "source_policy_trainable_parameter_count": 0,
        "optimizer": "not_instantiated",
        "scheduler": "not_instantiated",
        "scaler": "not_instantiated",
    }


def decision_evaluation_contract(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Pin the only strict roots allowed to authorize formal continuation."""

    if args.mode != "formal":
        return {
            "macro0_reference_root": None,
            "macro0_reference_commit": None,
            "macro10_registered_root": None,
            "support_gate": None,
        }
    reference = config["formal_run"]["decision_evaluation"]
    gates = config["formal_run"]["decision_gates"]
    macro0_root = getattr(args, "macro0_evaluation_root", None)
    macro10_root = getattr(args, "macro10_evaluation_root", None)
    return {
        "macro0_reference_root": (
            str(macro0_root) if macro0_root is not None else None
        ),
        "macro0_reference_commit": reference["macro0_reference_commit"],
        "macro10_registered_root": (
            str(macro10_root) if macro10_root is not None else None
        ),
        "support_gate": {
            name: gates[name]
            for name in (
                "macro10_support_correct_min",
                "macro10_support_lost_to_macro0_max",
                "macro10_support_breadth_min",
            )
        },
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
    warm_start: V6PriorWarmStart,
    ownership: V6PriorOwnership,
    writer: FrozenV6ConditionResidualWriter,
    reconciliation: ProgramReconciliationState,
    repo_root: Any,
    git_state_fn: Any = residual_git_state,
    rank_topology_fn: Any = rank_topology,
) -> dict[str, Any]:
    schedule_start = segment.schedule_origin
    schedule_stop = schedule_start + segment.total_macros
    git = dict(git_state_fn(repo_root))
    return {
        "schema_version": V6_PRIOR_RUN_SCHEMA,
        "mode": args.mode,
        "git": {
            "branch": git["branch"],
            "commit": git["commit"],
            "authority_ref": git["authority_ref"],
            "dirty_paths": list(git["dirty_paths"]),
        },
        "config": {
            "path": str(args.config),
            "schema": V6_PRIOR_CONFIG_SCHEMA,
            "bytes": args.config.stat().st_size,
        },
        "source": dict(source),
        "tokenizer": dict(tokenizer),
        "initialization": {
            "mode": "strict_historical_v6_macro400_all_frozen",
            "checkpoint": str(warm_start.checkpoint),
            "writer_state_tensor_count": warm_start.state_tensor_count,
            "writer_state_value_count": warm_start.state_value_count,
            "residual_memory": (
                "fresh_zero_and_identity_reconciliation_then_joint_exact_resume"
            ),
        },
        "data": _data_contract(
            args,
            config,
            tasks,
            sampler,
            video_schedule,
            schedule_start=schedule_start,
            schedule_stop=schedule_stop,
        ),
        "method": dict(config["method"]),
        "information_wall": dict(config["information_wall"]),
        "writer": dict(config["writer"]),
        "condition_feature": dict(config["condition_feature"]),
        "program_residual": dict(config["program_residual"]),
        "reconciliation": dict(config["reconciliation"]),
        "update": dict(config["update"]),
        "objective": dict(config["objective"]),
        "optimization": dict(config["optimization"]),
        "ownership": _ownership_contract(ownership, writer, reconciliation),
        "decision_evaluation": decision_evaluation_contract(args, config),
        "runtime": {
            "host": socket.gethostname(),
            "device": torch.cuda.get_device_name(context.device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "world_size": context.world_size,
            "tasks_per_rank": 24 // context.world_size,
            "rank_topology": rank_topology_fn(context),
            "total_macros": segment.total_macros,
            "schedule_origin": segment.schedule_origin,
            "checkpoint_macros": list(segment.checkpoint_macros),
            "num_workers_per_rank": args.num_workers,
            "action_loader_prefetch_factor": 2 if args.num_workers else None,
            "action_loader_persistent_workers": args.num_workers > 0,
            "logical_policy_batch_size": 20,
            "functional_policy_microbatch_size": 10,
            "physical_policy_forwards_per_task": 2,
            "negative_policy_forwards_per_task": 0,
            "policy_gradient_checkpointing": False,
            "writer_activation_checkpointing_effective": False,
            "distributed_model_wrapper": "none",
            "collectives": "two_all_gathers_no_memory_allreduce",
            "deferred_process_group": True,
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
            "nccl_algo": os.environ.get("NCCL_ALGO"),
            "nccl_proto": os.environ.get("NCCL_PROTO"),
            "cuda_allocator_conf_observed": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF"
            ),
        },
        "content_hash_policy": "disabled_by_owner",
    }


def checkpoint_contract(run_contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_schema": run_contract["schema_version"],
        "mode": run_contract["mode"],
        "git_commit": run_contract["git"]["commit"],
        "config": run_contract["config"],
        "source": run_contract["source"],
        "initialization": {
            name: run_contract["initialization"][name]
            for name in (
                "mode",
                "checkpoint",
                "writer_state_tensor_count",
                "writer_state_value_count",
                "residual_memory",
            )
        },
        "condition_feature": run_contract["condition_feature"],
        "program_residual": run_contract["program_residual"],
        "reconciliation": run_contract["reconciliation"],
        "update": run_contract["update"],
        "ownership": run_contract["ownership"],
        "world_size": run_contract["runtime"]["world_size"],
        "rank_topology": run_contract["runtime"]["rank_topology"],
        "content_hash_policy": "disabled_by_owner",
    }


def cursor_contract(config: Mapping[str, Any], macro: int) -> dict[str, Any]:
    if macro < 0:
        raise ExpertManifoldError("residual Writer cursor is negative")
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
        "full48_order": "correct_0_to_23_then_negative_0_to_23",
        "assimilated_rows": macro * int(config["reconciliation"]["rows_per_macro"]),
    }


def publish_contract(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    context: DistributedContext,
    *,
    continuation_gate_evidence: Mapping[str, Any] | None = None,
) -> None:
    path = args.output_dir / "run_contract.json"
    payload: list[Any] = [None]
    if context.is_main:
        try:
            if args.resume is None:
                if args.output_dir.exists() and any(args.output_dir.iterdir()):
                    raise ExpertManifoldError(
                        "fresh residual Writer output is not empty"
                    )
                args.output_dir.mkdir(parents=True, exist_ok=True)
                write_json_atomic(path, dict(contract))
            elif (
                args.resume.parent.parent.resolve() != args.output_dir
                or not path.is_file()
                or read_json(path) != dict(contract)
            ):
                raise ExpertManifoldError(
                    "residual Writer resume run contract changed"
                )
            append_jsonl(
                args.output_dir / "invocations.jsonl",
                {
                    "argv": os.sys.argv,
                    "started_unix": time.time(),
                    "resume": str(args.resume) if args.resume else None,
                    "requested_stop_after_macro": args.stop_after_macro,
                    "continuation_gate_evidence": (
                        dict(continuation_gate_evidence)
                        if continuation_gate_evidence is not None
                        else None
                    ),
                },
            )
            payload[0] = {"ok": True}
        except Exception as error:
            payload[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=context.device)
    if not isinstance(payload[0], Mapping) or payload[0].get("error"):
        raise ExpertManifoldError(
            "residual Writer launch contract publication failed: "
            f"{payload[0]}"
        )
