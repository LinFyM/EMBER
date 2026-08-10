"""Launch, ownership, and exact-resume records for Reward-Credit Program writes."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.expert_manifold.contract import ExpertManifoldError, ExpertTask
from ember.expert_manifold.v6_prior import V6PriorOwnership, V6PriorWarmStart
from ember.expert_manifold.v6_prior_contract import (
    V6_PRIOR_CONFIG_SCHEMA,
    V6_PRIOR_RUN_SCHEMA,
)
from ember.pi05_source_checkpoint import (
    DistributedContext,
    read_json,
    write_json_atomic,
)
from ember.pi05_source_contract import append_jsonl
from ember.writer.as_sampling import TeacherVideoSchedule
from ember.writer.condition_update import (
    FrozenV6ConditionResidualWriter,
    ProgramReconciliationState,
)
from ember.writer.topology import visible_physical_cuda_index


def residual_git_state(repo_root: Path) -> dict[str, Any]:
    """Seal either the clean authority branch or a clean detached descendant."""

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
    contains = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, authority_commit],
            cwd=repo_root,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    return {
        "branch": branch,
        "commit": commit,
        "upstream": upstream or None,
        "upstream_commit": run("rev-parse", upstream) if upstream else None,
        "authority_ref": authority_ref,
        "authority_commit": authority_commit,
        "authority_contains_commit": contains,
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
    args: argparse.Namespace, config: Mapping[str, Any]
) -> dict[str, Any]:
    decision = config["formal_run"]["decision_evaluation"]
    repo_root = Path(__file__).resolve().parents[3]

    def roots(name: str) -> dict[str, str]:
        return {
            condition: str((repo_root / path).resolve())
            for condition, path in decision[name].items()
        }

    return {
        "macro0_reference_root": str(
            (repo_root / decision["macro0_reference_root"]).resolve()
        ),
        "macro0_reference_commit": decision["macro0_reference_commit"],
        "macro1_registered_root": str(
            (repo_root / decision["macro1_registered_root"]).resolve()
        ),
        "macro2_registered_root": str(
            (repo_root / decision["macro2_registered_root"]).resolve()
        ),
        "macro1_control_registered_roots": roots("macro1_control_registered_roots"),
        "macro2_control_registered_roots": roots("macro2_control_registered_roots"),
        "support_gate": dict(config["formal_run"]["decision_gates"]),
        "active_for_this_invocation": args.mode == "formal",
    }


def _data_contract(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    tasks: Sequence[ExpertTask],
    video_schedule: TeacherVideoSchedule,
    *,
    total_macros: int,
) -> dict[str, Any]:
    demos = []
    for macro in range(total_macros):
        demos.append(
            {
                str(task.global_task_id): int(
                    video_schedule.demos_for_task_visit(task.global_task_id, macro)[0]
                )
                for task in tasks
            }
        )
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
        "scheduled_teacher_demos": demos,
        "teacher_action_reads": 0,
        "source_action_reads": 0,
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
    video_schedule: TeacherVideoSchedule,
    warm_start: V6PriorWarmStart,
    ownership: V6PriorOwnership,
    writer: FrozenV6ConditionResidualWriter,
    reconciliation: ProgramReconciliationState,
    repo_root: Path,
    git_state_fn: Any = residual_git_state,
    rank_topology_fn: Any = rank_topology,
) -> dict[str, Any]:
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
            video_schedule,
            total_macros=segment.total_macros,
        ),
        "method": dict(config["method"]),
        "information_wall": dict(config["information_wall"]),
        "writer": dict(config["writer"]),
        "condition_feature": dict(config["condition_feature"]),
        "program_residual": dict(config["program_residual"]),
        "reconciliation": dict(config["reconciliation"]),
        "update": dict(config["update"]),
        "environment": dict(config["environment"]),
        "objective": dict(config["objective"]),
        "rng": dict(config["rng"]),
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
            "rollout_policy_batch_size": 4,
            "reward_replay_chunk_batch_size": int(
                config["optimization"]["reward_replay_chunk_batch_size"]
            ),
            "flow_mc_samples": 4,
            "old_policy_forwards": 0,
            "negative_policy_forwards": 0,
            "distributed_model_wrapper": "none",
            "collectives": ("cuda_complete_file_rendezvous_then_two_fixed_all_gathers"),
            "deferred_process_group": True,
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
            "nccl_algo": os.environ.get("NCCL_ALGO"),
            "nccl_proto": os.environ.get("NCCL_PROTO"),
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
            "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            "cuda_allocator_conf_observed": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
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
        "environment": run_contract["environment"],
        "objective": run_contract["objective"],
        "rng": run_contract["rng"],
        "ownership": run_contract["ownership"],
        "world_size": run_contract["runtime"]["world_size"],
        "rank_topology": run_contract["runtime"]["rank_topology"],
        "content_hash_policy": "disabled_by_owner",
    }


def cursor_contract(config: Mapping[str, Any], macro: int) -> dict[str, Any]:
    if type(macro) is not int or macro < 0:
        raise ExpertManifoldError("Reward-Credit Writer cursor is invalid")
    return {
        "next_macro": macro,
        "macro_semantics": "one_complete_full24_reward_cycle",
        "completed_full24_cycles": macro,
        "task_visits_per_task": macro,
        "video_visits_per_task": macro,
        "global_rollouts": macro * 24 * 4,
        "rollouts_per_task": macro * 4,
        "flow_panels_per_task": macro,
        "program_updates": macro,
        "metrics_rows": macro,
        "teacher_video_seed": int(config["data"]["teacher_video_seed"]),
        "environment_seed_root": int(config["rng"]["environment_seed_root"]),
        "policy_noise_seed_root": int(config["rng"]["policy_noise_seed_root"]),
        "flow_credit_seed_root": int(config["rng"]["flow_credit_seed_root"]),
        "counterfactual_seed": int(config["data"]["counterfactual_seed"]),
        "counterfactual_phase": macro % 3,
        "full48_order": "correct_0_to_23_then_negative_0_to_23",
        "assimilated_rows": macro * int(config["reconciliation"]["rows_per_macro"]),
        "pending_environment_episodes": 0,
        "pending_policy_action_chunks": 0,
        "pending_replay_microbatches": 0,
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
                    raise ExpertManifoldError("fresh Reward-Credit output is not empty")
                args.output_dir.mkdir(parents=True, exist_ok=True)
                write_json_atomic(path, dict(contract))
            elif (
                args.resume.parent.parent.resolve() != args.output_dir
                or not path.is_file()
                or read_json(path) != dict(contract)
            ):
                raise ExpertManifoldError("Reward-Credit resume run contract changed")
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
            "Reward-Credit launch contract publication failed: " f"{payload[0]}"
        )
