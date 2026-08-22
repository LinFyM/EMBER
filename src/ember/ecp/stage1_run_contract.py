"""Run-contract publication and exact-resume checks for canonical ECP Stage 1."""

from __future__ import annotations

import os
import socket
import sys
from typing import Any, Mapping

import torch.distributed as dist

from ember.ecp.stage1_config import (
    REPO_ROOT,
    RUN_SCHEMA,
    STAGE,
    stage1_asset_authority,
)
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import barrier, read_json, write_json_atomic


def build_stage1_run_contract(
    runtime: Any, source: Mapping[str, Any]
) -> dict[str, Any]:
    support_manifest = stage1_asset_authority(
        runtime.config, "policy_support_bank", runtime.args.asset_root
    )
    libero_assets = stage1_asset_authority(
        runtime.config, "libero_assets_root", runtime.args.asset_root
    )
    local = {
        "rank": runtime.context.rank,
        "local_rank": runtime.context.local_rank,
        "device": str(runtime.context.device),
        "numa_node": runtime.context.numa_node,
        "cpu_affinity": list(runtime.context.cpu_affinity or ()),
    }
    topology: list[Any] = [None] * runtime.context.world_size
    if runtime.context.world_size > 1:
        dist.all_gather_object(topology, local)
    else:
        topology[0] = local
    state = git_state(REPO_ROOT)
    return {
        "schema_version": RUN_SCHEMA,
        "stage": STAGE,
        "mode": runtime.args.mode,
        "command": list(sys.argv),
        "git": {"branch": state["branch"], "commit": state["commit"]},
        "host": socket.gethostname(),
        "config": {
            "path": str(runtime.args.config),
            "bytes": runtime.args.config.stat().st_size,
        },
        "source": dict(source),
        "asset_root": str(runtime.args.asset_root),
        "data_root": str(runtime.args.data_root),
        "tokenizer": {
            "path": str(runtime.args.tokenizer_path),
            "bytes": runtime.args.tokenizer_path.stat().st_size,
        },
        "observer_authority": {**runtime.observer_authority, "frozen": True},
        "policy_support_bank": {
            "path": str(support_manifest.resolve()),
            "bytes": support_manifest.stat().st_size,
        },
        "libero_assets": {
            "path": str(libero_assets.resolve()),
            "directory": libero_assets.is_dir(),
        },
        "initialization": dict(runtime.initialization),
        "tasks": [
            {
                "ordinal": task.ordinal,
                "global_task_id": task.global_task_id,
                "suite": task.suite,
                "task_id": task.task_id,
                "fold_role": task.fold_role,
                "asset_key": task.asset_key,
                "domain": task.domain,
            }
            for task in runtime.tasks
        ],
        "successful_members": len(runtime.evidence_bank.members),
        "model": dict(runtime.config["model"]),
        "data": dict(runtime.config["data"]),
        "objective": dict(runtime.config["objective"]),
        "prior_calibration": dict(runtime.config["prior_calibration"]),
        "structured_calibration": dict(runtime.config["structured_calibration"]),
        "environment": dict(runtime.config["environment"]),
        "optimization": dict(runtime.config["optimization"]),
        "information_wall": dict(runtime.config["information_wall"]),
        "runtime": {
            "world_size": runtime.context.world_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
            "topology": topology,
            "start_task_visits": runtime.start_task_visits,
            "stop_after_task_visits": runtime.stop_after_task_visits,
            "loaded_policy_support_tasks_by_rank": len(runtime.support_bank.tasks),
            "loaded_policy_support_panels_by_rank": len(runtime.support_panels),
            "structured_calibration_pending": runtime.calibration.pending,
            "structured_calibration_assignments": [
                list(values) for values in runtime.calibration.assignments
            ],
        },
        "trainable_parameters": sum(
            value.numel() for value in runtime.trainable_parameters
        ),
        "trainable_modules": ["policy_teacher", "compiler"],
        "frozen_writer_modules": ["visible_program"],
        "source_policy_trainable_parameters": 0,
        "observer_trainable_parameters": 0,
        "content_hash_policy": "disabled_by_owner",
    }


def initialize_stage1_run_contract(
    runtime: Any, source: Mapping[str, Any]
) -> None:
    contract = build_stage1_run_contract(runtime, source)
    path = runtime.args.output_dir / "run_contract.json"
    if runtime.context.is_main and runtime.args.resume is None:
        runtime.args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(path, contract)
    elif runtime.context.is_main:
        existing = read_json(path)
        if (
            existing.get("schema_version") != RUN_SCHEMA
            or existing.get("stage") != STAGE
            or existing.get("git", {}).get("commit") != contract["git"]["commit"]
            or existing.get("config", {}).get("bytes") != contract["config"]["bytes"]
            or existing.get("source", {}).get("checkpoint")
            != contract["source"]["checkpoint"]
            or existing.get("runtime", {}).get("world_size")
            != runtime.context.world_size
        ):
            raise ValueError("ECP Stage 1 resume run contract changed")
    barrier(runtime.context)
