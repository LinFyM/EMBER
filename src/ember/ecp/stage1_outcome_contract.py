"""Formal run contract for OCPB Stage 1 calibration."""

from __future__ import annotations

import os
import socket
from typing import TYPE_CHECKING, Any, Mapping

import torch.distributed as dist

from ember.ecp.stage1_outcome_config import (
    RUN_SCHEMA,
    STAGE,
    outcome_repo_authority,
)
from ember.ecp.stage1_training import REPO_ROOT
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import barrier, read_json, write_json_atomic

if TYPE_CHECKING:
    from ember.ecp.stage1_outcome_training import ECPStage1OutcomeRuntime


def _run_contract(
    runtime: "ECPStage1OutcomeRuntime",
    *,
    source: Mapping[str, Any],
    initialization: Mapping[str, Any],
) -> dict[str, Any]:
    repository = git_state(REPO_ROOT)
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
    return {
        "schema_version": RUN_SCHEMA,
        "stage": STAGE,
        "mode": runtime.args.mode,
        "git": {"branch": repository["branch"], "commit": repository["commit"]},
        "host": socket.gethostname(),
        "config": {
            "path": str(runtime.args.config),
            "bytes": runtime.args.config.stat().st_size,
        },
        "base_stage1_config": str(
            outcome_repo_authority(runtime.config, "base_stage1_config").resolve()
        ),
        "policy_support_bank": str(
            (runtime.support_bank.root / "manifest.json").resolve()
        ),
        "source": dict(source),
        "initialization": dict(initialization),
        "asset_root": str(runtime.args.asset_root),
        "data_root": str(runtime.args.data_root),
        "tokenizer": str(runtime.args.tokenizer_path),
        "tasks": {
            "role": "train24_fit19_outcome_calibration",
            "count": len(runtime.tasks),
            "ordinals": [task.ordinal for task in runtime.tasks],
            "global_task_ids": [task.global_task_id for task in runtime.tasks],
            "equal_weight_per_macro": True,
            "assignments": [list(values) for values in runtime.assignments],
            "held_zero_reward_ordinals": list(
                runtime.config["roles"]["held_task_ordinals"]
            ),
        },
        "outcome_calibration": dict(runtime.config["outcome_calibration"]),
        "functional_anchor": {
            **dict(runtime.base_config["objective"]),
            "support_preservation": runtime.config["outcome_calibration"][
                "support_preservation"
            ],
        },
        "environment": dict(runtime.config["environment"]),
        "optimization": dict(runtime.config["optimization"]),
        "information_wall": dict(runtime.config["information_wall"]),
        "runtime": {
            "world_size": runtime.context.world_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
            "topology": topology,
        },
        "trainable_parameters": sum(
            parameter.numel() for parameter in runtime.trainable_parameters
        ),
        "source_policy_trainable_parameters": 0,
        "observer_trainable_parameters": 0,
        "content_hash_policy": "disabled_by_owner",
    }


def publish_outcome_run_contract(
    runtime: "ECPStage1OutcomeRuntime",
    *,
    source: Mapping[str, Any],
    initialization: Mapping[str, Any],
) -> None:
    contract = _run_contract(
        runtime, source=source, initialization=initialization
    )
    path = runtime.args.output_dir / "run_contract.json"
    if runtime.context.is_main:
        if runtime.args.resume is None:
            if runtime.args.output_dir.exists():
                raise ValueError("fresh outcome output directory already exists")
            runtime.args.output_dir.mkdir(parents=True)
            write_json_atomic(path, contract)
        elif not path.is_file() or read_json(path) != contract:
            raise ValueError("outcome resume run contract changed")
    barrier(runtime.context)
