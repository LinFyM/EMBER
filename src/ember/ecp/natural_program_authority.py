"""Run provenance and information-wall authority for G2 Natural Program."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import (
    DistributedContext,
    read_json,
    write_json_atomic,
)
from ember.pi05_source_contract import append_jsonl
from ember.writer.meta_lora import MetaLoRAProjection, MetaLoRAStack


RUN_SCHEMA = "ember_ecp_natural_program_g2_run_v3"


def _run_authority_commit(repository: Mapping[str, Any], mode: str) -> str:
    # Formal G2 already requires a clean detached commit contained by
    # origin/main.  Exact resume must keep that frozen authority stable when
    # origin/main later advances with documentation or other integrated work.
    return str(
        repository["commit"]
        if mode == "formal"
        else repository["authority_commit"]
    )


def _pure_native_inventory(
    policy: torch.nn.Module, model: torch.nn.Module
) -> dict[str, Any]:
    action_meta_modules = [
        f"{prefix}.{name}:{type(module).__name__}"
        for root, prefix in ((policy, "policy"), (model, "natural_program"))
        for name, module in root.named_modules()
        if isinstance(module, (MetaLoRAStack, MetaLoRAProjection))
    ]
    policy_trainable = [
        name for name, value in policy.named_parameters() if value.requires_grad
    ]
    observer_trainable = [
        name for name, value in model.encoder.named_parameters() if value.requires_grad
    ]
    observer_training = bool(model.encoder.training)
    if (
        action_meta_modules
        or policy_trainable
        or observer_trainable
        or observer_training
    ):
        raise ValueError(
            "G2 accidentally loaded Action Meta or a trainable frozen authority"
        )
    return {
        "loader": "load_frozen_native_observer_then_train_program_heads",
        "action_meta_argument": None,
        "install_action_meta_lora": False,
        "action_meta_module_instances": action_meta_modules,
        "action_meta_module_count": 0,
        "action_meta_parameter_count": 0,
        "source_policy_trainable_parameters": policy_trainable,
        "source_policy_trainable_parameter_count": 0,
        "native_observer_trainable_parameters": observer_trainable,
        "native_observer_trainable_parameter_count": 0,
        "native_observer_training": observer_training,
        "natural_program_trainable_parameter_count": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
        "natural_program_parameter_tensors": sum(
            value.requires_grad for value in model.parameters()
        ),
    }


def _topology(context: DistributedContext) -> list[Any]:
    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "device": str(context.device),
        "numa_node": context.numa_node,
        "cpu_affinity": list(context.cpu_affinity or ()),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    rows: list[Any] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(rows, local)
    else:
        rows[0] = local
    return rows


def _behavior_contract(
    runtime_args: argparse.Namespace,
    config: Mapping[str, Any],
    behavior_codes: Any | None,
) -> dict[str, Any]:
    if behavior_codes is None:
        return {"behavior_alignment": None, "behavior_contract": None}
    return {
        "behavior_alignment": {
            "schema_version": behavior_codes.manifest["schema_version"],
            "manifest": str(
                runtime_args.asset_root / config["authorities"]["behavior_codes"]
            ),
            "tensor_bytes": int(behavior_codes.manifest["tensor_bytes"]),
            "fit_tasks": len(behavior_codes.fit_task_ids),
            "held_zero_gradient_tasks": len(behavior_codes.held_task_ids),
            "selected_targets": list(behavior_codes.selected_targets),
            "dimension": behavior_codes.dimension,
            "deployment_forward_reads_behavior_codes": False,
        },
        "behavior_contract": dict(config["behavior_alignment"]),
    }


def build_natural_program_run_contract(
    *,
    runtime_args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    tasks: Sequence[Any],
    source: Mapping[str, Any],
    policy: torch.nn.Module,
    model: torch.nn.Module,
    total_macros: int,
    checkpoint_macros: tuple[int, ...],
    optimizer_steps_per_macro: int,
    tasks_per_role_per_optimizer_step: int,
    repo_root: Path,
    native_checkpoint: Path,
    behavior_codes: Any | None,
    initialization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    repository = git_state(repo_root)
    native_weights = native_checkpoint / "ecp.safetensors"
    return {
        "schema_version": RUN_SCHEMA,
        "stage": "g2_natural_program",
        "mode": runtime_args.mode,
        "git": {
            "branch": repository["branch"],
            "commit": repository["commit"],
            "authority_commit": _run_authority_commit(
                repository, runtime_args.mode
            ),
        },
        "config_path": str(runtime_args.config),
        "source": dict(source),
        "tokenizer": {
            "path": str(runtime_args.tokenizer_path),
            "bytes": runtime_args.tokenizer_path.stat().st_size,
        },
        "data_root": str(runtime_args.data_root),
        "label_authority": {
            "root": str(runtime_args.label_root),
            "manifest_bytes": (runtime_args.label_root / "manifest.json").stat().st_size,
            "deployment_forward_reads_labels": False,
        },
        **_behavior_contract(runtime_args, config, behavior_codes),
        "initialization": dict(initialization) if initialization else None,
        "native_observer": {
            "checkpoint": str(native_checkpoint),
            "checkpoint_macro": 10,
            "weights_bytes": native_weights.stat().st_size,
            **_pure_native_inventory(policy, model),
        },
        "information_wall": dict(config["information_wall"]),
        "fold": dict(config["fold"]),
        "tasks": [
            {
                "authority_id": task.authority_id,
                "domain": task.domain,
                "domain_task_id": task.domain_task_id,
                "role": task.role,
                "language": task.language,
                "path": str(task.path),
                "bytes": task.expected_bytes,
            }
            for task in tasks
        ],
        "model": dict(config["model"]),
        "data": dict(config["data"]),
        "objective": dict(config["objective"]),
        "optimization": dict(config["optimization"]),
        "runtime": {
            "world_size": context.world_size,
            "topology": _topology(context),
            "total_macros": total_macros,
            "checkpoint_macros": list(checkpoint_macros),
            "optimizer_steps_per_macro": optimizer_steps_per_macro,
            "total_optimizer_steps": total_macros * optimizer_steps_per_macro,
            "tasks_per_role_per_optimizer_step": (
                tasks_per_role_per_optimizer_step
            ),
            "global_tasks_per_macro": (
                38
                if runtime_args.mode == "formal"
                else int(config["profile_defaults"]["tasks_per_rank_per_macro"])
                * context.world_size
            ),
            "task_weight": (
                "one visit per task per macro; every optimizer step is role-"
                "balanced and task-mean weighted; the short tail pair rotates"
                if runtime_args.mode == "formal"
                else "one role-balanced optimizer group for execution smoke; "
                "task-mean weighted"
            ),
            "assignment": (
                "role-balanced optimizer groups then cost-balanced uneven ranks"
            ),
            "contrastive_negatives": (
                "fixed_count_role_balanced_fit_language_content_"
                "independent_of_rank_and_world_size"
            ),
        },
        "gradient_wall": {
            "fit_roles": ["meta_fit", "target_fit"],
            "held_roles": ["meta_held", "target_held"],
            "held_task_gradient_count": 0,
            "source_policy_trainable_parameter_count": 0,
            "native_observer_trainable_parameter_count": 0,
        },
    }


def publish_natural_program_run_contract(
    args: argparse.Namespace,
    context: DistributedContext,
    contract: Mapping[str, Any],
) -> None:
    result: list[Any] = [None]
    if context.is_main:
        try:
            path = args.output_dir / "run_contract.json"
            if args.resume is None:
                if args.output_dir.exists() and any(args.output_dir.iterdir()):
                    raise ValueError("fresh G2 output directory is not empty")
                args.output_dir.mkdir(parents=True, exist_ok=True)
                write_json_atomic(path, contract)
            elif not path.is_file() or read_json(path) != contract:
                raise ValueError("exact-resume G2 run contract changed")
            append_jsonl(
                args.output_dir / "invocations.jsonl",
                {
                    "argv": sys.argv,
                    "host": socket.gethostname(),
                    "resume": str(args.resume) if args.resume else None,
                    "started_unix": time.time(),
                },
            )
            result[0] = {"ok": True}
        except Exception as error:
            result[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(result, src=0, device=context.device)
    if result[0].get("error"):
        raise ValueError(result[0]["error"])
