"""Run provenance and information-wall inventory for G3."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.ecp.native_factors import native_capture_modes
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import DistributedContext, read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.writer.meta_lora import MetaLoRAProjection, MetaLoRAStack


RUN_SCHEMA = "ember_ecp_shared_compiler_g3_run_v2"


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


def pure_shared_compiler_inventory(
    *,
    policy: torch.nn.Module,
    program: torch.nn.Module,
    compiler: torch.nn.Module,
    owners: Sequence[Any],
) -> dict[str, Any]:
    action_meta = [
        f"{prefix}.{name}:{type(module).__name__}"
        for root, prefix in ((policy, "policy"), (program, "program"))
        for name, module in root.named_modules()
        if isinstance(module, (MetaLoRAStack, MetaLoRAProjection))
    ]
    policy_trainable = [
        name for name, value in policy.named_parameters() if value.requires_grad
    ]
    program_trainable = [
        name for name, value in program.named_parameters() if value.requires_grad
    ]
    compiler_trainable = [
        name for name, value in compiler.named_parameters() if value.requires_grad
    ]
    modes = native_capture_modes(policy, owners)
    forbidden_free_parameters = [
        name
        for name in compiler_trainable
        if any(
            token in name
            for token in ("task_lookup", "video_lookup", "frame_lookup", "free_logits")
        )
    ]
    if (
        action_meta
        or policy_trainable
        or program_trainable
        or program.training
        or not compiler.training
        or not compiler_trainable
        or forbidden_free_parameters
        or set(modes) != {"identity_lora_base_layer"}
    ):
        raise ValueError("G3 pure-Native or shared-only trainable wall changed")
    return {
        "loader": "load_frozen_native_observer_and_frozen_g2_program",
        "action_meta_argument": None,
        "install_action_meta_lora": False,
        "action_meta_module_instances": action_meta,
        "action_meta_module_count": 0,
        "action_meta_parameter_count": 0,
        "source_policy_trainable_parameters": policy_trainable,
        "source_policy_trainable_parameter_count": 0,
        "natural_program_trainable_parameters": program_trainable,
        "natural_program_trainable_parameter_count": 0,
        "compiler_trainable_parameters": compiler_trainable,
        "compiler_trainable_parameter_count": sum(
            value.numel() for value in compiler.parameters() if value.requires_grad
        ),
        "native_capture_modes": list(modes),
        "task_video_frame_free_parameter_count": 0,
    }


def build_shared_compiler_run_contract(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    tasks: Sequence[Any],
    members: Sequence[Any],
    source: Mapping[str, Any],
    policy: torch.nn.Module,
    program: torch.nn.Module,
    compiler: torch.nn.Module,
    native_teacher_store: Any | None,
    owners: Sequence[Any],
    total_macros: int,
    checkpoint_macros: Sequence[int],
    optimizer_steps_per_macro: int,
    repo_root: Path,
) -> dict[str, Any]:
    repository = git_state(repo_root)
    effect_root = read_json(args.effect_bank_root)
    native_teacher = None
    if native_teacher_store is not None:
        teacher_root = read_json(native_teacher_store.root_manifest)
        if native_teacher_store.tensor_reads != 0:
            raise ValueError("G3 native teacher tensors were read before training")
        native_teacher = {
            "root_manifest": str(native_teacher_store.root_manifest),
            "root_manifest_bytes": native_teacher_store.root_manifest.stat().st_size,
            "schema_version": teacher_root.get("schema_version"),
            "full_fit_task_count": teacher_root.get("fit_authority_task_count"),
            "K1_covered_task_count": teacher_root.get("K1_covered_task_count"),
            "teacher_count": teacher_root.get("teacher_count"),
            "deployment_reads": False,
            "task_video_member_keys_are_training_only": True,
        }
    return {
        "schema_version": RUN_SCHEMA,
        "stage": "g3_shared_compiler",
        "mode": args.mode,
        "git": {
            "branch": repository["branch"],
            "commit": repository["commit"],
            "authority_commit": (
                repository["commit"]
                if args.mode == "formal"
                else repository["authority_commit"]
            ),
        },
        "config": {
            "path": str(args.config),
            "bytes": args.config.stat().st_size,
        },
        "source": dict(source),
        "tokenizer": {
            "path": str(args.tokenizer_path),
            "bytes": args.tokenizer_path.stat().st_size,
        },
        "data_root": str(args.data_root),
        "effect_bank": {
            "root_manifest": str(args.effect_bank_root),
            "root_manifest_bytes": args.effect_bank_root.stat().st_size,
            "schema_version": effect_root.get("schema_version"),
            "task_count": effect_root.get("task_count"),
            "member_count": effect_root.get("member_count"),
            "deployment_forward_reads_effects": False,
        },
        "native_teacher": native_teacher,
        "frozen_program": {
            "checkpoint": str(
                args.asset_root / config["authorities"]["g2_program_checkpoint"]
            ),
            **pure_shared_compiler_inventory(
                policy=policy,
                program=program,
                compiler=compiler,
                owners=owners,
            ),
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
                "members": [
                    {
                        "name": member.name,
                        "step": member.step,
                        "adapter": str(member.adapter),
                        "adapter_bytes": member.adapter_bytes,
                        "successes": member.successes,
                    }
                    for row in members
                    if row.task.authority_id == task.authority_id
                    for member in row.members
                ],
            }
            for task in tasks
        ],
        "model": dict(config["model"]),
        "data": dict(config["data"]),
        "functional_action_queries": {
            "fit_task_count": sum(
                task.role in {"meta_fit", "target_fit"} for task in tasks
            ),
            "held_task_count": 0,
            "video_and_action_episodes_disjoint": True,
            "queries_per_task_step": int(
                config["optimization"]["functional_query_count"]
            ),
            "action_chunk_size": int(config["data"]["action_chunk_size"]),
            "validation_or_test_action_reads": 0,
            "deployment_action_reads": 0,
        },
        "optimization": dict(config["optimization"]),
        "runtime": {
            "world_size": context.world_size,
            "topology": _topology(context),
            "total_macros": total_macros,
            "checkpoint_macros": list(checkpoint_macros),
            "optimizer_steps_per_macro": optimizer_steps_per_macro,
            "total_optimizer_steps": total_macros * optimizer_steps_per_macro,
            "global_tasks_per_optimizer_step": 2,
            "task_role_weighting": "one target-fit plus one meta-fit per step",
            "assignment": "role-balanced pair then cost-balanced ranks",
        },
        "gradient_wall": {
            "fit_roles": ["meta_fit", "target_fit"],
            "held_roles": ["meta_held", "target_held"],
            "held_task_gradient_count": 0,
            "source_policy_trainable_parameter_count": 0,
            "native_observer_trainable_parameter_count": 0,
            "natural_program_trainable_parameter_count": 0,
        },
    }


def publish_shared_compiler_run_contract(
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
                    raise ValueError("fresh G3 output directory is not empty")
                args.output_dir.mkdir(parents=True, exist_ok=True)
                write_json_atomic(path, contract)
            elif not path.is_file() or read_json(path) != contract:
                raise ValueError("exact-resume G3 contract changed")
            append_jsonl(
                args.output_dir / "invocations.jsonl",
                {
                    "mode": args.mode,
                    "resume": str(args.resume) if args.resume else None,
                    "stop_after_macro": args.stop_after_macro,
                },
            )
            result[0] = {"ok": True}
        except Exception as error:  # pragma: no cover - distributed propagation
            result[0] = {"ok": False, "error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(result, src=0)
    if not result[0]["ok"]:
        raise ValueError(result[0]["error"])
