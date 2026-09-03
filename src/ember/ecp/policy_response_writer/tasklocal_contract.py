"""Formal authority and result records for task-local Writer qualification."""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from ember.ecp.policy_response_writer.capture import FrozenPolicyResponseVideo
from ember.ecp.policy_response_writer.training import REPO_ROOT, PolicyResponseRuntime
from ember.ecp.shared_compiler_assets import authority_path
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic


def _resolved_functional_panel_authority(
    runtime: PolicyResponseRuntime, task: int
) -> dict[str, Any]:
    panel = runtime.panels[task]
    if panel.task_id != task:
        raise ValueError("task-local functional panel task changed")
    path = panel.path.resolve()
    return {
        "kind": "resolved_task_record",
        "task": task,
        "path": str(path),
        "bytes": path.stat().st_size,
    }


def build_tasklocal_run_contract(
    runtime: PolicyResponseRuntime,
    *,
    schema: str,
    stage: str,
    task: int,
    fit_demos: tuple[int, int],
    held_demo: int,
    reference_path: Path,
    parameters: tuple[torch.nn.Parameter, ...],
    stop: int,
) -> dict[str, Any]:
    base_path = (
        runtime.args.asset_root
        / str(runtime.config["authorities"]["base_g3_config"])
    ).resolve()
    return {
        "schema_version": schema,
        "stage": stage,
        "mode": runtime.args.mode,
        "git": git_state(REPO_ROOT),
        "config": {
            "path": str(runtime.args.config),
            "bytes": runtime.args.config.stat().st_size,
        },
        "base_config": {"path": str(base_path), "bytes": base_path.stat().st_size},
        "functional_panel_authority": _resolved_functional_panel_authority(
            runtime, task
        ),
        "source_checkpoint": str(
            authority_path(
                runtime.base,
                "source_checkpoint",
                asset_root=runtime.args.asset_root,
            )
        ),
        "native_observer_checkpoint": str(
            authority_path(
                runtime.base,
                "native_observer_checkpoint",
                asset_root=runtime.args.asset_root,
            )
        ),
        "stable_carrier": str(
            authority_path(
                runtime.base,
                "stable_carrier",
                asset_root=runtime.args.asset_root,
            )
        ),
        "data_root": str(runtime.args.data_root),
        "task": task,
        "fit_videos": list(fit_demos),
        "held_video": held_demo,
        "task_local_reference": str(reference_path),
        "representation": runtime.args.representation,
        "stop_step": stop,
        "model": dict(runtime.config["model"]),
        "optimization": dict(runtime.config["optimization"]["task_local"]),
        "information_wall": dict(runtime.config["information_wall"]),
        "initialization": runtime.initialization,
        "inventory": {
            "trainable_parameter_count": sum(value.numel() for value in parameters),
            "trainable_parameter_names": [
                name
                for name, value in runtime.writer.named_parameters()
                if value.requires_grad
            ],
            "process_trainable_parameter_count": 0,
            "source_policy_trainable_parameter_count": 0,
            "action_meta_installed": False,
            "frozen_process_cache": True,
        },
        "world_topology": [
            {
                "rank": runtime.context.rank,
                "local_rank": runtime.context.local_rank,
                "device": str(runtime.context.device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cuda_device_name": torch.cuda.get_device_name(
                    runtime.context.device
                ),
                "hostname": socket.gethostname(),
                "numa_node": runtime.context.numa_node,
                "cpu_affinity": list(runtime.context.cpu_affinity or ()),
            }
        ],
    }


def seal_or_validate_tasklocal_run_contract(
    runtime: PolicyResponseRuntime, contract: Mapping[str, Any]
) -> None:
    output = runtime.args.output_dir
    if runtime.args.resume is None:
        if output.exists() and any(output.iterdir()):
            raise ValueError("fresh task-local Composer output root is not empty")
        output.mkdir(parents=True, exist_ok=True)
        if runtime.args.mode == "formal":
            write_json_atomic(output / "run_contract.json", contract)
        return
    if runtime.args.mode != "formal":
        raise ValueError("task-local Composer resume is formal-only")
    if runtime.args.resume.parent.parent.resolve() != output:
        raise ValueError("task-local Composer resume escaped its output root")
    existing = read_json(output / "run_contract.json")
    current_git = contract.get("git", {})
    original_git = existing.get("git", {})
    if current_git.get("commit") != original_git.get("commit"):
        raise ValueError("task-local Composer code commit changed on resume")
    # origin/main may advance while this frozen commit keeps running.  The
    # formal launcher already proves the current checkout is clean, detached,
    # and still contained by the authority; preserve the launch-time Git
    # snapshot when comparing every other exact-resume field.
    normalized = {**contract, "git": original_git}
    if existing != normalized:
        raise ValueError("task-local Composer run contract changed on resume")


def build_tasklocal_result(
    runtime: PolicyResponseRuntime,
    *,
    schema: str,
    task: int,
    fit_demos: tuple[int, int],
    held_demo: int,
    stop: int,
    start_step: int,
    total: int,
    training: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    evaluations: Mapping[str, Any],
    capture_records: list[dict[str, Any]],
    capture_seconds: float,
    cache: Mapping[int, FrozenPolicyResponseVideo],
    reference_path: Path,
    parameters: tuple[torch.nn.Parameter, ...],
    evaluation_seconds: float,
    started: float,
) -> dict[str, Any]:
    return {
        "schema_version": schema,
        "status": "complete",
        "phase": "task-local",
        "mode": runtime.args.mode,
        "representation": runtime.args.representation,
        "task": task,
        "fit_videos": list(fit_demos),
        "held_video": held_demo,
        "optimizer_steps": stop,
        "resume_start_step": start_step,
        "configured_total_steps": total,
        "curve": training["curve"],
        "evaluation": evaluation,
        "evaluation_by_checkpoint": evaluations,
        "capture": capture_records,
        "capture_seconds": capture_seconds,
        "frozen_evidence_cache": "ephemeral_cpu_evidence_plus_gpu_process",
        "frozen_evidence_tensor_bytes": sum(
            value.tensor_bytes for value in cache.values()
        ),
        "task_local_reference": str(reference_path),
        "initialization": runtime.initialization,
        "trainable_parameter_count": sum(value.numel() for value in parameters),
        "task_query_parameter_count": runtime.writer.composer.task_query.numel(),
        "process_trainable_parameter_count": 0,
        "source_policy_trainable_parameter_count": 0,
        "action_meta_installed": False,
        "wrong_video_backward_calls": 0,
        "held_video_backward_calls": 0,
        "panel_b_backward_calls": 0,
        "single_complete_rank16": True,
        "checkpoints": [
            str(path)
            for path in sorted(
                (runtime.args.output_dir / "checkpoints").glob("macro_*")
            )
        ],
        "metrics_rows": training["metrics_rows"],
        "train_seconds": training["train_seconds"],
        "evaluation_seconds": evaluation_seconds,
        "total_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": training["max_cuda_allocated_bytes"],
        "max_cuda_reserved_bytes": training["max_cuda_reserved_bytes"],
    }
