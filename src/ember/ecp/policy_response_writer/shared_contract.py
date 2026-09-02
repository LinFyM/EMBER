"""Formal authority records for shared Policy-Response Writer training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ember.ecp.policy_response_writer.training import REPO_ROOT, PolicyResponseRuntime
from ember.ecp.shared_compiler_assets import authority_path
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import barrier, read_json, write_json_atomic


def build_shared_run_contract(
    runtime: PolicyResponseRuntime,
    *,
    schema: str,
    stage: str,
    stop: int,
    parameters: Sequence[torch.nn.Parameter],
    topology: Sequence[Mapping[str, Any]],
    task_owners: Sequence[Sequence[int]],
    video_splits: Mapping[int, tuple[tuple[int, int], int]],
) -> dict[str, Any]:
    base_path = (
        runtime.args.asset_root
        / str(runtime.config["authorities"]["base_g3_config"])
    ).resolve()
    panel_path = (
        runtime.args.asset_root
        / str(runtime.config["authorities"]["functional_panel_config"])
    ).resolve()
    split = {
        str(task): {"fit": list(fit), "held": held}
        for task, (fit, held) in sorted(video_splits.items())
    }
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
        "functional_panel_config": {
            "path": str(panel_path),
            "bytes": panel_path.stat().st_size,
        },
        "source_checkpoint": str(
            authority_path(
                runtime.base, "source_checkpoint", asset_root=runtime.args.asset_root
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
                runtime.base, "stable_carrier", asset_root=runtime.args.asset_root
            )
        ),
        "data_root": str(runtime.args.data_root),
        "representation": runtime.args.representation,
        "initialization_request": runtime.args.initialization,
        "initialization": runtime.initialization,
        "stop_step": stop,
        "model": dict(runtime.config["model"]),
        "optimization": dict(runtime.config["optimization"]["shared"]),
        "loss_normalization": {
            "functional": "per_task_frozen_panel_a_carrier_rms",
            "process": "per_task_initial_two_fit_prefix_mean",
            "preservation": "same_as_functional",
            "resume_authority": "normalizers.json",
        },
        "task_split": dict(runtime.config["task_split"]),
        "video_splits": split,
        "information_wall": dict(runtime.config["information_wall"]),
        "task_ownership": [list(map(int, row)) for row in task_owners],
        "inventory": {
            "trainable_parameter_count": sum(value.numel() for value in parameters),
            "trainable_parameter_names": [
                name
                for name, value in runtime.writer.named_parameters()
                if value.requires_grad
            ],
            "task_local_parameter_count": 0,
            "source_policy_trainable_parameter_count": 0,
            "native_observer_trainable_parameter_count": 0,
            "action_meta_installed": False,
            "single_complete_rank16": True,
        },
        "world_topology": list(topology),
    }


def seal_or_validate_shared_run_contract(
    runtime: PolicyResponseRuntime, contract: Mapping[str, Any]
) -> None:
    output = runtime.args.output_dir
    if runtime.args.resume is None:
        if runtime.context.is_main:
            if output.exists() and any(output.iterdir()):
                raise ValueError("fresh shared Writer output root is not empty")
            output.mkdir(parents=True, exist_ok=True)
            if runtime.args.mode == "formal":
                write_json_atomic(output / "run_contract.json", contract)
        barrier(runtime.context)
        return
    if runtime.args.mode != "formal":
        raise ValueError("shared Writer resume is formal-only")
    if runtime.args.resume.parent.parent.resolve() != output:
        raise ValueError("shared Writer resume escaped its output root")
    if runtime.context.is_main:
        existing = read_json(output / "run_contract.json")
        current_git = contract.get("git", {})
        original_git = existing.get("git", {})
        if current_git.get("commit") != original_git.get("commit"):
            raise ValueError("shared Writer code commit changed on resume")
        normalized = {**contract, "git": original_git}
        if existing != normalized:
            raise ValueError("shared Writer run contract changed on resume")
    barrier(runtime.context)


def reference_result_path(runtime: PolicyResponseRuntime, task: int) -> Path:
    return (
        runtime.args.asset_root
        / str(runtime.config["authorities"]["task_local_reference_root"])
        / f"task_{task:03d}"
        / "result.json"
    ).resolve()
