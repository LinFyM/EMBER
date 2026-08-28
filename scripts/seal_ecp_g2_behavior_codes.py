#!/usr/bin/env python3
"""Seal fit-only rank-16 policy-behavior coordinates for G2 alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import save_file

from ember.ecp.behavior.gate import (
    behavior_gram,
    fit_behavior_basis,
    load_behavior_panels,
)
from ember.ecp.behavior.codes import BEHAVIOR_CODE_SCHEMA
from ember.ecp.natural_program_data import load_natural_program_tasks
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic


REPO_ROOT = Path(__file__).resolve().parents[1]


def _tasks(config: dict, asset_root: Path, data_root: Path):
    fold = config["fold"]
    authority = config["authorities"]
    return load_natural_program_tasks(
        meta_protocol_path=asset_root / authority["meta_protocol"],
        source_manifest_path=asset_root / authority["source_manifest"],
        target_manifest_path=asset_root / authority["target_manifest"],
        data_root=data_root,
        target_fit_ids=fold["target_fit_task_ids"],
        target_held_ids=fold["target_held_task_ids"],
        held_meta_fold=int(fold["meta_held_fold"]),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--factor-roots", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("config", "asset_root", "data_root", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def _task_fold(tasks: Sequence[Any]) -> tuple[
    dict[int, str], tuple[int, ...], tuple[int, ...], tuple[int, ...], dict[str, int]
]:
    roles = {task.authority_id: task.role for task in tasks}
    all_tasks = tuple(sorted(roles))
    fit = tuple(
        task for task in all_tasks if roles[task] in {"meta_fit", "target_fit"}
    )
    held = tuple(task for task in all_tasks if task not in fit)
    if len(fit) != 75 or len(held) != 20:
        raise ValueError("behavior-code fold changed")
    role_counts = {
        role: sum(roles[task] == role for task in all_tasks)
        for role in ("meta_fit", "meta_held", "target_fit", "target_held")
    }
    return roles, all_tasks, fit, held, role_counts


def _load_consensus(
    roots: Sequence[Path],
    tasks: Sequence[int],
    selected_targets: Sequence[int],
    device: torch.device,
) -> dict[int, tuple[Any, ...]]:
    behavior = {}
    for task in tasks:
        _, _, behavior[task] = load_behavior_panels(
            roots, task, selected_targets, device
        )
    return behavior


def _fit_tensors(
    behavior: Mapping[int, tuple[Any, ...]],
    *,
    all_tasks: tuple[int, ...],
    fit: tuple[int, ...],
    held: tuple[int, ...],
    roles: Mapping[int, str],
    selected_targets: tuple[int, ...],
    dimension: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    all_index = {task: index for index, task in enumerate(all_tasks)}
    train_index = torch.tensor([all_index[task] for task in fit], device=device)
    train_weights = torch.tensor(
        [0.5 / sum(roles[value] == roles[task] for value in fit) for task in fit],
        device=device,
    )
    bases = tuple(
        fit_behavior_basis(
            behavior_gram(behavior, all_tasks, target),
            train_index,
            train_weights,
            dimension=dimension,
        )
        for target in range(len(selected_targets))
    )
    return {
        "task_ids": torch.tensor(all_tasks, dtype=torch.int64),
        "fit_task_ids": torch.tensor(fit, dtype=torch.int64),
        "held_task_ids": torch.tensor(held, dtype=torch.int64),
        "selected_targets": torch.tensor(selected_targets, dtype=torch.int64),
        "coordinates": torch.stack([basis.coordinates for basis in bases], dim=1)
        .cpu()
        .contiguous(),
        "mean": torch.stack([basis.mean for basis in bases]).cpu().contiguous(),
        "scale": torch.stack([basis.scale for basis in bases]).cpu().contiguous(),
        "eigenvectors": torch.stack([basis.eigenvectors for basis in bases])
        .cpu()
        .contiguous(),
        "eigenvalues": torch.stack([basis.eigenvalues for basis in bases])
        .cpu()
        .contiguous(),
        "norms": torch.stack([basis.norms for basis in bases]).cpu().contiguous(),
        "train_sqrt_weights": bases[0].train_sqrt_weights.cpu().contiguous(),
    }


def _manifest(
    *,
    tensor_path: Path,
    dimension: int,
    selected_targets: tuple[int, ...],
    all_tasks: tuple[int, ...],
    fit: tuple[int, ...],
    held: tuple[int, ...],
    role_counts: Mapping[str, int],
    roots: tuple[Path, ...],
    asset_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": BEHAVIOR_CODE_SCHEMA,
        "status": "complete",
        "git_commit": git_state(REPO_ROOT)["commit"],
        "tensor_file": tensor_path.name,
        "tensor_bytes": tensor_path.stat().st_size,
        "dimension": dimension,
        "selected_targets": list(selected_targets),
        "task_count": len(all_tasks),
        "fit_tasks": list(fit),
        "held_zero_gradient_tasks": list(held),
        "role_counts": dict(role_counts),
        "fit_role_weighting": "meta_fit_0.5_target_fit_0.5",
        "factor_roots": [str(root.relative_to(asset_root)) for root in roots],
        "behavior_target": "two_disjoint_256_row_cross_episode_flow_gradient_panels",
        "basis": "fit75_role_equal_consensus_normalized_kernel_pca",
        "held_used_for_training_or_checkpoint_selection": False,
        "validation_or_test_reads": 0,
        "action_meta_loaded": False,
        "source_policy_loaded": False,
    }


def main() -> None:
    args = _parse_args()
    if args.output_dir.exists():
        raise ValueError("behavior-code output already exists")
    config = read_json(args.config)
    cell = config["behavior_alignment"]
    selected_targets = tuple(map(int, cell["selected_targets"]))
    dimension = int(cell["dimension"])
    roots = tuple(Path(value).resolve() for value in args.factor_roots.split(":"))
    if len(roots) != 2 or any(not root.is_dir() for root in roots):
        raise ValueError("behavior factor roots changed")
    tasks = _tasks(config, args.asset_root, args.data_root)
    roles, all_tasks, fit, held, role_counts = _task_fold(tasks)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    behavior = _load_consensus(roots, all_tasks, selected_targets, device)
    tensors = _fit_tensors(
        behavior,
        all_tasks=all_tasks,
        fit=fit,
        held=held,
        roles=roles,
        selected_targets=selected_targets,
        dimension=dimension,
        device=device,
    )
    args.output_dir.mkdir(parents=True)
    tensor_path = args.output_dir / "behavior_codes.safetensors"
    save_file(tensors, str(tensor_path))
    manifest = _manifest(
        tensor_path=tensor_path,
        dimension=dimension,
        selected_targets=selected_targets,
        all_tasks=all_tasks,
        fit=fit,
        held=held,
        role_counts=role_counts,
        roots=roots,
        asset_root=args.asset_root,
    )
    write_json_atomic(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
