#!/usr/bin/env python3
"""Prepare the fixed ECP Stage 1A rollout and static-adapter authorities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from ember.expert_manifold.projection import (
    ECP_STAGE1_STATIC_LORA_MANIFEST_SCHEMA,
    ECP_STAGE1_STATIC_LORA_PURPOSES,
)
from ember.pi05_eval.occupancy_selection import (
    ECP_STAGE1_HELD_KEYS,
    ECP_STAGE1_OCCUPANCY_SELECTION_SCHEMA,
    ECP_STAGE1_PROFILE_KEYS,
)
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic


REPO_ROOT = Path(__file__).resolve().parents[1]
INITIAL_IDS = (0, 7, 14, 21, 28, 35, 42, 49)
RECOVERY_IDS = (1, 26, 2, 27)


def _repository() -> dict[str, Any]:
    state = git_state(REPO_ROOT)
    if not git_state_is_clean_pushed_or_frozen_authority(state):
        raise ValueError(
            "formal ECP Stage 1A preparation requires clean pushed authority"
        )
    return state


def _row(suite: str, task_id: int, global_id: int, init_id: int, category: str):
    return {
        "suite": suite,
        "task_id": task_id,
        "global_task_id": global_id,
        "init_state_id": init_id,
        "category": category,
    }


def _panel_keys(panel: str) -> tuple[tuple[str, int, int], ...]:
    return (
        ECP_STAGE1_PROFILE_KEYS
        if panel.startswith("profile_")
        else ECP_STAGE1_HELD_KEYS
    )


def _base_panel(panel: str) -> str:
    return panel.removeprefix("profile_")


def _successful_rows(
    results_path: Path, keys: tuple[tuple[str, int, int], ...]
) -> list[dict[str, Any]]:
    results = read_json(results_path.resolve())
    rows = tuple(dict(row) for row in results.get("rows", ()))
    selected = []
    for suite, task_id, global_id in keys:
        successes = sorted(
            (
                row
                for row in rows
                if str(row.get("suite")) == suite
                and int(row.get("task_id", -1)) == task_id
                and bool(row.get("success"))
            ),
            key=lambda row: int(row["init_state_id"]),
        )
        if not successes:
            raise ValueError(
                f"independent ECP particle has no success: {suite}/{task_id}"
            )
        selected.append(
            _row(
                suite,
                task_id,
                global_id,
                int(successes[0]["init_state_id"]),
                "successful",
            )
        )
    return selected


def write_selection(args: argparse.Namespace) -> None:
    keys = _panel_keys(args.panel)
    panel = _base_panel(args.panel)
    if panel == "source_support":
        rows = [
            _row(suite, task_id, global_id, init_id, category)
            for suite, task_id, global_id in keys
            for category, values in (
                ("initial", INITIAL_IDS),
                ("recovery_candidate", RECOVERY_IDS),
            )
            for init_id in values
        ]
    elif panel == "candidate_policy":
        rows = [
            _row(suite, task_id, global_id, 0, "candidate")
            for suite, task_id, global_id in keys
        ]
    else:
        if args.results is None:
            raise ValueError("independent-particle selection requires fixed50 results")
        rows = _successful_rows(args.results, keys)
    write_json_atomic(
        args.output.resolve(),
        {
            "schema_version": ECP_STAGE1_OCCUPANCY_SELECTION_SCHEMA,
            "status": "preregistered_or_fixed_after_member_qualification",
            "panel": args.panel,
            "repository": _repository(),
            "selection_rule": (
                "lowest fixed init ID strict success per held task"
                if panel == "independent_particle"
                else "fixed preregistered state IDs"
            ),
            "rows": rows,
        },
    )


def _file(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size}


def _adapter_for(root: Path, global_id: int, purpose: str) -> Path:
    root = root.resolve()
    if purpose == "stage1a_independent_particle_step2000":
        candidates = list(
            root.glob(
                f"worker_*/task_*_global_{global_id:02d}/checkpoints/"
                "step_00002000/adapter.safetensors"
            )
        )
    else:
        candidates = sorted(
            root.rglob(f"task_*_global_{global_id:02d}.safetensors")
        )
    if len(candidates) != 1:
        raise ValueError(
            f"ECP static adapter is not unique for global task {global_id}"
        )
    return candidates[0].resolve()


def write_projection(args: argparse.Namespace) -> None:
    base_path = args.base_projection_manifest.resolve()
    base = read_json(base_path)
    base_rows = {int(row["global_task_id"]): dict(row) for row in base.get("tasks", ())}
    tasks = []
    keys = (
        ECP_STAGE1_PROFILE_KEYS
        if args.task_panel == "profile_fit"
        else ECP_STAGE1_HELD_KEYS
    )
    for suite, task_id, global_id in keys:
        source = base_rows.get(global_id)
        adapter = _adapter_for(args.adapter_root, global_id, args.purpose)
        if source is None:
            raise ValueError("ECP base projection is missing a held task")
        tasks.append(
            {
                "suite": suite,
                "task_id": task_id,
                "ordinal": int(source["ordinal"]),
                "global_task_id": global_id,
                "expert_checkpoint": source["expert_checkpoint"],
                "projected_adapter": str(adapter),
                "projected_adapter_bytes": adapter.stat().st_size,
            }
        )
    write_json_atomic(
        args.output.resolve(),
        {
            "schema_version": ECP_STAGE1_STATIC_LORA_MANIFEST_SCHEMA,
            "projection_kind": "ecp_stage1_privileged_static_lora",
            "purpose": args.purpose,
            "task_panel": args.task_panel,
            "repository": {
                "commit": _repository()["commit"],
                "dirty_paths": [],
            },
            "base_projection_manifest": _file(base_path),
            "optimization": {
                "held_shared_gradient_steps": 0,
                "single_complete_lora": True,
                "final_lora_averaging": False,
                "rank": 16,
                "second_adapter_deployed": False,
                "parameterization": "one complete rank16 static LoRA",
            },
            "information_wall": {
                "role": "development_train_leave_task_out_oracle_only",
                "deployment_carrier": False,
                "validation_action_or_reward_reads": 0,
                "test_action_or_reward_reads": 0,
                "second_adapter_deployed": False,
            },
            "tasks": tasks,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    selection = commands.add_parser("selection")
    selection.add_argument(
        "--panel",
        choices=(
            "source_support",
            "candidate_policy",
            "independent_particle",
            "profile_source_support",
            "profile_candidate_policy",
            "profile_independent_particle",
        ),
        required=True,
    )
    selection.add_argument("--results", type=Path)
    selection.add_argument("--output", type=Path, required=True)
    projection = commands.add_parser("projection")
    projection.add_argument(
        "--purpose", choices=sorted(ECP_STAGE1_STATIC_LORA_PURPOSES), required=True
    )
    projection.add_argument(
        "--task-panel", choices=("held5", "profile_fit"), default="held5"
    )
    projection.add_argument("--base-projection-manifest", type=Path, required=True)
    projection.add_argument("--adapter-root", type=Path, required=True)
    projection.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "selection":
        write_selection(args)
    else:
        write_projection(args)


if __name__ == "__main__":
    main()
