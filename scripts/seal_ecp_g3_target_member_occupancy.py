#!/usr/bin/env python3
"""Seal one successful target-fit initial state for each G3 expert member."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from ember.ecp.shared_compiler_assets import (
    authority_path,
    load_shared_compiler_config,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic


SCHEMA = "ember_ecp_g3_verified_member_occupancy_selection_v1"


def _selection(
    *,
    results_path: Path,
    fit_ids: set[int],
    step: int,
) -> dict[str, Any]:
    results = read_json(results_path)
    selected: dict[int, Mapping[str, Any]] = {}
    for row in sorted(
        results.get("rows", ()), key=lambda value: int(value["init_state_id"])
    ):
        expert = row.get("task_expert", {})
        global_id = int(expert.get("global_task_id", -1))
        if (
            global_id in fit_ids
            and int(expert.get("step", -1)) == step
            and row.get("success") is True
            and global_id not in selected
        ):
            selected[global_id] = row
    expected = 19 if step == 1000 else 18
    if len(selected) != expected:
        raise ValueError(f"G3 step{step} successful fit-member panel changed")
    rows = []
    for global_id, row in sorted(selected.items()):
        expert = row["task_expert"]
        rows.append(
            {
                "suite": str(row["suite"]),
                "task_id": int(row["task_id"]),
                "global_task_id": global_id,
                "language": str(row["language"]),
                "init_state_id": int(row["init_state_id"]),
                "success": True,
                "member_step": step,
                "member_checkpoint": str(expert["checkpoint"]),
                "selection": "lowest_verified_successful_init_state",
            }
        )
    return {
        "schema_version": SCHEMA,
        "status": "sealed_g3_verified_member_occupancy",
        "member_step": step,
        "source_results": {
            "path": str(results_path.resolve()),
            "bytes": results_path.stat().st_size,
        },
        "rows": rows,
        "information_wall": {
            "role": "target_fit_training_only",
            "validation_or_test_rows": 0,
            "task_identity_role": "sampler_and_member_ownership_only",
            "deployment_use": False,
        },
        "claim_boundary": (
            "Each row is an already paired closed-loop success for this exact "
            "task expert and only seeds a verified training-state trajectory; "
            "it is not a deployment input or a task dictionary."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pi05_ecp_shared_compiler_g3_v1.json"),
    )
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_shared_compiler_config(args.config)
    fit_ids = set(map(int, config["fold"]["target_fit_task_ids"]))
    if len(fit_ids) != 19:
        raise ValueError("G3 target-fit fold changed")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for step, authority in (
        (1000, "target_step1000_results"),
        (2000, "target_step2000_results"),
    ):
        payload = _selection(
            results_path=authority_path(
                config, authority, asset_root=args.asset_root
            ),
            fit_ids=fit_ids,
            step=step,
        )
        write_json_atomic(
            args.output_dir / f"step_{step:08d}.json", payload
        )


if __name__ == "__main__":
    main()
