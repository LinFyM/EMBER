#!/usr/bin/env python3
"""Seal compact CPU-derived dynamic labels for G2 Natural Program."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ember.ecp.natural_program_data import load_natural_program_tasks
from ember.ecp.natural_program_labels import seal_natural_program_labels


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO_ROOT
            / "configs/pi05_ecp_natural_program_g2_behavior_kernel_v3.json"
        ),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume-partial", action="store_true")
    parser.add_argument(
        "--authority-id",
        type=int,
        action="append",
        help="profile only: seal selected authorities without a complete manifest",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    authorities = config["authorities"]
    fold = config["fold"]
    tasks = load_natural_program_tasks(
        meta_protocol_path=REPO_ROOT / authorities["meta_protocol"],
        source_manifest_path=REPO_ROOT / authorities["source_manifest"],
        target_manifest_path=REPO_ROOT / authorities["target_manifest"],
        data_root=args.data_root,
        target_fit_ids=fold["target_fit_task_ids"],
        target_held_ids=fold["target_held_task_ids"],
        held_meta_fold=int(fold["meta_held_fold"]),
    )
    if args.authority_id:
        selected = set(args.authority_id)
        tasks = tuple(task for task in tasks if task.authority_id in selected)
        if len(tasks) != len(selected):
            raise ValueError("unknown Natural Program profile authority")
    manifest = seal_natural_program_labels(
        tasks=tasks,
        output_root=args.output_root,
        assets_root=args.assets_root,
        predicate_slots=int(config["model"]["predicate_slots"]),
        workers=args.workers,
        resume_partial=args.resume_partial,
    )
    print(json.dumps({
        "status": manifest["status"],
        "tasks": len(manifest["tasks"]),
        "output_root": str(args.output_root.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
