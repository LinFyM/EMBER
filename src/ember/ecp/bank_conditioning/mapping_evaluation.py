"""Held-video and held-task qualification for G3 mapping checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ember.ecp.bank_conditioning.mapping_eval_runtime import (
    REPO_ROOT,
    evaluate_mapping_worker,
)
from ember.ecp.bank_conditioning.mapping_gate import aggregate_mapping_evaluation

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("worker", "aggregate"), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_shared_compiler_g3_v5.json",
    )
    parser.add_argument(
        "--program-causality-contract",
        type=Path,
        default=(
            REPO_ROOT
            / "configs/pi05_ecp_shared_compiler_g3_f3_program_causality_v1.json"
        ),
    )
    parser.add_argument("--phase", choices=("f3",), required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--training-run", type=Path)
    parser.add_argument("--compiler-checkpoint", type=Path)
    parser.add_argument("--previous-report", type=Path)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
        "program_causality_contract",
        "asset_root",
        "output_dir",
        "source_run",
        "source_checkpoint",
        "tokenizer_path",
        "data_root",
        "training_run",
        "compiler_checkpoint",
        "previous_report",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if not 1 <= args.worker_count <= 6:
        raise ValueError("mapping evaluation worker count must be in [1, 6]")
    worker_required = (
        "worker_index",
        "source_run",
        "source_checkpoint",
        "tokenizer_path",
        "data_root",
        "training_run",
        "compiler_checkpoint",
    )
    if args.mode == "worker" and any(getattr(args, name) is None for name in worker_required):
        raise ValueError("mapping evaluation worker arguments are incomplete")
    if args.mode == "worker" and not 0 <= args.worker_index < args.worker_count:
        raise ValueError("mapping evaluation worker index changed")
    return args


def main() -> None:
    args = finalize_args(build_parser().parse_args())
    if args.mode == "worker":
        evaluate_mapping_worker(args)
    else:
        report = aggregate_mapping_evaluation(args)
        print(json.dumps(report["gate"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
