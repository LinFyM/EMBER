#!/usr/bin/env python3
"""Prepare, run, or aggregate the EBSRI S2 shared LOTO evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ember.ecp.joint_program_primal.bank_set_shared_aggregate import (
    aggregate_shared_evaluation,
)
from ember.ecp.joint_program_primal.bank_set_shared_evaluation import (
    evaluate_worker,
    prepare_job_queue,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/pi05_ecp_event_bank_set_s2_shared_loto_v1.json"
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs/pi05_ecp_shared_compiler_g3_v5.json"


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--asset-root", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="seal the dynamic job queue")
    _common(prepare)
    prepare.add_argument("--compiler-run", type=Path, required=True)
    prepare.add_argument("--compiler-checkpoints", type=Path, nargs=2, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--worker-count", type=int, required=True)

    worker = commands.add_parser("worker", help="run one persistent GPU worker")
    _common(worker)
    worker.add_argument("--mode", choices=("profile", "formal"), required=True)
    worker.add_argument("--source-run", type=Path, required=True)
    worker.add_argument("--checkpoint", type=Path, required=True)
    worker.add_argument("--tokenizer-path", type=Path, required=True)
    worker.add_argument("--data-root", type=Path, required=True)
    worker.add_argument("--compiler-run", type=Path, required=True)
    worker.add_argument("--condition-cache-root", type=Path, required=True)
    worker.add_argument("--program-bank-condition-cache-root", type=Path, required=True)
    worker.add_argument("--s1-gate-aggregate", type=Path, required=True)
    worker.add_argument("--output-dir", type=Path, required=True)
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--worker-count", type=int, required=True)

    aggregate = commands.add_parser("aggregate", help="seal the adjacent Gate")
    aggregate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    return parser


def _arguments() -> argparse.Namespace:
    args = _parser().parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
        elif isinstance(value, list) and all(isinstance(row, Path) for row in value):
            setattr(args, name, [row.resolve() for row in value])
    if args.command == "worker":
        args.phase = "shared_loto"
        args.task = None
        args.resume = None
        # Runtime construction reuses the training cursor validator even though
        # workers never step an optimizer.  Select an allowed cursor for each
        # mode; the checkpoint loader immediately replaces it with 70 or 110.
        args.stop_after_step = 1 if args.mode == "profile" else 70
        args.log_every = 1
        args.skip_routing_initialization = True
    return args


def main() -> int:
    args = _arguments()
    if args.command == "prepare":
        value = prepare_job_queue(
            config_path=args.config,
            base_config_path=args.base_config,
            asset_root=args.asset_root,
            compiler_run=args.compiler_run,
            checkpoint_paths=args.compiler_checkpoints,
            output_dir=args.output_dir,
            worker_count=args.worker_count,
        )
        summary = {"job_count": len(value["jobs"]), "worker_count": value["worker_count"]}
    elif args.command == "worker":
        summary = evaluate_worker(args)
    else:
        value = aggregate_shared_evaluation(
            output_dir=args.output_dir, config_path=args.config
        )
        summary = {
            "primary_pass": value["primary_pass"],
            "adjacent_checkpoint_pass": value["adjacent_checkpoint"]["pass"],
            "gate_pass": value["gate_pass"],
        }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
