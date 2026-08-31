#!/usr/bin/env python3
"""Run or aggregate the EBSRI S0/S1 task-local qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ember.ecp.joint_program_primal.bank_set_tasklocal import run
from ember.ecp.joint_program_primal.bank_set_tasklocal_evaluation import (
    aggregate_tasklocal,
)
from ember.ecp.joint_program_primal.runtime import REPO_ROOT


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run", help="run one task on one GPU")
    execute.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_event_bank_set_s0_free_summary_v1.json",
    )
    execute.add_argument(
        "--base-config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_shared_compiler_g3_v5.json",
    )
    execute.add_argument("--mode", choices=("profile", "formal"), required=True)
    execute.add_argument("--task", type=int, choices=(1, 93), required=True)
    execute.add_argument("--asset-root", type=Path, required=True)
    execute.add_argument("--source-run", type=Path, required=True)
    execute.add_argument("--checkpoint", type=Path, required=True)
    execute.add_argument("--tokenizer-path", type=Path, required=True)
    execute.add_argument("--data-root", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, required=True)
    execute.add_argument("--condition-cache-root", type=Path, required=True)
    execute.add_argument("--program-bank-condition-cache-root", type=Path, required=True)
    execute.add_argument("--resume", type=Path)
    execute.add_argument("--stop-after-step", type=int)
    execute.add_argument("--log-every", type=int, default=10)
    aggregate = commands.add_parser("aggregate", help="seal the two formal tasks")
    aggregate.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_event_bank_set_s0_free_summary_v1.json",
    )
    aggregate.add_argument("--task-output-dirs", type=Path, nargs=2, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    return parser


def _arguments() -> argparse.Namespace:
    args = _parser().parse_args()
    for name in vars(args):
        value = getattr(args, name)
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
        elif name == "task_output_dirs":
            setattr(args, name, tuple(path.resolve() for path in value))
    if args.command == "run" and args.log_every <= 0:
        raise ValueError("bank-set log interval must be positive")
    if args.command == "run":
        args.phase = "joint"
    return args


if __name__ == "__main__":
    arguments = _arguments()
    if arguments.command == "run":
        run(arguments)
    else:
        report = aggregate_tasklocal(
            config_path=arguments.config,
            task_output_dirs=arguments.task_output_dirs,
            output_dir=arguments.output_dir,
        )
        print(json.dumps({"gate": report["gate"], "tasks": report["tasks"]}, sort_keys=True))
