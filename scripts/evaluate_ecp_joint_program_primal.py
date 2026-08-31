#!/usr/bin/env python3
"""Evaluate or aggregate a J3 or routing-control checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ember.ecp.joint_program_primal.evaluation import evaluate_worker
from ember.ecp.joint_program_primal.evaluation_gate import aggregate_evaluation
from ember.ecp.joint_program_primal.routing_control import ROUTING_CONTROL_SCHEMA
from ember.ecp.joint_program_primal.routing_control import (
    PROGRAM_BANK_INTERACTION_SCHEMA,
)
from ember.ecp.joint_program_primal.routing_control_evaluation import (
    ROUTING_GATE_SCHEMA,
    aggregate_routing_evaluation,
    evaluate_routing_worker,
)
from ember.pi05_source_checkpoint import read_json


REPO_ROOT = Path(__file__).resolve().parents[1]


def _worker_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("worker")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_joint_program_primal_j3_v1.json",
    )
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=REPO_ROOT
        / "configs/pi05_ecp_joint_program_primal_j3_gate_v1.json",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_shared_compiler_g3_v5.json",
    )
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--compiler-run", type=Path, required=True)
    parser.add_argument(
        "--compiler-checkpoints", type=Path, nargs=2, required=True
    )
    parser.add_argument("--condition-cache-root", type=Path, required=True)
    parser.add_argument("--program-bank-condition-cache-root", type=Path)
    parser.add_argument("--endpoint-cache-root", type=Path, required=True)
    parser.add_argument("--output-dirs", type=Path, nargs=2, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)


def _aggregate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("aggregate")
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--compiler-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--previous-report", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _worker_parser(subparsers)
    _aggregate_parser(subparsers)
    return parser


def _resolve(args: argparse.Namespace) -> argparse.Namespace:
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
        elif isinstance(value, list) and value and all(
            isinstance(item, Path) for item in value
        ):
            setattr(args, name, [item.resolve() for item in value])
    return args


def main() -> int:
    args = _resolve(build_parser().parse_args())
    if args.command == "worker":
        schema = read_json(args.config).get("schema_version")
        if schema == PROGRAM_BANK_INTERACTION_SCHEMA:
            raise ValueError(
                "retired pointwise Program-bank interaction config is not executable"
            )
        elif schema == ROUTING_CONTROL_SCHEMA:
            evaluate_routing_worker(args)
        else:
            evaluate_worker(args)
        return 0
    gate_schema = read_json(args.gate_config).get("schema_version")
    if gate_schema == "ember_ecp_program_bank_candidate_interaction_gate_v4":
        raise ValueError(
            "retired pointwise Program-bank interaction Gate is not executable"
        )
    if gate_schema == ROUTING_GATE_SCHEMA:
        aggregate = aggregate_routing_evaluation
    else:
        aggregate = aggregate_evaluation
    report = aggregate(
        output_dir=args.output_dir,
        gate_config=args.gate_config,
        compiler_run=args.compiler_run,
        worker_count=args.worker_count,
        previous_report=args.previous_report,
    )
    print(
        json.dumps(
            {
                "checkpoint": report["checkpoint"]["optimizer_step"],
                "primary_pass": report["primary_pass"],
                "gate_pass": report["gate_pass"],
                "checks": report["checks"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
