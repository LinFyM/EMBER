#!/usr/bin/env python3
"""Build source response shards and assemble the MDCO Stage 1 authority."""

from __future__ import annotations

import argparse
from pathlib import Path

from ember.ecp.stage1_evidence_building import (
    assemble_mapping_diverse_authority,
    build_source_response_shard,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    shard = commands.add_parser("source-shard")
    shard.add_argument("--source-manifest", type=Path, required=True)
    shard.add_argument("--capture-run", type=Path, required=True)
    shard.add_argument("--source-checkpoint", type=Path, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--shard-count", type=int, required=True)
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--device", default="cuda:0")
    shard.add_argument("--microbatch-size", type=int, default=8)

    assemble = commands.add_parser("assemble")
    assemble.add_argument("--asset-root", type=Path, required=True)
    assemble.add_argument("--source-manifest", type=Path, required=True)
    assemble.add_argument("--meta-protocol", type=Path, required=True)
    assemble.add_argument("--capture-run", type=Path, required=True)
    assemble.add_argument("--source-shard", type=Path, action="append", required=True)
    assemble.add_argument("--target-manifest", type=Path, required=True)
    assemble.add_argument("--target-selection", type=Path, required=True)
    assemble.add_argument("--target-analysis", type=Path, required=True)
    assemble.add_argument("--output-dir", type=Path, required=True)
    return parser


def _finalize(args: argparse.Namespace) -> argparse.Namespace:
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
        elif isinstance(value, list) and value and isinstance(value[0], Path):
            setattr(args, name, [path.resolve() for path in value])
    if args.command == "source-shard" and (
        args.shard_count <= 0
        or not 0 <= args.shard_index < args.shard_count
        or args.microbatch_size <= 0
    ):
        raise ValueError("invalid MDCO source response shard")
    return args


if __name__ == "__main__":
    arguments = _finalize(_parser().parse_args())
    if arguments.command == "source-shard":
        build_source_response_shard(arguments)
    else:
        assemble_mapping_diverse_authority(arguments)
