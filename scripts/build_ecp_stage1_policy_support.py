#!/usr/bin/env python3

import argparse
from pathlib import Path

from ember.ecp.stage1_support_building import (
    assemble_support_bank,
    build_support_shard,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "assemble"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-frames-per-call", type=int)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
        "asset_root",
        "source_run",
        "checkpoint",
        "data_root",
        "output_dir",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid policy-support shard")
    required = ("asset_root", "source_run", "checkpoint", "data_root")
    if args.mode == "build" and any(
        getattr(args, name) is None for name in required
    ):
        raise ValueError("policy-support build requires source and asset authorities")
    return args


if __name__ == "__main__":
    arguments = finalize_args(build_parser().parse_args())
    if arguments.mode == "build":
        build_support_shard(arguments)
    else:
        assemble_support_bank(arguments)
