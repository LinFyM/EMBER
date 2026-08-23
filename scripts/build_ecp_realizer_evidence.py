#!/usr/bin/env python3
"""Capture or aggregate probe-preserving ECP realizer evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ember.ecp.realizer_capture import capture_effect_shard
from ember.ecp.realizer_code import fit_effect_code_authority
from ember.ecp.realizer_evidence import aggregate_effect_shards
from ember.ecp.two_sided_coordinate import build_centered_two_sided_coordinate
from ember.ecp.two_sided_oracle import materialize_centered_two_sided_oracle


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_fixed_effect_realizer.json",
    )
    capture.add_argument("--mode", choices=("profile", "formal"), required=True)
    capture.add_argument("--shard-index", type=int, default=0)
    capture.add_argument("--shard-count", type=int, default=1)
    capture.add_argument("--member-index", type=int)
    capture.add_argument("--source-run", type=Path, required=True)
    capture.add_argument("--checkpoint", type=Path, required=True)
    capture.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.add_argument("--device", default="cuda:0")
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--shard", type=Path, action="append", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    code = commands.add_parser("fit-code")
    code.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_fixed_effect_realizer.json",
    )
    code.add_argument("--particle-manifest", type=Path, required=True)
    code.add_argument("--fold", type=int, required=True)
    code.add_argument("--mode", choices=("profile", "formal"), required=True)
    code.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    code.add_argument("--output-dir", type=Path, required=True)
    code.add_argument("--device", default="cuda:0")
    two_sided = commands.add_parser("fit-two-sided")
    two_sided.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_centered_two_sided_coordinate.json",
    )
    two_sided.add_argument("--particle-manifest", type=Path, required=True)
    two_sided.add_argument("--mode", choices=("profile", "formal"), required=True)
    two_sided.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    two_sided.add_argument("--output-dir", type=Path, required=True)
    materialize = commands.add_parser("materialize-two-sided")
    materialize.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_centered_two_sided_coordinate.json",
    )
    materialize.add_argument("--coordinate-manifest", type=Path, required=True)
    materialize.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    materialize.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "capture":
        print(capture_effect_shard(args), flush=True)
    elif args.command == "aggregate":
        print(
            aggregate_effect_shards(shard_manifests=args.shard, output=args.output),
            flush=True,
        )
    elif args.command == "fit-code":
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        print(
            fit_effect_code_authority(
                config_path=args.config,
                particle_manifest=args.particle_manifest,
                output_dir=args.output_dir,
                fold=args.fold,
                asset_root=args.asset_root.resolve(),
                device=device,
                formal=args.mode == "formal",
            ),
            flush=True,
        )
    elif args.command == "fit-two-sided":
        print(build_centered_two_sided_coordinate(args), flush=True)
    else:
        print(materialize_centered_two_sided_oracle(args), flush=True)


if __name__ == "__main__":
    main()
