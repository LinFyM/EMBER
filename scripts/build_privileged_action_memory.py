#!/usr/bin/env python3
"""Bounded cache entrypoint for the registered train-only action memory diagnostic."""
from pathlib import Path
import argparse
from ember.pi05_eval.action_memory_cache import ROOT, build_source, prepare_geometry, seal_source

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("phase", choices=("geometry", "source", "seal"))
parser.add_argument("--root", type=Path, default=ROOT)
parser.add_argument("--asset-root", type=Path, required=True)
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--shards", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()
if args.root.resolve() != ROOT:
    raise ValueError("Unregistered memory root")
if args.phase == "geometry":
    prepare_geometry(args.asset_root, args.root)
elif args.phase == "source":
    build_source(args.root, args.shard, args.shards, args.batch_size, args.profile)
else:
    seal_source(args.root)
