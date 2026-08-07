#!/usr/bin/env python3
"""Extract or seal action-hidden frozen video features for Expert-Manifold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ember.expert_manifold.contract import REPO_ROOT
from ember.expert_manifold.feature_cache import (
    run_feature_worker,
    seal_feature_cache,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_video_expert_manifold_v1.json",
    )
    worker.add_argument("--mode", choices=("profile", "formal"), required=True)
    worker.add_argument("--source-run", type=Path, required=True)
    worker.add_argument("--checkpoint", type=Path, required=True)
    worker.add_argument("--tokenizer-path", type=Path, required=True)
    worker.add_argument("--data-root", type=Path, required=True)
    worker.add_argument("--output-dir", type=Path, required=True)
    worker.add_argument("--task-indices", required=True)
    seal = commands.add_parser("seal")
    seal.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_video_expert_manifold_v1.json",
    )
    seal.add_argument("--cache-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "worker":
        result = run_feature_worker(
            config_path=args.config,
            mode=args.mode,
            source_run=args.source_run,
            checkpoint=args.checkpoint,
            tokenizer_path=args.tokenizer_path,
            data_root=args.data_root,
            output_dir=args.output_dir.resolve(),
            task_indices=args.task_indices,
        )
        summary = {"event": "worker_complete", "tasks": result["task_count"]}
    else:
        result = seal_feature_cache(args.config, args.cache_root)
        summary = {"event": "cache_sealed", "tasks": result["task_count"]}
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
