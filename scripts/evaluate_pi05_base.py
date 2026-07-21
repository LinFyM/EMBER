#!/usr/bin/env python3
"""CLI for the sealed generic pi0.5 LIBERO feasibility evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ember.pi05_assets import compute_train_only_stats, create_model_manifest
from ember.pi05_evaluation import aggregate_results, evaluate_test_task


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / "configs" / "libero_24_8_8_v1" / "protocol.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    stats = subparsers.add_parser("stats")
    stats.add_argument("--dataset-root", type=Path, required=True)
    stats.add_argument("--output", type=Path, required=True)
    stats.add_argument("--workers", type=int, default=16)
    model = subparsers.add_parser("model-manifest")
    model.add_argument("--model-path", type=Path, required=True)
    model.add_argument("--output", type=Path, required=True)
    task = subparsers.add_parser("task")
    task.add_argument("--normalization", type=Path, required=True)
    task.add_argument("--model-path", type=Path, required=True)
    task.add_argument("--model-manifest", type=Path, required=True)
    task.add_argument("--tokenizer-path", type=Path, required=True)
    task.add_argument("--tokenizer-manifest", type=Path, required=True)
    task.add_argument("--suite", required=True)
    task.add_argument("--task-id", type=int, required=True)
    task.add_argument("--output-dir", type=Path, required=True)
    task.add_argument("--episode-limit", type=int)
    task.add_argument("--env-count", type=int)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--input-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "stats":
        compute_train_only_stats(PROTOCOL, args.dataset_root, args.output, args.workers)
    elif args.command == "model-manifest":
        create_model_manifest(args.model_path, args.output)
    elif args.command == "task":
        evaluate_test_task(
            repo_root=REPO_ROOT,
            protocol_path=PROTOCOL,
            normalization_path=args.normalization,
            model_path=args.model_path,
            model_manifest_path=args.model_manifest,
            tokenizer_path=args.tokenizer_path,
            tokenizer_manifest_path=args.tokenizer_manifest,
            suite_name=args.suite,
            task_id=args.task_id,
            output_dir=args.output_dir,
            episode_limit=args.episode_limit,
            env_count=args.env_count,
        )
    else:
        aggregate_results(PROTOCOL, args.input_root, args.output)


if __name__ == "__main__":
    main()
