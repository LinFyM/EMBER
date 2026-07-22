"""Eight-rank zero-action-warmup PI05 Reward-Trained Writer training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch.distributed as dist

from ember.pi05_source_setup import initialize_distributed
from ember.rl_writer.contract import REPO_ROOT
from ember.rl_writer.loop import run_updates
from ember.rl_writer.runtime import RLWriterRuntime, build_runtime


def train(args: argparse.Namespace) -> None:
    context = initialize_distributed(require_numa=args.mode == "formal")
    runtime: RLWriterRuntime | None = None
    try:
        runtime = build_runtime(args, context)
        if context.is_main:
            print(
                json.dumps(
                    {
                        "event": "start",
                        "contract_sha256": runtime.contract_sha256,
                        "branch": args.branch,
                        "next_update": runtime.next_update,
                        "stop_after_update": args.stop_after_update,
                        "trainable": runtime.contract["trainable"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        run_updates(runtime)
    finally:
        if runtime is not None:
            runtime.env_pool.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_rl_writer_development_v1.json",
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--stage", choices=("development", "final"), required=True)
    parser.add_argument(
        "--branch",
        choices=("micro_as_warmup", "zero_as_warmup"),
        required=True,
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--total-updates", type=int)
    parser.add_argument("--stop-after-update", type=int)
    parser.add_argument("--checkpoint-updates", type=str)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
        "source_run",
        "checkpoint",
        "tokenizer_path",
        "feature_cache",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.resume is not None:
        args.resume = args.resume.resolve()
    return args


def main() -> None:
    train(finalize_args(build_parser().parse_args()))
