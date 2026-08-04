"""Task-relative on-policy Flow-Credit Writer training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch.distributed as dist

from ember.pi05_source_setup import initialize_distributed
from ember.rl_writer.contract import REPO_ROOT
from ember.rl_writer.loop import run_cycles
from ember.rl_writer.progress_diagnostic import run_progress_diagnostic
from ember.rl_writer.runtime import RLWriterRuntime, build_runtime


def train(args: argparse.Namespace) -> None:
    context = initialize_distributed(
        require_numa=args.mode == "formal",
        defer_process_group=True,
    )
    runtime: RLWriterRuntime | None = None
    try:
        runtime = build_runtime(args, context)
        if context.is_main:
            print(
                json.dumps(
                    {
                        "event": "start",
                        "contract_sha256": runtime.contract_sha256,
                        "next_cycle": runtime.next_cycle,
                        "stop_after_cycle": args.stop_after_cycle,
                        "trainable": runtime.contract["trainable"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.mode == "diagnostic":
            run_progress_diagnostic(runtime)
        else:
            run_cycles(runtime)
    finally:
        if runtime is not None:
            runtime.env_pool.close()
            runtime.video_store.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_rl_writer_development_v1.json",
    )
    parser.add_argument(
        "--mode", choices=("diagnostic", "profile", "formal"), required=True
    )
    parser.add_argument("--stage", choices=("development",), required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--coldstart-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--total-cycles", type=int)
    parser.add_argument("--stop-after-cycle", type=int)
    parser.add_argument("--checkpoint-cycles", type=str)
    parser.add_argument("--learning-epochs", type=int)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
        "source_run",
        "checkpoint",
        "tokenizer_path",
        "data_root",
        "coldstart_checkpoint",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.resume is not None:
        args.resume = args.resume.resolve()
    return args


def main() -> None:
    train(finalize_args(build_parser().parse_args()))
