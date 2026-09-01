"""Canonical joint Program--primal functional training entrypoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch.distributed as dist

from ember.ecp.checkpoint import save_ecp_checkpoint
from ember.ecp.joint_program_primal.runtime import (
    REPO_ROOT,
    JointProgramPrimalRuntime,
    joint_run_schema,
    joint_stage,
    prepare_joint_program_primal_runtime,
)
from ember.ecp.joint_program_primal.train_step import (
    run_joint_program_primal_optimizer_step,
)
from ember.pi05_source_checkpoint import write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import initialize_distributed


def train(args: argparse.Namespace) -> None:
    if args.phase != "joint":
        raise ValueError("J3 joint trainer received the positive-control phase")
    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    runtime: JointProgramPrimalRuntime | None = None
    try:
        runtime = prepare_joint_program_primal_runtime(args, context)
        started = time.monotonic()
        while runtime.optimizer_steps < runtime.stop_after_step:
            row = run_joint_program_primal_optimizer_step(runtime)
            row["elapsed_seconds"] = time.monotonic() - started
            if context.is_main:
                append_jsonl(args.output_dir / "metrics.jsonl", row)
                runtime.metrics_rows += 1
                if runtime.optimizer_steps % args.log_every == 0:
                    primary_metric = (
                        "mean_acquisition_loss"
                        if "mean_acquisition_loss" in row
                        else "mean_functional_loss"
                    )
                    console = {
                        name: row[name]
                        for name in (
                            "optimizer_step",
                            "effective_optimizer_step",
                            "global_step_seconds",
                            primary_metric,
                            "gradient_norm_before_clip",
                            "gradient_probe_norms",
                            "next_lr",
                            "task_group",
                            "role_counts",
                            "rank_assignments",
                            "native_teacher_tensor_reads",
                            "elapsed_seconds",
                        )
                    }
                    for name in (
                        "counterfactual_arm",
                        "counterfactual_view_index",
                        "mean_counterfactual_normalized_gap",
                        "mean_counterfactual_hinge_loss",
                        "active_counterfactual_fraction",
                    ):
                        if name in row:
                            console[name] = row[name]
                    console["rank_performance"] = [
                        {
                            name: value[name]
                            for name in (
                                "rank",
                                "tasks",
                                "seconds",
                                "max_cuda_allocated_bytes",
                                "max_cuda_reserved_bytes",
                            )
                        }
                        for value in row["rank_performance"]
                    ]
                    print(json.dumps(console, sort_keys=True), flush=True)
            if runtime.optimizer_steps in runtime.checkpoint_steps:
                save_ecp_checkpoint(
                    output_dir=args.output_dir,
                    macro=runtime.optimizer_steps,
                    stage=joint_stage(runtime.config),
                    context=context,
                    model=runtime.writer_state,
                    optimizer=runtime.optimizer,
                    scheduler=runtime.scheduler,
                    run_contract_schema=joint_run_schema(runtime.config),
                    metrics_rows=runtime.metrics_rows,
                )
        if context.is_main:
            completion = {
                "stage": joint_stage(runtime.config),
                "completed_optimizer_steps": runtime.optimizer_steps,
                "completed_effective_steps": max(0, runtime.optimizer_steps - 10),
            }
            write_json_atomic(args.output_dir / "segment_completion.json", completion)
            if runtime.optimizer_steps == max(runtime.checkpoint_steps):
                write_json_atomic(args.output_dir / "completion.json", completion)
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_joint_program_primal_j3_v1.json",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_shared_compiler_g3_v5.json",
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument(
        "--phase", choices=("positive-control", "joint"), required=True
    )
    parser.add_argument("--task", type=int)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition-cache-root", type=Path, required=True)
    parser.add_argument("--program-bank-condition-cache-root", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-checkpoint", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
        "base_config",
        "asset_root",
        "source_run",
        "checkpoint",
        "tokenizer_path",
        "data_root",
        "output_dir",
        "condition_cache_root",
        "program_bank_condition_cache_root",
        "resume",
        "evaluate_checkpoint",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.log_every <= 0:
        raise ValueError("J3 log interval must be positive")
    if (args.phase == "positive-control") != (args.task is not None):
        raise ValueError("J3 task is required only for the positive control")
    return args
