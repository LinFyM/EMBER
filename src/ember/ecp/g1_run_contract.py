"""Retained task-run contract for the formal G1 capacity oracle."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from typing import Any, Mapping

import torch

from ember.ecp.g1_assets import G1_CONFIG_SCHEMA, G1RankAssets, G1TaskAssets
from ember.ecp.g1_video import G1VideoRuntime
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl


def build_run_contract(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    task: G1TaskAssets,
    ranks: G1RankAssets,
    video: G1VideoRuntime,
    pure_native: Mapping[str, Any],
    initialization: Mapping[str, Any],
    sensitivity_raw: torch.Tensor,
    sensitivity_weights: torch.Tensor,
    repo_root: Any,
    schema: str,
) -> dict[str, Any]:
    return {
        "schema_version": schema,
        "mode": args.mode,
        "repository": git_state(repo_root),
        "host": socket.gethostname(),
        "device": str(args.torch_device),
        "runtime": {
            "world_size": 1,
            "torch_device": str(args.torch_device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": torch.cuda.get_device_name(args.torch_device),
        },
        "config": str(args.config),
        "config_schema": G1_CONFIG_SCHEMA,
        "authorities": dict(config["authorities"]),
        "task": {
            "ordinal": task.ordinal,
            "global_task_id": task.global_task_id,
            "suite": task.suite,
            "task_id": task.task_id,
            "language": task.language,
            "video_path": str(task.video_authority.path),
            "video_bytes": task.video_authority.expected_bytes,
        },
        "video": {
            "teacher_demo_index": video.teacher_demo_index,
            "raw_frame_count": video.raw_frame_count,
            "sampled_frame_indices": list(video.sampled_frame_indices),
            "sampled_frame_count": video.readout.frame_count,
            "K": 1,
            "cross_video_weight": "identity_k1",
        },
        "video_contract": dict(config["video"]),
        "functional_query": dict(config["functional_query"]),
        "native_factor": {
            "input_candidates": ["video", "frame", "probe", "horizon"],
            "output_candidates": [
                "video",
                "frame",
                "probe",
                "horizon",
                "abs_adj_init_goal_type",
            ],
            "positive_negative_softmax": True,
            "residual_rank": 4,
            "carrier_rank": 12,
            "output_rank": 16,
            "fit_experts_for_s_ref": ranks.fit_expert_count,
            "s_ref": ranks.s_ref.detach().cpu().tolist(),
            **dict(config["native_factor"]),
        },
        "native_factor_initialization": dict(initialization),
        "pure_native_stage0": dict(pure_native),
        "policy_sensitivity": {
            "calibration": "carrier directional functional derivative along each successful rank4 member",
            "raw": sensitivity_raw.detach().cpu().tolist(),
            "family_balanced_weights": sensitivity_weights.detach().cpu().tolist(),
        },
        "optimization": dict(config["optimization"]),
        "information_wall": dict(config["information_wall"]),
        "content_hash_policy": "disabled_by_owner",
    }


def publish_run_contract(
    *, args: argparse.Namespace, contract: Mapping[str, Any]
) -> None:
    path = args.output_dir / "run_contract.json"
    if args.resume is None:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("fresh G1 task output is not empty")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, dict(contract))
    elif not path.is_file() or read_json(path) != contract:
        raise ValueError("G1 exact-resume run contract changed")
    append_jsonl(
        args.output_dir / "invocations.jsonl",
        {
            "argv": sys.argv,
            "host": socket.gethostname(),
            "resume": str(args.resume) if args.resume else None,
            "stop_after_step": args.stop_after_step,
            "started_unix": time.time(),
        },
    )
