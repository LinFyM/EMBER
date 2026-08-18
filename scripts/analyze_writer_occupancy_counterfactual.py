#!/usr/bin/env python3
"""Re-query macro25/50 on the fixed union of their rollout occupancies."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file

from ember.lora import copy_task_lora_state_, validate_lora_state
from ember.pi05_eval.paired_metrics import episode_key, index_rows
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import load_policy
from ember.reward.rollout import policy_flow_noise_cpu
from ember.writer.as_config import authority_path, load_writer_config
from ember.writer.functional import prepare_frozen_writer_policy


SHARD_SCHEMA = "ember_writer_occupancy_counterfactual_shard_v1"
ANALYSIS_SCHEMA = "ember_writer_occupancy_counterfactual_analysis_v1"


def _entry_name(suite: str, task_id: int, init_state_id: int) -> str:
    return f"{suite}_task_{task_id:02d}_state_{init_state_id:03d}"


def _load_panel(
    root: Path,
) -> tuple[dict[str, Any], dict[tuple[str, int, int], dict[str, Any]]]:
    result = read_json(root / "results.json")
    contract = read_json(root / "run_contract.json")
    if (
        result.get("mode") != "formal"
        or result.get("role") != "validation"
        or result.get("adapter", {}).get("video_condition") != "correct"
        or contract.get("diagnostic_occupancy_capture") is None
    ):
        raise ValueError(
            "occupancy counterfactual requires one diagnostic correct-video panel"
        )
    return result, index_rows(result["rows"])


def _load_trajectory(row: Mapping[str, Any]) -> dict[str, Any]:
    record = row.get("occupancy_trajectory", {})
    path = Path(str(record.get("path", "")))
    if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError("occupancy trajectory sidecar changed")
    trajectory = torch.load(path, map_location="cpu", weights_only=False)
    if (
        trajectory.get("schema_version") != "ember_writer_occupancy_trajectory_v1"
        or len(trajectory.get("observations", ()))
        != len(trajectory.get("action_chunks", ()))
        or len(trajectory.get("observations", ()))
        != len(trajectory.get("policy_noise_seeds", ()))
    ):
        raise ValueError("occupancy trajectory layout changed")
    return trajectory


def _query(
    policy: torch.nn.Module,
    observations: Sequence[Mapping[str, torch.Tensor]],
    seeds: Sequence[int],
    *,
    device: torch.device,
    microbatch_size: int,
) -> tuple[torch.Tensor, ...]:
    outputs = []
    for start in range(0, len(observations), microbatch_size):
        rows = observations[start : start + microbatch_size]
        batch = {
            name: torch.cat([row[name] for row in rows]).to(device, non_blocking=True)
            for name in sorted(rows[0])
        }
        noise = torch.cat(
            [
                policy_flow_noise_cpu(
                    seed=int(seed),
                    chunk_size=int(policy.config.chunk_size),
                    max_action_dim=int(policy.config.max_action_dim),
                )
                for seed in seeds[start : start + microbatch_size]
            ]
        ).to(device, non_blocking=True)
        with torch.inference_mode():
            actions = policy.predict_action_chunk(batch, noise=noise, num_steps=10)
        outputs.extend(
            value.detach().to(device="cpu").contiguous() for value in actions.split(1)
        )
    return tuple(outputs)


def _rms(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).square().mean().sqrt())


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _episode_analysis(
    *,
    selection: Mapping[str, Any],
    row25: Mapping[str, Any],
    row50: Mapping[str, Any],
    run25: Path,
    run50: Path,
    policy: torch.nn.Module,
    lora: Any,
    device: torch.device,
    microbatch_size: int,
) -> dict[str, Any]:
    suite = str(selection["suite"])
    task_id = int(selection["task_id"])
    init_state_id = int(selection["init_state_id"])
    trajectory25 = _load_trajectory(row25)
    trajectory50 = _load_trajectory(row50)
    if bool(trajectory25["success"]) != bool(selection["macro25_success"]) or bool(
        trajectory50["success"]
    ) != bool(selection["macro50_success"]):
        raise ValueError("captured occupancy outcome differs from its selection panel")
    observations25 = tuple(trajectory25["observations"])
    observations50 = tuple(trajectory50["observations"])
    seeds25 = tuple(int(value) for value in trajectory25["policy_noise_seeds"])
    seeds50 = tuple(int(value) for value in trajectory50["policy_noise_seeds"])
    observations = (*observations25, *observations50)
    seeds = (*seeds25, *seeds50)
    name = _entry_name(suite, task_id, init_state_id)
    adapters = {
        "macro25": load_file(
            str(run25 / "writer_lora_cache" / "entries" / name / "lora.safetensors"),
            device="cpu",
        ),
        "macro50": load_file(
            str(run50 / "writer_lora_cache" / "entries" / name / "lora.safetensors"),
            device="cpu",
        ),
    }
    actions = {}
    for arm, state in adapters.items():
        validate_lora_state(state, lora)
        copy_task_lora_state_(policy, state, lora)
        policy.reset()
        actions[arm] = _query(
            policy,
            observations,
            seeds,
            device=device,
            microbatch_size=microbatch_size,
        )
    count25 = len(observations25)
    disagreement25 = [
        _rms(left, right)
        for left, right in zip(
            actions["macro25"][:count25],
            actions["macro50"][:count25],
            strict=True,
        )
    ]
    disagreement50 = [
        _rms(left, right)
        for left, right in zip(
            actions["macro25"][count25:],
            actions["macro50"][count25:],
            strict=True,
        )
    ]
    stored25 = tuple(trajectory25["action_chunks"])
    stored50 = tuple(trajectory50["action_chunks"])
    requery25 = [
        _rms(left, right)
        for left, right in zip(stored25, actions["macro25"][:count25], strict=True)
    ]
    requery50 = [
        _rms(left, right)
        for left, right in zip(stored50, actions["macro50"][count25:], strict=True)
    ]

    def halves(values: Sequence[float]) -> tuple[float, float]:
        split = max(1, len(values) // 2)
        return _mean(values[:split]), _mean(values[split:] or values[-1:])

    early25, late25 = halves(disagreement25)
    early50, late50 = halves(disagreement50)
    return {
        "suite": suite,
        "task_id": task_id,
        "init_state_id": init_state_id,
        "category": selection["category"],
        "macro25_success": bool(trajectory25["success"]),
        "macro50_success": bool(trajectory50["success"]),
        "macro25_steps": int(trajectory25["steps"]),
        "macro50_steps": int(trajectory50["steps"]),
        "macro25_occupancy_replans": count25,
        "macro50_occupancy_replans": len(observations50),
        "initial_state_action_rms": 0.5 * (disagreement25[0] + disagreement50[0]),
        "macro25_occupancy_action_rms_mean": _mean(disagreement25),
        "macro50_occupancy_action_rms_mean": _mean(disagreement50),
        "macro25_occupancy_action_rms_early_half": early25,
        "macro25_occupancy_action_rms_late_half": late25,
        "macro50_occupancy_action_rms_early_half": early50,
        "macro50_occupancy_action_rms_late_half": late50,
        "macro25_stored_requery_rms_mean": _mean(requery25),
        "macro50_stored_requery_rms_mean": _mean(requery50),
    }


def run_shard(args: argparse.Namespace) -> None:
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 0 or not 0 <= rank < world_size or args.microbatch_size != 8:
        raise ValueError("occupancy counterfactual runtime changed")
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    selection = read_json(args.selection)
    panel25, rows25 = _load_panel(args.macro25_run)
    panel50, rows50 = _load_panel(args.macro50_run)
    selected = tuple(selection["rows"])
    if selection.get("schema_version") != "ember_writer_occupancy_selection_v1":
        raise ValueError("occupancy selection schema changed")
    config = load_writer_config(args.writer_config)
    source_config = read_json(authority_path(config, "source_base_config"))
    policy = load_policy(args.source_checkpoint / "policy", source_config, device)
    from ember.pi05_lora import load_pi05_lora_contract

    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    prepare_frozen_writer_policy(policy, lora)
    policy.eval()
    rows = []
    for ordinal, record in enumerate(selected):
        if ordinal % world_size != rank:
            continue
        key = (
            str(record["suite"]),
            int(record["task_id"]),
            int(record["init_state_id"]),
        )
        rows.append(
            _episode_analysis(
                selection=record,
                row25=rows25[key],
                row50=rows50[key],
                run25=args.macro25_run,
                run50=args.macro50_run,
                policy=policy,
                lora=lora,
                device=device,
                microbatch_size=args.microbatch_size,
            )
        )
        print(json.dumps({"rank": rank, "complete": len(rows), "key": key}), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        args.output_dir / f"rank_{rank:02d}.json",
        {
            "schema_version": SHARD_SCHEMA,
            "rank": rank,
            "world_size": world_size,
            "selection": str(args.selection),
            "macro25_run": str(args.macro25_run),
            "macro50_run": str(args.macro50_run),
            "panel_arms": [panel25["arm"], panel50["arm"]],
            "rows": rows,
        },
    )


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def aggregate(args: argparse.Namespace) -> None:
    rows = []
    for rank in range(args.shard_count):
        shard = read_json(args.output_dir / f"rank_{rank:02d}.json")
        if (
            shard.get("schema_version") != SHARD_SCHEMA
            or int(shard["world_size"]) != args.shard_count
        ):
            raise ValueError("occupancy counterfactual shard changed")
        rows.extend(shard["rows"])
    selection = read_json(args.selection)
    expected = {
        (str(row["suite"]), int(row["task_id"]), int(row["init_state_id"]))
        for row in selection["rows"]
    }
    observed = {
        (str(row["suite"]), int(row["task_id"]), int(row["init_state_id"]))
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError("occupancy counterfactual lost selected rows")
    metrics = tuple(key for key in rows[0] if key.endswith("_rms") or "_rms_" in key)
    by_category = {}
    for category in ("lost", "gained", "retained"):
        selected_rows = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "rows": len(selected_rows),
            "metrics": {
                key: _quantiles([float(row[key]) for row in selected_rows])
                for key in metrics
            },
        }
    panel25, _ = _load_panel(args.macro25_run)
    checkpoint = Path(panel25["adapter"]["writer_asset"]["checkpoint"])
    metric_rows = [
        json.loads(line)
        for line in (checkpoint.parent.parent / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    loss_by_macro = {
        int(row["macro"]): float(row["global_mean_functional_loss"])
        for row in metric_rows
        if int(row["macro"]) in {25, 50}
    }
    if set(loss_by_macro) != {25, 50}:
        raise ValueError("occupancy analysis lost macro25/50 offline loss")
    result = {
        "schema_version": ANALYSIS_SCHEMA,
        "selection": str(args.selection),
        "macro25_run": str(args.macro25_run),
        "macro50_run": str(args.macro50_run),
        "rows": sorted(
            rows, key=lambda row: (row["suite"], row["task_id"], row["init_state_id"])
        ),
        "by_category": by_category,
        "offline_b20_functional_loss": {
            "macro25": loss_by_macro[25],
            "macro50": loss_by_macro[50],
            "macro50_minus_macro25": loss_by_macro[50] - loss_by_macro[25],
        },
        "reference_boundary": {
            "validation_task_experts_available": 0,
            "teacher_actions_read": 0,
            "analysis": "matched macro25/macro50 action disagreement on fixed S25 union S50",
            "causal_limit": "successful checkpoint behavior is a directional reference, not an expert-action oracle",
        },
        "training_gradient_use": False,
    }
    write_json_atomic(args.output_dir / "occupancy_analysis.json", result)
    print(json.dumps({"event": "complete", "rows": len(rows)}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--macro25-run", type=Path, required=True)
    parser.add_argument("--macro50-run", type=Path, required=True)
    parser.add_argument("--writer-config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--shard-count", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in (
        "selection",
        "macro25_run",
        "macro50_run",
        "writer_config",
        "source_checkpoint",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.aggregate_only:
        if args.shard_count is None:
            raise ValueError("occupancy aggregation requires its shard count")
        aggregate(args)
    else:
        if args.shard_count is not None:
            raise ValueError("shard count is aggregation-only")
        run_shard(args)


if __name__ == "__main__":
    main()
