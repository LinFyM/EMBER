#!/usr/bin/env python3
"""Measure policy-effective geometry across a complete train24 expert bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file

from ember.expert_manifold.contract import (
    REPO_ROOT,
    authority_path,
    load_expert_manifold_config,
)
from ember.expert_manifold.evaluation import inspect_task_expert_bank
from ember.pi05_eval_contract import (
    inspect_source_checkpoint,
    load_evaluation_authorities,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.adapter_analysis_metrics import (
    adapter_geometry,
    effective_inner,
    effective_update_variance,
    effective_variance,
)


ANALYSIS_SCHEMA = "ember_pi05_task_expert_bank_analysis_v1"


def _steps(value: str) -> tuple[int, ...]:
    try:
        steps = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("steps must be comma-separated integers") from error
    if not steps or tuple(sorted(set(steps))) != steps or steps[0] <= 0:
        raise argparse.ArgumentTypeError("steps must be positive, unique, and sorted")
    return steps


def _pairs(state: Mapping[str, torch.Tensor]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    suffixes = {
        ".lora_A.default.weight": "a",
        ".lora_B.default.weight": "b",
    }
    for name in state:
        for suffix, factor in suffixes.items():
            if name.endswith(suffix):
                result.setdefault(name.removesuffix(suffix), {})[factor] = name
                break
    if not result or any(set(value) != {"a", "b"} for value in result.values()):
        raise ValueError("task-expert LoRA state is not fully paired")
    return result


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _pairwise_effective(
    pairs: Mapping[str, Mapping[str, str]],
    states: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    size = len(states)
    gram = np.empty((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(left, size):
            gram[left, right] = gram[right, left] = effective_inner(
                pairs, states[left], states[right]
            )
    diagonal = np.maximum(np.diag(gram), 0.0)
    cosine = gram / np.sqrt(
        np.maximum(diagonal[:, None], 1e-30)
        * np.maximum(diagonal[None, :], 1e-30)
    )
    mask = ~np.eye(size, dtype=bool)
    off = cosine[mask]
    return {
        "mean_offdiagonal_cosine": float(off.mean()),
        "median_offdiagonal_cosine": float(np.median(off)),
        "mean_absolute_offdiagonal_cosine": float(np.abs(off).mean()),
        "negative_pair_fraction": float((off < 0).mean()),
        "minimum_pair_cosine": float(off.min()),
        "maximum_pair_cosine": float(off.max()),
        "effective_variance": effective_variance(pairs, states),
    }


def _step_analysis(
    adapter: Mapping[str, Any],
    *,
    rank: int,
) -> tuple[dict[str, Any], tuple[dict[str, torch.Tensor], ...]]:
    states = tuple(
        load_file(str(Path(row["checkpoint"]) / "adapter.safetensors"), device="cpu")
        for row in adapter["tasks"]
    )
    pairs = _pairs(states[0])
    if any(_pairs(state) != pairs for state in states[1:]):
        raise ValueError("task-expert states changed LoRA topology")
    writer = SimpleNamespace(PUBLIC_LORA_RANK=rank)
    geometry = [adapter_geometry(writer, pairs, state, 1.0) for state in states]
    coordinates = [
        row["rank_coordinate_geometry_gauge_dependent"] for row in geometry
    ]
    return (
        {
            "step": int(adapter["step"]),
            "task_count": len(states),
            "effective_lora_norm": _quantiles(
                [float(row["effective_lora_norm_unscaled"]) for row in geometry]
            ),
            "stable_rank_mean_across_targets": _quantiles(
                [float(row["stable_rank_mean"]) for row in geometry]
            ),
            "top_singular_energy_mean_across_targets": _quantiles(
                [float(row["top_singular_energy_mean"]) for row in geometry]
            ),
            "effective_target_count": _quantiles(
                [
                    float(row["target_energy_profile"]["effective_targets"])
                    for row in geometry
                ]
            ),
            "top4_target_energy_fraction": _quantiles(
                [
                    float(row["target_energy_profile"]["top4_targets_energy_fraction"])
                    for row in geometry
                ]
            ),
            "public_a_rms": _quantiles(
                [float(row["public_a_rms"]) for row in geometry]
            ),
            "public_b_rms": _quantiles(
                [float(row["public_b_rms"]) for row in geometry]
            ),
            "active_rank_coordinates": _quantiles(
                [float(row["all"]["active_coordinates"]) for row in coordinates]
            ),
            "top4_rank_coordinate_energy_fraction": _quantiles(
                [
                    float(row["all"]["top4_coordinate_energy_fraction"])
                    for row in coordinates
                ]
            ),
            "mean_absolute_rank_component_cosine": _quantiles(
                [
                    float(row["all"]["mean_absolute_offdiagonal_component_cosine"])
                    for row in coordinates
                ]
            ),
            "mean_absolute_b_column_cosine": {
                kind: _quantiles(
                    [
                        float(row[kind]["mean_absolute_b_column_cosine"])
                        for row in coordinates
                    ]
                )
                for kind in ("q", "v", "action")
            },
            "q_energy_fraction": _quantiles(
                [float(row["by_kind"]["q"]["energy_fraction"]) for row in geometry]
            ),
            "v_energy_fraction": _quantiles(
                [float(row["by_kind"]["v"]["energy_fraction"]) for row in geometry]
            ),
            "action_energy_fraction": _quantiles(
                [float(row["by_kind"]["action"]["energy_fraction"]) for row in geometry]
            ),
            "pairwise_task_geometry": _pairwise_effective(pairs, states),
            "per_task": [
                {
                    "suite": task["suite"],
                    "task_id": int(task["task_id"]),
                    "ordinal": int(task["ordinal"]),
                    "global_task_id": int(task["global_task_id"]),
                    "effective_lora_norm": float(row["effective_lora_norm_unscaled"]),
                    "stable_rank_mean": float(row["stable_rank_mean"]),
                    "top_singular_energy_mean": float(row["top_singular_energy_mean"]),
                    "effective_targets": float(
                        row["target_energy_profile"]["effective_targets"]
                    ),
                    "top4_target_energy_fraction": float(
                        row["target_energy_profile"]["top4_targets_energy_fraction"]
                    ),
                }
                for task, row in zip(adapter["tasks"], geometry, strict=True)
            ],
        },
        states,
    )


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = load_expert_manifold_config(config_path)
    formal_steps = set(int(value) for value in config["task_experts"]["formal_run"]["checkpoint_steps"])
    if any(step not in formal_steps for step in args.steps):
        raise ValueError("analysis steps are outside the formal task-expert schedule")
    authorities = load_evaluation_authorities(
        authority_path(config, "evaluation_config"), REPO_ROOT
    )
    source = inspect_source_checkpoint(
        authorities,
        args.source_run.resolve(),
        args.checkpoint.resolve(),
        evaluation_mode="formal",
    )
    manifest = read_json(authority_path(config, "target_data_manifest"))
    train_rows = tuple(
        sorted(
            (row for row in manifest["tasks"] if row["split_role"] == "train"),
            key=lambda row: int(row["global_task_id"]),
        )
    )
    task_keys = tuple((str(row["suite"]), int(row["task_id"])) for row in train_rows)
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    by_step = []
    states_by_step = []
    adapters = []
    for step in args.steps:
        adapter = inspect_task_expert_bank(
            config_path=config_path,
            bank_root=args.bank_root.resolve(),
            step=step,
            source=source,
            task_keys=task_keys,
            evaluation_role="development_train",
            require_formal=True,
        )
        summary, states = _step_analysis(adapter, rank=lora.rank)
        adapters.append(adapter)
        by_step.append(summary)
        states_by_step.append(states)
    pairs = _pairs(states_by_step[0][0])
    transitions = []
    for left, right, previous, current in zip(
        args.steps[:-1], args.steps[1:], states_by_step[:-1], states_by_step[1:], strict=True
    ):
        transitions.append(
            {
                "from_step": left,
                "to_step": right,
                "effective_update_variance": effective_update_variance(
                    pairs, previous, current
                ),
            }
        )
    result = {
        "schema_version": ANALYSIS_SCHEMA,
        "method": "independent_task_local_rank16_policy_experts",
        "config": {"path": str(config_path), "schema": config["schema_version"]},
        "bank_root": str(args.bank_root.resolve()),
        "source": adapters[0]["source"],
        "training_commit": adapters[0]["training_commit"],
        "steps": list(args.steps),
        "lora": adapters[0]["lora_contract"],
        "by_step": by_step,
        "transitions": transitions,
        "information_wall": adapters[0]["information_wall"],
        "content_hash_policy": "disabled_by_owner",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_video_expert_manifold_v1.json",
    )
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--steps", type=_steps, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = analyze(args)
    print(
        json.dumps(
            {
                "event": "complete",
                "steps": result["steps"],
                "tasks": result["by_step"][-1]["task_count"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
