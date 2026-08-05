"""Pure metrics and aggregation for the Program-Credit mechanism audit."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from ember.rl_writer.program_credit import program_cotangent, program_direction
from ember.writer.adapter_analysis_metrics import (
    distribution,
    effective_inner,
)


CHECKPOINT_LABELS = ("as125", "cycle1")
PROGRAM_SKETCH_WIDTH = 256


def tensor_variance(values: Sequence[torch.Tensor]) -> dict[str, float]:
    flat = torch.stack([value.double().reshape(-1) for value in values])
    gram = flat @ flat.T
    sample = float(gram.diag().mean())
    mean = float(gram.mean())
    return {
        "sample_energy": sample,
        "mean_energy": mean,
        "centered_variance_over_sample_energy": max(sample - mean, 0.0)
        / max(sample, 1e-30),
        "mean_energy_fraction": mean / max(sample, 1e-30),
    }


def _difference_summary_from_gram(gram: torch.Tensor) -> dict[str, float]:
    diagonal = gram.diag().clamp_min(0)
    sample = float(diagonal.mean())
    mean = float(gram.mean())
    normalized = gram / torch.sqrt(
        diagonal[:, None].clamp_min(1e-30) * diagonal[None].clamp_min(1e-30)
    )
    mask = ~torch.eye(gram.shape[0], dtype=torch.bool)
    off = normalized[mask]
    return {
        "sample_delta_energy": sample,
        "mean_delta_energy": mean,
        "mean_delta_energy_fraction": mean / max(sample, 1e-30),
        "pair_delta_cosine_median": float(off.median()),
        "negative_pair_fraction": float((off < 0).double().mean()),
    }


def tensor_difference_family(
    baseline: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    names: Sequence[str],
) -> tuple[dict[str, float], torch.Tensor]:
    differences = torch.stack(
        [(candidate[name] - baseline[name]).double().reshape(-1) for name in names]
    )
    summary = _difference_summary_from_gram(differences @ differences.T)
    return summary, differences.mean(0)


def _effective_difference_inner(
    pairs: Mapping[str, Mapping[str, str]],
    left_from: Mapping[str, torch.Tensor],
    left_to: Mapping[str, torch.Tensor],
    right_from: Mapping[str, torch.Tensor],
    right_to: Mapping[str, torch.Tensor],
) -> float:
    return (
        effective_inner(pairs, left_to, right_to)
        - effective_inner(pairs, left_to, right_from)
        - effective_inner(pairs, left_from, right_to)
        + effective_inner(pairs, left_from, right_from)
    )


def effective_difference_family(
    pairs: Mapping[str, Mapping[str, str]],
    baseline: Mapping[str, Mapping[str, torch.Tensor]],
    candidate: Mapping[str, Mapping[str, torch.Tensor]],
    names: Sequence[str],
) -> dict[str, float]:
    gram = torch.empty(len(names), len(names), dtype=torch.float64)
    for left, left_name in enumerate(names):
        for right in range(left, len(names)):
            right_name = names[right]
            value = _effective_difference_inner(
                pairs,
                baseline[left_name],
                candidate[left_name],
                baseline[right_name],
                candidate[right_name],
            )
            gram[left, right] = gram[right, left] = value
    return _difference_summary_from_gram(gram)


def program_sketch(value: torch.Tensor) -> list[float]:
    flat = value.detach().double().reshape(-1)
    generator = torch.Generator(device="cpu").manual_seed(2026080501)
    buckets = torch.randint(
        0, PROGRAM_SKETCH_WIDTH, (flat.numel(),), generator=generator
    )
    signs = torch.randint(0, 2, (flat.numel(),), generator=generator).double()
    signs.mul_(2).sub_(1)
    sketch = torch.zeros(PROGRAM_SKETCH_WIDTH, dtype=torch.float64)
    sketch.scatter_add_(0, buckets, flat * signs)
    return sketch.tolist()


def _pairwise_cosine_summary(matrix: torch.Tensor) -> dict[str, float]:
    norm = torch.linalg.vector_norm(matrix, dim=1)
    cosine = (matrix @ matrix.T) / (
        norm[:, None].clamp_min(1e-30) * norm[None].clamp_min(1e-30)
    )
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool)
    values = cosine[mask]
    mean = matrix.mean(0)
    return {
        "pair_cosine_mean": float(values.mean()),
        "pair_cosine_median": float(values.median()),
        "negative_pair_fraction": float((values < 0).double().mean()),
        "full24_energy_retention": float(
            mean.square().sum() / matrix.square().sum(1).mean().clamp_min(1e-30)
        ),
    }


def _credit_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    directions = []
    binary = []
    semantic = []
    mode_values: dict[str, list[float]] = {}
    for row in rows:
        credit = row["program_credit"]
        seeds = credit["direction_seeds"]
        values = credit["credits"]
        tensors = [program_direction(seed, (320, 256)).reshape(-1) for seed in seeds]
        directions.append(program_cotangent(tensors, values).double())
        binary_values = [
            value if mode == "binary_discordant" else 0.0
            for value, mode in zip(values, credit["credit_modes"], strict=True)
        ]
        semantic_values = [
            value if mode == "paired_failure_semantic" else 0.0
            for value, mode in zip(values, credit["credit_modes"], strict=True)
        ]
        binary.append(program_cotangent(tensors, binary_values).double())
        semantic.append(program_cotangent(tensors, semantic_values).double())
        for value, mode in zip(values, credit["credit_modes"], strict=True):
            mode_values.setdefault(mode, []).append(float(value))
    total = torch.stack(directions)
    binary_total = torch.stack(binary).mean(0)
    semantic_total = torch.stack(semantic).mean(0)
    combined = total.mean(0)
    return {
        "task_cotangent": _pairwise_cosine_summary(total),
        "pair_credit": {
            mode: {
                "pairs": len(values),
                "nonzero_pairs": sum(value != 0 for value in values),
                "absolute_value": distribution([abs(value) for value in values]),
            }
            for mode, values in sorted(mode_values.items())
        },
        "full24_component_energy": {
            "binary": float(binary_total.square().sum()),
            "semantic": float(semantic_total.square().sum()),
            "combined": float(combined.square().sum()),
            "binary_semantic_cosine": float(
                (binary_total @ semantic_total)
                / (
                    binary_total.norm().clamp_min(1e-30)
                    * semantic_total.norm().clamp_min(1e-30)
                )
            ),
        },
    }


def _checkpoint_summary(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Any]:
    selected = [row for row in rows if row["checkpoint_label"] == label]
    action_rows = [
        row for row in selected
        if row["fixed_action_condition_from_demo_0"] is not None
    ]
    conditions = ("demo_1", "wrong_0", "reversed_0", "shuffled_0")
    return {
        "tasks": len(selected),
        "geometry_demo_0": {
            key: distribution(
                [float(row["geometry_demo_0"][key]) for row in selected]
            )
            for key in (
                "effective_lora_norm_scaled",
                "stable_rank_mean",
                "top_singular_energy_mean",
            )
        },
        "same_task_video": {
            "program_centered_variance_over_sample_energy": distribution(
                [
                    float(
                        row["program_same_task_video_variance"]
                        ["centered_variance_over_sample_energy"]
                    )
                    for row in selected
                ]
            ),
            "effective_ba_centered_variance_over_sample_energy": distribution(
                [
                    float(
                        row["effective_ba_same_task_video_variance"]
                        ["centered_variance_over_sample_energy"]
                    )
                    for row in selected
                ]
            ),
        },
        "condition_relative_l2": {
            name: {
                "program": distribution(
                    [
                        float(row["program_condition_from_demo_0"][name]["relative_l2"])
                        for row in selected
                    ]
                ),
                "effective_ba": distribution(
                    [
                        float(row["effective_ba_condition_from_demo_0"][name]["relative_l2"])
                        for row in selected
                    ]
                ),
                "fixed_action": distribution(
                    [
                        float(
                            row["fixed_action_condition_from_demo_0"][name]["relative_l2"]
                        )
                        for row in action_rows
                    ]
                ),
            }
            for name in conditions
        },
    }


def _transition_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["checkpoint_label"] == "cycle1"]
    sketches = torch.tensor(
        [
            row["checkpoint_update"]["program_task_mean_delta_sketch"]
            for row in selected
        ],
        dtype=torch.float64,
    )
    action_rows = [
        row for row in selected
        if row["checkpoint_update"]["fixed_action_demo_0"] is not None
    ]
    result = {
        "program_demo_0_relative_l2": distribution(
            [
                float(row["checkpoint_update"]["program_demo_0"]["relative_l2"])
                for row in selected
            ]
        ),
        "effective_ba_demo_0_relative_l2": distribution(
            [
                float(row["checkpoint_update"]["effective_ba_demo_0"]["relative_l2"])
                for row in selected
            ]
        ),
        "fixed_action_demo_0_relative_l2": distribution(
            [
                float(row["checkpoint_update"]["fixed_action_demo_0"]["relative_l2"])
                for row in action_rows
            ]
        ),
        "same_task_update_family": {
            level: {
                key: distribution(
                    [
                        float(
                            row["checkpoint_update"]
                            [f"{level}_same_task_update_family"][key]
                        )
                        for row in selected
                    ]
                )
                for key in (
                    "mean_delta_energy_fraction",
                    "pair_delta_cosine_median",
                    "negative_pair_fraction",
                )
            }
            for level in ("program", "effective_ba")
        },
        "task_mean_program_delta_countsketch": _pairwise_cosine_summary(sketches),
        "train_program_credit": _credit_summary(selected),
    }
    result["per_task"] = [
        {
            "global_task_id": int(row["global_task_id"]),
            "suite": str(row["suite"]),
            "suite_task_id": int(row["suite_task_id"]),
            "program_relative_l2": float(
                row["checkpoint_update"]["program_demo_0"]["relative_l2"]
            ),
            "effective_ba_relative_l2": float(
                row["checkpoint_update"]["effective_ba_demo_0"]["relative_l2"]
            ),
            "program_video_update_mean_energy_fraction": float(
                row["checkpoint_update"]
                ["program_same_task_update_family"]
                ["mean_delta_energy_fraction"]
            ),
            "effective_ba_video_update_mean_energy_fraction": float(
                row["checkpoint_update"]
                ["effective_ba_same_task_update_family"]
                ["mean_delta_energy_fraction"]
            ),
        }
        for row in sorted(selected, key=lambda value: int(value["global_task_id"]))
    ]
    return result


def summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        **{label: _checkpoint_summary(rows, label) for label in CHECKPOINT_LABELS},
        "as125_to_cycle1": _transition_summary(rows),
    }
