"""Exact held-task qualification for policy-behavior Program coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open

from ember.ecp.bank_conditioning.consensus import truncated_mean_update
from ember.ecp.behavior.codes import BehaviorCodeAuthority


FAMILY_SLICES = {
    "q": (0, 3),
    "v": (3, 6),
    "action_in": (6, 7),
    "action_out": (7, 8),
}


@dataclass(frozen=True)
class BehaviorBasis:
    eigenvectors: torch.Tensor
    eigenvalues: torch.Tensor
    coordinates: torch.Tensor
    norms: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor
    train_sqrt_weights: torch.Tensor


def factor_inner(
    left: tuple[torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    left_a, left_b = left
    right_a, right_b = right
    return (left_b.T @ right_b * (left_a @ right_a.T)).sum()


def factor_cosine(
    left: tuple[torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor],
) -> float:
    dot = factor_inner(left, right)
    left_norm = factor_inner(left, left).clamp_min(0).sqrt()
    right_norm = factor_inner(right, right).clamp_min(0).sqrt()
    return float((dot / (left_norm * right_norm).clamp_min(1e-20)).clamp(-1, 1))


def _factor_path(roots: Sequence[Path], task: int) -> Path:
    rows = []
    for root in roots:
        rows.extend(root.glob(f"shard_*/task_{task:03d}.safetensors"))
    if len(rows) != 1:
        raise ValueError(f"task {task} behavior factor authority changed")
    return rows[0]


def load_behavior_panels(
    roots: Sequence[Path],
    task: int,
    selected_targets: Sequence[int],
    device: torch.device,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    panel_a = []
    panel_b = []
    consensus = []
    with safe_open(
        str(_factor_path(roots, task)), framework="pt", device=str(device)
    ) as handle:
        for target in selected_targets:
            first = (
                handle.get_tensor(f"target_{target:02d}.panel_a.a").float(),
                handle.get_tensor(f"target_{target:02d}.panel_a.b").float(),
            )
            second = (
                handle.get_tensor(f"target_{target:02d}.panel_b.a").float(),
                handle.get_tensor(f"target_{target:02d}.panel_b.b").float(),
            )
            panel_a.append(first)
            panel_b.append(second)
            consensus.append(truncated_mean_update((first, second), rank=4))
    return tuple(panel_a), tuple(panel_b), tuple(consensus)


def behavior_gram(
    behavior: Mapping[int, tuple[Any, ...]],
    tasks: Sequence[int],
    target: int,
) -> torch.Tensor:
    return torch.stack(
        [
            torch.stack(
                [factor_inner(behavior[left][target], behavior[right][target]) for right in tasks]
            )
            for left in tasks
        ]
    )


def fit_behavior_basis(
    gram: torch.Tensor,
    train_index: torch.Tensor,
    train_weights: torch.Tensor,
    *,
    dimension: int,
) -> BehaviorBasis:
    norms = gram.diag().clamp_min(1e-20).sqrt()
    cosine = gram / (norms[:, None] * norms[None]).clamp_min(1e-20)
    train_gram = cosine.index_select(0, train_index).index_select(1, train_index)
    sqrt_weights = train_weights.sqrt()
    eigenvalues, eigenvectors = torch.linalg.eigh(
        sqrt_weights[:, None] * train_gram * sqrt_weights[None]
    )
    order = eigenvalues.argsort(descending=True)[:dimension]
    eigenvalues = eigenvalues[order].clamp_min(1e-6)
    eigenvectors = eigenvectors[:, order]
    coordinates = (
        cosine[:, train_index]
        @ (sqrt_weights[:, None] * eigenvectors)
        / eigenvalues.sqrt()[None]
    )
    train_coordinates = coordinates.index_select(0, train_index)
    mean = (train_coordinates * train_weights[:, None]).sum(0)
    scale = (
        (train_coordinates - mean).square() * train_weights[:, None]
    ).sum(0).sqrt().clamp_min(1e-3)
    return BehaviorBasis(
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        coordinates=coordinates,
        norms=norms,
        mean=mean,
        scale=scale,
        train_sqrt_weights=sqrt_weights,
    )


def _rank4_pair(
    authority: BehaviorCodeAuthority,
    behavior: Mapping[int, tuple[Any, ...]],
    target: int,
    coefficients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = authority.train_sqrt_weights * (
        authority.eigenvectors[target]
        @ (coefficients / authority.eigenvalues[target].sqrt())
    )
    all_index = {task: index for index, task in enumerate(authority.task_ids)}
    pairs = []
    for position, task in enumerate(authority.fit_task_ids):
        a, b = behavior[task][target]
        pairs.append(
            (
                a,
                b * (weights[position] / authority.norms[target, all_index[task]]),
            )
        )
    return truncated_mean_update(tuple(pairs), rank=4)


def _family(values: Sequence[float]) -> dict[str, float]:
    return {
        family: sum(values[start:stop]) / (stop - start)
        for family, (start, stop) in FAMILY_SLICES.items()
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    rows = torch.tensor(tuple(values), dtype=torch.float32).sort().values
    return {
        "minimum": float(rows[0]),
        "p10": float(torch.quantile(rows, 0.1)),
        "median": float(rows.median()),
        "mean": float(rows.mean()),
        "maximum": float(rows[-1]),
    }


def _summarize(rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"tasks": list(rows), "by_role": {}}
    groups = {
        role: [row for row in rows if row["role"] == role]
        for role in ("meta_held", "target_held")
    }
    for role, selected in groups.items():
        result["by_role"][role] = {}
        for metric in metrics:
            result["by_role"][role][metric] = {
                family: _distribution(
                    [float(row[metric][family]) for row in selected]
                )
                for family in FAMILY_SLICES
            }
            result["by_role"][role][metric]["overall"] = _distribution(
                [
                    sum(float(row[metric][family]) for family in FAMILY_SLICES)
                    / len(FAMILY_SLICES)
                    for row in selected
                ]
            )
    result["role_equal_median"] = {}
    for metric in metrics:
        result["role_equal_median"][metric] = {}
        for family in (*FAMILY_SLICES, "overall"):
            result["role_equal_median"][metric][family] = sum(
                result["by_role"][role][metric][family]["median"]
                for role in groups
            ) / len(groups)
    return result


def _coordinate_rows(
    records: Sequence[Mapping[str, Any]], authority: BehaviorCodeAuthority
) -> list[dict[str, Any]]:
    rows = []
    mean = authority.mean
    view_names = ("same_a", "same_b", "k1", "k4")
    for record in records:
        task = int(record["authority_id"])
        target = authority.target(task, standardized=False)
        views = {
            name: record["behavior_predictions"][name].to(target.device)
            for name in view_names
        }
        values: dict[str, dict[str, float]] = {}
        for name, prediction in views.items():
            values[f"{name}_absolute"] = _family(
                F.cosine_similarity(prediction, target, dim=-1).tolist()
            )
            values[f"{name}_specific"] = _family(
                F.cosine_similarity(prediction - mean, target - mean, dim=-1).tolist()
            )
        pairs = []
        for left in range(len(view_names)):
            for right in range(left + 1, len(view_names)):
                pairs.append(
                    F.cosine_similarity(
                        views[view_names[left]] - mean,
                        views[view_names[right]] - mean,
                        dim=-1,
                    )
                )
        values["cross_view_specific"] = _family(torch.stack(pairs).mean(0).tolist())
        rows.append({"task": task, "role": record["role"], **values})
    return rows


def _exact_rows(
    records: Sequence[Mapping[str, Any]], authority: BehaviorCodeAuthority
) -> list[dict[str, Any]]:
    device = authority.coordinates.device
    panels_a: dict[int, tuple[Any, ...]] = {}
    panels_b: dict[int, tuple[Any, ...]] = {}
    behavior: dict[int, tuple[Any, ...]] = {}
    for task in authority.task_ids:
        panels_a[task], panels_b[task], behavior[task] = load_behavior_panels(
            authority.factor_roots,
            task,
            authority.selected_targets,
            device,
        )
    by_task = {int(row["authority_id"]): row for row in records}
    wrong = {}
    for role in ("meta_held", "target_held"):
        tasks = sorted(
            task for task, row in by_task.items() if row["role"] == role
        )
        for index, task in enumerate(tasks):
            wrong[task] = tasks[(index + 1) % len(tasks)]
    universal = []
    train_weights = authority.train_sqrt_weights.square()
    for target in range(len(authority.selected_targets)):
        universal.append(
            truncated_mean_update(
                tuple(
                    (
                        behavior[task][target][0],
                        behavior[task][target][1] * train_weights[position],
                    )
                    for position, task in enumerate(authority.fit_task_ids)
                ),
                rank=4,
            )
        )
    rows = []
    for task, record in sorted(by_task.items()):
        prediction = record["behavior_predictions"]["k4"].to(device)
        wrong_prediction = by_task[wrong[task]]["behavior_predictions"]["k4"].to(device)
        values = {
            name: []
            for name in (
                "behavior_panel_repeatability",
                "predicted_to_panel_b",
                "predicted_to_consensus",
                "wrong_program_margin_to_panel_b",
                "top16_span_oracle_to_panel_b",
                "top16_span_oracle_to_consensus",
                "universal_to_panel_b",
            )
        }
        for target in range(len(authority.selected_targets)):
            predicted = _rank4_pair(authority, behavior, target, prediction[target])
            incorrect = _rank4_pair(
                authority, behavior, target, wrong_prediction[target]
            )
            oracle = _rank4_pair(
                authority,
                behavior,
                target,
                authority.target(task, standardized=False)[target],
            )
            correct_b = factor_cosine(predicted, panels_b[task][target])
            values["behavior_panel_repeatability"].append(
                factor_cosine(panels_a[task][target], panels_b[task][target])
            )
            values["predicted_to_panel_b"].append(correct_b)
            values["predicted_to_consensus"].append(
                factor_cosine(predicted, behavior[task][target])
            )
            values["wrong_program_margin_to_panel_b"].append(
                correct_b - factor_cosine(incorrect, panels_b[task][target])
            )
            values["top16_span_oracle_to_panel_b"].append(
                factor_cosine(oracle, panels_b[task][target])
            )
            values["top16_span_oracle_to_consensus"].append(
                factor_cosine(oracle, behavior[task][target])
            )
            values["universal_to_panel_b"].append(
                factor_cosine(universal[target], panels_b[task][target])
            )
        rows.append(
            {
                "task": task,
                "role": record["role"],
                **{name: _family(value) for name, value in values.items()},
            }
        )
    return rows


def build_behavior_gate(
    records: Sequence[Mapping[str, Any]],
    authority: BehaviorCodeAuthority,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    if (
        len(records) != 20
        or {int(row["authority_id"]) for row in records}
        != set(authority.held_task_ids)
    ):
        raise ValueError("behavior-code held panel changed")
    coordinate_metrics = tuple(
        [f"{view}_{kind}" for view in ("same_a", "same_b", "k1", "k4") for kind in ("absolute", "specific")]
        + ["cross_view_specific"]
    )
    exact_metrics = (
        "behavior_panel_repeatability",
        "predicted_to_panel_b",
        "predicted_to_consensus",
        "wrong_program_margin_to_panel_b",
        "top16_span_oracle_to_panel_b",
        "top16_span_oracle_to_consensus",
        "universal_to_panel_b",
    )
    coordinate = _summarize(_coordinate_rows(records, authority), coordinate_metrics)
    exact = _summarize(_exact_rows(records, authority), exact_metrics)
    role_equal = exact["role_equal_median"]
    view_absolute = [
        coordinate["role_equal_median"][f"{view}_absolute"]["overall"]
        for view in ("same_a", "same_b", "k1", "k4")
    ]
    prediction = role_equal["predicted_to_panel_b"]
    checks = {
        "exact_overall": prediction["overall"]
        >= float(thresholds["exact_to_panel_b"]),
        "exact_consensus": role_equal["predicted_to_consensus"]["overall"]
        >= float(thresholds["exact_to_consensus"]),
        "q": prediction["q"] >= float(thresholds["q"]),
        "v": prediction["v"] >= float(thresholds["v"]),
        "action_in": prediction["action_in"] >= float(thresholds["action_in"]),
        "action_out": prediction["action_out"] >= float(thresholds["action_out"]),
        "wrong_program_margin": role_equal["wrong_program_margin_to_panel_b"][
            "overall"
        ]
        >= float(thresholds["wrong_program_margin"]),
        "meta_held": exact["by_role"]["meta_held"]["predicted_to_panel_b"][
            "overall"
        ]["median"]
        >= float(thresholds["minimum_role"]),
        "target_held": exact["by_role"]["target_held"]["predicted_to_panel_b"][
            "overall"
        ]["median"]
        >= float(thresholds["minimum_role"]),
        "language_reference_delta": prediction["overall"]
        - float(thresholds["language_reference"])
        >= float(thresholds["minimum_language_delta"]),
        "all_video_views": min(view_absolute)
        >= float(thresholds["minimum_view_coordinate"]),
        "cross_view_consistency": coordinate["role_equal_median"][
            "cross_view_specific"
        ]["overall"]
        >= float(thresholds["minimum_cross_view_specific"]),
    }
    return {
        "schema_version": "ember_ecp_g2_behavior_gate_v1",
        "thresholds": dict(thresholds),
        "coordinate": coordinate,
        "exact_rank4": exact,
        "checks": checks,
        "passed": all(checks.values()),
    }
