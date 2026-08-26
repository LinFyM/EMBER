"""Analytic-dual basis algebra for the fit-only G3 capacity probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional

from ember.ecp.native_factors import rms_normalize
from ember.ecp.shared_compiler_native_teacher import (
    factor_subspace_loss,
    low_rank_update_direction_loss,
    small_core_singular_values,
)


WORKER_SCHEMA = "ember_ecp_g3_dual_basis_worker_v1"
BASIS_SCHEMA = "ember_ecp_g3_dual_basis_loto_v1"
REPLAY_SCHEMA = "ember_ecp_g3_dual_basis_replay_worker_v1"
REPORT_SCHEMA = "ember_ecp_g3_dual_basis_report_v1"
DEFAULT_TARGETS = (20, 21, 36, 37)
DEFAULT_BASIS_DIMENSIONS = (16, 32, 64, 96, 128)


@dataclass(frozen=True)
class StableDualFactor:
    """One stable eigensystem of a fixed native candidate covariance."""

    scatter: torch.Tensor
    basis: torch.Tensor
    retained: torch.Tensor
    mean: torch.Tensor

    @property
    def stable_rank(self) -> int:
        return int(self.retained.numel())

    @property
    def condition(self) -> float:
        return float(self.retained[-1] / self.retained[0])

    def solve(
        self, desired: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        vectors = desired.reshape(-1, desired.shape[-1]).double()
        if vectors.shape[1] != self.scatter.shape[0]:
            raise RuntimeError("analytic dual desired width changed")
        coordinates = vectors @ self.basis
        projected = coordinates @ self.basis.transpose(0, 1)
        dual = (coordinates / self.retained[None]) @ self.basis.transpose(0, 1)
        norms = dual.norm(dim=-1).clamp_min(1e-30)
        unit = dual / norms[:, None]
        replayed = dual @ self.scatter
        projection_cosine = functional.cosine_similarity(
            projected.float(), vectors.float(), dim=-1
        )
        covariance_cosine = functional.cosine_similarity(
            replayed.float(), projected.float(), dim=-1
        )
        return unit, norms, projection_cosine, {
            "stable_rank": self.stable_rank,
            "condition": self.condition,
            "projection_cosine_mean": float(projection_cosine.mean()),
            "projection_cosine_minimum": float(projection_cosine.min()),
            "covariance_cosine_mean": float(covariance_cosine.mean()),
            "dual_norm_median": float(norms.median()),
            "dual_norm_maximum": float(norms.max()),
        }


def stable_dual_factor(
    values: torch.Tensor,
    mass: torch.Tensor,
    *,
    relative_singular_threshold: float,
) -> StableDualFactor:
    """Factor one fixed-measure centered covariance in FP64 exactly once."""

    flat = values.reshape(-1, values.shape[-1]).double()
    weights = mass.reshape(-1).double()
    if flat.shape[0] != weights.numel() or flat.shape[0] <= 1:
        raise RuntimeError("native values and fixed measure changed shape")
    weights = weights / weights.sum().clamp_min(1e-30)
    mean = torch.einsum("n,nd->d", weights, flat)
    centered = flat - mean
    scatter = centered.transpose(0, 1) @ (weights[:, None] * centered)
    eigenvalues, eigenvectors = torch.linalg.eigh(scatter)
    singular = eigenvalues.clamp_min(0).sqrt()
    maximum = singular[-1].clamp_min(torch.finfo(singular.dtype).tiny)
    keep = singular > maximum * relative_singular_threshold
    if not torch.any(keep):
        raise RuntimeError("analytic dual retained no stable native direction")
    return StableDualFactor(
        scatter=scatter,
        basis=eigenvectors[:, keep],
        retained=eigenvalues[keep].clamp_min(1e-30),
        mean=mean,
    )


def tensor_prefix(record_index: int, target: int) -> str:
    return f"record/{record_index:06d}/target/{target:02d}"


def side_tensor_blocks(
    tensors: Mapping[str, torch.Tensor],
    *,
    record_index: int,
    target_row: Mapping[str, Any],
    side: str,
) -> tuple[torch.Tensor, ...]:
    target = int(target_row["target"])
    prefix = tensor_prefix(record_index, target)
    if side == "input":
        value = tensors[f"{prefix}/input/unit"]
        expected = int(target_row["input_width"])
        if value.ndim != 3 or value.shape[-1] != expected:
            raise ValueError("dual-basis input block changed shape")
        return tuple(value[member] for member in range(value.shape[0]))
    if side != "output":
        raise ValueError("dual-basis side changed")
    groups = int(target_row["groups"])
    width = int(target_row["group_width"])
    blocks = []
    for group in range(groups):
        value = tensors[f"{prefix}/output/g{group:03d}/unit"]
        if value.ndim != 3 or value.shape[-1] != width:
            raise ValueError("dual-basis output block changed shape")
        blocks.extend(value[member] for member in range(value.shape[0]))
    return tuple(blocks)


def task_equal_scatter(rows: Sequence[torch.Tensor]) -> torch.Tensor:
    """Average normalized rank-block projectors so every task has unit mass."""

    if not rows:
        raise ValueError("dual-basis task has no rows")
    projectors = []
    width = rows[0].shape[-1]
    for value in rows:
        if value.ndim != 2 or value.shape[1] != width or value.shape[0] <= 0:
            raise ValueError("dual-basis rank block changed shape")
        _left, singular, right = torch.linalg.svd(
            value.float(), full_matrices=False
        )
        stable_rank = int(
            (singular > singular[0].clamp_min(1e-30) * 1e-5).sum()
        )
        if stable_rank <= 0:
            raise ValueError("dual-basis rank block has no stable row space")
        basis = right[:stable_rank].transpose(0, 1)
        projectors.append(basis @ basis.transpose(0, 1) / stable_rank)
    return torch.stack(projectors).mean(0)


def projected_duals(
    dual: torch.Tensor,
    basis: torch.Tensor,
    dimensions: Sequence[int],
) -> tuple[torch.Tensor, tuple[int, ...], tuple[float, ...]]:
    if dual.ndim != 3 or basis.ndim != 2 or dual.shape[-1] != basis.shape[0]:
        raise ValueError("dual-basis query projection changed shape")
    flat = dual.reshape(-1, dual.shape[-1]).double()
    values = [flat]
    effective = [dual.shape[-1]]
    projection_cosines = [1.0]
    for requested in dimensions:
        width = min(int(requested), basis.shape[1])
        if width <= 0:
            raise ValueError("dual-basis projection has zero width")
        current = basis[:, :width].to(flat)
        projected = (flat @ current) @ current.transpose(0, 1)
        projection_cosines.append(
            float(functional.cosine_similarity(projected, flat, dim=-1).mean())
        )
        values.append(projected)
        effective.append(width)
    return (
        torch.stack(values).reshape(len(values), *dual.shape),
        tuple(effective),
        tuple(projection_cosines),
    )


def centered_scores(
    queries: torch.Tensor,
    values: torch.Tensor,
    mass: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        queries.ndim != 4
        or values.shape[:-1] != mass.shape
        or queries.shape[-1] != values.shape[-1]
    ):
        raise ValueError("dual-basis signed score replay changed shape")
    flat_values = values.reshape(-1, values.shape[-1]).double()
    weights = mass.reshape(-1).double().clamp_min(1e-30)
    weights = weights / weights.sum()
    flat_queries = queries.reshape(-1, queries.shape[-1]).double()
    scores = flat_queries @ flat_values.transpose(0, 1)
    scores = scores - torch.einsum("qn,n->q", scores, weights)[:, None]
    return (
        scores.reshape(*queries.shape[:-1], flat_values.shape[0]),
        flat_values,
        weights,
    )


def pool_centered_scores(
    scores: torch.Tensor,
    flat_values: torch.Tensor,
    weights: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if scores.ndim != 4 or scale.shape != scores.shape[:-1]:
        raise ValueError("dual-basis signed pooling scale changed shape")
    scaled = scores * scale[..., None]
    log_mass = weights.log().reshape(1, 1, 1, -1)
    positive = (log_mass + scaled).softmax(-1)
    negative = (log_mass - scaled).softmax(-1)
    return (positive - negative) @ flat_values


def signed_pool_queries(
    queries: torch.Tensor,
    values: torch.Tensor,
    mass: torch.Tensor,
    *,
    score_bound: float,
) -> torch.Tensor:
    scores, flat_values, weights = centered_scores(queries, values, mass)
    scale = score_bound / scores.abs().amax(-1).clamp_min(1e-12)
    return rms_normalize(
        pool_centered_scores(scores, flat_values, weights, scale)
    )


def signed_pool_output_groups(
    queries: Sequence[torch.Tensor],
    grouped_values: torch.Tensor,
    mass: torch.Tensor,
    *,
    score_bound: float,
) -> torch.Tensor:
    if len(queries) != grouped_values.shape[0] or not queries:
        raise ValueError("dual-basis output group count changed")
    prepared = [
        centered_scores(query, grouped_values[group], mass)
        for group, query in enumerate(queries)
    ]
    maxima = torch.stack([scores.abs().amax(-1) for scores, _, _ in prepared])
    scale = score_bound / maxima.amax(0).clamp_min(1e-12)
    blocks = tuple(
        pool_centered_scores(scores, values, weights, scale)
        for scores, values, weights in prepared
    )
    return rms_normalize(torch.cat(blocks, dim=-1))


def factor_cosines(
    student: torch.Tensor, teacher: torch.Tensor
) -> tuple[float, float]:
    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError("dual-basis factor comparison changed shape")
    values = functional.cosine_similarity(student.float(), teacher.float(), dim=-1)
    return float(values.mean()), float(values.min())


def update_geometry(
    student_a: torch.Tensor,
    student_b: torch.Tensor,
    teacher_a: torch.Tensor,
    teacher_b: torch.Tensor,
    scales: torch.Tensor,
) -> dict[str, Any]:
    if scales.shape != (student_a.shape[0],):
        raise ValueError("dual-basis replay scale rank changed")
    student_b_scaled = (student_b * scales[:, None]).transpose(0, 1)
    teacher_b_scaled = (teacher_b * scales[:, None]).transpose(0, 1)
    loss = low_rank_update_direction_loss(
        student_a, student_b_scaled, teacher_a, teacher_b_scaled
    )
    student_singular = small_core_singular_values(student_a, student_b_scaled)
    teacher_singular = small_core_singular_values(teacher_a, teacher_b_scaled)
    ratio = student_singular / teacher_singular.clamp_min(1e-12)
    return {
        "update_cosine": float(1.0 - loss),
        "input_subspace_similarity": float(
            1.0 - factor_subspace_loss(student_a, teacher_a)
        ),
        "output_subspace_similarity": float(
            1.0 - factor_subspace_loss(student_b, teacher_b)
        ),
        "small_core_log_spectrum_rmse": float(
            (
                student_singular.clamp_min(1e-12).log()
                - teacher_singular.clamp_min(1e-12).log()
            ).square().mean().sqrt()
        ),
        "small_core_singular_to_teacher": ratio.detach().cpu().tolist(),
        "minimum_small_core_singular_to_teacher": float(ratio.min()),
    }


def quantiles(values: Sequence[float]) -> dict[str, float | int]:
    tensor = torch.tensor(tuple(map(float, values)), dtype=torch.float64)
    if tensor.numel() <= 0 or not torch.isfinite(tensor).all():
        raise ValueError("dual-basis report received invalid values")
    return {
        "count": tensor.numel(),
        "minimum": float(tensor.min()),
        "p10": float(torch.quantile(tensor, 0.1)),
        "median": float(torch.quantile(tensor, 0.5)),
        "mean": float(tensor.mean()),
        "maximum": float(tensor.max()),
    }


def hierarchical_task_scores(
    rows: Sequence[Mapping[str, Any]], value_key: str
) -> tuple[dict[int, float], dict[int, float], dict[int, str]]:
    videos: dict[int, dict[int, list[float]]] = {}
    roles: dict[int, str] = {}
    for row in rows:
        task = int(row["authority_id"])
        video = int(row["video_demo"])
        role = str(row["role"])
        if task in roles and roles[task] != role:
            raise ValueError("dual-basis task role changed during report")
        roles[task] = role
        videos.setdefault(task, {}).setdefault(video, []).append(float(row[value_key]))
    task_mean = {}
    task_worst_video = {}
    for task, by_video in videos.items():
        video_means = [sum(values) / len(values) for values in by_video.values()]
        task_mean[task] = sum(video_means) / len(video_means)
        task_worst_video[task] = min(video_means)
    return task_mean, task_worst_video, roles


def task_distribution(
    rows: Sequence[Mapping[str, Any]], value_key: str
) -> dict[str, Any]:
    task_mean, task_worst, roles = hierarchical_task_scores(rows, value_key)
    task_mean_summary = quantiles(list(task_mean.values()))
    task_mean_summary["at_least_0.95_count"] = sum(
        value >= 0.95 for value in task_mean.values()
    )
    task_worst_summary = quantiles(list(task_worst.values()))
    task_worst_summary["at_least_0.95_count"] = sum(
        value >= 0.95 for value in task_worst.values()
    )
    return {
        "task_mean": task_mean_summary,
        "task_worst_video": task_worst_summary,
        "roles_task_mean": {
            role: quantiles(
                [value for task, value in task_mean.items() if roles[task] == role]
            )
            for role in sorted(set(roles.values()))
        },
    }


def probe_thresholds(summary: Mapping[str, Any]) -> dict[str, bool]:
    checks = {}
    for name in ("task_mean", "task_worst_video"):
        values = summary[name]
        required = (9 * int(values["count"]) + 9) // 10
        checks[f"{name}_median_at_least_0.98"] = (
            float(values["median"]) >= 0.98
        )
        checks[f"{name}_p10_at_least_0.95"] = float(values["p10"]) >= 0.95
        checks[f"{name}_at_least_90pct_tasks_at_0.95"] = (
            int(values["at_least_0.95_count"]) >= required
        )
    return checks
