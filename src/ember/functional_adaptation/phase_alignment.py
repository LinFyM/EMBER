"""Task-equal functional coordinates and monotone trajectory phase alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class FunctionalWhitener:
    """A PCA/whitening transform fitted only on declared fit-task responses."""

    mean: torch.Tensor
    components: torch.Tensor
    scales: torch.Tensor
    explained_variance_ratio: float

    def transform(self, response: torch.Tensor) -> torch.Tensor:
        if response.ndim < 2:
            raise ValueError("functional response must retain a time dimension")
        flat = response.double().flatten(1)
        if flat.shape[1] != self.mean.numel():
            raise ValueError("functional response width changed")
        transformed = (flat - self.mean) @ self.components.T / self.scales
        if not torch.isfinite(transformed).all():
            raise ValueError("functional whitening produced nonfinite coordinates")
        return transformed

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        return {
            "mean": self.mean,
            "components": self.components,
            "scales": self.scales,
            "explained_variance_ratio": self.explained_variance_ratio,
        }


def fit_task_equal_whitener(
    responses: Sequence[torch.Tensor],
    task_ids: Sequence[int],
    *,
    width: int,
) -> FunctionalWhitener:
    """Fit weighted PCA so tasks, members within tasks, and states are equal."""

    if len(responses) != len(task_ids) or not responses or width <= 0:
        raise ValueError("functional whitening panel is invalid")
    task_count = len(set(int(value) for value in task_ids))
    members_per_task = {
        task_id: sum(int(value) == task_id for value in task_ids)
        for task_id in set(int(value) for value in task_ids)
    }
    rows, weights = [], []
    for response, task_id_value in zip(responses, task_ids, strict=True):
        task_id = int(task_id_value)
        flat = response.double().flatten(1)
        if flat.shape[0] < 2 or not torch.isfinite(flat).all():
            raise ValueError("functional response sequence is invalid")
        rows.append(flat)
        weights.append(
            torch.full(
                (flat.shape[0],),
                1.0 / (task_count * members_per_task[task_id] * flat.shape[0]),
                dtype=torch.double,
            )
        )
    matrix = torch.cat(rows)
    sample_weights = torch.cat(weights)
    mean = (sample_weights[:, None] * matrix).sum(dim=0)
    centered = matrix - mean
    covariance = centered.T @ (centered * sample_weights[:, None])
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[order]
    if width > matrix.shape[1] or float(eigenvalues[width - 1]) <= 0:
        raise ValueError("functional response rank is below the requested width")
    components = eigenvectors[:, order[:width]].T.contiguous()
    scales = eigenvalues[:width].sqrt()
    positive_total = eigenvalues.clamp_min(0).sum()
    return FunctionalWhitener(
        mean=mean,
        components=components,
        scales=scales,
        explained_variance_ratio=float(eigenvalues[:width].sum() / positive_total),
    )


def _resample_at_progress(
    sequence: torch.Tensor, progress: torch.Tensor, *, count: int
) -> torch.Tensor:
    if (
        sequence.ndim != 2
        or progress.shape != (sequence.shape[0],)
        or sequence.shape[0] < 2
        or count < 2
        or float(progress[0]) != 0.0
        or float(progress[-1]) != 1.0
    ):
        raise ValueError("functional phase sequence is invalid")
    targets = torch.linspace(0, 1, count, dtype=sequence.dtype)
    right = torch.searchsorted(progress, targets, right=True).clamp(
        min=1, max=sequence.shape[0] - 1
    )
    left = right - 1
    denominator = progress[right] - progress[left]
    fraction = torch.where(
        denominator > 0,
        (targets - progress[left]) / denominator,
        torch.zeros_like(targets),
    )
    return sequence[left] + fraction[:, None] * (sequence[right] - sequence[left])


def uniform_time_embedding(sequence: torch.Tensor, *, count: int = 8) -> torch.Tensor:
    """Resample a functional sequence at fixed normalized-time quantiles."""

    progress = torch.linspace(0, 1, sequence.shape[0], dtype=sequence.dtype)
    return _resample_at_progress(sequence, progress, count=count)


def arc_length_phase_embedding(
    sequence: torch.Tensor, *, count: int = 8
) -> torch.Tensor:
    """Resample at fixed cumulative functional-distance quantiles."""

    distances = torch.linalg.vector_norm(sequence[1:] - sequence[:-1], dim=1)
    cumulative = torch.cat(
        [torch.zeros(1, dtype=sequence.dtype), distances.cumsum(dim=0)]
    )
    if float(cumulative[-1]) <= 0:
        raise ValueError("functional trajectory has zero arc length")
    return _resample_at_progress(
        sequence, cumulative / cumulative[-1], count=count
    )


def arc_length_phase_indices(
    sequence: torch.Tensor, *, count: int = 8
) -> torch.Tensor:
    """Select distinct ordered real states near fixed functional-phase quantiles."""

    if sequence.ndim != 2 or sequence.shape[0] < count or count < 2:
        raise ValueError("functional phase index panel is invalid")
    distances = torch.linalg.vector_norm(sequence[1:] - sequence[:-1], dim=1)
    cumulative = torch.cat(
        [torch.zeros(1, dtype=sequence.dtype), distances.cumsum(dim=0)]
    )
    if float(cumulative[-1]) <= 0:
        raise ValueError("functional trajectory has zero arc length")
    progress = cumulative / cumulative[-1]
    targets = torch.linspace(0, 1, count, dtype=sequence.dtype)
    indices = torch.empty(count, dtype=torch.long)
    indices[0] = 0
    indices[-1] = sequence.shape[0] - 1
    for phase in range(1, count - 1):
        lower = int(indices[phase - 1]) + 1
        upper = sequence.shape[0] - (count - phase)
        allowed = progress[lower : upper + 1]
        indices[phase] = lower + int((allowed - targets[phase]).abs().argmin())
    return indices
