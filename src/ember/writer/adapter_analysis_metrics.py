"""Gauge-aware and gauge-invariant metrics for generated PI05 LoRA states."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def state_row(state: Mapping[str, torch.Tensor], row: int) -> dict[str, torch.Tensor]:
    return {
        name: value[row].detach().to(device="cpu", dtype=torch.float32)
        for name, value in state.items()
    }


def _kind(module: str) -> str:
    if module.endswith("q_proj"):
        return "q"
    if module.endswith("v_proj"):
        return "v"
    return "action"


def effective_inner(
    pairs: Mapping[str, Mapping[str, str]],
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> float:
    total = 0.0
    for names in pairs.values():
        la, lb = left[names["a"]].double(), left[names["b"]].double()
        ra, rb = right[names["a"]].double(), right[names["b"]].double()
        total += float(((lb.T @ rb) * (la @ ra.T)).sum())
    return total


def effective_metrics(
    pairs: Mapping[str, Mapping[str, str]],
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    left = effective_inner(pairs, reference, reference)
    right = effective_inner(pairs, candidate, candidate)
    dot = effective_inner(pairs, reference, candidate)
    return {
        "relative_l2": math.sqrt(max(left + right - 2.0 * dot, 0.0) / max(left, 1e-24)),
        "cosine": dot / max(math.sqrt(left * right), 1e-24),
        "reference_l2": math.sqrt(max(left, 0.0)),
        "candidate_l2": math.sqrt(max(right, 0.0)),
    }


def effective_delta_metrics(
    pairs: Mapping[str, Mapping[str, str]],
    reference: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Compare a candidate effective-BA update with a target update."""

    rr = effective_inner(pairs, reference, reference)
    tt = effective_inner(pairs, target, target)
    cc = effective_inner(pairs, candidate, candidate)
    rt = effective_inner(pairs, reference, target)
    rc = effective_inner(pairs, reference, candidate)
    tc = effective_inner(pairs, target, candidate)
    target_sq = max(tt + rr - 2.0 * rt, 0.0)
    candidate_sq = max(cc + rr - 2.0 * rc, 0.0)
    delta_dot = tc - rt - rc + rr
    residual_sq = max(tt + cc - 2.0 * tc, 0.0)
    return {
        "target_delta_l2": math.sqrt(target_sq),
        "candidate_delta_l2": math.sqrt(candidate_sq),
        "candidate_over_target_delta_l2": math.sqrt(
            candidate_sq / max(target_sq, 1e-24)
        ),
        "delta_cosine": delta_dot / max(math.sqrt(target_sq * candidate_sq), 1e-24),
        "residual_over_target_delta_l2": math.sqrt(residual_sq / max(target_sq, 1e-24)),
    }


def _off_diagonal_mean(value: torch.Tensor) -> float:
    mask = ~torch.eye(value.shape[0], dtype=torch.bool)
    return float(value[mask].mean())


def _component_summary(gram: torch.Tensor) -> dict[str, float | int]:
    diagonal = gram.diag().clamp_min(0)
    total = float(diagonal.sum())
    normalized = gram / torch.sqrt(
        diagonal[:, None].clamp_min(1e-30) * diagonal[None].clamp_min(1e-30)
    )
    return {
        "active_coordinates": int((diagonal > max(total, 1e-30) * 1e-10).sum()),
        "top4_coordinate_energy_fraction": float(
            diagonal.topk(min(4, diagonal.numel())).values.sum() / max(total, 1e-30)
        ),
        "mean_absolute_offdiagonal_component_cosine": _off_diagonal_mean(
            normalized.abs()
        ),
    }


def _energy_profile(energy: torch.Tensor, *, label: str) -> dict[str, float | int]:
    energy = energy.detach().double().clamp_min(0).reshape(-1)
    total = float(energy.sum())
    probability = energy / max(total, 1e-30)
    return {
        f"active_{label}": int((energy > max(total, 1e-30) * 1e-10).sum()),
        f"effective_{label}": float(1.0 / probability.square().sum().clamp_min(1e-30)),
        f"top4_{label}_energy_fraction": float(
            probability.topk(min(4, probability.numel())).values.sum()
        ),
        f"max_{label}_energy_fraction": float(probability.max()),
        "energy": total,
    }


def adapter_geometry(
    public_rank: int,
    pairs: Mapping[str, Mapping[str, str]],
    state: Mapping[str, torch.Tensor],
    scale: float,
) -> dict[str, Any]:
    spectra = []
    by_kind: dict[str, list[dict[str, float]]] = {"q": [], "v": [], "action": []}
    if public_rank <= 0:
        raise ValueError("public LoRA rank must be positive")
    component = torch.zeros(public_rank, public_rank, dtype=torch.float64)
    component_by_kind = {name: torch.zeros_like(component) for name in by_kind}
    b_cosines: dict[str, list[float]] = {name: [] for name in by_kind}
    a_sq = b_sq = a_count = b_count = 0.0
    for module, names in pairs.items():
        a = state[names["a"]].double()
        b = state[names["b"]].double()
        right_b = torch.linalg.qr(b, mode="r").R
        right_a = torch.linalg.qr(a.T, mode="r").R
        energy = torch.linalg.svdvals(right_b @ right_a.T).square()
        probability = energy / energy.sum().clamp_min(1e-30)
        spectrum = {
            "energy": float(energy.sum()),
            "stable_rank": float(energy.sum() / energy[0].clamp_min(1e-30)),
            "top_singular_energy": float(probability[0]),
            "rank90": float(
                torch.searchsorted(
                    probability.cumsum(0), torch.tensor(0.9, dtype=probability.dtype)
                )
                + 1
            ),
            "rank99": float(
                torch.searchsorted(
                    probability.cumsum(0), torch.tensor(0.99, dtype=probability.dtype)
                )
                + 1
            ),
        }
        kind = _kind(module)
        spectra.append(spectrum)
        by_kind[kind].append(spectrum)
        value = (b.T @ b) * (a @ a.T)
        component += value
        component_by_kind[kind] += value
        columns = b / torch.linalg.vector_norm(b, dim=0, keepdim=True).clamp_min(1e-30)
        b_cosines[kind].append(_off_diagonal_mean((columns.T @ columns).abs()))
        a_sq += float(a.square().sum())
        b_sq += float(b.square().sum())
        a_count += a.numel()
        b_count += b.numel()

    total_energy = sum(row["energy"] for row in spectra)
    target_energy = torch.tensor(
        [row["energy"] for row in spectra], dtype=torch.float64
    )

    def aggregate(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
        return {
            "targets": float(len(rows)),
            "energy_fraction": float(sum(row["energy"] for row in rows) / total_energy),
            "stable_rank_mean": float(np.mean([row["stable_rank"] for row in rows])),
            "top_singular_energy_mean": float(
                np.mean([row["top_singular_energy"] for row in rows])
            ),
            "rank90_mean": float(np.mean([row["rank90"] for row in rows])),
            "rank99_mean": float(np.mean([row["rank99"] for row in rows])),
        }

    return {
        "effective_ba_energy_unscaled": total_energy,
        "effective_lora_norm_unscaled": math.sqrt(total_energy),
        "effective_lora_norm_scaled": math.sqrt(total_energy) * scale,
        "stable_rank_mean": float(np.mean([row["stable_rank"] for row in spectra])),
        "top_singular_energy_mean": float(
            np.mean([row["top_singular_energy"] for row in spectra])
        ),
        "rank90_mean": float(np.mean([row["rank90"] for row in spectra])),
        "rank99_mean": float(np.mean([row["rank99"] for row in spectra])),
        "target_energy_profile": _energy_profile(target_energy, label="targets"),
        "public_a_rms": math.sqrt(a_sq / a_count),
        "public_b_rms": math.sqrt(b_sq / b_count),
        "rank_coordinate_geometry_gauge_dependent": {
            "all": _component_summary(component),
            **{
                name: {
                    **_component_summary(component_by_kind[name]),
                    "mean_absolute_b_column_cosine": float(np.mean(b_cosines[name])),
                }
                for name in by_kind
            },
        },
        "by_kind": {name: aggregate(rows) for name, rows in by_kind.items()},
    }


def effective_variance(
    pairs: Mapping[str, Mapping[str, str]],
    states: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, float]:
    size = len(states)
    gram = np.empty((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(left, size):
            gram[left, right] = gram[right, left] = effective_inner(
                pairs, states[left], states[right]
            )
    sample = max(float(np.diag(gram).mean()), 0.0)
    mean = max(float(gram.mean()), 0.0)
    centered = max(sample - mean, 0.0)
    row_mean = gram.mean(axis=1)
    delta_energy = np.diag(gram) - 2.0 * row_mean + mean
    scale_energy = np.square(row_mean - mean) / max(mean, 1e-24)
    scale_like = float(np.maximum(scale_energy, 0.0).mean())
    orthogonal = float(np.maximum(delta_energy - scale_energy, 0.0).mean())
    return {
        "sample_energy": sample,
        "task_mean_energy": mean,
        "centered_variance_over_sample_energy": centered / max(sample, 1e-24),
        "scale_like_video_variance_fraction": scale_like / max(centered, 1e-24),
        "orthogonal_direction_video_variance_fraction": orthogonal
        / max(centered, 1e-24),
    }


def effective_update_variance(
    pairs: Mapping[str, Mapping[str, str]],
    previous: Sequence[Mapping[str, torch.Tensor]],
    current: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, float]:
    """Measure condition variance of exact effective-BA checkpoint updates."""

    if len(previous) != len(current) or not previous:
        raise ValueError("effective update variance requires paired non-empty states")
    size = len(previous)
    gram = np.empty((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(left, size):
            value = (
                effective_inner(pairs, current[left], current[right])
                - effective_inner(pairs, current[left], previous[right])
                - effective_inner(pairs, previous[left], current[right])
                + effective_inner(pairs, previous[left], previous[right])
            )
            gram[left, right] = gram[right, left] = value
    return _variance_from_gram(gram)


def tensor_variance(values: Sequence[torch.Tensor]) -> dict[str, float]:
    """Measure condition variance for ordinary tensors."""

    if not values:
        raise ValueError("tensor variance requires non-empty values")
    flattened = [value.detach().double().reshape(-1) for value in values]
    if any(value.shape != flattened[0].shape for value in flattened[1:]):
        raise ValueError("tensor variance values changed shape")
    stacked = torch.stack(flattened)
    return _variance_from_gram((stacked @ stacked.T).cpu().numpy())


def tensor_update_variance(
    previous: Sequence[torch.Tensor],
    current: Sequence[torch.Tensor],
) -> dict[str, float]:
    """Measure condition variance of paired tensor checkpoint updates."""

    if len(previous) != len(current) or not previous:
        raise ValueError("tensor update variance requires paired non-empty values")
    return tensor_variance(
        [right.detach() - left.detach() for left, right in zip(previous, current)]
    )


def _variance_from_gram(gram: np.ndarray) -> dict[str, float]:
    if gram.ndim != 2 or gram.shape[0] == 0 or gram.shape[0] != gram.shape[1]:
        raise ValueError("variance Gram must be non-empty and square")
    sample = float(np.diag(gram).mean())
    mean = float(gram.mean())
    centered = max(sample - mean, 0.0)
    return {
        "sample_energy": sample,
        "task_mean_energy": mean,
        "task_mean_energy_over_sample_energy": mean / max(sample, 1e-24),
        "centered_variance_over_sample_energy": centered / max(sample, 1e-24),
    }


def tensor_metrics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    left = reference.detach().double().reshape(-1)
    right = candidate.detach().double().reshape(-1)
    left_sq = float(left.square().sum())
    right_sq = float(right.square().sum())
    dot = float((left * right).sum())
    return {
        "relative_l2": math.sqrt(
            max(left_sq + right_sq - 2.0 * dot, 0.0) / max(left_sq, 1e-24)
        ),
        "cosine": dot / max(math.sqrt(left_sq * right_sq), 1e-24),
        "reference_l2": math.sqrt(left_sq),
        "candidate_l2": math.sqrt(right_sq),
    }


def tensor_delta_metrics(
    reference: torch.Tensor,
    target: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    """Compare a candidate tensor update with a target tensor update."""

    reference = reference.detach().double().reshape(-1)
    target_delta = target.detach().double().reshape(-1) - reference
    candidate_delta = candidate.detach().double().reshape(-1) - reference
    target_sq = float(target_delta.square().sum())
    candidate_sq = float(candidate_delta.square().sum())
    dot = float((target_delta * candidate_delta).sum())
    residual_sq = float((target_delta - candidate_delta).square().sum())
    return {
        "target_delta_l2": math.sqrt(target_sq),
        "candidate_delta_l2": math.sqrt(candidate_sq),
        "candidate_over_target_delta_l2": math.sqrt(
            candidate_sq / max(target_sq, 1e-24)
        ),
        "delta_cosine": dot / max(math.sqrt(target_sq * candidate_sq), 1e-24),
        "residual_over_target_delta_l2": math.sqrt(residual_sq / max(target_sq, 1e-24)),
    }


def distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }
