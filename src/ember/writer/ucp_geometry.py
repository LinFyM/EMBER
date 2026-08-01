"""Gauge-robust effective-BA spectrum and rank-coordinate diagnostics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
import torch

from ember.writer.model import WriterModelError


def _spectrum_row(energy: torch.Tensor) -> dict[str, float | int]:
    total = float(energy.sum())
    if total <= 0 or not bool(torch.isfinite(energy).all()):
        raise WriterModelError("effective-BA singular spectrum is degenerate")
    probability = energy / total
    cumulative = probability.cumsum(0)
    return {
        "effective_ba_energy": total,
        "effective_ba_norm": math.sqrt(total),
        "stable_rank": total / max(float(energy[0]), 1e-24),
        "entropy_effective_rank": float(torch.exp(-(
            probability * probability.clamp_min(1e-30).log()
        ).sum())),
        "top_singular_energy": float(probability[0]),
        "rank90": int(torch.searchsorted(
            cumulative, torch.tensor(.9, dtype=cumulative.dtype),
        )) + 1,
        "rank99": int(torch.searchsorted(
            cumulative, torch.tensor(.99, dtype=cumulative.dtype),
        )) + 1,
    }


def effective_ba_spectra(
    factors: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, dict[str, float | int]]:
    """Batched reduced-QR spectra grouped by real target factor shape."""

    grouped: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    for name, (a, b) in factors.items():
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise WriterModelError("invalid effective-BA factor shapes")
        grouped[(a.shape[1], b.shape[0], a.shape[0])].append(name)
    if not grouped:
        raise WriterModelError("cannot analyze empty effective-BA factors")
    result = {}
    for names in grouped.values():
        a = torch.stack([factors[name][0].double().cpu() for name in names])
        b = torch.stack([factors[name][1].double().cpu() for name in names])
        rb = torch.linalg.qr(b, mode="r").R
        ra = torch.linalg.qr(a.transpose(-2, -1), mode="r").R
        energy = torch.linalg.svdvals(rb @ ra.transpose(-2, -1)).square()
        result.update({
            name: _spectrum_row(energy[index])
            for index, name in enumerate(names)
        })
    return result


def effective_ba_spectrum(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | int]:
    """Compute one target's true ``BA`` spectrum through the batched owner."""

    return effective_ba_spectra({"target": (a, b)})["target"]


def aggregate_effective_ba_spectra(
    spectra: Sequence[Mapping[str, float | int]],
) -> dict[str, float | int]:
    """Match the historical exact50 convention: target-wise ranks, then mean."""

    if not spectra:
        raise WriterModelError("cannot aggregate empty effective-BA spectra")
    energy = float(sum(float(value["effective_ba_energy"]) for value in spectra))
    if energy <= 0:
        raise WriterModelError("effective-BA group has zero energy")
    return {
        "targets": len(spectra),
        "effective_ba_energy": energy,
        "effective_lora_norm": math.sqrt(energy),
        "stable_rank_mean": float(np.mean([
            float(value["stable_rank"]) for value in spectra
        ])),
        "entropy_effective_rank_mean": float(np.mean([
            float(value["entropy_effective_rank"]) for value in spectra
        ])),
        "top_singular_energy_mean": float(np.mean([
            float(value["top_singular_energy"]) for value in spectra
        ])),
        "rank90_mean": float(np.mean([float(value["rank90"]) for value in spectra])),
        "rank99_mean": float(np.mean([float(value["rank99"]) for value in spectra])),
    }


def component_coordinate_geometry(
    gram: torch.Tensor,
    b_gram: torch.Tensor,
    layer_energy: Sequence[float],
) -> dict[str, object]:
    """Describe rank-coordinate components; never interpret this Gram as BA rank."""

    rank = gram.shape[0]
    layers = np.asarray(layer_energy, dtype=np.float64)
    diagonal = gram.diag().clamp_min(0)
    if (
        gram.shape != (rank, rank) or b_gram.shape != gram.shape
        or layers.size == 0 or not np.isfinite(layers).all()
        or float(diagonal.sum()) <= 0
    ):
        raise WriterModelError("UCP rank-coordinate component geometry is degenerate")
    component = gram / torch.sqrt(
        diagonal[:, None] * diagonal[None]
    ).clamp_min(1e-24)
    b_diagonal = b_gram.diag().clamp_min(0)
    b_cosine = b_gram / torch.sqrt(
        b_diagonal[:, None] * b_diagonal[None]
    ).clamp_min(1e-24)
    upper = torch.triu(torch.ones(rank, rank, dtype=torch.bool), diagonal=1)
    participation = diagonal / diagonal.sum()
    return {
        "coordinate_energy_participation": participation.tolist(),
        "active_coordinates_1e6": int((participation > 1e-6).sum()),
        "component_pair_cosine_mean": float(component[upper].mean()),
        "component_negative_pair_fraction": float(
            (component[upper] < 0).float().mean()
        ),
        "b_column_cosine_mean": float(b_cosine[upper].mean()),
        "b_column_negative_fraction": float(
            (b_cosine[upper] < 0).float().mean()
        ),
        "layer_energy_cv": float(layers.std() / max(layers.mean(), 1e-24)),
    }
