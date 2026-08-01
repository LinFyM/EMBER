"""Gauge-robust effective-LoRA Gram and spectrum summaries."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from ember.writer.model import WriterModelError


def gram_geometry(
    gram: torch.Tensor,
    b_gram: torch.Tensor,
    layer_energy: Sequence[float],
) -> dict[str, object]:
    rank = gram.shape[0]
    eigen = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    spectral_total, top = float(eigen.sum()), float(eigen[0])
    layers = np.asarray(layer_energy, dtype=np.float64)
    functional_total = float(layers.sum())
    if (
        spectral_total <= 0 or top <= 0 or functional_total <= 0
        or layers.size == 0 or not np.isfinite(layers).all()
    ):
        raise WriterModelError("UCP effective LoRA geometry is degenerate")
    probability = eigen / spectral_total
    cumulative = probability.cumsum(0)
    diagonal = gram.diag().clamp_min(0)
    component = gram / torch.sqrt(
        diagonal[:, None] * diagonal[None]
    ).clamp_min(1e-24)
    b_diagonal = b_gram.diag().clamp_min(0)
    b_cosine = b_gram / torch.sqrt(
        b_diagonal[:, None] * b_diagonal[None]
    ).clamp_min(1e-24)
    upper = torch.triu(torch.ones(rank, rank, dtype=torch.bool), diagonal=1)
    participation = diagonal / diagonal.sum().clamp_min(1e-24)
    return {
        "effective_lora_norm": math.sqrt(functional_total),
        "stable_rank": spectral_total / top,
        "entropy_effective_rank": float(torch.exp(-(
            probability * probability.clamp_min(1e-30).log()
        ).sum())),
        "top_singular_energy": top / spectral_total,
        "rank90": int(torch.searchsorted(
            cumulative, torch.tensor(.9, dtype=cumulative.dtype),
        )) + 1,
        "rank99": int(torch.searchsorted(
            cumulative, torch.tensor(.99, dtype=cumulative.dtype),
        )) + 1,
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
