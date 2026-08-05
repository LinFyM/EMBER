from __future__ import annotations

import math

import torch

from ember.writer.program_credit_analysis_metrics import (
    effective_difference_family,
    tensor_difference_family,
    tensor_variance,
)


def test_tensor_video_and_update_families_separate_mean_from_variance() -> None:
    videos = [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])]
    variance = tensor_variance(videos)
    assert variance["centered_variance_over_sample_energy"] == 0.0

    baseline = {"a": torch.zeros(2), "b": torch.zeros(2)}
    candidate = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([1.0, 0.0])}
    summary, mean = tensor_difference_family(
        baseline, candidate, ("a", "b")
    )
    assert summary["mean_delta_energy_fraction"] == 1.0
    assert summary["pair_delta_cosine_median"] == 1.0
    assert torch.equal(mean, torch.tensor([1.0, 0.0], dtype=torch.float64))


def test_effective_ba_update_family_uses_product_not_raw_factor_delta() -> None:
    pairs = {"target": {"a": "a", "b": "b"}}
    baseline = {
        "v0": {"a": torch.eye(2), "b": torch.zeros(2, 2)},
        "v1": {"a": torch.eye(2), "b": torch.zeros(2, 2)},
    }
    candidate = {
        "v0": {"a": torch.eye(2), "b": torch.eye(2)},
        "v1": {"a": -torch.eye(2), "b": -torch.eye(2)},
    }
    summary = effective_difference_family(
        pairs, baseline, candidate, ("v0", "v1")
    )
    assert math.isclose(summary["mean_delta_energy_fraction"], 1.0)
    assert math.isclose(summary["pair_delta_cosine_median"], 1.0)
    assert summary["negative_pair_fraction"] == 0.0
