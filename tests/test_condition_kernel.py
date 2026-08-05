from __future__ import annotations

import torch

from ember.writer.condition_kernel import (
    FactorizedConditionFeature,
    ProgramValueMemory,
    apply_program_value_delta,
    kernel_corrected_value_delta,
)
from ember.writer.video_program import temporal_video_descriptor


def test_factorized_condition_feature_is_fixed_normalized_product() -> None:
    module = FactorizedConditionFeature(
        task_center=torch.tensor([0.5, -0.5]),
        task_frequencies=torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
        video_frequencies=torch.tensor([[1.0, -1.0], [0.5, 0.5]]),
    )
    task = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    video = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    first = module(task, video)
    second = module(task, video)
    assert first.shape == (2, 16)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(
        first.square().sum(dim=-1), torch.ones(2), rtol=1e-6, atol=1e-6
    )
    assert not any(parameter.requires_grad for parameter in module.parameters())


def test_program_value_memory_read_and_update_are_exact() -> None:
    memory = ProgramValueMemory(
        feature_width=3,
        program_slots=2,
        program_width=4,
        initialization_seed=7,
    )
    features = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]])
    before = memory(features).detach().clone()
    cotangents = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4) / 10
    delta, summary = kernel_corrected_value_delta(
        features,
        cotangents,
        step_size=0.2,
    )
    predicted = torch.einsum("tf,fsw->tsw", features, delta)
    gram = features.double() @ features.double().T
    expected = -0.2 * (
        gram
        @ torch.linalg.solve(
            gram + 0.01 * torch.eye(2, dtype=torch.float64),
            cotangents.double().flatten(1),
        )
    ).reshape_as(cotangents)
    torch.testing.assert_close(predicted, expected.float(), rtol=2e-5, atol=2e-6)
    observed, application = apply_program_value_delta(memory, delta, features)
    torch.testing.assert_close(memory(features) - before, predicted)
    torch.testing.assert_close(observed, predicted)
    assert application.predicted_observed_max_abs < 1e-6
    assert application.predicted_observed_relative_rms < 1e-5
    assert summary.feature_rank == 2
    assert summary.applied_scale == 1.0


def test_kernel_update_uses_one_global_trust_scale() -> None:
    features = torch.eye(2)
    cotangents = torch.ones(2, 2, 2)
    uncapped, _ = kernel_corrected_value_delta(
        features, cotangents, step_size=1.0
    )
    capped, summary = kernel_corrected_value_delta(
        features,
        cotangents,
        step_size=1.0,
        induced_update_rms_cap=0.05,
    )
    ratio = capped / uncapped
    torch.testing.assert_close(ratio, torch.full_like(ratio, summary.applied_scale))
    assert summary.predicted_update_rms <= 0.0500001


def test_temporal_video_descriptor_is_order_sensitive_and_normalized() -> None:
    frames = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5], [0.5, -1.0]]
    )
    offsets = torch.tensor([0, 4], dtype=torch.long)
    normal = temporal_video_descriptor(frames, offsets)
    reversed_value = temporal_video_descriptor(frames.flip(0), offsets)
    shuffled = temporal_video_descriptor(frames[[0, 2, 1, 3]], offsets)
    assert normal.shape == (1, 8)
    torch.testing.assert_close(normal.square().sum(), torch.tensor(1.0))
    assert not torch.equal(normal, reversed_value)
    assert not torch.equal(normal, shuffled)
