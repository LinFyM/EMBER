from __future__ import annotations

import pytest
import torch

from ember.writer.condition_update import (
    ConditionUpdateError,
    PolicyInnovationCausalConditionFeature,
    ProgramResidualMemory,
    apply_program_residual_delta_with_evidence_,
    counterfactual_null_program_delta,
)


def test_policy_innovation_feature_is_zero_preserving_and_reads_real_order() -> None:
    encoder = PolicyInnovationCausalConditionFeature(
        innovation_width=3,
        feature_width=6,
        initialization_seed=17,
    )
    zero = torch.zeros(1, 4, 3)
    assert torch.equal(encoder(zero), torch.zeros(1, 6))

    innovation = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.5, 0.0],
        ]
    )[None]
    natural = encoder(innovation)
    reverse_order = torch.tensor([3, 2, 1, 0], dtype=torch.long)
    shuffle_order = torch.tensor([0, 2, 1, 3], dtype=torch.long)
    reversed_feature = encoder(innovation, phase_order=reverse_order)
    shuffled_feature = encoder(innovation, phase_order=shuffle_order)
    torch.testing.assert_close(
        natural.square().sum(dim=1), torch.ones(1), rtol=1e-6, atol=1e-6
    )
    assert not torch.equal(natural, reversed_feature)
    assert not torch.equal(natural, shuffled_feature)
    torch.testing.assert_close(reversed_feature, encoder(innovation.flip(1)))
    torch.testing.assert_close(shuffled_feature, encoder(innovation[:, [0, 2, 1, 3]]))
    assert not tuple(encoder.parameters())
    assert not encoder.state_dict()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert encoder(innovation).dtype == torch.float32


def test_balanced_static_and_causal_blocks_break_reverse_collinearity() -> None:
    encoder = PolicyInnovationCausalConditionFeature(
        innovation_width=3,
        feature_width=6,
        initialization_seed=17,
    )
    static = torch.tensor([2.0, -1.0, 0.5])
    dynamic = torch.tensor([0.25, 1.5, -0.75])
    innovations = torch.stack((static + dynamic, static - dynamic))[None]
    natural = encoder(innovations)
    reversed_feature = encoder(
        innovations,
        phase_order=torch.tensor([1, 0], dtype=torch.long),
    )
    torch.testing.assert_close(
        (natural * reversed_feature).sum(),
        torch.zeros(()),
        rtol=0,
        atol=2e-6,
    )


def test_blind_full48_update_preserves_negative_rows_and_closes_memory() -> None:
    correct = torch.eye(4, dtype=torch.float32)[:2]
    negative = torch.eye(4, dtype=torch.float32)[2:]
    cotangents = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
    delta, summary = counterfactual_null_program_delta(
        correct,
        negative,
        cotangents,
        step_size=1.0,
        relative_damping=0.01,
    )
    full = torch.cat((correct, negative))
    predicted = (full @ delta.flatten(1)).reshape(4, 3, 4)
    torch.testing.assert_close(predicted[:2], -cotangents / 1.01, rtol=2e-5, atol=2e-6)
    assert torch.equal(predicted[2:], torch.zeros_like(predicted[2:]))
    assert summary.feature_rank == 4
    assert summary.predicted_negative_to_correct_ratio == 0
    assert delta.dtype == torch.float32

    memory = ProgramResidualMemory(
        feature_width=4,
        program_slots=3,
        program_width=4,
    )
    evidence = apply_program_residual_delta_with_evidence_(memory, delta, full)
    torch.testing.assert_close(memory(full), predicted)
    assert evidence.predicted_observed_max_abs == 0
    assert evidence.predicted_observed_relative_rms == 0


def test_blind_update_matches_explicit_nonorthogonal_ridge() -> None:
    generator = torch.Generator().manual_seed(29)
    correct = torch.randn(3, 7, generator=generator)
    negative = torch.randn(3, 7, generator=generator)
    cotangent = torch.randn(3, 2, 4, generator=generator)
    delta, summary = counterfactual_null_program_delta(
        correct,
        negative,
        cotangent,
        step_size=0.7,
        relative_damping=0.01,
    )
    features = torch.cat((correct, negative)).to(torch.float64)
    gram = features @ features.T
    damping = 0.01 * gram.diagonal().mean()
    right = torch.cat(
        (-0.7 * cotangent.flatten(1), torch.zeros_like(cotangent).flatten(1))
    ).to(torch.float64)
    expected = features.T @ torch.linalg.solve(
        gram + damping * torch.eye(6, dtype=torch.float64), right
    )
    torch.testing.assert_close(
        delta.flatten(1).to(torch.float64), expected, rtol=2e-5, atol=2e-6
    )
    assert summary.regularized_gram_condition_number >= 1


def test_rank_deficient_features_and_zero_credit_remain_finite() -> None:
    correct = torch.ones(2, 3)
    negative = torch.ones(2, 3)
    cotangent = torch.zeros(2, 2, 2)
    delta, summary = counterfactual_null_program_delta(
        correct,
        negative,
        cotangent,
        step_size=1.0,
        relative_damping=0.01,
    )
    assert torch.isfinite(delta).all()
    assert summary.feature_rank == 1
    assert summary.predicted_negative_to_correct_ratio > 1e30
    assert torch.count_nonzero(delta) == 0


@pytest.mark.parametrize(
    "correct,negative,cotangent",
    (
        (torch.empty(0, 3), torch.empty(0, 3), torch.empty(0, 2, 2)),
        (torch.ones(2, 3), torch.ones(1, 3), torch.ones(2, 2, 2)),
        (
            torch.full((2, 3), torch.nan),
            torch.ones(2, 3),
            torch.ones(2, 2, 2),
        ),
    ),
)
def test_blind_update_rejects_invalid_batches(
    correct: torch.Tensor,
    negative: torch.Tensor,
    cotangent: torch.Tensor,
) -> None:
    with pytest.raises(ConditionUpdateError):
        counterfactual_null_program_delta(
            correct,
            negative,
            cotangent,
            step_size=1.0,
            relative_damping=0.01,
        )
