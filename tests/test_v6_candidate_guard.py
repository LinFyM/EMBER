from __future__ import annotations

import torch

from ember.expert_manifold.v6_candidate_guard import (
    classify_paired_candidate_outcomes,
    response_preserving_candidate_guard_correction,
)


def test_paired_classification_uses_statewise_losses_and_gains() -> None:
    base = torch.zeros(24, 2, dtype=torch.bool)
    candidate = torch.zeros_like(base)
    base[0] = torch.tensor([True, True])
    candidate[0] = torch.tensor([True, False])
    candidate[1] = torch.tensor([True, False])
    base[2] = True
    candidate[2] = True

    result = classify_paired_candidate_outcomes(base, candidate)
    assert result.harmful_mask.nonzero().flatten().tolist() == [0]
    assert result.beneficial_mask.nonzero().flatten().tolist() == [1]
    assert result.stable_success_mask.nonzero().flatten().tolist() == [2]
    assert result.summary.losses == 1
    assert result.summary.gains == 1
    assert result.summary.discordant_states == 2
    assert result.summary.harmful_task_ordinals == (0,)
    assert result.summary.beneficial_task_ordinals == (1,)
    assert result.summary.stable_success_task_ordinals == (2,)
    assert result.summary.indifferent_task_count == 22


def _projection_inputs(seed: int = 17):
    generator = torch.Generator().manual_seed(seed)
    correct = torch.randn(24, 64, generator=generator)
    negative = torch.randn(24, 64, generator=generator)
    equivariance = torch.randn(24, 64, generator=generator)
    blind = torch.randn(64, 3, 4, generator=generator)
    return blind, correct, negative, equivariance, torch.cat((correct, negative))


def test_no_guard_projection_is_elementwise_identity() -> None:
    blind, correct, negative, equivariance, full = _projection_inputs()
    empty = correct.new_empty((0, correct.shape[1]))
    mask = torch.zeros(24, dtype=torch.bool)
    responses = torch.cat((negative, equivariance))
    projected, summary = response_preserving_candidate_guard_correction(
        blind, empty, correct, responses, mask, mask, full, negative_rows=24
    )
    assert projected is blind
    assert torch.equal(projected, blind)
    assert summary.total_guard_rows == 0
    assert summary.projection_changed is False
    assert summary.projected_to_blind_energy_ratio == 1
    assert summary.blind_projected_cosine == 1


def test_current_and_persisted_guards_close_with_positive_retained_direction() -> None:
    blind, correct, negative, equivariance, full = _projection_inputs()
    stable = torch.zeros(24, dtype=torch.bool)
    harmful = torch.zeros(24, dtype=torch.bool)
    stable[1] = True
    harmful[2] = True
    persisted = correct[[0]]
    responses = torch.cat((negative, equivariance))
    projected, summary = response_preserving_candidate_guard_correction(
        blind, persisted, correct, responses, stable, harmful, full, negative_rows=24
    )
    guards = torch.cat((persisted, correct[[1, 2]]))
    motion = guards @ projected.flatten(1)
    unprotected_motion = correct[3:] @ projected.flatten(1)
    closure_ratio = (
        motion.square().mean().sqrt()
        / unprotected_motion.square().mean().sqrt()
    )
    assert closure_ratio < 1e-5
    torch.testing.assert_close(
        negative @ projected.flatten(1),
        negative @ blind.flatten(1),
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        equivariance @ projected.flatten(1),
        equivariance @ blind.flatten(1),
        rtol=2e-5,
        atol=2e-5,
    )
    assert summary.persisted_guard_rows == 1
    assert summary.current_stable_guard_rows == 1
    assert summary.current_harmful_guard_rows == 1
    assert summary.total_guard_rows == 3
    assert summary.guard_rank == 3
    assert summary.negative_rows == 24
    assert summary.negative_rank == 24
    assert summary.restricted_guard_rank == 3
    assert summary.final_guard_violation_count == 0
    assert summary.negative_preservation_violation_count == 0
    assert summary.equivariance_rows == 24
    assert summary.equivariance_rank == 24
    assert summary.equivariance_preservation_violation_count == 0
    assert summary.projection_changed
    assert summary.projected_to_blind_energy_ratio > 0
    assert summary.blind_projected_inner_product > 0
    assert summary.blind_projected_cosine > 0


def test_duplicate_permuted_persisted_rows_define_same_final_projection() -> None:
    blind, correct, negative, equivariance, full = _projection_inputs(29)
    responses = torch.cat((negative, equivariance))
    stable = torch.zeros(24, dtype=torch.bool)
    harmful = torch.zeros(24, dtype=torch.bool)
    harmful[[3, 7]] = True
    expected, _ = response_preserving_candidate_guard_correction(
        blind, correct[[0, 1]], correct, responses, stable, harmful, full, negative_rows=24
    )
    observed, summary = response_preserving_candidate_guard_correction(
        blind, correct[[1, 0, 1]], correct, responses, stable, harmful, full, negative_rows=24
    )
    torch.testing.assert_close(observed, expected, rtol=3e-5, atol=3e-6)
    assert summary.total_guard_rows == 5
    assert summary.guard_rank == 4
