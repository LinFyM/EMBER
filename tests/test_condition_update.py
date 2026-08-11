from __future__ import annotations

import pytest
import torch

from ember.writer.condition_update import (
    ConditionUpdateError,
    PolicyInnovationGoalCausalConditionFeature,
    ProgramResidualMemory,
    apply_program_residual_delta_with_evidence_,
    shared_reward_tangent_program_projection,
    success_key_nullspace_program_delta,
)


def test_policy_innovation_feature_is_zero_preserving_and_reads_real_order() -> None:
    encoder = PolicyInnovationGoalCausalConditionFeature(
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


def test_goal_causal_blocks_encode_terminal_role_and_internal_order() -> None:
    encoder = PolicyInnovationGoalCausalConditionFeature(
        innovation_width=3,
        feature_width=6,
        initialization_seed=17,
    )
    encoder.projection.copy_(torch.eye(3).repeat(2, 1, 1))
    innovations = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [-1.0, 1.0, 2.0],
        ]
    )[None]
    natural = encoder(innovations)
    whole = innovations.mean(dim=1)
    goal = innovations[:, -1] - whole
    centered = innovations - whole[:, None]
    scale = torch.arange(1, 5, dtype=torch.float32).sqrt()
    causal = (centered.cumsum(dim=1) / scale[None, :, None]).mean(dim=1)
    expected = torch.cat(
        (torch.nn.functional.normalize(goal), torch.nn.functional.normalize(causal)),
        dim=1,
    ) / 2**0.5
    torch.testing.assert_close(natural, expected, rtol=1e-6, atol=1e-6)

    reversed_feature = encoder(innovations.flip(1))
    reverse_goal = innovations[:, 0] - whole
    reverse_centered = innovations.flip(1) - whole[:, None]
    reverse_causal = (
        reverse_centered.cumsum(dim=1) / scale[None, :, None]
    ).mean(dim=1)
    reverse_expected = torch.cat(
        (
            torch.nn.functional.normalize(reverse_goal),
            torch.nn.functional.normalize(reverse_causal),
        ),
        dim=1,
    ) / 2**0.5
    torch.testing.assert_close(reversed_feature, reverse_expected, rtol=1e-6, atol=1e-6)
    assert not torch.equal(natural, reversed_feature)


def _solve(
    correct: torch.Tensor,
    negative: torch.Tensor,
    cotangents: torch.Tensor,
    *,
    anchors: torch.Tensor | None = None,
    protected: torch.Tensor | None = None,
    step_size: float = 1.0,
):
    return success_key_nullspace_program_delta(
        correct,
        negative,
        cotangents,
        (
            anchors
            if anchors is not None
            else correct.new_empty((0, correct.shape[1]))
        ),
        (
            protected
            if protected is not None
            else torch.zeros(correct.shape[0], dtype=torch.bool)
        ),
        step_size=step_size,
        relative_damping=0.01,
    )


def test_zero_anchor_full48_update_preserves_negative_rows_and_closes_memory() -> None:
    correct = torch.eye(4, dtype=torch.float32)[:2]
    negative = torch.eye(4, dtype=torch.float32)[2:]
    cotangents = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
    delta, summary = _solve(correct, negative, cotangents)
    full = torch.cat((correct, negative))
    predicted = (full @ delta.flatten(1)).reshape(4, 3, 4)
    torch.testing.assert_close(predicted[:2], -cotangents / 1.01, rtol=2e-5, atol=2e-6)
    assert torch.equal(predicted[2:], torch.zeros_like(predicted[2:]))
    assert summary.original_feature_rank == 4
    assert summary.projected_feature_rank == 4
    assert summary.anchor_rank == 0
    assert summary.predicted_negative_to_unprotected_ratio == 0
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


def test_zero_anchor_update_matches_explicit_nonorthogonal_ridge() -> None:
    generator = torch.Generator().manual_seed(29)
    correct = torch.randn(3, 7, generator=generator)
    negative = torch.randn(3, 7, generator=generator)
    cotangent = torch.randn(3, 2, 4, generator=generator)
    delta, summary = _solve(
        correct, negative, cotangent, step_size=0.7
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
    assert summary.active_regularized_gram_condition_number >= 1


def test_success_keys_hard_constrain_shared_write_without_masking_objective() -> None:
    correct = torch.eye(4, dtype=torch.float32)[:2]
    negative = torch.eye(4, dtype=torch.float32)[2:]
    cotangent = torch.arange(1, 17, dtype=torch.float32).reshape(2, 2, 4)
    anchors = correct[:1].clone()
    protected = torch.tensor([True, False])
    delta, summary = _solve(
        correct,
        negative,
        cotangent,
        anchors=anchors,
        protected=protected,
    )
    motion = torch.cat((correct, negative)) @ delta.flatten(1)
    torch.testing.assert_close(
        anchors @ delta.flatten(1),
        torch.zeros(1, 8),
        rtol=0,
        atol=1e-7,
    )
    assert torch.count_nonzero(motion[0]) == 0
    assert torch.count_nonzero(motion[1]) > 0
    assert torch.count_nonzero(motion[2:]) == 0
    assert summary.current_protected_conditions == 1
    assert summary.anchor_rank == 1
    assert summary.original_feature_rank == 4
    assert summary.projected_feature_rank == 3


def test_duplicate_and_permuted_success_keys_define_the_same_nullspace() -> None:
    generator = torch.Generator().manual_seed(91)
    correct = torch.randn(4, 9, generator=generator)
    negative = torch.randn(4, 9, generator=generator)
    cotangent = torch.randn(4, 2, 3, generator=generator)
    anchors = correct[[0, 2]]
    protected = torch.tensor([True, False, True, False])
    expected, _ = _solve(
        correct,
        negative,
        cotangent,
        anchors=anchors,
        protected=protected,
    )
    observed, summary = _solve(
        correct,
        negative,
        cotangent,
        anchors=anchors[[1, 0, 1]],
        protected=protected,
    )
    torch.testing.assert_close(observed, expected, rtol=3e-5, atol=3e-6)
    assert summary.anchor_constraint_rows == 3
    assert summary.anchor_rank == 2


def test_all_condition_directions_constrained_produces_finite_zero_write() -> None:
    correct = torch.eye(3)
    negative = torch.eye(3).roll(1, dims=0)
    cotangent = torch.randn(3, 2, 2, generator=torch.Generator().manual_seed(5))
    delta, summary = _solve(
        correct,
        negative,
        cotangent,
        anchors=torch.eye(3),
        protected=torch.ones(3, dtype=torch.bool),
    )
    assert torch.equal(delta, torch.zeros_like(delta))
    assert summary.projected_feature_rank == 0
    assert summary.anchor_rank == 3


def test_rank_deficient_features_and_zero_credit_remain_finite() -> None:
    correct = torch.ones(2, 3)
    negative = torch.ones(2, 3)
    cotangent = torch.zeros(2, 2, 2)
    delta, summary = _solve(correct, negative, cotangent)
    assert torch.isfinite(delta).all()
    assert summary.original_feature_rank == 1
    assert summary.predicted_negative_to_unprotected_ratio == 0
    assert torch.count_nonzero(delta) == 0


def test_reward_projection_is_exact_identity_without_constraint_or_violation() -> None:
    generator = torch.Generator().manual_seed(701)
    delta = torch.randn(5, 2, 3, generator=generator)
    features = torch.randn(4, 5, generator=generator)
    reward = torch.zeros(4, 2, 3)
    projected, summary = shared_reward_tangent_program_projection(
        delta,
        features,
        reward,
        torch.zeros(4, dtype=torch.bool),
        features.new_empty((0, 5)),
    )
    assert projected is delta
    assert torch.equal(projected, delta)
    assert summary.mixed_constraints == 0
    assert summary.projection_changed is False

    reward[0] = -(features[0] @ delta.flatten(1)).reshape(2, 3)
    projected, summary = shared_reward_tangent_program_projection(
        delta,
        features,
        reward,
        torch.tensor([True, False, False, False]),
        features.new_empty((0, 5)),
    )
    assert projected is delta
    assert summary.raw_violation_count == 0


def test_reward_projection_satisfies_correlated_halfspaces_and_anchors() -> None:
    delta = torch.zeros(4, 2, 2)
    delta[1, 0, 0] = 2.0
    delta[2, 0, 1] = 1.0
    features = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 2.0, 2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    reward = torch.zeros(4, 2, 2)
    reward[0, 0, 0] = 1.0
    reward[1, 0] = torch.tensor([1.0, 1.0])
    reward[2] = reward[1]
    reward[3, 0, 1] = -1.0
    projected, summary = shared_reward_tangent_program_projection(
        delta,
        features,
        reward,
        torch.ones(4, dtype=torch.bool),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    constraints = (
        reward * (features @ projected.flatten(1)).reshape(4, 2, 2)
    ).flatten(1).sum(dim=1)
    assert summary.raw_violation_count >= 2
    assert summary.final_violation_count == 0
    assert bool((constraints <= 2e-5).all())
    torch.testing.assert_close(
        projected[0], torch.zeros_like(projected[0]), rtol=0, atol=1e-7
    )
    assert summary.projection_changed is True
    assert summary.blind_projected_inner_product > 0
    assert summary.blind_projected_cosine > 0


def test_reward_projection_is_permutation_and_duplicate_invariant() -> None:
    generator = torch.Generator().manual_seed(911)
    delta = torch.randn(6, 2, 3, generator=generator)
    features = torch.randn(3, 6, generator=generator)
    reward = torch.randn(3, 2, 3, generator=generator)
    motion = (features @ delta.flatten(1)).reshape(3, 2, 3)
    reward.copy_(motion)
    mask = torch.ones(3, dtype=torch.bool)
    expected, _ = shared_reward_tangent_program_projection(
        delta, features, reward, mask, features.new_empty((0, 6))
    )
    order = torch.tensor([2, 0, 1, 1])
    observed, summary = shared_reward_tangent_program_projection(
        delta,
        features.index_select(0, order),
        reward.index_select(0, order),
        torch.ones(4, dtype=torch.bool),
        features.new_empty((0, 6)),
    )
    torch.testing.assert_close(observed, expected, rtol=3e-5, atol=3e-6)
    assert summary.final_violation_count == 0


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
        success_key_nullspace_program_delta(
            correct,
            negative,
            cotangent,
            correct.new_empty((0, correct.shape[1])),
            torch.zeros(correct.shape[0], dtype=torch.bool),
            step_size=1.0,
            relative_damping=0.01,
        )
