from __future__ import annotations

import pytest
import torch

from ember.writer.condition_update import (
    ConditionUpdateError,
    PolicyInnovationGoalCausalConditionFeature,
    ProgramResidualMemory,
    apply_program_residual_delta_with_evidence_,
    paired_video_joint_program_delta,
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


def test_magnitude_gated_causal_interaction_encodes_goal_and_order() -> None:
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
    goal_block = torch.nn.functional.normalize(goal)
    causal_block = torch.nn.functional.normalize(causal)
    odd = torch.nn.functional.normalize(goal_block.abs() * causal_block)
    even = torch.nn.functional.normalize(goal_block * causal_block)
    expected = torch.nn.functional.normalize(
        torch.cat((odd, even), dim=1)
    )
    torch.testing.assert_close(natural, expected, rtol=1e-6, atol=1e-6)

    reversed_feature = encoder(innovations.flip(1))
    reverse_goal = innovations[:, 0] - whole
    reverse_centered = innovations.flip(1) - whole[:, None]
    reverse_causal = (
        reverse_centered.cumsum(dim=1) / scale[None, :, None]
    ).mean(dim=1)
    reverse_goal_block = torch.nn.functional.normalize(reverse_goal)
    reverse_causal_block = torch.nn.functional.normalize(reverse_causal)
    reverse_odd = torch.nn.functional.normalize(
        reverse_goal_block.abs() * reverse_causal_block
    )
    reverse_even = torch.nn.functional.normalize(
        reverse_goal_block * reverse_causal_block
    )
    reverse_expected = torch.nn.functional.normalize(
        torch.cat((reverse_odd, reverse_even), dim=1)
    )
    torch.testing.assert_close(reversed_feature, reverse_expected, rtol=1e-6, atol=1e-6)
    assert not torch.equal(natural, reversed_feature)


def test_magnitude_gated_causal_interaction_separates_joint_sign_reversal() -> None:
    encoder = PolicyInnovationGoalCausalConditionFeature(
        innovation_width=2,
        feature_width=4,
        initialization_seed=19,
    )
    encoder.projection.copy_(torch.eye(2).repeat(2, 1, 1))
    first = torch.tensor([1.0, 2.0])
    second = torch.tensor([2.0, -1.0])
    innovations = torch.stack((first, second, -second, -first))[None]
    correct = encoder(innovations)
    reversed_feature = encoder(innovations.flip(1))
    half = correct.shape[1] // 2
    torch.testing.assert_close(
        reversed_feature[:, :half], -correct[:, :half], rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        reversed_feature[:, half:], correct[:, half:], rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        (correct * reversed_feature).sum(dim=1),
        torch.zeros(1),
        rtol=0,
        atol=1e-6,
    )


def test_magnitude_gated_blocks_have_equal_pre_normalization_norms() -> None:
    goal = torch.tensor([[1.0, -2.0, 3.0]])
    causal = torch.tensor([[-4.0, 5.0, -6.0]])
    odd = goal.abs() * causal
    even = goal * causal
    torch.testing.assert_close(
        torch.linalg.vector_norm(odd, dim=1),
        torch.linalg.vector_norm(even, dim=1),
        rtol=0,
        atol=0,
    )


def test_magnitude_gated_causal_interaction_requires_both_descriptors() -> None:
    encoder = PolicyInnovationGoalCausalConditionFeature(
        innovation_width=2,
        feature_width=4,
        initialization_seed=23,
    )
    encoder.projection.copy_(torch.eye(2).repeat(2, 1, 1))
    constant = torch.tensor([[[1.0, -2.0]]]).repeat(1, 4, 1)
    assert torch.equal(encoder(constant), torch.zeros(1, 4))

    # The terminal quartile equals the whole-video mean, while the ordered
    # prefix statistic remains nonzero: a zero goal must still zero both blocks.
    goal_zero = torch.tensor(
        [[[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]
    )
    assert torch.equal(encoder(goal_zero), torch.zeros(1, 4))


def _paired_video_solve(
    correct: torch.Tensor,
    negative: torch.Tensor,
    cotangent: torch.Tensor,
    *,
    task_count: int,
):
    return paired_video_joint_program_delta(
        correct,
        negative,
        cotangent,
        task_count=task_count,
        view_weights=torch.full(
            (correct.shape[0],), 0.5, dtype=torch.float32, device=correct.device
        ),
        step_size=1.0,
        relative_damping=0.01,
    )


def test_paired_video_duplicate_views_degenerate_to_single_view_solve() -> None:
    generator = torch.Generator().manual_seed(307)
    tasks = 4
    correct = torch.randn(tasks, 11, generator=generator)
    negative = torch.randn(tasks, 11, generator=generator)
    cotangent = torch.randn(tasks, 2, 3, generator=generator)
    features64 = torch.cat((correct, negative)).to(dtype=torch.float64)
    gram64 = features64 @ features64.T
    damping64 = 0.01 * gram64.diagonal().mean()
    rhs64 = torch.cat(
        (cotangent.flatten(1), torch.zeros_like(cotangent).flatten(1))
    ).to(dtype=torch.float64)
    single = -features64.T @ torch.linalg.solve(
        gram64 + damping64 * torch.eye(2 * tasks, dtype=torch.float64),
        rhs64,
    )
    paired, summary, motion = _paired_video_solve(
        torch.cat((correct, correct)),
        torch.cat((negative, negative)),
        torch.cat((cotangent, cotangent)),
        task_count=tasks,
    )
    torch.testing.assert_close(
        paired.flatten(1).to(dtype=torch.float64),
        single,
        rtol=2e-5,
        atol=2e-6,
    )
    assert summary.task_count == tasks
    assert summary.views_per_task == 2
    assert summary.row_count == 4 * tasks

    memory = ProgramResidualMemory(
        feature_width=correct.shape[1],
        program_slots=cotangent.shape[1],
        program_width=cotangent.shape[2],
    )
    all_rows = torch.cat((correct, correct, negative, negative))
    torch.testing.assert_close(
        motion,
        (all_rows @ paired.flatten(1)).reshape(4 * tasks, 2, 3),
    )
    evidence = apply_program_residual_delta_with_evidence_(memory, paired, all_rows)
    assert evidence.predicted_observed_max_abs == 0
    assert evidence.predicted_observed_relative_rms == 0


def test_paired_video_joint_solve_is_invariant_to_per_task_view_swaps() -> None:
    generator = torch.Generator().manual_seed(311)
    tasks = 5
    correct = torch.randn(2 * tasks, 13, generator=generator)
    negative = torch.randn(2 * tasks, 13, generator=generator)
    cotangent = torch.randn(2 * tasks, 2, 4, generator=generator)
    expected, _, _ = _paired_video_solve(
        correct, negative, cotangent, task_count=tasks
    )
    permutation = torch.arange(2 * tasks)
    for task in (0, 2, 4):
        permutation[task], permutation[tasks + task] = (
            permutation[tasks + task].clone(),
            permutation[task].clone(),
        )
    observed, _, _ = _paired_video_solve(
        correct[permutation],
        negative[permutation],
        cotangent[permutation],
        task_count=tasks,
    )
    torch.testing.assert_close(observed, expected, rtol=2e-5, atol=2e-6)


def test_paired_video_joint_solve_descends_both_views_and_limits_negatives() -> None:
    tasks = 3
    primary = torch.eye(6)[:tasks]
    companion = torch.eye(6)[tasks:]
    correct = torch.cat((primary, companion))
    negative = torch.zeros_like(correct)
    cotangent = torch.arange(1, 1 + 6 * 4, dtype=torch.float32).reshape(6, 2, 2)
    delta, summary, _ = _paired_video_solve(
        correct, negative, cotangent, task_count=tasks
    )
    assert summary.primary_directional_derivative < 0
    assert summary.companion_directional_derivative < 0
    assert summary.joint_directional_derivative < 0
    assert summary.negative_to_correct_motion_ratio == 0
    assert summary.value_delta_rms > 0
    assert torch.isfinite(delta).all()


def test_paired_video_joint_solve_rejects_changed_weights() -> None:
    correct = torch.eye(4)
    negative = torch.zeros_like(correct)
    cotangent = torch.ones(4, 2, 2)
    with pytest.raises(ConditionUpdateError, match="view weights"):
        paired_video_joint_program_delta(
            correct,
            negative,
            cotangent,
            task_count=2,
            view_weights=torch.tensor([0.5, 0.5, 0.25, 0.75]),
            step_size=1.0,
            relative_damping=0.01,
        )
