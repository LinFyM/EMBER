from __future__ import annotations

import pytest
import torch

from ember.rl_writer.flow_credit import (
    leave_one_out_binary_advantages,
    task_relative_aspo_loss,
)


def test_leave_one_out_advantage_is_task_centered_and_zero_without_contrast() -> None:
    mixed = leave_one_out_binary_advantages(
        torch.tensor([1.0, 0.0, 0.0, 1.0])
    )
    torch.testing.assert_close(
        mixed,
        torch.tensor([2 / 3, -2 / 3, -2 / 3, 2 / 3]),
    )
    assert float(mixed.sum()) == pytest.approx(0.0)
    assert not bool(
        leave_one_out_binary_advantages(torch.ones(4)).count_nonzero()
    )


def test_on_policy_flow_credit_lowers_success_loss_and_raises_failure_loss() -> None:
    current = torch.full((1, 4), 0.2, requires_grad=True)
    old = current.detach().clone()
    loss, metrics = task_relative_aspo_loss(
        current,
        old,
        torch.arange(4),
        torch.tensor([1.0, 0.0, 0.0, 1.0]),
        clip_epsilon=0.05,
        loss_value_clip=20.0,
        log_ratio_clip=5.0,
    )
    loss.backward()
    assert bool((current.grad[[0], :][:, [0, 3]] > 0).all())
    assert bool((current.grad[[0], :][:, [1, 2]] < 0).all())
    assert metrics.ratio_mean == pytest.approx(1.0)
    assert metrics.positive_episodes == metrics.negative_episodes == 2


def test_flow_credit_averages_chunks_inside_each_episode() -> None:
    current = torch.full((1, 3), 0.2, requires_grad=True)
    loss, _ = task_relative_aspo_loss(
        current,
        current.detach().clone(),
        torch.tensor([0, 0, 1]),
        torch.tensor([1.0, 0.0]),
        clip_epsilon=0.05,
        loss_value_clip=20.0,
        log_ratio_clip=5.0,
    )
    loss.backward()
    assert current.grad[0, 0] == pytest.approx(current.grad[0, 1])
    assert abs(float(current.grad[0, 0] + current.grad[0, 1])) == pytest.approx(
        abs(float(current.grad[0, 2]))
    )
