from __future__ import annotations

import math

import pytest
import torch

from ember.reward.protocol import RewardProtocolError
from ember.rl_writer.program_credit import (
    binary_first_pair_credit,
    program_cotangent,
    program_direction,
    program_direction_seed,
)


def test_program_directions_are_keyed_reproducible_rademacher_values() -> None:
    seed = program_direction_seed(
        31,
        cycle=2,
        global_task_id=7,
        pair_index=0,
    )
    first = program_direction(seed, (8, 16))
    second = program_direction(seed, (8, 16))
    other_pair = program_direction(
        program_direction_seed(
            31,
            cycle=2,
            global_task_id=7,
            pair_index=1,
        ),
        (8, 16),
    )
    assert torch.equal(first, second)
    assert set(first.unique().tolist()) == {-1.0, 1.0}
    assert not torch.equal(first, other_pair)


@pytest.mark.parametrize(
    ("success_plus", "success_minus", "progress_plus", "progress_minus", "value", "mode"),
    [
        (True, False, -1.0, 1.0, 1.0, "binary_discordant"),
        (False, True, 1.0, -1.0, -1.0, "binary_discordant"),
        (True, True, 0.7, -0.4, 0.0, "paired_success_zero"),
        (False, False, 0.8, -0.2, 0.5, "paired_failure_semantic"),
    ],
)
def test_binary_first_pair_credit_obeys_pre_registered_precedence(
    success_plus: bool,
    success_minus: bool,
    progress_plus: float,
    progress_minus: float,
    value: float,
    mode: str,
) -> None:
    credit = binary_first_pair_credit(
        success_plus=success_plus,
        success_minus=success_minus,
        progress_plus=progress_plus,
        progress_minus=progress_minus,
    )
    assert credit.value == pytest.approx(value)
    assert credit.mode == mode


def test_program_cotangent_is_exact_two_pair_normalized_sum() -> None:
    directions = (
        torch.tensor([[1.0, -1.0], [1.0, -1.0]]),
        torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
    )
    result = program_cotangent(directions, (1.0, -0.5))
    expected = (directions[0] - 0.5 * directions[1]) / (
        2.0 * math.sqrt(directions[0].numel())
    )
    torch.testing.assert_close(result, expected)


def test_program_credit_rejects_unbounded_or_mismatched_inputs() -> None:
    with pytest.raises(RewardProtocolError, match="unbounded"):
        binary_first_pair_credit(
            success_plus=False,
            success_minus=False,
            progress_plus=1.1,
            progress_minus=0.0,
        )
    with pytest.raises(RewardProtocolError, match="exactly two"):
        program_cotangent((torch.ones(2, 2),), (1.0,))
    with pytest.raises(RewardProtocolError, match="invalid antithetic"):
        program_cotangent(
            (torch.ones(2, 2), torch.ones(3, 2)),
            (1.0, 0.0),
        )
