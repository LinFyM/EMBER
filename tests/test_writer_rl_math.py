from __future__ import annotations

import copy
import itertools
import random

import pytest
import torch

from ember.writer.rl_math import (
    DecisionReservoir, adamw_trust_step, exploration_covariance, loo_advantages,
    rl_mean_cotangent, task_trust_kl,
)


def test_gaussian_credit_matches_distribution_score_and_unbiased_decision_sum():
    covariance = exploration_covariance(dtype=torch.float64)
    assert covariance[0, 7].item() == pytest.approx(0.8 * 0.05 ** 2)
    assert covariance[6, 34].item() == pytest.approx(0.8 ** 4 * 0.1 ** 2)
    assert covariance[0, 1].item() == 0
    precision = torch.linalg.inv(covariance)
    mean = torch.zeros(4, 35, dtype=torch.float64, requires_grad=True)
    z = torch.linspace(-0.1, 0.15, 140, dtype=torch.float64).reshape(4, 35)
    logprob = torch.distributions.MultivariateNormal(mean, covariance_matrix=covariance).log_prob(z)
    oracle, = torch.autograd.grad((-0.1 / 16 * (2 / 3) * logprob).sum(), mean)
    full = rl_mean_cotangent(z, mean, 2 / 3, 4, 4, precision=precision)
    torch.testing.assert_close(full, oracle)
    subset_estimates = [rl_mean_cotangent(z[list(s)], mean[list(s)], 2 / 3, 4, 2,
                                         precision=precision).sum(0)
                        for s in itertools.combinations(range(4), 2)]
    torch.testing.assert_close(torch.stack(subset_estimates).mean(0), full.sum(0))
    assert not full.requires_grad


def test_loo_zero_credit_and_independent_episode_baseline():
    rewards = torch.tensor([[0., 0., 0., 0.], [1., 1., 1., 1.], [1., 1., 0., 0.]], requires_grad=True)
    advantages = loo_advantages(rewards)
    torch.testing.assert_close(advantages, torch.tensor([[0.] * 4, [0.] * 4, [2 / 3, 2 / 3, -2 / 3, -2 / 3]]))
    assert not advantages.requires_grad


def test_reservoir_is_uniform_bounded_and_trust_subset_is_fixed_by_caller():
    counts = torch.zeros(40)
    for seed in range(2000):
        reservoir = DecisionReservoir(random.Random(seed))
        for decision in range(40):
            reservoir.add(decision)
        assert reservoir.total_seen == 40 and len(set(reservoir.items)) == 16
        counts[reservoir.items] += 1
    # A fixed-seed distribution check catches recency-biased replacement.
    assert torch.max((counts - 800).abs()) < 100
    subset = reservoir.sample(rng=random.Random(7))
    assert len(subset) == 4 and set(subset) <= set(reservoir.items)
    assert subset == reservoir.sample(rng=random.Random(7))


def test_trust_uses_full_35d_and_balances_episodes_before_tasks():
    old = torch.zeros(7, 35, dtype=torch.float64)
    new = old.clone()
    # Task a has one decision at KL .08 and four at zero. Its mean is .04,
    # so pooling decisions (.016) would incorrectly pass the .02 bound.
    new[0, -1] = 0.4
    new[5:, -1] = 0.1
    values = task_trust_kl(new, old, ["a"] * 5 + ["b"] * 2,
                           [0, 1, 1, 1, 1, 0, 0], precision=torch.eye(35, dtype=torch.float64))
    assert values == pytest.approx({"a": 0.04, "b": 0.005})
    assert task_trust_kl(old[:0], old[:0], [], []) == {}


def _optimizer(parameter):
    return torch.optim.AdamW([parameter], lr=0.03, betas=(0.9, 0.95), weight_decay=0.1, foreach=False)


def _assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_state_equal(a, b)
    else:
        assert actual == expected


def test_scaled_candidate_commits_one_adam_direction_and_one_moment_step():
    parameter = torch.nn.Parameter(torch.tensor([0.8, -0.4], dtype=torch.float64))
    optimizer = _optimizer(parameter)
    for gradient in ([0.3, -0.9], [0.5, 0.4]):
        parameter.grad = torch.tensor(gradient, dtype=torch.float64)
        optimizer.step()
    old = parameter.detach().clone()
    reference = torch.nn.Parameter(old.clone())
    reference_optimizer = _optimizer(reference)
    reference_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    parameter.grad = reference.grad = torch.tensor([-0.8, 0.2], dtype=torch.float64)
    reference_optimizer.step()
    direction = reference.detach() - old
    observed = []

    def score():
        observed.append(parameter.detach().clone())
        return {"task": (0.1, float("nan"), 0.01)[len(observed) - 1]}

    result = adamw_trust_step(optimizer, score)
    assert result.accepted and result.alpha == 0.25
    assert [a.alpha for a in result.attempts] == [1., 0.5, 0.25]
    for actual, alpha in zip(observed, [1., 0.5, 0.25]):
        torch.testing.assert_close(actual, old + alpha * direction)
    _assert_state_equal(optimizer.state_dict(), reference_optimizer.state_dict())
    assert optimizer.state[parameter]["step"].item() == 3


@pytest.mark.parametrize("initialized", [False, True])
def test_all_rejections_restore_parameters_moments_and_step(initialized):
    parameter = torch.nn.Parameter(torch.tensor([0.8, -0.4]))
    optimizer = _optimizer(parameter)
    parameter.grad = torch.tensor([0.3, -0.9])
    if initialized:
        optimizer.step()
    before = copy.deepcopy(optimizer.state_dict())
    old = parameter.detach().clone()
    result = adamw_trust_step(optimizer, lambda: {"a": 0.03, "b": 0.0})
    assert not result.accepted and result.alpha is None and len(result.attempts) == 4
    torch.testing.assert_close(parameter, old, rtol=0, atol=0)
    _assert_state_equal(optimizer.state_dict(), before)


def test_exception_rolls_back_and_is_reraised():
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = _optimizer(parameter)
    parameter.grad = torch.tensor([0.5])
    old = copy.deepcopy(optimizer.state_dict())

    def score():
        raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError, match="observer failed"):
        adamw_trust_step(optimizer, score)
    torch.testing.assert_close(parameter, torch.tensor([1.]), rtol=0, atol=0)
    _assert_state_equal(optimizer.state_dict(), old)


def test_nonfinite_direction_rejects_before_candidate_forward():
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = _optimizer(parameter)
    parameter.grad = torch.tensor([float("inf")])
    before = copy.deepcopy(optimizer.state_dict())

    def score():
        pytest.fail("nonfinite direction must not run a candidate forward")

    result = adamw_trust_step(optimizer, score)
    assert not result.accepted and not result.attempts
    torch.testing.assert_close(parameter, torch.tensor([1.]), rtol=0, atol=0)
    _assert_state_equal(optimizer.state_dict(), before)
