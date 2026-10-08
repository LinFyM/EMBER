"""Scientific normalization regressions: full latent score and per-query keep."""
import torch

from ember.experience_compiler.credit import stage_weights
from ember.experience_compiler.execution import Transition, score_cotangent
from ember.experience_compiler.learning import ValueBaseline


def test_score_is_four_conditions_two_queries_with_sampling_compensation():
    transition = Transition(3, .7, torch.zeros(1, 50, 32), torch.ones(1, 50, 32),
                            torch.full((1, 50, 32), 3.))
    value = score_cotangent(transition, 2., replans=40, retained=16)
    # Full unsampled episode/condition coefficient is 1/8, not the old 1/16.
    expected = 2 * (1 / 8) * (40 / 16) * (10 / 2) * 1.15 / .7 * 2
    assert torch.allclose(value, torch.full((1, 50, 32), expected))
    assert value[..., 31].abs().sum() > 0  # score credit includes latent tail.


def test_stage_mass_does_not_depend_on_number_of_actual_revisions():
    assert stage_weights(1) == [1.]
    for count in (2, 4, 7, 20):
        weights = stage_weights(count)
        assert weights[-1] == .5
        assert abs(sum(weights) - 1.) < 1e-12
        assert abs(sum(weights[:-1]) - .5) < 1e-12


def test_value_targets_cannot_backpropagate_into_compilation_context():
    baseline = ValueBaseline()
    context = torch.randn(2, 264, requires_grad=True)
    baseline(context).square().sum().backward()
    assert context.grad is None
    assert baseline.network[-1].weight.grad is not None


def test_keep_requires_querywise_regression_before_averaging():
    previous = torch.tensor([1., 9.], requires_grad=True)
    current = torch.tensor([4., 4.], requires_grad=True)
    keep = torch.relu(current - previous.detach()).mean() * .2
    keep.backward()
    assert abs(float(keep.detach()) - .3) < 1e-7
    assert current.grad.tolist() == [.10000000149011612, 0.]
    assert previous.grad is None
    # Positive and negative query changes must not cancel before the ReLU.
    assert torch.relu(current.detach().mean() - previous.detach().mean()) == 0
