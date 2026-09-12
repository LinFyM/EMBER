"""Direct-autograd oracles for complete main LoRA and native replay credit."""
import copy

import pytest
import torch

from ember.writer.function_credit import mean_velocity_loss
from ember.writer.supervised import replay_functional_credit
from test_video_program import inputs, small_cpu_work, unlock, writer


@pytest.mark.parametrize('activation_checkpoint', [False, True])
def test_main_replay_matches_direct_autograd(activation_checkpoint):
    model = writer(activation_checkpoint=activation_checkpoint)
    unlock(model)
    reference = copy.deepcopy(model)
    args = inputs((3, 4))
    direct_responses = tuple(value.detach().requires_grad_() for value in args[0])
    direct_visuals = tuple(value.detach().requires_grad_() for value in args[4])
    state = reference(direct_responses, *args[1:4], direct_visuals, *args[5:])
    targets = {name: torch.randn_like(value) for name, value in state.items()}
    correct = .125 * sum((value - targets[name]).square().mean() for name, value in state.items())
    expected = torch.autograd.grad(correct, (*reference.parameters(), *direct_responses, *direct_visuals), retain_graph=True)
    compiled = dict(zip(state, torch.autograd.grad(correct, tuple(state.values()))))
    response_grads, visual_grads = replay_functional_credit(model, args[0], args[1:], compiled)
    actual = [p.grad for p in model.parameters()] + list(response_grads) + list(visual_grads)
    for result, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, target, rtol=3e-4, atol=2e-6)


def test_velocity_loss_uses_all_horizon_only_real_action_dimensions():
    predicted = torch.zeros(2, 50, 32, requires_grad=True)
    loss = mean_velocity_loss(predicted, torch.ones_like(predicted), 7)
    loss.backward()
    assert loss == 1 and (predicted.grad[..., :7] != 0).all()
    assert predicted.grad[..., 7:].count_nonzero() == 0


@pytest.mark.parametrize('microbatch', [1, 3, 5])
def test_native_main_credit_matches_direct_autograd_and_query_slicing(microbatch):
    from ember.writer.function_credit import NativeFlowPrediction, flow_sample, paired_functional_credit
    from ember.writer.functional import prepare_frozen_writer_policy
    from test_writer_functional import _TinyPi05Policy, _tiny_pi05_contract
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    policy, contract = _TinyPi05Policy(), _tiny_pi05_contract()
    template = prepare_frozen_writer_policy(policy, contract)
    state = {name: value.detach().clone().requires_grad_() for name, value in template.items()}
    with torch.no_grad():
        for name, value in state.items():
            if 'lora_B' in name:
                value.fill_(.03)
    batch = {ACTION: torch.randn(5, 4, 3), 'image': torch.randn(5, 3, 4, 4),
             OBS_LANGUAGE_TOKENS: torch.ones(5, 4, dtype=torch.long),
             OBS_LANGUAGE_ATTENTION_MASK: torch.ones(5, 4, dtype=torch.bool)}
    sample = flow_sample(policy, batch, seed=37, device='cpu', random_batch=8, offset=2)
    prediction = torch.func.functional_call(NativeFlowPrediction(policy),
        {'policy.' + name: value for name, value in state.items()}, (sample,))
    expected = torch.autograd.grad(mean_velocity_loss(prediction, sample.target, 3) * .125, tuple(state.values()))
    credit = paired_functional_credit(policy, state, contract, batch, seed=37, device='cpu',
        random_batch=8, offset=2, microbatch=microbatch, condition_weight=.125)
    for actual, target in zip(credit['lora_cotangent'].values(), expected, strict=True):
        torch.testing.assert_close(actual, target, rtol=2e-4, atol=2e-6)
    assert credit['source_forward_calls'] == 0
    assert all(p.grad is None and not p.requires_grad for p in policy.parameters())
