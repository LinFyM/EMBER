"""Direct-autograd oracles for main LoRA replay and isolated local action credit."""
import copy

import pytest
import torch

from ember.writer.function_credit import local_flow_sample, mean_velocity_loss
from ember.writer.function_reader import LocalActionReader
from ember.writer.supervised import local_action_credit, replay_functional_credit
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


def test_local_flow_sample_preserves_global_rng_and_exact_flow_relationship():
    actions = torch.randn(15, 7)
    before = torch.get_rng_state().clone()
    noisy, time, target = local_flow_sample(actions, seed=37)
    assert torch.equal(torch.get_rng_state(), before)
    assert noisy.shape == target.shape == (8, 15, 7) and time.shape == (8,)
    torch.testing.assert_close(noisy, actions[None] + time[:, None, None] * target)
    assert ((time > .001) & (time < 1)).all() and time.unique().numel() == 8
    for actual, repeated in zip((noisy, time, target), local_flow_sample(actions, seed=37), strict=True):
        torch.testing.assert_close(actual, repeated)
    assert not torch.equal(noisy, local_flow_sample(actions, seed=38)[0])


@pytest.mark.parametrize('process_mode', ['ordered', 'frame_set'])
def test_local_credit_preserves_compiler_gradients_and_carries_native_cotangents(process_mode):
    model = writer(process_mode=process_mode)
    head = LocalActionReader(width=model.config.width, heads=model.config.heads)
    with torch.no_grad():
        head.output.weight.normal_(std=.1)
    args = inputs((4,))
    sample = local_flow_sample(torch.randn(15, 7), seed=37)
    # Existing main gradients cannot be erased or supplemented by the local path.
    compiler_before = []
    for parameter in model.compiler_parameters():
        parameter.grad = torch.randn_like(parameter)
        compiler_before.append(parameter.grad.clone())
    loss, cotangents = local_action_credit(model, head, args[0], args[1:], sample, weight=.25, backward=True)
    assert loss > 0 and all(torch.isfinite(value).all() for group in cotangents for value in group)
    assert all(value.norm() > 0 for group in cotangents for value in group)
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.encoder_parameters())
    for parameter, expected in zip(model.compiler_parameters(), compiler_before, strict=True):
        torch.testing.assert_close(parameter.grad, expected)
    before = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
    with torch.no_grad():
        repeated, gradients = local_action_credit(model, head, args[0], args[1:], sample, weight=.25, backward=False)
    assert repeated == pytest.approx(loss) and gradients is None
    for p, previous in zip(model.parameters(), before, strict=True):
        if previous is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, previous)
