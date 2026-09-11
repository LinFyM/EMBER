"""Direct-autograd oracle for distinct encoder, reader and compiler credit."""
import copy

import pytest
import torch

from ember.writer.function_credit import mean_velocity_loss
from ember.writer.function_reader import ExecutionVideoReader, distillation_weight
from ember.writer.supervised import replay_functional_credit
from test_video_program import inputs, small_cpu_work, unlock, writer


@pytest.mark.parametrize('rho', [0., .25])
@pytest.mark.parametrize('activation_checkpoint', [False, True])
def test_replay_matches_direct_grouped_objective(rho, activation_checkpoint):
    model = writer(activation_checkpoint=activation_checkpoint)
    unlock(model)
    reference = copy.deepcopy(model)
    args = inputs((3, 4))
    direct_responses = tuple(value.detach().requires_grad_() for value in args[0])
    videos = reference.encode(direct_responses, *args[1:])
    state = reference.decode(videos, args[1])
    targets = {name: torch.randn_like(value) for name, value in state.items()}
    teacher_targets = {name: torch.randn_like(value) for name, value in state.items()}
    weight = .125
    correct = weight * sum((value - targets[name]).square().mean() for name, value in state.items())
    distill = weight * sum((value - teacher_targets[name]).square().mean() for name, value in state.items())
    # Nonlinear auxiliary function of exactly the same E; independent compiler.
    memory = torch.cat([video.flatten(0, 1) for video in videos])
    auxiliary = weight * memory.sin().square().mean()
    encoder_params = tuple(reference.encoder_parameters())
    compiler_params = tuple(reference.compiler_parameters())
    expected_encoder = torch.autograd.grad(correct + auxiliary, encoder_params + direct_responses, retain_graph=True)
    expected_compiler = torch.autograd.grad((1-rho)*correct + rho*distill, compiler_params, retain_graph=True)
    compiled = dict(zip(state, torch.autograd.grad(correct, tuple(state.values()), retain_graph=True)))
    distilled = dict(zip(state, torch.autograd.grad(distill, tuple(state.values()), retain_graph=True)))
    memory_gradient = torch.autograd.grad(auxiliary, memory)[0]
    response_grads = replay_functional_credit(model, args[0], args[1:], compiled, distilled, memory_gradient, rho)
    actual = [p.grad for p in model.encoder_parameters()] + list(response_grads)
    for result, expected in zip(actual, expected_encoder, strict=True):
        torch.testing.assert_close(result, expected, rtol=3e-4, atol=2e-6)
    for parameter, expected in zip(model.compiler_parameters(), expected_compiler, strict=True):
        torch.testing.assert_close(parameter.grad, expected, rtol=3e-4, atol=2e-6)


def test_zero_initialized_reader_learns_output_then_video_and_query():
    reader = ExecutionVideoReader(12, 3, query_width=6, action_width=8)
    query, base = torch.randn(2, 4, 6), torch.randn(2, 4, 8)
    memory = torch.randn(9, 12, requires_grad=True)
    prior = torch.full((1, 9), -torch.log(torch.tensor(9.)))
    target = torch.randn_like(base)
    torch.testing.assert_close(reader(query, base, memory, prior), base)
    loss = mean_velocity_loss(reader(query, base, memory, prior), target, 7)
    loss.backward()
    assert reader.output.weight.grad.norm() > 0
    assert memory.grad.count_nonzero() == 0
    with torch.no_grad():
        reader.output.weight.add_(reader.output.weight.grad, alpha=-.1)
    reader.zero_grad(set_to_none=True)
    memory.grad = None
    query.requires_grad_()
    mean_velocity_loss(reader(query, base, memory, prior), target, 7).backward()
    assert memory.grad.norm() > 0 and query.grad.norm() > 0
    assert reader.read.key.weight.grad.norm() > 0
    assert reader.output.weight.grad[-1].count_nonzero() == 0


def test_velocity_loss_uses_all_horizon_only_real_action_dimensions():
    predicted = torch.zeros(2, 50, 32, requires_grad=True)
    target = torch.ones_like(predicted)
    loss = mean_velocity_loss(predicted, target, 7)
    loss.backward()
    assert loss == 1 and (predicted.grad[..., :7] != 0).all()
    assert predicted.grad[..., 7:].count_nonzero() == 0


def test_distillation_is_fixed_by_update_count():
    spec = dict(enabled=True, distill_start=32, distill_end=100, distill_max=.25)
    assert [distillation_weight(spec, step) for step in (0, 32, 66, 100, 200)] == [0, 0, .125, .25, .25]
    assert distillation_weight({**spec, 'enabled': False}, 200) == 0


@pytest.mark.parametrize('microbatch', [1, 3, 5])
@pytest.mark.parametrize('enabled', [False, True])
def test_paired_native_credit_matches_direct_autograd_and_query_slicing(microbatch, enabled):
    from ember.writer.function_credit import NativeFlowPrediction, flow_sample, paired_functional_credit
    from ember.writer.functional import prepare_frozen_writer_policy
    from test_writer_functional import _TinyPi05Policy, _tiny_pi05_contract
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    policy, contract = _TinyPi05Policy(), _tiny_pi05_contract()
    template = prepare_frozen_writer_policy(policy, contract)
    state = {name: value.detach().clone().requires_grad_() for name, value in template.items()}
    for name, value in state.items():
        if 'lora_B' in name:
            value.data.fill_(.03)
    batch = {ACTION: torch.randn(5, 4, 3), 'image': torch.randn(5, 3, 4, 4),
             OBS_LANGUAGE_TOKENS: torch.ones(5, 4, dtype=torch.long),
             OBS_LANGUAGE_ATTENTION_MASK: torch.ones(5, 4, dtype=torch.bool)}
    reader = ExecutionVideoReader(12, 3, query_width=4, action_width=3) if enabled else None
    if reader is not None:
        with torch.no_grad():
            reader.output.weight.normal_(std=.1)
    reference = copy.deepcopy(reader)
    memory, prior = torch.randn(9, 12, requires_grad=True), torch.full((1, 9), -.5)
    sample = flow_sample(policy, batch, seed=37, device='cpu', random_batch=8, offset=2)
    native = NativeFlowPrediction(policy)
    with torch.no_grad():
        base, query = native(sample)
    student, _ = torch.func.functional_call(native, {'policy.'+name:value for name,value in state.items()}, (sample,))
    expected_c = torch.autograd.grad(mean_velocity_loss(student,sample.target,3)*.125,
                                     tuple(state.values()), retain_graph=enabled)
    expected_d = expected_r = None
    if enabled:
        teacher = reference(query,base,memory,prior)
        expected_d = torch.autograd.grad(mean_velocity_loss(student,teacher.detach(),3)*.125, tuple(state.values()))
        expected_r = torch.autograd.grad(mean_velocity_loss(teacher,sample.target,3)*.125,
                                         (memory,*reference.parameters()))
    credit = paired_functional_credit(policy,state,contract,batch,reader=reader,memory=memory,prior=prior,
        seed=37,device='cpu',random_batch=8,offset=2,microbatch=microbatch,
        condition_weight=.125,auxiliary_weight=1.,rho=.25 if enabled else 0.)
    for actual,expected in zip(credit['lora_cotangent'].values(),expected_c,strict=True):
        torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-6)
    if enabled:
        for actual,expected in zip(credit['distill_cotangent'].values(),expected_d,strict=True):
            torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-6)
        for actual,expected in zip([credit['memory_cotangent'],*[p.grad for p in reader.parameters()]],expected_r,strict=True):
            torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-6)
    else:
        assert credit['source_forward_calls'] == 0 and credit['memory_cotangent'] is None
    assert all(p.grad is None and not p.requires_grad for p in policy.parameters())
