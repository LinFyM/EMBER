"""Behavioral contracts for process-only q and a shared native-coordinate outlet."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ember.lora import LoRATarget, LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.writer.video import VideoConditionedWriter, VideoWriterConfig


@pytest.fixture(autouse=True)
def small_cpu_work():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(32)
    yield
    torch.set_num_threads(previous)


def writer(**kwargs):
    contract = load_pi05_lora_contract(Path(__file__).resolve().parents[1] / "configs/pi05_lora_v1.json")
    contract = replace(contract, targets=tuple(LoRATarget(target.name, 3, 4) for target in contract.targets),
                       rank=2, alpha=2)
    config = replace(VideoWriterConfig(width=12, heads=3, horizon=4, native_width=6, language_width=8,
                                       query_chunk=2, memory_timescale=8), **kwargs)
    return VideoConditionedWriter(contract, config)


def inputs(lengths=(5,)):
    response = tuple(torch.randn(length, 4, 6) for length in lengths)
    indices = tuple(torch.arange(length) * 5 for length in lengths)
    language = torch.randn(6, 8)
    language_mask = torch.tensor([False, True, True, False, True, False])
    visual = tuple(torch.randn(length, 8, 8) for length in lengths)
    valid = tuple(torch.tensor([True, True, True, True, True, True, False, False]).expand(length, -1)
                  for length in lengths)
    task = tuple(torch.tensor([False, False, False, True, True, True, False, False]).expand(length, -1)
                 for length in lengths)
    return response, indices, language, language_mask, visual, valid, task


class _FixedCoordinates:
    """A fixed linear test operator; actual source/adjoint tests live separately."""

    def __init__(self, model, args):
        self.contract = model.contract
        self.compile_calls = 0
        self.predictions = torch.randn(*args[0][0].shape[:2], model.config.action_width)
        self.bases, self.maps = {}, {}
        for target in model.contract.targets:
            self.bases[target.name] = torch.linalg.qr(torch.randn(target.in_features, model.contract.rank))[0].T
            self.maps[target.name] = torch.randn(target.out_features * model.contract.rank, self.predictions.numel()) * .1

    def compile(self, q):
        self.compile_calls += 1
        state = {}
        for target in self.contract.targets:
            state[target.name + LORA_A_SUFFIX] = self.bases[target.name].to(q)
            state[target.name + LORA_B_SUFFIX] = (
                self.maps[target.name].to(q) @ q.flatten()).reshape(target.out_features, self.contract.rank)
        return state

    def adjoint(self, gradients):
        return sum(self.maps[target.name].T @ gradients[target.name + LORA_B_SUFFIX].detach().flatten()
                   for target in self.contract.targets).reshape_as(self.predictions)


def native_inputs(model, args):
    return _FixedCoordinates(model, args)


def unlock(model):
    with torch.no_grad():
        model.q_head.weight.normal_(std=.03)


def permute_content(args, order):
    return tuple((value[0][order],) if index in (0, 4, 5, 6) else value
                 for index, value in enumerate(args))


def test_complete_identity_and_the_first_real_update_opens_upstream_credit():
    model, args = writer(), inputs()
    native = native_inputs(model, args)
    state = model(*args, native_inputs=native)
    validate_lora_state(state, model.contract)
    assert len(state) == 76
    for target in model.contract.targets:
        assert state[target.name + LORA_A_SUFFIX].count_nonzero() > 0
        assert state[target.name + LORA_B_SUFFIX].count_nonzero() == 0
    objective = sum((state[target.name + LORA_B_SUFFIX] - 1).square().mean() for target in model.contract.targets)
    objective.backward()
    assert model.q_head.weight.grad.norm() > 0
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    model.zero_grad(set_to_none=True)
    state = model(*args, native_inputs=native)
    sum((state[target.name + LORA_B_SUFFIX] - 1).square().mean() for target in model.contract.targets).backward()
    assert model.encoder.visual_projection.weight.grad.norm() > 0
    assert model.encoder.native_projection.weight.grad.norm() > 0
    assert model.encoder.past.gates.weight.grad.norm() > 0
    assert model.encoder.future.gates.weight.grad.norm() > 0


@pytest.mark.parametrize("activation_checkpoint", [False, True])
def test_constant_video_cannot_generate_process_content_after_learning(activation_checkpoint):
    model, args = writer(activation_checkpoint=activation_checkpoint), list(inputs((11,)))
    unlock(model)
    with torch.no_grad():
        for parameter in model.outlet.parameters():
            parameter.normal_(std=.2)
    args[0] = (args[0][0][:1].expand_as(args[0][0]).clone(),)
    args[4] = (args[4][0][:1].expand_as(args[4][0]).clone(),)
    native = native_inputs(model, args)
    # Static query conditions may change arbitrarily; process Values stay zero.
    native.predictions *= 100
    videos = model.encode(*args)
    torch.testing.assert_close(videos[0][0], torch.zeros_like(videos[0][0]), rtol=0, atol=1e-6)
    q = model.action_cotangents(videos, native.predictions)
    torch.testing.assert_close(q, torch.zeros_like(q), rtol=0, atol=1e-6)
    state = model.decode(videos, native)
    for target in model.contract.targets:
        update = state[target.name + LORA_B_SUFFIX] @ state[target.name + LORA_A_SUFFIX]
        torch.testing.assert_close(update, torch.zeros_like(update), rtol=0, atol=1e-6)
    shifted = list(args)
    shifted[1] = (args[1][0] * 19 + 731,)
    torch.testing.assert_close(model(*shifted, native_inputs=native), model(*args, native_inputs=native))
    single = list(inputs((1,)))
    assert model.action_cotangents(model.encode(*single), native_inputs(model, single).predictions).count_nonzero() == 0


@pytest.mark.parametrize("activation_checkpoint", [False, True])
def test_complete_horizon_visual_and_both_directions_receive_real_credit(activation_checkpoint):
    model, args = writer(activation_checkpoint=activation_checkpoint), inputs()
    unlock(model)
    args[0][0].requires_grad_()
    args[4][0].requires_grad_()
    state = model(*args, native_inputs=native_inputs(model, args))
    sum(value.square().mean() for name, value in state.items() if name.endswith(LORA_B_SUFFIX)).backward()
    assert args[0][0].grad.abs().sum(-1).gt(0).all()
    assert args[4][0].grad[args[5][0]].abs().sum(-1).gt(0).all()
    assert model.encoder.past.gates.weight.grad.norm() > 0
    assert model.encoder.future.gates.weight.grad.norm() > 0
    assert model.encoder.horizon_read.query.weight.grad.norm() > 0
    assert model.process_read.value.weight.grad.norm() > 0


def test_exact_task_span_and_masked_padding_are_respected():
    model, args = writer(), inputs()
    unlock(model)
    native = native_inputs(model, args)
    reference = model(*args, native_inputs=native)
    changed = list(args)
    changed[2] = args[2].clone()
    changed[2][~args[3]] += 100
    changed[4] = (args[4][0].clone(),)
    changed[4][0][~args[5][0]] += 100
    torch.testing.assert_close(model(*changed, native_inputs=native), reference)
    invalid = list(args)
    invalid[6] = (args[6][0].clone(),)
    invalid[6][0][:, 3] = False
    with pytest.raises(ValueError, match="task-token"):
        model.encode(*invalid)


def test_real_order_matters_while_absolute_video_clock_does_not():
    model, args = writer(), inputs()
    unlock(model)
    native = native_inputs(model, args)
    original = model.action_cotangents(model.encode(*args), native.predictions)
    order = torch.tensor([3, 0, 4, 1, 2])
    changed = permute_content(args, order)
    reordered = model.action_cotangents(model.encode(*changed), native.predictions[order])
    assert not torch.allclose(reordered, original[order])
    changed = list(args)
    changed[1] = (args[1][0] * 19 + 731,)
    torch.testing.assert_close(model.action_cotangents(model.encode(*changed), native.predictions), original)
    with pytest.raises(ValueError, match="strictly increasing"):
        changed[1] = (args[1][0][order],)
        model.encode(*changed)


def test_all_fifty_native_horizon_slots_remain_live():
    model = writer(horizon=50)
    args = list(inputs((3,)))
    args[0] = (torch.randn(3, 50, 6, requires_grad=True),)
    unlock(model)
    q = model.action_cotangents(model.encode(*args), native_inputs(model, args).predictions)
    assert q.shape == (3, 50, 7)
    q.square().mean().backward()
    assert args[0][0].grad.abs().sum(-1).gt(0).all()


def test_k1_and_frozen_source_prediction_information_wall():
    model, args = writer(), inputs()
    with pytest.raises(ValueError, match="K1"):
        model.encode(*inputs((3, 7)))
    native = native_inputs(model, args)
    native.predictions.requires_grad_()
    with pytest.raises(ValueError, match="frozen"):
        model(*args, native_inputs=native)
    with pytest.raises(ValueError, match="identity"):
        writer(process_mode="frame_set")


def test_parameter_partition_covers_exactly_encoder_q_reader_and_outlet():
    model = writer()
    encoder, compiler = map(lambda it: {id(value) for value in it},
                            (model.encoder_parameters(), model.compiler_parameters()))
    assert encoder and compiler and not encoder & compiler
    assert encoder | compiler == {id(value) for value in model.parameters()}
    assert {id(value) for value in model.outlet.parameters()} <= compiler


def test_old_architecture_requires_its_frozen_runtime():
    with pytest.raises(ValueError, match="identity"):
        writer(schema="video_conditioned_writer_v8", architecture="source_pullback_process_lora_v1")
