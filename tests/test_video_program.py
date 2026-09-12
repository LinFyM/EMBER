"""Behavioral contracts for native video evidence and complete LoRA compilation."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ember.lora import LoRATarget, identity_lora_state, validate_lora_state
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
    # Exercise every canonical target slot without allocating production-size factors.
    contract = replace(contract, targets=tuple(LoRATarget(target.name, 3, 4) for target in contract.targets),
                       rank=2, alpha=2)
    config = replace(VideoWriterConfig(width=12, heads=3, horizon=4, native_width=6, language_width=8,
                                       prior_width=7, factor_width=5, edge_chunk=2), **kwargs)
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
    prior = tuple(torch.randn(length, 5, 7) for length in lengths)
    return response, indices, language, language_mask, visual, valid, task, prior


def unlock(model):
    with torch.no_grad():
        for group in model.decoder.groups:
            group.a_factors.normal_(std=0.03)
            group.b_factors.normal_(std=0.03)


def permute(args, order):
    return tuple((value[0][order],) if index in (0, 1, 4, 5, 6, 7) else value
                 for index, value in enumerate(args))


def test_complete_identity_then_video_conditioned_factors():
    model, args = writer(), inputs()
    initial = model(*args)
    validate_lora_state(initial, model.contract)
    assert len(initial) == 76
    for name, value in identity_lora_state(model.contract).items():
        torch.testing.assert_close(initial[name], value)
    unlock(model)
    changed = list(args)
    changed[0] = (args[0][0] + torch.randn_like(args[0][0]) * 3,)
    original, perturbed = model(*args), model(*changed)
    assert all(not torch.allclose(original[name], perturbed[name]) for name in original)


def test_frozen_prior_values_affect_complete_lora_and_trainable_grounding():
    model, args = writer(), inputs((4,))
    unlock(model)
    original = model(*args)
    changed = list(args)
    changed[7] = (args[7][0] + torch.randn_like(args[7][0]) * 3,)
    perturbed = model(*changed)
    assert all(not torch.allclose(original[name], perturbed[name]) for name in original)
    sum(value.square().mean() for value in original.values()).backward()
    assert model.encoder.prior_projection.weight.grad.norm() > 0
    assert all(value.grad is None and not value.requires_grad for value in args[7])
    assert len(original) == 76


@pytest.mark.parametrize("activation_checkpoint", [False, True])
def test_full_horizon_and_real_patches_receive_encoder_gradient(activation_checkpoint):
    model, args = writer(activation_checkpoint=activation_checkpoint), inputs((4,))
    args[0][0].requires_grad_()
    args[4][0].requires_grad_()
    evidence = model.encode(*args)[0]
    assert evidence.shape == (4, 3, 12)
    grad_h, grad_z = torch.autograd.grad(evidence[-1].square().sum(), (args[0][0], args[4][0]))
    assert (grad_h.abs().sum(-1) > 0).all()  # Every frame and every full-H slot can contribute.
    assert (grad_z[:, :3].abs().sum(-1) > 0).all()
    assert grad_z[:, 6:].count_nonzero() == 0


def test_exact_contextual_task_span_and_padding_wall():
    model, args = writer(), inputs()
    original = model.encode(*args)[0]
    args[4][0][:, 6:] = torch.randn_like(args[4][0][:, 6:]) * 100
    args[2][~args[3]] = 200
    torch.testing.assert_close(model.encode(*args)[0], original)
    args[4][0][:, 3] += torch.randn_like(args[4][0][:, 3]) * 2
    assert not torch.allclose(model.encode(*args)[0], original)
    invalid = list(args)
    invalid[6] = (args[6][0].clone(),)
    invalid[6][0][:, 4] = False
    with pytest.raises(ValueError, match="task-token span"):
        model.encode(*invalid)
    invalid[6] = (args[6][0].clone(),)
    invalid[6][0][:, 5] = False
    invalid[6][0][:, 7] = True
    with pytest.raises(ValueError, match="task-token span"):
        model.encode(*invalid)


def test_ordered_prefix_causality_and_past_dependence():
    model, args = writer(), inputs((6,))
    original = model.encode(*args)[0]
    prefix = tuple((value[0][:3],) if index in (0, 1, 4, 5, 6, 7) else value
                   for index, value in enumerate(args))
    torch.testing.assert_close(model.encode(*prefix)[0], original[:3], rtol=2e-5, atol=2e-6)
    args[0][0][3:] += torch.randn_like(args[0][0][3:]) * 3
    args[4][0][3:] += torch.randn_like(args[4][0][3:]) * 3
    args[7][0][3:] += torch.randn_like(args[7][0][3:]) * 3
    changed = model.encode(*args)[0]
    torch.testing.assert_close(changed[:3], original[:3], rtol=0, atol=0)
    assert not torch.allclose(changed[3:], original[3:])
    args[0][0][0] += torch.randn_like(args[0][0][0]) * 3
    assert not torch.allclose(model.encode(*args)[0][-1], changed[-1])


def test_frame_set_is_permutation_equivariant_and_time_blind():
    ordered, model, args = writer(), writer(process_mode="frame_set"), inputs()
    assert {name: value.shape for name, value in ordered.named_parameters()} == {
        name: value.shape for name, value in model.named_parameters()}
    order = torch.tensor([3, 1, 4, 0, 2])
    unlock(model)
    shuffled = permute(args, order)
    original = model.encode(*args)[0]
    torch.testing.assert_close(model.encode(*shuffled)[0], original[order], rtol=2e-5, atol=2e-6)
    changed_times = list(args)
    changed_times[1] = (args[1][0] * 100 + 70,)
    torch.testing.assert_close(model.encode(*changed_times)[0], original)
    original_lora, shuffled_lora = model(*args), model(*shuffled)
    for name, value in original_lora.items():
        torch.testing.assert_close(shuffled_lora[name], value, rtol=2e-5, atol=2e-6)


def test_all_fifty_native_horizon_slots_remain_live():
    model, args = writer(horizon=50), list(inputs((3,)))
    args[0] = (torch.randn(3, 50, 6, requires_grad=True),)
    evidence = model.encode(*args)[0]
    gradient = torch.autograd.grad(evidence[-1].square().sum(), args[0][0])[0]
    assert (gradient.abs().sum(-1) > 0).all()


@pytest.mark.parametrize("mode", ["ordered", "frame_set"])
def test_video_set_order_invariance_and_equal_video_prior(mode):
    model, args = writer(process_mode=mode), inputs((3, 5))
    unlock(model)
    videos = model.encode(*args)
    memory, routing, prior = model.memory(videos, args[1])
    assert memory.shape == routing.shape == (24, 12)
    torch.testing.assert_close(prior.exp()[0, :9].sum(), torch.tensor(1.))
    torch.testing.assert_close(prior.exp()[0, 9:].sum(), torch.tensor(1.))
    original, reversed_set = model.decode(videos, args[1]), model.decode(videos[::-1], args[1][::-1])
    for name in original:
        torch.testing.assert_close(reversed_set[name], original[name], rtol=2e-5, atol=2e-6)


def test_parameter_partition_covers_exactly_encoder_and_compiler():
    model = writer()
    encoder = {id(parameter) for parameter in model.encoder_parameters()}
    compiler = {id(parameter) for parameter in model.compiler_parameters()}
    assert encoder and compiler and not encoder & compiler
    assert encoder | compiler == {id(parameter) for parameter in model.parameters()}
    assert encoder == {id(parameter) for parameter in model.encoder.parameters()}
    with pytest.raises(ValueError, match="identity"):
        VideoWriterConfig(architecture="old_horizon")
