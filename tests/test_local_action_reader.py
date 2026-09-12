"""Local action head contracts, independent of source policy and LoRA compilation."""
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ember.writer.attention import position_encoding
from ember.writer.function_reader import LocalActionReader
from ember.lora import LoRATarget
from ember.pi05_lora import load_pi05_lora_contract
from ember.writer.video import VideoConditionedWriter, VideoWriterConfig


@pytest.fixture(autouse=True)
def small_cpu_work():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(32)
    yield
    torch.set_num_threads(previous)


def inputs(width=24, *, static=False):
    noisy_actions, time = torch.randn(8, 15, 7), torch.linspace(.001, .999, 8)
    memory = torch.randn(4, 3, width).flatten(0, 1)
    routing = position_encoding(torch.arange(4).repeat_interleave(3), width, memory.dtype)
    if static:
        routing.zero_()
    prior = memory.new_full((1, len(memory)), -math.log(len(memory)))
    return noisy_actions, time, memory, routing, prior


def unlock(reader):
    with torch.no_grad():
        reader.output.weight.normal_(std=.03)


def test_default_local_velocity_shape_and_zero_initialization():
    reader = LocalActionReader()
    result = reader(*inputs(width=256))
    assert result.shape == (8, 15, 7)
    assert result.count_nonzero() == 0
    loss = (result - torch.randn_like(result)).square().mean()
    loss.backward()
    assert reader.output.weight.grad.abs().sum() > 0


def test_every_frame_token_and_local_queries_receive_gradient_after_output_unlock():
    reader = LocalActionReader(width=24, heads=4)
    unlock(reader)
    actions, time, memory, routing, prior = inputs()
    actions.requires_grad_()
    time.requires_grad_()
    memory.requires_grad_()
    prediction = reader(actions, time, memory, routing, prior)
    gradients = torch.autograd.grad(prediction.square().mean(), (actions, time, memory))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert (gradients[0].abs().sum(-1) > 0).all()
    assert (gradients[1].abs() > 0).all()
    assert (gradients[2].abs().sum(-1) > 0).all()
    changed = memory.detach().clone()
    changed[:3] += torch.randn_like(changed[:3])
    assert not torch.allclose(prediction, reader(actions, time, changed, routing, prior))


@pytest.mark.parametrize("static", [True, False])
def test_joint_memory_route_prior_permutation_keeps_action_output_positions(static):
    reader = LocalActionReader(width=24, heads=4)
    unlock(reader)
    actions, time, memory, routing, prior = inputs(static=static)
    # Nonuniform priors exercise alignment, rather than relying on constant masks.
    prior = torch.randn_like(prior).log_softmax(-1)
    permutation = torch.tensor([8, 2, 10, 5, 1, 11, 7, 3, 0, 4, 9, 6])
    expected = reader(actions, time, memory, routing, prior)
    actual = reader(actions, time, memory[permutation], routing[permutation], prior[:, permutation])
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_noise_batch_matches_independent_queries_and_static_route_stays_absent():
    reader = LocalActionReader(width=24, heads=4)
    unlock(reader)
    actions, time, memory, routing, prior = inputs(static=True)
    expected = reader(actions, time, memory, routing, prior)
    independent = torch.cat([reader(actions[i:i + 1], time[i:i + 1], memory, routing, prior)
                             for i in range(len(actions))])
    torch.testing.assert_close(expected, independent, rtol=2e-5, atol=2e-6)
    with torch.no_grad():
        reader.routing_projection.weight.normal_(std=4)
    torch.testing.assert_close(reader(actions, time, memory, routing, prior), expected)


def test_reader_owns_local_route_without_a_writer_or_compiler_dependency():
    contract = load_pi05_lora_contract(Path(__file__).resolve().parents[1] / "configs/pi05_lora_v1.json")
    contract = replace(contract, targets=(LoRATarget("target", 3, 4),), rank=2, alpha=2)
    writer = VideoConditionedWriter(contract, VideoWriterConfig(width=24, heads=4, horizon=4,
        native_width=6, language_width=8, factor_width=4, activation_checkpoint=False))
    reader = LocalActionReader(width=24, heads=4)
    unlock(reader)
    response, visual = torch.randn(4, 4, 6), torch.randn(4, 6, 8)
    task_mask = torch.tensor([False, False, False, True, True, True]).expand(4, -1)
    evidence = writer.encode((response,), (torch.arange(4) * 5,), torch.randn(3, 8),
        torch.ones(3, dtype=torch.bool), (visual,), (torch.ones_like(task_mask),), (task_mask,))[0]
    actions, time, _, routing, prior = inputs()
    reader(actions, time, evidence.flatten(0, 1), routing, prior).square().mean().backward()
    assert reader.routing_projection.weight.grad.abs().sum() > 0
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in writer.encoder_parameters())
    assert all(parameter.grad is None for parameter in writer.compiler_parameters())
    assert not {id(parameter) for parameter in reader.parameters()} & {
        id(parameter) for parameter in writer.parameters()}


@pytest.mark.parametrize("field", ["actions", "time", "memory", "routing", "prior"])
def test_rejects_misaligned_action_or_memory_axes(field):
    reader = LocalActionReader(width=24, heads=4)
    args = list(inputs())
    index = ["actions", "time", "memory", "routing", "prior"].index(field)
    args[index] = args[index][:-1]
    with pytest.raises(ValueError, match="local"):
        reader(*args)
