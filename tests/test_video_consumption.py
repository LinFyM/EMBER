from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ember.lora import LoRATarget
from ember.pi05_lora import load_pi05_lora_contract
from ember.writer.horizon import HorizonRelationWriter, HorizonWriterConfig


@pytest.fixture(autouse=True)
def small_cpu_work():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(previous)


def writer(mode):
    contract = load_pi05_lora_contract(Path(__file__).resolve().parents[1] / "configs/pi05_lora_v1.json")
    contract = replace(contract, targets=(LoRATarget("left", 3, 4), LoRATarget("right", 3, 4)), rank=2, alpha=2)
    config = HorizonWriterConfig(width=12, heads=3, horizon=4, native_width=6, language_width=8,
                                 factor_width=5, edge_chunk=2, activation_checkpoint=True,
                                 consumer_mode="semantic_process", process_mode=mode,
                                 compiler_language_mode="semantic_then_process_v1")
    return HorizonRelationWriter(contract, config)


def inputs(length=5):
    response = torch.randn(length, 4, 6, requires_grad=True)
    visual = torch.randn(length, 5, 8, requires_grad=True)
    valid = torch.tensor([True, True, True, True, False]).expand(length, -1)
    task = torch.tensor([False, False, True, True, False]).expand(length, -1)
    return ([response], [torch.arange(length) * 5], torch.randn(3, 8),
            torch.tensor([True, True, False]), [visual], [valid], [task])


def unlock(model):
    with torch.no_grad():
        for group in model.decoder.groups:
            group.a_factors.normal_(std=.03)
            group.b_factors.normal_(std=.03)


@pytest.mark.parametrize("mode", ["past_relation", "frame_set"])
def test_complete_output_learns_from_native_response_and_visual_semantics(mode):
    model, values = writer(mode), inputs()
    initial = model(*values)
    assert all(torch.count_nonzero(tensor) == 0 for name, tensor in initial.items() if name.endswith("lora_B.weight"))
    unlock(model)
    state = model(*values)
    sum(tensor.square().sum() for tensor in state.values()).backward()
    for value in (values[0][0], values[4][0], model.semantic.projection.weight,
                  model.semantic_compiler.process_read.value.weight, model.input_projection.weight):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    assert not hasattr(model, "compiler")
    assert hasattr(model, "process_groups") == (mode == "past_relation")


def test_frame_set_consumes_all_frames_without_order_or_clock_dependence():
    model, values = writer("frame_set"), inputs(6)
    unlock(model)
    original = model(*values)
    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    changed = list(values)
    for index in (0, 4, 5, 6):
        changed[index] = [values[index][0][permutation]]
    changed[1] = [torch.arange(6) * 100 + 370]
    permuted = model(*changed)
    for name in original:
        torch.testing.assert_close(original[name], permuted[name], rtol=3e-5, atol=3e-6)
    # A different genuine image collection must still be available to the model.
    changed[4] = [changed[4][0] + torch.randn_like(changed[4][0])]
    different = model(*changed)
    assert any(not torch.allclose(original[name], different[name]) for name in original)


def test_centering_does_not_reintroduce_common_content_through_value_bias():
    compiler = writer("past_relation").semantic_compiler
    constant = torch.randn(1, 12).expand(5, -1).clone()
    key, value, prior = compiler.process_memory([constant], [torch.arange(5) * 5], ordered=True)
    assert compiler.process_read.value.bias is None and compiler.process_read.output.bias is None
    result = compiler.process_read(torch.randn(4, 12), key, value, prior)
    torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=1e-6)


def test_semantic_and_decoder_initialization_is_shared_without_rng_drift():
    torch.manual_seed(19)
    ordered = writer("past_relation")
    after_ordered = torch.rand(4)
    torch.manual_seed(19)
    unordered = writer("frame_set")
    after_unordered = torch.rand(4)
    torch.testing.assert_close(after_ordered, after_unordered)
    # Representative shared owners, rather than a full parameter scan.
    torch.testing.assert_close(ordered.semantic.projection.weight, unordered.semantic.projection.weight)
    torch.testing.assert_close(ordered.semantic_compiler.fusion[0].weight, unordered.semantic_compiler.fusion[0].weight)
    torch.testing.assert_close(ordered.decoder.a_code.weight, unordered.decoder.a_code.weight)
