"""Bounded synthetic CPU contracts for actual-parameter rank128 editing."""

from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from ember.experience_compiler.decoder import CoordinateDecoder
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX


NAMES = ("attention.q_proj", "action_out_proj")


def _mt_state():
    generator = torch.Generator(device="cpu").manual_seed(21)
    state = {}
    for name, in_width, out_width in zip(NAMES, (70, 65), (130, 32), strict=True):
        state[name + LORA_A_SUFFIX] = torch.randn(128, in_width, generator=generator) * 0.2
        state[name + LORA_B_SUFFIX] = torch.arange(out_width * 128).reshape(out_width, 128) / 10000
    return state


def _delta(seed=37, *, zero=False):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.zeros(len(NAMES), 128, 256) if zero else torch.randn(len(NAMES), 128, 256, generator=generator) * .1
    return values.requires_grad_()


def _functional_loss(state):
    loss = 0.
    for name in NAMES:
        a, b = state[name + LORA_A_SUFFIX], state[name + LORA_B_SUFFIX]
        inputs = torch.linspace(-1., 1., 3 * a.shape[1]).reshape(3, a.shape[1])
        prediction = F.linear(F.linear(inputs, a), b)
        loss = loss + (prediction - .2).square().mean()
    return loss


@pytest.mark.parametrize("autocast", (False, True))
def test_zero_edit_retains_actual_incoming_even_after_shared_parameter_changes(autocast):
    mt = _mt_state()
    incoming = {key: value * .7 + .03 for key, value in mt.items()}
    decoder = CoordinateDecoder(mt, NAMES, chunk_rows=127)
    units = decoder.matrix_rms.clone()
    assert decoder.coordinate_embeddings.shape[1] == 64
    assert decoder.output_projection.bias is None and decoder.edit_projection.bias is None
    assert [(name, buffer.numel()) for name, buffer in decoder.named_buffers() if buffer.is_floating_point()] == [
        ("matrix_rms", 2 * len(NAMES))]
    with torch.no_grad():
        for value in mt.values():
            value.zero_()
        decoder.coordinate_embeddings.add_(.13)
        decoder.input_projection.bias.add_(.5)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            output = decoder(incoming, _delta(zero=True))
    for key, value in output.items():
        assert value.shape == incoming[key].shape and value.is_contiguous()
        torch.testing.assert_close(value, incoming[key], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(decoder.matrix_rms, units)


def test_zero_edit_has_immediate_functional_derivative_without_incoming_or_common_drift():
    incoming = {key: value.requires_grad_() for key, value in _mt_state().items()}
    decoder = CoordinateDecoder(incoming, NAMES, chunk_rows=127)
    delta = _delta(zero=True)
    output = decoder(incoming, delta)
    _functional_loss(output).backward()
    assert delta.grad is not None and torch.isfinite(delta.grad).all() and delta.grad.norm() > 0
    assert all(value.grad is None for value in incoming.values())
    common_norm = sum(parameter.grad.square().sum() for parameter in decoder.parameters()).sqrt()
    assert common_norm < 1e-10
    decoder.zero_grad(set_to_none=True)
    _functional_loss(decoder(incoming, _delta())).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               and parameter.grad.norm() > 0 for parameter in decoder.parameters())


def test_B_column_block_matches_registered_formula_and_conditions_on_actual_values():
    mt, delta = _mt_state(), _delta()
    decoder = CoordinateDecoder(mt, NAMES)
    incoming = {key: value * .7 + .03 for key, value in mt.items()}
    key, rank, block = NAMES[0] + LORA_B_SUFFIX, 19, 1
    address = decoder.coordinate_embeddings[128 * 2 + rank * 3 + block]
    values, scale = incoming[key][64:128, rank], decoder.matrix_rms[1]
    with torch.no_grad():
        output = decoder(incoming, delta)
        base = decoder.input_projection(values / scale) + decoder.address_projection(address)
        edit = decoder.edit_projection(delta)[0, rank]
        expected = values + scale * decoder.output_projection(F.silu(base + edit) - F.silu(base))
        torch.testing.assert_close(output[key][64:128, rank], expected, atol=3e-7, rtol=1e-5)
        changed = dict(incoming)
        changed[key] = incoming[key] + scale
        another = decoder(changed, delta)
    first_edit, second_edit = output[key] - incoming[key], another[key] - changed[key]
    assert (first_edit - second_edit).norm() > 1e-5


def test_all38_targets_A_and_B_are_editable_and_condition_projection_runs_once():
    names = tuple(f"target{index}" for index in range(38))
    generator = torch.Generator().manual_seed(81)
    incoming = {name + suffix: torch.randn((128, 3) if suffix == LORA_A_SUFFIX else (32, 128), generator=generator) * .1
                for name in names for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX)}
    decoder = CoordinateDecoder(incoming, names, chunk_rows=97)
    calls = []
    hook = decoder.edit_projection.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        output = decoder(incoming, torch.randn(38, 128, 256, generator=generator) * .1)
    hook.remove()
    assert set(output) == set(incoming) and len(output) == 76 and calls == [1]
    assert all(value.shape == incoming[key].shape and (value - incoming[key]).norm() > 1e-6
               for key, value in output.items())


def test_physical_chunks_preserve_outputs_and_complete_functional_gradients():
    incoming = _mt_state()
    decoder = CoordinateDecoder(incoming, NAMES)
    chunked = deepcopy(decoder)
    whole_delta, chunked_delta = _delta(), _delta()
    whole = decoder(incoming, whole_delta, chunk_rows=100000)
    pieces = chunked(incoming, chunked_delta, chunk_rows=37)
    for key in whole:
        torch.testing.assert_close(whole[key], pieces[key], atol=3e-7, rtol=1e-5)
    _functional_loss(whole).backward()
    _functional_loss(pieces).backward()
    torch.testing.assert_close(whole_delta.grad, chunked_delta.grad, atol=1e-7, rtol=1e-4)
    first, second = tuple(decoder.parameters()), tuple(chunked.parameters())
    assert all(parameter.grad is not None for parameter in (*first, *second))
    first_grad = torch.cat([parameter.grad.flatten() for parameter in first])
    second_grad = torch.cat([parameter.grad.flatten() for parameter in second])
    assert torch.isfinite(first_grad).all() and torch.isfinite(second_grad).all()
    # Matrix sizes change floating-point reduction order; compare useful
    # gradient precision instead of enforcing low-bit equality near zero.
    assert (first_grad - second_grad).norm() / first_grad.norm() < 2e-5


def test_seeded_addresses_RMS_floor_and_narrow_shape_finite_guards_preserve_CPU_rng():
    mt = _mt_state()
    mt[NAMES[0] + LORA_B_SUFFIX].zero_()
    before = torch.random.get_rng_state().clone()
    decoder = CoordinateDecoder(mt, NAMES, seed=7)
    same, other = CoordinateDecoder(mt, NAMES, seed=7), CoordinateDecoder(mt, NAMES, seed=8)
    assert torch.equal(torch.random.get_rng_state(), before)
    torch.testing.assert_close(decoder.coordinate_embeddings, same.coordinate_embeddings)
    assert not torch.allclose(decoder.coordinate_embeddings, other.coordinate_embeddings)
    assert decoder.matrix_rms[1] == pytest.approx(1e-6)
    wrong = dict(mt)
    wrong[NAMES[0] + LORA_B_SUFFIX] = torch.zeros(130, 64)
    with pytest.raises(ValueError, match="rank128"):
        CoordinateDecoder(wrong, NAMES)
    with pytest.raises(ValueError, match="delta Q shape"):
        decoder(mt, torch.zeros(2, 64, 256))
    with pytest.raises(ValueError, match="complete target"):
        decoder({}, _delta())
    with pytest.raises(ValueError, match="incoming factor shape"):
        decoder(wrong, _delta())
    with pytest.raises(ValueError, match="positive"):
        decoder(mt, _delta(), chunk_rows=0)
    with pytest.raises(ValueError, match="finite"):
        decoder(mt, torch.full((2, 128, 256), torch.nan))
