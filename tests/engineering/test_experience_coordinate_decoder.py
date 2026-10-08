"""Synthetic CPU checks for the active complete rank128 coordinate decoder."""

from copy import deepcopy

import pytest
import torch

from ember.experience_compiler.decoder import CoordinateDecoder
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX


NAMES = ("attention.q_proj", "action_out_proj")


def _mt_state():
    generator = torch.Generator(device="cpu").manual_seed(21)
    state = {}
    for name, in_width, out_width in zip(NAMES, (70, 65), (130, 32), strict=True):
        state[name + LORA_A_SUFFIX] = torch.randn(128, in_width, generator=generator) * 0.2
        # Asymmetric entries detect a mistaken reshape of [out, rank] B.
        state[name + LORA_B_SUFFIX] = torch.arange(out_width * 128).reshape(out_width, 128) / 10000
    return state


def _query(seed=37):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(len(NAMES), 128, 256, generator=generator, requires_grad=True)


def test_complete_mt_identity_preserves_B_axis_and_partial_blocks():
    mt = _mt_state()
    expected = {key: value.clone() for key, value in mt.items()}
    decoder = CoordinateDecoder(mt, NAMES)
    assert all(parameter.requires_grad for parameter in decoder.parameters())
    assert not any(buffer.is_floating_point() for buffer in decoder.buffers())
    # The coordinates are copied values, not references or a persistent MT skip.
    for value in mt.values():
        value.zero_()
    with torch.no_grad():
        first = decoder(_query())
        second = decoder(_query(42))
    assert set(first) == set(expected)
    for key, value in first.items():
        assert value.shape == expected[key].shape and value.is_contiguous()
        torch.testing.assert_close(value, expected[key], atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(value, second[key], atol=0, rtol=0)
    b_key = NAMES[0] + LORA_B_SUFFIX
    assert first[b_key][67, 19] == pytest.approx(float(expected[b_key][67, 19]), abs=2e-7)


def test_initialization_is_seeded_without_consuming_global_cpu_rng():
    mt = _mt_state()
    before = torch.random.get_rng_state().clone()
    first = CoordinateDecoder(mt, NAMES, seed=7)
    torch.testing.assert_close(torch.random.get_rng_state(), before, atol=0, rtol=0)
    same = CoordinateDecoder(mt, NAMES, seed=7)
    different = CoordinateDecoder(mt, NAMES, seed=8)
    torch.testing.assert_close(first.coordinate_embeddings, same.coordinate_embeddings, atol=0, rtol=0)
    torch.testing.assert_close(first.input_projection.weight, same.input_projection.weight, atol=0, rtol=0)
    assert not torch.equal(first.coordinate_embeddings[:, 64:], different.coordinate_embeddings[:, 64:])
    torch.testing.assert_close(torch.random.get_rng_state(), before, atol=0, rtol=0)


def test_head_update_opens_real_Q_normalization_and_coordinate_gradients():
    decoder = CoordinateDecoder(_mt_state(), NAMES, chunk_rows=127)
    q = _query()
    target = {key: torch.zeros_like(value) for key, value in decoder(q).items()}

    def objective(state):
        return sum((value - target[key]).square().mean() for key, value in state.items())

    objective(decoder(q)).backward()
    assert q.grad is not None and q.grad.count_nonzero() == 0
    assert decoder.coordinate_embeddings.grad[:, :64].norm() > 0
    assert decoder.coordinate_embeddings.grad[:, 64:].count_nonzero() == 0
    optimizer = torch.optim.SGD(
        [*decoder.input_projection.parameters(), *decoder.output_projection.parameters()], lr=0.01)
    optimizer.step()
    decoder.zero_grad(set_to_none=True)
    q.grad = None
    updated = decoder(q)
    objective(updated).backward()
    for gradient in (q.grad, decoder.q_norm.weight.grad, decoder.q_norm.bias.grad,
                     decoder.input_projection.weight.grad, decoder.output_projection.weight.grad,
                     decoder.coordinate_embeddings.grad[:, 64:]):
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0
    with torch.no_grad():
        another = decoder(_query(42))
    assert any(not torch.allclose(updated[key], another[key]) for key in updated)


def test_physical_chunks_preserve_outputs_and_complete_gradients():
    decoder = CoordinateDecoder(_mt_state(), NAMES)
    generator = torch.Generator(device="cpu").manual_seed(53)
    with torch.no_grad():
        decoder.output_projection.weight[:, 128:].normal_(std=0.01, generator=generator)
    chunked = deepcopy(decoder)
    q_whole, q_chunked = _query(), _query()
    whole = decoder(q_whole, chunk_rows=100000)
    pieces = chunked(q_chunked, chunk_rows=37)
    for key in whole:
        torch.testing.assert_close(whole[key], pieces[key], atol=5e-7, rtol=5e-6)
    sum(value.square().mean() for value in whole.values()).backward()
    sum(value.square().mean() for value in pieces.values()).backward()
    torch.testing.assert_close(q_whole.grad, q_chunked.grad, atol=1e-7, rtol=5e-5)
    for parameter, other in zip(decoder.parameters(), chunked.parameters(), strict=True):
        assert parameter.grad is not None and other.grad is not None
        torch.testing.assert_close(parameter.grad, other.grad, atol=5e-7, rtol=5e-5)


def test_rejects_incompatible_rank_Q_and_invalid_chunk_size():
    mt = _mt_state()
    wrong = dict(mt)
    wrong[NAMES[0] + LORA_B_SUFFIX] = torch.zeros(130, 64)
    with pytest.raises(ValueError, match="rank128"):
        CoordinateDecoder(wrong, NAMES)
    decoder = CoordinateDecoder(mt, NAMES)
    with pytest.raises(ValueError, match="Q shape"):
        decoder(torch.zeros(2, 64, 256))
    with pytest.raises(ValueError, match="positive"):
        decoder(_query(), chunk_rows=0)
