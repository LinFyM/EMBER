from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRATarget, identity_lora_state, validate_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.writer.layered import LayeredRelationWriter, LayeredWriterConfig
from ember.writer.relation import LocalRelationBlock, directional_correspondence, relative_correspondence


@pytest.fixture(autouse=True)
def _small_cpu_work() -> None:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(32)
    yield
    torch.set_num_threads(previous)


def _contract(full: bool = False):
    contract = load_pi05_lora_contract(Path(__file__).resolve().parents[1] / "configs/pi05_lora_v1.json")
    if full:
        return contract
    return replace(contract, targets=(LoRATarget("left", 3, 4), LoRATarget("right", 3, 4),
                                      LoRATarget("input", 2, 5), LoRATarget("output", 5, 2)),
                   rank=2, alpha=2)


def _config(**kwargs) -> LayeredWriterConfig:
    return replace(LayeredWriterConfig(width=12, heads=3, layers=2, horizon=4,
                                      native_width=6, language_width=8, blocks=2, radius=1,
                                      coordinate_width=5, edge_chunk=2, coordinate_chunk=3,
                                      activation_checkpoint=False), **kwargs)


def _input(length: int, config: LayeredWriterConfig):
    response = torch.randn(length, config.layers, config.horizon, config.native_width)
    indices = torch.arange(length) * 5
    if length > 1:
        indices[-1] -= 2  # Exercise a real shortened final interval.
    return response, indices


def _language(config: LayeredWriterConfig):
    return torch.randn(5, config.language_width), torch.tensor([True, True, True, False, False])


def test_directional_row_normalization_and_relative_index_recovery() -> None:
    scores = torch.tensor([[0., 3., 1.], [0., 0., 2.], [2., 0., 0.]])
    late, early = directional_correspondence(scores)
    torch.testing.assert_close(late.sum(-1), torch.ones(3))
    torch.testing.assert_close(early.sum(-1), torch.ones(3))
    torch.testing.assert_close(early, scores.T.softmax(-1))
    assert not torch.allclose(early, late.T)
    attention = torch.stack((late, early))[None, None]
    rho = relative_correspondence(attention)
    h = torch.arange(3)
    offsets = h[None, :] - h[:, None] + 2
    recovered = rho.gather(-1, offsets.expand_as(attention))
    torch.testing.assert_close(recovered, attention)
    torch.testing.assert_close(rho.sum(-1), torch.ones_like(rho.sum(-1)))


def test_rho_mlp_first_projection_equals_relative_position_value_read() -> None:
    block = LocalRelationBlock(12, 3, 4, 1, 2, False)
    attention = torch.randn(2, 2, 3, 4, 4).softmax(-1)
    offsets = block.offsets + 3
    position_table = block.message.relative[:, offsets]
    explicit = torch.einsum("ejahg,ahgd->ejhd", attention, position_table)
    torch.testing.assert_close(block.message.relative_read(attention), explicit)


def test_pair_messages_share_one_score_and_reverse_signed_gap() -> None:
    block = LocalRelationBlock(12, 3, 4, 1, 2, False)
    x = torch.randn(3, 2, 4, 12)
    features, values = block._heads(block.content(x)), block._heads(block.value(x))
    late, early, gap = torch.tensor([2, 1]), torch.tensor([1, 0]), torch.tensor([3., 5.])
    received = []
    handle = block.message.register_forward_pre_hook(lambda _module, args: received.append(args))
    block._pair_messages(x, features, values, late, early, gap)
    handle.remove()
    score = block.chronological_score(features[late], features[early], gap)
    torch.testing.assert_close(received[0][2], score.softmax(-1))
    torch.testing.assert_close(received[1][2], score.transpose(-1, -2).softmax(-1))
    torch.testing.assert_close(received[0][3], gap)
    torch.testing.assert_close(received[1][3], -gap)


def test_local_blocks_update_synchronously_and_allow_both_neighbors() -> None:
    config = _config(blocks=2, radius=1)
    writer = LayeredRelationWriter(_contract(), config)
    responses, times = _input(9, config)
    language = writer.encode_language(*_language(config))
    original = writer.encode_video(responses, times, language)
    # t=4 reaches only [2,6] after two blocks. Sequential in-place propagation
    # in either traversal direction would violate one of these boundaries.
    for distant in (1, 7):
        outside = responses.clone()
        outside[distant] = torch.randn_like(outside[distant]) * 3
        changed = writer.encode_video(outside, times, language)
        torch.testing.assert_close(original[4], changed[4], rtol=0, atol=0)
    for neighbor in (3, 5):
        local = responses.clone()
        local[neighbor] = torch.randn_like(local[neighbor]) * 3
        assert not torch.allclose(original[4], writer.encode_video(local, times, language)[4])
    # Native layer identities are batch dimensions, never edges between j.
    other_layer = responses.clone()
    other_layer[:, 1] = torch.randn_like(other_layer[:, 1])
    torch.testing.assert_close(original[:, 0], writer.encode_video(other_layer, times, language)[:, 0])


def test_single_frame_has_zero_neighbor_update_and_finite_horizon_read() -> None:
    config = _config()
    writer = LayeredRelationWriter(_contract(), config)
    response, times = _input(1, config)
    block = writer.relation_blocks[0]
    states = torch.randn(1, config.layers, config.horizon, config.width)
    torch.testing.assert_close(block(states, times), states + block.ffn(block.ffn_norm(states)))
    output = writer.encode_video(response, times, writer.encode_language(*_language(config)))
    assert output.shape == (1, config.layers, config.width)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("cardinality", [1, 2, 4])
def test_variable_length_video_set_is_permutation_invariant(cardinality: int) -> None:
    config = _config()
    writer = LayeredRelationWriter(_contract(), config)
    with torch.no_grad():
        writer.decoder.a_readout.normal_(std=0.05)
        writer.decoder.b_readout.normal_(std=0.05)
    inputs = [_input(length, config) for length in (1, 3, 5, 2)[:cardinality]]
    videos, times = map(list, zip(*inputs, strict=True))
    embeddings, mask = _language(config)
    original = writer(videos, times, embeddings, mask)
    permuted = writer(videos[::-1], times[::-1], embeddings, mask)
    for name in original:
        torch.testing.assert_close(original[name], permuted[name], rtol=2e-5, atol=2e-6)
    poisoned = embeddings.clone()
    poisoned[~mask] = 1000
    torch.testing.assert_close(writer.encode_language(embeddings, mask), writer.encode_language(poisoned, mask))


def test_memory_prior_equalizes_video_base_mass_and_values_exclude_routing() -> None:
    config = _config()
    writer = LayeredRelationWriter(_contract(), config)
    videos = [torch.randn(length, config.layers, config.width) for length in (2, 5)]
    times = [torch.arange(len(video)) * 5 for video in videos]
    memory, routing, prior = writer._memory(videos, times)
    weights = prior.softmax(-1).squeeze(0)
    torch.testing.assert_close(weights[:4].sum(), torch.tensor(0.5))
    torch.testing.assert_close(weights[4:].sum(), torch.tensor(0.5))
    memory_later, routing_later, _ = writer._memory(videos, [time + 17 for time in times])
    torch.testing.assert_close(memory, memory_later, rtol=0, atol=0)
    assert not torch.allclose(routing, routing_later)
    observed_values = []
    hook = writer.compiler[0].cross.register_forward_pre_hook(lambda _module, args: observed_values.append(args[2]))
    writer.compile(videos, times, torch.randn(config.width))
    hook.remove()
    torch.testing.assert_close(observed_values[0], writer.compiler[0].memory_norm(memory))


def test_identity_output_covers_all_38_native_targets() -> None:
    config, contract = _config(blocks=0), _contract(full=True)
    writer = LayeredRelationWriter(contract, config)
    response, times = _input(1, config)
    generated = writer([response], [times], *_language(config))
    validate_lora_state(generated, contract)
    assert len(generated) == 76
    assert sum(value.numel() for value in generated.values()) == 1_287_168
    expected = identity_lora_state(contract)
    for name in generated:
        torch.testing.assert_close(generated[name], expected[name], rtol=0, atol=0)


def _functional_tensor_loss(generated, contract, inputs, targets):
    # A small, explicit linear execution function tests the chain-rule algebra;
    # this is a CPU contract test, not native policy or scientific evidence.
    losses = []
    for target, x, y in zip(contract.targets, inputs, targets, strict=True):
        a, b = generated[target.name + LORA_A_SUFFIX], generated[target.name + LORA_B_SUFFIX]
        losses.append(F.mse_loss(F.linear(F.linear(x, a), b), y))
    return torch.stack(losses).mean()


def test_initial_readout_update_is_local_to_one_target_and_rank() -> None:
    config, contract = _config(), _contract()
    writer = LayeredRelationWriter(contract, config)
    codes = torch.randn(len(contract.targets), contract.rank, config.width)
    optimizer = torch.optim.SGD(writer.decoder.parameters(), lr=0.1)
    initial = writer.decoder(codes)
    # The second target shares a native-shape group with the first. A scalar
    # loss on one B column must not update another rank or another target.
    key = "right" + LORA_B_SUFFIX
    loss = F.mse_loss(initial[key][:, 1], torch.ones(contract.targets[1].out_features))
    loss.backward()
    optimizer.step()
    updated = writer.decoder(codes)
    assert F.mse_loss(updated[key][:, 1], torch.ones(contract.targets[1].out_features)) < loss
    for name, value in initial.items():
        if name == key:
            torch.testing.assert_close(updated[name][:, 0], value[:, 0], rtol=0, atol=0)
        else:
            torch.testing.assert_close(updated[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_identity_start_then_functional_update_reaches_upstream(checkpointed: bool) -> None:
    config, contract = _config(activation_checkpoint=checkpointed), _contract()
    writer = LayeredRelationWriter(contract, config)
    response, times = _input(4, config)
    response.requires_grad_()
    language = _language(config)
    inputs = [torch.randn(5, target.in_features) for target in contract.targets]
    targets = [torch.randn(5, target.out_features) for target in contract.targets]
    optimizer = torch.optim.SGD(writer.parameters(), lr=0.02)
    initial = writer([response], [times], *language)
    _functional_tensor_loss(initial, contract, inputs, targets).backward()
    assert writer.decoder.b_readout.grad.abs().sum() > 0
    assert writer.input_projection.weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    response.grad = None
    generated = writer([response], [times], *language)
    _functional_tensor_loss(generated, contract, inputs, targets).backward()
    for parameter in (writer.input_projection.weight, writer.language_input.weight,
                      writer.relation_blocks[0].message.relative, writer.horizon_read.value.weight,
                      writer.compiler[0].cross.value.weight, writer.decoder.groups[0].b_coordinates):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert response.grad is not None and response.grad.abs().sum() > 0


def test_chunked_checkpoint_graph_matches_dense_edge_graph_and_vjp() -> None:
    config = _config(blocks=1)
    dense = LayeredRelationWriter(_contract(), replace(config, edge_chunk=100, coordinate_chunk=100))
    chunked = LayeredRelationWriter(_contract(), replace(config, activation_checkpoint=True))
    with torch.no_grad():
        dense.decoder.b_readout.normal_(std=0.03)
    chunked.load_state_dict(dense.state_dict())
    response, times = _input(4, config)
    left, right = response.requires_grad_(), response.detach().clone().requires_grad_()
    language = _language(config)
    dense_output = dense([left], [times], *language)
    chunked_output = chunked([right], [times], *language)
    key = "left" + LORA_B_SUFFIX
    torch.testing.assert_close(dense_output[key], chunked_output[key])
    vector = torch.randn_like(dense_output[key])
    dense_grad = torch.autograd.grad((dense_output[key] * vector).sum(), left)[0]
    chunked_grad = torch.autograd.grad((chunked_output[key] * vector).sum(), right)[0]
    torch.testing.assert_close(dense_grad, chunked_grad, rtol=2e-4, atol=2e-6)


def test_invalid_masks_and_unordered_times_are_rejected() -> None:
    config = _config()
    writer = LayeredRelationWriter(_contract(), config)
    embeddings, mask = _language(config)
    with pytest.raises(ValueError, match="valid token"):
        writer.encode_language(embeddings, torch.zeros_like(mask))
    response, times = _input(3, config)
    with pytest.raises(ValueError, match="increasing"):
        writer([response], [times.flip(0)], embeddings, mask)
    with pytest.raises(ValueError, match="one or more"):
        writer([], [], embeddings, mask)
