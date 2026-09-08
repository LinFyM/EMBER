from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRATarget, identity_lora_state, validate_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.writer.horizon import HorizonRelationWriter, HorizonWriterConfig
from ember.writer.relation import past_edges, relative_correspondence


@pytest.fixture(autouse=True)
def _small_cpu_work():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(32)
    yield
    torch.set_num_threads(previous)


def _contract(full=False):
    contract = load_pi05_lora_contract(Path(__file__).resolve().parents[1] / "configs/pi05_lora_v1.json")
    return contract if full else replace(contract, targets=(LoRATarget("left", 3, 4), LoRATarget("right", 3, 4),
                                                           LoRATarget("input", 2, 5)), rank=2, alpha=2)


def _config(**kwargs):
    return replace(HorizonWriterConfig(width=12, heads=3, horizon=4, native_width=6, language_width=8,
                                       factor_width=5, edge_chunk=2, activation_checkpoint=False), **kwargs)


def _input(length, config):
    response = torch.randn(length, config.horizon, config.native_width)
    times = torch.arange(length) * 5
    if length > 1:
        times[-1] -= 2
    visual = torch.randn(length, 6, config.language_width)
    mask = torch.tensor([True, True, True, True, False, False]).expand(length, -1)
    return response, times, visual, mask


def _language(config):
    return torch.randn(5, config.language_width), torch.tensor([True, True, True, False, False])


def _call(writer, inputs, language):
    responses, times, visuals, masks = map(list, zip(*inputs, strict=True))
    return writer(responses, times, *language, visuals, masks, [_task_mask(mask) for mask in masks])


def _task_mask(mask):
    result = mask.clone()
    result[:, 0] = False
    return result


def _frame_language(writer, visual, mask, language_mask):
    return writer.contextual_language(visual, _task_mask(mask), language_mask)


def _unlock(writer):
    with torch.no_grad():
        for group in writer.decoder.groups:
            group.a_factors.normal_(std=0.03)
            group.b_factors.normal_(std=0.03)


def test_complete_h_query_has_cross_row_relation_dependence():
    config = _config()
    block = HorizonRelationWriter(_contract(), config).process_groups[0].local
    current, values = torch.randn(2, 4, 12), torch.randn(2, 3, 4, 4)
    pi = torch.randn(2, 3, 4, 5).softmax(-1)[..., :-1].detach().requires_grad_()
    query, _, _, _ = block.form_query(current, values, pi, torch.tensor([5., 8.]),
                                      torch.randn(4, 12), torch.randn(2, 12))
    gradient = torch.autograd.grad(query[0, 0].square().sum(), pi)[0]
    assert gradient[0, :, 1:].abs().sum() > 0
    assert gradient[1].abs().sum() == 0
    assert not block.horizon_query.causal


def test_null_mass_relative_distribution_and_true_gap_features():
    config = _config()
    block = HorizonRelationWriter(_contract(), config).process_groups[0].local
    current = torch.randn(2, 4, 12)
    late, early = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
    gaps = torch.tensor([3., 10.])
    observed = []
    handle = block.bias.register_forward_pre_hook(lambda _, args: observed.append(args[0]))
    pi = block.correspondence(current, late, early, gaps)
    handle.remove()
    assert (pi.sum(-1) < 1).all()
    rho = relative_correspondence(pi)
    torch.testing.assert_close(rho.sum(-1), pi.sum(-1))
    offsets = block.offsets + config.horizon - 1
    torch.testing.assert_close(rho.gather(-1, offsets.expand_as(pi)), pi)
    torch.testing.assert_close(observed[0][..., 0], (gaps[:, None, None] / 5).expand(2, 4, 4))
    torch.testing.assert_close(observed[0][..., 1], (block.offsets - gaps[:, None, None]) / 4)
    assert block.bias[-1].weight.count_nonzero() == 0
    assert block.bias[-1].bias.count_nonzero() == 0
    # Force almost all null mass: relative/matched reads must retain the tiny
    # nonempty mass, rather than renormalizing to a confident correspondence.
    with torch.no_grad():
        block.null[-1].weight.zero_()
        block.null[-1].bias.fill_(30)
    assert block.correspondence(current, late, early, gaps).sum(-1).max() < 1e-10
    _, matched, relative, mass = block.form_query(current, early, torch.zeros_like(pi), gaps,
                                                  torch.randn(4, 12), torch.randn(2, 12))
    assert matched.count_nonzero() == relative.count_nonzero() == mass.count_nonzero() == 0


def test_four_groups_three_writebacks_and_future_independence():
    config = _config()
    writer = HorizonRelationWriter(_contract(), config)
    response, times, visual, mask = _input(8, config)
    language_mask = _language(config)[1]
    language = _frame_language(writer, visual, mask, language_mask)
    original = writer.encode_video(response, times, language, visual, mask)
    assert len(writer.process_groups) == 4
    assert sum(group.writeback is not None for group in writer.process_groups) == 3
    assert all(group.temporal.causal for group in writer.process_groups)
    response[5:] = torch.randn_like(response[5:]) * 7
    visual[5:] = torch.randn_like(visual[5:]) * 7
    language = _frame_language(writer, visual, mask, language_mask)
    changed = writer.encode_video(response, times, language, visual, mask)
    torch.testing.assert_close(original[:5], changed[:5], rtol=0, atol=0)
    assert not torch.allclose(original[5:], changed[5:])
    # The whole four-group graph also equals a separately encoded prefix.
    prefix = writer.encode_video(response[:5], times[:5], language[:5], visual[:5], mask[:5])
    torch.testing.assert_close(original[:5], prefix, rtol=2e-5, atol=2e-6)


def test_local_edges_are_synchronous_and_visual_projections_are_reused():
    config = _config(radius=1)
    writer = HorizonRelationWriter(_contract(), config)
    block = writer.process_groups[0].local
    states, times = torch.randn(5, 4, 12), torch.arange(5).float() * 5
    visual, mask, language = torch.randn(5, 6, 8), torch.ones(5, 6, dtype=torch.bool), torch.randn(5, 12)
    counts = [0, 0]
    def count(index):
        def hook(*_):
            counts[index] += 1
        return hook
    handles = [block.visual_read.key.register_forward_hook(count(0)), block.visual_read.value.register_forward_hook(count(1))]
    original = block(states, times, language, visual, mask, writer.horizon_embedding)
    for handle in handles:
        handle.remove()
    assert counts == [1, 1]
    modified = states.clone()
    modified[0] *= 9
    changed = block(modified, times, language, visual, mask, writer.horizon_embedding)
    torch.testing.assert_close(original[2:], changed[2:], rtol=0, atol=0)
    assert not torch.allclose(original[1], changed[1])
    # Both real visual endpoints influence messages; masks remove padding.
    for endpoint in (0, 1):
        modified_visual = visual.clone()
        modified_visual[endpoint] *= 9
        assert not torch.allclose(original[1], block(states, times, language, modified_visual, mask, writer.horizon_embedding)[1])


def test_gru_uses_ordered_real_messages_and_correct_spacing():
    config = _config()
    block = HorizonRelationWriter(_contract(), config).process_groups[0].local
    states, times = torch.randn(6, 4, 12), torch.tensor([0., 5., 10., 15., 20., 23.])
    current, past, slots = past_edges(6, 4, states.device)
    assert past[current == 5].tolist() == [1, 2, 3, 4]
    messages = torch.randn(len(current), 4, 12)
    actual = block.aggregate(states, times, messages, current, past, slots)
    initial = block.initial(block.initial_norm(states)).tanh()
    final = []
    for t in range(len(states)):
        hidden = initial[t]
        edge_ids = (current == t).nonzero().flatten().tolist()
        for ordinal, edge in enumerate(edge_ids):
            u = past[edge]
            delta = (times[u] - times[past[edge_ids[ordinal - 1]]]) / 5 if ordinal else 0
            gamma = torch.tensor([(times[t] - times[u]) / 5, delta, float(ordinal > 0)])
            inputs = block.message_norm(messages[edge]) + block.time_input(gamma)
            hidden = block.gru(inputs, hidden)
        final.append(hidden)
    expected = states + block.neighbor_output(torch.stack(final) - initial)
    expected = expected + block.ffn(block.ffn_norm(expected))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[0], states[0] + block.ffn(block.ffn_norm(states[0])))
    swapped = messages.clone()
    indices = (current == 5).nonzero().flatten()
    swapped[indices] = swapped[indices.flip(0)]
    assert not torch.allclose(actual[5], block.aggregate(states, times, swapped, current, past, slots)[5])


def test_writeback_depends_on_each_h_and_language_positions_and_masks():
    writer = HorizonRelationWriter(_contract(), _config())
    block = writer.process_groups[0].writeback
    states, process = torch.randn(2, 4, 12), torch.randn(2, 12)
    update = block(states, process) - states
    assert not torch.allclose(update[:, 0], update[:, 1])
    embeddings, mask = _language(writer.config)
    original = writer.encode_language(embeddings, mask)
    embeddings[~mask] = 1000
    torch.testing.assert_close(original, writer.encode_language(embeddings, mask))
    permuted = embeddings.clone()
    permuted[:3] = embeddings[:3].flip(0)
    assert not torch.allclose(original, writer.encode_language(permuted, mask))


@pytest.mark.parametrize("cardinality", [1, 2, 4])
def test_variable_video_set_is_permutation_invariant(cardinality):
    writer = HorizonRelationWriter(_contract(), _config())
    _unlock(writer)
    inputs = [_input(n, writer.config) for n in (1, 3, 5, 2)[:cardinality]]
    language = _language(writer.config)
    original, reverse = _call(writer, inputs, language), _call(writer, inputs[::-1], language)
    for name in original:
        torch.testing.assert_close(original[name], reverse[name], rtol=2e-5, atol=2e-6)
    response, times, visual, mask = inputs[-1]
    poisoned = visual.clone()
    poisoned[~mask] = 1000
    clean = writer.encode_video(response, times, _frame_language(writer, visual, mask, language[1]), visual, mask)
    masked = writer.encode_video(response, times, _frame_language(writer, poisoned, mask, language[1]), poisoned, mask)
    torch.testing.assert_close(clean, masked)


def test_memory_prior_equalizes_video_mass_and_excludes_routing_from_values():
    writer = HorizonRelationWriter(_contract(), _config())
    videos = [torch.randn(n, 12) for n in (2, 5)]
    times = [torch.arange(len(v)) * 5 for v in videos]
    memory, route, prior = writer._memory(videos, times)
    torch.testing.assert_close(prior.softmax(-1)[0, :2].sum(), torch.tensor(.5))
    torch.testing.assert_close(prior.softmax(-1)[0, 2:].sum(), torch.tensor(.5))
    later, later_route, _ = writer._memory(videos, [t + 17 for t in times])
    torch.testing.assert_close(memory, later)
    assert not torch.allclose(route, later_route)
    observed = []
    handle = writer.compiler[0].cross.register_forward_pre_hook(lambda _, args: observed.append(args[2]))
    writer.compile(videos, times, torch.randn(12))
    handle.remove()
    torch.testing.assert_close(observed[0], writer.compiler[0].memory_norm(memory))


def test_compiler_language_can_route_but_cannot_supply_residual_content():
    writer = HorizonRelationWriter(_contract(), _config(activation_checkpoint=False))
    language = torch.randn(12)
    # With a single real memory value, attention cannot select different
    # content. A language-dependent result would reveal a residual bypass.
    one_value, one_time = [torch.randn(1, 12)], [torch.tensor([0])]
    first = writer.compile(one_value, one_time, language)
    second = writer.compile(one_value, one_time, -language)
    torch.testing.assert_close(first, second)

    # Multiple distinct values make language-guided lookup meaningful, with
    # gradients to both the existing query projection and video memory.
    memory = torch.randn(5, 12, requires_grad=True)
    output = writer.compile([memory], [torch.arange(5) * 5], language)
    gradient = torch.autograd.grad(output.square().sum(), (writer.query_language.weight, memory))
    assert all(value.abs().sum() > 0 for value in gradient)


@pytest.mark.parametrize("mode", [None, "language_residual_v1", "first_query_only_v1"])
def test_runtime_rejects_unmarked_or_old_compiler_before_loading_assets(tmp_path, mode):
    from ember.writer.runtime import build_runtime

    model = {} if mode is None else {"compiler_language_mode": mode}
    with pytest.raises(ValueError, match="architecture identity"):
        build_runtime(tmp_path, {"model": model}, torch.device("cpu"))


def test_identity_output_covers_all_native_targets_and_independent_rank_decoders():
    writer = HorizonRelationWriter(_contract(True), _config())
    generated = _call(writer, [_input(1, writer.config)], _language(writer.config))
    validate_lora_state(generated, writer.contract)
    assert len(generated) == 76
    assert sum(value.numel() for value in generated.values()) == 1_287_168
    for name, value in identity_lora_state(writer.contract).items():
        torch.testing.assert_close(generated[name], value, rtol=0, atol=0)
    writer = HorizonRelationWriter(_contract(), _config())
    code = torch.randn(3, 2, 12)
    initial = writer.decoder(code)
    optimizer = torch.optim.SGD(writer.decoder.parameters(), lr=.1)
    loss = F.mse_loss(initial['right' + LORA_B_SUFFIX][:, 1], torch.ones(4))
    loss.backward()
    optimizer.step()
    updated = writer.decoder(code)
    assert F.mse_loss(updated['right' + LORA_B_SUFFIX][:, 1], torch.ones(4)) < loss
    for name in initial:
        if name == 'right' + LORA_B_SUFFIX:
            torch.testing.assert_close(initial[name][:, 0], updated[name][:, 0], rtol=0, atol=0)
        else:
            torch.testing.assert_close(initial[name], updated[name], rtol=0, atol=0)


def _functional_loss(generated, contract, inputs, targets):
    return torch.stack([F.mse_loss(F.linear(F.linear(x, generated[t.name + LORA_A_SUFFIX]),
                                          generated[t.name + LORA_B_SUFFIX]), y)
                        for t, x, y in zip(contract.targets, inputs, targets, strict=True)]).mean()


@pytest.mark.parametrize("checkpointed", [False, True])
def test_identity_then_functional_update_reaches_complete_graph(checkpointed):
    writer = HorizonRelationWriter(_contract(), _config(activation_checkpoint=checkpointed))
    response, times, visual, mask = _input(5, writer.config)
    response.requires_grad_()
    visual.requires_grad_()
    condition, language = [(response, times, visual, mask)], _language(writer.config)
    inputs = [torch.randn(5, t.in_features) for t in writer.contract.targets]
    targets = [torch.randn(5, t.out_features) for t in writer.contract.targets]
    optimizer = torch.optim.SGD(writer.parameters(), lr=.02)
    _functional_loss(_call(writer, condition, language), writer.contract, inputs, targets).backward()
    assert writer.decoder.groups[0].b_factors.grad.abs().sum() > 0
    assert writer.input_projection.weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    response.grad, visual.grad = None, None
    _functional_loss(_call(writer, condition, language), writer.contract, inputs, targets).backward()
    required = [writer.input_projection.weight, writer.language_input.weight,
                writer.compiler[0].cross.value.weight]
    for group in writer.process_groups:
        required.extend([group.local.content.weight, group.local.relative_read.weight,
                         group.local.horizon_query.attention.query.weight, group.local.visual_read.key.weight,
                         group.local.visual_read.value.weight, group.local.gru.weight_hh,
                         group.horizon_read.value.weight, group.temporal.attention.query.weight])
        if group.writeback is not None:
            required.append(group.writeback.process.weight)
    # A-side U is still blocked after only B-D's first update; the BA loss must
    # now reach A-D, then its U projection becomes learnable on the next update.
    for parameter in required:
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert writer.decoder.groups[0].a_factors.grad.abs().sum() > 0
    assert response.grad.abs().sum() > 0 and visual.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _functional_loss(_call(writer, condition, language), writer.contract, inputs, targets).backward()
    assert writer.decoder.a_code.weight.grad.abs().sum() > 0
    assert writer.decoder.b_code.weight.grad.abs().sum() > 0


def test_dense_chunk_checkpoint_outputs_and_vjp_agree():
    dense = HorizonRelationWriter(_contract(), _config(edge_chunk=100))
    chunked = HorizonRelationWriter(_contract(), _config(activation_checkpoint=True))
    _unlock(dense)
    chunked.load_state_dict(dense.state_dict())
    response, times, visual, mask = _input(5, dense.config)
    left, right = response.requires_grad_(), response.detach().clone().requires_grad_()
    language = _language(dense.config)
    lhs = _call(dense, [(left, times, visual, mask)], language)
    rhs = _call(chunked, [(right, times, visual, mask)], language)
    vectors = {name: torch.randn_like(value) for name, value in lhs.items()}
    for name in lhs:
        torch.testing.assert_close(lhs[name], rhs[name], rtol=2e-5, atol=2e-6)
    loss1, loss2 = [sum((state[n] * vectors[n]).sum() for n in state) for state in (lhs, rhs)]
    grad1 = torch.autograd.grad(loss1, (left, *dense.parameters()), allow_unused=True)
    grad2 = torch.autograd.grad(loss2, (right, *chunked.parameters()), allow_unused=True)
    for a, b in zip(grad1, grad2, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=5e-4, atol=2e-6)


def test_invalid_inputs_are_rejected():
    writer = HorizonRelationWriter(_contract(), _config())
    embeddings, mask = _language(writer.config)
    with pytest.raises(ValueError, match="valid token"):
        writer.encode_language(embeddings, torch.zeros_like(mask))
    response, times, visual, vmask = _input(3, writer.config)
    with pytest.raises(ValueError, match="increasing"):
        _call(writer, [(response, times.flip(0), visual, vmask)], (embeddings, mask))
    with pytest.raises(ValueError, match="contextual task tokens"):
        _call(writer, [(response, times, visual, torch.zeros_like(vmask))], (embeddings, mask))
    with pytest.raises(ValueError, match="one or more"):
        writer([], [], embeddings, mask, [], [], [])


def test_bfloat16_preserves_real_frame_gaps_before_feature_cast():
    writer = HorizonRelationWriter(_contract(), _config()).bfloat16()
    response, _, visual, mask = _input(3, writer.config)
    times = torch.tensor([510, 515, 518])
    embeddings, lmask = _language(writer.config)
    language = _frame_language(writer, visual.bfloat16(), mask, lmask)
    seen = []
    hook = writer.process_groups[0].register_forward_pre_hook(lambda _, args: seen.append(args[1]))
    output = writer.encode_video(response.bfloat16(), times, language, visual.bfloat16(), mask)
    hook.remove()
    assert torch.isfinite(output).all()
    assert seen[0].dtype == torch.float32
    torch.testing.assert_close(seen[0], times.float(), rtol=0, atol=0)


def test_contextual_language_reads_exact_span_per_frame_with_live_gradient():
    writer = HorizonRelationWriter(_contract(), _config())
    _, _, visual, mask = _input(4, writer.config)
    language_mask = _language(writer.config)[1]
    task_mask = _task_mask(mask)
    visual.requires_grad_()
    code = writer.contextual_language(visual, task_mask, language_mask)
    assert code.shape == (4, writer.config.width)
    changed = visual.detach().clone()
    changed[~task_mask] = 1000
    changed[3, task_mask[3]] *= 7
    other = writer.contextual_language(changed, task_mask, language_mask)
    torch.testing.assert_close(code[:3], other[:3])
    assert not torch.allclose(code[3], other[3])
    gradient = torch.autograd.grad(code[:2].square().sum(), visual)[0]
    assert gradient[:2][task_mask[:2]].abs().sum() > 0
    assert gradient[~task_mask].count_nonzero() == gradient[2:].count_nonzero() == 0
    with pytest.raises(ValueError, match="exactly its contextual task tokens"):
        writer.contextual_language(visual, mask, language_mask)


def test_static_embedding_content_only_conditions_the_compiler_lookup():
    writer = HorizonRelationWriter(_contract(), _config())
    _unlock(writer)
    inputs, language = [_input(4, writer.config)], _language(writer.config)
    observed = []
    hook = writer.process_groups[-1].register_forward_hook(lambda _, args, output: observed.append(output[1]))
    first = _call(writer, inputs, language)
    second = _call(writer, inputs, (language[0] * -9, language[1]))
    hook.remove()
    torch.testing.assert_close(observed[0], observed[1])
    assert any(not torch.allclose(first[name], second[name]) for name in first)
