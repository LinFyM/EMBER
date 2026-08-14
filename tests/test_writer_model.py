from __future__ import annotations

import torch

from fixtures.legacy_v6_writer import _model as _legacy_v6_model
from fixtures.writer_model import _inputs, _model
from ember.writer.slot_set import PolicyProcedureCommonValueFusion


def _sum(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(value.float().sum() for value in state.values())


def test_procedure_common_value_is_raw_mean_under_uniform_attention() -> None:
    fusion = PolicyProcedureCommonValueFusion(width=4)
    with torch.no_grad():
        fusion.query.weight.zero_()
        fusion.key.weight.zero_()
        fusion.output.weight.copy_(torch.eye(4))
    slots = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    shared, diagnostics = fusion(slots, torch.tensor([0, 2]))
    common = slots.mean(dim=0)
    torch.testing.assert_close(diagnostics.shared_corrections[0], common)
    torch.testing.assert_close(shared[0], common + common)
    torch.testing.assert_close(diagnostics.attention[0], torch.full((2, 3), 0.5))

    singleton, singleton_diagnostics = fusion(slots[:1], torch.tensor([0, 1]))
    assert torch.equal(singleton, slots[:1])
    assert not singleton_diagnostics.shared_corrections.count_nonzero()


def test_native_compiler_public_stages_are_exact_old_graph() -> None:
    model, _ = _legacy_v6_model()
    compiler = model.compiler
    generator = torch.Generator(device="cpu").manual_seed(29)
    core = torch.randn(2, 4, 256, generator=generator)
    procedure = torch.randn(2, 5, 256, generator=generator)
    valid_core = torch.tensor([[True, True, True, False], [True] * 4])
    valid_procedure = torch.tensor([[True, True, True, False, False], [True] * 5])
    positions = torch.arange(5).expand(2, -1)

    routing = compiler.routing_norm(compiler._routing())[None].expand(2, -1, -1)
    core_slots = compiler.core_reader(routing, core, valid_core)
    procedure_slots, normalized_core, centered = compiler.procedure_reader(
        routing,
        core_slots,
        procedure,
        positions,
        valid_procedure,
    )
    gamma, beta = compiler.modulation(compiler.procedure_norm(procedure_slots)).chunk(
        2, dim=-1
    )
    fused = (1.0 + gamma) * normalized_core + beta
    expected = compiler.post_fusion(fused, routing)

    actual, diagnostics = compiler.fused_slots(
        core,
        valid_core,
        procedure,
        positions,
        valid_procedure,
    )
    assert torch.equal(actual, expected)
    assert torch.equal(diagnostics["core_slots"], core_slots)
    assert torch.equal(diagnostics["procedure_centered"], centered)
    assert torch.equal(diagnostics["procedure_slots"], procedure_slots)
    assert torch.equal(diagnostics["adaln_gamma"], gamma)
    assert torch.equal(diagnostics["adaln_beta"], beta)
    assert torch.equal(diagnostics["fused_slots"], fused)


def test_v6_layerwise_conditioner_step0_k1_is_exact_native_v6() -> None:
    model, _ = _model()
    frames, indices, _, _, tokens, masks, spans = _inputs()
    inputs = (
        frames[5:],
        indices[5:],
        torch.tensor([0, 3]),
        torch.tensor([0, 1]),
        tokens[1:],
        masks[1:],
        spans[1:],
    )
    encoded = model.encode_program(*inputs, policy=torch.nn.Identity())
    frames1, indices1, video_offsets1, _, tokens1, masks1, spans1 = inputs
    evidence = model.base_writer.encode_video_evidence(
        torch.nn.Identity(),
        frames1,
        video_offsets1,
        tokens1,
        masks1,
        spans1,
    )
    memories = model.base_writer.build_memories(evidence, indices1)
    native = model.base_writer.decode_slots(model.base_writer.compile_slots(memories))
    bridged = model(*inputs, policy=torch.nn.Identity())
    assert all(torch.equal(bridged[name], native[name]) for name in native)

    torch.nn.init.normal_(model.query_delta.weight, std=1.0)
    bridged_after = model(*inputs, policy=torch.nn.Identity())
    assert any(not torch.equal(bridged_after[name], native[name]) for name in native)


def test_v6_procedure_common_value_is_set_invariant_and_order_sensitive() -> None:
    model, _ = _model()
    inputs = _inputs()
    natural = model.encode_program(*inputs, policy=torch.nn.Identity())

    frames, _, _, condition_offsets, tokens, masks, spans = inputs
    video_order = torch.tensor([3, 4, 0, 1, 2, 5, 6, 7])
    permuted = model.encode_program(
        frames[video_order],
        torch.tensor([0, 5, 0, 5, 10, 0, 5, 10]),
        torch.tensor([0, 2, 5, 8]),
        condition_offsets,
        tokens,
        masks,
        spans,
        policy=torch.nn.Identity(),
    )
    torch.testing.assert_close(natural.program, permuted.program)

    reversed_frames = frames.clone()
    reversed_frames[:3] = frames[:3].flip(0)
    reversed_program = model.encode_program(
        reversed_frames,
        inputs[1],
        inputs[2],
        condition_offsets,
        tokens,
        masks,
        spans,
        policy=torch.nn.Identity(),
    )
    assert not torch.allclose(natural.program, reversed_program.program)


def test_v6_layerwise_conditioner_gradient_staging_and_freeze_boundary() -> None:
    model, _ = _model()
    generated, auxiliary = model.forward_training(
        *_inputs(), policy=torch.nn.Identity()
    )
    assert auxiliary.item() == 0.0
    _sum(generated).backward()
    assert model.query_delta.weight.grad is not None
    assert model.query_delta.weight.grad.abs().sum() > 0
    assert all(
        parameter.grad is None or not parameter.grad.count_nonzero()
        for parameter in model.probe_conditioner.parameters()
    )
    assert all(parameter.grad is None for parameter in model.base_writer.parameters())
    assert all(parameter.grad is None for parameter in model.procedure_set.parameters())

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.query_delta.weight.normal_(std=0.01)
    generated, _ = model.forward_training(*_inputs(), policy=torch.nn.Identity())
    _sum(generated).backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.probe_conditioner.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.layer_probe_reader.parameters()
    )


def test_cached_readouts_recompile_exactly_without_video_backbone() -> None:
    model, _ = _model()
    torch.nn.init.normal_(model.query_delta.weight, std=0.01)
    encoded = model.encode_program(*_inputs(), policy=torch.nn.Identity())
    cached = model.compile_readouts(
        encoded.diagnostics.shared_core_slots,
        encoded.diagnostics.per_video_procedure_slots,
        _inputs()[3],
        per_video_query_conditioners=(
            encoded.diagnostics.per_video_query_conditioners
        ),
        per_video_query_deltas=encoded.diagnostics.per_video_query_deltas,
    )
    torch.testing.assert_close(cached.program, encoded.program, rtol=0, atol=0)
    _sum(model.decode_program(cached.program)).backward()
    assert model.query_delta.weight.grad is not None
    assert model.query_delta.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.base_writer.parameters())
    assert all(parameter.grad is None for parameter in model.procedure_set.parameters())


def test_k1_uses_same_graph_and_has_exact_zero_auxiliary() -> None:
    model, _ = _model()
    frames, indices, _, _, tokens, masks, spans = _inputs()
    generated, auxiliary = model.forward_training(
        frames[5:],
        indices[5:],
        torch.tensor([0, 3]),
        torch.tensor([0, 1]),
        tokens[1:],
        masks[1:],
        spans[1:],
        policy=torch.nn.Identity(),
    )
    assert auxiliary.item() == 0.0
    assert all(
        value.shape == model.template_state()[name].shape
        for name, value in generated.items()
    )


def test_layerwise_conditioner_step0_is_as139_graph() -> None:
    model, _ = _model()
    inputs = _inputs()
    actual = model.encode_program(*inputs, policy=torch.nn.Identity()).program
    frames, indices, video_offsets, condition_offsets, tokens, masks, spans = inputs
    video_bounds = tuple(int(value) for value in video_offsets.tolist())
    condition_bounds = tuple(int(value) for value in condition_offsets.tolist())
    counts = torch.tensor(
        [right - left for left, right in zip(condition_bounds, condition_bounds[1:])]
    )
    condition_ids = torch.repeat_interleave(torch.arange(len(counts)), counts)
    evidence = model.base_writer.encode_video_evidence(
        torch.nn.Identity(),
        frames,
        video_offsets,
        tokens.index_select(0, condition_ids),
        masks.index_select(0, condition_ids),
        spans.index_select(0, condition_ids),
    )
    memories = model.base_writer.build_memories(evidence, indices)
    compiler = model.base_writer.compiler
    shared_core = []
    for left, right in zip(condition_bounds, condition_bounds[1:]):
        core = memories.core[left:right].reshape(1, -1, 256)
        valid = memories.valid_core[left:right].reshape(1, -1)
        routing = compiler.routing(1)
        slots = compiler.read_core_slots(routing, core, valid)
        shared_core.append(compiler.normalize_core_slots(slots)[0])
    shared_core = torch.stack(shared_core)
    routing = compiler.routing(len(video_bounds) - 1)
    procedure, _ = compiler.read_procedure_slots(
        routing,
        shared_core.index_select(0, condition_ids),
        memories.procedure,
        memories.positions,
        memories.valid_procedure,
    )
    mean_procedure = torch.stack(
        [
            procedure[left:right].mean(dim=0)
            for left, right in zip(condition_bounds, condition_bounds[1:])
        ]
    )
    expected = compiler.fuse_readouts(
        compiler.routing(len(shared_core)), shared_core, mean_procedure
    )[0]
    assert torch.equal(actual, expected)


def test_layerwise_conditioner_constant_video_has_zero_dynamic_value() -> None:
    model, _ = _model()
    frames, indices, offsets, condition_offsets, tokens, masks, spans = _inputs()
    frames[:] = frames[0]
    encoded = model.encode_program(
        frames,
        indices,
        offsets,
        condition_offsets,
        tokens,
        masks,
        spans,
        policy=torch.nn.Identity(),
    )
    assert not encoded.diagnostics.per_video_query_conditioners.count_nonzero()
