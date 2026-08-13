from __future__ import annotations

import torch

from fixtures.legacy_v6_writer import _model as _legacy_v6_model
from fixtures.writer_model import _inputs, _model


def _sum(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(value.float().sum() for value in state.values())


def test_native_compiler_public_stages_are_exact_old_graph() -> None:
    model, _ = _legacy_v6_model()
    compiler = model.compiler
    generator = torch.Generator(device="cpu").manual_seed(29)
    core = torch.randn(2, 4, 256, generator=generator)
    procedure = torch.randn(2, 5, 256, generator=generator)
    valid_core = torch.tensor([[True, True, True, False], [True] * 4])
    valid_procedure = torch.tensor(
        [[True, True, True, False, False], [True] * 5]
    )
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
    gamma, beta = compiler.modulation(
        compiler.procedure_norm(procedure_slots)
    ).chunk(2, dim=-1)
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


def test_v6_memory_set_k1_is_exact_native_v6_before_and_after_training() -> None:
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

    torch.nn.init.normal_(model.procedure_set.output.weight, std=0.01)
    bridged_after = model(*inputs, policy=torch.nn.Identity())
    assert all(torch.equal(bridged_after[name], native[name]) for name in native)


def test_v6_memory_set_is_video_invariant_and_order_sensitive() -> None:
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


def test_v6_memory_set_gradient_staging_and_freeze_boundary() -> None:
    model, _ = _model()
    generated, auxiliary = model.forward_training(
        *_inputs(), policy=torch.nn.Identity()
    )
    assert auxiliary.item() == 0.0
    _sum(generated).backward()
    assert model.procedure_set.output.weight.grad is not None
    assert model.procedure_set.output.weight.grad.abs().sum() > 0
    assert model.procedure_set.query.weight.grad is not None
    assert not model.procedure_set.query.weight.grad.count_nonzero()
    assert all(parameter.grad is None for parameter in model.base_writer.parameters())

    model.zero_grad(set_to_none=True)
    torch.nn.init.normal_(model.procedure_set.output.weight, std=0.01)
    generated, _ = model.forward_training(*_inputs(), policy=torch.nn.Identity())
    _sum(generated).backward()
    assert model.procedure_set.query.weight.grad is not None
    assert model.procedure_set.query.weight.grad.abs().sum() > 0
    assert model.procedure_set.key.weight.grad is not None
    assert model.procedure_set.key.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.base_writer.parameters())


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
    assert all(value.shape == model.template_state()[name].shape for name, value in generated.items())
