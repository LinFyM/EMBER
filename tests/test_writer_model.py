from __future__ import annotations

import torch

from fixtures.writer_model import _inputs, _model


def _sum(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(value.float().sum() for value in state.values())


def test_v6_slot_set_k1_is_exact_native_v6_before_and_after_training() -> None:
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
    native = model.base_writer.decode_slots(encoded.diagnostics.per_video_slots)
    bridged = model(*inputs, policy=torch.nn.Identity())
    assert all(torch.equal(bridged[name], native[name]) for name in native)

    torch.nn.init.normal_(model.slot_set.output.weight, std=0.01)
    bridged_after = model(*inputs, policy=torch.nn.Identity())
    assert all(torch.equal(bridged_after[name], native[name]) for name in native)


def test_v6_slot_set_is_video_invariant_and_order_sensitive() -> None:
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


def test_v6_slot_set_gradient_staging_and_freeze_boundary() -> None:
    model, _ = _model()
    generated, auxiliary = model.forward_training(
        *_inputs(), policy=torch.nn.Identity()
    )
    assert auxiliary.item() == 0.0
    _sum(generated).backward()
    assert model.slot_set.output.weight.grad is not None
    assert model.slot_set.output.weight.grad.abs().sum() > 0
    assert model.slot_set.query.weight.grad is not None
    assert not model.slot_set.query.weight.grad.count_nonzero()
    assert all(parameter.grad is None for parameter in model.base_writer.parameters())

    model.zero_grad(set_to_none=True)
    torch.nn.init.normal_(model.slot_set.output.weight, std=0.01)
    generated, _ = model.forward_training(*_inputs(), policy=torch.nn.Identity())
    _sum(generated).backward()
    assert model.slot_set.query.weight.grad is not None
    assert model.slot_set.query.weight.grad.abs().sum() > 0
    assert model.slot_set.key.weight.grad is not None
    assert model.slot_set.key.weight.grad.abs().sum() > 0
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
