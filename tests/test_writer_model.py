from __future__ import annotations

import torch

from fixtures.writer_model import _inputs, _model


def test_dynamic_k_writer_starts_at_exact_complete_rank8_identity() -> None:
    model, template = _model()
    output = model(*_inputs(), policy=torch.nn.Identity())
    assert set(output) == set(template)
    for name, value in output.items():
        assert value.shape == (2, *template[name].shape)
        assert torch.equal(value[0], template[name])
        assert torch.equal(value[1], template[name])


def test_dynamic_k_writer_program_is_video_ordered_and_set_invariant() -> None:
    model, _ = _model()
    inputs = _inputs()
    natural = model.encode_program(*inputs, policy=torch.nn.Identity())

    frames, indices, video_offsets, condition_offsets, *language = inputs
    video_order = torch.tensor([3, 4, 0, 1, 2, 5, 6, 7])
    permuted_video_offsets = torch.tensor([0, 2, 5, 8], dtype=torch.long)
    permuted = model.encode_program(
        frames[video_order],
        torch.tensor([0, 5, 0, 5, 10, 0, 5, 10]),
        permuted_video_offsets,
        condition_offsets,
        *language,
        policy=torch.nn.Identity(),
    )
    torch.testing.assert_close(
        natural.program, permuted.program, rtol=2e-5, atol=3e-5
    )

    reversed_frames = frames.clone()
    reversed_frames[:3] = frames[:3].flip(0)
    reversed_program = model.encode_program(
        reversed_frames,
        indices,
        video_offsets,
        condition_offsets,
        *language,
        policy=torch.nn.Identity(),
    )
    assert not torch.allclose(natural.program, reversed_program.program)


def test_dynamic_k_writer_training_consistency_and_mapper_gradient_staging() -> None:
    model, _ = _model()
    generated, consistency = model.forward_training(
        *_inputs(), policy=torch.nn.Identity()
    )
    assert consistency.ndim == 0 and consistency.item() > 0
    sum(value.float().sum() for value in generated.values()).backward()
    assert model.lora_mapper.families["q"].b.weight.grad is not None
    assert model.lora_mapper.families["q"].b.weight.grad.abs().sum() > 0
    assert model.lora_mapper.families["q"].a.weight.grad is None
    assert model.memory_program.dynamic_projection.weight.grad is not None
    assert not model.memory_program.dynamic_projection.weight.grad.count_nonzero()

    model.zero_grad(set_to_none=True)
    torch.nn.init.normal_(model.lora_mapper.families["q"].b.weight, std=0.01)
    torch.nn.init.normal_(model.lora_mapper.families["v"].b.weight, std=0.01)
    generated, consistency = model.forward_training(
        *_inputs(), policy=torch.nn.Identity()
    )
    (sum(value.float().sum() for value in generated.values()) + consistency).backward()
    assert model.memory_program.dynamic_projection.weight.grad is not None
    assert model.memory_program.dynamic_projection.weight.grad.abs().sum() > 0


def test_k1_uses_same_graph_and_has_exact_zero_consistency() -> None:
    model, _ = _model()
    frames, indices, video_offsets, _, tokens, masks, spans = _inputs()
    generated, consistency = model.forward_training(
        frames[5:],
        indices[5:],
        torch.tensor([0, 3]),
        torch.tensor([0, 1]),
        tokens[1:],
        masks[1:],
        spans[1:],
        policy=torch.nn.Identity(),
    )
    assert consistency.item() == 0.0
    assert all(value.shape == model.template_state()[name].shape for name, value in generated.items())
