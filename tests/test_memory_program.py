from __future__ import annotations

import pytest
import torch

from ember.writer.memory_program import DynamicKMemoryProgram, MemoryProgramError


def _offsets(lengths: tuple[int, ...]) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.long)


def _visual(frames: int, tokens: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(frames, tokens, 256), torch.ones(
        frames, tokens, dtype=torch.bool
    )


def test_memory_program_is_zero_preserving_and_k1_uses_the_set_path() -> None:
    torch.manual_seed(7)
    module = DynamicKMemoryProgram()
    memory = torch.randn(1, 18, 8, 1024).expand(3, -1, -1, -1).clone()
    visual = torch.zeros(3, 3, 256)
    valid = torch.ones(3, 3, dtype=torch.bool)
    output, diagnostics = module(
        memory,
        visual,
        valid,
        torch.tensor([0, 5, 10]),
        _offsets((3,)),
        _offsets((1,)),
    )

    assert output.shape == (1, 20, 8, 256)
    assert diagnostics.semantic_addresses.shape == (1, 18, 8, 256)
    assert diagnostics.visual_transition_readouts.shape == (3, 18, 8, 256)
    assert not diagnostics.visual_transition_readouts.count_nonzero()
    assert not diagnostics.visual_goal_readouts.count_nonzero()
    assert diagnostics.semantic_addresses.abs().sum().item() > 0.0
    assert diagnostics.video_programs.shape == (1, 18, 8, 256)
    assert torch.equal(output, torch.zeros_like(output))
    assert torch.equal(
        diagnostics.video_programs, torch.zeros_like(diagnostics.video_programs)
    )
    assert torch.equal(diagnostics.shared_program, diagnostics.singleton_program)
    assert diagnostics.consistency_loss.item() == 0.0
    assert all(
        layer.bias is None
        for layer in module.modules()
        if isinstance(layer, torch.nn.Linear)
    )


def test_absolute_memory_only_addresses_dynamic_procedure_queries() -> None:
    torch.manual_seed(9)
    module = DynamicKMemoryProgram().eval()
    memory = torch.randn(4, 18, 8, 1024)
    visual, valid = _visual(4)
    indices = torch.tensor([0, 5, 10, 15])
    video_offsets = _offsets((4,))
    condition_offsets = _offsets((1,))
    natural, diagnostics = module(
        memory, visual, valid, indices, video_offsets, condition_offsets
    )
    shifted, shifted_diagnostics = module(
        memory + 3.0, visual, valid, indices, video_offsets, condition_offsets
    )

    assert not torch.allclose(
        diagnostics.semantic_addresses,
        shifted_diagnostics.semantic_addresses,
    )
    assert not torch.allclose(natural, shifted)

    with torch.no_grad():
        module.semantic_address_projection.weight.zero_()
    unaddressed, _ = module(
        memory, visual, valid, indices, video_offsets, condition_offsets
    )
    shifted_unaddressed, _ = module(
        memory + 3.0, visual, valid, indices, video_offsets, condition_offsets
    )
    torch.testing.assert_close(unaddressed, shifted_unaddressed)

    with torch.no_grad():
        module.dynamic_projection.weight.zero_()
    static_only, static_diagnostics = module(
        memory,
        torch.zeros_like(visual),
        valid,
        indices,
        video_offsets,
        condition_offsets,
    )
    assert static_diagnostics.semantic_addresses.abs().sum().item() == 0.0
    assert torch.equal(static_only, torch.zeros_like(static_only))

    visual_only, visual_diagnostics = module(
        memory, visual, valid, indices, video_offsets, condition_offsets
    )
    assert visual_diagnostics.visual_transition_readouts.abs().sum().item() > 0.0
    assert visual_only.abs().sum().item() > 0.0


def test_memory_program_is_invariant_to_video_permutation() -> None:
    torch.manual_seed(11)
    module = DynamicKMemoryProgram().eval()
    memory = torch.randn(9, 18, 8, 1024)
    visual, valid = _visual(9)
    indices = torch.tensor([0, 5, 10] * 3)
    output, diagnostics = module(
        memory,
        visual,
        valid,
        indices,
        _offsets((3, 3, 3)),
        _offsets((3,)),
    )
    order = torch.tensor([2, 0, 1])
    permuted_memory = memory.reshape(3, 3, 18, 8, 1024)[order].flatten(0, 1)
    permuted_indices = indices.reshape(3, 3)[order].flatten()
    permuted, permuted_diagnostics = module(
        permuted_memory,
        visual.reshape(3, 3, 3, 256)[order].flatten(0, 1),
        valid.reshape(3, 3, 3)[order].flatten(0, 1),
        permuted_indices,
        _offsets((3, 3, 3)),
        _offsets((3,)),
    )

    torch.testing.assert_close(output, permuted, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        diagnostics.shared_program,
        permuted_diagnostics.shared_program,
        rtol=2e-5,
        atol=2e-6,
    )


def test_memory_program_reads_order_and_backpropagates_through_all_axes() -> None:
    torch.manual_seed(13)
    module = DynamicKMemoryProgram()
    memory = torch.randn(8, 18, 8, 1024, requires_grad=True)
    visual, valid = _visual(8)
    visual.requires_grad_()
    frame_indices = torch.tensor([0, 5, 10, 15, 0, 5, 10, 15])
    video_offsets = _offsets((4, 4))
    condition_offsets = _offsets((2,))
    natural, diagnostics = module(
        memory, visual, valid, frame_indices, video_offsets, condition_offsets
    )

    shuffled_order = torch.tensor([0, 2, 1, 3, 4, 6, 5, 7])
    shuffled, _ = module(
        memory[shuffled_order],
        visual[shuffled_order],
        valid[shuffled_order],
        frame_indices,
        video_offsets,
        condition_offsets,
    )
    reversed_order = torch.tensor([3, 2, 1, 0, 7, 6, 5, 4])
    reversed_program, _ = module(
        memory[reversed_order],
        visual[reversed_order],
        valid[reversed_order],
        frame_indices,
        video_offsets,
        condition_offsets,
    )
    assert not torch.allclose(natural, shuffled)
    assert not torch.allclose(natural, reversed_program)
    assert diagnostics.consistency_loss.item() > 0.0

    loss = natural.square().mean() + diagnostics.consistency_loss
    loss.backward()
    assert memory.grad is not None and memory.grad.abs().sum().item() > 0.0
    assert visual.grad is not None and visual.grad.abs().sum().item() > 0.0
    for parameter in (
        module.dynamic_projection.weight,
        module.semantic_address_projection.weight,
        module.visual_reader.query.weight,
        module.visual_reader.key.weight,
        module.visual_reader.output.weight,
        module.temporal_blocks[0].value.weight,
        module.set_blocks[0].value.weight,
        module.endpoint_reader.value.weight,
        module.layer_axis.value.weight,
        module.rank_axis.value.weight,
    ):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum().item() > 0.0


@pytest.mark.parametrize(
    "frame_indices",
    (torch.tensor([5, 10, 15]), torch.tensor([0, 5, 5])),
)
def test_memory_program_rejects_invalid_per_video_ordinals(
    frame_indices: torch.Tensor,
) -> None:
    module = DynamicKMemoryProgram()
    memory = torch.randn(3, 18, 8, 1024)
    visual, valid = _visual(3)

    with pytest.raises(MemoryProgramError, match="start at zero and increase"):
        module(
            memory,
            visual,
            valid,
            frame_indices,
            _offsets((3,)),
            _offsets((1,)),
        )
