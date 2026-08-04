from __future__ import annotations

import torch

from ember.writer.program_compiler import TargetOwnedFactorHead


def test_target_owned_factor_head_starts_at_exact_zero() -> None:
    torch.manual_seed(19)
    head = TargetOwnedFactorHead(128, 32, 16)
    value = torch.randn(2, 5, 128)
    assert not bool(head(value).count_nonzero())
    assert bool(head.input.weight.count_nonzero())
    assert not bool(head.output.weight.count_nonzero())


def test_target_owned_factor_heads_have_disjoint_parameters_and_gradients() -> None:
    torch.manual_seed(23)
    first = TargetOwnedFactorHead(12, 7, 5)
    second = TargetOwnedFactorHead(12, 7, 5)
    torch.nn.init.normal_(first.output.weight, std=0.01)
    torch.nn.init.normal_(second.output.weight, std=0.01)
    first(torch.randn(2, 4, 12)).sum().backward()
    assert first.input.weight.data_ptr() != second.input.weight.data_ptr()
    assert first.output.weight.data_ptr() != second.output.weight.data_ptr()
    assert bool(first.input.weight.grad.count_nonzero())
    assert bool(first.output.weight.grad.count_nonzero())
    assert second.input.weight.grad is None
    assert second.output.weight.grad is None
