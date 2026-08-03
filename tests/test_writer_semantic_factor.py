from __future__ import annotations

import torch

from ember.writer.program_compiler import SemanticFactorHead, SemanticFactorRouter


def test_semantic_factor_router_preserves_amplitude_and_cannot_create_value() -> None:
    torch.manual_seed(19)
    router = SemanticFactorRouter(width=32, basis_count=4, initialization_seed=7)
    head = SemanticFactorHead(128, 32, 16, 4)
    torch.nn.init.normal_(head.output.weight, std=0.01)
    routing = router(torch.randn(2, 3, 5, 32))
    assert routing.shape == (2, 3, 5, 4)
    assert torch.allclose(routing.sum(-1), torch.full((2, 3, 5), 4.0), atol=1e-6)
    assert not torch.allclose(routing[0], routing[1])
    assert not bool(head(torch.zeros(2, 5, 128), routing[:, 0]).count_nonzero())
