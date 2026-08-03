from __future__ import annotations

import torch

from ember.writer.program_compiler import (
    SemanticDirectionRouter,
    SemanticDirectionStoreHead,
)


def test_semantic_direction_store_route_is_fixed_and_value_free() -> None:
    torch.manual_seed(19)
    centers = torch.eye(8, 32)
    router = SemanticDirectionRouter(
        centers, anchor_mean=torch.zeros(32), top_k=2
    )
    head = SemanticDirectionStoreHead(128, 32, 16, 8)
    anchors = torch.stack((centers[1], centers[5]))
    indices, weights = router(anchors)
    assert indices.shape == weights.shape == (2, 2)
    assert indices[:, 0].tolist() == [1, 5]
    assert torch.equal(weights, torch.full((2, 2), 0.5))
    assert not bool(
        head(torch.zeros(2, 5, 128), indices, weights).count_nonzero()
    )


def test_semantic_direction_store_updates_only_selected_stores() -> None:
    torch.manual_seed(23)
    head = SemanticDirectionStoreHead(12, 7, 5, 8)
    torch.nn.init.normal_(head.output_weight, std=0.01)
    indices = torch.tensor([[1, 3], [3, 6]], dtype=torch.long)
    weights = torch.full((2, 2), 0.5)
    head(torch.randn(2, 4, 12), indices, weights).sum().backward()
    active = {1, 3, 6}
    assert {
        index
        for index in range(8)
        if bool(head.input_weight.grad[index].count_nonzero())
    } == active
    assert {
        index
        for index in range(8)
        if bool(head.output_weight.grad[index].count_nonzero())
    } == active
