from __future__ import annotations

import torch

from ember.ecp.g1_initialization import (
    CandidateMeasure,
    _scaled_group_probabilities,
)


def test_signed_reference_projection_preserves_simplexes_and_group_direction() -> None:
    generator = torch.Generator().manual_seed(17)
    measures = tuple(
        CandidateMeasure.build(
            torch.randn(96, 8, generator=generator),
            relative_singular_threshold=1e-3,
        )
        for _ in range(3)
    )
    desired = torch.randn(4, 24, generator=generator)

    probabilities, signed = _scaled_group_probabilities(
        measures=measures,
        desired=desired,
        probability_floor_mass=1e-4,
    )

    assert all(
        torch.allclose(
            value.sum(-1), torch.ones_like(value.sum(-1)), atol=1e-6, rtol=0
        )
        for value in probabilities
    )
    assert torch.all(
        torch.nn.functional.cosine_similarity(signed, desired, dim=-1) > 0.999
    )
