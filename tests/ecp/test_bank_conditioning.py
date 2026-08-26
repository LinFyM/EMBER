from __future__ import annotations

import torch

from ember.ecp.bank_conditioning import (
    StreamingBankStatistics,
    StreamingSignedPool,
    bounded_relative_group_gain,
    materialized_bank_statistics,
    materialized_signed_pool,
    spectral_bank_query,
)


def _condition(seed: int = 17):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(73, 12, generator=generator, dtype=torch.float64)
    mass = torch.rand(73, generator=generator, dtype=torch.float64) + 0.1
    mass = mass / mass.sum()
    reference = torch.randn(4, 12, generator=generator, dtype=torch.float64)
    mean = torch.einsum("n,nd->d", mass, values)
    score = reference @ (values - mean).T
    score_maximum = score.abs().amax(-1)
    compatibility = score * (0.1 / score_maximum)[:, None]
    return values, mass, reference, score_maximum, compatibility


def test_streaming_bank_statistics_and_spectral_query_match_materialized() -> None:
    values, mass, reference, score_maximum, compatibility = _condition()
    expected = materialized_bank_statistics(values, mass, compatibility)
    accumulator = StreamingBankStatistics(
        width=values.shape[-1],
        query_shape=(reference.shape[0],),
        device=values.device,
        dtype=torch.float64,
    )
    for start, stop in ((0, 11), (11, 39), (39, 73)):
        accumulator.add(
            values[start:stop], mass[start:stop], compatibility[:, start:stop]
        )
    observed = accumulator.finalize()
    torch.testing.assert_close(observed.mean, expected.mean, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        observed.covariance, expected.covariance, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        observed.anchor, expected.anchor, rtol=1e-12, atol=1e-12
    )

    solved = spectral_bank_query(observed, relative_eigenvalue_floor=1e-10)
    target = reference * (0.1 / score_maximum)[:, None]
    cosine = torch.nn.functional.cosine_similarity(solved.query, target, dim=-1)
    assert float(cosine.min()) > 0.999999
    assert solved.relative_residual_maximum < 1e-10

    loss = solved.query.square().mean()
    compatibility = compatibility.clone().requires_grad_()
    differentiable = materialized_bank_statistics(values, mass, compatibility)
    spectral_bank_query(
        differentiable, relative_eigenvalue_floor=1e-10
    ).query.square().mean().backward()
    assert compatibility.grad is not None
    assert bool(torch.isfinite(compatibility.grad).all())
    assert bool(torch.count_nonzero(compatibility.grad))
    assert float(loss) > 0.0


def test_signed_pool_is_chunk_equivalent_and_group_gain_is_bounded() -> None:
    values, mass, reference, score_maximum, _ = _condition(seed=23)
    expected = materialized_signed_pool(reference, values, mass)
    accumulator = StreamingSignedPool(reference, dtype=torch.float64)
    for start, stop in ((0, 7), (7, 54), (54, 73)):
        accumulator.add(values[start:stop], mass[start:stop])
    torch.testing.assert_close(
        accumulator.signed_mean(), expected, rtol=1e-12, atol=1e-12
    )

    maxima = torch.stack((score_maximum, score_maximum * 0.25, score_maximum * 2.0))
    gain = bounded_relative_group_gain(maxima)
    torch.testing.assert_close(gain[2], torch.ones_like(gain[2]))
    torch.testing.assert_close(gain[1], torch.full_like(gain[1], 0.125))
    assert float(gain.min()) >= 0.0
    assert float(gain.max()) <= 1.0
