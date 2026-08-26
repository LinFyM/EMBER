from __future__ import annotations

import torch

from ember.ecp.bank_conditioning import (
    StreamingBankStatistics,
    StreamingFeatureStatistics,
    StreamingSignedPool,
    batched_feature_whiteners,
    bounded_relative_group_gain,
    materialized_bank_statistics,
    materialized_signed_pool,
    spectral_bank_query,
)


def test_streaming_feature_whitening_is_chunk_equivalent_and_differentiable() -> None:
    generator = torch.Generator().manual_seed(11)
    keys = torch.randn(73, 6, generator=generator)
    mass = torch.rand(3, 73, generator=generator) + 0.1
    mass = mass / mass.sum(-1, keepdim=True)

    expected = StreamingFeatureStatistics(
        events=3, width=6, device=keys.device
    )
    expected.add(keys, mass)
    streamed = StreamingFeatureStatistics(
        events=3, width=6, device=keys.device
    )
    for start, stop in ((0, 11), (11, 39), (39, 73)):
        streamed.add(keys[start:stop], mass[:, start:stop])
    left, right = batched_feature_whiteners(
        (expected.finalize(), streamed.finalize()),
        relative_eigenvalue_floor=1e-8,
    )
    torch.testing.assert_close(left.mean, right.mean, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        left.inverse_sqrt, right.inverse_sqrt, rtol=1e-5, atol=1e-5
    )
    assert left.retained_ranks == (6, 6, 6)

    differentiable = keys.clone().requires_grad_()
    whitened = left.whiten(differentiable)
    assert whitened.shape == (3, 73, 6)
    whitened.square().mean().backward()
    assert differentiable.grad is not None
    assert bool(torch.isfinite(differentiable.grad).all())
    assert bool(torch.count_nonzero(differentiable.grad))


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

    branches = torch.stack((reference, -0.7 * reference), dim=1).requires_grad_()
    bias = torch.randn(
        reference.shape[0], values.shape[0], generator=torch.Generator().manual_seed(29)
    ) * 0.05
    explicit = materialized_signed_pool(
        branches,
        values,
        mass,
        explicit_branches=True,
        logit_bias=bias,
    )
    streamed = StreamingSignedPool(
        branches, dtype=torch.float64, explicit_branches=True
    )
    for start, stop in ((0, 7), (7, 54), (54, 73)):
        streamed.add(
            values[start:stop],
            mass[start:stop],
            bias[:, start:stop],
        )
    torch.testing.assert_close(
        streamed.signed_mean(), explicit, rtol=1e-12, atol=1e-12
    )
    explicit.square().mean().backward()
    assert branches.grad is not None
    assert bool(torch.isfinite(branches.grad).all())
    assert bool(torch.count_nonzero(branches.grad))
