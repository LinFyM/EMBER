from __future__ import annotations

import torch

from ember.ecp.bank_conditioning import (
    StreamingBankStatistics,
    StreamingFeatureStatistics,
    StreamingFunctionalBankStatistics,
    StreamingSignedPool,
    batched_feature_whiteners,
    bound_functional_queries,
    bounded_relative_group_gain,
    functional_polar_queries,
    materialized_bank_statistics,
    materialized_signed_pool,
    normalize_replay_queries,
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


def test_functional_polar_statistics_are_chunk_equivalent_and_bounded() -> None:
    generator = torch.Generator().manual_seed(37)
    candidates, native_width, key_width = 79, 10, 6
    events, ranks = 3, 4
    values = torch.randn(candidates, native_width, generator=generator)
    keys = torch.randn(events, candidates, key_width, generator=generator)
    base = torch.rand(candidates, generator=generator) + 0.1
    base = base / base.sum()
    replay = torch.rand(ranks, candidates, generator=generator) + 0.1
    replay = replay / replay.sum(-1, keepdim=True)
    event = torch.rand(events, candidates, generator=generator) + 0.1
    event = event / event.sum(-1, keepdim=True)

    def accumulator() -> StreamingFunctionalBankStatistics:
        return StreamingFunctionalBankStatistics(
            native_width=native_width,
            key_width=key_width,
            events=events,
            ranks=ranks,
            device=values.device,
        )

    expected = accumulator()
    expected.add(values, base, replay, event, keys)
    streamed = accumulator()
    for start, stop in ((0, 13), (13, 51), (51, candidates)):
        streamed.add(
            values[start:stop],
            base[start:stop],
            replay[:, start:stop],
            event[:, start:stop],
            keys[:, start:stop],
        )
    left = expected.finalize()
    right = streamed.finalize()
    for lhs, rhs in (
        (left.mean, right.mean),
        (left.covariance, right.covariance),
        (left.replay_covariances, right.replay_covariances),
        (left.key_images, right.key_images),
    ):
        torch.testing.assert_close(lhs, rhs, rtol=1e-12, atol=1e-12)

    raw = torch.randn(
        ranks, events, 2, key_width, generator=generator, requires_grad=True
    )
    event_weights = torch.rand(ranks, events, generator=generator)
    event_weights = event_weights / event_weights.sum(-1, keepdim=True)
    polar = functional_polar_queries(
        raw,
        event_weights,
        right,
        covariance_floor=1e-8,
        image_floor=1e-8,
    )
    bounded, _ = bound_functional_queries((polar.queries,), score_bound=1e-3)
    assert float(bounded[0].detach().norm(dim=-1).amax()) <= 1.00001e-3
    bounded[0].square().mean().backward()
    assert raw.grad is not None
    assert bool(torch.isfinite(raw.grad).all())
    assert bool(torch.count_nonzero(raw.grad))

    replay_query = torch.randn(ranks, 2, native_width, generator=generator)
    normalized, _ = normalize_replay_queries(
        (replay_query,), (right,), score_rms=0.02
    )
    score_rms = torch.einsum(
        "rbd,de,rbe->rb",
        normalized[0].double(),
        right.covariance,
        normalized[0].double(),
    ).clamp_min(0).sqrt()
    torch.testing.assert_close(
        score_rms.amax(-1), torch.full((ranks,), 0.02, dtype=torch.float64)
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
