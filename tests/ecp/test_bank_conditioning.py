from __future__ import annotations

from types import SimpleNamespace

import torch

from ember.ecp.bank_conditioning import (
    StreamingProjectedFunctionalStatistics,
    StreamingSketchCrossImage,
    StreamingBankStatistics,
    StreamingFeatureStatistics,
    StreamingFunctionalBankStatistics,
    StreamingNativeCovariance,
    StreamingSignedPool,
    batched_functional_polar_queries,
    batched_feature_whiteners,
    batched_spectral_native_covariances,
    bank_adaptive_basis,
    bound_functional_queries,
    bounded_relative_group_gain,
    functional_polar_queries,
    fixed_nested_projection,
    functional_target_queries,
    materialized_bank_statistics,
    materialized_signed_pool,
    normalize_replay_queries,
    spectral_bank_query,
)
from ember.ecp.bank_conditioning.functional_polar import _economy_svd_right
from ember.ecp.bank_conditioning.primal_dual import SpectralNativeCovariance
from ember.ecp.bank_conditioning.primal_dual_runtime import (
    PrimalDualVideoOperator,
)
from ember.ecp.contracts import TargetFamily, TargetOwner


def test_functional_sketch_is_chunk_equivalent_nested_and_target_recovering() -> None:
    generator = torch.Generator().manual_seed(83)
    candidates, native_width, key_width = 71, 12, 8
    events, ranks, maximum_rank = 3, 4, 6
    values = torch.randn(candidates, native_width, generator=generator).double()
    key_map = torch.randn(
        events, native_width, key_width, generator=generator
    ).double()
    keys = torch.einsum("nd,edm->enm", values, key_map)
    keys += 0.01 * torch.randn(keys.shape, generator=generator).double()
    base = torch.rand(candidates, generator=generator).double() + 0.1
    base /= base.sum()
    event = torch.rand(events, candidates, generator=generator).double() + 0.1
    event /= event.sum(-1, keepdim=True)
    replay = torch.rand(ranks, candidates, generator=generator).double() + 0.1
    replay /= replay.sum(-1, keepdim=True)

    def cross_accumulator() -> StreamingSketchCrossImage:
        return StreamingSketchCrossImage(
            native_width=native_width,
            key_width=key_width,
            events=events,
            device=values.device,
            dtype=torch.float64,
        )

    materialized = cross_accumulator()
    materialized.add(values, base, event, keys)
    streamed = cross_accumulator()
    for start, stop in ((0, 9), (9, 38), (38, candidates)):
        streamed.add(
            values[start:stop],
            base[start:stop],
            event[:, start:stop],
            keys[:, start:stop],
        )
    full_cross = materialized.finalize()
    chunk_cross = streamed.finalize()
    torch.testing.assert_close(full_cross.mean, chunk_cross.mean)
    torch.testing.assert_close(full_cross.key_images, chunk_cross.key_images)

    for mode in ("global", "per_event"):
        projection = fixed_nested_projection(
            events=events,
            key_width=key_width,
            maximum_rank=maximum_rank,
            mode=mode,
            seed=20260828,
            device=values.device,
            dtype=torch.float64,
        )
        repeated = fixed_nested_projection(
            events=events,
            key_width=key_width,
            maximum_rank=maximum_rank,
            mode=mode,
            seed=20260828,
            device=values.device,
            dtype=torch.float64,
        )
        torch.testing.assert_close(projection, repeated)
        basis = bank_adaptive_basis(
            full_cross,
            projection,
            requested_rank=maximum_rank,
            mode=mode,
            relative_singular_floor=1e-10,
        )
        assert basis.retained_rank == maximum_rank

        def projected_accumulator() -> StreamingProjectedFunctionalStatistics:
            return StreamingProjectedFunctionalStatistics(
                basis, ranks=ranks, device=values.device, dtype=torch.float64
            )

        expected = projected_accumulator()
        expected.add(values, base, replay)
        observed = projected_accumulator()
        for start, stop in ((0, 9), (9, 38), (38, candidates)):
            observed.add(
                values[start:stop], base[start:stop], replay[:, start:stop]
            )
        left = expected.finalize(full_cross)
        right = observed.finalize(chunk_cross)
        torch.testing.assert_close(left.covariance, right.covariance)
        torch.testing.assert_close(left.replay_covariances, right.replay_covariances)
        torch.testing.assert_close(left.replay_images, right.replay_images)
        torch.testing.assert_close(left.key_images, right.key_images)

        coefficients = torch.randn(
            2, ranks, maximum_rank, generator=generator, dtype=torch.float64
        )
        desired = torch.einsum("mrs,rds->mrd", coefficients, right.replay_images)
        solved = functional_target_queries(
            desired,
            right,
            relative_floor=1e-10,
        )
        assert solved.native.shape == desired.shape
        assert solved.reduced.shape == (2, ranks, maximum_rank)
        assert float(solved.linear_recovery.min()) > 0.999999


def test_economy_svd_preserves_right_gram_for_rectangular_batches() -> None:
    generator = torch.Generator().manual_seed(5)
    for rows, columns in ((23, 5), (5, 23), (7, 7)):
        matrix = torch.randn(3, rows, columns, generator=generator).double()
        singular, right = _economy_svd_right(matrix)
        reconstructed = right.transpose(-1, -2) @ torch.diag_embed(
            singular.square()
        ) @ right
        torch.testing.assert_close(reconstructed, matrix.transpose(-1, -2) @ matrix)


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

    per_event_raw = raw.detach().clone().requires_grad_()
    per_event = batched_functional_polar_queries(
        (per_event_raw, per_event_raw),
        (event_weights, event_weights),
        (right, right),
        covariance_floor=1e-8,
        image_floor=1e-8,
        mode="per_event",
    )
    references = []
    for event_index in range(events):
        event_statistics = type(right)(
            mean=right.mean,
            covariance=right.covariance,
            replay_covariances=right.replay_covariances,
            key_images=right.key_images[event_index : event_index + 1],
            total_mass=right.total_mass,
        )
        references.append(
            functional_polar_queries(
                per_event_raw[:, event_index : event_index + 1],
                torch.ones(ranks, 1),
                event_statistics,
                covariance_floor=1e-8,
                image_floor=1e-8,
            ).queries
        )
    reference = torch.cat(references, dim=1)
    torch.testing.assert_close(per_event[0].queries, reference)
    torch.testing.assert_close(per_event[1].queries, reference)

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

    canonical = StreamingSignedPool(
        reference,
        trusted_positive_measure=True,
        canonical_block_candidates=17,
    )
    canonical.add(values, mass)
    canonical_streamed = StreamingSignedPool(
        reference,
        trusted_positive_measure=True,
        canonical_block_candidates=17,
    )
    for start, stop in ((0, 7), (7, 54), (54, 73)):
        canonical_streamed.add(values[start:stop], mass[start:stop])
    assert torch.equal(canonical_streamed.signed_mean(), canonical.signed_mean())

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


def test_global_primal_dual_replays_the_retained_native_direction() -> None:
    generator = torch.Generator().manual_seed(131)
    values = torch.randn(211, 9, generator=generator, dtype=torch.float64)
    values[:, -1] = values[:, 0] + 1e-4 * values[:, -1]
    mass = torch.rand(211, generator=generator, dtype=torch.float64) + 0.1
    mass /= mass.sum()
    materialized = StreamingNativeCovariance(
        width=9,
        device=values.device,
        dtype=torch.float64,
        canonical_block_candidates=37,
    )
    materialized.add(values, mass)
    streamed = StreamingNativeCovariance(
        width=9,
        device=values.device,
        dtype=torch.float64,
        canonical_block_candidates=37,
    )
    for start, stop in ((0, 17), (17, 83), (83, 211)):
        streamed.add(values[start:stop], mass[start:stop])
    expected = materialized.finalize()
    observed = streamed.finalize()
    assert torch.equal(observed.covariance, expected.covariance)

    operator = batched_spectral_native_covariances(
        (observed,), relative_eigenvalue_floor=1e-6
    )[0]
    primal = torch.randn(
        4, 9, generator=generator, dtype=torch.float64, requires_grad=True
    )
    query, score_rms, projection_fraction = operator.dual_and_score_rms(primal)
    explicit_query, explicit_rms, explicit_projection = operator.dual_and_score_rms(
        primal, inverse_covariance_power=1.0
    )
    assert torch.equal(query, explicit_query)
    assert torch.equal(score_rms, explicit_rms)
    assert torch.equal(projection_fraction, explicit_projection)

    basis = operator.basis.to(primal)
    eigenvalues = operator.eigenvalues.to(primal)
    relative = eigenvalues / eigenvalues[-1]
    transported = ((primal @ basis) / relative.sqrt()[None]) @ basis.T
    half_query, half_rms, _ = operator.dual_and_score_rms(
        transported, inverse_covariance_power=0.5
    )
    assert float(
        torch.nn.functional.cosine_similarity(
            query.detach(), half_query.detach(), dim=-1
        ).min()
    ) > 0.999999
    assert bool(torch.isfinite(half_rms).all())
    quarter_transported = (
        (primal @ basis) / relative.pow(0.25)[None]
    ) @ basis.T
    tempered_query, tempered_rms, _ = operator.dual_and_score_rms(
        quarter_transported, inverse_covariance_power=0.75
    )
    assert float(
        torch.nn.functional.cosine_similarity(
            query.detach(), tempered_query.detach(), dim=-1
        ).min()
    ) > 0.999999
    assert bool(torch.isfinite(tempered_rms).all())
    query = query * (1e-3 / score_rms.clamp_min(1e-12))[:, None]
    replayed = materialized_signed_pool(query, values, mass)
    projection = (primal @ operator.basis) @ operator.basis.T
    cosine = torch.nn.functional.cosine_similarity(replayed, projection, dim=-1)
    assert float(cosine.detach().min()) > 0.99999
    assert float(projection_fraction.detach().min()) > 0.0
    (replayed * torch.randn(replayed.shape, generator=generator)).sum().backward()
    assert primal.grad is not None
    assert bool(torch.isfinite(primal.grad).all())
    assert bool(torch.count_nonzero(primal.grad))


def test_bank_compatibility_support_is_differentiable_and_routes_hard() -> None:
    owners = tuple(
        TargetOwner(index, f"v_{index}", TargetFamily.V, index, 4, 4)
        for index in range(38)
    )
    runtime = PrimalDualVideoOperator(
        owners,
        program_width=8,
        event_slots=4,
        relative_eigenvalue_floor=1e-6,
        replay_score_rms=0.02,
        covariance_frame_chunk=2,
        compatibility_support_threshold=0.906622976064682,
    )

    def spectral(basis: torch.Tensor) -> SpectralNativeCovariance:
        rank = basis.shape[1]
        return SpectralNativeCovariance(
            basis=basis,
            eigenvalues=torch.ones(rank),
            native_width=4,
            retained_rank=rank,
            eigenvalue_floor=torch.tensor(1e-6),
            retained_condition=torch.tensor(1.0),
            retained_trace_fraction=torch.tensor(1.0),
        )

    full = spectral(torch.eye(4))
    supported = spectral(torch.eye(4)[:, :3])
    low = spectral(torch.eye(4)[:, :1])
    high_primals = []
    for index in range(38):
        value = 0.05 * torch.randn(
            4, 4, generator=torch.Generator().manual_seed(index + 7)
        )
        value[:, 0] += 1.0
        high_primals.append(value.requires_grad_())
    high_primals = tuple(high_primals)
    low_primals = tuple(
        torch.tensor([[0.1, 1.0, 0.0, 0.0]]).expand(4, -1).clone()
        for _ in owners
    )
    output_primals = tuple(torch.ones(1, 4, 4) for _ in owners)

    route, training = runtime.input_projection_supports(
        high_primals, tuple(supported for _ in owners)
    )
    values = torch.cat(
        tuple(supported.retained_projection(value) for value in high_primals)
    )
    assert torch.equal(route, values.kthvalue(16).values)
    assert torch.equal(training, values.sort().values[11:20].mean())
    training.backward()
    assert all(value.grad is not None for value in high_primals)
    assert any(bool(torch.count_nonzero(value.grad)) for value in high_primals)

    def prepared(operator: SpectralNativeCovariance) -> SimpleNamespace:
        return SimpleNamespace(
            frame_measure=torch.ones(1),
            input_operators=tuple(operator for _ in owners),
            output_operators=tuple((full,) for _ in owners),
        )

    high_plan = runtime._plan(prepared(supported), high_primals, output_primals)
    low_plan = runtime._plan(prepared(low), low_primals, output_primals)
    decoupled_plan = runtime._plan(
        prepared(supported),
        low_primals,
        output_primals,
        compatibility_input_primals=high_primals,
    )
    forced = runtime._plan(
        prepared(low),
        low_primals,
        output_primals,
        inverse_covariance_power_override=1.0,
    )
    forced_supported = runtime._plan(
        prepared(supported),
        low_primals,
        output_primals,
        inverse_covariance_power_override=1.0,
    )
    assert high_plan.selected_inverse_covariance_power == 1.0
    assert low_plan.selected_inverse_covariance_power == 0.5
    assert decoupled_plan.selected_inverse_covariance_power == 1.0
    for observed, expected in zip(
        decoupled_plan.input_queries,
        forced_supported.input_queries,
        strict=True,
    ):
        torch.testing.assert_close(observed, expected)
    assert low_plan.compatibility_support is not None
    assert forced.selected_inverse_covariance_power == 1.0
    assert forced.compatibility_support is None
