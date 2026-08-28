from __future__ import annotations

import torch

from ember.ecp.bank_conditioning.native_bank_runtime import NativeCandidateBank
from ember.ecp.bank_conditioning.set_summary import (
    SetConditionedScalarEnergy,
    SetSummaryFactorSelector,
    StreamingSetMoments,
    StreamingSetSignedPool,
    TaskLocalSelectionCode,
    materialized_set_signed_pool,
)
from ember.ecp.native_factors import G1_RESIDUAL_RANK


def _energy(*, global_events: bool = True) -> SetConditionedScalarEnergy:
    torch.manual_seed(7)
    return SetConditionedScalarEnergy(
        feature_width=8,
        event_slots=3,
        global_events=global_events,
        hidden_width=16,
        logit_bound=4.0,
    )


def _flat_candidates() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    values = torch.randn(13, 6, generator=generator)
    keys = torch.randn(13, 8, generator=generator)
    keys = torch.nn.functional.normalize(keys, dim=-1)
    mass = torch.rand(3, 13, generator=generator)
    mass[0, :2] = 0
    return values, keys, mass


def _code() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(13)
    code = torch.randn(G1_RESIDUAL_RANK, 3, 8, generator=generator)
    weights = torch.randn(G1_RESIDUAL_RANK, 3, generator=generator).softmax(-1)
    return code, weights


def test_set_summary_candidate_permutation_invariance() -> None:
    energy = _energy()
    values, keys, mass = _flat_candidates()
    code, event_weights = _code()
    statistics = energy.summarize(keys, mass)
    logits = energy.score(keys, code, statistics)
    expected = materialized_set_signed_pool(
        values, mass, logits, event_weights
    )

    permutation = torch.tensor((8, 2, 12, 0, 6, 5, 4, 9, 1, 11, 3, 10, 7))
    permuted_statistics = energy.summarize(keys[permutation], mass[:, permutation])
    permuted_logits = energy.score(
        keys[permutation], code, permuted_statistics
    )
    actual = materialized_set_signed_pool(
        values[permutation],
        mass[:, permutation],
        permuted_logits,
        event_weights,
    )
    torch.testing.assert_close(statistics.value, permuted_statistics.value)
    torch.testing.assert_close(expected, actual, atol=2e-6, rtol=2e-6)


def test_set_summary_streaming_matches_materialized_across_chunks() -> None:
    energy = _energy()
    values, keys, mass = _flat_candidates()
    code, event_weights = _code()
    statistics = energy.summarize(keys, mass)
    logits = energy.score(keys, code, statistics)
    expected = materialized_set_signed_pool(
        values, mass, logits, event_weights
    )

    pool = StreamingSetSignedPool(
        ranks=G1_RESIDUAL_RANK,
        events=3,
        width=values.shape[-1],
        reference=logits,
    )
    moments = StreamingSetMoments(events=3, width=8, reference=keys)
    encoded = energy.summary_features(keys)
    for start, stop in ((0, 2), (2, 7), (7, 8), (8, 13)):
        pool.add(values[start:stop], mass[:, start:stop], logits[:, :, start:stop])
        moments.add(encoded[start:stop], mass[:, start:stop])
    actual = pool.signed_factor(event_weights)
    streamed_statistics = moments.finalize()
    torch.testing.assert_close(expected, actual, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        statistics.value, streamed_statistics.value, atol=2e-6, rtol=2e-6
    )


def _bank(*, value_width: int, seed: int) -> NativeCandidateBank:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(5, 2, value_width, generator=generator)
    keys = torch.randn(5, 2, 8, generator=generator)
    keys = torch.nn.functional.normalize(keys, dim=-1)
    event_mass = torch.rand(3, 5, 2, generator=generator)
    base_mass = torch.rand(5, 2, generator=generator).clamp_min(1e-3)
    replay_mass = torch.rand(
        G1_RESIDUAL_RANK, 5, 2, generator=generator
    ).clamp_min(1e-3)
    return NativeCandidateBank(
        values=values,
        content_keys=keys,
        base_mass=base_mass,
        event_mass=event_mass,
        replay_mass=replay_mass,
    )


def test_set_summary_selector_has_real_gradient_without_video_parameters() -> None:
    torch.manual_seed(17)
    selector = SetSummaryFactorSelector(
        feature_width=8,
        event_slots=3,
        output_groups=2,
        global_events=True,
        hidden_width=16,
        logit_bound=4.0,
    )
    free = TaskLocalSelectionCode(events=3, width=8)
    code, event_weights = free()
    a, b = selector(
        input_bank=_bank(value_width=6, seed=19),
        output_banks=(
            _bank(value_width=5, seed=23),
            _bank(value_width=5, seed=29),
        ),
        code=code,
        event_weights=event_weights,
    )
    assert a.shape == (G1_RESIDUAL_RANK, 6)
    assert b.shape == (G1_RESIDUAL_RANK, 10)
    loss = a.square().mean() + b.square().mean() + 0.01 * a.sum()
    loss.backward()
    parameters = tuple(selector.parameters()) + tuple(free.parameters())
    assert parameters
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
