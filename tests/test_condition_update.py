from __future__ import annotations

import torch

from ember.writer.condition_update import (
    compact_rank2_effective_tangent,
    deterministic_mgs_column_pivots,
    FixedBalancedCausalConditionFeature,
    pivot_preserving_base_factors,
    ProgramReconciliationState,
    ProgramResidualMemory,
    anchored_reconciliation_program_delta,
    apply_anchored_reconciliation_update_,
    program_residual_delta_application_evidence,
    stable_factor_head_linearization,
)
from ember.writer.model import FactorHead, WriterVideoEvidence


def _evidence(frame_values: torch.Tensor, text: torch.Tensor) -> WriterVideoEvidence:
    frames, tokens, width = frame_values.shape
    assert text.shape == (tokens, width)
    return WriterVideoEvidence(
        text_queries=text[None],
        frame_evidence=frame_values,
        grounded_evidence=frame_values.clone(),
        interactions=torch.zeros(frames, width),
        valid_task_tokens=torch.ones(1, tokens, dtype=torch.bool),
        offsets=(0, frames),
    )


def test_fixed_temporal_feature_is_zero_preserving_and_reads_real_order() -> None:
    encoder = FixedBalancedCausalConditionFeature(
        program_width=3,
        feature_width=6,
        initialization_seed=17,
    )
    text = torch.tensor([[2.0, -1.0, 0.5], [1.0, 3.0, -2.0]])
    zero = _evidence(text[None].expand(4, -1, -1).clone(), text)
    indices = torch.tensor([0, 5, 10, 15], dtype=torch.long)
    zero_feature = encoder(zero, indices)
    assert torch.equal(zero_feature, torch.zeros_like(zero_feature))

    innovation = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.5, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.5]],
            [[-1.0, 0.5, 0.0], [-0.5, 0.25, 0.0]],
        ]
    )
    evidence = _evidence(text[None] + innovation, text)
    natural = encoder(evidence, indices)
    reversed_feature = encoder(
        evidence,
        indices,
        frame_order=torch.tensor([3, 2, 1, 0], dtype=torch.long),
    )
    shuffled = encoder(
        evidence,
        indices,
        frame_order=torch.tensor([0, 2, 1, 3], dtype=torch.long),
    )
    physically_reversed = encoder(
        _evidence(evidence.frame_evidence.flip(0), text), indices
    )
    physically_shuffled = encoder(
        _evidence(evidence.frame_evidence[[0, 2, 1, 3]], text), indices
    )
    assert natural.shape == (1, 6)
    torch.testing.assert_close(
        natural.square().sum(dim=1), torch.ones(1), rtol=1e-6, atol=1e-6
    )
    assert not torch.equal(natural, reversed_feature)
    assert not torch.equal(natural, shuffled)
    torch.testing.assert_close(reversed_feature, physically_reversed)
    torch.testing.assert_close(shuffled, physically_shuffled)
    assert not tuple(encoder.parameters())
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_feature = encoder(evidence, indices)
        memory = ProgramResidualMemory(
            feature_width=6,
            program_slots=3,
            program_width=4,
        )
        autocast_read = memory(autocast_feature)
    assert autocast_feature.dtype == torch.float32
    assert autocast_read.dtype == torch.float32


def test_balanced_causal_feature_breaks_static_reverse_collinearity() -> None:
    encoder = FixedBalancedCausalConditionFeature(
        program_width=3,
        feature_width=6,
        initialization_seed=17,
    )
    text = torch.zeros(1, 3)
    static = torch.tensor([2.0, -1.0, 0.5])
    dynamic = torch.tensor([0.25, 1.5, -0.75])
    evidence = _evidence(
        torch.stack((static + dynamic, static - dynamic))[:, None], text
    )
    indices = torch.tensor([0, 5], dtype=torch.long)
    natural = encoder(evidence, indices)
    reversed_feature = encoder(
        evidence,
        indices,
        frame_order=torch.tensor([1, 0], dtype=torch.long),
    )
    torch.testing.assert_close(
        natural.square().sum(dim=1), torch.ones(1), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        reversed_feature.square().sum(dim=1),
        torch.ones(1),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        (natural * reversed_feature).sum(),
        torch.zeros(()),
        rtol=0,
        atol=2e-6,
    )


def test_first_anchored_update_matches_blind_and_preserves_negative_rows() -> None:
    correct_features = torch.eye(4, dtype=torch.float32)[:2]
    negative_features = torch.eye(4, dtype=torch.float32)[2:]
    cotangents = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
    reconciliation = ProgramReconciliationState(feature_width=4)
    delta, next_precision, summary = anchored_reconciliation_program_delta(
        correct_features,
        negative_features,
        cotangents,
        reconciliation,
        step_size=1.0,
        relative_damping=0.01,
    )
    full_features = torch.cat((correct_features, negative_features))
    predicted = (full_features @ delta.flatten(1)).reshape(4, 3, 4)
    torch.testing.assert_close(
        predicted[:2],
        -cotangents / 1.01,
        rtol=2e-5,
        atol=2e-6,
    )
    assert torch.equal(predicted[2:], torch.zeros_like(predicted[2:]))
    assert summary.feature_rank == 4
    assert summary.predicted_correct_motion_rms > 0
    assert summary.predicted_negative_motion_rms == 0
    assert summary.predicted_negative_to_correct_ratio == 0
    assert summary.current_motion_to_blind_ratio == 1
    assert summary.assimilated_rows_before == 0
    assert summary.assimilated_rows_after == 4
    assert delta.dtype == torch.float32

    memory = ProgramResidualMemory(
        feature_width=4,
        program_slots=3,
        program_width=4,
    )
    assert torch.equal(memory(full_features), torch.zeros(4, 3, 4))
    before = memory(full_features).clone()
    apply_anchored_reconciliation_update_(
        memory,
        reconciliation,
        delta,
        next_precision,
        assimilated_rows_after=summary.assimilated_rows_after,
    )
    application = program_residual_delta_application_evidence(
        memory,
        delta,
        full_features,
        before,
        predicted=predicted,
    )
    torch.testing.assert_close(memory(full_features), predicted)
    assert application.predicted_observed_max_abs == 0
    assert application.predicted_observed_relative_rms == 0
    assert reconciliation.assimilated_rows == 4
    assert not torch.equal(reconciliation.precision, torch.eye(4, dtype=torch.float64))


def test_zero_correct_motion_uses_finite_gate_failure_value() -> None:
    features = torch.eye(4, dtype=torch.float32)
    reconciliation = ProgramReconciliationState(feature_width=4)
    delta, next_precision, summary = anchored_reconciliation_program_delta(
        features[:2],
        features[2:],
        torch.zeros(2, 3, 4),
        reconciliation,
        step_size=1.0,
        relative_damping=0.01,
    )
    assert torch.isfinite(torch.tensor(summary.predicted_negative_to_correct_ratio))
    assert summary.predicted_negative_to_correct_ratio > 1e30
    assert torch.count_nonzero(delta) == 0
    assert not torch.equal(next_precision, reconciliation.precision)
    memory = ProgramResidualMemory(feature_width=4, program_slots=3, program_width=4)
    before = memory.value.clone()
    apply_anchored_reconciliation_update_(
        memory,
        reconciliation,
        delta,
        next_precision,
        assimilated_rows_after=summary.assimilated_rows_after,
    )
    assert torch.equal(memory.value, before)
    assert torch.equal(reconciliation.precision, next_precision)
    assert reconciliation.assimilated_rows == 4


def test_first_anchored_update_matches_nonorthogonal_blind_ridge() -> None:
    generator = torch.Generator().manual_seed(29)
    correct = torch.randn(3, 7, generator=generator)
    negative = torch.randn(3, 7, generator=generator)
    cotangent = torch.randn(3, 2, 4, generator=generator)
    reconciliation = ProgramReconciliationState(feature_width=7)
    delta, _, summary = anchored_reconciliation_program_delta(
        correct,
        negative,
        cotangent,
        reconciliation,
        step_size=0.7,
        relative_damping=0.01,
    )
    features = torch.cat((correct, negative)).to(torch.float64)
    gram = features @ features.T
    damping = 0.01 * gram.diagonal().mean()
    right = torch.cat(
        (-0.7 * cotangent.flatten(1), torch.zeros_like(cotangent).flatten(1))
    ).to(torch.float64)
    expected = features.T @ torch.linalg.solve(
        gram + damping * torch.eye(6, dtype=torch.float64), right
    )
    torch.testing.assert_close(
        delta.flatten(1).to(torch.float64), expected, rtol=2e-5, atol=2e-6
    )
    assert abs(summary.current_motion_to_blind_ratio - 1.0) <= 1e-6


def test_reference_diagnostics_match_explicit_current_and_blind_motion() -> None:
    generator = torch.Generator().manual_seed(31)
    reconciliation = ProgramReconciliationState(feature_width=6)
    memory = ProgramResidualMemory(feature_width=6, program_slots=2, program_width=3)
    prior_correct = torch.randn(3, 6, generator=generator)
    prior_negative = torch.randn(3, 6, generator=generator)
    prior_cotangent = torch.randn(3, 2, 3, generator=generator)
    prior_delta, prior_precision, prior_summary = anchored_reconciliation_program_delta(
        prior_correct,
        prior_negative,
        prior_cotangent,
        reconciliation,
        step_size=1.0,
        relative_damping=0.01,
    )
    apply_anchored_reconciliation_update_(
        memory,
        reconciliation,
        prior_delta,
        prior_precision,
        assimilated_rows_after=prior_summary.assimilated_rows_after,
    )

    correct = torch.randn(3, 6, generator=generator)
    negative = torch.randn(3, 6, generator=generator)
    cotangent = torch.randn(3, 2, 3, generator=generator)
    delta, _, summary = anchored_reconciliation_program_delta(
        correct,
        negative,
        cotangent,
        reconciliation,
        step_size=1.0,
        relative_damping=0.01,
        reference_correct_features=prior_correct,
    )
    reference_motion = prior_correct @ delta.flatten(1)
    features = torch.cat((correct, negative)).to(torch.float64)
    gram = features @ features.T
    damping = 0.01 * gram.diagonal().mean()
    right = torch.cat(
        (-cotangent.flatten(1), torch.zeros_like(cotangent).flatten(1))
    ).to(torch.float64)
    blind_delta = features.T @ torch.linalg.solve(
        gram + damping * torch.eye(6, dtype=torch.float64), right
    )
    blind_reference_motion = prior_correct.to(torch.float64) @ blind_delta
    reference_rows = reference_motion.square().mean(dim=1).sqrt()
    blind_rows = blind_reference_motion.square().mean(dim=1).sqrt()
    expected_fraction = float((reference_rows < blind_rows).float().mean())
    expected_rms = float(reference_motion.square().mean().sqrt())
    expected_blind_rms = float(blind_reference_motion.square().mean().sqrt())
    assert abs(summary.reference_motion_rms - expected_rms) <= 1e-6
    assert abs(summary.blind_reference_motion_rms - expected_blind_rms) <= 1e-6
    assert summary.reference_rows_improved_fraction == expected_fraction


def test_streaming_anchored_reconciliation_matches_direct_cumulative_ridge() -> None:
    generator = torch.Generator().manual_seed(41)
    reconciliation = ProgramReconciliationState(feature_width=5)
    memory = ProgramResidualMemory(feature_width=5, program_slots=2, program_width=3)
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = []
    targets: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    for _ in range(4):
        correct = torch.randn(2, 5, generator=generator)
        negative = torch.randn(2, 5, generator=generator)
        cotangent = torch.randn(2, 2, 3, generator=generator)
        full = torch.cat((correct, negative))
        gram = full.to(torch.float64) @ full.to(torch.float64).T
        damping = 0.01 * float(gram.diagonal().mean())
        before = full.to(torch.float64) @ memory.value.flatten(1).to(torch.float64)
        increment = torch.cat(
            (-cotangent.flatten(1), torch.zeros_like(cotangent).flatten(1))
        ).to(torch.float64)
        targets.append((full.to(torch.float64), before + increment, damping))
        batches.append((correct, negative, cotangent, damping))
        delta, next_precision, summary = anchored_reconciliation_program_delta(
            correct,
            negative,
            cotangent,
            reconciliation,
            step_size=1.0,
            relative_damping=0.01,
        )
        apply_anchored_reconciliation_update_(
            memory,
            reconciliation,
            delta,
            next_precision,
            assimilated_rows_after=summary.assimilated_rows_after,
        )

    precision = torch.eye(5, dtype=torch.float64)
    right = torch.zeros(5, 6, dtype=torch.float64)
    for features, target, damping in targets:
        precision.add_(features.T @ features / damping)
        right.add_(features.T @ target / damping)
    direct = torch.linalg.solve(precision, right)
    torch.testing.assert_close(
        memory.value.flatten(1).to(torch.float64),
        direct,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        reconciliation.precision, precision, rtol=1e-13, atol=1e-13
    )
    assert reconciliation.assimilated_rows == 16


def test_rank_deficient_repeated_features_remain_finite() -> None:
    correct = torch.ones(2, 3)
    negative = torch.ones(2, 3)
    cotangent = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    reconciliation = ProgramReconciliationState(feature_width=3)
    delta, precision, summary = anchored_reconciliation_program_delta(
        correct,
        negative,
        cotangent,
        reconciliation,
        step_size=1.0,
        relative_damping=0.01,
    )
    assert torch.isfinite(delta).all()
    assert torch.isfinite(precision).all()
    assert summary.feature_rank == 1


def test_stable_factor_head_linearization_matches_exact_fp32_jvp() -> None:
    torch.manual_seed(23)
    head = FactorHead(5, 7, 4)
    with torch.no_grad():
        head.network[-1].weight.normal_(std=0.2)
    source = torch.randn(2, 3, 5)
    residual = torch.randn_like(source) * 0.01
    rows, tangent = stable_factor_head_linearization(head, source, residual)
    expected_rows, expected_tangent = torch.autograd.functional.jvp(
        head,
        source,
        residual,
        create_graph=False,
        strict=True,
    )
    torch.testing.assert_close(rows, expected_rows)
    assert tangent is not None
    torch.testing.assert_close(tangent, expected_tangent, rtol=2e-5, atol=2e-6)


def test_pivot_base_keeps_native_columns_and_reconstructs_rank14_pair() -> None:
    generator = torch.Generator().manual_seed(29)
    left = torch.randn(3, 41, 14, generator=generator)
    coordinates = torch.randn(3, 14, 16, generator=generator)
    base_b = torch.matmul(left, coordinates)
    base_a = torch.randn(3, 16, 19, generator=generator)
    reduced_a, reduced_b, pivots = pivot_preserving_base_factors(
        base_a, base_b, keep=14
    )
    assert pivots.shape == (3, 14)
    assert all(len(set(row.tolist())) == 14 for row in pivots)
    gathered = torch.gather(
        base_b,
        -1,
        pivots.unsqueeze(-2).expand(3, base_b.shape[-2], 14),
    )
    assert torch.equal(reduced_b, gathered)
    torch.testing.assert_close(
        torch.matmul(reduced_b, reduced_a),
        torch.matmul(base_b, base_a),
        rtol=3e-4,
        atol=3e-4,
    )
    tied = torch.zeros(2, 7, 16)
    tied_pivots = deterministic_mgs_column_pivots(tied, keep=14)
    assert torch.equal(tied_pivots, torch.arange(14).expand(2, -1))

    native_a = base_a.to(torch.bfloat16)
    native_b = base_b.to(torch.bfloat16)
    native_reduced_a, native_reduced_b, native_pivots = pivot_preserving_base_factors(
        native_a, native_b, keep=14
    )
    native_gathered = torch.gather(
        native_b,
        -1,
        native_pivots.unsqueeze(-2).expand(3, native_b.shape[-2], 14),
    )
    assert native_reduced_a.dtype == native_reduced_b.dtype == torch.bfloat16
    assert torch.equal(native_reduced_b, native_gathered)


def test_compact_rank2_matches_effective_tangent_without_second_order_term() -> None:
    generator = torch.Generator().manual_seed(31)
    base_a = torch.randn(2, 16, 35, generator=generator)
    base_b = torch.randn(2, 40, 16, generator=generator)
    delta_a = torch.randn(2, 16, 35, generator=generator) * 0.01
    delta_b = torch.randn(2, 40, 16, generator=generator) * 0.01
    residual_a, residual_b = compact_rank2_effective_tangent(
        base_a, base_b, delta_a, delta_b
    )
    expected = torch.matmul(base_b, delta_a) + torch.matmul(delta_b, base_a)
    observed = torch.matmul(residual_b, residual_a)
    u, singular, vh = torch.linalg.svd(expected, full_matrices=False)
    expected_top2 = torch.matmul(
        u[..., :, :2] * singular[..., :2].unsqueeze(-2),
        vh[..., :2, :],
    )
    torch.testing.assert_close(observed, expected_top2, rtol=3e-5, atol=3e-5)
    finite_factor_delta = torch.matmul(
        base_b + delta_b,
        base_a + delta_a,
    ) - torch.matmul(base_b, base_a)
    assert not torch.allclose(expected, finite_factor_delta)
    assert residual_a.shape == (2, 2, 35)
    assert residual_b.shape == (2, 40, 2)

    zero_a, zero_b = compact_rank2_effective_tangent(
        base_a,
        base_b,
        torch.zeros_like(delta_a),
        torch.zeros_like(delta_b),
    )
    assert torch.isfinite(zero_a).all() and torch.isfinite(zero_b).all()
    assert not torch.count_nonzero(zero_a)
    assert not torch.count_nonzero(zero_b)


def test_compact_rank2_keeps_small_svd_fp32_under_outer_autocast(
    monkeypatch,
) -> None:
    generator = torch.Generator().manual_seed(37)
    base_a = torch.randn(2, 16, 35, generator=generator, dtype=torch.bfloat16)
    base_b = torch.randn(2, 40, 16, generator=generator, dtype=torch.bfloat16)
    delta_a = torch.randn(2, 16, 35, generator=generator, dtype=torch.bfloat16)
    delta_b = torch.randn(2, 40, 16, generator=generator, dtype=torch.bfloat16)
    original_matmul = torch.matmul
    original_svd = torch.linalg.svd
    observed_svd_dtypes = []

    def emulate_cuda_autocast_matmul(left, right, *args, **kwargs):
        result = original_matmul(left, right, *args, **kwargs)
        if torch.is_autocast_enabled("cpu") and result.dtype == torch.float32:
            return result.to(dtype=torch.bfloat16)
        return result

    def require_fp32_svd(value, *args, **kwargs):
        observed_svd_dtypes.append(value.dtype)
        assert value.dtype == torch.float32
        return original_svd(value, *args, **kwargs)

    monkeypatch.setattr(torch, "matmul", emulate_cuda_autocast_matmul)
    monkeypatch.setattr(torch.linalg, "svd", require_fp32_svd)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        residual_a, residual_b = compact_rank2_effective_tangent(
            base_a,
            base_b,
            delta_a,
            delta_b,
        )

    assert observed_svd_dtypes == [torch.float32]
    assert residual_a.dtype == residual_b.dtype == torch.bfloat16
    assert torch.isfinite(residual_a).all() and torch.isfinite(residual_b).all()
