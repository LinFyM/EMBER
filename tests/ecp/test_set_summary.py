from __future__ import annotations

import torch

from ember.ecp.bank_conditioning.set_summary import (
    EventBankSetEncoder,
    StreamingEventBankSummary,
    candidate_metadata,
    event_candidate_measure,
    program_relative_coordinates,
)


def _candidates():
    generator = torch.Generator().manual_seed(20260831)
    values = torch.randn(5, 2, 3, 7, generator=generator)
    query = torch.randn(4, 3, 7, generator=generator)
    mean = torch.randn(7, generator=generator)
    assignment = torch.rand(5, 3, generator=generator).softmax(-1)
    frame = torch.rand(5, generator=generator)
    frame = frame / frame.sum()
    return values, query, mean, assignment, frame


def test_candidate_indices_keep_input_and_output_measures_distinct() -> None:
    values, _, _, assignment, frame = _candidates()
    positions = torch.linspace(0.0, 1.0, 5)
    input_metadata = candidate_metadata(positions, output=False, like=values)
    output_metadata = candidate_metadata(positions, output=True, like=values)
    input_mass = event_candidate_measure(frame, assignment, output=False)
    output_mass = event_candidate_measure(frame, assignment, output=True)

    assert input_metadata.shape == (5, 2, 50, 3)
    assert output_metadata.shape == (5, 2, 50, 4, 7)
    assert input_mass.shape == (3, 5, 2, 50)
    assert output_mass.shape == (3, 5, 2, 50, 4)
    torch.testing.assert_close(input_mass.sum(), output_mass.sum())
    torch.testing.assert_close(input_mass.sum(), assignment.mul(frame[:, None]).sum())


def test_program_relative_coordinates_are_basis_invariant() -> None:
    values, query, mean, _, _ = _candidates()
    coordinate = program_relative_coordinates(query, values, mean)
    orthogonal, _ = torch.linalg.qr(torch.randn(7, 7))
    rotated = program_relative_coordinates(
        torch.einsum("red,df->ref", query, orthogonal),
        values @ orthogonal,
        mean @ orthogonal,
    )
    assert coordinate.shape == (5, 2, 3, 12)
    torch.testing.assert_close(coordinate, rotated, atol=2e-5, rtol=2e-5)


def _summary_inputs():
    generator = torch.Generator().manual_seed(17)
    coordinates = torch.randn(13, 12, generator=generator)
    mass = torch.rand(3, 13, generator=generator).clamp_min(1e-4)
    query = torch.randn(4, 3, 12, generator=generator, requires_grad=True)
    values = torch.randn(13, 5, generator=generator, requires_grad=True)
    native = torch.randn(13, 7, generator=generator, requires_grad=True)
    return coordinates, mass, query, values, native


def _accumulate(chunks):
    coordinates, mass, query, values, native = _summary_inputs()
    accumulator = StreamingEventBankSummary(
        events=3,
        coordinate_width=12,
        value_width=5,
        reference=coordinates,
        native_width=7,
    )
    for start, stop in chunks:
        accumulator.add(
            coordinates[start:stop],
            mass[:, start:stop],
            query,
            values[start:stop],
            native[start:stop],
        )
    return accumulator.finalize(), query, values, native


def test_event_summary_matches_irregular_chunks_and_has_gradient() -> None:
    full, _, _, _ = _accumulate(((0, 13),))
    chunked, query, values, native = _accumulate(
        ((0, 2), (2, 7), (7, 8), (8, 13))
    )
    for left, right in zip(
        (
            full.mean,
            full.log_variance,
            full.induced_positive,
            full.induced_negative,
            full.native_anchor,
            full.log_partition,
        ),
        (
            chunked.mean,
            chunked.log_variance,
            chunked.induced_positive,
            chunked.induced_negative,
            chunked.native_anchor,
            chunked.log_partition,
        ),
        strict=True,
    ):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)
    (chunked.condition.square().mean() + chunked.native_anchor.square().mean()).backward()
    assert query.grad is not None and bool(torch.isfinite(query.grad).all())
    assert values.grad is not None and bool(torch.isfinite(values.grad).all())
    assert native.grad is None


def test_real_encoder_is_candidate_permutation_invariant() -> None:
    torch.manual_seed(31)
    encoder = EventBankSetEncoder(
        context_width=11,
        coordinate_width=12,
        summary_value_width=5,
        hidden_width=16,
    )
    coordinates = torch.randn(13, 12)
    metadata = torch.randn(13, 3)
    mass = torch.rand(3, 13).clamp_min(1e-4)
    context = torch.randn(4, 3, 11)
    native = torch.randn(13, 7)
    expected = encoder.summarize(
        coordinates=coordinates,
        metadata=metadata,
        event_mass=mass,
        event_context=context,
        output=False,
        native_values=native,
    )
    permutation = torch.tensor((8, 2, 12, 0, 6, 5, 4, 9, 1, 11, 3, 10, 7))
    actual = encoder.summarize(
        coordinates=coordinates[permutation],
        metadata=metadata[permutation],
        event_mass=mass[:, permutation],
        event_context=context,
        output=False,
        native_values=native[permutation],
    )
    torch.testing.assert_close(expected.condition, actual.condition, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        expected.native_anchor, actual.native_anchor, atol=2e-6, rtol=2e-6
    )
