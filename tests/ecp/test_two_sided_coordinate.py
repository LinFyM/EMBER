import torch

from ember.ecp.two_sided_coordinate import (
    decode_two_sided_code,
    fit_weighted_sketch_basis,
    fixed_two_sided_probes,
    reconstruct_rank4_factors,
    transform_two_sided_sketch,
    two_sided_sketch,
)


def test_fixed_two_sided_probes_are_deterministic_and_orthonormal() -> None:
    first = fixed_two_sided_probes(
        in_features=13, out_features=11, width=8, seed=17
    )
    second = fixed_two_sided_probes(
        in_features=13, out_features=11, width=8, seed=17
    )
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)
        assert torch.allclose(left.T @ left, torch.eye(8), atol=1e-5)


def test_rank4_two_sided_reconstruction_recovers_exact_effective_update() -> None:
    generator = torch.Generator().manual_seed(23)
    a = torch.randn(4, 13, generator=generator)
    b = torch.randn(11, 4, generator=generator)
    omega, psi = fixed_two_sided_probes(
        in_features=13, out_features=11, width=8, seed=29
    )
    sketch = two_sided_sketch(a, b, omega=omega, psi=psi)
    predicted_a, predicted_b = reconstruct_rank4_factors(
        sketch,
        omega=omega,
        psi=psi,
        out_features=11,
        in_features=13,
    )
    expected = b @ a
    predicted = predicted_b @ predicted_a
    assert torch.linalg.norm(predicted - expected) / torch.linalg.norm(expected) < 2e-5


def test_weighted_coordinate_decodes_the_fit_affine_span() -> None:
    generator = torch.Generator().manual_seed(31)
    origin = torch.randn(9, generator=generator)
    directions, _ = torch.linalg.qr(torch.randn(9, 3, generator=generator))
    coefficients = torch.tensor(
        [
            [-2.0, 0.0, 1.0],
            [-1.0, 1.0, -1.0],
            [0.0, -2.0, 0.5],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, -0.5],
        ]
    )
    values = origin + coefficients @ directions.T
    mean, components, scales, mask, _ = fit_weighted_sketch_basis(
        values,
        torch.tensor([0.1, 0.2, 0.2, 0.2, 0.3]),
        width=6,
        relative_eigenvalue_floor=1e-7,
        scale_floor=1e-6,
    )
    held = origin + torch.tensor([0.25, -0.75, 1.5]) @ directions.T
    code = transform_two_sided_sketch(
        held,
        mean=mean,
        components=components,
        scales=scales,
        active_mask=mask,
    )
    decoded = decode_two_sided_code(
        code,
        mean=mean,
        components=components,
        scales=scales,
        active_mask=mask,
    )
    assert int(mask.sum()) == 3
    assert torch.allclose(decoded, held, atol=2e-5)
    assert torch.count_nonzero(code[3:]) == 0
