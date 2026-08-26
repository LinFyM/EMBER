from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.ecp.native_factors import rms_normalize
from ember.ecp.shared_compiler_dual_basis import (
    probe_thresholds,
    projected_duals,
    signed_pool_output_groups,
    signed_pool_queries,
    stable_dual_factor,
    task_distribution,
    task_equal_scatter,
)
from ember.ecp.shared_compiler_dual_basis_artifacts import (
    validate_formal_arguments,
)


def _ill_conditioned_values(seed: int, *, scale: float = 1.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(40, 1, 1, 4, generator=generator, dtype=torch.float64)
    return values * torch.tensor(
        [1.0, 0.1, 0.01, 0.003], dtype=torch.float64
    ) * scale


def _dual_for(values: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    mass = torch.linspace(1.0, 2.0, values.shape[0], dtype=torch.float64)
    mass = mass[:, None, None]
    factor = stable_dual_factor(
        values, mass, relative_singular_threshold=1e-8
    )
    unit, norm, _projection, _geometry = factor.solve(desired)
    dual = unit * norm[:, None]
    torch.testing.assert_close(
        dual @ factor.scatter,
        desired.reshape(-1, desired.shape[-1]),
        rtol=1e-6,
        atol=1e-8,
    )
    return dual


def test_fp64_analytic_dual_replays_through_antithetic_pooling() -> None:
    values = _ill_conditioned_values(7)
    mass = torch.linspace(1.0, 2.0, values.shape[0], dtype=torch.float64)[
        :, None, None
    ]
    desired = torch.tensor([[0.5, -0.7, 0.4, 0.2]], dtype=torch.float64)
    dual = _dual_for(values, desired).reshape(1, 1, 1, 4)

    pooled = signed_pool_queries(dual, values, mass, score_bound=0.05)

    expected = rms_normalize(desired).reshape_as(pooled)
    cosine = torch.nn.functional.cosine_similarity(pooled, expected, dim=-1)
    assert float(cosine) > 0.999


def test_output_groups_share_one_gain_and_preserve_relative_amplitude() -> None:
    first = _ill_conditioned_values(11)
    second = _ill_conditioned_values(13, scale=1.7)
    grouped = torch.stack((first, second))
    mass = torch.linspace(1.0, 2.0, first.shape[0], dtype=torch.float64)[
        :, None, None
    ]
    desired_first = torch.tensor([[0.8, -0.2, 0.4, 0.1]], dtype=torch.float64)
    desired_second = torch.tensor([[0.04, 0.02, -0.03, 0.01]], dtype=torch.float64)
    queries = (
        _dual_for(first, desired_first).reshape(1, 1, 1, 4),
        _dual_for(second, desired_second).reshape(1, 1, 1, 4),
    )

    actual = signed_pool_output_groups(
        queries, grouped, mass, score_bound=0.05
    )

    expected = rms_normalize(
        torch.cat((desired_first, desired_second), dim=-1)
    ).reshape_as(actual)
    cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=-1)
    assert float(cosine) > 0.995


def test_task_scatter_is_rank_gauge_invariant() -> None:
    generator = torch.Generator().manual_seed(19)
    block = torch.randn(4, 12, generator=generator)
    gauge = torch.tensor(
        [
            [2.0, 0.3, 0.0, 0.0],
            [0.1, 0.8, 0.2, 0.0],
            [0.0, 0.2, 1.3, 0.1],
            [0.0, 0.0, 0.4, 0.7],
        ]
    )

    original = task_equal_scatter((block,))
    transformed = task_equal_scatter((gauge @ block,))

    torch.testing.assert_close(original, transformed, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(original.trace(), torch.tensor(1.0))


def test_projection_reports_effective_basis_width() -> None:
    generator = torch.Generator().manual_seed(23)
    dual = torch.randn(2, 4, 8, generator=generator, dtype=torch.float64)
    basis = torch.eye(8)[:, :4]

    projected, effective, cosine = projected_duals(dual, basis, (2, 8))

    assert projected.shape == (3, 2, 4, 8)
    assert effective == (8, 2, 4)
    assert cosine[0] == 1.0
    torch.testing.assert_close(projected[0], dual)


def test_task_distribution_is_task_and_video_hierarchical() -> None:
    rows = [
        {
            "authority_id": 1,
            "video_demo": 10,
            "role": "meta_fit",
            "score": 1.0,
        },
        {
            "authority_id": 1,
            "video_demo": 10,
            "role": "meta_fit",
            "score": 1.0,
        },
        {
            "authority_id": 1,
            "video_demo": 11,
            "role": "meta_fit",
            "score": 0.0,
        },
        {
            "authority_id": 2,
            "video_demo": 20,
            "role": "target_fit",
            "score": 0.8,
        },
    ]

    summary = task_distribution(rows, "score")

    assert summary["task_mean"]["median"] == 0.65
    assert summary["task_worst_video"]["median"] == 0.4
    assert summary["roles_task_mean"]["meta_fit"]["mean"] == 0.5
    assert summary["roles_task_mean"]["target_fit"]["mean"] == 0.8


def test_probe_gate_cannot_average_away_a_bad_same_task_video() -> None:
    rows = []
    for task in range(10):
        rows.extend(
            (
                {
                    "authority_id": task,
                    "video_demo": 2 * task,
                    "role": "meta_fit",
                    "score": 1.0,
                },
                {
                    "authority_id": task,
                    "video_demo": 2 * task + 1,
                    "role": "meta_fit",
                    "score": 0.96 if task else 0.0,
                },
            )
        )

    checks = probe_thresholds(task_distribution(rows, "score"))

    assert checks["task_mean_median_at_least_0.98"]
    assert not checks["task_worst_video_p10_at_least_0.95"]


def test_formal_arguments_cannot_use_exploratory_escape_or_custom_config() -> None:
    root = Path(__file__).resolve().parents[2]
    args = SimpleNamespace(
        task_ids=None,
        shard_count=6,
        target_indices=(20, 21, 36, 37),
        max_videos_per_task=2,
        basis_dimensions=(16, 32, 64, 96, 128),
        score_bound=0.1,
        config=root / "configs/pi05_ecp_shared_compiler_g3_v2.json",
        g1_config=root / "configs/pi05_ecp_native_factor_g1_v1.json",
    )
    validate_formal_arguments(args, repo_root=root)

    args.task_ids = (9,)
    with pytest.raises(ValueError, match="canonical contract"):
        validate_formal_arguments(args, repo_root=root)
    args.task_ids = None
    args.config = root / "configs/pi05_ecp_shared_compiler_g3_v1.json"
    with pytest.raises(ValueError, match="canonical contract"):
        validate_formal_arguments(args, repo_root=root)
