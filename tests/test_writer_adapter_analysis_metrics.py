from types import SimpleNamespace

import pytest
import torch

from ember.writer.adapter_analysis_metrics import (
    effective_delta_metrics,
    policy_lane_batch_records,
    policy_lane_checkpoint_summary,
    tensor_delta_metrics,
)


PAIRS = {"module": {"a": "a", "b": "b"}}


def _state(value: float) -> dict[str, torch.Tensor]:
    return {
        "a": torch.tensor([[1.0]]),
        "b": torch.tensor([[value]]),
    }


def test_effective_delta_metrics_measure_recovery_and_residual() -> None:
    metrics = effective_delta_metrics(
        PAIRS,
        _state(1.0),
        _state(3.0),
        _state(2.0),
    )

    assert metrics["target_delta_l2"] == pytest.approx(2.0)
    assert metrics["candidate_over_target_delta_l2"] == pytest.approx(0.5)
    assert metrics["delta_cosine"] == pytest.approx(1.0)
    assert metrics["residual_over_target_delta_l2"] == pytest.approx(0.5)


def test_effective_delta_metrics_preserve_opposite_direction() -> None:
    metrics = effective_delta_metrics(
        PAIRS,
        _state(1.0),
        _state(3.0),
        _state(0.0),
    )

    assert metrics["candidate_over_target_delta_l2"] == pytest.approx(0.5)
    assert metrics["delta_cosine"] == pytest.approx(-1.0)
    assert metrics["residual_over_target_delta_l2"] == pytest.approx(1.5)


def test_tensor_delta_metrics_use_the_same_update_geometry() -> None:
    metrics = tensor_delta_metrics(
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 2.0]),
        torch.tensor([2.0, 2.0]),
    )

    assert metrics["target_delta_l2"] == pytest.approx(2.0)
    assert metrics["candidate_over_target_delta_l2"] == pytest.approx(0.5)
    assert metrics["delta_cosine"] == pytest.approx(1.0)
    assert metrics["residual_over_target_delta_l2"] == pytest.approx(0.5)


def test_policy_lane_records_measure_storage_and_output_participation() -> None:
    writer = SimpleNamespace(
        hyperdecoder=SimpleNamespace(
            a_output=torch.nn.Parameter(torch.ones(2, 5, 3)),
            b_output=torch.nn.Parameter(torch.ones(2, 7, 3)),
        )
    )
    hidden = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        ]
    )
    records = policy_lane_batch_records(
        writer,
        {
            "lanes": torch.ones(2, 2, 4),
            "hidden": hidden,
            "a_energy": torch.ones(2, 2),
            "b_energy": torch.ones(2, 2),
        },
        ("demo_0", "demo_1"),
    )

    assert records is not None
    assert records["lane_storage"]["combined"]["effective_lanes"] == pytest.approx(2.0)
    demo = records["conditions"]["demo_0"]
    assert demo["combined_output_lane_participation"]["active_lanes"] == 2
    assert demo["combined_output_lane_participation"]["effective_lanes"] == pytest.approx(2.0)
    assert demo["hidden_row_geometry"]["stable_row_rank"] == pytest.approx(2.0)


def test_policy_lane_summary_allows_action_panel_only_conditions() -> None:
    writer = SimpleNamespace(
        hyperdecoder=SimpleNamespace(
            a_output=torch.nn.Parameter(torch.ones(2, 5, 3)),
            b_output=torch.nn.Parameter(torch.ones(2, 7, 3)),
        )
    )
    names = (
        "demo_0",
        "demo_1",
        "demo_2",
        "demo_3",
        "demo_4",
        "reversed_0",
        "shuffled_0",
    )
    lanes = torch.arange(7 * 2 * 4, dtype=torch.float32).reshape(7, 2, 4) + 1
    hidden = torch.arange(7 * 2 * 3, dtype=torch.float32).reshape(7, 2, 3) + 1
    capture = {
        "lanes": lanes,
        "hidden": hidden,
        "a_energy": torch.ones(7, 2),
        "b_energy": torch.ones(7, 2),
    }
    full = policy_lane_batch_records(
        writer, capture, names
    )
    partial = policy_lane_batch_records(
        writer,
        {key: value[:5] for key, value in capture.items()},
        names[:5],
    )

    summary = policy_lane_checkpoint_summary(
        [{"policy_lane": partial}, {"policy_lane": full}]
    )

    assert summary is not None
    assert "reversed_0" in summary["demo_0_condition_relative_l2"]
    assert summary["demo_0_condition_relative_l2"]["reversed_0"]["mean"] > 0
