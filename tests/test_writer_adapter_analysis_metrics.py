from types import SimpleNamespace

import pytest
import torch

from ember.writer.adapter_analysis_metrics import (
    effective_delta_metrics,
    policy_dictionary_batch_records,
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


def test_policy_dictionary_records_measure_atom_participation() -> None:
    writer = SimpleNamespace(
        policy_atoms=SimpleNamespace(
            a_atoms=torch.nn.ParameterList(
                [torch.nn.Parameter(torch.ones(4, 2))]
            ),
            b_atoms=torch.nn.ParameterList(
                [torch.nn.Parameter(torch.ones(2, 4))]
            ),
        )
    )
    mix_a = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        ]
    )
    records = policy_dictionary_batch_records(
        writer,
        {"mix_a": mix_a, "mix_b": mix_a.clone()},
        ("demo_0", "demo_1"),
    )

    assert records is not None
    assert records["dictionary_storage"]["combined"]["effective_atoms"] == pytest.approx(4.0)
    demo = records["conditions"]["demo_0"]
    assert demo["combined"]["active_atoms"] == 2
    assert demo["combined"]["effective_atoms"] == pytest.approx(2.0)
    assert demo["storage_norm_weighted"]["effective_atoms"] == pytest.approx(2.0)
    assert demo["a"]["stable_row_rank"] == pytest.approx(2.0)
