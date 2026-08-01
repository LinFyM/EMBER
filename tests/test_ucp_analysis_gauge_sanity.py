from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.writer.model import WriterModelError


def _analysis_script() -> object:
    path = Path(__file__).parents[1] / "scripts/analyze_as_writer_ucp.py"
    spec = importlib.util.spec_from_file_location("ucp_gauge_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patched_routing(
    monkeypatch: pytest.MonkeyPatch, *, ba_relative_l2: float,
) -> tuple[object, SimpleNamespace, torch.Tensor]:
    script = _analysis_script()
    metric = {
        "relative_l2": 0.1, "cosine": 0.99,
        "reference_rms": 1.0, "candidate_rms": 1.0,
    }
    action = torch.ones(2)
    writer = SimpleNamespace(
        PUBLIC_LORA_RANK=16,
        compiler=SimpleNamespace(target_count=1),
        tensor_specs=(SimpleNamespace(name="factor", module="module"),),
        _decoding={"factor": ("q_a", 0)},
    )
    monkeypatch.setattr(script, "_variant_result", lambda **_kwargs: {})
    monkeypatch.setattr(
        script, "_variant_comparison",
        lambda *_args, **_kwargs: {
            "coordinates": metric, "effective_ba": metric,
            "policy_action": metric,
        },
    )
    monkeypatch.setattr(
        script, "rank_gauge_permute",
        lambda *_args: ({"factor": torch.ones(1)}, {}),
    )
    monkeypatch.setattr(script, "validate_lora_state", lambda *_args: None)
    monkeypatch.setattr(script, "policy_action", lambda **_kwargs: action * 1.002)
    monkeypatch.setattr(
        script, "effective_ba_error",
        lambda *_args: {
            "relative_l2": ba_relative_l2, "difference_rms": 1e-11,
            "max_absolute_error": 1e-8,
        },
    )
    monkeypatch.setattr(script, "mapping_metrics", lambda *_args, **_kwargs: metric)
    return script, writer, action


def _run_routing(script: object, writer: SimpleNamespace, action: torch.Tensor) -> object:
    return script._routing_diagnostics(
        writer=writer, policy=SimpleNamespace(), processor=SimpleNamespace(),
        identity={}, lora=None, device=torch.device("cpu"),
        encoded={
            "initial": torch.ones(1), "endpoints": torch.ones(1),
            "valid_intervals": torch.ones(1), "valid_semantics": torch.ones(1),
            "prepared": {}, "action_seed": 7,
        },
        shared={}, full={"public": {"factor": torch.ones(1)}, "action": action},
    )


def test_rank_gauge_sanity_gates_ba_not_low_precision_action_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script, writer, action = _patched_routing(monkeypatch, ba_relative_l2=1e-9)
    result = _run_routing(script, writer, action)["rank_gauge_permutation"]

    assert result["effective_ba_relative_l2_tolerance"] == pytest.approx(2e-5)
    assert result["fixed_policy_action_numerical_error"]["relative_l2"] == pytest.approx(
        .002, abs=1e-6,
    )
    assert "bf16" in result["sanity_gate"]


def test_rank_gauge_sanity_still_rejects_effective_ba_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script, writer, action = _patched_routing(monkeypatch, ba_relative_l2=1e-3)
    with pytest.raises(WriterModelError, match="rank gauge permutation"):
        _run_routing(script, writer, action)
