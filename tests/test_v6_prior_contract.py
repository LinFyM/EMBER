from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_contract import (
    load_v6_prior_config,
    runtime_for_mode,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pi05_v6_prior_policy_effective_writer_v1.json"


def test_v6_prior_config_blocks_all_gpu_modes_before_the_online_smoke() -> None:
    config = load_v6_prior_config(CONFIG)
    with pytest.raises(ExpertManifoldError, match="gradient profile is not ready"):
        runtime_for_mode(config, "gradient-profile")
    ready = deepcopy(config)
    ready["gradient_profile"]["status"] = (
        "ready_after_cpu_and_single_a40_warmstart_reproduction_smoke"
    )
    assert runtime_for_mode(ready, "gradient-profile") == (1, ())
    with pytest.raises(ExpertManifoldError, match="profile runtime is not sealed"):
        runtime_for_mode(config, "profile")
    with pytest.raises(ExpertManifoldError, match="formal runtime is not sealed"):
        runtime_for_mode(config, "formal")


def test_v6_prior_config_rejects_language_bypass_and_unprofiled_weights(
    tmp_path: Path,
) -> None:
    baseline = json.loads(CONFIG.read_text(encoding="utf-8"))
    bypass = deepcopy(baseline)
    bypass["method"]["language_only_lora_path"] = True
    bypass_path = tmp_path / "bypass.json"
    bypass_path.write_text(json.dumps(bypass), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="scientific boundary"):
        load_v6_prior_config(bypass_path)

    weights = deepcopy(baseline)
    weights["objective"]["auxiliary_weights"]["expert"] = 0.1
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(json.dumps(weights), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="scientific boundary"):
        load_v6_prior_config(weights_path)

    batched = deepcopy(baseline)
    batched["evaluation"]["writer_model_batch_size"] = 8
    batched_path = tmp_path / "batched.json"
    batched_path.write_text(json.dumps(batched), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="scientific boundary"):
        load_v6_prior_config(batched_path)
