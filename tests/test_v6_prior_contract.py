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


def test_v6_prior_config_opens_only_the_preregistered_gradient_profile() -> None:
    config = load_v6_prior_config(CONFIG)
    assert runtime_for_mode(config, "gradient-profile") == (1, ())
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
