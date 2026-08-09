from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_contract import load_v6_prior_config


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/pi05_v6_ecp_policy_effective_writer_v2.json"
)


def _pre_gradient_config() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["gradient_profile"].update(
        {
            "status": "ready_after_cpu_and_single_a40_throughput_smoke",
            "artifact_evidence": None,
        }
    )
    config["objective"]["auxiliary_weights"].update(
        {
            "status": "blocked_until_live_train24_gradient_profile",
            "projection": None,
            "ranking": None,
        }
    )
    config["profile_run"].update(
        {"status": "blocked_until_live_gradient_weights", "artifact_evidence": None}
    )
    config["formal_run"]["status"] = (
        "blocked_until_live_a40_resume_profile_evidence"
    )
    return config


def test_v6_prior_config_allows_only_predeclared_policy_microbatches(
    tmp_path: Path,
) -> None:
    base = _pre_gradient_config()
    for microbatch in (16, 10):
        candidate = deepcopy(base)
        candidate["optimization"]["functional_policy_microbatch_size"] = microbatch
        candidate["gradient_profile"]["functional_policy_microbatch_size"] = microbatch
        path = tmp_path / f"microbatch-{microbatch}.json"
        path.write_text(json.dumps(candidate), encoding="utf-8")
        assert (
            load_v6_prior_config(path)["optimization"][
                "functional_policy_microbatch_size"
            ]
            == microbatch
        )

    for microbatch in (1, 20):
        candidate = deepcopy(base)
        candidate["optimization"]["functional_policy_microbatch_size"] = microbatch
        candidate["gradient_profile"]["functional_policy_microbatch_size"] = microbatch
        candidate["gradient_profile"]["physical_policy_forwards_per_task"] = (
            20 // microbatch
        )
        path = tmp_path / f"rejected-microbatch-{microbatch}.json"
        path.write_text(json.dumps(candidate), encoding="utf-8")
        with pytest.raises(ExpertManifoldError, match="scientific boundary"):
            load_v6_prior_config(path)
