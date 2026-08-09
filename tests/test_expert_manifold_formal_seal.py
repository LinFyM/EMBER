import copy
import json
from pathlib import Path

import pytest

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.inference import inspect_v6_prior_writer_asset
from ember.expert_manifold.v6_prior_contract import (
    REPO_ROOT,
    load_v6_prior_config,
)
from ember.pi05_source_checkpoint import read_json


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/pi05_v6_prior_policy_effective_writer_v1.json"


def _smoke_evidence() -> dict:
    return {
        "commit": "clean-pushed",
        "root": "runs/outputs/smoke",
        "device": "NVIDIA A40",
        "checkpoint_kind": "historical_v6_macro400_load_only",
        "video_condition": "correct",
        "video_sampling": "without_replacement",
        "validation_task_count": 8,
        "state_count": 1,
        "scientific_rows": 8,
        "generated_entries": 8,
        "cache_entries": 8,
        "writer_state_tensor_count": 600,
        "writer_model_batch_size": 1,
        "writer_modules_released": True,
        "source_policy_reused_for_rollout": True,
        "source_policy_reloaded": False,
        "staged_path_matches_direct_v6_forward": True,
        "staged_path_max_abs_difference": 0.0,
        "retry_count": 0,
        "failure_count": 0,
        "teacher_action_reads": 0,
        "teacher_state_reads": 0,
        "reward_reads": 0,
        "terminal_reads": 0,
        "oom_count": 0,
        "nonfinite_count": 0,
        "success_interpretation": "execution_smoke_only_not_performance_evidence",
    }


def test_v6_prior_evaluation_stays_blocked_before_live_smoke() -> None:
    evaluation = load_v6_prior_config(CONFIG)["evaluation"]
    assert evaluation == {
        "writer_model_batch_size": 1,
        "formal_status": "blocked_until_live_a40_warmstart_reproduction_smoke",
        "online_smoke_evidence": None,
    }


def test_formal_seal_accepts_only_complete_live_smoke_evidence(
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    config["evaluation"] = {
        "writer_model_batch_size": 1,
        "formal_status": "sealed",
        "online_smoke_evidence": _smoke_evidence(),
    }
    path = tmp_path / "sealed.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert load_v6_prior_config(path)["evaluation"]["formal_status"] == "sealed"

    del config["evaluation"]["online_smoke_evidence"]["writer_modules_released"]
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="scientific boundary changed"):
        load_v6_prior_config(path)


def test_old_expert_asset_config_cannot_enter_canonical_runtime() -> None:
    old = REPO_ROOT / "configs/pi05_video_expert_manifold_v1.json"
    with pytest.raises(ExpertManifoldError, match="scientific boundary changed"):
        load_v6_prior_config(old)


def test_historical_v6_warm_start_is_a_real_load_only_evaluation_asset() -> None:
    config = load_v6_prior_config(CONFIG)
    checkpoint = (REPO_ROOT / config["initialization"]["checkpoint"]).resolve()
    historical_source = read_json(checkpoint.parent.parent / "run_contract.json")[
        "source"
    ]

    asset = inspect_v6_prior_writer_asset(
        config,
        checkpoint,
        historical_source,
        require_formal=False,
    )

    assert asset["kind"] == "historical_v6_macro400_load_only"
    assert asset["source_macro"] == 400
    assert asset["method_macro"] == 0
    assert asset["writer_state"]["state_tensor_count"] == 600
    assert asset["writer_state"]["state_value_count"] == 12_064_064
