import copy
import json
from pathlib import Path

import pytest

from ember.expert_manifold.contract import (
    ExpertManifoldError,
    load_barycentric_writer_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT
    / "configs/pi05_video_expert_manifold_policy_effective_barycentric_v1.json"
)


def _smoke_evidence() -> dict:
    return {
        "commit": "clean-pushed",
        "host": "gpu02",
        "physical_gpu": 0,
        "device": "NVIDIA A40",
        "root": "runs/outputs/smoke",
        "validation_task_count": 8,
        "state_count": 1,
        "scientific_rows": 8,
        "video_condition": "correct",
        "video_sampling": "without_replacement",
        "generated_entries": 8,
        "cache_entries": 8,
        "writer_modules_released": True,
        "source_policy_reused_for_rollout": True,
        "source_policy_reloaded": False,
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


def test_policy_effective_evaluation_is_sealed_by_its_own_live_smoke() -> None:
    evaluation = load_barycentric_writer_config(CONFIG)["evaluation"]
    assert evaluation["formal_status"] == "sealed"
    assert evaluation["online_smoke_evidence"]["device"] == "NVIDIA A40"
    assert evaluation["online_smoke_evidence"]["scientific_rows"] == 8
    assert evaluation["online_smoke_evidence"]["source_policy_reloaded"] is False
    assert (
        evaluation["cpu_policy_effective_compiler"]["selected_effective_basis_rank"]
        == 96
    )


def test_formal_seal_accepts_only_complete_live_smoke_evidence(
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    config["evaluation"]["formal_status"] = "sealed"
    config["evaluation"]["online_smoke_evidence"] = _smoke_evidence()
    path = tmp_path / "sealed.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert (
        load_barycentric_writer_config(path)["evaluation"]["formal_status"] == "sealed"
    )

    del config["evaluation"]["online_smoke_evidence"]["writer_modules_released"]
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="scientific boundary changed"):
        load_barycentric_writer_config(path)


def test_old_learned_writer_config_cannot_enter_canonical_runtime() -> None:
    old = REPO_ROOT / "configs/pi05_video_expert_manifold_v1.json"
    with pytest.raises(ExpertManifoldError, match="scientific boundary changed"):
        load_barycentric_writer_config(old)
