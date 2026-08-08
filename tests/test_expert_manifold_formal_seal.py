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
    / "configs/pi05_video_expert_manifold_hard_routed_policy_effective_v2.json"
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


def _hard_route_evidence() -> dict:
    return {
        "artifact": (
            "runs/outputs/pi05_expert_manifold_hard_routed_"
            "cpu_real_assets_20260809/analysis.json"
        ),
        "cpu_only": True,
        "deployed_coefficient_rule": "deterministic_signed_argmax_one_hot",
        "argmax_tie_break": "lowest_expert_ordinal",
        "zero_identity_exact": True,
        "all_nonzero_support_one": True,
        "all_nonzero_sum_one": True,
        "all_coefficients_finite": True,
        "all_states_finite": True,
        "train_centroid_self_route_all_tasks": True,
        "ordered_reversed_selection_changed": True,
        "expert_count": 24,
        "train_centroid_count": 24,
        "train_centroid_self_route_count": 24,
        "train_video_count": 1200,
        "deployed_support_min": 1,
        "deployed_support_max": 1,
    }


def test_hard_routed_evaluation_is_blocked_only_on_live_smoke() -> None:
    evaluation = load_barycentric_writer_config(CONFIG)["evaluation"]
    assert evaluation["formal_status"] == "blocked_until_live_a40_online_smoke"
    assert evaluation["cpu_hard_route_evidence"][
        "train_centroid_self_route_count"
    ] == 24
    assert evaluation["cpu_hard_route_evidence"][
        "ordered_reversed_selection_change_count"
    ] == 1200
    assert "online_smoke_evidence" not in evaluation
    assert (
        evaluation["cpu_policy_effective_compiler"]["selected_effective_basis_rank"]
        == 96
    )


def test_formal_seal_accepts_only_complete_live_smoke_evidence(
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    config["evaluation"]["formal_status"] = "sealed"
    config["evaluation"]["cpu_hard_route_evidence"] = _hard_route_evidence()
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
