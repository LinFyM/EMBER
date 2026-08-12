from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ember.expert_manifold.v6_prior_contract as contract_module
import ember.expert_manifold.v6_prior_runtime as runtime_module
from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior import V6PriorOwnership
from ember.expert_manifold.v6_prior_contract import (
    V6_PRIOR_CANONICAL_CONFIG,
    V6_PRIOR_CONFIG_SCHEMA,
    load_v6_prior_config,
    runtime_for_mode,
)
from ember.expert_manifold.v6_prior_run_contract import (
    _ownership_contract,
    cursor_contract,
)
from ember.expert_manifold.v6_prior_runtime import _resolve_segment
from ember.pi05_source_checkpoint import DistributedContext, write_json_atomic


def _raw_config() -> dict:
    return json.loads(V6_PRIOR_CANONICAL_CONFIG.read_text(encoding="utf-8"))


def _load_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict,
) -> None:
    path = tmp_path / V6_PRIOR_CANONICAL_CONFIG.name
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(contract_module, "V6_PRIOR_CANONICAL_CONFIG", path)
    load_v6_prior_config(path)


def _formal_ready_config() -> dict:
    config = _raw_config()
    config["status"] = "active_formal_ready"
    config["profile_run"]["status"] = "sealed_from_live_a40_fresh0_to1_profile"
    config["profile_run"]["artifact_evidence"] = {"path": "profile.json"}
    config["formal_run"]["status"] = "ready_after_live_profile_seal"
    return config


def _preprofile_config() -> dict:
    config = _raw_config()
    config["status"] = "active_cpu_ready_awaiting_live_profile"
    config["profile_run"]["status"] = "awaiting_live_a40_fresh0_to1_profile"
    config["profile_run"]["artifact_evidence"] = None
    config["formal_run"]["status"] = (
        "blocked_until_live_profile_passes_and_is_sealed"
    )
    return config


def _context(world_size: int = 6) -> DistributedContext:
    return DistributedContext(0, 0, world_size, torch.device("cpu"))


def _args(
    *, output_dir: Path, resume: Path | None, stop: int | None, mode: str = "formal"
) -> argparse.Namespace:
    return argparse.Namespace(
        mode=mode,
        output_dir=output_dir,
        resume=resume,
        stop_after_macro=stop,
        num_workers=0,
    )


def _git_state(commit: str = "a" * 40) -> dict:
    return {
        "commit": commit,
        "authority_commit": commit,
        "authority_contains_commit": True,
        "dirty_paths": [],
    }


def test_pvjfc_config_seals_two_train_views_and_one_deployment_view() -> None:
    config = load_v6_prior_config()
    assert config["schema_version"] == V6_PRIOR_CONFIG_SCHEMA
    assert config["method"]["name"] == "frozen_v6_paired_video_joint_functional_credit"
    assert config["method"]["training_views_per_task"] == 2
    assert config["method"]["deployment_views_per_task"] == 1
    assert config["information_wall"]["training_reward_reads"] == 0
    assert config["information_wall"]["training_outcome_rollouts"] == 0
    assert config["update"]["correct_conditions"] == 48
    assert config["update"]["negative_conditions"] == 48
    assert config["update"]["view_weights"] == [0.5, 0.5]
    assert config["update"]["view_swap_invariant"] is True
    assert config["update"]["duplicate_view_degenerates_to_single_view"] is True
    assert config["update"]["history_replay"] is False
    assert config["objective"]["reward_use"] == "none"
    assert config["data"]["action_queries_per_task"] == 20
    assert config["data"]["training_companion_videos_per_task_per_macro"] == 1
    assert config["formal_run"]["allowed_world_sizes"] == list(range(1, 7))
    assert "partial_occupancy_is_allowed" in config["evaluation"]["device_selection"]
    assert "success_key_bank" not in config
    assert "environment" not in config


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("condition_feature", "innovation_width", 1024),
        ("condition_feature", "phase_slots", 8),
        ("condition_feature", "projection_seed", 1),
        ("update", "kind", "single-view"),
        ("update", "view_weights", [1, 1]),
        ("update", "relative_damping", 0.02),
        ("data", "teacher_video_seed", 1),
        ("objective", "name", "language-only"),
        ("rng", "view_policy_rng", "independent"),
        ("optimization", "functional_policy_microbatch_size", 1),
        ("profile_run", "gates", {}),
        ("formal_run", "decision_gates", {}),
    ),
)
def test_scientific_contract_mutations_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    key: str,
    value: object,
) -> None:
    config = _raw_config()
    config[section][key] = value
    with pytest.raises(ExpertManifoldError, match="fail-closed contract"):
        _load_mutation(tmp_path, monkeypatch, config)


def test_unknown_fields_and_retired_configs_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _raw_config()
    config["condition_feature"]["fallback"] = "language"
    with pytest.raises(ExpertManifoldError, match="fail-closed contract"):
        _load_mutation(tmp_path, monkeypatch, config)
    for name in (
        "pi05_v6_paired_candidate_update_guard_v1.json",
        "pi05_v6_success_key_nullspace_consolidation_v1.json",
        "pi05_v6_policy_innovation_goal_causal_key_v1.json",
    ):
        with pytest.raises(ExpertManifoldError, match="canonical config path"):
            load_v6_prior_config(V6_PRIOR_CANONICAL_CONFIG.parent / name)


def test_profile_is_authorized_before_formal_and_formal_opens_only_after_seal() -> None:
    sealed = _raw_config()
    with pytest.raises(ExpertManifoldError, match="not authorized"):
        runtime_for_mode(sealed, "mechanism-profile")
    with pytest.raises(ExpertManifoldError, match="blocked by live gates"):
        runtime_for_mode(sealed, "formal")
    preseal = _preprofile_config()
    assert runtime_for_mode(preseal, "mechanism-profile") == (1, (), 0)
    with pytest.raises(ExpertManifoldError, match="blocked by live gates"):
        runtime_for_mode(preseal, "formal")
    ready = _formal_ready_config()
    assert runtime_for_mode(ready, "formal") == (10, tuple(range(1, 11)), 0)


def test_preprofile_artifact_injection_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _preprofile_config()
    config["profile_run"]["artifact_evidence"] = {"path": "unsealed.json"}
    with pytest.raises(ExpertManifoldError, match="fail-closed contract"):
        _load_mutation(tmp_path, monkeypatch, config)


def test_runtime_segments_are_only_zero_to_five_then_exact_five_to_ten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _formal_ready_config()
    monkeypatch.setattr(runtime_module, "residual_git_state", lambda _root: _git_state())
    fresh = _resolve_segment(
        _args(output_dir=tmp_path / "run", resume=None, stop=5), config, _context()
    )
    assert (fresh.start_macro, fresh.stop_macro) == (0, 5)
    assert fresh.checkpoint_macros == tuple(range(1, 11))
    for stop in (None, 1, 10):
        with pytest.raises(ExpertManifoldError, match="sealed segment"):
            _resolve_segment(
                _args(output_dir=tmp_path / "run", resume=None, stop=stop),
                config,
                _context(),
            )
    checkpoint = tmp_path / "run/checkpoints/macro_00000005"
    checkpoint.mkdir(parents=True)
    write_json_atomic(
        checkpoint.parent.parent / "run_contract.json", {"git": {"commit": "a" * 40}}
    )
    resumed = _resolve_segment(
        _args(output_dir=tmp_path / "run", resume=checkpoint, stop=10),
        config,
        _context(),
    )
    assert (resumed.start_macro, resumed.stop_macro) == (5, 10)


def test_runtime_accepts_one_to_six_cards_including_partially_available_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _formal_ready_config()
    monkeypatch.setattr(runtime_module, "residual_git_state", lambda _root: _git_state())
    arguments = _args(output_dir=tmp_path / "run", resume=None, stop=5)
    for world_size in range(1, 7):
        assert _resolve_segment(arguments, config, _context(world_size)).stop_macro == 5
    with pytest.raises(ExpertManifoldError):
        _resolve_segment(arguments, config, _context(7))


def test_cursor_seals_paired_views_same_rng_and_no_outcome_state() -> None:
    cursor = cursor_contract(_raw_config(), 10)
    assert cursor == {
        "next_macro": 10,
        "task_visits_per_task": 10,
        "sampler_seed": 20260721,
        "teacher_video_seed": 20260722,
        "counterfactual_seed": 20260809,
        "counterfactual_phase": 1,
        "videos_per_task_visit": 1,
        "training_companion_videos_per_task_visit": 1,
        "action_queries_per_task": 20,
        "condition_order": (
            "primary_correct_0_to_23_then_companion_correct_0_to_23_then_"
            "primary_negative_0_to_23_then_companion_negative_0_to_23"
        ),
        "view_weights": [0.5, 0.5],
        "policy_rng_reuse": "same_task_B20_keyed_seed_for_both_views",
    }
    assert all("outcome" not in key and "bank" not in key for key in cursor)


def test_ownership_records_only_deployment_program_memory() -> None:
    ownership = V6PriorOwnership(10_775_296, 523, 600)
    writer = SimpleNamespace(
        condition_feature=SimpleNamespace(
            projection=torch.empty((2, 128, 3072), dtype=torch.float32)
        ),
        policy_innovation=SimpleNamespace(
            feature_width=3072,
            phase_slots=16,
            fixed_suffix_noise=torch.empty((50, 32), dtype=torch.float32),
            state_dict=lambda: {},
        ),
        program_memory=SimpleNamespace(
            value=torch.empty((256, 320, 256), dtype=torch.float32, device="meta")
        ),
    )
    observed = _ownership_contract(ownership, writer)
    assert observed["fixed_policy_innovation_encoder"]["checkpoint_owned"] is False
    assert observed["program_residual_memory"]["checkpoint_owned"] is True
    assert observed["program_residual_memory"]["deployment_owned"] is True
    assert "success_key_anchor_bank" not in observed
