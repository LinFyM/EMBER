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
from ember.expert_manifold.v6_success_key import SuccessKeyAnchorBank
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
    config["formal_run"]["artifact_evidence"] = None
    config["evaluation"]["formal_status"] = (
        "sealed_from_live_srtp_deployment_smoke"
    )
    config["evaluation"]["online_smoke_evidence"] = {"path": "vertical.json"}
    return config


def _context(world_size: int = 6) -> DistributedContext:
    return DistributedContext(0, 0, world_size, torch.device("cpu"))


def _args(
    *,
    output_dir: Path,
    resume: Path | None,
    stop: int | None,
    mode: str = "formal",
) -> argparse.Namespace:
    return argparse.Namespace(
        mode=mode,
        output_dir=output_dir,
        resume=resume,
        stop_after_macro=stop,
        num_workers=2,
    )


def _git_state(commit: str = "a" * 40) -> dict:
    return {
        "commit": commit,
        "authority_commit": commit,
        "authority_contains_commit": True,
        "dirty_paths": [],
    }


def test_srtp_config_changes_only_final_shared_write_and_fixed_reward_credit() -> None:
    config = load_v6_prior_config()
    assert config["schema_version"] == V6_PRIOR_CONFIG_SCHEMA
    assert config["method"]["language_only_lora_path"] is False
    assert config["condition_feature"]["innovation_width"] == 3072
    assert config["condition_feature"]["phase_slots"] == 16
    assert config["condition_feature"]["goal_block"].startswith(
        "terminal_quartile"
    )
    assert config["condition_feature"]["causal_block"].startswith(
        "sqrt_normalized_causal_prefix"
    )
    assert config["condition_feature"]["projection_shape"] == [2, 128, 3072]
    assert config["condition_feature"]["learned_parameters"] == 0
    assert config["update"]["kind"] == (
        "full48_success_key_nullspace_then_shared_reward_tangent_projection"
    )
    assert config["update"]["blind_parameterization"].endswith(
        "orthogonal_complement"
    )
    assert config["update"]["reward_projection"].startswith("euclidean_nearest")
    assert config["update"]["task_scalar_gate_or_mask"] is False
    assert config["update"]["persistent_precision_or_optimizer_state"] is False
    assert "reconciliation" not in config
    assert config["environment"]["rollouts_per_task"] == 4
    assert config["environment"]["retain_success_replay"] is False
    assert config["environment"]["retain_failure_replay"] is False
    assert config["objective"]["trajectory_replay"] == (
        "fixed_landmarks_only_no_full_prefix_replay"
    )
    assert config["objective"]["maximum_logical_landmark_batch"] == 16
    assert config["objective"]["flow_mc_samples"] == 4
    assert config["success_key_bank"]["task_slots"] == 24
    assert config["success_key_bank"]["deployment_read"] is False
    assert config["data"]["action_queries_per_task"] == 20
    distributed = config["optimization"]["distributed_update"]
    assert distributed["world_size"] == "fresh_live_1_to_6_then_exact_resume_locked"
    assert config["formal_run"]["allowed_world_sizes"] == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("condition_feature", "innovation_width", 1024),
        ("condition_feature", "phase_slots", 8),
        ("condition_feature", "projection_seed", 1),
        ("update", "kind", "anchored"),
        ("update", "relative_damping", 0.02),
        ("data", "teacher_video_seed", 1),
        ("objective", "name", "reward_credit"),
        ("rng", "environment_seed_root", 1),
        ("environment", "retain_failure_replay", True),
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


def test_unknown_fields_and_retired_config_paths_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _raw_config()
    config["condition_feature"]["fallback"] = "language"
    with pytest.raises(ExpertManifoldError, match="fail-closed contract"):
        _load_mutation(tmp_path, monkeypatch, config)
    with pytest.raises(ExpertManifoldError, match="canonical config path"):
        load_v6_prior_config(
            V6_PRIOR_CANONICAL_CONFIG.parent
            / "pi05_v6_reward_credit_program_cotangent_v1.json"
        )
    with pytest.raises(ExpertManifoldError, match="canonical config path"):
        load_v6_prior_config(
            V6_PRIOR_CANONICAL_CONFIG.parent
            / "pi05_v6_policy_innovation_goal_causal_key_v1.json"
        )
    with pytest.raises(ExpertManifoldError, match="canonical config path"):
        load_v6_prior_config(
            V6_PRIOR_CANONICAL_CONFIG.parent
            / "pi05_v6_success_key_nullspace_consolidation_v1.json"
        )


def test_profile_is_authorized_before_formal_and_formal_opens_only_after_seal() -> None:
    sealed = load_v6_prior_config()
    with pytest.raises(ExpertManifoldError, match="blocked by live gates"):
        runtime_for_mode(sealed, "formal")
    ready = deepcopy(sealed)
    ready["status"] = "active_formal_ready"
    ready["formal_run"]["status"] = "ready_after_live_profile_seal"
    ready["formal_run"]["artifact_evidence"] = None
    preseal = deepcopy(ready)
    preseal["status"] = "active_cpu_ready_awaiting_live_profile"
    preseal["profile_run"]["status"] = "awaiting_live_a40_fresh0_to1_profile"
    preseal["profile_run"]["artifact_evidence"] = None
    preseal["formal_run"]["status"] = (
        "blocked_until_live_profile_passes_and_is_sealed"
    )
    preseal["evaluation"]["formal_status"] = (
        "awaiting_live_srtp_deployment_smoke"
    )
    preseal["evaluation"]["online_smoke_evidence"] = None
    assert runtime_for_mode(preseal, "mechanism-profile") == (1, (), 0)
    with pytest.raises(ExpertManifoldError, match="blocked by live gates"):
        runtime_for_mode(preseal, "formal")
    assert runtime_for_mode(ready, "formal") == (10, (5, 10), 0)


def test_preprofile_artifact_injection_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _raw_config()
    config["status"] = "active_cpu_ready_awaiting_live_profile"
    config["profile_run"]["status"] = "awaiting_live_a40_fresh0_to1_profile"
    config["formal_run"]["status"] = (
        "blocked_until_live_profile_passes_and_is_sealed"
    )
    config["evaluation"]["formal_status"] = (
        "awaiting_live_srtp_deployment_smoke"
    )
    config["evaluation"]["online_smoke_evidence"] = None
    config["profile_run"]["artifact_evidence"] = {"path": "unsealed.json"}
    with pytest.raises(ExpertManifoldError, match="fail-closed contract"):
        _load_mutation(tmp_path, monkeypatch, config)


def test_runtime_segments_are_fresh_zero_to_five_then_exact_five_to_ten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _formal_ready_config()
    monkeypatch.setattr(
        runtime_module, "residual_git_state", lambda _root: _git_state()
    )
    fresh = _resolve_segment(
        _args(output_dir=tmp_path / "run", resume=None, stop=5),
        config,
        _context(),
    )
    assert (fresh.start_macro, fresh.stop_macro) == (0, 5)
    assert fresh.checkpoint_macros == (5, 10)
    for stop in (None, 10):
        with pytest.raises(ExpertManifoldError, match="sealed segment"):
            _resolve_segment(
                _args(output_dir=tmp_path / "run", resume=None, stop=stop),
                config,
                _context(),
            )

    checkpoint = tmp_path / "run/checkpoints/macro_00000005"
    checkpoint.mkdir(parents=True)
    write_json_atomic(
        checkpoint.parent.parent / "run_contract.json",
        {"git": {"commit": "a" * 40}},
    )
    resumed = _resolve_segment(
        _args(output_dir=tmp_path / "run", resume=checkpoint, stop=10),
        config,
        _context(),
    )
    assert (resumed.start_macro, resumed.stop_macro) == (5, 10)


def test_runtime_accepts_every_live_world_up_to_six_and_rejects_larger_world(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _formal_ready_config()
    monkeypatch.setattr(
        runtime_module, "residual_git_state", lambda _root: _git_state()
    )
    arguments = _args(output_dir=tmp_path / "run", resume=None, stop=5)
    for world_size in range(1, 7):
        segment = _resolve_segment(arguments, config, _context(world_size))
        assert (segment.start_macro, segment.stop_macro) == (0, 5)
    with pytest.raises(ExpertManifoldError):
        _resolve_segment(arguments, config, _context(7))
    arguments.num_workers = 0
    with pytest.raises(ExpertManifoldError):
        _resolve_segment(arguments, config, _context())
    arguments.num_workers = 2
    monkeypatch.setattr(
        runtime_module,
        "residual_git_state",
        lambda _root: {**_git_state(), "dirty_paths": ["M source.py"]},
    )
    with pytest.raises(ExpertManifoldError):
        _resolve_segment(arguments, config, _context())


def test_cursor_seals_k4_randomness_without_cumulative_precision_state() -> None:
    cursor = cursor_contract(_raw_config(), 10)
    assert cursor == {
        "next_macro": 10,
        "task_visits_per_task": 10,
        "sampler_seed": 20260721,
        "teacher_video_seed": 20260722,
        "counterfactual_seed": 20260809,
        "counterfactual_phase": 1,
        "videos_per_task_visit": 1,
        "action_queries_per_task": 20,
        "full48_order": "correct_0_to_23_then_negative_0_to_23",
        "rollouts_per_task": 4,
        "next_rollout_cursor_per_task": 40,
        "environment_seed_root": 2026081101,
        "policy_noise_seed_root": 2026081102,
        "landmark_seed_root": 2026081201,
        "flow_credit_seed_root": 2026081103,
        "landmark_policy": "first_last_plus_two_seeded_uniform_reservoir_interior",
        "success_key_anchor_policy": "first_all_success_per_train_task",
    }
    assert all("precision" not in key for key in cursor)


def test_ownership_records_deployment_memory_and_training_only_success_bank() -> None:
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
    bank = SuccessKeyAnchorBank(
        range(24), feature_width=256, device=torch.device("cpu")
    )
    observed = _ownership_contract(ownership, writer, bank)
    assert observed["fixed_policy_innovation_encoder"]["checkpoint_owned"] is False
    assert observed["fixed_projection"]["shape"] == [2, 128, 3072]
    assert observed["program_residual_memory"]["checkpoint_owned"] is True
    assert observed["success_key_anchor_bank"]["checkpoint_owned"] is True
    assert observed["success_key_anchor_bank"]["deployment_owned"] is False
    assert "reconciliation_precision" not in observed
