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
    config["formal_run"]["artifact_evidence"] = None
    config["evaluation"]["formal_status"] = (
        "sealed_from_live_osg_pc_deployment_smoke"
    )
    config["evaluation"]["online_smoke_evidence"] = {"path": "vertical.json"}
    return config


def _context() -> DistributedContext:
    return DistributedContext(0, 0, 4, torch.device("cpu"))


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


def test_osg_pc_config_changes_only_credit_and_adds_success_guards() -> None:
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
        "full48_on_policy_success_guarded_counterfactual_null_condition_kernel"
    )
    assert config["update"]["projection"].startswith("parameter_free")
    assert config["update"]["persistent_precision_or_optimizer_state"] is False
    assert "reconciliation" not in config
    assert config["environment"]["rollouts_per_task"] == 4
    assert config["environment"]["retain_failure_replay"] is False
    assert config["objective"]["retention_flow_panel_row_identity"] == (
        "historical_complete_k4_task_panel_then_success_row_select"
    )
    assert config["objective"]["retention_flow_mc_samples"] == 4
    assert config["data"]["action_queries_per_task"] == 20
    distributed = config["optimization"]["distributed_update"]
    assert distributed["world_size"] == 4
    assert distributed["tasks_per_rank"] == 6


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


def test_profile_is_authorized_before_formal_and_formal_opens_only_after_seal() -> None:
    result_sealed = load_v6_prior_config()
    ready = _formal_ready_config()
    preseal = deepcopy(ready)
    preseal["status"] = "active_cpu_ready_awaiting_live_profile"
    preseal["profile_run"]["status"] = "awaiting_live_a40_fresh0_to1_profile"
    preseal["profile_run"]["artifact_evidence"] = None
    preseal["formal_run"]["status"] = (
        "blocked_until_live_profile_passes_and_is_sealed"
    )
    preseal["evaluation"]["formal_status"] = (
        "awaiting_live_osg_pc_deployment_smoke"
    )
    preseal["evaluation"]["online_smoke_evidence"] = None
    assert runtime_for_mode(preseal, "mechanism-profile") == (1, (), 0)
    with pytest.raises(ExpertManifoldError, match="blocked by live gates"):
        runtime_for_mode(preseal, "formal")
    assert runtime_for_mode(ready, "formal") == (10, (5, 10), 0)
    with pytest.raises(ExpertManifoldError, match="blocked by live gates"):
        runtime_for_mode(result_sealed, "formal")


def test_preprofile_artifact_injection_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _raw_config()
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


def test_runtime_rejects_wrong_world_workers_dirty_or_unpushed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _formal_ready_config()
    monkeypatch.setattr(
        runtime_module, "residual_git_state", lambda _root: _git_state()
    )
    arguments = _args(output_dir=tmp_path / "run", resume=None, stop=5)
    with pytest.raises(ExpertManifoldError):
        _resolve_segment(
            arguments,
            config,
            DistributedContext(0, 0, 2, torch.device("cpu")),
        )
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
        "retention_flow_seed_root": 2026081103,
    }
    assert all("precision" not in key for key in cursor)


def test_ownership_records_fixed_policy_encoder_but_checkpoints_only_memory() -> None:
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
    assert observed["fixed_projection"]["shape"] == [2, 128, 3072]
    assert observed["program_residual_memory"]["checkpoint_owned"] is True
    assert "reconciliation_precision" not in observed
