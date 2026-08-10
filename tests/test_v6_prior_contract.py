from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ember.expert_manifold.v6_prior_contract as contract_module
import ember.expert_manifold.inference as inference_module
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
    checkpoint_contract,
    cursor_contract,
)
from ember.expert_manifold.v6_prior_runtime import _resolve_segment
from ember.pi05_source_checkpoint import DistributedContext


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


def test_canonical_residual_is_formal_ready_and_tangent_fails_closed() -> None:
    config = load_v6_prior_config(V6_PRIOR_CANONICAL_CONFIG)
    assert config["schema_version"] == V6_PRIOR_CONFIG_SCHEMA
    assert config["profile_run"]["status"] == "sealed_from_live_a40_macro49_profile"
    assert config["formal_run"]["status"] == (
        "ready_after_live_mechanism_and_deployment_seals"
    )
    assert runtime_for_mode(config, "formal") == (50, (10, 25, 50), 0)
    assert config["method"]["language_only_lora_path"] is False
    assert config["method"]["dynamic_value"] == "one_raw_teacher_video_only"
    assert config["program_residual"]["value_count"] == 20_971_520
    old = V6_PRIOR_CANONICAL_CONFIG.with_name(
        "pi05_v6_condition_local_tangent_tube_writer_v3.json"
    )
    with pytest.raises(ExpertManifoldError, match="non-canonical"):
        load_v6_prior_config(old)


def test_formal_runtime_fails_closed_before_deployment_seal() -> None:
    config = _raw_config()
    config["status"] = "active_mechanism_sealed_awaiting_deployment_seal"
    config["formal_run"][
        "status"
    ] = "blocked_until_live_deployment_profile_and_smoke_seal"
    config["evaluation"][
        "formal_status"
    ] = "blocked_until_new_residual_deployment_graph_live_profile"
    config["evaluation"]["online_smoke_evidence"] = None
    with pytest.raises(
        ExpertManifoldError,
        match="blocked by mechanism or deployment state",
    ):
        runtime_for_mode(config, "formal")


def test_profile_seal_recomputes_macro_instead_of_trusting_passed_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ember.expert_manifold.v6_prior_profile as profile_module

    config = _raw_config()
    checks = {name: True for name in contract_module._PROFILE_CHECKS}
    gate_evidence = {"checks": checks, "derived": 1.0}
    monkeypatch.setattr(
        profile_module,
        "profile_passes",
        lambda _config, _macro: (True, gate_evidence),
    )
    result = {
        "schema_version": contract_module.V6_PRIOR_PROFILE_SCHEMA,
        "passed": True,
        "schedule_macro": 49,
        "retain_weight": False,
        "gates": config["profile_run"]["gates"],
        "gate_evidence": gate_evidence,
        "macro": {"raw": "recomputed"},
        "content_hash_policy": "disabled_by_owner",
    }
    assert contract_module._profile_result_matches(config, result)
    result["gate_evidence"] = {"checks": checks, "derived": 2.0}
    assert not contract_module._profile_result_matches(config, result)


def test_formal_states_cannot_be_asserted_without_retained_artifacts() -> None:
    config = _raw_config()
    config["formal_run"]["status"] = "formal_result_sealed"
    assert not contract_module._formal_state_matches(config)
    config["formal_run"]["status"] = "formal_running_or_resumable"
    assert not contract_module._formal_state_matches(config)


def test_deployment_checkpoint_requires_active_authority_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _raw_config()
    source = {"source": "sealed"}
    historical = {"writer_state": {"state_value_count": 12_064_064}}
    configured_writer = (
        Path(config["initialization"]["checkpoint"]) / "writer.safetensors"
    )
    commit = "a" * 40
    contract = {
        "run_schema": contract_module.V6_PRIOR_RUN_SCHEMA,
        "mode": "formal",
        "git_commit": commit,
        "config": {
            "path": str(contract_module.V6_PRIOR_CANONICAL_CONFIG),
            "schema": contract_module.V6_PRIOR_CONFIG_SCHEMA,
            "bytes": 1,
        },
        "source": source,
        "initialization": {
            "mode": "strict_historical_v6_macro400_all_frozen",
            "checkpoint": str(configured_writer),
            "writer_state_tensor_count": 600,
            "writer_state_value_count": 12_064_064,
            "residual_memory": "fresh_zero_then_memory_only_exact_resume",
        },
        "condition_feature": config["condition_feature"],
        "program_residual": config["program_residual"],
        "update": config["update"],
        "ownership": inference_module._expected_residual_ownership(config),
        "world_size": 6,
        "rank_topology": [{} for _ in range(6)],
        "content_hash_policy": "disabled_by_owner",
    }
    monkeypatch.setattr(
        inference_module,
        "git_commit_in_active_authority_lineage",
        lambda value: value == commit,
    )
    assert inference_module._residual_contract_matches(
        contract, config, source, historical, configured_writer
    )
    contract["git_commit"] = "b" * 40
    assert not inference_module._residual_contract_matches(
        contract, config, source, historical, configured_writer
    )


def test_evaluation_profile_only_cannot_seal_deployment() -> None:
    config = _raw_config()
    evaluation = config["evaluation"]
    evaluation["formal_status"] = "sealed_from_live_residual_deployment_profile"
    evaluation["online_smoke_evidence"] = {
        "path": "runs/outputs/profile/writer_generation_profile.json",
        "bytes": 1,
        "schema": "ember_pi05_writer_generation_profile_v1",
        "run_commit": "e" * 40,
        "writer_model_batch_size": 32,
    }
    assert not contract_module._evaluation_artifact_matches(config)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("initialization", "checkpoint", "runs/outputs/not-the-frozen-v6"),
        ("data", "sampler_seed", 1),
        ("data", "teacher_video_seed", 1),
        ("data", "counterfactual_seed", 1),
        ("condition_feature", "projection_seed", 1),
        ("update", "relative_damping", 0.02),
    ),
)
def test_scientific_authorities_and_schedules_are_fixed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    key: str,
    value: object,
) -> None:
    config = _raw_config()
    config[section][key] = value
    with pytest.raises(ExpertManifoldError, match="sealed contract"):
        _load_mutation(tmp_path, monkeypatch, config)


def _formal_ready_config() -> dict:
    config = _raw_config()
    config["profile_run"]["status"] = "sealed_from_live_a40_macro49_profile"
    config["profile_run"]["artifact_evidence"] = {"unit": "sealed"}
    config["status"] = "active_deployment_sealed_formal_ready"
    config["formal_run"]["status"] = "ready_after_live_mechanism_and_deployment_seals"
    config["evaluation"][
        "formal_status"
    ] = "sealed_from_live_residual_deployment_profile"
    config["evaluation"]["online_smoke_evidence"] = {"unit": "sealed"}
    return config


def _args(
    *, resume: int | None, stop: int | None, mode: str = "formal"
) -> argparse.Namespace:
    return argparse.Namespace(
        mode=mode,
        resume=(
            None
            if resume is None
            else Path("/tmp/run/checkpoints") / f"macro_{resume:08d}"
        ),
        stop_after_macro=stop,
        num_workers=2,
    )


def _context() -> DistributedContext:
    return DistributedContext(0, 0, 6, torch.device("cpu"))


def test_formal_segments_must_stop_at_every_predeclared_decision_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "residual_git_state",
        lambda _root: {
            "commit": "a" * 40,
            "authority_commit": "a" * 40,
            "authority_contains_commit": True,
            "dirty_paths": [],
        },
    )
    monkeypatch.setattr(
        runtime_module,
        "read_json",
        lambda _path: {"git": {"commit": "a" * 40}},
    )
    config = _formal_ready_config()
    assert runtime_for_mode(config, "formal") == (50, (10, 25, 50), 0)
    for start, stop in ((0, 10), (10, 25), (25, 50)):
        segment = _resolve_segment(
            _args(resume=None if start == 0 else start, stop=stop),
            config,
            _context(),
        )
        assert (segment.start_macro, segment.stop_macro) == (start, stop)
    for start, stop in ((0, 25), (0, 50), (10, 50), (25, 25)):
        with pytest.raises(ExpertManifoldError, match="sealed segment"):
            _resolve_segment(
                _args(resume=None if start == 0 else start, stop=stop),
                config,
                _context(),
            )


def test_profile_is_one_fresh_macro49_and_cannot_retain_or_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "residual_git_state",
        lambda _root: {
            "commit": "b" * 40,
            "authority_commit": "b" * 40,
            "authority_contains_commit": True,
            "dirty_paths": [],
        },
    )
    sealed = _raw_config()
    with pytest.raises(ExpertManifoldError, match="not in its launch state"):
        runtime_for_mode(sealed, "mechanism-profile")
    config = deepcopy(sealed)
    config["status"] = "active_implementation_cpu_sealed_awaiting_live_a40_profile"
    config["profile_run"]["status"] = "awaiting_live_a40_macro49_profile"
    config["profile_run"]["artifact_evidence"] = None
    config["formal_run"]["status"] = "blocked_until_live_profile_passes_and_is_sealed"
    segment = _resolve_segment(
        _args(resume=None, stop=1, mode="mechanism-profile"), config, _context()
    )
    assert (
        segment.total_macros,
        segment.schedule_origin,
        segment.checkpoint_macros,
    ) == (
        1,
        49,
        (),
    )
    with pytest.raises(ExpertManifoldError, match="sealed segment"):
        _resolve_segment(
            _args(resume=None, stop=2, mode="mechanism-profile"), config, _context()
        )


def test_exact_resume_keeps_original_frozen_commit_after_authority_advances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = "c" * 40
    advanced = "d" * 40
    state = {
        "commit": original,
        "authority_commit": advanced,
        "authority_contains_commit": True,
        "dirty_paths": [],
    }
    monkeypatch.setattr(runtime_module, "residual_git_state", lambda _root: state)
    monkeypatch.setattr(
        runtime_module,
        "read_json",
        lambda _path: {"git": {"commit": original}},
    )
    segment = _resolve_segment(
        _args(resume=10, stop=25),
        _formal_ready_config(),
        _context(),
    )
    assert (segment.start_macro, segment.stop_macro) == (10, 25)
    with pytest.raises(ExpertManifoldError, match="sealed segment"):
        _resolve_segment(
            _args(resume=None, stop=10),
            _formal_ready_config(),
            _context(),
        )
    state["authority_contains_commit"] = False
    with pytest.raises(ExpertManifoldError, match="sealed segment"):
        _resolve_segment(
            _args(resume=10, stop=25),
            _formal_ready_config(),
            _context(),
        )


def test_checkpoint_owns_only_fp32_program_memory_and_preserves_zero_lineage() -> None:
    ownership = V6PriorOwnership(10_775_296, 523, 600)
    writer = SimpleNamespace(
        condition_feature=SimpleNamespace(
            projection=torch.empty((2, 128, 256), dtype=torch.float32, device="meta")
        ),
        program_memory=SimpleNamespace(
            value=torch.empty((256, 320, 256), dtype=torch.float32, device="meta")
        ),
    )
    observed = _ownership_contract(ownership, writer)
    assert observed["historical_v6_base"]["checkpoint_owned"] is False
    assert observed["fixed_projection"]["persistent"] is False
    assert observed["program_residual_memory"] == {
        "shape": [256, 320, 256],
        "dtype": "torch.float32",
        "value_count": 20_971_520,
        "trainable": False,
        "manual_update": True,
        "checkpoint_owned": True,
        "deployment_owned": True,
    }
    run = {
        "schema_version": "launch-v1",
        "mode": "formal",
        "git": {"commit": "c" * 40},
        "config": {"path": "/repo/config.json", "schema": "config-v1", "bytes": 1},
        "source": {"checkpoint": "/source"},
        "initialization": {
            "mode": "strict_historical_v6_macro400_all_frozen",
            "checkpoint": "/v6/writer.safetensors",
            "writer_state_tensor_count": 600,
            "writer_state_value_count": 10_000,
            "residual_memory": "fresh_zero_then_memory_only_exact_resume",
        },
        "condition_feature": {"kind": "fixed"},
        "program_residual": {"value_count": 20_971_520},
        "update": {"kind": "full48"},
        "ownership": observed,
        "runtime": {"world_size": 6, "rank_topology": [{"rank": i} for i in range(6)]},
    }
    checkpoint = checkpoint_contract(run)
    assert checkpoint["initialization"]["residual_memory"] == (
        "fresh_zero_then_memory_only_exact_resume"
    )
    assert checkpoint["ownership"] == observed
    assert cursor_contract(_raw_config(), 25) == {
        "next_macro": 25,
        "task_visits_per_task": 25,
        "sampler_seed": 20260721,
        "teacher_video_seed": 20260722,
        "counterfactual_seed": 20260809,
        "counterfactual_phase": 1,
        "videos_per_task_visit": 1,
        "action_queries_per_task": 20,
        "full48_order": "correct_0_to_23_then_negative_0_to_23",
    }
