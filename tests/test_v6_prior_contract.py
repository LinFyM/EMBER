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
    checkpoint_contract,
    cursor_contract,
    decision_evaluation_contract,
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
    config["status"] = "formal_ready"
    config["profile_run"]["status"] = "sealed_from_live_a40_fresh0_to1_reward_profile"
    config["profile_run"]["artifact_evidence"] = {"synthetic": True}
    config["formal_run"]["status"] = "ready_after_live_reward_profile_seal"
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
        resume=resume,
        stop_after_macro=stop,
        num_workers=0,
        output_dir=output_dir,
    )


def _git_state(commit: str = "a" * 40) -> dict:
    return {
        "commit": commit,
        "authority_commit": commit,
        "authority_contains_commit": True,
        "dirty_paths": [],
    }


def test_active_reward_credit_is_profile_only_until_live_full24_seal() -> None:
    config = load_v6_prior_config(V6_PRIOR_CANONICAL_CONFIG)
    assert config["schema_version"] == V6_PRIOR_CONFIG_SCHEMA
    assert config["status"] == "awaiting_live_a40_reward_credit_profile"
    assert runtime_for_mode(config, "mechanism-profile") == (1, (), 0)
    with pytest.raises(ExpertManifoldError, match="blocked by the live profile"):
        runtime_for_mode(config, "formal")
    retired = V6_PRIOR_CANONICAL_CONFIG.with_name(
        "pi05_v6_exact_anchored_reconciliation_program_residual_v3.json"
    )
    with pytest.raises(ExpertManifoldError, match="non-canonical"):
        load_v6_prior_config(retired)
    assert config["method"]["dynamic_value"] == "one_raw_teacher_video_only"
    assert config["method"]["language_only_lora_path"] is False
    assert config["information_wall"]["source_action_reads"] == 0
    assert config["objective"]["old_policy_forwards"] == 0
    assert config["objective"]["learning_epochs"] == 1


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("initialization", "checkpoint", "runs/outputs/not-frozen-v6"),
        ("data", "teacher_video_seed", 1),
        ("environment", "rollouts_per_task", 3),
        ("objective", "flow_mc_samples", 2),
        ("optimization", "reward_replay_chunk_batch_size", 1),
        ("rng", "environment_seed_root", 1),
        ("rng", "policy_noise_seed_root", 1),
        ("rng", "flow_credit_seed_root", 1),
        ("update", "relative_damping", 0.02),
    ),
)
def test_scientific_authorities_and_throughput_recipe_fail_closed(
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


def test_unknown_behavioral_config_fields_are_not_silently_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _raw_config()
    config["objective"]["precision_safety_fallback"] = True
    with pytest.raises(ExpertManifoldError, match="fail-closed contract"):
        _load_mutation(tmp_path, monkeypatch, config)


def test_profile_segment_is_discarded_fresh_zero_to_one_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _raw_config()
    config["profile_run"]["registered_output_root"] = "runs/outputs/profile"
    output = tmp_path / "runs/outputs/profile"
    monkeypatch.setattr(runtime_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        runtime_module, "residual_git_state", lambda _root: _git_state()
    )
    segment = _resolve_segment(
        _args(output_dir=output, resume=None, stop=1, mode="mechanism-profile"),
        config,
        _context(),
    )
    assert (segment.start_macro, segment.stop_macro) == (0, 1)
    assert segment.checkpoint_macros == ()
    for resume, stop in (
        (tmp_path / "checkpoints/macro_00000001", 1),
        (None, 2),
    ):
        with pytest.raises(ExpertManifoldError):
            _resolve_segment(
                _args(
                    output_dir=output,
                    resume=resume,
                    stop=stop,
                    mode="mechanism-profile",
                ),
                config,
                _context(),
            )


def _formal_layout(tmp_path: Path, config: dict) -> tuple[Path, Path, Path, Path]:
    config["formal_run"]["registered_output_root"] = "runs/outputs/formal"
    decision = config["formal_run"]["decision_evaluation"]
    decision["macro0_reference_root"] = "runs/outputs/baseline"
    decision["macro1_registered_root"] = "runs/outputs/macro1"
    decision["macro2_registered_root"] = "runs/outputs/macro2"
    for macro in (1, 2):
        decision[f"macro{macro}_control_registered_roots"] = {
            condition: f"runs/outputs/macro{macro}-{condition}"
            for condition in (
                "same_task_other",
                "cross_suite_wrong",
                "shuffled",
                "reversed",
                "no_video",
            )
        }
    baseline = tmp_path / decision["macro0_reference_root"]
    macro1 = tmp_path / decision["macro1_registered_root"]
    macro2 = tmp_path / decision["macro2_registered_root"]
    output = tmp_path / config["formal_run"]["registered_output_root"]
    baseline.mkdir(parents=True)
    return baseline, macro1, macro2, output


def test_formal_segments_are_exact_zero_to_one_then_gated_one_to_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _formal_ready_config()
    _, macro1, _, output = _formal_layout(tmp_path, config)
    monkeypatch.setattr(runtime_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        runtime_module, "residual_git_state", lambda _root: _git_state()
    )
    fresh = _resolve_segment(
        _args(output_dir=output, resume=None, stop=1), config, _context()
    )
    assert (fresh.start_macro, fresh.stop_macro) == (0, 1)
    assert fresh.checkpoint_macros == (1, 2)
    for stop in (None, 2):
        with pytest.raises(ExpertManifoldError, match="registered frozen segment"):
            _resolve_segment(
                _args(output_dir=output, resume=None, stop=stop),
                config,
                _context(),
            )

    checkpoint = output / "checkpoints/macro_00000001"
    checkpoint.mkdir(parents=True)
    write_json_atomic(output / "run_contract.json", {"git": {"commit": "a" * 40}})
    macro1.mkdir(parents=True)
    monkeypatch.setattr(
        "ember.pi05_eval.reward_credit_gate.load_reward_credit_decision_evidence",
        lambda **_kwargs: {"passed": True},
    )
    resumed = _resolve_segment(
        _args(output_dir=output, resume=checkpoint, stop=2),
        config,
        _context(),
    )
    assert (resumed.start_macro, resumed.stop_macro) == (1, 2)
    assert resumed.continuation_gate_evidence == {"passed": True}


def test_macro1_resume_requires_pre_registered_strict_support_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _formal_ready_config()
    _, macro1, _, output = _formal_layout(tmp_path, config)
    checkpoint = output / "checkpoints/macro_00000001"
    checkpoint.mkdir(parents=True)
    write_json_atomic(output / "run_contract.json", {"git": {"commit": "a" * 40}})
    macro1.mkdir(parents=True)
    monkeypatch.setattr(runtime_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        runtime_module, "residual_git_state", lambda _root: _git_state()
    )
    monkeypatch.setattr(
        "ember.pi05_eval.reward_credit_gate.load_reward_credit_decision_evidence",
        lambda **_kwargs: {"passed": False},
    )
    with pytest.raises(ExpertManifoldError, match="did not pass"):
        _resolve_segment(
            _args(output_dir=output, resume=checkpoint, stop=2),
            config,
            _context(),
        )


def test_macro1_score_trigger_requires_registered_six_arm_before_cycle2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _formal_ready_config()
    _, macro1, _, output = _formal_layout(tmp_path, config)
    checkpoint = output / "checkpoints/macro_00000001"
    checkpoint.mkdir(parents=True)
    write_json_atomic(output / "run_contract.json", {"git": {"commit": "a" * 40}})
    macro1.mkdir(parents=True)
    monkeypatch.setattr(runtime_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        runtime_module, "residual_git_state", lambda _root: _git_state()
    )
    monkeypatch.setattr(
        "ember.pi05_eval.reward_credit_gate.load_reward_credit_decision_evidence",
        lambda **_kwargs: {"passed": True, "six_arm_required": True},
    )
    arguments = _args(output_dir=output, resume=checkpoint, stop=2)
    with pytest.raises(ExpertManifoldError, match="six-arm audit"):
        _resolve_segment(arguments, config, _context())
    for path in config["formal_run"]["decision_evaluation"][
        "macro1_control_registered_roots"
    ].values():
        (tmp_path / path).mkdir(parents=True)
    monkeypatch.setattr(
        "ember.pi05_eval.reward_credit_gate.load_reward_credit_six_arm_evidence",
        lambda **_kwargs: {"goal_passed": False},
    )
    segment = _resolve_segment(arguments, config, _context())
    assert segment.continuation_gate_evidence == {
        "passed": True,
        "six_arm_required": True,
        "six_arm": {"goal_passed": False},
    }


def test_exact_resume_requires_original_frozen_commit_to_remain_authorized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _formal_ready_config()
    _, macro1, _, output = _formal_layout(tmp_path, config)
    original = "c" * 40
    checkpoint = output / "checkpoints/macro_00000001"
    checkpoint.mkdir(parents=True)
    write_json_atomic(output / "run_contract.json", {"git": {"commit": original}})
    macro1.mkdir(parents=True)
    monkeypatch.setattr(runtime_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        runtime_module,
        "residual_git_state",
        lambda _root: {
            **_git_state(original),
            "authority_commit": "d" * 40,
            "authority_contains_commit": True,
        },
    )
    monkeypatch.setattr(
        "ember.pi05_eval.reward_credit_gate.load_reward_credit_decision_evidence",
        lambda **_kwargs: {"passed": True},
    )
    segment = _resolve_segment(
        _args(output_dir=output, resume=checkpoint, stop=2),
        config,
        _context(),
    )
    assert segment.start_macro == 1


def test_run_contract_pre_registers_all_strict_roots_without_cli_overrides() -> None:
    config = _raw_config()
    arguments = SimpleNamespace(mode="formal")
    record = decision_evaluation_contract(arguments, config)
    decision = config["formal_run"]["decision_evaluation"]
    assert (
        Path(record["macro0_reference_root"]).resolve()
        == (
            V6_PRIOR_CANONICAL_CONFIG.parents[1] / decision["macro0_reference_root"]
        ).resolve()
    )
    assert (
        Path(record["macro1_registered_root"]).resolve()
        == (
            V6_PRIOR_CANONICAL_CONFIG.parents[1] / decision["macro1_registered_root"]
        ).resolve()
    )
    assert (
        Path(record["macro2_registered_root"]).resolve()
        == (
            V6_PRIOR_CANONICAL_CONFIG.parents[1] / decision["macro2_registered_root"]
        ).resolve()
    )
    for macro in (1, 2):
        for condition, path in decision[
            f"macro{macro}_control_registered_roots"
        ].items():
            assert (
                Path(
                    record[f"macro{macro}_control_registered_roots"][condition]
                ).resolve()
                == (V6_PRIOR_CANONICAL_CONFIG.parents[1] / path).resolve()
            )
    assert record["active_for_this_invocation"] is True
    arguments.mode = "mechanism-profile"
    assert (
        decision_evaluation_contract(arguments, config)["active_for_this_invocation"]
        is False
    )


def test_cursor_records_reward_rollouts_flow_panels_and_empty_pending_queues() -> None:
    config = _raw_config()
    cursor = cursor_contract(config, 2)
    assert cursor["completed_full24_cycles"] == 2
    assert cursor["global_rollouts"] == 192
    assert cursor["rollouts_per_task"] == 8
    assert cursor["flow_panels_per_task"] == 2
    assert cursor["assimilated_rows"] == 96
    assert cursor["full48_order"] == "correct_0_to_23_then_negative_0_to_23"
    assert all(
        cursor[name] == 0
        for name in (
            "pending_environment_episodes",
            "pending_policy_action_chunks",
            "pending_replay_microbatches",
        )
    )


def test_checkpoint_contract_owns_only_program_and_training_precision() -> None:
    ownership = V6PriorOwnership(10_775_296, 523, 600)
    writer = SimpleNamespace(
        condition_feature=SimpleNamespace(
            projection=torch.empty((2, 128, 256), dtype=torch.float32)
        ),
        program_memory=SimpleNamespace(
            value=torch.empty((256, 320, 256), dtype=torch.float32, device="meta")
        ),
    )
    reconciliation = SimpleNamespace(
        precision=torch.empty((256, 256), dtype=torch.float64, device="meta")
    )
    observed = _ownership_contract(ownership, writer, reconciliation)
    assert observed["program_residual_memory"]["checkpoint_owned"] is True
    assert observed["reconciliation_precision"]["deployment_owned"] is False
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
            "writer_state_value_count": 12_064_064,
            "residual_memory": (
                "fresh_zero_and_identity_reconciliation_then_joint_exact_resume"
            ),
        },
        "condition_feature": {"kind": "fixed"},
        "program_residual": {"value_count": 20_971_520},
        "reconciliation": {"kind": "exact_rls"},
        "update": {"kind": "full48"},
        "environment": {"rollouts_per_task": 4},
        "objective": {"flow_mc_samples": 4},
        "rng": {"scheme": "splitmix64"},
        "ownership": observed,
        "runtime": {
            "world_size": 6,
            "rank_topology": [{"rank": rank} for rank in range(6)],
        },
    }
    checkpoint = checkpoint_contract(run)
    assert checkpoint["environment"] == run["environment"]
    assert checkpoint["objective"] == run["objective"]
    assert checkpoint["rng"] == run["rng"]
    assert "optimizer" not in checkpoint
