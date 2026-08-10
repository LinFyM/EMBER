from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import ember.pi05_eval.rank_reserved_gate as gate_module
from ember.expert_manifold.rank_reserved_contract import (
    RANK_RESERVED_ADAPTER_SCHEMA,
    RANK_RESERVED_CANONICAL_CONFIG,
    RANK_RESERVED_CONFIG_SCHEMA,
    RANK_RESERVED_PROGRAM_REFERENCE,
    RANK_RESERVED_PROGRAM_REFERENCE_SCHEMA,
    load_rank_reserved_config,
)
from ember.pi05_assets import Pi05EvaluationError


def _config() -> dict:
    return {
        "schema_version": RANK_RESERVED_CONFIG_SCHEMA,
        "assets": {
            "macro0": {
                "kind": "v6_qv_rank14_zero_program_load_only",
                "method_macro": 0,
                "checkpoint": "assets/macro0",
                "enable_program_residual": False,
            },
            "cycle1": {
                "kind": "v6_qv_rank14_plus2_reward_program_load_only",
                "method_macro": 1,
                "checkpoint": "assets/cycle1.json",
                "enable_program_residual": True,
            },
        },
        "evaluation": {
            "formal_status": "sealed_from_live_a40_rank_reserved_deployment_profile",
            "registered_roots": {
                "macro0_correct": "formal/macro0-correct",
                "cycle1_correct": "formal/cycle1-correct",
                "cycle1_controls": {
                    condition: f"formal/cycle1-{condition}"
                    for condition in (
                        "same_task_other",
                        "cross_suite_wrong",
                        "shuffled",
                        "reversed",
                        "no_video",
                    )
                },
            },
        },
    }


def _arguments(
    config_path: Path, checkpoint: Path, condition: str
) -> argparse.Namespace:
    return argparse.Namespace(
        mode="formal",
        expert_manifold_config=config_path,
        expert_manifold_checkpoint=checkpoint,
        expert_manifold_video_condition=condition,
    )


def _contract(
    *,
    repo_root: Path,
    output: Path,
    config_path: Path,
    macro: int,
    condition: str,
) -> dict:
    config = _config()
    asset = config["assets"]["macro0" if macro == 0 else "cycle1"]
    return {
        "mode": "formal",
        "role": "validation",
        "output_dir": str(output.resolve()),
        "git": {"commit": "a" * 40, "dirty_paths": []},
        "tasks": [
            {
                "split_role": "validation",
                "init_state_ids": tuple(range(50)),
            }
            for _ in range(8)
        ],
        "adapter": {
            "schema_version": RANK_RESERVED_ADAPTER_SCHEMA,
            "config": {
                "path": str(config_path),
                "schema": RANK_RESERVED_CONFIG_SCHEMA,
            },
            "writer_asset": {
                "kind": asset["kind"],
                "method_macro": macro,
                "checkpoint": str((repo_root / asset["checkpoint"]).resolve()),
                "enable_program_residual": asset["enable_program_residual"],
            },
            "video_condition": condition,
            "video_schedule": {"sampling_mode": "without_replacement"},
        },
        "writer_lora_cache": {
            "root": str((output / "writer_lora_cache").resolve()),
            "identity": {"implementation_commit": "a" * 40},
        },
    }


@pytest.fixture
def sealed_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict]:
    config_path = tmp_path / "configs/rank-reserved.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps({"schema_version": RANK_RESERVED_CONFIG_SCHEMA}),
        encoding="utf-8",
    )
    config = _config()
    monkeypatch.setattr(
        gate_module,
        "_root",
        lambda relative: (tmp_path / relative).resolve(),
    )
    monkeypatch.setattr(
        gate_module,
        "_asset_checkpoint",
        lambda selected, name: (
            tmp_path / selected["assets"][name]["checkpoint"]
        ).resolve(),
    )
    monkeypatch.setattr(
        gate_module,
        "load_rank_reserved_config",
        lambda path: config if path == config_path.resolve() else None,
    )
    monkeypatch.setattr(
        gate_module,
        "git_state_is_clean_pushed_or_frozen_authority",
        lambda _git: True,
    )
    monkeypatch.setattr(
        gate_module,
        "_implementation_lineage_matches",
        lambda _config, _commit: True,
    )
    return tmp_path, config_path, config


def test_macro0_requires_the_exact_registered_correct_root(
    sealed_gate: tuple[Path, Path, dict],
) -> None:
    repo_root, config_path, config = sealed_gate
    output = repo_root / config["evaluation"]["registered_roots"]["macro0_correct"]
    checkpoint = repo_root / config["assets"]["macro0"]["checkpoint"]
    args = _arguments(config_path, checkpoint, "correct")
    contract = _contract(
        repo_root=repo_root,
        output=output,
        config_path=config_path,
        macro=0,
        condition="correct",
    )
    gate_module.validate_registered_rank_reserved_output(args, output, contract)

    wrong = repo_root / "formal/unregistered"
    contract["output_dir"] = str(wrong.resolve())
    with pytest.raises(Pi05EvaluationError, match="pre-registered root"):
        gate_module.validate_registered_rank_reserved_output(args, wrong, contract)


def test_cycle1_and_controls_require_the_ordered_behavior_gates(
    sealed_gate: tuple[Path, Path, dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, config_path, config = sealed_gate
    checkpoint = repo_root / config["assets"]["cycle1"]["checkpoint"]
    correct = repo_root / config["evaluation"]["registered_roots"]["cycle1_correct"]
    args = _arguments(config_path, checkpoint, "correct")
    contract = _contract(
        repo_root=repo_root,
        output=correct,
        config_path=config_path,
        macro=1,
        condition="correct",
    )
    monkeypatch.setattr(
        gate_module,
        "rank_reserved_macro0_evidence",
        lambda _config: {"passed": False},
    )
    with pytest.raises(Pi05EvaluationError, match="macro0 base gate"):
        gate_module.validate_registered_rank_reserved_output(args, correct, contract)

    monkeypatch.setattr(
        gate_module,
        "rank_reserved_macro0_evidence",
        lambda _config: {"passed": True},
    )
    gate_module.validate_registered_rank_reserved_output(args, correct, contract)

    condition = "reversed"
    control = (
        repo_root
        / config["evaluation"]["registered_roots"]["cycle1_controls"][condition]
    )
    args.expert_manifold_video_condition = condition
    control_contract = _contract(
        repo_root=repo_root,
        output=control,
        config_path=config_path,
        macro=1,
        condition=condition,
    )
    monkeypatch.setattr(
        gate_module,
        "rank_reserved_cycle1_evidence",
        lambda _config: {
            "passed": False,
            "new_rank14_plus2_cycle1": {"commit": "a" * 40},
        },
    )
    with pytest.raises(Pi05EvaluationError, match="cycle1 gate"):
        gate_module.validate_registered_rank_reserved_output(
            args,
            control,
            control_contract,
        )
    monkeypatch.setattr(
        gate_module,
        "rank_reserved_cycle1_evidence",
        lambda _config: {
            "passed": True,
            "new_rank14_plus2_cycle1": {"commit": "a" * 40},
        },
    )
    gate_module.validate_registered_rank_reserved_output(
        args,
        control,
        control_contract,
    )


def test_rank_reserved_registration_rejects_kind_or_enablement_drift(
    sealed_gate: tuple[Path, Path, dict],
) -> None:
    repo_root, config_path, config = sealed_gate
    output = repo_root / config["evaluation"]["registered_roots"]["macro0_correct"]
    checkpoint = repo_root / config["assets"]["macro0"]["checkpoint"]
    args = _arguments(config_path, checkpoint, "correct")
    contract = _contract(
        repo_root=repo_root,
        output=output,
        config_path=config_path,
        macro=0,
        condition="correct",
    )
    contract["adapter"]["writer_asset"]["enable_program_residual"] = True
    with pytest.raises(Pi05EvaluationError, match="pre-registered root"):
        gate_module.validate_registered_rank_reserved_output(args, output, contract)


def test_cycle1_uses_the_tracked_program_asset_not_the_output_root_resolver() -> None:
    config = load_rank_reserved_config(RANK_RESERVED_CANONICAL_CONFIG)

    assert gate_module._asset_checkpoint(config, "cycle1") == (
        RANK_RESERVED_PROGRAM_REFERENCE.resolve()
    )
    with pytest.raises(Pi05EvaluationError, match="escaped canonical outputs"):
        gate_module._root(str(config["assets"]["cycle1"]["checkpoint"]))


def test_config_evidence_identity_allows_another_same_commit_worktree(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "current/configs/rank-reserved.json"
    expected.parent.mkdir(parents=True)
    expected.write_text("same-content", encoding="utf-8")
    other = tmp_path / "frozen/configs/rank-reserved.json"

    record = {
        "path": str(other),
        "bytes": expected.stat().st_size,
        "schema": RANK_RESERVED_CONFIG_SCHEMA,
    }

    assert gate_module._config_record_matches(record, expected)
    record["bytes"] += 1
    assert not gate_module._config_record_matches(record, expected)


def test_program_checkpoint_identity_allows_another_same_commit_worktree(
    tmp_path: Path,
) -> None:
    other = tmp_path / "frozen/configs" / RANK_RESERVED_PROGRAM_REFERENCE.name
    asset = {
        "checkpoint": str(other),
        "manifest": {
            "path": str(other),
            "bytes": RANK_RESERVED_PROGRAM_REFERENCE.stat().st_size,
            "schema": RANK_RESERVED_PROGRAM_REFERENCE_SCHEMA,
        },
    }

    assert gate_module._checkpoint_record_matches(
        asset, RANK_RESERVED_PROGRAM_REFERENCE
    )
    asset["manifest"]["bytes"] += 1
    assert not gate_module._checkpoint_record_matches(
        asset, RANK_RESERVED_PROGRAM_REFERENCE
    )
