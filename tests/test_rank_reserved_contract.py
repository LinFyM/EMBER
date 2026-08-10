from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import ember.expert_manifold.rank_reserved_contract as contract_module
import ember.expert_manifold.rank_reserved_authority as authority_module
import ember.expert_manifold.v6_prior_training as old_training
from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.inference import (
    load_expert_manifold_deployment_config,
)
from ember.expert_manifold.rank_reserved_contract import (
    RANK_RESERVED_CANONICAL_CONFIG,
    RANK_RESERVED_CONFIG_SCHEMA,
    RANK_RESERVED_PROGRAM_REFERENCE,
    load_rank_reserved_config,
    load_rank_reserved_program_reference,
    rank_reserved_output_path,
    rank_reserved_asset,
    seal_rank_reserved_deployment,
)
from ember.pi05_eval.rank_reserved_launch import _tracked_authority_head


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_rank_reserved_config_is_load_only_and_inherits_the_frozen_v6_owner() -> None:
    config = load_rank_reserved_config(RANK_RESERVED_CANONICAL_CONFIG)

    assert config["schema_version"] == RANK_RESERVED_CONFIG_SCHEMA
    assert config["method"]["training_enabled"] is False
    assert config["compiler"]["qv_base_rank"] == 14
    assert config["compiler"]["qv_residual_rank"] == 2
    assert config["compiler"]["action_path"].startswith("unchanged_full_rank16")
    assert config["writer"]["frame_stride"] == 5
    assert config["information_wall"]["teacher_action_reads"] == 0
    assert not {
        "reconciliation",
        "update",
        "optimization",
        "profile_run",
        "formal_run",
    }.intersection(config)

    old_reward = REPO_ROOT / "configs/pi05_v6_reward_credit_program_cotangent_v1.json"
    with pytest.raises(ExpertManifoldError, match="read-only result evidence"):
        load_expert_manifold_deployment_config(old_reward)


def test_deployment_seal_requires_the_tracked_canonical_branch_head() -> None:
    commit = "a" * 40
    tracked = {
        "branch": "codex/bci-continuation",
        "commit": commit,
        "upstream": "origin/codex/bci-continuation",
        "upstream_commit": commit,
        "authority_ref": "origin/codex/bci-continuation",
        "authority_commit": commit,
        "authority_contains_commit": True,
        "dirty_paths": [],
    }
    assert _tracked_authority_head(tracked)

    detached = {**tracked, "branch": "", "upstream": None}
    assert not _tracked_authority_head(detached)
    assert not _tracked_authority_head({**tracked, "authority_commit": "b" * 40})
    assert not _tracked_authority_head({**tracked, "dirty_paths": [" M config"]})


def test_cycle1_program_reference_reads_only_the_registered_tensor_header() -> None:
    reference = load_rank_reserved_program_reference(RANK_RESERVED_PROGRAM_REFERENCE)

    assert reference["program_memory"]["bytes"] == 83_886_184
    assert reference["program_memory"]["key"] == "program_memory.value"
    assert reference["program_memory"]["dtype"] == "F32"
    assert reference["program_memory"]["shape"] == [256, 320, 256]
    assert reference["source_checkpoint"]["next_macro"] == 1
    assert reference["source_checkpoint"]["metrics_rows"] == 1
    assert reference["source_checkpoint"]["training_commit"] == (
        "e3857f73ce92fa7f790a7e49f8166d7e5ef5b9e5"
    )
    assert reference["source_checkpoint"]["world_size"] == 6
    assert reference["optimizer_or_rng_read"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (("optimizer_or_rng_read", True), ("copy_or_symlink", True)),
)
def test_program_reference_rejects_expanded_checkpoint_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: bool,
) -> None:
    reference = json.loads(RANK_RESERVED_PROGRAM_REFERENCE.read_text())
    reference[field] = value
    path = tmp_path / "program_reference.json"
    path.write_text(json.dumps(reference), encoding="utf-8")
    monkeypatch.setattr(
        authority_module,
        "RANK_RESERVED_PROGRAM_REFERENCE",
        path,
    )

    with pytest.raises(ExpertManifoldError, match="Program reference changed"):
        load_rank_reserved_program_reference(path)


def test_rank_reserved_assets_are_exactly_macro0_or_cycle1() -> None:
    config = load_rank_reserved_config(RANK_RESERVED_CANONICAL_CONFIG)
    macro0 = REPO_ROOT / config["assets"]["macro0"]["checkpoint"]
    cycle1 = REPO_ROOT / config["assets"]["cycle1"]["checkpoint"]

    zero = rank_reserved_asset(config, macro0)
    reward = rank_reserved_asset(config, cycle1)
    assert (zero["method_macro"], zero["enable_program_residual"]) == (0, False)
    assert (reward["method_macro"], reward["enable_program_residual"]) == (
        1,
        True,
    )
    with pytest.raises(ExpertManifoldError, match="not registered"):
        rank_reserved_asset(config, RANK_RESERVED_CANONICAL_CONFIG)


def test_rank_reserved_config_rejects_an_unregistered_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = json.loads(RANK_RESERVED_CANONICAL_CONFIG.read_text())
    modified = deepcopy(config)
    modified["compiler"]["qv_base_rank"] = 15
    path = tmp_path / RANK_RESERVED_CANONICAL_CONFIG.name
    path.write_text(json.dumps(modified), encoding="utf-8")
    monkeypatch.setattr(contract_module, "RANK_RESERVED_CANONICAL_CONFIG", path)

    with pytest.raises(ExpertManifoldError, match="config changed"):
        load_rank_reserved_config(path)


def test_rank_reserved_output_path_accepts_only_the_frozen_runs_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = tmp_path / "worktree"
    canonical = tmp_path / "canonical"
    (canonical / "runs/outputs/registered").mkdir(parents=True)
    worktree.mkdir()
    (worktree / "runs").symlink_to(canonical / "runs", target_is_directory=True)
    artifact = canonical / "runs/outputs/registered/result.json"
    artifact.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(authority_module, "REPO_ROOT", worktree)

    observed = rank_reserved_output_path(
        "runs/outputs/registered/result.json",
        label="test artifact",
        expected_bytes=2,
        require_file=True,
    )
    assert observed == artifact.resolve()

    outside = tmp_path / "outside"
    outside.mkdir()
    (canonical / "runs/outputs/escape").symlink_to(
        outside,
        target_is_directory=True,
    )
    with pytest.raises(ExpertManifoldError, match="escaped canonical outputs"):
        rank_reserved_output_path(
            "runs/outputs/escape/result.json",
            label="escaped artifact",
        )
    with pytest.raises(ExpertManifoldError, match="escaped canonical outputs"):
        rank_reserved_output_path(
            "runs/outputs/../secret",
            label="traversal artifact",
        )


def test_retired_reward_training_fails_before_distributed_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(**_kwargs):
        raise AssertionError("distributed runtime must not initialize")

    monkeypatch.setattr(old_training, "initialize_distributed", forbidden)
    with pytest.raises(ExpertManifoldError, match="training is retired"):
        old_training.train(SimpleNamespace())


def test_deployment_seal_only_updates_the_registered_evidence_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / RANK_RESERVED_CANONICAL_CONFIG.name
    raw = json.loads(RANK_RESERVED_CANONICAL_CONFIG.read_text(encoding="utf-8"))
    path.write_text(json.dumps(raw), encoding="utf-8")
    evidence = {
        "schema": "ember_pi05_v6_qv_rank_reserved_deployment_seal_v1",
        "run_commit": "a" * 40,
        "writer_model_batch_size": 8,
        "profile": {"path": "profile.json", "bytes": 1},
        "vertical": {"path": "vertical.json", "bytes": 2},
    }
    monkeypatch.setattr(contract_module, "RANK_RESERVED_CANONICAL_CONFIG", path)
    monkeypatch.setattr(
        contract_module,
        "load_rank_reserved_config",
        lambda selected: json.loads(selected.read_text(encoding="utf-8")),
    )
    monkeypatch.setattr(
        contract_module,
        "build_rank_reserved_deployment_evidence",
        lambda _config, require_run_commit: evidence,
    )
    monkeypatch.setattr(
        contract_module,
        "_deployment_evidence_matches",
        lambda value: value == evidence,
    )

    observed = seal_rank_reserved_deployment(path, require_run_commit="a" * 40)

    sealed = json.loads(path.read_text(encoding="utf-8"))
    assert observed == evidence
    assert sealed["status"] == ("sealed_from_live_a40_rank_reserved_deployment_profile")
    assert sealed["evaluation"]["formal_status"] == sealed["status"]
    assert sealed["evaluation"]["online_smoke_evidence"] == evidence
