from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from ember.expert_manifold.contract import ExpertManifoldError, load_task_expert_config
from ember.expert_manifold.inference import (
    _expected_residual_ownership,
    _trained_writer_asset,
    inspect_v6_prior_writer_asset,
)
from ember.expert_manifold.v6_prior_checkpoint import V6_PRIOR_CHECKPOINT_SCHEMA
from ember.expert_manifold.v6_prior_contract import (
    V6_PRIOR_CANONICAL_CONFIG,
    V6_PRIOR_CONFIG_SCHEMA,
    V6_PRIOR_RUN_SCHEMA,
    load_v6_prior_config,
)
from ember.expert_manifold.v6_prior_run_contract import cursor_contract
from ember.pi05_source_checkpoint import read_json


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = V6_PRIOR_CANONICAL_CONFIG


def test_active_cveg_and_prior_candidate_guard_nonpasses_remain_sealed() -> None:
    cveg = load_v6_prior_config(CONFIG)
    assert cveg["status"] == "active_formal_ready"
    assert cveg["schema_version"] == V6_PRIOR_CONFIG_SCHEMA
    assert cveg["method"]["name"] == (
        "frozen_v6_cross_video_equivariant_candidate_guard"
    )
    assert cveg["data"]["videos_per_task_per_macro"] == 1
    assert cveg["data"]["training_companion_videos_per_task_per_macro"] == 1
    assert cveg["information_wall"]["deployment_companion_video_count"] == 0
    profile = cveg["profile_run"]["artifact_evidence"]
    assert profile["passed"] is True
    assert profile["checks_passed"] == 22
    assert profile["equivariance_rank"] == 24
    assert profile["blind_equivariance_to_primary_motion_ratio"] < 1e-5
    assert profile["final_equivariance_to_primary_motion_ratio"] < 1e-5
    assert profile["negative_null_per_kind"] == {
        "wrong": 8,
        "shuffled": 8,
        "reversed": 8,
    }
    assert cveg["formal_run"]["status"] == "ready_after_live_profile_seal"
    assert cveg["formal_run"]["artifact_evidence"] is None
    assert cveg["evaluation"]["formal_status"] == (
        "sealed_from_live_cveg_deployment_smoke"
    )
    smoke = cveg["evaluation"]["online_smoke_evidence"]
    assert smoke["writer_model_batch_size"] == 32
    assert smoke["longest_sampled_video_frames"] == 67
    assert smoke["oom_count"] == 0
    assert smoke["nonfinite_count"] == 0

    profile = read_json(
        REPO_ROOT
        / "runs/outputs/pi05_npcg_constraint_precision_reprofile_macro0_"
        "r3_b20_4156012_20260812/mechanism_profile.json"
    )
    assert profile["passed"] is True
    assert len(profile["gate_evidence"]["checks"]) == 20
    assert all(profile["gate_evidence"]["checks"].values())
    assert profile["gate_evidence"]["negative_null_per_kind"] == {
        "reversed": 8,
        "shuffled": 8,
        "wrong": 8,
    }
    completion = read_json(
        REPO_ROOT
        / "runs/outputs/pi05_npcg_negative_preserving_candidate_guard_"
        "formal_fresh0to5_r3_b20_f8491e9_retry1_20260812/completion.json"
    )
    assert completion["completed_macro"] == 5
    assert completion["metrics_rows"] == 5
    result = read_json(
        REPO_ROOT
        / "runs/outputs/pi05_npcg_negative_preserving_candidate_guard_"
        "correct400_noreplacement_seed7_macro0005_5235d05_gpu02_retry1_"
        "20260812/npcg_formal_decision_evidence.json"
    )
    assert result["strict_correct400"]["successes"] == 135
    assert result["strict_correct400"]["breadth"] == 5
    assert result["paired_old134_to_npcg"]["retained_success"] == 117
    assert result["paired_old134_to_npcg"]["gained"] == 18
    assert result["paired_old134_to_npcg"]["lost"] == 17
    assert result["decision"]["passed"] is False
    assert result["decision"]["exact_resume_macro5_to10_authorized"] is False

    work_queue = read_json(
        REPO_ROOT
        / "runs/outputs/pi05_wqpcug_work_queue_candidate_guard_full24_"
        "reprofile_macro0_r3_b20_d799758_20260812/mechanism_profile.json"
    )
    assert work_queue["passed"] is False
    evidence = work_queue["gate_evidence"]
    assert [name for name, passed in evidence["checks"].items() if not passed] == [
        "negative_null"
    ]
    assert evidence["paired_outcomes"]["paired_states"] == 48
    assert evidence["paired_outcomes"]["discordant_states"] == 7
    assert evidence["paired_outcomes"]["harmful_task_count"] == 3
    assert evidence["negative_to_unprotected_program_motion_ratio"] > 0.15

    pcug_profile = read_json(
        REPO_ROOT
        / "runs/outputs/pi05_pcug_paired_candidate_guard_full24_profile_"
        "macro0_r4_b20_238cab4_20260812/failure.json"
    )
    assert pcug_profile["passed"] is False
    assert pcug_profile["wall_ratio_lower_bound"] > 1.5
    assert pcug_profile["paired_probe_started"] is False
    assert pcug_profile["mechanism_profile_written"] is False

    historical_srtp = json.loads(
        (
            REPO_ROOT
            / "configs/pi05_v6_shared_reward_tangent_projection_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert historical_srtp["status"] == "profile_result_sealed_nonpass"
    srtp = historical_srtp["profile_run"]["artifact_evidence"]
    assert srtp["passed"] is False
    assert srtp["failed_rank_count"] == 3
    assert srtp["mechanism_profile_written"] is False
    assert historical_srtp["formal_run"]["status"] == "blocked_by_profile_nonpass"
    assert historical_srtp["formal_run"]["artifact_evidence"] is None
    assert historical_srtp["evaluation"]["formal_status"] == (
        "not_run_after_profile_nonpass"
    )
    assert historical_srtp["evaluation"]["online_smoke_evidence"] is None

    config = json.loads(
        (
            REPO_ROOT
            / "configs/pi05_v6_success_key_nullspace_consolidation_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert config["status"] == "formal_result_sealed"
    assert config["profile_run"]["allowed_world_sizes"] == [1, 2, 3, 4, 5, 6]
    assert config["profile_run"]["maximum_world_size"] == 6
    profile = config["profile_run"]["artifact_evidence"]
    assert profile["passed"] is True
    assert profile["world_size"] == 3
    assert profile["all_checks_passed"] is True
    assert config["formal_run"]["status"] == "formal_result_sealed"
    result = config["formal_run"]["artifact_evidence"]
    assert result["training"]["completed_macro"] == 5
    assert result["strict_correct400"]["successes"] == 137
    assert result["strict_correct400"]["breadth"] == 7
    assert result["paired_old134"]["retained_gained_lost"] == [121, 16, 13]
    assert result["decision"] == {
        "macro5_gate_passed": False,
        "exact_resume_macro5_to10_authorized": False,
        "six_arm_controls_authorized": False,
        "scope": "retire_sknc_plus_blind_b20_only",
    }
    smoke = config["evaluation"]["online_smoke_evidence"]
    assert smoke["writer_model_batch_size"] == 32
    assert smoke["profile"]["passed"] is True

    osg_path = (
        REPO_ROOT
        / "configs/pi05_v6_on_policy_success_guarded_program_credit_v1.json"
    )
    osg = json.loads(osg_path.read_text(encoding="utf-8"))
    assert osg["status"] == "profile_result_sealed_nonpass"
    profile = osg["profile_run"]["artifact_evidence"]
    assert profile["exit_code"] == 1
    assert profile["watchdog_count"] == 1
    assert profile["production_wall_ratio_lower_bound"] > 1.25
    assert profile["passed"] is False
    assert osg["formal_run"]["status"] == "blocked_by_profile_nonpass"
    assert osg["evaluation"]["formal_status"] == (
        "not_run_after_profile_nonpass"
    )

    pick_gc_path = (
        REPO_ROOT / "configs/pi05_v6_policy_innovation_goal_causal_key_v1.json"
    )
    pick_gc = json.loads(pick_gc_path.read_text(encoding="utf-8"))
    assert pick_gc["status"] == "formal_result_sealed"
    result = pick_gc["formal_run"]["artifact_evidence"]
    assert result["training"]["completed_macro"] == 10
    assert result["strict_correct400"]["results"]["successes"] == 138
    assert result["strict_correct400"]["results"]["breadth"] == 6
    assert result["strict_correct400"]["decision_evidence"]["passed"] is False
    assert result["decision"] == {
        "macro10_gate_passed": False,
        "resume_macro10_to25_authorized": False,
        "six_arm_controls_authorized": False,
        "retained_gained_lost_to_macro0": [118, 20, 16],
        "scope": "retire_pick_gc_plus_blind_offline_source_action_credit_only",
    }


def test_old_expert_asset_config_cannot_enter_residual_runtime() -> None:
    old = REPO_ROOT / "configs/pi05_video_expert_manifold_v1.json"
    with pytest.raises(ExpertManifoldError, match="canonical config path"):
        load_v6_prior_config(old)


def test_task_expert_authority_remains_independent_of_retired_writer_paths(
    tmp_path: Path,
) -> None:
    old = REPO_ROOT / "configs/pi05_video_expert_manifold_v1.json"
    config = json.loads(old.read_text(encoding="utf-8"))
    config["topological_writer"] = {"retired": True}
    config["meta_training"] = {"retired": True}
    path = tmp_path / "task_experts.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert load_task_expert_config(path)["task_experts"]["task_count"] == 24


def test_historical_v6_is_real_load_only_asset_with_exact_zero_residual() -> None:
    config = load_v6_prior_config(CONFIG)
    checkpoint = (REPO_ROOT / config["initialization"]["checkpoint"]).resolve()
    source = read_json(checkpoint.parent.parent / "run_contract.json")["source"]
    asset = inspect_v6_prior_writer_asset(
        config,
        checkpoint,
        source,
        require_formal=False,
    )
    assert asset["kind"] == "historical_v6_macro400_load_only"
    assert asset["source_macro"] == 400
    assert asset["method_macro"] == 0
    assert asset["writer_state"]["state_tensor_count"] == 600
    assert asset["writer_state"]["state_value_count"] == 12_064_064
    assert asset["residual_state"] == {
        "kind": "fresh_elementwise_zero",
        "path": None,
        "bytes": 0,
        "tensor_count": 0,
        "dtype": "torch.float32",
        "shape": [256, 320, 256],
        "value_count": 20_971_520,
    }


def _synthetic_inspection(config: dict, source: dict, checkpoint: Path) -> dict:
    configured = (
        REPO_ROOT / str(config["initialization"]["checkpoint"])
    ).resolve() / "writer.safetensors"
    ownership = _expected_residual_ownership(config)
    contract = {
        "run_schema": V6_PRIOR_RUN_SCHEMA,
        "mode": "formal",
        "git_commit": "a" * 40,
        "config": {
            "path": f"/frozen/{CONFIG.name}",
            "schema": V6_PRIOR_CONFIG_SCHEMA,
            "bytes": 1,
        },
        "source": source,
        "initialization": {
            "mode": "strict_historical_v6_macro400_all_frozen",
            "checkpoint": str(configured),
            "writer_state_tensor_count": 600,
            "writer_state_value_count": 12_064_064,
            "residual_memory": "fresh_zero_then_memory_only_exact_resume",
            "success_key_bank": "fresh_empty_then_exact_resume",
        },
        "condition_feature": config["condition_feature"],
        "program_residual": config["program_residual"],
        "success_key_bank": config["success_key_bank"],
        "update": config["update"],
        "environment": config["environment"],
        "objective": config["objective"],
        "rng": config["rng"],
        "ownership": ownership,
        "world_size": 5,
        "rank_topology": [{"rank": rank} for rank in range(5)],
        "content_hash_policy": "disabled_by_owner",
    }
    return {
        "checkpoint_schema": V6_PRIOR_CHECKPOINT_SCHEMA,
        "next_macro": 10,
        "metrics_rows": 10,
        "world_size": 5,
        "cursor_contract": cursor_contract(config, 10),
        "checkpoint_contract": contract,
        "program_memory": {
            "file": "program_memory.safetensors",
            "key": "program_memory.value",
            "tensor_count": 1,
            "dtype": "torch.float32",
            "shape": [256, 320, 256],
            "value_count": 20_971_520,
            "finite": None,
        },
        "success_key_bank": {
            "file": "success_key_bank.safetensors",
            "keys": [
                "success_key_bank.features",
                "success_key_bank.present",
                "success_key_bank.task_global_ids",
            ],
            "tensor_count": 3,
            "feature_dtype": "torch.float32",
            "feature_shape": [24, 256],
            "feature_value_count": 6144,
            "present_dtype": "torch.uint8",
            "task_global_ids_dtype": "torch.int64",
            "present_count": None,
            "present_ordinals": [],
            "task_global_ids": [],
            "finite": None,
        },
        "payload_value_validation": "deployment_metadata_only",
        "content_hash_policy": "disabled_by_owner",
    }


def test_trained_asset_accepts_only_formal_memory_owner_and_exact_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_v6_prior_config(CONFIG)
    source = {"checkpoint": "/synthetic/source"}
    checkpoint = tmp_path / "checkpoints/macro_00000010"
    checkpoint.mkdir(parents=True)
    (checkpoint / "manifest.json").write_text(
        json.dumps({"world_size": 5}), encoding="utf-8"
    )
    (checkpoint / "program_memory.safetensors").write_bytes(b"formal-memory")
    inspection = _synthetic_inspection(config, source, checkpoint)
    inspection_arguments = {}

    def inspect_checkpoint(
        observed_checkpoint: Path, **arguments: object
    ) -> dict:
        assert observed_checkpoint == checkpoint
        inspection_arguments.update(arguments)
        return inspection

    monkeypatch.setattr(
        "ember.expert_manifold.inference.inspect_v6_prior_checkpoint",
        inspect_checkpoint,
    )
    monkeypatch.setattr(
        "ember.expert_manifold.inference._historical_writer_asset",
        lambda *_args, **_kwargs: {
            "writer_state": {
                "path": "/historical/writer.safetensors",
                "bytes": 1,
                "state_tensor_count": 600,
                "state_value_count": 12_064_064,
            }
        },
    )
    monkeypatch.setattr(
        "ember.expert_manifold.inference.git_commit_in_active_authority_lineage",
        lambda commit: commit == "a" * 40,
    )
    asset = _trained_writer_asset(
        config,
        checkpoint,
        source,
        require_formal=True,
    )
    assert asset["kind"] == "v6_condition_program_residual_checkpoint"
    assert asset["residual_state"]["tensor_count"] == 1
    assert inspection_arguments == {
        "expected_world_size": 5,
        "validate_payload_values": False,
    }

    invalid = deepcopy(inspection)
    invalid["checkpoint_contract"] = deepcopy(inspection["checkpoint_contract"])
    invalid["checkpoint_contract"]["ownership"] = deepcopy(
        inspection["checkpoint_contract"]["ownership"]
    )
    invalid["checkpoint_contract"]["ownership"]["dynamic_anchor"] = {
        "checkpoint_owned": True
    }
    monkeypatch.setattr(
        "ember.expert_manifold.inference.inspect_v6_prior_checkpoint",
        lambda _checkpoint, **_kwargs: invalid,
    )
    with pytest.raises(ExpertManifoldError, match="residual checkpoint changed"):
        _trained_writer_asset(config, checkpoint, source, require_formal=True)

    invalid_cursor = deepcopy(inspection)
    invalid_cursor["cursor_contract"]["teacher_video_seed"] += 1
    monkeypatch.setattr(
        "ember.expert_manifold.inference.inspect_v6_prior_checkpoint",
        lambda _checkpoint, **_kwargs: invalid_cursor,
    )
    with pytest.raises(ExpertManifoldError, match="residual checkpoint changed"):
        _trained_writer_asset(config, checkpoint, source, require_formal=True)
