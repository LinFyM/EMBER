from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from ember.expert_manifold.contract import ExpertManifoldError, load_task_expert_config
from ember.expert_manifold.inference import (
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


def test_pick_gc_world4_deployment_is_sealed_from_its_live_evidence() -> None:
    config = load_v6_prior_config(CONFIG)
    evaluation = config["evaluation"]
    assert config["status"] == "active_formal_ready"
    assert config["profile_run"]["expected_world_size"] == 4
    assert config["profile_run"]["tasks_per_rank"] == 6
    assert config["profile_run"]["status"] == (
        "sealed_from_live_a40_fresh0_to1_profile"
    )
    profile = config["profile_run"]["artifact_evidence"]
    assert profile["run_commit"] == "09bbed368b81f6ba77bf86636340f2ad27ee3be3"
    assert profile["world_size"] == 4
    assert profile["tasks_per_rank"] == 6
    assert profile["passed"] is True
    assert profile["completion"]["passed"] is True
    assert config["formal_run"]["status"] == "ready_after_live_profile_seal"
    assert evaluation["throughput_policy"] == (
        "highest_measured_batch_throughput_with_device_memory_headroom"
    )
    assert evaluation["required_writer_model_batch_sizes"] == [8, 16, 32]
    assert evaluation["formal_status"] == (
        "sealed_from_live_pick_gc_deployment_profile"
    )
    evidence = evaluation["online_smoke_evidence"]
    assert evidence["run_commit"] == "717b561518b9a654f81fd9d27643f7272ddf93ff"
    assert evidence["writer_model_batch_size"] == 32
    assert evidence["completion"]["passed"] is True


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
    ownership = {
        "historical_v6_base": {
            "state_tensor_count": 600,
            "parameter_tensor_count": 523,
            "parameter_count": 10_775_296,
            "trainable_parameter_count": 0,
            "checkpoint_owned": False,
            "deployment_owned": True,
        },
        "fixed_projection": {
            "shape": [2, 128, 3072],
            "dtype": "torch.float32",
            "trainable": False,
            "persistent": False,
            "checkpoint_owned": False,
        },
        "fixed_policy_innovation_encoder": {
            "feature_width": 3072,
            "phase_slots": 16,
            "fixed_suffix_noise_shape": [50, 32],
            "trainable_parameter_count": 0,
            "persistent_state_tensor_count": 0,
            "checkpoint_owned": False,
            "deployment_owned": True,
        },
        "program_residual_memory": {
            "shape": [256, 320, 256],
            "dtype": "torch.float32",
            "value_count": 20_971_520,
            "trainable": False,
            "manual_update": True,
            "checkpoint_owned": True,
            "deployment_owned": True,
        },
        "source_policy_trainable_parameter_count": 0,
        "optimizer": "not_instantiated",
        "scheduler": "not_instantiated",
        "scaler": "not_instantiated",
    }
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
        },
        "condition_feature": config["condition_feature"],
        "program_residual": config["program_residual"],
        "update": config["update"],
        "ownership": ownership,
        "world_size": 4,
        "rank_topology": [{"rank": rank} for rank in range(4)],
        "content_hash_policy": "disabled_by_owner",
    }
    return {
        "checkpoint_schema": V6_PRIOR_CHECKPOINT_SCHEMA,
        "next_macro": 10,
        "metrics_rows": 10,
        "world_size": 4,
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
    (checkpoint / "manifest.json").write_text("{}", encoding="utf-8")
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
        "expected_world_size": 4,
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
