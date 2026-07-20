from __future__ import annotations

from pathlib import Path

from ember.lora import canonical_contract_sha256, load_lora_contract
from ember.source_base_checkpoint import canonical_hash, sha256_file
from ember.writer.inference import _resolve_writer_stage
from ember.writer.training import load_writer_config
from ember.writer_rl_protocol import load_writer_rl_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_writer_profile_config_seals_measured_formal_launch() -> None:
    config = load_writer_config(REPO_ROOT / "configs/writer_cold_start_v1.json")
    formal = config["formal_run"]
    assert formal["status"] == "sealed"
    assert formal["expected_world_size"] == 8
    assert formal["per_rank_batch_size"] == 384
    assert formal["total_steps"] == 1575
    assert formal["checkpoint_steps"] == [525, 1050, 1575]
    assert config["writer"]["vision_feature_dim"] == 960
    assert config["data"]["demo_indices"] == [0, 49]
    assert config["optimization"]["precision"] == "bfloat16"


def test_writer_inference_distinguishes_cold_and_reward_checkpoints() -> None:
    writer_path = REPO_ROOT / "configs/writer_cold_start_v1.json"
    writer_config = load_writer_config(writer_path)
    lora_sha = canonical_contract_sha256(
        load_lora_contract(REPO_ROOT / writer_config["protocol"]["lora_contract"])
    )
    policy_files = {"model.safetensors": "a" * 64}
    cold_contract = {
        "schema_version": "ember_writer_cold_start_launch_v1",
        "mode": "formal",
        "config_sha256": sha256_file(writer_path),
        "writer": writer_config["writer"],
        "source_policy_files": policy_files,
        "trainable": {"lora_contract_sha256": lora_sha},
    }
    assert _resolve_writer_stage(
        training_contract=cold_contract,
        checkpoint_manifest={
            "contract_sha256": canonical_hash(cold_contract),
            "consumed": {"next_step": 525},
        },
        writer_config=writer_config,
        writer_config_path=writer_path,
        writer_rl_config_path=None,
        policy_files=policy_files,
        lora_contract_sha256=lora_sha,
        require_formal=True,
    ) == ("cold_start", sha256_file(writer_path), 525)

    reward_path = REPO_ROOT / "configs/writer_only_rl_v1.json"
    reward_config = load_writer_rl_config(reward_path)
    reward_contract = {
        "schema_version": "ember_writer_only_rl_launch_v1",
        "mode": "profile",
        "config_sha256": sha256_file(reward_path),
        "protocol": reward_config["protocol"],
        "algorithm": reward_config["algorithm"],
        "environment": reward_config["environment"],
        "source_policy_files": policy_files,
        "trainable": {
            "object": "shared_writer_only",
            "lora_contract_sha256": lora_sha,
            "source_policy_parameter_count": 0,
            "generated_lora_in_place_parameter_count": 0,
            "critic_parameter_count": 0,
        },
        "runtime": {"generated_lora_in_place_updates": False},
    }
    assert _resolve_writer_stage(
        training_contract=reward_contract,
        checkpoint_manifest={
            "schema_version": "ember_writer_only_rl_checkpoint_v1",
            "contract_sha256": canonical_hash(reward_contract),
            "consumed": {"next_update": 9},
        },
        writer_config=writer_config,
        writer_config_path=writer_path,
        writer_rl_config_path=reward_path,
        policy_files=policy_files,
        lora_contract_sha256=lora_sha,
        require_formal=False,
    ) == ("writer_only_rl", sha256_file(reward_path), 9)
