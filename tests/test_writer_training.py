from __future__ import annotations

from pathlib import Path

from ember.writer.training import load_writer_config


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
