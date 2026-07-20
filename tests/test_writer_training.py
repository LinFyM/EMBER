from __future__ import annotations

from pathlib import Path

from ember.writer.training import load_writer_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_writer_profile_config_keeps_formal_launch_closed() -> None:
    config = load_writer_config(REPO_ROOT / "configs/writer_cold_start_v1.json")
    assert config["formal_run"]["status"] == "pending_profile"
    assert config["writer"]["vision_feature_dim"] == 960
    assert config["data"]["demo_indices"] == [0, 49]
    assert config["optimization"]["precision"] == "bfloat16"
