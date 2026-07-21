from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from ember.pi05_source_checkpoint import DistributedContext, sha256_file
from ember.writer.as_contract import (
    load_writer_config,
    parse_checkpoint_steps,
    resolve_runtime,
    resume_step,
)
from ember.writer.model import WriterModelError


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_as_writer_config_is_pi05_one_video_and_pending_real_profile() -> None:
    path = REPO_ROOT / "configs/pi05_as_writer_v1.json"
    config = load_writer_config(path)
    assert config["writer"]["vision_feature_dim"] == 2048
    assert config["writer"]["language_feature_dim"] == 2048
    assert config["writer"]["generated_adapter"] == "complete_pi05_task_specific_lora"
    assert config["data"]["task_count"] == 24
    assert config["data"]["episodes_per_task"] == 50
    assert config["data"]["sampler_seed"] != config["data"]["teacher_video_seed"]
    assert config["formal_run"]["status"] == "pending_profile"
    assert config["formal_run"]["total_steps"] == 0
    assert config["information_wall"]["validation_actions_read"] == 0
    assert config["information_wall"]["test_actions_read"] == 0
    assert config["information_wall"]["test_video_values_read"] == 0
    assert sha256_file(path) == (
        REPO_ROOT / "configs/pi05_as_writer_v1.sha256"
    ).read_text(encoding="utf-8").split()[0]


def test_as_writer_checkpoint_schedule_and_cursor_are_fail_closed() -> None:
    assert parse_checkpoint_steps("2,4,4", 4) == (2, 4)
    assert resume_step(Path("/tmp/step_00000004")) == 4
    with pytest.raises(WriterModelError, match="must end at total_steps"):
        parse_checkpoint_steps("2,3", 4)
    with pytest.raises(WriterModelError, match="not a step checkpoint"):
        resume_step(Path("/tmp/trainer_state.pt"))


def test_pending_as_writer_config_cannot_launch_formal() -> None:
    config = load_writer_config(REPO_ROOT / "configs/pi05_as_writer_v1.json")
    args = argparse.Namespace(
        mode="formal",
        total_steps=None,
        batch_size=None,
        checkpoint_steps=None,
        stop_after_step=None,
        resume=None,
        skip_data_sha=False,
    )
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=8,
        device=torch.device("cpu"),
        numa_node=0,
        cpu_affinity=(0,),
    )
    with pytest.raises(WriterModelError, match="pending a real profile"):
        resolve_runtime(args, config, context)


def test_retired_smolvla_cold_start_config_is_not_an_active_writer_config() -> None:
    with pytest.raises(WriterModelError, match="unsupported PI05 AS-Writer"):
        load_writer_config(REPO_ROOT / "configs/writer_cold_start_v1.json")
