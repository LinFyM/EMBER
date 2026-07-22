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
from ember.writer.conditioning import (
    batch_size_cycle,
    conditioning_cycle,
    matching_objective,
    pack_writer_conditions,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_as_writer_config_is_pi05_one_video_and_formal_sealed() -> None:
    path = REPO_ROOT / "configs/pi05_as_writer_v2.json"
    config = load_writer_config(path)
    assert config["writer"]["vision_feature_dim"] == 2048
    assert config["writer"]["vision_spatial_tokens"] == 16
    assert config["writer"]["language_feature_dim"] == 2048
    assert config["writer"]["generated_adapter"] == "complete_pi05_task_specific_lora"
    assert config["data"]["task_count"] == 24
    assert config["data"]["episodes_per_task"] == 50
    assert config["data"]["sampler_seed"] != config["data"]["teacher_video_seed"]
    assert config["formal_run"]["status"] == "sealed"
    assert config["formal_run"]["total_steps"] == 1500
    assert config["formal_run"]["per_rank_batch_size"] == 16
    assert config["formal_run"]["checkpoint_steps"] == [
        250,
        500,
        750,
        1000,
        1250,
        1500,
    ]
    assert len(config["conditioning_training"]["video_task_pairs"]) == 12
    assert conditioning_cycle(config) == (
        "normal",
        "full_language_contrast",
        "generic_language_contrast",
    )
    assert batch_size_cycle(16, config) == (16, 8, 8)
    assert config["conditioning_training"]["generic_writer_language"] == (
        "perform the demonstrated task"
    )
    assert config["conditioning_training"]["policy_language_contract"] == (
        "correct_action_query_task_language_on_every_branch"
    )
    assert config["information_wall"]["validation_actions_read"] == 0
    assert config["information_wall"]["test_actions_read"] == 0
    assert config["information_wall"]["test_video_values_read"] == 0
    assert sha256_file(path) == (
        REPO_ROOT / "configs/pi05_as_writer_v2.sha256"
    ).read_text(encoding="utf-8").split()[0]


def test_as_writer_checkpoint_schedule_and_cursor_are_fail_closed() -> None:
    assert parse_checkpoint_steps("2,4,4", 4) == (2, 4)
    assert resume_step(Path("/tmp/step_00000004")) == 4
    with pytest.raises(WriterModelError, match="must end at total_steps"):
        parse_checkpoint_steps("2,3", 4)
    with pytest.raises(WriterModelError, match="not a step checkpoint"):
        resume_step(Path("/tmp/trainer_state.pt"))


def test_video_matching_manual_gradient_coefficients_equal_autograd() -> None:
    losses = tuple(torch.tensor(value, requires_grad=True) for value in (0.11, 0.10))
    config = {
        "contrast_correct_loss_weight": 1.0,
        "matching_loss_weight": 0.5,
        "matching_margin": 0.01,
        "matching_temperature": 0.01,
    }
    objective, coefficients, probability = matching_objective(losses, config)
    observed = torch.autograd.grad(objective, losses)
    for actual, expected in zip(observed, coefficients, strict=True):
        torch.testing.assert_close(actual, expected)
    assert 0 < float(probability.detach()) < 1


def test_writer_condition_packing_uses_real_generic_tokens_and_two_arms_only() -> None:
    language = torch.full((2, 4), 1.0)
    generic = torch.full((3, 4), 7.0)
    correct = torch.full((5, 2, 4), 2.0)
    wrong = torch.full((6, 2, 4), 3.0)
    packed = pack_writer_conditions(
        language, generic, correct, wrong, "generic_language_contrast"
    )
    assert packed[2].tolist() == [0, 3, 6]
    assert packed[3].tolist() == [0, 5, 11]
    assert torch.all(packed[0] == 7)
    assert packed[1].shape == (11, 2, 4)

    normal = pack_writer_conditions(language, generic, correct, None, "normal")
    assert normal[2].tolist() == [0, 2]
    assert normal[3].tolist() == [0, 5]


def test_sealed_as_writer_config_resolves_profile_and_formal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_writer_config(REPO_ROOT / "configs/pi05_as_writer_v2.json")
    args = argparse.Namespace(
        mode="profile",
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
    assert resolve_runtime(args, config, context) == (
        6,
        2,
        (6,),
    )
    assert args.stop_after_step == 6
    args.mode = "formal"
    args.stop_after_step = None
    monkeypatch.setattr(
        "ember.writer.as_contract.git_state",
        lambda _root: {"dirty_paths": [], "commit": "sealed", "origin_main": "sealed"},
    )
    assert resolve_runtime(args, config, context) == (
        1500,
        16,
        (250, 500, 750, 1000, 1250, 1500),
    )
    assert args.stop_after_step == 1500


def test_retired_smolvla_cold_start_config_is_not_an_active_writer_config() -> None:
    with pytest.raises(WriterModelError, match="unsupported PI05 AS-Writer"):
        load_writer_config(REPO_ROOT / "configs/writer_cold_start_v1.json")


def test_collapsed_writer_v1_config_is_not_active() -> None:
    with pytest.raises(WriterModelError, match="unsupported PI05 AS-Writer"):
        load_writer_config(REPO_ROOT / "configs/pi05_as_writer_v1.json")
