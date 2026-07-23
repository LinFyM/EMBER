from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from ember.pi05_source_checkpoint import (
    DistributedContext,
    sha256_file,
    write_json_atomic,
)
from ember.writer.as_contract import (
    _contract_stop_step,
    load_writer_config,
    parse_checkpoint_steps,
    reconcile_resume_contract,
    resolve_runtime,
    resume_step,
    writer_split_roles,
)
from ember.writer.model import WriterModelError
from ember.writer.conditioning import (
    batch_size_cycle,
    conditioning_cycle,
    matching_objective,
    pack_writer_conditions,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_CONFIG = REPO_ROOT / "configs/pi05_as_writer_final_v1.json"


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


def test_final_as_writer_reuses_one_runner_for_exact_32_source_role() -> None:
    config = load_writer_config(FINAL_CONFIG)
    assert config["sealed_stage"] == "final"
    assert writer_split_roles(config) == ("train", "validation")
    assert config["data"]["task_count"] == 32
    pairs = config["conditioning_training"]["video_task_pairs"]
    assert len(pairs) == 16
    assert len({task for pair in pairs for task in pair}) == 32
    assert config["formal_run"]["total_steps"] == 1500
    assert config["formal_run"]["selected_stop_step"] == 500
    assert config["formal_run"]["checkpoint_steps"] == [
        250,
        500,
        750,
        1000,
        1250,
        1500,
    ]
    assert config["optimization"]["scheduler"]["decay_steps"] == 1500
    assert sha256_file(FINAL_CONFIG) == (
        REPO_ROOT / "configs/pi05_as_writer_final_v1.sha256"
    ).read_text(encoding="utf-8").split()[0]


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


def test_zero_matching_weight_keeps_only_correct_arm_gradient() -> None:
    losses = tuple(torch.tensor(value, requires_grad=True) for value in (0.11, 0.10))
    config = {
        "contrast_correct_loss_weight": 1.0,
        "matching_loss_weight": 0.0,
        "matching_margin": 0.01,
        "matching_temperature": 0.01,
    }
    objective, coefficients, probability = matching_objective(losses, config)
    observed = torch.autograd.grad(objective, losses)
    torch.testing.assert_close(objective, losses[0])
    torch.testing.assert_close(observed[0], torch.ones_like(losses[0]))
    torch.testing.assert_close(observed[1], torch.zeros_like(losses[1]))
    torch.testing.assert_close(coefficients[0], torch.ones_like(losses[0]))
    torch.testing.assert_close(coefficients[1], torch.zeros_like(losses[1]))
    assert 0 < float(probability.detach()) < 1


def test_no_matching_ablation_preserves_conditioning_cycle() -> None:
    path = REPO_ROOT / "configs/pi05_as_writer_v2_no_matching.json"
    config = load_writer_config(path)
    assert config["conditioning_training"]["matching_loss_weight"] == 0.0
    assert conditioning_cycle(config) == (
        "normal",
        "full_language_contrast",
        "generic_language_contrast",
    )
    assert batch_size_cycle(16, config) == (16, 8, 8)
    assert config["formal_run"]["checkpoint_steps"] == [
        250,
        500,
        750,
        1000,
        1250,
        1500,
    ]
    assert config["formal_run"]["selected_stop_step"] == 500
    assert config["formal_run"]["stage_stop_steps"] == [
        250,
        500,
        750,
        1000,
        1250,
        1500,
    ]
    assert sha256_file(path) == (
        REPO_ROOT / "configs/pi05_as_writer_v2_no_matching.sha256"
    ).read_text(encoding="utf-8").split()[0]


def test_normal_only_ablation_preserves_architecture_and_full_positive_batch() -> None:
    baseline = load_writer_config(
        REPO_ROOT / "configs/pi05_as_writer_v2_no_matching.json"
    )
    path = REPO_ROOT / "configs/pi05_as_writer_v2_normal_only.json"
    config = load_writer_config(path)
    assert config["writer"] == baseline["writer"]
    assert config["data"] == baseline["data"]
    assert config["optimization"]["optimizer"] == baseline["optimization"]["optimizer"]
    assert config["optimization"]["scheduler"] == baseline["optimization"]["scheduler"]
    assert config["conditioning_training"]["matching_loss_weight"] == 0.0
    assert conditioning_cycle(config) == ("normal",)
    assert batch_size_cycle(16, config) == (16,)
    assert config["formal_run"]["selected_stop_step"] == 250
    assert sha256_file(path) == (
        REPO_ROOT / "configs/pi05_as_writer_v2_normal_only.sha256"
    ).read_text(encoding="utf-8").split()[0]


def test_no_matching_ablation_allows_declared_exact_resume_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_writer_config(
        REPO_ROOT / "configs/pi05_as_writer_v2_no_matching.json"
    )
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=8,
        device=torch.device("cpu"),
        numa_node=0,
        cpu_affinity=(0,),
    )
    monkeypatch.setattr(
        "ember.writer.as_contract.git_state",
        lambda _root: {"dirty_paths": [], "commit": "sealed", "origin_main": "sealed"},
    )
    args = argparse.Namespace(
        mode="formal",
        total_steps=None,
        batch_size=None,
        checkpoint_steps=None,
        stop_after_step=750,
        resume=Path("/tmp/step_00000500"),
        skip_data_sha=False,
    )
    assert resolve_runtime(args, config, context) == (
        1500,
        16,
        (250, 500, 750, 1000, 1250, 1500),
    )
    assert args.stop_after_step == 750
    args.stop_after_step = 600
    with pytest.raises(WriterModelError, match="sealed profile"):
        resolve_runtime(args, config, context)


def test_formal_stage_extension_does_not_change_writer_contract_stop() -> None:
    config = load_writer_config(
        REPO_ROOT / "configs/pi05_as_writer_v2_no_matching.json"
    )
    args = argparse.Namespace(mode="formal", stop_after_step=750)
    assert _contract_stop_step(args, config, 1500) == 500
    args.mode = "profile"
    assert _contract_stop_step(args, config, 1500) == 750


def test_code_compatible_resume_allows_only_recorded_commit_change(
    tmp_path: Path,
) -> None:
    existing = {
        "schema_version": "contract",
        "git": {"branch": "main", "commit": "old"},
        "runtime": {"selected_stop_step": 500, "total_steps": 1500},
    }
    write_json_atomic(tmp_path / "run_contract.json", existing)
    args = argparse.Namespace(
        output_dir=tmp_path,
        resume=tmp_path / "checkpoints/step_00000500",
        allow_contract_compatible_code_resume=True,
    )
    candidate = {
        **existing,
        "git": {"branch": "main", "commit": "new"},
    }
    assert reconcile_resume_contract(args, candidate) == existing

    changed = {
        **candidate,
        "runtime": {"selected_stop_step": 500, "total_steps": 2000},
    }
    with pytest.raises(WriterModelError, match="scientific contract"):
        reconcile_resume_contract(args, changed)

    args.allow_contract_compatible_code_resume = False
    with pytest.raises(WriterModelError, match="launch contract changed"):
        reconcile_resume_contract(args, candidate)


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

    final = load_writer_config(FINAL_CONFIG)
    args.stop_after_step = None
    assert resolve_runtime(args, final, context) == (
        1500,
        16,
        (250, 500, 750, 1000, 1250, 1500),
    )
    assert args.stop_after_step == 500


def test_retired_smolvla_cold_start_config_is_not_an_active_writer_config() -> None:
    with pytest.raises(WriterModelError, match="unsupported PI05 AS-Writer"):
        load_writer_config(REPO_ROOT / "configs/writer_cold_start_v1.json")


def test_collapsed_writer_v1_config_is_not_active() -> None:
    with pytest.raises(WriterModelError, match="unsupported PI05 AS-Writer"):
        load_writer_config(REPO_ROOT / "configs/pi05_as_writer_v1.json")
