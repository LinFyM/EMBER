from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.pi05_source_checkpoint import DistributedContext, write_json_atomic
from ember.writer.as_contract import (
    _contract_stop_step,
    load_writer_config,
    parse_checkpoint_steps,
    reconcile_resume_contract,
    resolve_runtime,
    resume_step,
    writer_split_roles,
)
from ember.writer.conditioning import (
    batch_size_cycle,
    conditioning_cycle,
    matching_objective,
)
from ember.writer.as_step import _functional_arm_gradient, _policy_microbatches
from ember.writer.model import WriterModelError


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/pi05_as_writer_action_memory_v1.json"


def test_action_memory_writer_config_seals_architecture_and_information_wall() -> None:
    config = load_writer_config(CONFIG)
    writer = config["writer"]
    assert writer["architecture"] == "pi05_action_memory_writer_v1"
    assert writer["prefix_owner"] == "frozen_pi05_paligemma_full_frame_text_prefix"
    assert writer["expert_layers"] == 18
    assert writer["memory_slots"] == 16
    assert writer["meta_lora_rank"] == 8
    assert writer["hidden_dim"] == 320
    assert writer["frame_stride"] == 4
    assert writer["frame_microbatch"] == 16
    assert writer["conditional_linear_bias"] is True
    assert config["data"]["task_count"] == 24
    assert config["data"]["episodes_per_task"] == 50
    assert writer_split_roles(config) == ("train",)
    assert conditioning_cycle(config) == ("normal",)
    assert batch_size_cycle(128, config) == (128,)
    assert (
        config["information_wall"][
            "validation_actions_read_by_training_optimizer"
        ]
        == 0
    )
    assert (
        config["information_wall"][
            "validation_action_queries_per_checkpoint_monitor"
        ]
        == 512
    )
    assert config["conditioning_training"]["functional_policy_microbatch_size"] == 16
    assert config["information_wall"]["test_actions_read"] == 0
    assert config["information_wall"]["test_video_values_read"] == 0
    assert config["optimization"]["maximum_formal_wall_clock_minutes"] is None
    assert config["profile_defaults"]["expected_world_size"] == 4
    assert config["formal_run"]["expected_world_size"] == 4
    assert config["optimization"]["scheduler"]["decay_steps"] == 800
    assert config["formal_run"]["selected_stop_step"] == 800
    assert config["formal_run"]["stage_stop_steps"] == [
        800,
        1100,
        1400,
        1700,
        2000,
        2300,
        2400,
    ]


def test_action_memory_checkpoint_schedule_and_cursor_are_fail_closed() -> None:
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


def test_zero_matching_weight_keeps_only_correct_arm_gradient() -> None:
    losses = tuple(torch.tensor(value, requires_grad=True) for value in (0.11, 0.10))
    config = {
        "contrast_correct_loss_weight": 1.0,
        "matching_loss_weight": 0.0,
        "matching_margin": 0.01,
        "matching_temperature": 0.01,
    }
    objective, coefficients, _ = matching_objective(losses, config)
    observed = torch.autograd.grad(objective, losses)
    torch.testing.assert_close(objective, losses[0])
    torch.testing.assert_close(observed[0], torch.ones_like(losses[0]))
    torch.testing.assert_close(observed[1], torch.zeros_like(losses[1]))
    torch.testing.assert_close(coefficients[0], torch.ones_like(losses[0]))
    torch.testing.assert_close(coefficients[1], torch.zeros_like(losses[1]))


def test_formal_runtime_uses_four_rank_query_scaled_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_writer_config(CONFIG)
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=4,
        device=torch.device("cpu"),
        numa_node=0,
        cpu_affinity=(0,),
    )
    profile = argparse.Namespace(
        mode="profile",
        total_steps=None,
        batch_size=None,
        checkpoint_steps=None,
        stop_after_step=None,
        resume=None,
        skip_data_sha=False,
    )
    assert resolve_runtime(profile, config, context) == (2, 128, (2,))
    assert profile.stop_after_step == 2

    formal = argparse.Namespace(
        mode="formal",
        total_steps=None,
        batch_size=None,
        checkpoint_steps=None,
        stop_after_step=None,
        resume=None,
        skip_data_sha=False,
    )
    if config["formal_run"]["status"] != "sealed":
        with pytest.raises(WriterModelError, match="pending a real profile"):
            resolve_runtime(formal, config, context)
        return
    monkeypatch.setattr(
        "ember.writer.as_contract.git_state",
        lambda _root: {"dirty_paths": [], "commit": "sealed", "origin_main": "sealed"},
    )
    expected_checkpoints = tuple(config["formal_run"]["checkpoint_steps"])
    assert resolve_runtime(formal, config, context) == (
        2400,
        128,
        expected_checkpoints,
    )
    assert formal.stop_after_step == 800
    formal.resume = Path("/tmp/step_00000800")
    formal.stop_after_step = 1100
    assert resolve_runtime(formal, config, context)[0] == 2400
    assert formal.stop_after_step == 1100


def test_formal_extension_keeps_original_contract_stop() -> None:
    config = load_writer_config(CONFIG)
    args = argparse.Namespace(mode="formal", stop_after_step=1100)
    assert _contract_stop_step(args, config, 2400) == 800
    args.mode = "profile"
    assert _contract_stop_step(args, config, 2400) == 1100


def test_policy_microbatches_preserve_all_rows_and_tensor_alignment() -> None:
    batch = {
        "images": torch.arange(30).reshape(10, 3),
        "tokens": torch.arange(20).reshape(10, 2),
    }
    chunks = _policy_microbatches(batch, 4)
    assert [chunk["images"].shape[0] for chunk in chunks] == [4, 4, 2]
    assert torch.equal(torch.cat([chunk["images"] for chunk in chunks]), batch["images"])
    assert torch.equal(torch.cat([chunk["tokens"] for chunk in chunks]), batch["tokens"])

    with pytest.raises(WriterModelError, match="dimensions disagree"):
        _policy_microbatches(
            {"images": torch.zeros(2, 3), "tokens": torch.zeros(3, 2)},
            2,
        )


def test_functional_arm_gradient_is_query_weighted_across_microbatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_gradient(
        _policy: object,
        _state: object,
        _contract: object,
        *,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        value = batch["value"].mean()
        return value, {"loss": float(value)}, {"adapter": value.reshape(1)}

    monkeypatch.setattr(
        "ember.writer.as_step.functional_lora_loss_gradient",
        fake_gradient,
    )
    runtime = SimpleNamespace(
        policy=object(),
        lora_contract=object(),
        config={
            "conditioning_training": {
                "functional_policy_microbatch_size": 4,
            }
        },
    )
    values = torch.arange(10, dtype=torch.float32)
    loss, detail, gradient = _functional_arm_gradient(
        runtime,  # type: ignore[arg-type]
        {"adapter": torch.zeros(1)},
        {"value": values, "other": values[:, None]},
    )
    torch.testing.assert_close(loss, values.mean())
    torch.testing.assert_close(gradient["adapter"], values.mean().reshape(1))
    assert detail["loss"] == pytest.approx(float(values.mean()))
    assert detail["policy_forward_calls"] == 3


def test_code_compatible_resume_allows_only_recorded_commit_change(
    tmp_path: Path,
) -> None:
    existing = {
        "schema_version": "contract",
        "git": {"branch": "main", "commit": "old"},
        "runtime": {"selected_stop_step": 500, "total_steps": 1200},
    }
    write_json_atomic(tmp_path / "run_contract.json", existing)
    args = argparse.Namespace(
        output_dir=tmp_path,
        resume=tmp_path / "checkpoints/step_00000500",
        allow_contract_compatible_code_resume=True,
    )
    candidate = {**existing, "git": {"branch": "main", "commit": "new"}}
    assert reconcile_resume_contract(args, candidate) == existing
    changed = {
        **candidate,
        "runtime": {"selected_stop_step": 500, "total_steps": 1300},
    }
    with pytest.raises(WriterModelError, match="scientific contract"):
        reconcile_resume_contract(args, changed)


def test_retired_writer_configs_are_not_active() -> None:
    for name in (
        "writer_cold_start_v1.json",
        "pi05_as_writer_v1.json",
        "pi05_as_writer_v2.json",
        "pi05_as_writer_v3_normal_only.json",
    ):
        with pytest.raises(WriterModelError, match="unsupported PI05 AS-Writer"):
            load_writer_config(REPO_ROOT / "configs" / name)
