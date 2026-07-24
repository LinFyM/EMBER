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
from ember.writer.as_step import (
    _cumulative_counts,
    _functional_arm_gradient,
    _policy_microbatches,
    run_writer_step,
)
from ember.writer.model import WriterModelError
from ember.writer.checkpoint import initialize_writer_phase


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
    assert batch_size_cycle(16, config) == (16,)
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
    assert (
        config["conditioning_training"][
            "independent_conditions_per_optimizer_step"
        ]
        == 2
    )
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
    assert resolve_runtime(profile, config, context) == (2, 16, (2,))
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
        16,
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


def test_logical_optimizer_update_counts_two_independent_conditions_per_rank() -> None:
    runtime = SimpleNamespace(
        conditions_per_optimizer_step=2,
        batch_size=64,
        sampler=SimpleNamespace(batch_size_for_step=lambda _step: 64),
        context=SimpleNamespace(world_size=4),
        config={"conditioning_training": {"step_cycle": ["normal"]}},
    )
    assert _cumulative_counts(runtime, completed=3) == (
        3 * 2 * 4 * 64,
        3 * 2 * 4 * 64,
        3 * 2 * 4,
    )


def test_writer_update_averages_two_independent_task_conditions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    writer = torch.nn.ParameterList([parameter])
    optimizer = torch.optim.SGD(writer.parameters(), lr=1.0)
    scheduler_steps: list[bool] = []
    calls: list[tuple[int, float]] = []
    observed: dict[str, object] = {}

    def fake_pack(
        _runtime: object,
        *,
        task_id: int,
        correct_demo_index: int,
        mode: str,
        wrong_task_id: int | None,
        wrong_demo_index: int | None,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, int | None]]:
        del correct_demo_index, mode, wrong_task_id, wrong_demo_index
        return (torch.tensor(task_id),) * 5, {
            "correct_video_raw_frames": 1,
            "correct_video_sampled_frames": 1,
            "wrong_video_raw_frames": None,
            "wrong_video_sampled_frames": None,
        }

    def fake_differentiate(
        _runtime: object,
        _packed: object,
        policy_batch: dict[str, torch.Tensor],
        _mode: str,
        gradient_scale: float,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[dict[str, float]],
        None,
    ]:
        task_id = int(policy_batch["task_id"].item())
        calls.append((task_id, gradient_scale))
        contribution = torch.tensor(float(task_id) * gradient_scale)
        parameter.grad = (
            contribution
            if parameter.grad is None
            else parameter.grad + contribution
        )
        value = torch.tensor(float(task_id))
        return value, [value], [{"loss": float(value), "policy_forward_calls": 1}], None

    def fake_metrics(
        _runtime: object,
        *,
        step: int,
        conditions: list[dict[str, object]],
        **_kwargs: object,
    ) -> dict[str, int]:
        observed["conditions"] = conditions
        return {"optimizer_step": step + 1}

    monkeypatch.setattr("ember.writer.as_step._pack_raw_conditions", fake_pack)
    monkeypatch.setattr(
        "ember.writer.as_step._differentiate_condition_batch",
        fake_differentiate,
    )
    monkeypatch.setattr("ember.writer.as_step._step_metrics", fake_metrics)

    runtime = SimpleNamespace(
        iterator=iter(
            [
                {"task_id": torch.tensor([2])},
                {"task_id": torch.tensor([6])},
            ]
        ),
        sampler=SimpleNamespace(
            task_visit_for_step=lambda data_step: ((2, 0), (6, 0))[data_step],
            batch_size_for_step=lambda _data_step: 1,
        ),
        video_schedule=SimpleNamespace(
            demo_for_task_visit=lambda _task_id, _visit: 0
        ),
        video_partner={2: 6, 6: 2},
        processor=SimpleNamespace(training_batch=lambda batch: batch),
        optimizer=optimizer,
        scheduler=SimpleNamespace(step=lambda: scheduler_steps.append(True)),
        writer=writer,
        policy=torch.nn.Identity(),
        config={
            "conditioning_training": {"step_cycle": ["normal"]},
            "optimization": {"optimizer": {"gradient_clip_norm": 100.0}},
        },
        conditions_per_optimizer_step=2,
    )
    row = run_writer_step(runtime, step=0, started=0.0)
    assert row == {"optimizer_step": 1}
    assert calls == [(2, 0.5), (6, 0.5)]
    assert [item["data_step"] for item in observed["conditions"]] == [0, 1]  # type: ignore[index]
    assert float(parameter.detach()) == pytest.approx(-4.0)
    assert scheduler_steps == [True]


def test_writer_weight_warm_start_resets_optimization_and_records_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "prior" / "checkpoints" / "step_00000005"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint_manifest.json").write_text("{}")
    writer = torch.nn.Linear(2, 2)
    warm_state = {
        name: torch.full_like(value, 3.0)
        for name, value in writer.state_dict().items()
    }
    config = {
        "sealed_stage": "development",
        "authorities": {"target_data_manifest": {"sha256": "target"}},
        "writer": {"architecture": "same"},
    }
    source = {"checkpoint": "same"}
    training = {
        "schema_version": "ember_pi05_action_memory_as_writer_launch_v1",
        "stage": "development",
        "source": source,
        "authorities": config["authorities"],
        "writer": config["writer"],
        "trainable": {"lora_contract_sha256": "lora"},
    }
    manifest = {
        "consumed": {"next_step": 5},
        "files": {"writer.safetensors": {"sha256": "a" * 64}},
    }
    monkeypatch.setattr(
        "ember.writer.checkpoint.inspect_writer_checkpoint",
        lambda _checkpoint: (training, manifest, "contract-sha"),
    )
    monkeypatch.setattr(
        "ember.writer.checkpoint.load_file",
        lambda _path, device: warm_state,
    )
    record = initialize_writer_phase(
        checkpoint,
        SimpleNamespace(is_main=True, world_size=1, device=torch.device("cpu")),
        "development",
        source,
        config["authorities"],
        config["writer"],
        writer,  # type: ignore[arg-type]
        "lora",
    )
    assert record["mode"] == "writer_weight_warm_start"
    assert record["source_optimizer_step"] == 5
    assert record["optimizer"] == "fresh"
    assert record["scheduler"] == "fresh"
    assert record["rng"] == "fresh_seed"
    assert all(torch.equal(value, warm_state[name]) for name, value in writer.state_dict().items())


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
