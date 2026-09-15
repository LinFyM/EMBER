from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import (
    Pi05SourceTrainingError,
    canonical_hash,
    checkpoint_files,
    verify_checkpoint,
    write_json_atomic,
)
from ember.pi05_source_contract import reconcile_metrics, resolve_runtime, validate_formal
from ember.pi05_source_setup import load_config, make_scheduler


ROOT = Path(__file__).resolve().parents[1]


class _Tokenizer:
    def encode(self, prompts: list[str], *, add_bos: bool) -> list[list[int]]:
        assert add_bos
        assert all(prompt.startswith("Task: ") and prompt.endswith(";\nAction: ") for prompt in prompts)
        return [[1, row + 2] for row in range(len(prompts))]


def _processor() -> Pi05LiberoProcessor:
    value = object.__new__(Pi05LiberoProcessor)
    value._tokenizer = _Tokenizer()
    value._max_length = 5
    value._device = "cpu"
    value._state_q01 = torch.zeros(8)
    value._state_q99 = torch.full((8,), 2.0)
    value._action_q01 = torch.zeros(7)
    value._action_q99 = torch.full((7,), 2.0)
    return value


def test_training_processor_maps_cameras_tokens_and_source_actions() -> None:
    batch = {
        "observation.images.camera1": torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
        "observation.images.camera2": torch.full((2, 3, 8, 8), 255, dtype=torch.uint8),
        "observation.state": torch.ones(2, 8),
        "action": torch.stack((torch.zeros(50, 7), torch.full((50, 7), 2.0))),
        "task": ["close_the_drawer", "pick up the bowl"],
    }
    result = _processor().training_batch(batch)

    assert tuple(result["observation.images.base_0_rgb"].shape) == (2, 3, 8, 8)
    assert result["observation.images.base_0_rgb"].dtype == torch.float32
    assert torch.all(result["observation.images.left_wrist_0_rgb"] == 1)
    assert "observation.images.right_wrist_0_rgb" not in result
    assert result["observation.language.tokens"].tolist() == [[1, 2, 0, 0, 0], [1, 3, 0, 0, 0]]
    assert result["observation.language.attention_mask"].sum().item() == 4
    assert torch.all(result["action"][0] == -1)
    assert torch.allclose(result["action"][1], torch.ones_like(result["action"][1]))


def test_missing_right_wrist_uses_official_false_image_mask() -> None:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    class StubPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(
                image_features=(
                    "observation.images.base_0_rgb",
                    "observation.images.left_wrist_0_rgb",
                    "observation.images.right_wrist_0_rgb",
                ),
                image_resolution=(8, 8),
            )

    batch = {
        "observation.images.base_0_rgb": torch.zeros(2, 3, 8, 8),
        "observation.images.left_wrist_0_rgb": torch.ones(2, 3, 8, 8),
    }
    images, masks = PI05Policy._preprocess_images(StubPolicy(), batch)
    assert len(images) == len(masks) == 3
    assert torch.all(masks[0]) and torch.all(masks[1])
    assert not torch.any(masks[2])
    assert torch.all(images[2] == -1)


def test_official_warmup_starts_at_peak_over_warmup_plus_one() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=5e-5)
    scheduler = make_scheduler(optimizer, warmup_steps=10_000, peak_lr=5e-5)

    assert optimizer.param_groups[0]["lr"] == 5e-5 / 10_001
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 2 * 5e-5 / 10_001


def test_source_config_binds_every_sealed_authority() -> None:
    config = load_config(ROOT / "configs/pi05_source_base_v1.json")
    assert len(config["data"]["active_task_ids"]) == 71
    assert config["optimization"]["method"] == "full_action_sft"
    assert config["optimization"]["global_batch_size"] == 256
    assert config["formal_run"]["locked"] is True
    assert config["formal_run"]["micro_batch_size_per_rank"] == 32
    assert config["formal_run"]["gradient_accumulation_steps"] == 1
    assert config["models"]["forbidden"] == ["pi05_libero"]


def test_smoke_runtime_can_profile_without_unlocking_formal_config() -> None:
    config = load_config(ROOT / "configs/pi05_source_base_v1.json")
    args = argparse.Namespace(
        mode="smoke",
        optimizer_steps=3,
        micro_batch_size=2,
        gradient_accumulation=4,
        checkpoint_interval=0,
        ema="off",
        task_limit=4,
        skip_data_sha=True,
        stop_after_optimizer_step=None,
    )
    context = argparse.Namespace(world_size=1)
    assert resolve_runtime(args, config, context) == (3, 2, 4, 0, False)


def test_formal_resume_is_pinned_to_contract_not_moving_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs/pi05_source_base_v1.json")
    context = argparse.Namespace(world_size=8, numa_node=0, cpu_affinity=(0, 1))
    common = {
        "mode": "formal",
        "task_limit": None,
        "skip_data_sha": False,
        "stop_after_optimizer_step": None,
    }
    monkeypatch.setattr(
        "ember.pi05_source_contract.git_state",
        lambda: {
            "branch": "source-launch",
            "commit": "a" * 40,
            "origin_main": "b" * 40,
            "dirty_paths": [],
        },
    )
    kwargs = {
        "ema_enabled": True,
        "optimizer_steps": 1_000,
        "micro_batch_size": 32,
        "gradient_accumulation": 1,
        "checkpoint_interval": 1_000,
    }
    with pytest.raises(Pi05SourceTrainingError, match="must already be pushed"):
        validate_formal(argparse.Namespace(resume=None, **common), config, context, **kwargs)
    validate_formal(
        argparse.Namespace(resume=Path("checkpoint"), **common),
        config,
        context,
        **kwargs,
    )
    with pytest.raises(Pi05SourceTrainingError, match="NUMA affinity"):
        validate_formal(
            argparse.Namespace(resume=Path("checkpoint"), **common),
            config,
            argparse.Namespace(world_size=8, numa_node=None, cpu_affinity=None),
            **kwargs,
        )


def test_metrics_resume_preserves_orphaned_post_checkpoint_rows(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "".join(json.dumps({"optimizer_step": step}) + "\n" for step in (1, 2, 3)),
        encoding="utf-8",
    )

    assert reconcile_metrics(path, optimizer_step=2, expected_rows=2) == 2
    assert [json.loads(line)["optimizer_step"] for line in path.read_text().splitlines()] == [1, 2]
    packet = tmp_path / "failure_packets/orphaned_after_step_00000002.jsonl"
    assert json.loads(packet.read_text())["optimizer_step"] == 3


def test_checkpoint_manifest_detects_file_changes(tmp_path: Path) -> None:
    contract = "a" * 64
    (tmp_path / "policy").mkdir()
    (tmp_path / "policy/model.safetensors").write_bytes(b"weights")
    write_json_atomic(
        tmp_path / "trainer_state.json",
        {
            "schema_version": "ember_pi05_source_trainer_state_v1",
            "contract_sha256": contract,
            "optimizer_step": 1,
            "micro_step": 2,
            "metrics_rows": 1,
        },
    )
    files = checkpoint_files(tmp_path)
    write_json_atomic(
        tmp_path / "checkpoint_manifest.json",
        {
            "schema_version": "ember_pi05_source_checkpoint_v1",
            "contract_sha256": contract,
            "files": files,
            "aggregate_sha256": canonical_hash(files),
        },
    )
    assert verify_checkpoint(tmp_path, contract)["optimizer_step"] == 1

    (tmp_path / "policy/model.safetensors").write_bytes(b"changed")
    try:
        verify_checkpoint(tmp_path, contract)
    except Exception as error:
        assert "hashes changed" in str(error)
    else:
        raise AssertionError("checkpoint mutation was not detected")


def test_accumulated_source_updates_match_full_batch_and_preserve_ema(monkeypatch) -> None:
    from ember.pi05_source_training import _optimizer_step

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1, bias=False)

        def forward(self, batch):
            return (self.linear(batch["x"]) - batch["y"]).square().mean(), {}

    torch.manual_seed(7)
    policy = Policy()
    reference, ema = copy.deepcopy(policy), copy.deepcopy(policy)
    expected_ema = ema.linear.weight.detach().clone()
    optimizer = torch.optim.SGD(policy.parameters(), lr=.2)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    x, y = torch.randn(6, 2), torch.randn(6, 1)
    runtime = SimpleNamespace(
        policy=policy, wrapped=policy, ema_policy=ema, optimizer=optimizer, scheduler=scheduler,
        context=SimpleNamespace(world_size=1, device=torch.device("cpu")), micro_batch=2, accumulation=3,
        processor=SimpleNamespace(training_batch=lambda batch: batch),
        config={"optimization": {"optimizer": {"gradient_clip_norm": 1.}, "ema_decay": .9}},
    )
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda _: 0)
    for step in range(2):
        runtime.iterator = iter({"x": x[i:i+2], "y": y[i:i+2]} for i in range(0, 6, 2))
        row = _optimizer_step(runtime, step, time.monotonic())
        reference_optimizer.zero_grad()
        reference({"x": x, "y": y})[0].backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
        reference_optimizer.step()
        expected_ema.mul_(.9).add_(reference.linear.weight.detach(), alpha=.1)
        torch.testing.assert_close(policy.linear.weight, reference.linear.weight)
        torch.testing.assert_close(ema.linear.weight, expected_ema)
        assert row["global_examples"] == (step + 1) * 6
        assert row["micro_step"] == (step + 1) * 3


def test_aligned_source_provenance_rejects_old_action_alignment() -> None:
    from ember.pi05_eval_contract import Pi05EvaluationError, _validate_source_checkpoint_provenance

    config = load_config(ROOT / "configs/pi05_source_aligned.json")
    authorities = SimpleNamespace(source_base_config=config)
    run = {"schema_version": "ember_pi05_source_launch_v1", "models": config["models"],
           "features": config["features"], "optimization": config["optimization"],
           "task_ids": config["data"]["active_task_ids"], "data": config["data"]}
    trainer = {"schema_version": "ember_pi05_source_trainer_state_v1", "optimizer_step": 1000,
               "micro_step": 16000, "ema_enabled": True}
    manifest = {"schema_version": "ember_pi05_source_checkpoint_v1", "optimizer_step": 1000,
                "micro_step": 16000}
    _validate_source_checkpoint_provenance(authorities, run, manifest, trainer)
    for data in (None, {**config["data"], "action_start_offset": 0}):
        with pytest.raises(Pi05EvaluationError, match="provenance"):
            _validate_source_checkpoint_provenance(authorities, {**run, "data": data}, manifest, trainer)
