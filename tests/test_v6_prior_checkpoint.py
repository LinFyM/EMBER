from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_checkpoint import (
    PROGRAM_MEMORY_FILE,
    PROGRAM_MEMORY_KEY,
    V6_PRIOR_CHECKPOINT_SCHEMA,
    inspect_v6_prior_checkpoint,
    load_v6_prior_checkpoint,
    save_v6_prior_checkpoint,
)
from ember.pi05_source_checkpoint import DistributedContext
from ember.writer.condition_update import ProgramResidualMemory


_SHAPE = (4, 3, 2)


def _memory() -> ProgramResidualMemory:
    return ProgramResidualMemory(feature_width=4, program_slots=3, program_width=2)


def _cursor(macro: int = 1) -> dict[str, object]:
    return {
        "next_macro": macro,
        "task_visits_per_task": macro,
        "view_weights": [0.5, 0.5],
    }


def _contract() -> dict[str, object]:
    return {
        "run_schema": "paired-video",
        "mode": "formal",
        "world_size": 1,
    }


def _context() -> DistributedContext:
    return DistributedContext(0, 0, 1, torch.device("cpu"))


def _save(tmp_path: Path, memory: ProgramResidualMemory, macro: int = 1) -> Path:
    return save_v6_prior_checkpoint(
        output_dir=tmp_path,
        macro=macro,
        memory=memory,
        context=_context(),
        metrics_rows=macro,
        cursor_contract=_cursor(macro),
        checkpoint_contract=_contract(),
        expected_memory_shape=_SHAPE,
    )


def _refresh_size(checkpoint: Path, name: str) -> None:
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][name] = (checkpoint / name).stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_checkpoint_contains_only_program_memory_rng_and_manifest(tmp_path: Path) -> None:
    memory = _memory()
    memory.value.copy_(torch.arange(24, dtype=torch.float32).reshape(_SHAPE))
    checkpoint = _save(tmp_path, memory)
    assert {path.name for path in checkpoint.iterdir()} == {
        "manifest.json",
        PROGRAM_MEMORY_FILE,
        "rng_rank_000.pt",
    }
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert manifest["schema_version"] == V6_PRIOR_CHECKPOINT_SCHEMA
    assert "success_key_bank_shape" not in manifest
    inspection = inspect_v6_prior_checkpoint(
        checkpoint,
        expected_memory_shape=_SHAPE,
        expected_world_size=1,
    )
    assert inspection["program_memory"]["tensor_count"] == 1
    assert "success_key_bank" not in inspection
    assert inspection["rng"]["rank_count"] == 1


def test_load_restores_memory_cursor_and_current_rank_rng(tmp_path: Path) -> None:
    random.seed(5)
    np.random.seed(6)
    torch.manual_seed(7)
    memory = _memory()
    memory.value.normal_()
    expected = memory.value.clone()
    checkpoint = _save(tmp_path, memory)
    expected_draw = (random.random(), float(np.random.rand()), torch.rand(3))
    memory.value.zero_()
    random.seed(50)
    np.random.seed(60)
    torch.manual_seed(70)
    macro, rows = load_v6_prior_checkpoint(
        checkpoint=checkpoint,
        memory=memory,
        context=_context(),
        expected_cursor_contract=_cursor(),
        expected_checkpoint_contract=_contract(),
        expected_memory_shape=_SHAPE,
    )
    observed_draw = (random.random(), float(np.random.rand()), torch.rand(3))
    assert (macro, rows) == (1, 1)
    torch.testing.assert_close(memory.value, expected, rtol=0, atol=0)
    assert observed_draw[0] == expected_draw[0]
    assert observed_draw[1] == expected_draw[1]
    torch.testing.assert_close(observed_draw[2], expected_draw[2], rtol=0, atol=0)


def test_fresh_step_then_resume_preserves_program_and_rng_trajectory(tmp_path: Path) -> None:
    random.seed(10)
    np.random.seed(11)
    torch.manual_seed(12)
    first = _memory()
    first.value.add_(torch.rand_like(first.value) + random.random() + np.random.rand())
    checkpoint = _save(tmp_path, first)
    first.value.add_(torch.rand_like(first.value) + random.random() + np.random.rand())
    expected = first.value.clone()

    resumed = _memory()
    load_v6_prior_checkpoint(
        checkpoint=checkpoint,
        memory=resumed,
        context=_context(),
        expected_cursor_contract=_cursor(),
        expected_checkpoint_contract=_contract(),
        expected_memory_shape=_SHAPE,
    )
    resumed.value.add_(
        torch.rand_like(resumed.value) + random.random() + np.random.rand()
    )
    torch.testing.assert_close(resumed.value, expected, rtol=0, atol=0)


def test_checkpoint_rejects_wrong_schema_contract_extra_file_and_tensor(tmp_path: Path) -> None:
    checkpoint = _save(tmp_path, _memory())
    manifest_path = checkpoint / "manifest.json"
    original = json.loads(manifest_path.read_text())
    manifest = dict(original)
    manifest["schema_version"] = (
        "ember_pi05_v6_causal_goal_interaction_key_joint_credit_checkpoint_v1"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="manifest"):
        inspect_v6_prior_checkpoint(
            checkpoint, expected_memory_shape=_SHAPE, expected_world_size=1
        )
    manifest_path.write_text(json.dumps(original), encoding="utf-8")
    (checkpoint / "success_key_bank.safetensors").write_bytes(b"retired")
    with pytest.raises(ExpertManifoldError, match="manifest"):
        inspect_v6_prior_checkpoint(
            checkpoint, expected_memory_shape=_SHAPE, expected_world_size=1
        )
    (checkpoint / "success_key_bank.safetensors").unlink()
    save_file({"wrong": torch.zeros(_SHAPE)}, str(checkpoint / PROGRAM_MEMORY_FILE))
    _refresh_size(checkpoint, PROGRAM_MEMORY_FILE)
    with pytest.raises(ExpertManifoldError, match="Program memory key"):
        inspect_v6_prior_checkpoint(
            checkpoint, expected_memory_shape=_SHAPE, expected_world_size=1
        )


def test_checkpoint_rejects_cgik_rng_schema(tmp_path: Path) -> None:
    checkpoint = _save(tmp_path, _memory())
    rng_path = checkpoint / "rng_rank_000.pt"
    rng = torch.load(rng_path, map_location="cpu", weights_only=False)
    rng["schema_version"] = (
        "ember_pi05_v6_causal_goal_interaction_key_joint_credit_rank_rng_v1"
    )
    torch.save(rng, rng_path)
    _refresh_size(checkpoint, rng_path.name)
    with pytest.raises(ExpertManifoldError, match="rank 0 RNG"):
        inspect_v6_prior_checkpoint(
            checkpoint, expected_memory_shape=_SHAPE, expected_world_size=1
        )


def test_checkpoint_rejects_nonfinite_program_and_wrong_resume_contract(
    tmp_path: Path,
) -> None:
    checkpoint = _save(tmp_path, _memory())
    state = load_file(str(checkpoint / PROGRAM_MEMORY_FILE))
    state[PROGRAM_MEMORY_KEY][0, 0, 0] = torch.nan
    save_file(state, str(checkpoint / PROGRAM_MEMORY_FILE))
    _refresh_size(checkpoint, PROGRAM_MEMORY_FILE)
    with pytest.raises(ExpertManifoldError, match="non-finite"):
        inspect_v6_prior_checkpoint(
            checkpoint, expected_memory_shape=_SHAPE, expected_world_size=1
        )

    second = _save(tmp_path / "second", _memory())
    with pytest.raises(ExpertManifoldError, match="checkpoint contract"):
        load_v6_prior_checkpoint(
            checkpoint=second,
            memory=_memory(),
            context=_context(),
            expected_cursor_contract=_cursor(),
            expected_checkpoint_contract={"different": True},
            expected_memory_shape=_SHAPE,
        )
