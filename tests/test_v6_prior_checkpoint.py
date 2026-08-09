from __future__ import annotations

import inspect
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
from ember.pi05_source_checkpoint import (
    DistributedContext,
    read_json,
    write_json_atomic,
)
from ember.writer.condition_update import ProgramResidualMemory


_SHAPE = (2, 3, 4)


def _memory() -> ProgramResidualMemory:
    return ProgramResidualMemory(
        feature_width=_SHAPE[0],
        program_slots=_SHAPE[1],
        program_width=_SHAPE[2],
    )


def _cursor(macro: int = 1) -> dict[str, int]:
    return {
        "next_macro": macro,
        "task_visits_per_task": macro,
        "sampler_seed": 11,
        "teacher_video_seed": 13,
        "counterfactual_seed": 17,
        "counterfactual_phase": macro % 3,
    }


def _contract() -> dict[str, object]:
    return {
        "schema_version": "synthetic_condition_residual_run_v1",
        "mode": "profile",
        "world_size": 1,
        "manual_update": {
            "optimizer": False,
            "scheduler": False,
            "scaler": False,
        },
    }


def _context() -> DistributedContext:
    return DistributedContext(0, 0, 1, torch.device("cpu"))


def _save(root: Path, memory: ProgramResidualMemory) -> Path:
    return save_v6_prior_checkpoint(
        output_dir=root,
        macro=1,
        memory=memory,
        context=_context(),
        metrics_rows=1,
        cursor_contract=_cursor(),
        checkpoint_contract=_contract(),
        expected_memory_shape=_SHAPE,
    )


def _refresh_size(checkpoint: Path, name: str) -> None:
    manifest = read_json(checkpoint / "manifest.json")
    manifest["files"][name] = (checkpoint / name).stat().st_size
    write_json_atomic(checkpoint / "manifest.json", manifest)


def _assert_rng_equal(
    left: tuple[object, object, torch.Tensor],
    right: tuple[object, object, torch.Tensor],
) -> None:
    assert left[0] == right[0]
    assert left[1] == right[1]
    assert torch.equal(left[2], right[2])


def _draw_rng() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.random()), torch.rand(5)


def test_checkpoint_is_atomic_single_tensor_memory_only_and_inspectable(
    tmp_path: Path,
) -> None:
    memory = _memory()
    with torch.no_grad():
        memory.value.copy_(torch.arange(24, dtype=torch.float32).reshape(_SHAPE))
    checkpoint = _save(tmp_path, memory)

    assert {path.name for path in checkpoint.iterdir()} == {
        "manifest.json",
        PROGRAM_MEMORY_FILE,
        "rng_rank_000.pt",
    }
    assert not list(checkpoint.parent.glob(".*.tmp"))
    state = load_file(str(checkpoint / PROGRAM_MEMORY_FILE), device="cpu")
    assert set(state) == {PROGRAM_MEMORY_KEY}
    assert state[PROGRAM_MEMORY_KEY].dtype == torch.float32
    assert tuple(state[PROGRAM_MEMORY_KEY].shape) == _SHAPE
    assert torch.equal(state[PROGRAM_MEMORY_KEY], memory.value)

    manifest = read_json(checkpoint / "manifest.json")
    assert manifest["schema_version"] == V6_PRIOR_CHECKPOINT_SCHEMA
    assert manifest["content_hash_policy"] == "disabled_by_owner"
    encoded_manifest = json.dumps(manifest, sort_keys=True).lower()
    assert "writer.safetensors" not in encoded_manifest
    assert "trainer.pt" not in encoded_manifest
    assert "projection" not in encoded_manifest
    public_arguments = set(inspect.signature(save_v6_prior_checkpoint).parameters)
    assert public_arguments.isdisjoint(
        {"writer", "base_writer", "projection", "optimizer", "scheduler", "scaler"}
    )

    inspected = inspect_v6_prior_checkpoint(
        checkpoint,
        expected_memory_shape=_SHAPE,
        expected_world_size=1,
        expected_cursor_contract=_cursor(),
        expected_checkpoint_contract=_contract(),
    )
    assert inspected["next_macro"] == 1
    assert inspected["metrics_rows"] == 1
    assert inspected["program_memory"] == {
        "file": PROGRAM_MEMORY_FILE,
        "key": PROGRAM_MEMORY_KEY,
        "tensor_count": 1,
        "dtype": "torch.float32",
        "shape": list(_SHAPE),
        "value_count": 24,
        "finite": True,
    }
    assert "writer" not in inspected
    assert "trainer" not in inspected


def test_load_restores_only_memory_cursor_and_current_rank_rng(tmp_path: Path) -> None:
    random.seed(101)
    np.random.seed(103)
    torch.manual_seed(107)
    source = _memory()
    with torch.no_grad():
        source.value.normal_()
    saved_value = source.value.clone()
    checkpoint = _save(tmp_path, source)
    expected_rng = _draw_rng()

    random.seed(1001)
    np.random.seed(1003)
    torch.manual_seed(1007)
    destination = _memory()
    with torch.no_grad():
        destination.value.fill_(91.0)
    macro, rows = load_v6_prior_checkpoint(
        checkpoint=checkpoint,
        memory=destination,
        context=_context(),
        expected_cursor_contract=_cursor(),
        expected_checkpoint_contract=_contract(),
        expected_memory_shape=_SHAPE,
    )
    assert (macro, rows) == (1, 1)
    assert torch.equal(destination.value, saved_value)
    _assert_rng_equal(_draw_rng(), expected_rng)

    untouched = destination.value.clone()
    with pytest.raises(ExpertManifoldError, match="cursor contract"):
        load_v6_prior_checkpoint(
            checkpoint=checkpoint,
            memory=destination,
            context=_context(),
            expected_cursor_contract={**_cursor(), "teacher_video_seed": 19},
            expected_checkpoint_contract=_contract(),
            expected_memory_shape=_SHAPE,
        )
    assert torch.equal(destination.value, untouched)


def test_fresh_one_step_then_resume_has_identical_memory_and_rng_trajectory(
    tmp_path: Path,
) -> None:
    def seed() -> None:
        random.seed(211)
        np.random.seed(223)
        torch.manual_seed(227)

    def update(memory: ProgramResidualMemory) -> None:
        scale = random.random() + float(np.random.random())
        with torch.no_grad():
            memory.value.add_(scale * torch.randn_like(memory.value))

    seed()
    uninterrupted = _memory()
    update(uninterrupted)
    update(uninterrupted)
    uninterrupted_rng = _draw_rng()

    seed()
    interrupted = _memory()
    update(interrupted)
    checkpoint = _save(tmp_path, interrupted)
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    resumed = _memory()
    load_v6_prior_checkpoint(
        checkpoint=checkpoint,
        memory=resumed,
        context=_context(),
        expected_cursor_contract=_cursor(),
        expected_checkpoint_contract=_contract(),
        expected_memory_shape=_SHAPE,
    )
    update(resumed)
    resumed_rng = _draw_rng()
    assert torch.equal(resumed.value, uninterrupted.value)
    _assert_rng_equal(resumed_rng, uninterrupted_rng)


@pytest.mark.parametrize(
    "replacement, match",
    [
        ({"wrong.key": torch.zeros(_SHAPE)}, "Program memory key"),
        (
            {PROGRAM_MEMORY_KEY: torch.zeros(_SHAPE, dtype=torch.float64)},
            "tensor schema",
        ),
        ({PROGRAM_MEMORY_KEY: torch.zeros(2, 3, 5)}, "tensor schema"),
        (
            {PROGRAM_MEMORY_KEY: torch.full(_SHAPE, torch.nan)},
            "non-finite Program memory",
        ),
    ],
)
def test_inspection_rejects_wrong_key_dtype_shape_and_nonfinite(
    tmp_path: Path,
    replacement: dict[str, torch.Tensor],
    match: str,
) -> None:
    checkpoint = _save(tmp_path, _memory())
    save_file(replacement, str(checkpoint / PROGRAM_MEMORY_FILE))
    _refresh_size(checkpoint, PROGRAM_MEMORY_FILE)
    with pytest.raises(ExpertManifoldError, match=match):
        inspect_v6_prior_checkpoint(
            checkpoint,
            expected_memory_shape=_SHAPE,
            expected_world_size=1,
        )


def test_checkpoint_rejects_extra_files_old_schema_and_invalid_live_memory(
    tmp_path: Path,
) -> None:
    checkpoint = _save(tmp_path / "extra", _memory())
    (checkpoint / "trainer.pt").touch()
    with pytest.raises(ExpertManifoldError, match="manifest"):
        inspect_v6_prior_checkpoint(
            checkpoint, expected_memory_shape=_SHAPE, expected_world_size=1
        )

    legacy = _save(tmp_path / "legacy", _memory())
    manifest = read_json(legacy / "manifest.json")
    manifest["schema_version"] = "ember_pi05_v6_tangent_tube_writer_checkpoint_v3"
    write_json_atomic(legacy / "manifest.json", manifest)
    with pytest.raises(ExpertManifoldError, match="manifest"):
        inspect_v6_prior_checkpoint(
            legacy, expected_memory_shape=_SHAPE, expected_world_size=1
        )

    wrong_dtype = torch.zeros(_SHAPE, dtype=torch.float64)
    with pytest.raises(ExpertManifoldError, match="tensor schema"):
        save_v6_prior_checkpoint(
            output_dir=tmp_path / "wrong_dtype",
            macro=1,
            memory=wrong_dtype,
            context=_context(),
            metrics_rows=1,
            cursor_contract=_cursor(),
            checkpoint_contract=_contract(),
            expected_memory_shape=_SHAPE,
        )
    nonfinite = torch.full(_SHAPE, torch.inf, dtype=torch.float32)
    with pytest.raises(ExpertManifoldError, match="non-finite"):
        save_v6_prior_checkpoint(
            output_dir=tmp_path / "nonfinite",
            macro=1,
            memory=nonfinite,
            context=_context(),
            metrics_rows=1,
            cursor_contract=_cursor(),
            checkpoint_contract=_contract(),
            expected_memory_shape=_SHAPE,
        )
