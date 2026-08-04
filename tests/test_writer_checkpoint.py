from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from ember.lora import canonical_contract_sha256
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import (
    canonical_hash,
    read_json,
    sha256_file,
    source_reference_matches,
    write_json_atomic,
)
from ember.writer.as_contract import (
    AS_WRITER_LAUNCH_SCHEMA,
    load_writer_config,
)
from ember.writer.checkpoint import (
    AS_WRITER_CHECKPOINT_SCHEMA,
    _state_schemas,
    validate_writer_checkpoint_files,
)
from ember.writer.checkpoint_schema import (
    TARGET_OWNED_FACTOR_TASK_QUERY_RAW_CHECKPOINT_SCHEMA,
)
from ember.writer.inference import inspect_as_writer_evaluation
from ember.writer.model import WriterModelError
from ember.writer.update_contract import checkpoint_state_family


ROOT = Path(__file__).resolve().parents[1]
AS_CONFIG = (
    ROOT
    / "configs/pi05_as_writer_target_owned_factor_full24_decay400_bci_v1.json"
)


def test_target_owned_factor_uses_fresh_incompatible_checkpoint_family() -> None:
    config = load_writer_config(AS_CONFIG)
    family = checkpoint_state_family(config)
    assert family == "target_owned_factor_task_query_keyed_rawfull24_v1"
    assert _state_schemas(1, family)[0] == (
        TARGET_OWNED_FACTOR_TASK_QUERY_RAW_CHECKPOINT_SCHEMA
    )
    with pytest.raises(WriterModelError, match="unsupported"):
        _state_schemas(1, "cvadr_task_query_keyed_rawfull24_v1")


def _checkpoint(tmp_path: Path, contract_sha256: str) -> Path:
    checkpoint = tmp_path / "step_00000003"
    checkpoint.mkdir()
    for name, value in {
        "writer.safetensors": b"writer",
        "trainer_state.pt": b"trainer",
        "rank_00_state.pt": b"rank",
    }.items():
        (checkpoint / name).write_bytes(value)
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(checkpoint.iterdir())
    }
    manifest = {
        "schema_version": AS_WRITER_CHECKPOINT_SCHEMA,
        "contract_sha256": contract_sha256,
        "consumed": {"next_step": 3},
        "files": files,
    }
    manifest["canonical_payload_sha256"] = canonical_hash(manifest)
    write_json_atomic(checkpoint / "checkpoint_manifest.json", manifest)
    return checkpoint


def test_as_writer_checkpoint_verifies_every_file_before_pickle_load(
    tmp_path: Path,
) -> None:
    contract = "a" * 64
    checkpoint = _checkpoint(tmp_path, contract)
    manifest = validate_writer_checkpoint_files(
        checkpoint, world_size=1, contract_sha256=contract
    )
    assert manifest["consumed"]["next_step"] == 3
    (checkpoint / "trainer_state.pt").write_bytes(b"changed")
    with pytest.raises(WriterModelError, match="checkpoint file changed"):
        validate_writer_checkpoint_files(
            checkpoint, world_size=1, contract_sha256=contract
        )


def _tensor_checkpoint(
    run: Path,
    *,
    step: int,
    training: dict,
    state: dict[str, torch.Tensor],
) -> Path:
    checkpoint = run / "checkpoints" / f"step_{step:08d}"
    checkpoint.mkdir(parents=True)
    save_file(state, str(checkpoint / "writer.safetensors"))
    (checkpoint / "trainer_state.pt").write_bytes(b"trainer")
    for rank in range(int(training["runtime"]["world_size"])):
        (checkpoint / f"rank_{rank:02d}_state.pt").write_bytes(
            f"rank{rank}".encode()
        )
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(checkpoint.iterdir())
    }
    manifest = {
        "schema_version": AS_WRITER_CHECKPOINT_SCHEMA,
        "contract_sha256": canonical_hash(training),
        "consumed": {"next_step": step},
        "files": files,
    }
    manifest["canonical_payload_sha256"] = canonical_hash(manifest)
    write_json_atomic(checkpoint / "checkpoint_manifest.json", manifest)
    return checkpoint


def _sparse_video_data(tmp_path: Path, rows: list[dict]) -> Path:
    root = tmp_path / "target_data"
    for row in rows:
        path = root / row["hdf5"]["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            data = handle.create_group("data")
            for demo_index in range(50):
                observation = data.create_group(
                    f"demo_{demo_index}"
                ).create_group("obs")
                observation.create_dataset(
                    "agentview_rgb",
                    data=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                )
        with path.open("r+b") as handle:
            handle.truncate(int(row["hdf5"]["bytes"]))
    return root


def _static_as_evaluation_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict, tuple[tuple[str, int], ...]]:
    config = load_writer_config(AS_CONFIG)
    target_path = ROOT / config["authorities"]["target_data_manifest"]["path"]
    target = read_json(target_path)
    validation_rows = [
        row for row in target["tasks"] if row["split_role"] == "validation"
    ]
    validation_keys = tuple(
        (str(row["suite"]), int(row["task_id"])) for row in validation_rows
    )
    data_root = _sparse_video_data(tmp_path, validation_rows)
    source = {
        "source_run": "/data1/user/ymdai/projects/EMBER/runs/outputs/source",
        "checkpoint": "/data1/user/ymdai/projects/EMBER/runs/outputs/source/checkpoints/step_00001000",
        "model_path": "/data1/user/ymdai/projects/EMBER/runs/outputs/source/checkpoints/step_00001000/policy",
        "source_run_contract_sha256": "1" * 64,
        "checkpoint_manifest_sha256": "2" * 64,
        "optimizer_step": 1000,
        "source_run_summary_sha256": "3" * 64,
        "source_training_commit": "4" * 40,
        "frozen_policy_subdir": "policy",
        "source_base_config_sha256": config["authorities"]["source_base_config"][
            "sha256"
        ],
        "source_authority_hashes": {"normalization": "5" * 64},
        "model_files": [
            {"path": "policy/model.safetensors", "bytes": 1, "sha256": "6" * 64}
        ],
    }
    run = tmp_path / "run"
    run.mkdir()
    lora = load_pi05_lora_contract(
        ROOT / config["authorities"]["lora_contract"]["path"]
    )
    training_video = {
        "root": "/data/ymdai/ember_data/LIBERO-datasets/historical-revision",
        "dataset": target["dataset"],
        "target_data_manifest_file_sha256": sha256_file(target_path),
    }
    training = {
        "schema_version": AS_WRITER_LAUNCH_SCHEMA,
        "mode": "profile",
        "stage": "development",
        "git": {"commit": "8" * 40},
        "config_sha256": sha256_file(AS_CONFIG),
        "authorities": config["authorities"],
        "source": {
            **source,
            "source_run": "/data/ymdai/outputs/ember/source",
            "checkpoint": "/data/ymdai/outputs/ember/source/checkpoints/step_00001000",
            "model_path": "/data/ymdai/outputs/ember/source/checkpoints/step_00001000/policy",
            "source_run_contract_sha256": "a" * 64,
            "checkpoint_manifest_sha256": "b" * 64,
        },
        "video_data": training_video,
        "information_wall": config["information_wall"],
        "writer": config["writer"],
        "data": config["data"],
        "task_ids": target["summary"]["roles"]["train"],
        "trainable": {
            "object": "shared_action_supervised_writer_only",
            "lora_contract_sha256": canonical_contract_sha256(lora),
        },
        "runtime": {
            "world_size": config["formal_run"]["expected_world_size"],
            "checkpoint_steps": [4, 6],
        },
    }
    write_json_atomic(run / "run_contract.json", training)
    checkpoint = _tensor_checkpoint(
        run,
        step=4,
        training=training,
        state={"weight": torch.tensor([1.0])},
    )
    _tensor_checkpoint(
        run,
        step=6,
        training=training,
        state={"weight": torch.tensor([3.0])},
    )
    return checkpoint, data_root, source, validation_keys


def test_source_reference_match_is_host_and_hash_independent() -> None:
    recorded = {
        "source_run": "/data/ymdai/outputs/ember/source",
        "checkpoint": "/data/ymdai/outputs/ember/source/checkpoints/step_00001000",
        "optimizer_step": 1000,
        "frozen_policy_subdir": "policy",
        "source_training_commit": "1" * 40,
        "checkpoint_manifest_sha256": "2" * 64,
        "model_files": [
            {"path": "policy/model.safetensors", "bytes": 123, "sha256": "3" * 64}
        ],
    }
    current = {
        **recorded,
        "source_run": "/data1/user/ymdai/projects/EMBER/runs/outputs/source",
        "checkpoint": "/data1/user/ymdai/projects/EMBER/runs/outputs/source/checkpoints/step_00001000",
        "checkpoint_manifest_sha256": "4" * 64,
        "model_files": [
            {"path": "policy/model.safetensors", "bytes": 123, "sha256": "5" * 64}
        ],
    }
    assert source_reference_matches(recorded, current)
    assert not source_reference_matches(recorded, {**current, "optimizer_step": 999})


def test_as_writer_evaluation_seals_raw_video_authority_and_wrong_map(
    tmp_path: Path,
) -> None:
    checkpoint, data_root, source, validation_keys = _static_as_evaluation_fixture(
        tmp_path
    )
    adapter = inspect_as_writer_evaluation(
        config_path=AS_CONFIG,
        checkpoint=checkpoint,
        video_data_root=data_root,
        source=source,
        task_keys=validation_keys,
        video_condition="cross_suite_wrong",
        video_seed=7,
        require_formal=False,
    )
    assert adapter["arm"] == "as_writer_cross_suite_wrong_video"
    assert adapter["checkpoint"]["cursor"] == 4
    assert "kind" not in adapter["checkpoint"]
    assert adapter["video_data"]["task_count"] == 8
    assert all(
        row["suite"] != row["video_suite"]
        and row["language_split_role"] == row["video_split_role"] == "validation"
        for row in adapter["task_video_mapping"]
    )

    shuffled = inspect_as_writer_evaluation(
        config_path=AS_CONFIG,
        checkpoint=checkpoint,
        video_data_root=data_root,
        source=source,
        task_keys=validation_keys,
        video_condition="shuffled",
        video_seed=7,
        require_formal=False,
    )
    assert shuffled["video_condition"] == "shuffled"
    assert shuffled["wrong_video_mapping"] == "identity"

    exploratory_source = {**source, "source_run_summary_sha256": None}
    exploratory = inspect_as_writer_evaluation(
        config_path=AS_CONFIG,
        checkpoint=checkpoint,
        video_data_root=data_root,
        source=exploratory_source,
        task_keys=validation_keys,
        video_condition="correct",
        video_seed=7,
        require_formal=False,
    )
    assert exploratory["checkpoint"]["cursor"] == 4

    changed = {**source, "optimizer_step": 999}
    with pytest.raises(WriterModelError, match="authority changed"):
        inspect_as_writer_evaluation(
            config_path=AS_CONFIG,
            checkpoint=checkpoint,
            video_data_root=data_root,
            source=changed,
            task_keys=validation_keys,
            video_condition="correct",
            video_seed=7,
            require_formal=False,
        )


def test_as_writer_evaluation_rejects_non_single_checkpoint_paths(
    tmp_path: Path,
) -> None:
    checkpoint, data_root, source, validation_keys = _static_as_evaluation_fixture(
        tmp_path
    )
    derived = checkpoint.parent.parent / "derived_checkpoints" / "mean"
    derived.mkdir(parents=True)
    with pytest.raises(WriterModelError, match="outside a training run"):
        inspect_as_writer_evaluation(
            config_path=AS_CONFIG,
            checkpoint=derived,
            video_data_root=data_root,
            source=source,
            task_keys=validation_keys,
            video_condition="correct",
            video_seed=7,
            require_formal=False,
        )
