from __future__ import annotations

from pathlib import Path

import pytest

from ember.lora import canonical_contract_sha256
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import (
    canonical_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.writer.as_contract import (
    AS_WRITER_LAUNCH_SCHEMA,
    load_writer_config,
)
from ember.writer.checkpoint import (
    AS_WRITER_CHECKPOINT_SCHEMA,
    validate_writer_checkpoint_files,
)
from ember.writer.inference import inspect_as_writer_evaluation
from ember.writer.model import WriterModelError


ROOT = Path(__file__).resolve().parents[1]
AS_CONFIG = ROOT / "configs/pi05_as_writer_action_memory_v1.json"


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


def _sparse_video_data(tmp_path: Path, rows: list[dict]) -> Path:
    root = tmp_path / "target_data"
    for row in rows:
        path = root / row["hdf5"]["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
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
        "source_run_contract_sha256": "1" * 64,
        "checkpoint_manifest_sha256": "2" * 64,
        "optimizer_step": 1000,
        "source_run_summary_sha256": "3" * 64,
        "source_training_commit": "4" * 40,
        "source_base_config_sha256": config["authorities"]["source_base_config"][
            "sha256"
        ],
        "source_authority_hashes": {"normalization": "5" * 64},
        "model_files": [
            {"path": "policy/model.safetensors", "bytes": 1, "sha256": "6" * 64}
        ],
    }
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "step_00000004"
    checkpoint.mkdir(parents=True)
    lora = load_pi05_lora_contract(
        ROOT / config["authorities"]["lora_contract"]["path"]
    )
    training_video = {
        "root": str(data_root.resolve()),
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
        "source": source,
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
            "checkpoint_steps": [4],
        },
    }
    write_json_atomic(run / "run_contract.json", training)
    for name in (
        "writer.safetensors",
        "trainer_state.pt",
        *(
            f"rank_{rank:02d}_state.pt"
            for rank in range(config["formal_run"]["expected_world_size"])
        ),
    ):
        (checkpoint / name).write_bytes(name.encode("utf-8"))
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(checkpoint.iterdir())
    }
    manifest = {
        "schema_version": AS_WRITER_CHECKPOINT_SCHEMA,
        "contract_sha256": canonical_hash(training),
        "consumed": {"next_step": 4},
        "files": files,
    }
    manifest["canonical_payload_sha256"] = canonical_hash(manifest)
    write_json_atomic(checkpoint / "checkpoint_manifest.json", manifest)
    return checkpoint, data_root, source, validation_keys


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
    assert adapter["video_data"]["task_count"] == 8
    assert all(
        row["suite"] != row["video_suite"]
        and row["language_split_role"] == row["video_split_role"] == "validation"
        for row in adapter["task_video_mapping"]
    )

    generic = inspect_as_writer_evaluation(
        config_path=AS_CONFIG,
        checkpoint=checkpoint,
        video_data_root=data_root,
        source=source,
        task_keys=validation_keys,
        video_condition="generic_correct",
        video_seed=7,
        require_formal=False,
    )
    assert generic["writer_language_condition"] == "generic_neutral"
    assert generic["wrong_video_mapping"] == "identity"

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
