from __future__ import annotations

from pathlib import Path

import pytest

from ember.lora import canonical_contract_sha256
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import canonical_hash, read_json, sha256_file, write_json_atomic
from ember.writer.as_contract import inspect_feature_cache, load_writer_config
from ember.writer.checkpoint import (
    AS_WRITER_CHECKPOINT_SCHEMA,
    validate_writer_checkpoint_files,
)
from ember.writer.feature_cache import (
    PI05_FEATURE_CACHE_MANIFEST_SCHEMA,
    PI05_TASK_FEATURE_CACHE_SCHEMA,
)
from ember.writer.inference import inspect_as_writer_evaluation
from ember.writer.model import WriterModelError


ROOT = Path(__file__).resolve().parents[1]
AS_CONFIG = ROOT / "configs/pi05_as_writer_v2.json"


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


def _static_as_evaluation_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    config = load_writer_config(AS_CONFIG)
    target = read_json(ROOT / config["authorities"]["target_data_manifest"]["path"])
    development_ids = sorted(
        row["global_task_id"]
        for row in target["tasks"]
        if row["split_role"] in {"train", "validation"}
    )
    source = {
        "source_run_contract_sha256": "1" * 64,
        "checkpoint_manifest_sha256": "2" * 64,
        "optimizer_step": 30_000,
        "source_run_summary_sha256": "3" * 64,
        "source_training_commit": "4" * 40,
        "source_base_config_sha256": config["authorities"]["source_base_config"]["sha256"],
        "source_authority_hashes": {"normalization": "5" * 64},
        "model_files": [
            {"path": "policy/model.safetensors", "bytes": 1, "sha256": "6" * 64}
        ],
    }
    cache = tmp_path / "cache"
    cache.mkdir()
    extraction = "7" * 64
    cache_contract = {
        "schema_version": "ember_pi05_writer_feature_cache_launch_v2",
        "mode": "formal",
        "role": "development",
        "config_sha256": config["authorities"]["feature_cache_config"]["sha256"],
        "source": source,
        "test_video_values_read": 0,
        "extraction_sha256": extraction,
    }
    cache_contract["contract_sha256"] = canonical_hash(cache_contract)
    write_json_atomic(cache / "run_contract.json", cache_contract)
    records = [
        {
            "schema_version": PI05_TASK_FEATURE_CACHE_SCHEMA,
            "task_id": task_id,
            "extraction_sha256": extraction,
        }
        for task_id in development_ids
    ]
    cache_manifest = {
        "schema_version": PI05_FEATURE_CACHE_MANIFEST_SCHEMA,
        "contract_sha256": cache_contract["contract_sha256"],
        "extraction_sha256": extraction,
        "task_count": 32,
        "episode_count": 1600,
        "frame_count": 123,
        "task_records": records,
    }
    cache_manifest["canonical_payload_sha256"] = canonical_hash(cache_manifest)
    write_json_atomic(cache / "cache_manifest.json", cache_manifest)
    cache_summary = inspect_feature_cache(
        cache, config, source, target["summary"]["roles"]["validation"]
    )

    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "step_00000004"
    checkpoint.mkdir(parents=True)
    lora = load_pi05_lora_contract(ROOT / config["authorities"]["lora_contract"]["path"])
    training = {
        "schema_version": "ember_pi05_as_writer_launch_v2",
        "mode": "profile",
        "git": {"commit": "8" * 40},
        "config_sha256": sha256_file(AS_CONFIG),
        "authorities": config["authorities"],
        "source": source,
        "feature_cache": cache_summary,
        "information_wall": config["information_wall"],
        "writer": config["writer"],
        "data": config["data"],
        "task_ids": target["summary"]["roles"]["train"],
        "trainable": {
            "object": "shared_action_supervised_writer_only",
            "lora_contract_sha256": canonical_contract_sha256(lora),
        },
        "runtime": {"world_size": 8, "checkpoint_steps": [4]},
    }
    write_json_atomic(run / "run_contract.json", training)
    for name in (
        "writer.safetensors",
        "trainer_state.pt",
        *(f"rank_{rank:02d}_state.pt" for rank in range(8)),
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
    return checkpoint, cache, source


def test_as_writer_evaluation_seals_source_checkpoint_cache_and_video_map(
    tmp_path: Path,
) -> None:
    checkpoint, cache, source = _static_as_evaluation_fixture(tmp_path)
    validation_keys = (
        ("libero_spatial", 1),
        ("libero_spatial", 3),
        ("libero_object", 1),
        ("libero_object", 3),
        ("libero_goal", 3),
        ("libero_goal", 6),
        ("libero_10", 1),
        ("libero_10", 2),
    )
    adapter = inspect_as_writer_evaluation(
        config_path=AS_CONFIG,
        checkpoint=checkpoint,
        feature_cache=cache,
        source=source,
        task_keys=validation_keys,
        video_condition="cross_suite_wrong",
        video_seed=7,
        require_formal=False,
    )
    assert adapter["arm"] == "as_writer_cross_suite_wrong_video"
    assert adapter["checkpoint"]["cursor"] == 4
    assert adapter["feature_cache"]["task_count"] == 32
    assert all(
        row["suite"] != row["video_suite"]
        and row["language_split_role"] == row["video_split_role"] == "validation"
        for row in adapter["task_video_mapping"]
    )

    changed = {**source, "optimizer_step": 29_999}
    with pytest.raises(WriterModelError, match="authority changed"):
        inspect_as_writer_evaluation(
            config_path=AS_CONFIG,
            checkpoint=checkpoint,
            feature_cache=cache,
            source=changed,
            task_keys=validation_keys,
            video_condition="correct",
            video_seed=7,
            require_formal=False,
        )

    test_keys = (
        ("libero_spatial", 6),
        ("libero_spatial", 8),
        ("libero_object", 0),
        ("libero_object", 7),
        ("libero_goal", 4),
        ("libero_goal", 7),
        ("libero_10", 0),
        ("libero_10", 3),
    )
    with pytest.raises(WriterModelError, match="feature cache changed"):
        inspect_as_writer_evaluation(
            config_path=AS_CONFIG,
            checkpoint=checkpoint,
            feature_cache=cache,
            source=source,
            task_keys=test_keys,
            video_condition="correct",
            video_seed=7,
            require_formal=False,
        )
