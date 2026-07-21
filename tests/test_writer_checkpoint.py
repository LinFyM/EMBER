from __future__ import annotations

from pathlib import Path

import pytest

from ember.pi05_source_checkpoint import canonical_hash, sha256_file, write_json_atomic
from ember.writer.checkpoint import (
    AS_WRITER_CHECKPOINT_SCHEMA,
    validate_writer_checkpoint_files,
)
from ember.writer.model import WriterModelError


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
