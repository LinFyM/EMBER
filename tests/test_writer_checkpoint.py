from __future__ import annotations

from pathlib import Path

import pytest

from ember.pi05_source_checkpoint import write_json_atomic
from ember.writer.as_config import load_writer_config
from ember.writer.checkpoint import _state_schemas, validate_writer_checkpoint_files
from ember.writer.checkpoint_schema import K4_LAYER_TRACE_M2P_CHECKPOINT_SCHEMA
from ember.writer.model import WriterModelError
from ember.writer.update_contract import checkpoint_state_family


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pi05_as_writer_k4_layer_trace_m2p_bci_v1.json"
FAMILY = "k4_policy_layer_trace_m2p_full24_v1"


def test_k4_uses_fresh_incompatible_checkpoint_family() -> None:
    config = load_writer_config(CONFIG)
    assert checkpoint_state_family(config) == FAMILY
    assert _state_schemas(1, FAMILY)[0] == K4_LAYER_TRACE_M2P_CHECKPOINT_SCHEMA
    with pytest.raises(WriterModelError, match="unsupported"):
        _state_schemas(1, "cvadr_task_query_keyed_rawfull24_v1")


def _checkpoint(tmp_path: Path, contract_reference: str) -> Path:
    checkpoint = tmp_path / "step_00000003"
    checkpoint.mkdir()
    for name, value in {
        "writer.safetensors": b"writer",
        "trainer_state.pt": b"trainer",
        "rank_00_state.pt": b"rank",
    }.items():
        (checkpoint / name).write_bytes(value)
    files = {
        path.name: {"bytes": path.stat().st_size}
        for path in sorted(checkpoint.iterdir())
    }
    write_json_atomic(
        checkpoint / "checkpoint_manifest.json",
        {
            "schema_version": K4_LAYER_TRACE_M2P_CHECKPOINT_SCHEMA,
            "contract_reference": contract_reference,
            "consumed": {
                "next_step": 3,
                "optimizer_updates_per_task_cycle": 1,
                "checkpoint_state_family": FAMILY,
            },
            "files": files,
        },
    )
    return checkpoint


def test_k4_checkpoint_uses_schema_and_sizes_without_content_hashing(tmp_path: Path) -> None:
    reference = "ember_pi05_k4_policy_layer_trace_m2p_launch_v1"
    checkpoint = _checkpoint(tmp_path, reference)
    manifest = validate_writer_checkpoint_files(
        checkpoint, world_size=1, contract_sha256=reference
    )
    assert manifest["consumed"]["next_step"] == 3
    assert all("sha256" not in record for record in manifest["files"].values())
    (checkpoint / "trainer_state.pt").write_bytes(b"size-changed")
    with pytest.raises(WriterModelError, match="checkpoint file changed"):
        validate_writer_checkpoint_files(
            checkpoint, world_size=1, contract_sha256=reference
        )


def test_historical_checkpoint_schema_is_rejected_for_k4(tmp_path: Path) -> None:
    reference = "ember_pi05_k4_policy_layer_trace_m2p_launch_v1"
    checkpoint = _checkpoint(tmp_path, reference)
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = __import__("json").loads(manifest_path.read_text())
    manifest["schema_version"] = "ember_pi05_language_axial_writer_checkpoint_v6"
    write_json_atomic(manifest_path, manifest)
    with pytest.raises(WriterModelError, match="manifest changed"):
        validate_writer_checkpoint_files(
            checkpoint,
            world_size=1,
            contract_sha256=reference,
            allow_historical_v6_warmstart=True,
        )
