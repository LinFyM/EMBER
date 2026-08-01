from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

import ember.writer.endpoint_provenance as endpoint_provenance
from ember.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    LoRATarget,
    SmolVLALoRAContract,
    canonical_contract_sha256,
    lora_state_sha256,
)
from ember.pi05_source_checkpoint import (
    canonical_hash,
    sha256_file,
    write_json_atomic,
)
from ember.writer.endpoint_provenance import (
    PORTABLE_CACHE_SCHEMA,
    PORTABLE_EXTENSION_PATHS,
    PORTABLE_INFORMATION_WALL,
    SEALED_PANEL_PAYLOAD_SHA256,
    EndpointLoRAEntry,
    _expected_generation_descriptor,
    _portable_cache_entries,
    _resolve_generation_training_config,
    _validate_generation_git_and_config,
    _validate_generation_hdf5,
    _validated_payload,
)
from ember.writer.endpoint_validation import _verify_lora_entry
from ember.writer.model import WriterModelError


def _portable_cache_case(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    dict[str, Any],
    SmolVLALoRAContract,
    dict[tuple[int, int], int],
]:
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    write_json_atomic(evaluation / "run_contract.json", {})
    write_json_atomic(evaluation / "results.json", {})
    lora = SmolVLALoRAContract(
        targets=(LoRATarget("proj", 2, 2),),
        rank=1,
        alpha=1,
        dropout=0.0,
        identity_seed=7,
    )
    contract = {
        "contract_sha256": "1" * 64,
        "adapter": {
            "config": {"sha256": "5" * 64},
            "training_run": {"git_commit": "4" * 40},
            "checkpoint": {
                "path": str(tmp_path / "checkpoint"),
                "cursor": 10,
                "manifest_file_sha256": "2" * 64,
                "manifest_payload_sha256": "6" * 64,
                "writer_state_sha256": "3" * 64,
            },
        },
    }
    panel_conditions = {(1, 4): 0, (3, 5): 1}
    portable = {
        "schema_version": PORTABLE_CACHE_SCHEMA,
        "candidate": {
            "family": "v52",
            "candidate_id": "v52_step00000010",
            "evaluation_root": str(evaluation),
            "run_contract_file_sha256": sha256_file(
                evaluation / "run_contract.json"
            ),
            "run_contract_sha256": contract["contract_sha256"],
            "results_file_sha256": sha256_file(
                evaluation / "results.json"
            ),
            "adapter_sha256": canonical_hash(contract["adapter"]),
            "writer_constructor_git_commit": "4" * 40,
            "correct400": 7,
            "task_breadth": 2,
            "checkpoint": dict(contract["adapter"]["checkpoint"]),
        },
        "generation_run": {},
        "training_config": {},
        "panel_manifest_payload_sha256": SEALED_PANEL_PAYLOAD_SHA256,
        "lora_contract_sha256": canonical_contract_sha256(lora),
        "information_wall": dict(PORTABLE_INFORMATION_WALL),
        "entries": [
            {
                "global_task_id": task_id,
                "teacher_demo_index": demo,
                "lora_file": {
                    "path": (
                        f"loras/task_{task_id:03d}_demo_{demo:03d}.safetensors"
                    ),
                    "bytes": 10,
                    "sha256": str(task_id) * 64,
                },
                "lora_state_sha256": str(demo) * 64,
                "generation_evidence": {
                    "language_global_task_id": task_id,
                    "video_global_task_id": task_id,
                    "video_group": video_group,
                    "teacher_demo_index": demo,
                    "condition": "correct",
                    "video_order": "forward",
                    "frame_stride": 5,
                    "writer_checkpoint_cursor": 10,
                    "writer_checkpoint_manifest_sha256": "2" * 64,
                    "writer_state_sha256": "3" * 64,
                    "lora_sha256": str(demo) * 64,
                    "lora_tensor_count": 76,
                    "rank": 0,
                    "generation_wall_seconds": 0.1,
                },
            }
            for (task_id, demo), video_group in panel_conditions.items()
        ],
    }
    portable["canonical_payload_sha256"] = canonical_hash(portable)
    path = tmp_path / "portable.json"
    write_json_atomic(path, portable)
    return path, evaluation, contract, lora, panel_conditions


def test_validated_payload_fails_closed(tmp_path: Path) -> None:
    payload = {"schema_version": "example_v1", "value": 3}
    payload["canonical_payload_sha256"] = canonical_hash(payload)
    path = tmp_path / "payload.json"
    write_json_atomic(path, payload)
    assert _validated_payload(path, "example_v1")["value"] == 3
    payload["value"] = 4
    write_json_atomic(path, payload)
    with pytest.raises(WriterModelError, match="artifact changed"):
        _validated_payload(path, "example_v1")


def test_portable_cache_information_wall_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, evaluation, contract, lora, panel_conditions = _portable_cache_case(
        tmp_path
    )
    monkeypatch.setattr(
        endpoint_provenance,
        "_validate_portable_generation_run",
        lambda *_args, **_kwargs: {},
    )
    entries, _manifest_sha = _portable_cache_entries(
        path,
        evaluation,
        contract,
        panel_conditions,
        lora,
        "v52",
        7,
        2,
    )
    assert set(entries) == set(panel_conditions)

    portable = json.loads(path.read_text())
    portable.pop("canonical_payload_sha256")
    portable["information_wall"][
        "validation_action_values_read_during_generation"
    ] = 1
    portable["canonical_payload_sha256"] = canonical_hash(portable)
    write_json_atomic(path, portable)
    with pytest.raises(WriterModelError, match="authority changed"):
        _portable_cache_entries(
            path,
            evaluation,
            contract,
            panel_conditions,
            lora,
            "v52",
            7,
            2,
        )


def test_portable_generation_git_config_and_descriptor_are_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor = "1" * 40
    extension = "2" * 40
    training_config = {
        "writer": {
            "teacher_prompt": "predict the action",
            "max_frames_per_encoder_call": 16,
            "frame_stride": 5,
            "camera_dataset": "obs/agentview_rgb",
            "camera_transform": "libero_opengl_rotate_180_chw_uint8",
            "include_final_frame": True,
        }
    }
    blob = json.dumps(training_config).encode()
    digest = hashlib.sha256(blob).hexdigest()
    run = {
        "git": {
            "constructor_commit": constructor,
            "extension_commit": extension,
            "origin_main_commit": "3" * 40,
            "dirty_paths": [],
            "diff_paths": list(PORTABLE_EXTENSION_PATHS),
        },
        "training_run": {
            "config": {
                "repo_relative_path": "configs/writer.json",
                "file_sha256": digest,
                "git_blob_sha256": digest,
            }
        },
    }
    contract = {
        "adapter": {
            "training_run": {"git_commit": constructor},
            "config": {"sha256": digest},
        }
    }
    monkeypatch.setattr(
        endpoint_provenance,
        "_generation_git_diff",
        lambda *_args: PORTABLE_EXTENSION_PATHS,
    )
    monkeypatch.setattr(
        endpoint_provenance, "_git_blob_bytes", lambda *_args: blob
    )
    assert _validate_generation_git_and_config(run, contract) == training_config
    descriptor = _expected_generation_descriptor(training_config)
    assert descriptor["video"]["frame_stride"] == 5
    assert descriptor["execution"]["policy_loss_computed"] is False
    run["git"]["dirty_paths"] = ["src/ember/writer/model.py"]
    with pytest.raises(WriterModelError, match="Git authority"):
        _validate_generation_git_and_config(run, contract)


def test_portable_v6_recipe_overlay_resolves_only_its_pinned_git_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = {
        "schema_version": "ember_pi05_language_axial_as_writer_v6",
        "base_config": None,
        "writer": {
            "teacher_prompt": "predict the action",
            "max_frames_per_encoder_call": 32,
            "frame_stride": 5,
            "camera_dataset": "obs/agentview_rgb",
            "camera_transform": "libero_opengl_rotate_180_chw_uint8",
            "include_final_frame": True,
        },
    }
    blob = json.dumps(base).encode()
    overlay = {
        "schema_version": (
            "ember_pi05_language_axial_as_writer_recipe_overlay_v1"
        ),
        "base_config": "configs/pi05_as_writer_language_axial_v6.json",
        "base_sha256": hashlib.sha256(blob).hexdigest(),
        "replace": {
            "data": {},
            "conditioning_training": {},
            "optimization": {},
            "profile_defaults": {},
            "profile_evidence": {},
            "formal_run": {},
        },
    }
    run = {"git": {"constructor_commit": "1" * 40}}
    monkeypatch.setattr(
        endpoint_provenance, "_git_blob_bytes", lambda *_args: blob
    )
    resolved = _resolve_generation_training_config(run, overlay)
    assert resolved["writer"] == base["writer"]
    assert _expected_generation_descriptor(resolved)["video"]["frame_stride"] == 5

    changed = {**overlay, "base_sha256": "0" * 64}
    with pytest.raises(WriterModelError, match="recipe base changed"):
        _resolve_generation_training_config(run, changed)


def test_portable_hdf5_authority_is_metadata_only_and_fail_closed() -> None:
    expected = {
        task_id: {"bytes": 1000 + task_id, "sha256": f"{task_id + 1:x}" * 64}
        for task_id in range(8)
    }
    files = [
        {
            "global_task_id": task_id,
            "bytes": authority["bytes"],
            "sealed_expected_sha256": authority["sha256"],
            "episodes": [
                {
                    "demo_index": demo_index,
                    "action_rows_from_shape_metadata": 3,
                    "video_rows_from_shape_metadata": 3,
                    "action_width": 7,
                    "video_channels": 3,
                    "video_dtype": "uint8",
                }
                for demo_index in range(50)
            ],
        }
        for task_id, authority in expected.items()
    ]
    record = {
        "task_count": 8,
        "sealed_manifest_file_sha256": "f" * 64,
        "exact_file_size_verified": True,
        "hdf5_schema_metadata_verified": True,
        "runtime_full_sha256_computed": False,
        "validation_action_values_read": 0,
        "validation_video_values_read_during_schema_check": 0,
        "identity_sha256": canonical_hash(files),
        "files": files,
    }
    _validate_generation_hdf5(record, expected, set(range(8)), "f" * 64)
    record["runtime_full_sha256_computed"] = True
    with pytest.raises(WriterModelError, match="data authority"):
        _validate_generation_hdf5(record, expected, set(range(8)), "f" * 64)


def test_public_lora_tensor_hashes_and_state_hash_fail_closed(tmp_path: Path) -> None:
    lora = SmolVLALoRAContract(
        targets=(LoRATarget("proj", 2, 2),),
        rank=1,
        alpha=1,
        dropout=0.0,
        identity_seed=7,
    )
    state = {
        "proj" + LORA_A_SUFFIX: torch.tensor([[1.0, 2.0]]),
        "proj" + LORA_B_SUFFIX: torch.tensor([[3.0], [4.0]]),
    }
    path = tmp_path / "lora.safetensors"
    save_file(state, str(path))
    entry = EndpointLoRAEntry(
        path=path,
        bytes=path.stat().st_size,
        file_sha256=sha256_file(path),
        state_sha256=lora_state_sha256(state),
    )
    loaded = _verify_lora_entry(entry, lora, torch.device("cpu"))
    assert lora_state_sha256(loaded) == entry.state_sha256
    with pytest.raises(WriterModelError, match="state changed"):
        _verify_lora_entry(
            EndpointLoRAEntry(
                path=entry.path,
                bytes=entry.bytes,
                file_sha256=entry.file_sha256,
                state_sha256="0" * 64,
            ),
            lora,
            torch.device("cpu"),
        )
    nonfinite_state = {name: value.clone() for name, value in state.items()}
    nonfinite_state["proj" + LORA_A_SUFFIX][0, 0] = float("nan")
    nonfinite_path = tmp_path / "nonfinite.safetensors"
    save_file(nonfinite_state, str(nonfinite_path))
    with pytest.raises(WriterModelError, match="public LoRA tensor"):
        _verify_lora_entry(
            EndpointLoRAEntry(
                path=nonfinite_path,
                bytes=nonfinite_path.stat().st_size,
                file_sha256=sha256_file(nonfinite_path),
                state_sha256=lora_state_sha256(nonfinite_state),
            ),
            lora,
            torch.device("cpu"),
        )
