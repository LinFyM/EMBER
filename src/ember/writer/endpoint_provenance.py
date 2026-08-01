"""Fail-closed provenance for historical public-LoRA endpoint caches.

Historical Writer constructors stay executable only in their frozen Git
worktrees.  The active endpoint diagnostic consumes their public LoRA tensors
only after this module has reconciled the generator, training, outcome, panel,
data, and per-entry authorities.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ember.lora import canonical_contract_sha256
from ember.pi05_source_checkpoint import canonical_hash, read_json, sha256_file
from ember.writer.as_contract import REPO_ROOT
from ember.writer.model import WriterModelError
from ember.writer.validation_panel import PANEL_MANIFEST_SCHEMA, PANEL_SCHEMA


PORTABLE_CACHE_SCHEMA = "ember_pi05_endpoint_public_lora_cache_v2"
PORTABLE_GENERATION_RUN_SCHEMA = (
    "ember_pi05_historical_endpoint_public_lora_generation_run_v1"
)
SEALED_PANEL_PAYLOAD_SHA256 = (
    "97ba7b95c48124858f01b50a1400172ad69eae62e7796f54357caed140174b4d"
)
PORTABLE_EXTENSION_PATHS = (
    "scripts/evaluate_as_writer_validation_loss.py",
    "src/ember/writer/validation.py",
    "tests/test_pi05_validation_loss.py",
)
PORTABLE_INFORMATION_WALL = {
    "validation_action_values_read_during_generation": 0,
    "test_action_reads": 0,
    "test_video_value_reads": 0,
    "environment_steps": 0,
}


@dataclass(frozen=True)
class EndpointLoRAEntry:
    path: Path
    bytes: int
    file_sha256: str
    state_sha256: str


def _validated_payload(path: Path, schema: str) -> dict[str, Any]:
    payload = read_json(path)
    digest = payload.get("canonical_payload_sha256")
    unhashed = {
        key: value
        for key, value in payload.items()
        if key != "canonical_payload_sha256"
    }
    if payload.get("schema_version") != schema or canonical_hash(unhashed) != digest:
        raise WriterModelError(f"endpoint artifact changed: {path}")
    return payload


def _git_blob_bytes(commit: str, relative_path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise WriterModelError(
            "portable endpoint generation config blob is unavailable"
        ) from error


def _generation_git_diff(
    constructor: str,
    extension: str,
) -> tuple[str, ...]:
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", constructor, extension],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{constructor}..{extension}"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise WriterModelError(
            "portable endpoint generation commit left its constructor"
        ) from error
    return tuple(value for value in result.splitlines() if value)


def _expected_generation_descriptor(
    training_config: Mapping[str, Any],
) -> dict[str, Any]:
    writer = training_config.get("writer", {})
    teacher_prompt = str(writer.get("teacher_prompt", ""))
    max_frames = int(writer.get("max_frames_per_encoder_call", -1))
    if (
        not teacher_prompt
        or max_frames <= 0
        or int(writer.get("frame_stride", -1)) != 5
        or writer.get("camera_dataset") != "obs/agentview_rgb"
        or writer.get("camera_transform")
        != "libero_opengl_rotate_180_chw_uint8"
        or writer.get("include_final_frame") is not True
    ):
        raise WriterModelError(
            "portable endpoint generation training config changed"
        )
    return {
        "operation": "cache_only_public_lora",
        "writer_inputs": {
            "task_language": True,
            "exactly_one_raw_action_hidden_teacher_video": True,
            "teacher_action": False,
            "state_or_proprio": False,
            "reward_or_terminal": False,
            "task_id_or_filename": False,
        },
        "video": {
            "camera_dataset": "obs/agentview_rgb",
            "camera_transform": "libero_opengl_rotate_180_chw_uint8",
            "frame_stride": 5,
            "include_final_frame": True,
            "order": "forward",
            "multiple_video_average": False,
        },
        "preprocessing": {
            "video_loader": "RawTeacherVideoStore",
            "language_tokenizer": "Pi05TeacherPrefixTokenizer",
            "teacher_prompt": teacher_prompt,
            "max_frames_per_encoder_call": max_frames,
        },
        "execution": {
            "policy_loss_computed": False,
            "environment_created": False,
            "gradients_computed": False,
            "optimizer_updates": 0,
            "writer_eval_mode": True,
            "writer_forward_precision": "cuda_bfloat16_autocast",
        },
        "public_lora": {
            "target_count": 38,
            "rank": 16,
            "tensor_count": 76,
            "sets_per_video": 1,
            "checkpoint_fusion": False,
            "multiple_lora_average": False,
            "saved_format": "safetensors_native_dtype_complete_state",
        },
    }


def _validate_generation_git_and_config(
    run: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    git = run.get("git", {})
    adapter = contract["adapter"]
    constructor = str(git.get("constructor_commit", ""))
    extension = str(git.get("extension_commit", ""))
    origin_main = str(git.get("origin_main_commit", ""))
    config = run.get("training_run", {}).get("config", {})
    relative = str(config.get("repo_relative_path", ""))
    relative_path = PurePosixPath(relative)
    if (
        not re.fullmatch(r"[0-9a-f]{40}", constructor)
        or not re.fullmatch(r"[0-9a-f]{40}", extension)
        or not re.fullmatch(r"[0-9a-f]{40}", origin_main)
        or constructor != adapter.get("training_run", {}).get("git_commit")
        or git.get("dirty_paths") != []
        or tuple(git.get("diff_paths", [])) != PORTABLE_EXTENSION_PATHS
        or relative_path.is_absolute()
        or not relative
        or ".." in relative_path.parts
        or not relative.startswith("configs/")
    ):
        raise WriterModelError(
            "portable endpoint generation Git authority changed"
        )
    if _generation_git_diff(constructor, extension) != PORTABLE_EXTENSION_PATHS:
        raise WriterModelError(
            "portable endpoint generation extension paths changed"
        )
    blob = _git_blob_bytes(constructor, relative)
    blob_sha256 = hashlib.sha256(blob).hexdigest()
    if (
        blob_sha256 != config.get("git_blob_sha256")
        or blob_sha256 != config.get("file_sha256")
        or blob_sha256 != adapter.get("config", {}).get("sha256")
    ):
        raise WriterModelError(
            "portable endpoint generation config authority changed"
        )
    try:
        payload = json.loads(blob)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WriterModelError(
            "portable endpoint generation config is not JSON"
        ) from error
    if not isinstance(payload, Mapping):
        raise WriterModelError(
            "portable endpoint generation config payload changed"
        )
    return payload


def _resolve_portable_generation_run(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Path, Path, Mapping[str, Any]]:
    generation = manifest.get("generation_run", {})
    if (
        not isinstance(generation, Mapping)
        or generation.get("run_contract_path") != "../../run_contract.json"
    ):
        raise WriterModelError(
            "portable endpoint generation run path changed"
        )
    generation_root = manifest_path.parents[2].resolve()
    run_path = (
        manifest_path.parent / str(generation["run_contract_path"])
    ).resolve()
    if run_path != generation_root / "run_contract.json":
        raise WriterModelError(
            "portable endpoint generation run escaped its output root"
        )
    run = _validated_payload(run_path, PORTABLE_GENERATION_RUN_SCHEMA)
    expected_keys = {
        "schema_version",
        "mode",
        "host",
        "command",
        "git",
        "training_run",
        "checkpoints",
        "outcome_evaluations",
        "source",
        "tokenizer",
        "data",
        "panel",
        "lora",
        "generation_descriptor",
        "generation_descriptor_sha256",
        "information_wall",
        "world_size",
        "physical_gpu_ids",
        "canonical_payload_sha256",
    }
    if set(run) != expected_keys:
        raise WriterModelError(
            "portable endpoint generation run fields changed"
        )
    return generation, generation_root, run_path, run


def _validate_generation_run_header(
    run_path: Path,
    run: Mapping[str, Any],
    generation: Mapping[str, Any],
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    adapter = contract["adapter"]
    training = run.get("training_run", {})
    valid = (
        sha256_file(run_path) == generation.get("run_contract_file_sha256")
        and run.get("canonical_payload_sha256")
        == generation.get("run_contract_payload_sha256")
        and run.get("git", {}).get("extension_commit")
        == generation.get("extension_git_commit")
        and run.get("git", {}).get("constructor_commit")
        == generation.get("constructor_git_commit")
        and run.get("generation_descriptor_sha256")
        == generation.get("generation_descriptor_sha256")
        and run.get("mode") == "formal"
        and isinstance(run.get("command"), list)
        and "--cache-only-public-lora" in run.get("command", [])
        and int(run.get("world_size", -1)) == 4
        and run.get("physical_gpu_ids") == [4, 5, 6, 7]
        and Path(str(training.get("path", ""))).resolve()
        == Path(str(adapter["training_run"]["path"])).resolve()
        and training.get("contract_file_sha256")
        == adapter["training_run"]["run_contract_file_sha256"]
        and training.get("contract_payload_sha256")
        == adapter["training_run"]["run_contract_sha256"]
        and training.get("git_commit")
        == adapter["training_run"]["git_commit"]
        and manifest.get("training_config") == training.get("config")
        and run.get("source") == contract.get("model")
        and run.get("tokenizer") == contract.get("tokenizer")
        and run.get("information_wall") == PORTABLE_INFORMATION_WALL
        and manifest.get("information_wall") == PORTABLE_INFORMATION_WALL
    )
    if not valid:
        raise WriterModelError(
            "portable endpoint generation run authority changed"
        )


def _validate_generation_descriptor(
    run: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    config_payload = _validate_generation_git_and_config(run, contract)
    descriptor = _expected_generation_descriptor(config_payload)
    if (
        run.get("generation_descriptor") != descriptor
        or canonical_hash(descriptor)
        != run.get("generation_descriptor_sha256")
    ):
        raise WriterModelError(
            "portable endpoint generation descriptor changed"
        )


def _validate_generation_panel(
    generation_root: Path,
    run: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    panel = run.get("panel", {})
    panel_path = (generation_root / str(panel.get("manifest_path", ""))).resolve()
    if (
        panel_path != generation_root / "panel_manifest.json"
        or not panel_path.is_file()
        or sha256_file(panel_path) != panel.get("manifest_file_sha256")
    ):
        raise WriterModelError(
            "portable endpoint generation panel file changed"
        )
    panel_manifest = _validated_payload(panel_path, PANEL_MANIFEST_SCHEMA)
    panel_config_path = (
        REPO_ROOT / "configs/pi05_validation_functional_loss_panel_v1.json"
    )
    valid = (
        panel.get("schema_version") == PANEL_SCHEMA
        and panel_config_path.is_file()
        and panel.get("config_file_sha256") == sha256_file(panel_config_path)
        and panel.get("manifest_payload_sha256")
        == SEALED_PANEL_PAYLOAD_SHA256
        and panel_manifest.get("canonical_payload_sha256")
        == SEALED_PANEL_PAYLOAD_SHA256
        and int(panel.get("row_count", -1)) == 512
        and int(panel.get("unique_teacher_video_conditions", -1)) == 64
        and manifest.get("panel_manifest_payload_sha256")
        == SEALED_PANEL_PAYLOAD_SHA256
    )
    if not valid:
        raise WriterModelError(
            "portable endpoint generation panel authority changed"
        )


def _validate_generation_lora(
    run: Mapping[str, Any],
    manifest: Mapping[str, Any],
    lora: Any,
) -> None:
    lora_record = run.get("lora", {})
    authority_path = REPO_ROOT / "configs/pi05_lora_v1.json"
    contract_sha256 = canonical_contract_sha256(lora)
    valid = (
        lora_record.get("canonical_contract_sha256") == contract_sha256
        and manifest.get("lora_contract_sha256") == contract_sha256
        and int(lora_record.get("target_count", -1)) == 38
        and int(lora_record.get("rank", -1)) == 16
        and int(lora_record.get("tensor_count", -1)) == 76
        and authority_path.is_file()
        and lora_record.get("authority_file_sha256")
        == sha256_file(authority_path)
    )
    if not valid:
        raise WriterModelError(
            "portable endpoint generation LoRA authority changed"
        )


def _valid_hdf5_episode(
    episode: Any,
    demo_index: int,
) -> bool:
    if not isinstance(episode, Mapping):
        return False
    return (
        set(episode)
        == {
            "demo_index",
            "action_rows_from_shape_metadata",
            "video_rows_from_shape_metadata",
            "action_width",
            "video_channels",
            "video_dtype",
        }
        and int(episode.get("demo_index", -1)) == demo_index
        and int(episode.get("action_rows_from_shape_metadata", -1)) > 0
        and int(episode.get("video_rows_from_shape_metadata", -1))
        == int(episode.get("action_rows_from_shape_metadata", -2))
        and int(episode.get("action_width", -1)) == 7
        and int(episode.get("video_channels", -1)) == 3
        and episode.get("video_dtype") == "uint8"
    )


def _valid_hdf5_record(
    record: Any,
    expected_hdf5: Mapping[int, Mapping[str, Any]],
) -> bool:
    if not isinstance(record, Mapping):
        return False
    task_id = int(record.get("global_task_id", -1))
    authority = expected_hdf5.get(task_id, {})
    episodes = record.get("episodes", [])
    return (
        set(record)
        == {
            "global_task_id",
            "bytes",
            "sealed_expected_sha256",
            "episodes",
        }
        and int(record.get("bytes", -1)) > 0
        and int(record.get("bytes", -1)) == int(authority.get("bytes", -2))
        and re.fullmatch(
            r"[0-9a-f]{64}",
            str(record.get("sealed_expected_sha256", "")),
        )
        is not None
        and record.get("sealed_expected_sha256") == authority.get("sha256")
        and isinstance(episodes, list)
        and len(episodes) == 50
        and all(
            _valid_hdf5_episode(episode, demo_index)
            for demo_index, episode in enumerate(episodes)
        )
    )


def _validate_generation_hdf5(
    hdf5: Mapping[str, Any],
    expected_hdf5: Mapping[int, Mapping[str, Any]],
    task_ids: set[int],
    manifest_sha256: str,
) -> None:
    files = hdf5.get("files", [])
    observed_ids = {
        int(record.get("global_task_id", -1))
        for record in files
        if isinstance(record, Mapping)
    }
    valid = (
        set(hdf5)
        == {
            "task_count",
            "sealed_manifest_file_sha256",
            "exact_file_size_verified",
            "hdf5_schema_metadata_verified",
            "runtime_full_sha256_computed",
            "validation_action_values_read",
            "validation_video_values_read_during_schema_check",
            "identity_sha256",
            "files",
        }
        and int(hdf5.get("task_count", -1)) == 8
        and hdf5.get("sealed_manifest_file_sha256") == manifest_sha256
        and hdf5.get("exact_file_size_verified") is True
        and hdf5.get("hdf5_schema_metadata_verified") is True
        and hdf5.get("runtime_full_sha256_computed") is False
        and int(hdf5.get("validation_action_values_read", -1)) == 0
        and int(
            hdf5.get("validation_video_values_read_during_schema_check", -1)
        )
        == 0
        and isinstance(files, list)
        and len(files) == 8
        and len(observed_ids) == 8
        and observed_ids == task_ids
        and all(
            _valid_hdf5_record(record, expected_hdf5)
            for record in files
        )
        and canonical_hash(files) == hdf5.get("identity_sha256")
    )
    if not valid:
        raise WriterModelError(
            "portable endpoint generation data authority changed"
        )


def _validate_generation_data(
    run: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    data = run.get("data", {})
    video_data = contract["adapter"].get("video_data", {})
    target_manifest = data.get("target_data_manifest", {})
    relative = Path(str(target_manifest.get("path", "")))
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.parts[0] != "configs"
    ):
        raise WriterModelError(
            "portable endpoint generation target manifest changed"
        )
    target_manifest_path = REPO_ROOT / relative
    manifest_sha256 = str(target_manifest.get("sha256", ""))
    if (
        not target_manifest_path.is_file()
        or manifest_sha256
        != video_data.get("target_data_manifest_file_sha256")
        or sha256_file(target_manifest_path) != manifest_sha256
        or Path(str(data.get("root", ""))).resolve()
        != Path(str(video_data.get("root", ""))).resolve()
        or data.get("candidate_video_data") != video_data
    ):
        raise WriterModelError(
            "portable endpoint generation target manifest changed"
        )
    target_payload = read_json(target_manifest_path)
    expected_hdf5 = {
        int(record["global_task_id"]): record["hdf5"]
        for record in target_payload.get("tasks", [])
    }
    _validate_generation_hdf5(
        data.get("validation_hdf5", {}),
        expected_hdf5,
        set(map(int, video_data.get("task_ids", []))),
        manifest_sha256,
    )


def _validate_generation_candidate(
    run: Mapping[str, Any],
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    candidate = manifest.get("candidate", {})
    checkpoints = run.get("checkpoints", [])
    outcomes = run.get("outcome_evaluations", [])
    checkpoint = candidate.get("checkpoint")
    matching = [
        record
        for record in outcomes
        if record.get("candidate_id") == candidate.get("candidate_id")
    ]
    video_data = contract["adapter"].get("video_data", {})
    valid = (
        isinstance(checkpoints, list)
        and len(checkpoints)
        == len({int(record.get("cursor", -1)) for record in checkpoints})
        and checkpoint in checkpoints
        and len(matching) == 1
        and matching[0]
        == {
            **dict(candidate),
            "source": dict(contract["model"]),
            "tokenizer": dict(contract["tokenizer"]),
            "video_data": dict(video_data),
        }
    )
    if not valid:
        raise WriterModelError(
            "portable endpoint generation candidate authority changed"
        )


def _validate_portable_generation_run(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    evaluation_root: Path,
    contract: Mapping[str, Any],
    lora: Any,
) -> Mapping[str, Any]:
    if (
        Path(str(manifest.get("candidate", {}).get("evaluation_root", "")))
        .resolve()
        != evaluation_root
    ):
        raise WriterModelError(
            "portable endpoint generation candidate authority changed"
        )
    generation, generation_root, run_path, run = (
        _resolve_portable_generation_run(manifest_path, manifest)
    )
    _validate_generation_run_header(
        run_path,
        run,
        generation,
        manifest,
        contract,
    )
    _validate_generation_descriptor(run, contract)
    _validate_generation_panel(generation_root, run, manifest)
    _validate_generation_lora(run, manifest, lora)
    _validate_generation_data(run, contract)
    _validate_generation_candidate(run, manifest, contract)
    return run


def _portable_candidate_checkpoint(
    manifest: Mapping[str, Any],
    evaluation_root: Path,
    contract: Mapping[str, Any],
    lora: Any,
    family: str,
    correct400: int,
    task_breadth: int,
) -> Mapping[str, Any]:
    candidate = manifest.get("candidate", {})
    adapter_checkpoint = contract["adapter"]["checkpoint"]
    checkpoint = {
        key: adapter_checkpoint[key]
        for key in (
            "path",
            "cursor",
            "manifest_file_sha256",
            "manifest_payload_sha256",
            "writer_state_sha256",
        )
    }
    candidate_id = str(candidate.get("candidate_id", ""))
    expected = {
        "family": family,
        "candidate_id": candidate_id,
        "evaluation_root": str(evaluation_root),
        "run_contract_file_sha256": sha256_file(
            evaluation_root / "run_contract.json"
        ),
        "run_contract_sha256": contract["contract_sha256"],
        "results_file_sha256": sha256_file(evaluation_root / "results.json"),
        "adapter_sha256": canonical_hash(dict(contract["adapter"])),
        "writer_constructor_git_commit": contract["adapter"]["training_run"]
        ["git_commit"],
        "correct400": correct400,
        "task_breadth": task_breadth,
        "checkpoint": dict(checkpoint),
    }
    valid = (
        candidate == expected
        and re.fullmatch(r"[a-z0-9_]+", candidate_id) is not None
        and Path(str(candidate.get("evaluation_root", ""))).resolve()
        == evaluation_root
        and manifest.get("information_wall") == PORTABLE_INFORMATION_WALL
        and manifest.get("panel_manifest_payload_sha256")
        == SEALED_PANEL_PAYLOAD_SHA256
        and manifest.get("lora_contract_sha256")
        == canonical_contract_sha256(lora)
    )
    if not valid:
        raise WriterModelError("portable endpoint cache authority changed")
    return checkpoint


def _portable_lora_entry(
    row: Any,
    manifest_root: Path,
    panel_conditions: Mapping[tuple[int, int], int],
    checkpoint: Mapping[str, Any],
    seen: set[tuple[int, int]],
) -> tuple[tuple[int, int], EndpointLoRAEntry]:
    if not isinstance(row, Mapping):
        raise WriterModelError("portable endpoint cache entry changed")
    key = (
        int(row.get("global_task_id", -1)),
        int(row.get("teacher_demo_index", -1)),
    )
    evidence = row.get("generation_evidence", {})
    file_record = row.get("lora_file", {})
    if not isinstance(evidence, Mapping) or not isinstance(file_record, Mapping):
        raise WriterModelError("portable endpoint cache entry changed")
    path = (manifest_root / str(file_record.get("path", ""))).resolve()
    try:
        wall_seconds = float(evidence.get("generation_wall_seconds"))
    except (TypeError, ValueError):
        wall_seconds = float("nan")
    expected_evidence = {
        "language_global_task_id": key[0],
        "video_global_task_id": key[0],
        "video_group": panel_conditions.get(key),
        "teacher_demo_index": key[1],
        "condition": "correct",
        "video_order": "forward",
        "frame_stride": 5,
        "writer_checkpoint_cursor": int(checkpoint["cursor"]),
        "writer_checkpoint_manifest_sha256": checkpoint[
            "manifest_file_sha256"
        ],
        "writer_state_sha256": checkpoint["writer_state_sha256"],
        "lora_sha256": row.get("lora_state_sha256"),
        "lora_tensor_count": 76,
    }
    observed_evidence = {
        field: evidence.get(field) for field in expected_evidence
    }
    valid = (
        key in panel_conditions
        and key not in seen
        and set(row)
        == {
            "global_task_id",
            "teacher_demo_index",
            "lora_file",
            "lora_state_sha256",
            "generation_evidence",
        }
        and set(file_record) == {"path", "bytes", "sha256"}
        and set(evidence)
        == {*expected_evidence, "rank", "generation_wall_seconds"}
        and path.is_relative_to(manifest_root.resolve())
        and str(file_record.get("path", ""))
        == f"loras/task_{key[0]:03d}_demo_{key[1]:03d}.safetensors"
        and int(file_record.get("bytes", -1)) > 0
        and re.fullmatch(
            r"[0-9a-f]{64}", str(file_record.get("sha256", ""))
        )
        is not None
        and re.fullmatch(
            r"[0-9a-f]{64}", str(row.get("lora_state_sha256", ""))
        )
        is not None
        and observed_evidence == expected_evidence
        and int(evidence.get("rank", -1)) in range(4)
        and wall_seconds >= 0
        and wall_seconds < float("inf")
    )
    if not valid:
        raise WriterModelError("portable endpoint cache entry changed")
    return (
        key,
        EndpointLoRAEntry(
            path=path,
            bytes=int(file_record["bytes"]),
            file_sha256=str(file_record["sha256"]),
            state_sha256=str(row["lora_state_sha256"]),
        ),
    )


def _portable_cache_entries(
    manifest_path: Path,
    evaluation_root: Path,
    contract: Mapping[str, Any],
    panel_conditions: Mapping[tuple[int, int], int],
    lora: Any,
    family: str,
    correct400: int,
    task_breadth: int,
) -> tuple[dict[tuple[int, int], EndpointLoRAEntry], str]:
    """Load 64 public LoRAs emitted by the checkpoint's historical code."""

    manifest = _validated_payload(manifest_path, PORTABLE_CACHE_SCHEMA)
    checkpoint = _portable_candidate_checkpoint(
        manifest,
        evaluation_root,
        contract,
        lora,
        family,
        correct400,
        task_breadth,
    )
    _validate_portable_generation_run(
        manifest_path,
        manifest,
        evaluation_root,
        contract,
        lora,
    )
    selected = {}
    for row in manifest.get("entries", []):
        key, entry = _portable_lora_entry(
            row,
            manifest_path.parent,
            panel_conditions,
            checkpoint,
            set(selected),
        )
        selected[key] = entry
    if set(selected) != set(panel_conditions):
        raise WriterModelError(
            "portable endpoint cache does not cover the sealed panel"
        )
    return selected, sha256_file(manifest_path)
