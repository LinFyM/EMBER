"""Seal target-40 LIBERO demonstrations without decoding held numeric values."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.libero_data import ManifestError, audit_demonstration_file
from ember.pi05_assets import load_protocol, write_json_atomic
from ember.pi05_source_checkpoint import canonical_hash, sha256_file


DATASET_REPOSITORY = "yifengzhu-hf/LIBERO-datasets"
DATASET_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
SUITE_ORDER = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TARGET_DATA_SCHEMA = "ember_pi05_target_data_manifest_v1"
PROTOCOL_SHA256 = "db6d575758b69380803a7032b8cdb2f120d1348b63783da0bb1aa1f84031068f"
OVERLAP_AUDIT_SHA256 = "fe7311272fc7b854a84f799692f96fbb069e74a88ba5008757fa5ec7fb7cc003"


class Pi05TargetDataError(RuntimeError):
    """Raised when target task or demonstration authority differs."""


@dataclass(frozen=True)
class HubFileAuthority:
    relative_path: str
    bytes: int
    sha256: str


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Pi05TargetDataError(f"invalid target-data JSON: {path}") from error
    if not isinstance(value, dict):
        raise Pi05TargetDataError(f"target-data authority is not an object: {path}")
    return value


def _split_role(protocol: Mapping[str, Any], suite: str, task_id: int) -> str:
    roles = protocol["split"]["suites"][suite]
    matches = [role for role in ("train", "validation", "test") if task_id in roles[role]]
    if len(matches) != 1:
        raise Pi05TargetDataError(f"ambiguous target split role: {suite}/{task_id}")
    return matches[0]


def target_global_task_id(suite: str, task_id: int) -> int:
    if suite not in SUITE_ORDER or not 0 <= task_id < 10:
        raise Pi05TargetDataError(f"invalid target task key: {suite}/{task_id}")
    return SUITE_ORDER.index(suite) * 10 + task_id


def target_hdf5_relative_path(task: Mapping[str, Any]) -> str:
    suite = str(task["suite"])
    task_name = str(task["task_name"])
    if suite not in SUITE_ORDER or not task_name or "/" in task_name:
        raise Pi05TargetDataError("invalid target task filename authority")
    return f"{suite}/{task_name}_demo.hdf5"


def fetch_hub_authorities() -> dict[str, HubFileAuthority]:
    """Fetch only immutable Hub metadata; no demonstration file content."""

    from huggingface_hub import HfApi

    info = HfApi().dataset_info(
        DATASET_REPOSITORY,
        revision=DATASET_REVISION,
        files_metadata=True,
    )
    if info.sha != DATASET_REVISION:
        raise Pi05TargetDataError("target dataset revision moved")
    result: dict[str, HubFileAuthority] = {}
    for item in info.siblings:
        relative = str(item.rfilename)
        if not relative.endswith(".hdf5") or relative.split("/", 1)[0] not in SUITE_ORDER:
            continue
        if item.lfs is None or item.size is None:
            raise Pi05TargetDataError(f"target HDF5 lacks LFS metadata: {relative}")
        result[relative] = HubFileAuthority(
            relative_path=relative,
            bytes=int(item.size),
            sha256=str(item.lfs.sha256),
        )
    if len(result) != 40:
        raise Pi05TargetDataError(f"expected 40 target HDF5 files, found {len(result)}")
    return result


def build_target_rows(
    *,
    protocol: Mapping[str, Any],
    overlap_audit: Mapping[str, Any],
    hub_files: Mapping[str, HubFileAuthority],
) -> tuple[dict[str, Any], ...]:
    tasks = overlap_audit.get("target_tasks", [])
    if (
        overlap_audit.get("schema_version") != "ember_pi05_source_overlap_v1"
        or overlap_audit.get("summary", {}).get("target_tasks") != 40
    ):
        raise Pi05TargetDataError("overlap audit is not the sealed target-40 authority")
    by_key = {(str(row["suite"]), int(row["task_id"])): row for row in tasks}
    expected = {(suite, task_id) for suite in SUITE_ORDER for task_id in range(10)}
    if set(by_key) != expected or len(tasks) != 40:
        raise Pi05TargetDataError("overlap audit is not the target-40 authority")
    rows: list[dict[str, Any]] = []
    used_files: set[str] = set()
    for suite in SUITE_ORDER:
        for task_id in range(10):
            task = by_key[(suite, task_id)]
            if (
                not str(task.get("language", "")).strip()
                or int(task.get("bddl_bytes", 0)) <= 0
                or re.fullmatch(r"[0-9a-f]{64}", str(task.get("bddl_sha256", "")))
                is None
            ):
                raise Pi05TargetDataError(
                    f"invalid target specification authority: {suite}/{task_id}"
                )
            relative = target_hdf5_relative_path(task)
            hub = hub_files.get(relative)
            if hub is None:
                raise Pi05TargetDataError(f"Hub is missing target HDF5: {relative}")
            used_files.add(relative)
            rows.append(
                {
                    "global_task_id": target_global_task_id(suite, task_id),
                    "suite": suite,
                    "task_id": task_id,
                    "split_role": _split_role(protocol, suite, task_id),
                    "task_name": str(task["task_name"]),
                    "language": str(task["language"]),
                    "problem_folder": str(task["problem_folder"]),
                    "bddl": {
                        "filename": str(task["bddl_file"]),
                        "bytes": int(task["bddl_bytes"]),
                        "sha256": str(task["bddl_sha256"]),
                    },
                    "hdf5": {
                        "relative_path": relative,
                        "bytes": hub.bytes,
                        "sha256": hub.sha256,
                    },
                }
            )
    if used_files != set(hub_files):
        raise Pi05TargetDataError("Hub target HDF5 set contains an unexpected file")
    return tuple(rows)


def _audit_local_row(data_root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    hdf5 = row["hdf5"]
    try:
        result = audit_demonstration_file(
            data_root / str(hdf5["relative_path"]),
            task_index=int(row["global_task_id"]),
            task_name=str(row["task_name"]),
            split=str(row["split_role"]),
            language=str(row["language"]),
            bddl_basename=str(row["bddl"]["filename"]),
            expected_tag="libero-v1",
            expected_demos=50,
            expected_sha256=str(hdf5["sha256"]),
            expected_bytes=int(hdf5["bytes"]),
            normalization_episodes=(),
        )
    except (OSError, ManifestError) as error:
        raise Pi05TargetDataError(
            f"target HDF5 metadata audit failed: {row['suite']}/{row['task_id']}"
        ) from error
    if result.state_samples is not None or result.action_samples is not None:
        raise Pi05TargetDataError("target metadata audit decoded forbidden numeric values")
    audited = result.record
    return {
        **dict(row),
        "hdf5": {
            **dict(row["hdf5"]),
            "verified_local_sha256": audited["hdf5"]["sha256"],
        },
        "demonstrations": audited["demonstrations"],
        "camera": audited["camera"],
        "controller": audited["controller"],
        "robot": audited["robot"],
        "producer_notes": audited["quality"],
        "access_policy": {
            "sealed_at_this_stage": "metadata_and_opaque_file_hash_only",
            "decoded_action_values": 0,
            "decoded_state_values": 0,
            "decoded_reward_values": 0,
            "decoded_terminal_values": 0,
            "decoded_video_values": 0,
        },
    }


def seal_target_data(
    *,
    protocol_path: Path,
    overlap_audit_path: Path,
    data_root: Path,
    output_path: Path,
    sealed_utc: str,
) -> dict[str, Any]:
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise Pi05TargetDataError("target split protocol differs from the approved authority")
    if sha256_file(overlap_audit_path) != OVERLAP_AUDIT_SHA256:
        raise Pi05TargetDataError("source-overlap audit differs from the approved authority")
    protocol = load_protocol(protocol_path)
    overlap = _read_object(overlap_audit_path)
    remote = fetch_hub_authorities()
    initial = build_target_rows(protocol=protocol, overlap_audit=overlap, hub_files=remote)
    rows = tuple(_audit_local_row(data_root, row) for row in initial)
    roles = {
        role: [row["global_task_id"] for row in rows if row["split_role"] == role]
        for role in ("train", "validation", "test")
    }
    if {role: len(values) for role, values in roles.items()} != {
        "train": 24,
        "validation": 8,
        "test": 8,
    }:
        raise Pi05TargetDataError("target data roles differ from 24/8/8")
    aggregate = hashlib.sha256()
    for row in rows:
        hdf5 = row["hdf5"]
        aggregate.update(
            f"{hdf5['relative_path']}\0{hdf5['bytes']}\0{hdf5['sha256']}\n".encode()
        )
    manifest = {
        "schema_version": TARGET_DATA_SCHEMA,
        "sealed_utc": sealed_utc,
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
        },
        "authorities": {
            "protocol_sha256": sha256_file(protocol_path),
            "overlap_audit_sha256": sha256_file(overlap_audit_path),
        },
        "information_wall": {
            "seal_surface": "specification, Hub LFS metadata, HDF5 schema/shape metadata, opaque file SHA256",
            "decoded_trajectory_or_video_values": 0,
            "policy_outcomes_read": 0,
            "task_selection_changes": 0,
        },
        "summary": {
            "tasks": 40,
            "roles": roles,
            "episodes": sum(int(row["demonstrations"]["count"]) for row in rows),
            "frames": sum(int(row["demonstrations"]["steps"]) for row in rows),
            "hdf5_bytes": sum(int(row["hdf5"]["bytes"]) for row in rows),
            "hdf5_authority_sha256": aggregate.hexdigest(),
        },
        "tasks": list(rows),
    }
    manifest["canonical_payload_sha256"] = canonical_hash(manifest)
    write_json_atomic(output_path, manifest)
    return manifest
