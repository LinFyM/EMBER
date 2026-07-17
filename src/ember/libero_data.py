"""Leakage-safe HDF5 and normalization audit primitives for LIBERO-90."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


class ManifestError(RuntimeError):
    """Raised when a pinned LIBERO authority or demonstration invariant fails."""


@dataclass
class AuditResult:
    record: dict[str, Any]
    state_samples: np.ndarray | None
    action_samples: np.ndarray | None


REQUIRED_DATASETS: dict[str, tuple[str, tuple[int, ...] | None]] = {
    "actions": ("float64", (7,)),
    "dones": ("uint8", ()),
    "obs/agentview_rgb": ("uint8", (128, 128, 3)),
    "obs/ee_ori": ("float64", (3,)),
    "obs/ee_pos": ("float64", (3,)),
    "obs/ee_states": ("float64", (6,)),
    "obs/eye_in_hand_rgb": ("uint8", (128, 128, 3)),
    "obs/gripper_states": ("float64", (2,)),
    "obs/joint_states": ("float64", (7,)),
    "rewards": ("uint8", ()),
    "robot_states": ("float64", (9,)),
    "states": ("float64", None),
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_attr(group: h5py.Group, name: str) -> dict[str, Any]:
    try:
        value = group.attrs[name]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        decoded = json.loads(value)
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as error:
        raise ManifestError(f"invalid HDF5 JSON attribute {name}") from error
    if not isinstance(decoded, dict):
        raise ManifestError(f"HDF5 JSON attribute {name} must be an object")
    return decoded


def load_hub_surface(
    tree_metadata_path: Path,
    *,
    subdir: str,
    expected_file_count: int,
    expected_total_bytes: int,
) -> dict[str, Any]:
    """Load immutable LFS identities from a Hugging Face local tree manifest."""

    try:
        tree = json.loads(tree_metadata_path.read_text(encoding="utf-8"))
        raw_files = tree["files"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ManifestError(f"invalid Hub tree metadata: {tree_metadata_path}") from error
    prefix = f"{subdir.rstrip('/')}/"
    selected: dict[str, dict[str, Any]] = {}
    for relative_path, metadata in raw_files.items():
        if not relative_path.startswith(prefix) or not relative_path.endswith(".hdf5"):
            continue
        basename = relative_path.removeprefix(prefix)
        if "/" in basename:
            continue
        sha256 = metadata.get("lfs_sha256")
        size = metadata.get("lfs_size", metadata.get("size"))
        if not isinstance(sha256, str) or SHA256_RE.fullmatch(sha256) is None:
            raise ManifestError(f"Hub entry lacks an immutable LFS SHA256: {relative_path}")
        if not isinstance(size, int) or size <= 0:
            raise ManifestError(f"Hub entry has invalid LFS bytes: {relative_path}")
        selected[basename] = {"sha256": sha256, "bytes": size}
    total_bytes = sum(entry["bytes"] for entry in selected.values())
    if len(selected) != expected_file_count:
        raise ManifestError(
            f"Hub surface has {len(selected)} files, expected {expected_file_count}"
        )
    if total_bytes != expected_total_bytes:
        raise ManifestError(
            f"Hub surface total bytes are {total_bytes}, expected {expected_total_bytes}"
        )
    return {
        "file_count": len(selected),
        "total_bytes": total_bytes,
        "files": dict(sorted(selected.items())),
    }


def _validate_dataset(
    demo: h5py.Group, path: str, steps: int, expected_dtype: str, tail: tuple[int, ...] | None
) -> None:
    if path not in demo:
        raise ManifestError(f"missing demonstration dataset: {path}")
    dataset = demo[path]
    if not isinstance(dataset, h5py.Dataset) or not dataset.shape or dataset.shape[0] != steps:
        raise ManifestError(f"invalid leading dimension for demonstration dataset: {path}")
    if np.dtype(dataset.dtype) != np.dtype(expected_dtype):
        raise ManifestError(f"invalid dtype for demonstration dataset: {path}")
    if tail is not None and dataset.shape[1:] != tail:
        raise ManifestError(f"invalid shape for demonstration dataset: {path}")
    if tail is None and dataset.ndim != 2:
        raise ManifestError(f"invalid rank for demonstration dataset: {path}")


def _audit_demo_groups(
    data: h5py.Group,
    *,
    task_index: int,
    expected_demos: int,
    normalization_episodes: tuple[int, ...],
) -> tuple[list[int], np.ndarray | None, np.ndarray | None]:
    expected_names = {f"demo_{index}" for index in range(expected_demos)}
    if set(data.keys()) != expected_names:
        raise ManifestError(f"non-contiguous demonstration IDs: task {task_index}")
    lengths: list[int] = []
    state_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    for episode in range(expected_demos):
        demo = data[f"demo_{episode}"]
        if "actions" not in demo:
            raise ManifestError(f"missing actions dataset: task {task_index}, demo {episode}")
        steps = int(demo["actions"].shape[0])
        if int(demo.attrs.get("num_samples", -1)) != steps:
            raise ManifestError(f"num_samples mismatch: task {task_index}, demo {episode}")
        if "init_state" not in demo.attrs:
            raise ManifestError(f"missing demo init state: task {task_index}, demo {episode}")
        for dataset_path, (dtype, tail) in REQUIRED_DATASETS.items():
            _validate_dataset(demo, dataset_path, steps, dtype, tail)
        lengths.append(steps)
        if episode in normalization_episodes:
            ee_state = np.asarray(demo["obs/ee_states"], dtype=np.float64)
            gripper = np.asarray(demo["obs/gripper_states"], dtype=np.float64)
            state_chunks.append(np.concatenate((ee_state, gripper), axis=1))
            action_chunks.append(np.asarray(demo["actions"], dtype=np.float64))
    states = np.concatenate(state_chunks) if state_chunks else None
    actions = np.concatenate(action_chunks) if action_chunks else None
    return lengths, states, actions


def _environment_metadata(
    data: h5py.Group, *, task_index: int, bddl_basename: str, language: str
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if Path(str(data.attrs.get("bddl_file_name", ""))).name != bddl_basename:
        raise ManifestError(f"HDF5 BDDL basename mismatch: task {task_index}")
    if _json_attr(data, "problem_info").get("language_instruction") != language:
        raise ManifestError(f"HDF5 language mismatch: task {task_index}")
    env_args = _json_attr(data, "env_args")
    env_kwargs = env_args.get("env_kwargs")
    if not isinstance(env_kwargs, dict):
        raise ManifestError(f"missing HDF5 environment kwargs: task {task_index}")
    env_bddl = str(env_args.get("bddl_file", env_kwargs.get("bddl_file_name", "")))
    warnings = []
    if Path(env_bddl).name != bddl_basename:
        warnings.append(
            {
                "code": "legacy_env_bddl_basename_mismatch",
                "message": (
                    "producer env_args uses a legacy BDDL basename while the canonical "
                    "HDF5 authority matches"
                ),
            }
        )
    legacy_suites = re.findall(r"libero_\d+", env_bddl)
    legacy_suite = legacy_suites[-1] if legacy_suites else "unspecified"
    if legacy_suite != "libero_90":
        warnings.append(
            {
                "code": "legacy_env_bddl_suite",
                "message": "producer env_args names a legacy suite while the canonical BDDL basename matches",
            }
        )
    controller = env_kwargs.get("controller_configs")
    cameras = env_kwargs.get("camera_names")
    if not isinstance(controller, dict):
        raise ManifestError(f"missing controller authority: task {task_index}")
    if not isinstance(cameras, list) or not all(isinstance(value, str) for value in cameras):
        raise ManifestError(f"missing camera authority: task {task_index}")
    return {
        "camera": {
            "names": cameras,
            "height": env_kwargs.get("camera_heights"),
            "width": env_kwargs.get("camera_widths"),
            "depth": env_kwargs.get("camera_depths"),
            "enabled": env_kwargs.get("use_camera_obs"),
            "image_convention": data.attrs.get("macros_image_convention"),
        },
        "controller": {
            "type": controller.get("type"),
            "control_delta": controller.get("control_delta"),
            "control_frequency_hz": env_kwargs.get("control_freq"),
            "output_min": controller.get("output_min"),
            "output_max": controller.get("output_max"),
        },
        "robot": env_kwargs.get("robots"),
        "legacy_producer_suite": legacy_suite,
    }, warnings


def audit_demonstration_file(
    path: Path,
    *,
    task_index: int,
    task_name: str,
    split: str,
    language: str,
    bddl_basename: str,
    expected_tag: str,
    expected_demos: int,
    expected_sha256: str,
    expected_bytes: int,
    normalization_episodes: tuple[int, ...],
) -> AuditResult:
    """Audit one task while reading values only for source normalization episodes."""

    if normalization_episodes and split != "source":
        raise ManifestError("normalization values may only be read from source tasks")
    if path.name != f"{task_name}_demo.hdf5":
        raise ManifestError(f"demonstration filename does not match task {task_index}")
    if path.stat().st_size != expected_bytes:
        raise ManifestError(f"demonstration bytes differ from Hub authority: task {task_index}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ManifestError(f"demonstration SHA256 differs from Hub authority: task {task_index}")
    with h5py.File(path, "r") as handle:
        if "data" not in handle or not isinstance(handle["data"], h5py.Group):
            raise ManifestError(f"missing HDF5 data group: task {task_index}")
        data = handle["data"]
        if data.attrs.get("tag") != expected_tag:
            raise ManifestError(f"wrong HDF5 tag: task {task_index}")
        if int(data.attrs.get("num_demos", -1)) != expected_demos:
            raise ManifestError(f"wrong demonstration count: task {task_index}")
        environment, warnings = _environment_metadata(
            data, task_index=task_index, bddl_basename=bddl_basename, language=language
        )
        lengths, state_samples, action_samples = _audit_demo_groups(
            data,
            task_index=task_index,
            expected_demos=expected_demos,
            normalization_episodes=normalization_episodes,
        )
        total_steps = sum(lengths)
        if int(data.attrs.get("total", -1)) != total_steps:
            raise ManifestError(f"HDF5 total frame count mismatch: task {task_index}")
        record = {
            "task_index": task_index,
            "task_name": task_name,
            "split": split,
            "language": language,
            "hdf5": {"filename": path.name, "bytes": expected_bytes, "sha256": actual_sha256},
            "demonstrations": {
                "count": expected_demos,
                "steps": total_steps,
                "min_steps": min(lengths),
                "max_steps": max(lengths),
                "episode_lengths": lengths,
            },
            **environment,
            "access_policy": (
                "source_normalization_values" if normalization_episodes else "metadata_only"
            ),
            "normalization_episode_indices": list(normalization_episodes),
            "quality": {
                "status": "pass_with_note" if warnings else "pass",
                "warning_count": len(warnings),
                "warnings": warnings,
            },
        }
    return AuditResult(record, state_samples, action_samples)


def _stats(samples: np.ndarray) -> dict[str, Any]:
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ManifestError("normalization requires at least two vector samples")
    quantiles = np.quantile(samples, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return {
        "count": int(samples.shape[0]),
        "mean": samples.mean(axis=0).tolist(),
        "std": samples.std(axis=0).tolist(),
        "min": samples.min(axis=0).tolist(),
        "max": samples.max(axis=0).tolist(),
        "q01": quantiles[0].tolist(),
        "q10": quantiles[1].tolist(),
        "q50": quantiles[2].tolist(),
        "q90": quantiles[3].tolist(),
        "q99": quantiles[4].tolist(),
    }


def compute_normalization(
    results: list[AuditResult], *, source_task_indices: list[int], episode_bounds: list[int]
) -> dict[str, Any]:
    selected = [result for result in results if result.state_samples is not None]
    observed_indices = sorted(result.record["task_index"] for result in selected)
    if observed_indices != sorted(source_task_indices):
        raise ManifestError("normalization task set differs from the declared source split")
    if len(episode_bounds) != 2 or episode_bounds[0] > episode_bounds[1]:
        raise ManifestError("invalid source_base_fit episode bounds")
    expected_episodes = list(range(episode_bounds[0], episode_bounds[1] + 1))
    for result in selected:
        if result.record["split"] != "source":
            raise ManifestError("normalization includes a non-source task")
        if result.record["normalization_episode_indices"] != expected_episodes:
            raise ManifestError("normalization includes the wrong source episode pool")
    states = np.concatenate([result.state_samples for result in selected])
    actions = np.concatenate([result.action_samples for result in selected])
    return {
        "schema_version": 1,
        "authority": {
            "split": "source",
            "task_indices": sorted(source_task_indices),
            "episode_pool": "source_base_fit",
            "episode_bounds_inclusive": episode_bounds,
            "forbidden_surfaces": ["validation", "held_out"],
        },
        "feature_definitions": {
            "observation.state": [
                "ee_x", "ee_y", "ee_z", "ee_axis_angle_x", "ee_axis_angle_y",
                "ee_axis_angle_z", "gripper_left", "gripper_right",
            ],
            "action": ["dx", "dy", "dz", "dax", "day", "daz", "gripper"],
        },
        "observation.state": _stats(states),
        "action": _stats(actions),
    }
