"""Deterministic, resumable frozen-VLM features for full teaching videos."""

from __future__ import annotations

import math
import os
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from ember.source_base_checkpoint import (
    canonical_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.writer.data import WriterTaskAuthority


class FeatureCacheError(RuntimeError):
    """Raised when a cache input, artifact, or resume boundary is invalid."""


@dataclass(frozen=True)
class FeatureCacheTask:
    task_id: int
    language: str
    authority: WriterTaskAuthority
    expected_hdf5_sha256: str
    episode_lengths: tuple[int, ...]

    @property
    def frame_count(self) -> int:
        return sum(self.episode_lengths)


@dataclass(frozen=True)
class CachedWriterInput:
    language_features: torch.Tensor
    video_features: torch.Tensor
    episode_offsets: torch.Tensor
    demo_indices: torch.Tensor


class WriterFeatureStore:
    """Bounded task-level LRU over a completed frozen-feature cache."""

    def __init__(
        self,
        root: Path,
        *,
        task_ids: Sequence[int],
        expected_extraction_sha256: str,
        max_cached_tasks: int,
        expected_dim: int = 960,
    ) -> None:
        if (
            not task_ids
            or len(set(task_ids)) != len(task_ids)
            or max_cached_tasks <= 0
            or expected_dim <= 0
        ):
            raise FeatureCacheError("invalid Writer feature-store request")
        manifest = read_json(root / "cache_manifest.json")
        records = manifest.get("task_records", [])
        record_ids = tuple(sorted(int(record["task_id"]) for record in records))
        expected_ids = tuple(sorted(int(task_id) for task_id in task_ids))
        if (
            manifest.get("schema_version")
            != "ember_writer_feature_cache_manifest_v1"
            or manifest.get("extraction_sha256") != expected_extraction_sha256
            or record_ids != expected_ids
            or int(manifest.get("task_count", -1)) != len(expected_ids)
        ):
            raise FeatureCacheError("Writer feature-cache manifest changed")
        self.root = root
        self.task_ids = expected_ids
        self.extraction_sha256 = expected_extraction_sha256
        self.max_cached_tasks = max_cached_tasks
        self.expected_dim = expected_dim
        self._cached: OrderedDict[int, CachedWriterInput] = OrderedDict()
        self._verified: set[int] = set()

    def load(self, task_id: int) -> CachedWriterInput:
        if task_id not in self.task_ids:
            raise FeatureCacheError(f"task is outside the Writer cache: {task_id}")
        if task_id in self._cached:
            self._cached.move_to_end(task_id)
            return self._cached[task_id]
        if task_id not in self._verified:
            if not task_cache_is_complete(
                self.root,
                task_id,
                extraction_sha256=self.extraction_sha256,
            ):
                raise FeatureCacheError(f"task feature cache changed: {task_id}")
            self._verified.add(task_id)
        tensor_path, _ = task_cache_paths(self.root, task_id)
        cached = load_task_cache(tensor_path, expected_dim=self.expected_dim)
        self._cached[task_id] = cached
        while len(self._cached) > self.max_cached_tasks:
            self._cached.popitem(last=False)
        return cached

    @property
    def cached_task_ids(self) -> tuple[int, ...]:
        return tuple(self._cached)


def load_feature_cache_config(path: Path, repo_root: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "ember_writer_feature_cache_v1":
        raise FeatureCacheError("unsupported Writer feature-cache schema")
    protocol = config.get("protocol", {})
    manifest_path = repo_root / str(protocol.get("manifest", ""))
    if not manifest_path.is_file() or sha256_file(manifest_path) != protocol.get(
        "manifest_sha256"
    ):
        raise FeatureCacheError("sealed data manifest changed")
    required = {
        "required_split": "train",
        "task_count": 70,
        "demo_count_per_task": 50,
    }
    if any(protocol.get(key) != value for key, value in required.items()):
        raise FeatureCacheError("feature cache must cover the sealed 70x50 train pool")
    features = config.get("features", {})
    feature_required = {
        "camera_dataset": "obs/agentview_rgb",
        "camera_transform": "libero_opengl_rotate_180_chw_unit_float_v1",
        "vision_token_count": 64,
        "vision_feature_dim": 960,
        "vision_normalization": "multiply_sqrt_feature_dim",
        "vision_pooling": "mean_over_spatial_tokens_per_frame",
        "language_feature_dim": 960,
        "language_max_tokens": 48,
        "language_suffix": "newline",
        "language_padding": "max_length_right",
        "stored_dtype": "bfloat16",
        "preserve_episode_order_and_boundaries": True,
    }
    if any(features.get(key) != value for key, value in feature_required.items()):
        raise FeatureCacheError("frozen feature extraction contract changed")
    if int(features.get("frame_batch_size_per_rank", 0)) <= 0:
        raise FeatureCacheError("frame batch size must be positive")
    parallel = config.get("parallel", {})
    if (
        parallel.get("world_size") != 8
        or parallel.get("policy_processes_per_gpu") != 1
        or parallel.get("assignment")
        != "deterministic_lpt_by_manifest_frame_count"
    ):
        raise FeatureCacheError("feature cache requires eight symmetric CUDA ranks")
    return config


def load_train_tasks(
    config: Mapping[str, Any], repo_root: Path, data_root: Path
) -> tuple[FeatureCacheTask, ...]:
    manifest = read_json(repo_root / str(config["protocol"]["manifest"]))
    if (
        manifest.get("protocol_references", {}).get("split_sha256")
        != config["protocol"]["split_sha256"]
    ):
        raise FeatureCacheError("manifest and sealed split disagree")
    tasks: list[FeatureCacheTask] = []
    for record in manifest.get("tasks", []):
        if record.get("split") != config["protocol"]["required_split"]:
            continue
        task_id = int(record["task_index"])
        hdf5 = record["hdf5"]
        lengths = tuple(int(value) for value in record["demonstrations"]["episode_lengths"])
        if (
            len(lengths) != int(config["protocol"]["demo_count_per_task"])
            or any(length <= 0 for length in lengths)
            or sum(lengths) != int(record["demonstrations"]["steps"])
        ):
            raise FeatureCacheError(f"invalid episode authority for task {task_id}")
        authority = WriterTaskAuthority(
            task_id=task_id,
            language=str(record["language"]),
            path=data_root / str(hdf5["filename"]),
            expected_bytes=int(hdf5["bytes"]),
            expected_sha256=None,
        )
        tasks.append(
            FeatureCacheTask(
                task_id=task_id,
                language=authority.language,
                authority=authority,
                expected_hdf5_sha256=str(hdf5["sha256"]),
                episode_lengths=lengths,
            )
        )
    tasks.sort(key=lambda task: task.task_id)
    expected_count = int(config["protocol"]["task_count"])
    if len(tasks) != expected_count or len({task.task_id for task in tasks}) != expected_count:
        raise FeatureCacheError("feature cache requires exactly 70 unique train tasks")
    return tuple(tasks)


def balanced_task_assignments(
    tasks: Sequence[FeatureCacheTask], world_size: int
) -> tuple[tuple[FeatureCacheTask, ...], ...]:
    """Deterministic LPT schedule using only sealed manifest frame counts."""

    if not tasks or world_size <= 0 or len({task.task_id for task in tasks}) != len(tasks):
        raise FeatureCacheError("invalid feature-cache task schedule")
    loads = [0] * world_size
    assignments: list[list[FeatureCacheTask]] = [[] for _ in range(world_size)]
    for task in sorted(tasks, key=lambda item: (-item.frame_count, item.task_id)):
        rank = min(range(world_size), key=lambda index: (loads[index], index))
        assignments[rank].append(task)
        loads[rank] += task.frame_count
    return tuple(
        tuple(sorted(rank_tasks, key=lambda item: item.task_id))
        for rank_tasks in assignments
    )


def pool_visual_tokens(
    embeddings: torch.Tensor, *, expected_tokens: int, expected_dim: int
) -> torch.Tensor:
    if (
        embeddings.ndim != 3
        or embeddings.shape[1] != expected_tokens
        or embeddings.shape[2] != expected_dim
    ):
        raise FeatureCacheError(
            f"unexpected SmolVLA image embedding shape: {tuple(embeddings.shape)}"
        )
    return (embeddings * math.sqrt(expected_dim)).mean(dim=1)


def select_language_tokens(
    embeddings: torch.Tensor, attention_mask: torch.Tensor, *, expected_dim: int
) -> torch.Tensor:
    if (
        embeddings.ndim != 3
        or embeddings.shape[0] != 1
        or embeddings.shape[2] != expected_dim
        or attention_mask.shape != embeddings.shape[:2]
    ):
        raise FeatureCacheError("unexpected SmolVLA language embedding shape")
    mask = attention_mask[0].to(dtype=torch.bool, device=embeddings.device)
    if not bool(mask.any()):
        raise FeatureCacheError("language tokenization produced no valid tokens")
    return embeddings[0, mask] * math.sqrt(expected_dim)


def _validate_cached_tensors(
    tensors: Mapping[str, torch.Tensor], *, expected_dim: int
) -> CachedWriterInput:
    required = {"language_features", "video_features", "episode_offsets", "demo_indices"}
    if set(tensors) != required:
        raise FeatureCacheError("task cache contains the wrong tensor set")
    language = tensors["language_features"]
    video = tensors["video_features"]
    offsets = tensors["episode_offsets"]
    demos = tensors["demo_indices"]
    if (
        language.ndim != 2
        or language.shape[0] < 1
        or language.shape[1] != expected_dim
        or language.dtype != torch.bfloat16
        or video.ndim != 2
        or video.shape[0] < 1
        or video.shape[1] != expected_dim
        or video.dtype != torch.bfloat16
        or offsets.ndim != 1
        or offsets.dtype != torch.int64
        or demos.ndim != 1
        or demos.dtype != torch.int64
        or offsets.numel() != demos.numel() + 1
    ):
        raise FeatureCacheError("task cache tensor shapes or dtypes changed")
    offset_values = offsets.tolist()
    if (
        offset_values[0] != 0
        or offset_values[-1] != video.shape[0]
        or any(right <= left for left, right in zip(offset_values, offset_values[1:]))
        or len(set(demos.tolist())) != demos.numel()
    ):
        raise FeatureCacheError("task cache episode boundaries are invalid")
    return CachedWriterInput(language, video, offsets, demos)


def save_task_cache(
    path: Path,
    *,
    language_features: torch.Tensor,
    video_features: torch.Tensor,
    episode_offsets: torch.Tensor,
    demo_indices: torch.Tensor,
    metadata: Mapping[str, str],
) -> dict[str, Any]:
    tensors = {
        "language_features": language_features.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
        "video_features": video_features.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
        "episode_offsets": episode_offsets.detach().to(device="cpu", dtype=torch.int64).contiguous(),
        "demo_indices": demo_indices.detach().to(device="cpu", dtype=torch.int64).contiguous(),
    }
    cached = _validate_cached_tensors(tensors, expected_dim=video_features.shape[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    save_file(tensors, str(temporary), metadata=dict(metadata))
    os.replace(temporary, path)
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "language_tokens": int(cached.language_features.shape[0]),
        "frames": int(cached.video_features.shape[0]),
        "episodes": int(cached.demo_indices.numel()),
    }


def load_task_cache(path: Path, *, expected_dim: int = 960) -> CachedWriterInput:
    if not path.is_file():
        raise FeatureCacheError(f"task feature cache is missing: {path}")
    return _validate_cached_tensors(load_file(path, device="cpu"), expected_dim=expected_dim)


def task_cache_paths(output_dir: Path, task_id: int) -> tuple[Path, Path]:
    stem = f"task_{task_id:03d}"
    return output_dir / "tasks" / f"{stem}.safetensors", output_dir / "tasks" / f"{stem}.json"


def task_cache_is_complete(
    output_dir: Path, task_id: int, *, extraction_sha256: str
) -> bool:
    tensor_path, record_path = task_cache_paths(output_dir, task_id)
    if not tensor_path.is_file() or not record_path.is_file():
        return False
    try:
        record = read_json(record_path)
    except Exception:
        return False
    return (
        record.get("schema_version") == "ember_writer_task_feature_cache_v1"
        and record.get("task_id") == task_id
        and record.get("extraction_sha256") == extraction_sha256
        and record.get("file", {}).get("bytes") == tensor_path.stat().st_size
        and record.get("file", {}).get("sha256") == sha256_file(tensor_path)
    )


def write_task_record(output_dir: Path, task_id: int, value: Mapping[str, Any]) -> None:
    _, record_path = task_cache_paths(output_dir, task_id)
    write_json_atomic(record_path, dict(value))


def extraction_contract_sha256(contract: Mapping[str, Any]) -> str:
    """Hash only deterministic feature values, not host paths or scheduling state."""

    return canonical_hash(
        {
            "schema_version": contract["schema_version"],
            "config_sha256": contract["config_sha256"],
            "policy_files": contract["policy_files"],
            "task_ids": contract["task_ids"],
            "demo_indices": contract["demo_indices"],
            "features": contract["features"],
        }
    )
