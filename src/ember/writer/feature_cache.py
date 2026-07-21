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

from ember.pi05_source_checkpoint import (
    canonical_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.writer.data import WriterTaskAuthority


class FeatureCacheError(RuntimeError):
    """Raised when a cache input, artifact, or resume boundary is invalid."""


PI05_FEATURE_CACHE_CONFIG_SCHEMA = "ember_pi05_writer_feature_cache_v1"
PI05_FEATURE_CACHE_MANIFEST_SCHEMA = "ember_pi05_writer_feature_cache_manifest_v1"
PI05_TASK_FEATURE_CACHE_SCHEMA = "ember_pi05_writer_task_feature_cache_v1"


@dataclass(frozen=True)
class FeatureCacheTask:
    task_id: int
    language: str
    authority: WriterTaskAuthority
    expected_hdf5_sha256: str
    episode_lengths: tuple[int, ...]
    suite: str | None = None
    local_task_id: int | None = None
    split_role: str | None = None

    @property
    def frame_count(self) -> int:
        return sum(self.episode_lengths)


@dataclass(frozen=True)
class CachedWriterInput:
    language_features: torch.Tensor
    video_features: torch.Tensor
    episode_offsets: torch.Tensor
    demo_indices: torch.Tensor


@dataclass(frozen=True)
class OneVideoWriterInput:
    """Only the tensors permitted to enter one Writer invocation."""

    language_features: torch.Tensor
    video_features: torch.Tensor
    episode_offsets: torch.Tensor


def _load_pi05_store_manifest(
    root: Path,
    *,
    expected_extraction_sha256: str,
    expected_run_contract_file_sha256: str,
    expected_manifest_file_sha256: str,
) -> dict[str, Any]:
    manifest_path = root / "cache_manifest.json"
    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path) != expected_manifest_file_sha256
    ):
        raise FeatureCacheError("PI05 cache manifest file identity changed")
    manifest = read_json(manifest_path)
    payload = dict(manifest)
    digest = payload.pop("canonical_payload_sha256", None)
    if canonical_hash(payload) != digest:
        raise FeatureCacheError("PI05 cache manifest canonical payload changed")
    contract_path = root / "run_contract.json"
    if (
        not contract_path.is_file()
        or sha256_file(contract_path) != expected_run_contract_file_sha256
    ):
        raise FeatureCacheError("PI05 cache run-contract file identity changed")
    contract = read_json(contract_path)
    contract_payload = dict(contract)
    contract_digest = contract_payload.pop("contract_sha256", None)
    if (
        canonical_hash(contract_payload) != contract_digest
        or manifest.get("contract_sha256") != contract_digest
        or contract.get("extraction_sha256") != expected_extraction_sha256
    ):
        raise FeatureCacheError("PI05 cache run-contract linkage changed")
    return manifest


def _validate_store_records(
    manifest: Mapping[str, Any],
    *,
    task_ids: Sequence[int],
    expected_extraction_sha256: str,
) -> tuple[int, ...]:
    records = manifest.get("task_records", [])
    if not isinstance(records, list):
        raise FeatureCacheError("Writer feature-cache records are invalid")
    record_ids = tuple(sorted(int(record["task_id"]) for record in records))
    expected_ids = tuple(sorted(int(task_id) for task_id in task_ids))
    if (
        manifest.get("schema_version") != PI05_FEATURE_CACHE_MANIFEST_SCHEMA
        or manifest.get("extraction_sha256") != expected_extraction_sha256
        or len(record_ids) != len(set(record_ids))
        or not set(expected_ids) <= set(record_ids)
        or int(manifest.get("task_count", -1)) != len(record_ids)
        or any(
            record.get("schema_version") != PI05_TASK_FEATURE_CACHE_SCHEMA
            or record.get("extraction_sha256") != expected_extraction_sha256
            for record in records
            if int(record["task_id"]) in expected_ids
        )
    ):
        raise FeatureCacheError("Writer feature-cache manifest changed")
    return expected_ids


class WriterFeatureStore:
    """Bounded task-level LRU over a completed frozen-feature cache."""

    def __init__(
        self,
        root: Path,
        *,
        task_ids: Sequence[int],
        expected_extraction_sha256: str,
        max_cached_tasks: int,
        expected_dim: int,
        expected_run_contract_file_sha256: str,
        expected_manifest_file_sha256: str,
    ) -> None:
        if (
            not task_ids
            or len(set(task_ids)) != len(task_ids)
            or max_cached_tasks <= 0
            or expected_dim <= 0
        ):
            raise FeatureCacheError("invalid Writer feature-store request")
        manifest = _load_pi05_store_manifest(
            root,
            expected_extraction_sha256=expected_extraction_sha256,
            expected_run_contract_file_sha256=expected_run_contract_file_sha256,
            expected_manifest_file_sha256=expected_manifest_file_sha256,
        )
        expected_ids = _validate_store_records(
            manifest,
            task_ids=task_ids,
            expected_extraction_sha256=expected_extraction_sha256,
        )
        self.root = root
        self.task_ids = expected_ids
        self.extraction_sha256 = expected_extraction_sha256
        self.max_cached_tasks = max_cached_tasks
        self.expected_dim = expected_dim
        self.expected_task_record_schema = PI05_TASK_FEATURE_CACHE_SCHEMA
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
                record_schema=self.expected_task_record_schema,
            ):
                raise FeatureCacheError(f"task feature cache changed: {task_id}")
            self._verified.add(task_id)
        tensor_path, _ = task_cache_paths(self.root, task_id)
        cached = load_task_cache(tensor_path, expected_dim=self.expected_dim)
        self._cached[task_id] = cached
        while len(self._cached) > self.max_cached_tasks:
            self._cached.popitem(last=False)
        return cached

    def load_one_video(
        self,
        *,
        language_task_id: int,
        video_task_id: int,
        demo_index: int,
    ) -> OneVideoWriterInput:
        """Resolve language and video independently, then expose one episode only."""

        language_cache = self.load(language_task_id)
        video_cache = self.load(video_task_id)
        matches = (video_cache.demo_indices == int(demo_index)).nonzero().flatten()
        if matches.numel() != 1:
            raise FeatureCacheError(
                f"teaching demo is not unique in task cache: {video_task_id}/{demo_index}"
            )
        position = int(matches.item())
        start = int(video_cache.episode_offsets[position])
        stop = int(video_cache.episode_offsets[position + 1])
        video = video_cache.video_features[start:stop]
        if video.ndim != 2 or video.shape[0] <= 0:
            raise FeatureCacheError("teaching video slice is empty")
        return OneVideoWriterInput(
            language_features=language_cache.language_features,
            video_features=video,
            episode_offsets=torch.tensor([0, stop - start], dtype=torch.int64),
        )

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
    required_split = protocol.get("required_split")
    expected_task_count = {"train": 70, "validation": 10}.get(required_split)
    if (
        expected_task_count is None
        or protocol.get("task_count") != expected_task_count
        or protocol.get("demo_count_per_task") != 50
    ):
        raise FeatureCacheError(
            "feature cache must cover one sealed train or validation video pool"
        )
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


def _load_pi05_feature_authorities(
    config: Mapping[str, Any], repo_root: Path
) -> dict[str, Any]:
    authorities = config.get("authorities", {})
    required_authorities = {
        "target_data_manifest",
        "evaluation_config",
        "tokenizer_manifest",
        "lora_contract",
    }
    if set(authorities) != required_authorities:
        raise FeatureCacheError("PI05 feature-cache authority set changed")
    for name, authority in authorities.items():
        artifact = repo_root / str(authority.get("path", ""))
        if not artifact.is_file() or sha256_file(artifact) != authority.get("sha256"):
            raise FeatureCacheError(f"sealed PI05 feature-cache authority changed: {name}")
    return read_json(repo_root / str(authorities["target_data_manifest"]["path"]))


def _validate_pi05_video_protocol(
    config: Mapping[str, Any], target_manifest: Mapping[str, Any]
) -> None:
    dataset = target_manifest.get("dataset", {})
    protocol = config.get("protocol", {})
    expected_roles = {"development": ["train", "validation"]}
    expected_counts = {"development": 32}
    if (
        target_manifest.get("schema_version") != "ember_pi05_target_data_manifest_v1"
        or int(target_manifest.get("summary", {}).get("tasks", -1)) != 40
        or int(target_manifest.get("summary", {}).get("episodes", -1)) != 2000
        or protocol.get("dataset_repository") != dataset.get("repository")
        or protocol.get("dataset_revision") != dataset.get("revision")
        or protocol.get("role_split_roles") != expected_roles
        or protocol.get("role_task_counts") != expected_counts
        or int(protocol.get("demo_count_per_task", 0)) != 50
    ):
        raise FeatureCacheError("PI05 target-video protocol changed")


def _validate_pi05_feature_values(config: Mapping[str, Any]) -> None:
    if config.get("model") != {
        "policy_type": "pi05",
        "source_checkpoint": "final_formal_ema_only",
        "vision_owner": "policy.model.paligemma_with_expert.embed_image",
        "language_owner": "policy.model.paligemma_with_expert.embed_language_tokens",
        "forbidden_checkpoint": "pi05_libero",
    }:
        raise FeatureCacheError("PI05 feature model owner changed")
    if config.get("information_wall") != {
        "authorized_video_split_roles": ["train", "validation"],
        "test_video_values_read": 0,
        "trajectory_action_state_reward_terminal_reads": 0,
        "writer_input": "pure task language plus exactly one action-hidden agentview video",
    }:
        raise FeatureCacheError("PI05 development feature information wall changed")

    features = config.get("features", {})
    required_features = {
        "camera_dataset": "obs/agentview_rgb",
        "camera_transform": "libero_opengl_rotate_180_chw_unit_float_v1",
        "model_preprocessing": "PI05Policy._preprocess_images_resize_with_pad_224_neg_one_to_one",
        "vision_token_count": 256,
        "vision_feature_dim": 2048,
        "vision_pooling": "mean_over_projected_spatial_tokens_per_frame",
        "vision_normalization": "none_after_pi05_projection",
        "language_feature_dim": 2048,
        "language_max_tokens": 64,
        "observed_target40_max_language_tokens": 23,
        "language_prompt": "Task: {cleaned_task}\n",
        "language_normalization": "none_after_pi05_embedding",
        "stored_dtype": "bfloat16",
        "preserve_episode_order_and_boundaries": True,
        "writer_invocation_video_count": 1,
    }
    if any(features.get(key) != value for key, value in required_features.items()):
        raise FeatureCacheError("PI05 frozen feature extraction contract changed")
    if int(features.get("frame_batch_size_per_rank", 0)) <= 0:
        raise FeatureCacheError("PI05 frame batch size must be positive")


def _validate_pi05_feature_runtime(config: Mapping[str, Any]) -> None:
    features = config["features"]
    profile = config.get("profile", {})
    if (
        profile.get("status") not in {"pending_source_base", "sealed"}
        or profile.get("candidate_frame_batch_size_per_rank")
        != features["frame_batch_size_per_rank"]
        or profile.get("selection_metric")
        != "valid cached frames per second with zero OOM and exact tensor contract"
    ):
        raise FeatureCacheError("PI05 feature-cache profile contract changed")
    parallel = config.get("parallel", {})
    if (
        parallel.get("world_size") != 8
        or parallel.get("policy_processes_per_gpu") != 1
        or parallel.get("gpu0_extra_cuda_roles") != 0
        or parallel.get("assignment")
        != "deterministic_lpt_by_manifest_frame_count"
        or parallel.get("task_level_atomic_resume") is not True
    ):
        raise FeatureCacheError("PI05 feature cache requires eight symmetric CUDA ranks")


def load_pi05_feature_cache_config(path: Path, repo_root: Path) -> dict[str, Any]:
    """Load the one active PI05 Writer feature-cache contract."""

    config = read_json(path)
    if config.get("schema_version") != PI05_FEATURE_CACHE_CONFIG_SCHEMA:
        raise FeatureCacheError("unsupported PI05 Writer feature-cache schema")
    target_manifest = _load_pi05_feature_authorities(config, repo_root)
    _validate_pi05_video_protocol(config, target_manifest)
    _validate_pi05_feature_values(config)
    _validate_pi05_feature_runtime(config)
    return config


def load_pi05_feature_tasks(
    config: Mapping[str, Any],
    repo_root: Path,
    data_root: Path,
    *,
    role: str,
) -> tuple[FeatureCacheTask, ...]:
    """Resolve a sealed target-video role without reading demonstration values."""

    protocol = config.get("protocol", {})
    role_splits = protocol.get("role_split_roles", {})
    role_counts = protocol.get("role_task_counts", {})
    if role not in role_splits or role not in role_counts:
        raise FeatureCacheError(f"unsupported PI05 feature-cache role: {role}")
    manifest_ref = config["authorities"]["target_data_manifest"]
    manifest = read_json(repo_root / str(manifest_ref["path"]))
    selected_roles = set(role_splits[role])
    tasks: list[FeatureCacheTask] = []
    for record in manifest.get("tasks", []):
        if record.get("split_role") not in selected_roles:
            continue
        hdf5 = record["hdf5"]
        lengths = tuple(
            int(value) for value in record["demonstrations"]["episode_lengths"]
        )
        if (
            len(lengths) != int(protocol["demo_count_per_task"])
            or any(length <= 0 for length in lengths)
            or sum(lengths) != int(record["demonstrations"]["steps"])
        ):
            raise FeatureCacheError(
                f"invalid target video authority: {record['suite']}/{record['task_id']}"
            )
        authority = WriterTaskAuthority(
            task_id=int(record["global_task_id"]),
            language=str(record["language"]),
            path=data_root / str(hdf5["relative_path"]),
            expected_bytes=int(hdf5["bytes"]),
            expected_sha256=str(hdf5["sha256"]),
        )
        tasks.append(
            FeatureCacheTask(
                task_id=authority.task_id,
                language=authority.language,
                authority=authority,
                expected_hdf5_sha256=str(hdf5["sha256"]),
                episode_lengths=lengths,
                suite=str(record["suite"]),
                local_task_id=int(record["task_id"]),
                split_role=str(record["split_role"]),
            )
        )
    tasks.sort(key=lambda task: task.task_id)
    expected_count = int(role_counts[role])
    if (
        len(tasks) != expected_count
        or len({task.task_id for task in tasks}) != expected_count
        or any(task.task_id < 0 or task.task_id >= 40 for task in tasks)
    ):
        raise FeatureCacheError("PI05 feature-cache task role changed")
    return tuple(tasks)


def load_feature_tasks(
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
        raise FeatureCacheError(
            "feature cache task count differs from its sealed split role"
        )
    return tuple(tasks)


def load_train_tasks(
    config: Mapping[str, Any], repo_root: Path, data_root: Path
) -> tuple[FeatureCacheTask, ...]:
    """Load the source-only pool used by Writer training."""

    if config.get("protocol", {}).get("required_split") != "train":
        raise FeatureCacheError("Writer training cache must remain source-only")
    return load_feature_tasks(config, repo_root, data_root)


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


def pool_pi05_visual_tokens(
    embeddings: torch.Tensor, *, expected_tokens: int, expected_dim: int
) -> torch.Tensor:
    """Pool projected PI05 SigLIP tokens without SmolVLA's extra scaling."""

    if (
        embeddings.ndim != 3
        or embeddings.shape[1] != expected_tokens
        or embeddings.shape[2] != expected_dim
    ):
        raise FeatureCacheError(
            f"unexpected PI05 image embedding shape: {tuple(embeddings.shape)}"
        )
    return embeddings.mean(dim=1)


def select_pi05_language_tokens(
    embeddings: torch.Tensor, attention_mask: torch.Tensor, *, expected_dim: int
) -> torch.Tensor:
    """Select valid PI05 language embeddings without adding hidden state tokens."""

    if (
        embeddings.ndim != 3
        or embeddings.shape[0] != 1
        or embeddings.shape[2] != expected_dim
        or attention_mask.shape != embeddings.shape[:2]
    ):
        raise FeatureCacheError("unexpected PI05 language embedding shape")
    mask = attention_mask[0].to(dtype=torch.bool, device=embeddings.device)
    if not bool(mask.any()):
        raise FeatureCacheError("PI05 language tokenization produced no valid tokens")
    return embeddings[0, mask]


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
    output_dir: Path,
    task_id: int,
    *,
    extraction_sha256: str,
    record_schema: str = "ember_writer_task_feature_cache_v1",
) -> bool:
    tensor_path, record_path = task_cache_paths(output_dir, task_id)
    if not tensor_path.is_file() or not record_path.is_file():
        return False
    try:
        record = read_json(record_path)
    except Exception:
        return False
    return (
        record.get("schema_version") == record_schema
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

    source = contract.get("source", {})
    tokenizer = contract.get("tokenizer", {})
    authority_hashes = {
        name: value.get("sha256")
        for name, value in contract.get("authorities", {}).items()
    }
    return canonical_hash(
        {
            "schema_version": contract["schema_version"],
            "config_sha256": contract["config_sha256"],
            "git_commit": contract.get("git", {}).get("commit"),
            "authority_hashes": authority_hashes,
            "source": {
                key: source.get(key)
                for key in (
                    "source_run_contract_sha256",
                    "checkpoint_manifest_sha256",
                    "optimizer_step",
                    "source_run_summary_sha256",
                    "source_training_commit",
                    "source_base_config_sha256",
                    "source_authority_hashes",
                    "model_files",
                )
            },
            "policy_files": contract["policy_files"],
            "tokenizer": {
                "sha256": tokenizer.get("sha256"),
                "manifest_sha256": tokenizer.get("manifest_sha256"),
            },
            "task_ids": contract["task_ids"],
            "tasks": contract.get("tasks"),
            "demo_indices": contract["demo_indices"],
            "features": contract["features"],
            "runtime_versions": contract.get("runtime_versions"),
        }
    )
