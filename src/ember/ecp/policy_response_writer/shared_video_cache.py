"""Node-local mmap cache for frozen full policy-response videos.

The cache contains deployment-visible, action-hidden video evidence only.  A
single safetensors file backs each task/video pair, so independent torchrun
workers can map the same physical pages without duplicating the full cache in
host memory.  Files are run-local operational artifacts, not checkpoints or
scientific evidence.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ember.ecp.policy_response_writer.capture import FrozenPolicyResponseVideo
from ember.pi05_source_checkpoint import read_json, write_json_atomic


SHARED_VIDEO_CACHE_SCHEMA = "ember_policy_response_writer_shared_video_cache_v1"
_VIDEO_FIELDS = (
    "patch_states",
    "language_states",
    "language_mask",
    "layer_states",
    "flow_velocity",
    "suffix_noise",
    "frame_positions",
)


@dataclass(frozen=True)
class SharedVideoCacheResult:
    video: FrozenPolicyResponseVideo
    capture: dict[str, Any]
    hit: bool
    file_bytes: int
    build_seconds: float
    load_seconds: float


def _cpu_contiguous(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").contiguous()


def _video_tensors(video: FrozenPolicyResponseVideo) -> dict[str, torch.Tensor]:
    tensors = {
        name: _cpu_contiguous(getattr(video, name)) for name in _VIDEO_FIELDS
    }
    for index, value in enumerate(video.native_inputs):
        tensors[f"native_inputs.{index:03d}"] = _cpu_contiguous(value)
    for index, value in enumerate(video.native_outputs):
        tensors[f"native_outputs.{index:03d}"] = _cpu_contiguous(value)
    for index, value in enumerate(video.final_outputs):
        # final_outputs are views into native_outputs in live capture.  Seal a
        # tiny independent boundary tensor because safetensors rejects shared
        # storage aliases.
        tensors[f"final_outputs.{index:03d}"] = _cpu_contiguous(value).clone()
    return tensors


def _decode_video(
    tensors: Mapping[str, torch.Tensor], *, target_count: int
) -> FrozenPolicyResponseVideo:
    expected = {
        *_VIDEO_FIELDS,
        *(f"native_inputs.{index:03d}" for index in range(target_count)),
        *(f"native_outputs.{index:03d}" for index in range(target_count)),
        *(f"final_outputs.{index:03d}" for index in range(target_count)),
    }
    if set(tensors) != expected:
        raise ValueError("shared policy-response video tensor inventory changed")
    video = FrozenPolicyResponseVideo(
        **{name: tensors[name] for name in _VIDEO_FIELDS},
        native_inputs=tuple(
            tensors[f"native_inputs.{index:03d}"] for index in range(target_count)
        ),
        native_outputs=tuple(
            tensors[f"native_outputs.{index:03d}"] for index in range(target_count)
        ),
        final_outputs=tuple(
            tensors[f"final_outputs.{index:03d}"] for index in range(target_count)
        ),
    )
    if video.frame_count <= 0:
        raise ValueError("shared policy-response video lost its frame axis")
    return video


class SharedPolicyResponseVideoCache:
    """Atomic mmap cache shared by same-node Writer workers."""

    def __init__(self, root: Path, *, authority: Mapping[str, Any]) -> None:
        self.root = root.resolve()
        if not self.root.is_absolute() or self.root in {
            Path("/"),
            Path("/tmp"),
            Path("/dev/shm"),
        }:
            raise ValueError("shared policy-response cache root is not task-scoped")
        self.root.mkdir(parents=True, exist_ok=True)
        self.authority = {
            "schema_version": SHARED_VIDEO_CACHE_SCHEMA,
            **dict(authority),
            "checkpoint_payload": False,
            "formal_evidence": False,
            "deployment_input": False,
            "storage": "one_safetensors_mmap_per_task_video",
        }
        lock_path = self.root / ".manifest.lock"
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            manifest = self.root / "manifest.json"
            if manifest.is_file():
                if read_json(manifest) != self.authority:
                    raise ValueError("shared policy-response cache authority changed")
            else:
                write_json_atomic(manifest, self.authority)

    def _path(self, task: int, demo: int) -> Path:
        if task < 0 or demo < 0:
            raise ValueError("shared policy-response cache key changed")
        return self.root / f"task_{task:03d}_video_{demo:03d}.safetensors"

    @staticmethod
    def _metadata(path: Path) -> dict[str, str]:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        if (
            metadata is None
            or metadata.get("schema_version") != SHARED_VIDEO_CACHE_SCHEMA
        ):
            raise ValueError("shared policy-response cache schema changed")
        return dict(metadata)

    @staticmethod
    def _save(
        path: Path,
        video: FrozenPolicyResponseVideo,
        capture: Mapping[str, Any],
        *,
        task: int,
        demo: int,
    ) -> None:
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        metadata = {
            "schema_version": SHARED_VIDEO_CACHE_SCHEMA,
            "task": str(task),
            "demo": str(demo),
            "target_count": str(len(video.native_inputs)),
            "tensor_bytes": str(video.tensor_bytes),
            "capture_json": json.dumps(
                dict(capture), sort_keys=True, separators=(",", ":")
            ),
        }
        try:
            save_file(_video_tensors(video), str(temporary), metadata=metadata)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def _load(
        cls, path: Path, *, task: int, demo: int
    ) -> tuple[FrozenPolicyResponseVideo, dict[str, Any]]:
        metadata = cls._metadata(path)
        if (
            int(metadata.get("task", -1)) != task
            or int(metadata.get("demo", -1)) != demo
            or int(metadata.get("target_count", 0)) <= 0
        ):
            raise ValueError("shared policy-response cache pairing changed")
        tensors = load_file(str(path), device="cpu", backend="mmap")
        video = _decode_video(
            tensors, target_count=int(metadata["target_count"])
        )
        if video.tensor_bytes != int(metadata.get("tensor_bytes", -1)):
            raise ValueError("shared policy-response cache tensor bytes changed")
        capture = json.loads(metadata["capture_json"])
        if not isinstance(capture, dict):
            raise ValueError("shared policy-response cache capture changed")
        return video, capture

    def get_or_build(
        self,
        *,
        task: int,
        demo: int,
        builder: Callable[[], tuple[FrozenPolicyResponseVideo, Mapping[str, Any]]],
    ) -> SharedVideoCacheResult:
        path = self._path(task, demo)
        lock_path = path.with_suffix(".lock")
        build_seconds = 0.0
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            hit = path.is_file()
            if not hit:
                tick = time.monotonic()
                video, capture = builder()
                self._save(path, video, capture, task=task, demo=demo)
                del video
                build_seconds = time.monotonic() - tick
        load_tick = time.monotonic()
        video, capture = self._load(path, task=task, demo=demo)
        return SharedVideoCacheResult(
            video=video,
            capture=capture,
            hit=hit,
            file_bytes=path.stat().st_size,
            build_seconds=build_seconds,
            load_seconds=time.monotonic() - load_tick,
        )
