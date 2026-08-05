"""Action-hidden full-video inputs and source-only functional-query data."""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import h5py
import numpy as np

from ember.writer.model import WriterModelError


@dataclass(frozen=True)
class WriterTaskAuthority:
    """Immutable source/validation task data identity."""

    task_id: int
    language: str
    path: Path
    expected_bytes: int
    expected_sha256: str | None = None


@dataclass(frozen=True)
class RawTeacherVideo:
    """One sampled third-person teaching video and its original frame indices."""

    frames: Any
    frame_indices: Any
    raw_frame_count: int


def verify_authority(authority: WriterTaskAuthority) -> None:
    if (
        not authority.path.is_file()
        or authority.path.stat().st_size != authority.expected_bytes
    ):
        raise WriterModelError(f"task authority changed: {authority.task_id}")


def _camera(value: np.ndarray) -> np.ndarray:
    if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
        raise WriterModelError("camera frame changed shape or dtype")
    return np.ascontiguousarray(value[::-1, ::-1].transpose(2, 0, 1))


def _camera_batch(value: np.ndarray) -> np.ndarray:
    if value.ndim != 4 or value.shape[-1] != 3 or value.dtype != np.uint8:
        raise WriterModelError("camera video changed shape or dtype")
    return np.ascontiguousarray(value[:, ::-1, ::-1].transpose(0, 3, 1, 2))


def iter_action_hidden_video_chunks(
    authority: WriterTaskAuthority,
    demo_indices: Sequence[int],
    *,
    chunk_size: int,
) -> Iterator[tuple[int, int, int, np.ndarray]]:
    """Stream every third-person frame without reading privileged fields.

    Yields demo index, frame start, episode length, and a frame batch. Neither
    the number of demonstrations nor their lengths is fixed by this interface.
    """

    verify_authority(authority)
    if not demo_indices or len(set(demo_indices)) != len(demo_indices):
        raise WriterModelError("video request needs unique non-empty demo indices")
    if chunk_size <= 0:
        raise WriterModelError("video chunk size must be positive")
    with h5py.File(authority.path, "r") as handle:
        for demo_index in demo_indices:
            demo = handle.get(f"data/demo_{demo_index}")
            if not isinstance(demo, h5py.Group):
                raise WriterModelError(f"missing teaching episode {demo_index}")
            pixels = demo.get("obs/agentview_rgb")
            if (
                not isinstance(pixels, h5py.Dataset)
                or pixels.ndim != 4
                or pixels.shape[0] <= 0
            ):
                raise WriterModelError("invalid action-hidden teaching video")
            episode_length = int(pixels.shape[0])
            for start in range(0, episode_length, chunk_size):
                stop = min(start + chunk_size, episode_length)
                yield (
                    int(demo_index),
                    start,
                    episode_length,
                    _camera_batch(np.asarray(pixels[start:stop])),
                )


class RawTeacherVideoStore:
    """Load fixed-stride, variable-length videos without privileged fields."""

    def __init__(
        self,
        authorities: Sequence[WriterTaskAuthority],
        *,
        frame_stride: int,
        max_open_files: int = 2,
    ) -> None:
        if (
            not authorities
            or frame_stride <= 0
            or max_open_files <= 0
            or len({item.task_id for item in authorities}) != len(authorities)
        ):
            raise WriterModelError("invalid action-hidden video store")
        self.authorities = {item.task_id: item for item in authorities}
        self.frame_stride = int(frame_stride)
        self.max_open_files = int(max_open_files)
        self._handles: OrderedDict[int, h5py.File] = OrderedDict()
        for authority in authorities:
            verify_authority(authority)

    def _handle(self, task_id: int) -> h5py.File:
        if task_id not in self.authorities:
            raise WriterModelError("teaching video task is outside its authority")
        if task_id in self._handles:
            self._handles.move_to_end(task_id)
        else:
            self._handles[task_id] = h5py.File(
                self.authorities[task_id].path, "r"
            )
            while len(self._handles) > self.max_open_files:
                _, stale = self._handles.popitem(last=False)
                stale.close()
        return self._handles[task_id]

    def load(self, task_id: int, demo_index: int) -> RawTeacherVideo:
        if demo_index < 0:
            raise WriterModelError("teaching video demo index must be non-negative")
        demo = self._handle(task_id).get(f"data/demo_{demo_index}")
        if not isinstance(demo, h5py.Group):
            raise WriterModelError("teaching video episode is missing")
        pixels = demo.get("obs/agentview_rgb")
        if (
            not isinstance(pixels, h5py.Dataset)
            or pixels.ndim != 4
            or pixels.shape[0] <= 0
            or pixels.shape[-1] != 3
            or pixels.dtype != np.uint8
        ):
            raise WriterModelError("invalid action-hidden teaching video")
        raw_count = int(pixels.shape[0])
        indices = list(range(0, raw_count, self.frame_stride))
        if indices[-1] != raw_count - 1:
            indices.append(raw_count - 1)
        # Camera convention is identical to the execution policy: rotate 180°.
        frames = _camera_batch(np.asarray(pixels[indices]))
        return RawTeacherVideo(
            frames=frames,
            frame_indices=np.asarray(indices, dtype=np.int64),
            raw_frame_count=raw_count,
        )

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


class FunctionalQueryDataset:
    """Lazy source-only observation/action chunks for Writer or direct LoRA."""

    def __init__(
        self,
        authorities: Sequence[WriterTaskAuthority],
        *,
        demo_indices: Sequence[int],
        action_chunk_size: int,
        max_open_files_per_worker: int = 8,
    ) -> None:
        if (
            not authorities
            or not demo_indices
            or len(set(demo_indices)) != len(demo_indices)
            or action_chunk_size <= 0
            or max_open_files_per_worker <= 0
        ):
            raise WriterModelError("invalid functional-query data request")
        self.authorities = {item.task_id: item for item in authorities}
        if len(self.authorities) != len(authorities):
            raise WriterModelError("duplicate task authority")
        self.action_chunk_size = action_chunk_size
        self.max_open_files_per_worker = max_open_files_per_worker
        self._index: list[tuple[int, int, int]] = []
        self._task_rows: dict[int, list[int]] = {}
        self._task_episode_rows: dict[int, dict[int, list[int]]] = {}
        self._handles: OrderedDict[tuple[int, int], h5py.File] = OrderedDict()

        for authority in sorted(authorities, key=lambda item: item.task_id):
            verify_authority(authority)
            with h5py.File(authority.path, "r") as handle:
                for demo_index in demo_indices:
                    demo = handle.get(f"data/demo_{demo_index}")
                    if not isinstance(demo, h5py.Group):
                        raise WriterModelError("functional-query episode is missing")
                    actions = demo.get("actions")
                    if (
                        not isinstance(actions, h5py.Dataset)
                        or actions.ndim != 2
                        or actions.shape[1] != 7
                    ):
                        raise WriterModelError("functional-query actions are invalid")
                    episode_rows = self._task_episode_rows.setdefault(
                        authority.task_id, {}
                    ).setdefault(int(demo_index), [])
                    for frame_index in range(actions.shape[0]):
                        flat = len(self._index)
                        self._index.append(
                            (authority.task_id, int(demo_index), frame_index)
                        )
                        self._task_rows.setdefault(authority.task_id, []).append(flat)
                        episode_rows.append(flat)

    @property
    def frame_index(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(self._index)

    @property
    def task_rows(self) -> dict[int, tuple[int, ...]]:
        return {key: tuple(value) for key, value in self._task_rows.items()}

    @property
    def task_episode_rows(self) -> dict[int, dict[int, tuple[int, ...]]]:
        return {
            task_id: {
                demo_index: tuple(rows)
                for demo_index, rows in episodes.items()
            }
            for task_id, episodes in self._task_episode_rows.items()
        }

    def __len__(self) -> int:
        return len(self._index)

    def _handle(self, task_id: int) -> h5py.File:
        pid = os.getpid()
        for key in [key for key in self._handles if key[0] != pid]:
            self._handles.pop(key).close()
        key = (pid, task_id)
        if key in self._handles:
            self._handles.move_to_end(key)
        else:
            self._handles[key] = h5py.File(self.authorities[task_id].path, "r")
            while len(self._handles) > self.max_open_files_per_worker:
                _, stale = self._handles.popitem(last=False)
                stale.close()
        return self._handles[key]

    def __getitem__(self, item: int) -> dict[str, Any]:
        task_id, demo_index, frame_index = self._index[item]
        demo = self._handle(task_id)[f"data/demo_{demo_index}"]
        actions_ds = demo["actions"]
        stop = min(frame_index + self.action_chunk_size, actions_ds.shape[0])
        valid = stop - frame_index
        valid_actions = np.asarray(
            actions_ds[frame_index:stop], dtype=np.float32
        )
        actions = np.repeat(valid_actions[-1:], self.action_chunk_size, axis=0)
        actions[:valid] = valid_actions
        action_is_pad = np.ones(self.action_chunk_size, dtype=np.bool_)
        action_is_pad[:valid] = False
        obs = demo["obs"]
        state = np.concatenate(
            (
                np.asarray(obs["ee_states"][frame_index], dtype=np.float32),
                np.asarray(obs["gripper_states"][frame_index], dtype=np.float32),
            )
        )
        return {
            "observation.images.camera1": _camera(
                np.asarray(obs["agentview_rgb"][frame_index])
            ),
            "observation.images.camera2": _camera(
                np.asarray(obs["eye_in_hand_rgb"][frame_index])
            ),
            "observation.state": state,
            "action": actions,
            "action_is_pad": action_is_pad,
            "task": self.authorities[task_id].language,
            "task_id": task_id,
            "demo_index": demo_index,
            "frame_index": frame_index,
        }

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        return state
