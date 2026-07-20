"""Leakage-safe action-hidden specification and functional-query data access."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import h5py
import numpy as np

from ember.writer.core import WriterColdStartError


@dataclass(frozen=True)
class WriterSpecAuthority:
    task_id: int
    language: str
    path: Path
    expected_bytes: int
    expected_sha256: str | None


def verify_authority(authority: WriterSpecAuthority) -> None:
    if not authority.path.is_file() or authority.path.stat().st_size != authority.expected_bytes:
        raise WriterColdStartError(f"Writer HDF5 authority changed for task {authority.task_id}")


def _camera(value: np.ndarray) -> np.ndarray:
    if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
        raise WriterColdStartError("Writer camera frame changed shape or dtype")
    return np.ascontiguousarray(value[::-1, ::-1].transpose(2, 0, 1))


def _camera_batch(value: np.ndarray) -> np.ndarray:
    if value.ndim != 4 or value.shape[-1] != 3 or value.dtype != np.uint8:
        raise WriterColdStartError("Writer camera video changed shape or dtype")
    return np.ascontiguousarray(value[:, ::-1, ::-1].transpose(0, 3, 1, 2))


def iter_action_hidden_video_chunks(
    authority: WriterSpecAuthority,
    demo_indices: Sequence[int],
    *,
    chunk_size: int,
) -> Iterator[tuple[int, int, int, np.ndarray]]:
    """Stream every third-person frame while never touching privileged fields.

    Yields ``(demo_index, frame_start, episode_length, frames)``. Neither the
    number of demonstrations nor their lengths is fixed by this interface.
    """

    verify_authority(authority)
    if not demo_indices or chunk_size <= 0:
        raise WriterColdStartError("invalid action-hidden video request")
    with h5py.File(authority.path, "r") as handle:
        for demo_index in demo_indices:
            name = f"data/demo_{demo_index}"
            if name not in handle:
                raise WriterColdStartError(f"missing Writer specification {name}")
            pixels = handle[name].get("obs/agentview_rgb")
            if not isinstance(pixels, h5py.Dataset) or pixels.ndim != 4 or pixels.shape[0] <= 0:
                raise WriterColdStartError("invalid action-hidden teaching video")
            episode_length = int(pixels.shape[0])
            for start in range(0, episode_length, chunk_size):
                stop = min(start + chunk_size, episode_length)
                yield (
                    int(demo_index),
                    start,
                    episode_length,
                    _camera_batch(np.asarray(pixels[start:stop])),
                )


def read_action_hidden_spec_frames(
    authority: WriterSpecAuthority,
    demo_indices: Sequence[int],
    positions: Sequence[str],
) -> np.ndarray:
    """Read only third-person pixels; action/proprio datasets are never touched."""

    verify_authority(authority)
    if list(positions) != ["first", "middle", "last"]:
        raise WriterColdStartError("teaching-video frame rule changed")
    frames: list[np.ndarray] = []
    with h5py.File(authority.path, "r") as handle:
        for demo_index in demo_indices:
            name = f"data/demo_{demo_index}"
            if name not in handle:
                raise WriterColdStartError(f"missing Writer specification {name}")
            demo = handle[name]
            pixels = demo.get("obs/agentview_rgb")
            if not isinstance(pixels, h5py.Dataset) or pixels.ndim != 4 or pixels.shape[0] <= 0:
                raise WriterColdStartError("invalid action-hidden teaching video")
            last = pixels.shape[0] - 1
            for frame_index in (0, last // 2, last):
                frames.append(_camera(np.asarray(pixels[frame_index])))
    if not frames:
        raise WriterColdStartError("empty action-hidden teaching video")
    return np.stack(frames)


class WriterQueryDataset:
    """Lazy HDF5 functional-query dataset over sealed source tasks."""

    def __init__(
        self,
        authorities: Sequence[WriterSpecAuthority],
        *,
        demo_indices: Sequence[int],
        action_chunk_size: int,
    ) -> None:
        if not authorities or not demo_indices or action_chunk_size <= 0:
            raise WriterColdStartError("invalid Writer query dataset bounds")
        self.authorities = {item.task_id: item for item in authorities}
        self.action_chunk_size = action_chunk_size
        self._index: list[tuple[int, int, int]] = []
        self._task_rows: dict[int, list[int]] = {}
        self._handles: dict[tuple[int, int], h5py.File] = {}
        for authority in sorted(authorities, key=lambda item: item.task_id):
            verify_authority(authority)
            with h5py.File(authority.path, "r") as handle:
                for demo_index in demo_indices:
                    demo = handle.get(f"data/demo_{demo_index}")
                    if not isinstance(demo, h5py.Group):
                        raise WriterColdStartError("functional query demo is missing")
                    actions = demo.get("actions")
                    if not isinstance(actions, h5py.Dataset) or actions.ndim != 2 or actions.shape[1] != 7:
                        raise WriterColdStartError("functional query actions are invalid")
                    for frame_index in range(actions.shape[0]):
                        flat = len(self._index)
                        self._index.append((authority.task_id, demo_index, frame_index))
                        self._task_rows.setdefault(authority.task_id, []).append(flat)

    @property
    def frame_index(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(self._index)

    @property
    def task_rows(self) -> dict[int, tuple[int, ...]]:
        return {key: tuple(value) for key, value in self._task_rows.items()}

    def __len__(self) -> int:
        return len(self._index)

    def _handle(self, task_id: int) -> h5py.File:
        pid = os.getpid()
        for key in [key for key in self._handles if key[0] != pid]:
            self._handles.pop(key).close()
        key = (pid, task_id)
        if key not in self._handles:
            self._handles[key] = h5py.File(self.authorities[task_id].path, "r")
        return self._handles[key]

    def __getitem__(self, item: int) -> dict[str, Any]:
        task_id, demo_index, frame_index = self._index[item]
        demo = self._handle(task_id)[f"data/demo_{demo_index}"]
        actions_ds = demo["actions"]
        stop = min(frame_index + self.action_chunk_size, actions_ds.shape[0])
        valid = stop - frame_index
        valid_actions = np.asarray(actions_ds[frame_index:stop], dtype=np.float32)
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
            "observation.images.camera1": _camera(np.asarray(obs["agentview_rgb"][frame_index])),
            "observation.images.camera2": _camera(np.asarray(obs["eye_in_hand_rgb"][frame_index])),
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
        state["_handles"] = {}
        return state


class WriterTaskBatchSampler:
    """Deterministic single-task batches with O(1) step resume on each rank."""

    def __init__(
        self,
        dataset: WriterQueryDataset,
        *,
        task_ids: Sequence[int],
        per_rank_batch_size: int,
        start_step: int,
        stop_step: int,
        rank: int,
        world_size: int,
        seed: int,
    ) -> None:
        if (
            not task_ids
            or len(set(task_ids)) != len(task_ids)
            or per_rank_batch_size <= 0
            or not 0 <= start_step <= stop_step
            or not 0 <= rank < world_size
        ):
            raise WriterColdStartError("invalid Writer task sampler")
        missing = set(task_ids) - set(dataset.task_rows)
        if missing:
            raise WriterColdStartError(f"Writer task sampler lacks query rows: {sorted(missing)}")
        if any(len(dataset.task_rows[task]) < per_rank_batch_size for task in task_ids):
            raise WriterColdStartError("Writer task has fewer unique frames than one batch")
        self.dataset = dataset
        self.task_ids = tuple(sorted(task_ids))
        self.per_rank_batch_size = per_rank_batch_size
        self.start_step = start_step
        self.stop_step = stop_step
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

    def __len__(self) -> int:
        return self.stop_step - self.start_step

    def _task_for_slot(self, slot: int) -> int:
        cycle, offset = divmod(slot, len(self.task_ids))
        order = np.random.default_rng(np.random.SeedSequence([self.seed, cycle])).permutation(
            self.task_ids
        )
        return int(order[offset])

    def __iter__(self) -> Iterator[list[int]]:
        task_rows = self.dataset.task_rows
        for step in range(self.start_step, self.stop_step):
            task_id = self._task_for_slot(step * self.world_size + self.rank)
            rows = task_rows[task_id]
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, step, self.rank]))
            chosen = rng.choice(len(rows), size=self.per_rank_batch_size, replace=False)
            yield [rows[int(index)] for index in chosen]
