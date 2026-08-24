"""Lazy balanced data for composite-context recovery experts."""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from ember.ecp.process_meta import ProcessMetaError


@dataclass(frozen=True)
class CompositeRecoverySpec:
    variant_name: str
    phase_key: str
    language: str
    composite_path: Path
    composite_bytes: int
    composite_segments: tuple[tuple[int, int, int], ...]
    primitive_path: Path
    primitive_bytes: int
    primitive_demo_indices: tuple[int, ...]
    composite_query_count: int
    primitive_query_count: int
    initial_adapter_path: Path
    initial_adapter_bytes: int


class CompositeRecoveryDataset:
    """Second-phase successful rows mixed with original primitive rows."""

    def __init__(
        self,
        spec: CompositeRecoverySpec,
        *,
        task_id: int = 0,
        action_chunk_size: int = 50,
    ) -> None:
        self.spec = spec
        self.task_id = int(task_id)
        self.action_chunk_size = int(action_chunk_size)
        self._paths = (spec.composite_path, spec.primitive_path)
        self._index: list[tuple[int, int, int, int]] = []
        self._domain_rows: list[list[int]] = [[], []]
        self._handles: OrderedDict[tuple[int, int], h5py.File] = OrderedDict()
        if self.task_id < 0 or self.action_chunk_size <= 0:
            raise ProcessMetaError("invalid composite recovery dataset")
        self._index_composite_segments(spec.composite_segments)
        self._index_primitive_episodes(spec.primitive_demo_indices)
        counts = tuple(len(rows) for rows in self._domain_rows)
        if counts != (spec.composite_query_count, spec.primitive_query_count):
            raise ProcessMetaError("composite recovery query counts changed")

    def _append_segment(
        self, domain: int, demo_index: int, start: int, stop: int
    ) -> None:
        if not 0 <= start < stop:
            raise ProcessMetaError("composite recovery segment is empty")
        for frame_index in range(start, stop):
            flat = len(self._index)
            self._index.append((domain, demo_index, frame_index, stop))
            self._domain_rows[domain].append(flat)

    def _index_composite_segments(
        self, segments: Sequence[tuple[int, int, int]]
    ) -> None:
        with h5py.File(self._paths[0], "r") as handle:
            for demo_index, start, stop in segments:
                actions = handle.get(f"data/demo_{demo_index}/actions")
                if not isinstance(actions, h5py.Dataset) or actions.shape != (stop, 7):
                    raise ProcessMetaError("composite recovery episode changed")
                self._append_segment(0, demo_index, start, stop)

    def _index_primitive_episodes(self, demo_indices: Sequence[int]) -> None:
        with h5py.File(self._paths[1], "r") as handle:
            for demo_index in demo_indices:
                actions = handle.get(f"data/demo_{demo_index}/actions")
                if (
                    not isinstance(actions, h5py.Dataset)
                    or actions.ndim != 2
                    or actions.shape[0] <= 0
                    or actions.shape[1] != 7
                ):
                    raise ProcessMetaError("primitive recovery episode changed")
                self._append_segment(1, demo_index, 0, int(actions.shape[0]))

    @property
    def task_rows(self) -> dict[int, tuple[int, ...]]:
        return {self.task_id: tuple(range(len(self._index)))}

    @property
    def domain_rows(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return tuple(self._domain_rows[0]), tuple(self._domain_rows[1])

    def __len__(self) -> int:
        return len(self._index)

    def _handle(self, domain: int) -> h5py.File:
        key = (os.getpid(), domain)
        if key not in self._handles:
            self._handles[key] = h5py.File(self._paths[domain], "r")
        self._handles.move_to_end(key)
        return self._handles[key]

    @staticmethod
    def _camera(value: np.ndarray) -> np.ndarray:
        if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
            raise ProcessMetaError("composite recovery camera changed")
        return np.ascontiguousarray(value[::-1, ::-1].transpose(2, 0, 1))

    def __getitem__(self, item: int) -> dict[str, Any]:
        domain, demo_index, frame_index, segment_stop = self._index[item]
        demo = self._handle(domain)[f"data/demo_{demo_index}"]
        stop = min(frame_index + self.action_chunk_size, segment_stop)
        valid_actions = np.asarray(demo["actions"][frame_index:stop], dtype=np.float32)
        actions = np.repeat(valid_actions[-1:], self.action_chunk_size, axis=0)
        actions[: len(valid_actions)] = valid_actions
        action_is_pad = np.ones(self.action_chunk_size, dtype=np.bool_)
        action_is_pad[: len(valid_actions)] = False
        obs = demo["obs"]
        state = np.concatenate(
            (
                np.asarray(obs["ee_states"][frame_index], dtype=np.float32),
                np.asarray(obs["gripper_states"][frame_index], dtype=np.float32),
            )
        )
        return {
            "observation.images.camera1": self._camera(
                np.asarray(obs["agentview_rgb"][frame_index])
            ),
            "observation.images.camera2": self._camera(
                np.asarray(obs["eye_in_hand_rgb"][frame_index])
            ),
            "observation.state": state,
            "action": actions,
            "action_is_pad": action_is_pad,
            "task": self.spec.language,
            "task_id": self.task_id,
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
