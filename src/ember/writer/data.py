"""Action-hidden full-video inputs and source-only functional-query data."""

from __future__ import annotations

import hashlib
import os
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_authority(authority: WriterTaskAuthority) -> None:
    if (
        not authority.path.is_file()
        or authority.path.stat().st_size != authority.expected_bytes
    ):
        raise WriterModelError(f"task authority changed: {authority.task_id}")
    if (
        authority.expected_sha256 is not None
        and _sha256(authority.path) != authority.expected_sha256
    ):
        raise WriterModelError(f"task SHA256 changed: {authority.task_id}")


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


def _task_complete_batch_cycle(
    *,
    task_ids: Sequence[int],
    per_rank_batch_size: int,
    per_rank_batch_cycle: Sequence[int] | None,
    start_step: int,
    stop_step: int,
    rank: int,
    world_size: int,
    tasks_per_rank_per_update: int,
) -> tuple[int, ...]:
    cycle = tuple(
        int(value)
        for value in (per_rank_batch_cycle or (per_rank_batch_size,))
    )
    invalid = (
        not task_ids
        or len(set(task_ids)) != len(task_ids)
        or per_rank_batch_size <= 0
        or cycle != (per_rank_batch_size,)
        or not 0 <= start_step <= stop_step
        or not 0 <= rank < world_size
        or tasks_per_rank_per_update <= 0
        or len(task_ids) != world_size * tasks_per_rank_per_update
    )
    if invalid:
        raise WriterModelError("invalid task-complete sampler")
    return cycle


def _task_complete_episode_rows(
    dataset: FunctionalQueryDataset,
    task_ids: Sequence[int],
) -> tuple[dict[int, dict[int, tuple[int, ...]]], int]:
    episode_rows = dataset.task_episode_rows
    missing = set(task_ids) - set(episode_rows)
    if missing:
        raise WriterModelError(f"query rows missing for tasks: {sorted(missing)}")
    active = {task_id: episode_rows[task_id] for task_id in task_ids}
    if any(
        not episodes or any(not rows for rows in episodes.values())
        for episodes in active.values()
    ):
        raise WriterModelError("a task has an empty episode")
    episode_counts = {len(episodes) for episodes in active.values()}
    if len(episode_counts) != 1:
        raise WriterModelError("source tasks must declare the same episode count")
    return episode_rows, episode_counts.pop()


def _task_complete_video_costs(
    *,
    task_ids: tuple[int, ...],
    episode_rows: Mapping[int, Mapping[int, Sequence[int]]],
    video_schedule: TeacherVideoSchedule,
    task_video_costs: Mapping[int, Mapping[int, int]],
) -> dict[int, dict[int, int]]:
    costs = {
        int(task_id): {
            int(demo_index): int(value)
            for demo_index, value in demo_costs.items()
        }
        for task_id, demo_costs in task_video_costs.items()
    }
    declared_demos = {
        task_id: set(episode_rows[task_id]) for task_id in task_ids
    }
    invalid = (
        video_schedule.task_ids != task_ids
        or set(costs) != set(task_ids)
        or any(
            set(costs[task_id]) != declared_demos[task_id]
            or any(value <= 0 for value in costs[task_id].values())
            for task_id in task_ids
        )
    )
    if invalid:
        raise WriterModelError("task-complete video costs changed")
    return costs


class MixedTaskBatchSampler:
    """Yield one cost-balanced, task-complete macro update at a time.

    Every optimizer update contains every declared task exactly once globally.
    Each physical rank processes a cost-balanced group sequentially, and groups
    rotate across ranks so task identity is not permanently coupled to rank RNG.
    Each yielded physical batch remains task-pure for one video-conditioned LoRA.
    """

    _GROUP_SEED_TAG = 0xC057

    def __init__(
        self,
        dataset: FunctionalQueryDataset,
        *,
        task_ids: Sequence[int],
        per_rank_batch_size: int,
        per_rank_batch_cycle: Sequence[int] | None = None,
        start_step: int,
        stop_step: int,
        rank: int,
        world_size: int,
        seed: int,
        tasks_per_rank_per_update: int,
        video_schedule: TeacherVideoSchedule,
        task_video_costs: Mapping[int, Mapping[int, int]],
    ) -> None:
        batch_cycle = _task_complete_batch_cycle(
            task_ids=task_ids,
            per_rank_batch_size=per_rank_batch_size,
            per_rank_batch_cycle=per_rank_batch_cycle,
            start_step=start_step,
            stop_step=stop_step,
            rank=rank,
            world_size=world_size,
            tasks_per_rank_per_update=tasks_per_rank_per_update,
        )
        episode_rows, episodes_per_task = _task_complete_episode_rows(
            dataset,
            task_ids,
        )
        self.dataset = dataset
        self.task_ids = tuple(sorted(task_ids))
        self.per_rank_batch_size = per_rank_batch_size
        self.per_rank_batch_cycle = batch_cycle
        self.start_step = start_step
        self.stop_step = stop_step
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.tasks_per_rank_per_update = int(tasks_per_rank_per_update)
        self.episodes_per_task = episodes_per_task
        self.episode_rows = episode_rows
        self.video_schedule = video_schedule
        self.task_video_costs = _task_complete_video_costs(
            task_ids=self.task_ids,
            episode_rows=episode_rows,
            video_schedule=video_schedule,
            task_video_costs=task_video_costs,
        )
        self.episode_orders = {
            task_id: tuple(
                int(value)
                for value in np.random.default_rng(
                    np.random.SeedSequence([self.seed, task_id, 0xE91])
                ).permutation(tuple(sorted(episode_rows[task_id])))
            )
            for task_id in self.task_ids
        }

    def __len__(self) -> int:
        return (
            self.stop_step - self.start_step
        ) * self.tasks_per_rank_per_update

    def _cost_balanced_groups(self, step: int) -> tuple[tuple[int, ...], ...]:
        current_costs = {
            task_id: self.task_video_costs[task_id][
                self.video_schedule.demo_for_task_visit(task_id, step)
            ]
            for task_id in self.task_ids
        }
        tie_order = np.random.default_rng(
            np.random.SeedSequence([self.seed, step, self._GROUP_SEED_TAG])
        ).permutation(self.task_ids)
        tie_rank = {int(task_id): index for index, task_id in enumerate(tie_order)}
        ordered = sorted(
            self.task_ids,
            key=lambda task_id: (-current_costs[task_id], tie_rank[task_id]),
        )
        groups: list[list[int]] = [[] for _ in range(self.world_size)]
        loads = [0] * self.world_size
        rank_offset = (self.seed + step) % self.world_size
        for task_id in ordered:
            candidates = [
                rank
                for rank, group in enumerate(groups)
                if len(group) < self.tasks_per_rank_per_update
            ]
            rank = min(
                candidates,
                key=lambda value: (
                    loads[value],
                    (value - rank_offset) % self.world_size,
                ),
            )
            groups[rank].append(task_id)
            loads[rank] += current_costs[task_id]
        result = tuple(
            tuple(
                sorted(
                    group,
                    key=lambda task_id: (
                        -current_costs[task_id],
                        tie_rank[task_id],
                    ),
                )
            )
            for group in groups
        )
        if (
            any(len(group) != self.tasks_per_rank_per_update for group in result)
            or set().union(*(set(group) for group in result))
            != set(self.task_ids)
        ):
            raise WriterModelError("task-complete cost balancing failed")
        return result

    def tasks_for_step(
        self,
        step: int,
        *,
        rank: int | None = None,
    ) -> tuple[int, ...]:
        """Return long-first tasks for one rank in a complete macro update."""

        selected_rank = self.rank if rank is None else rank
        if step < 0 or not 0 <= selected_rank < self.world_size:
            raise WriterModelError("task-complete step or rank is invalid")
        group = (selected_rank + step) % self.world_size
        return self._cost_balanced_groups(step)[group]

    def task_visit_for_step(self, step: int, microtask: int) -> tuple[int, int]:
        """Return one task and its visit index inside a macro update."""

        if (
            not self.start_step <= step < self.stop_step
            or not 0 <= microtask < self.tasks_per_rank_per_update
        ):
            raise WriterModelError("task-complete microtask is outside the sampler")
        return self.tasks_for_step(step)[microtask], step

    def batch_size_for_step(self, step: int) -> int:
        if step < 0:
            raise WriterModelError("data step must be non-negative")
        return self.per_rank_batch_cycle[step % len(self.per_rank_batch_cycle)]

    def _episode_for_task_visit(
        self, task_id: int, task_visit: int, batch_offset: int
    ) -> tuple[int, int]:
        position = task_visit * self.per_rank_batch_size + batch_offset
        episode_cycle, episode_offset = divmod(position, self.episodes_per_task)
        demo_index = self.episode_orders[task_id][episode_offset]
        return demo_index, episode_cycle

    def _sample_for_task_visit(
        self, task_id: int, task_visit: int, batch_offset: int
    ) -> int:
        demo_index, episode_cycle = self._episode_for_task_visit(
            task_id, task_visit, batch_offset
        )
        rows = self.episode_rows[task_id][demo_index]
        row_offset = int(
            np.random.default_rng(
                np.random.SeedSequence(
                    [self.seed, task_id, demo_index, episode_cycle, 0xF4A]
                )
            ).integers(len(rows))
        )
        return rows[row_offset]

    def coverage_for_steps(
        self, start_step: int, stop_step: int
    ) -> dict[int, tuple[int, ...]]:
        """Return exact cross-rank episode coverage for a half-open step range."""

        if not 0 <= start_step <= stop_step:
            raise WriterModelError("invalid coverage step range")
        coverage = {task_id: set() for task_id in self.task_ids}
        for step in range(start_step, stop_step):
            for task_id in self.task_ids:
                batch_size = self.batch_size_for_step(step)
                if batch_size >= self.episodes_per_task:
                    coverage[task_id].update(self.episode_orders[task_id])
                    continue
                for batch_offset in range(batch_size):
                    demo_index, _ = self._episode_for_task_visit(
                        task_id, step, batch_offset
                    )
                    coverage[task_id].add(demo_index)
        return {
            task_id: tuple(sorted(demo_indices))
            for task_id, demo_indices in coverage.items()
        }

    def consumed_identity_summary(
        self, start_step: int, stop_step: int
    ) -> dict[str, Any]:
        """Digest the exact cross-rank task/episode/frame query schedule."""

        if not 0 <= start_step <= stop_step:
            raise WriterModelError("invalid consumed-query step range")
        digest = hashlib.sha256()
        unique_rows: set[int] = set()
        task_examples = {task_id: 0 for task_id in self.task_ids}
        frame_index = self.dataset.frame_index
        for step in range(start_step, stop_step):
            for rank in range(self.world_size):
                for microtask, task_id in enumerate(
                    self.tasks_for_step(step, rank=rank)
                ):
                    for batch_offset in range(self.batch_size_for_step(step)):
                        row = self._sample_for_task_visit(
                            task_id, step, batch_offset
                        )
                        row_task, demo_index, frame = frame_index[row]
                        if row_task != task_id:
                            raise WriterModelError(
                                "sampler query crossed task authority"
                            )
                        digest.update(
                            struct.pack(
                                ">8q",
                                step,
                                rank,
                                microtask,
                                batch_offset,
                                row,
                                task_id,
                                demo_index,
                                frame,
                            )
                        )
                        unique_rows.add(row)
                        task_examples[task_id] += 1
        counts = tuple(task_examples.values())
        return {
            "start_step": start_step,
            "stop_step": stop_step,
            "global_examples": (
                (stop_step - start_step)
                * len(self.task_ids)
                * self.per_rank_batch_size
            ),
            "unique_query_rows": len(unique_rows),
            "min_examples_per_task": min(counts),
            "max_examples_per_task": max(counts),
            "identity_sha256": digest.hexdigest(),
        }

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.start_step, self.stop_step):
            for task_id in self.tasks_for_step(step):
                yield [
                    self._sample_for_task_visit(task_id, step, batch_offset)
                    for batch_offset in range(self.batch_size_for_step(step))
                ]


class TeacherVideoSchedule:
    """Deterministic one-video schedule with no-replacement task cycles."""

    _SEED_TAG = 0x71DE0

    def __init__(
        self,
        *,
        task_ids: Sequence[int],
        demo_indices: Sequence[int],
        seed: int,
    ) -> None:
        if (
            not task_ids
            or len(set(task_ids)) != len(task_ids)
            or not demo_indices
            or len(set(demo_indices)) != len(demo_indices)
            or seed < 0
        ):
            raise WriterModelError("invalid teacher-video schedule")
        self.task_ids = tuple(sorted(int(value) for value in task_ids))
        self.demo_indices = tuple(sorted(int(value) for value in demo_indices))
        self.seed = int(seed)

    def demo_for_task_visit(self, task_id: int, task_visit: int) -> int:
        if task_id not in self.task_ids or task_visit < 0:
            raise WriterModelError("teacher-video request is outside the schedule")
        cycle, offset = divmod(task_visit, len(self.demo_indices))
        order = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, task_id, cycle, self._SEED_TAG]
            )
        ).permutation(self.demo_indices)
        return int(order[offset])

    def identity_for_task_visits(
        self, task_id: int, start_visit: int, stop_visit: int
    ) -> dict[str, Any]:
        if not 0 <= start_visit <= stop_visit:
            raise WriterModelError("invalid teacher-video visit range")
        demos = tuple(
            self.demo_for_task_visit(task_id, visit)
            for visit in range(start_visit, stop_visit)
        )
        digest = hashlib.sha256()
        for visit, demo_index in enumerate(demos, start=start_visit):
            digest.update(struct.pack(">3q", task_id, visit, demo_index))
        return {
            "task_id": task_id,
            "start_visit": start_visit,
            "stop_visit": stop_visit,
            "demo_indices": demos,
            "unique_demo_indices": tuple(sorted(set(demos))),
            "identity_sha256": digest.hexdigest(),
        }

    def consumed_identity_summary(
        self,
        sampler: MixedTaskBatchSampler,
        start_step: int,
        stop_step: int,
    ) -> dict[str, Any]:
        """Digest action queries and the one video used by every task visit."""

        declared_sets = {
            tuple(sorted(sampler.episode_rows[task_id]))
            for task_id in sampler.task_ids
        }
        declared_demos = next(iter(declared_sets)) if len(declared_sets) == 1 else ()
        if (
            self.task_ids != sampler.task_ids
            or self.demo_indices != declared_demos
            or not 0 <= start_step <= stop_step
        ):
            raise WriterModelError("Writer consumed schedule authorities differ")
        query = sampler.consumed_identity_summary(start_step, stop_step)
        digest = hashlib.sha256()
        coverage = {task_id: set() for task_id in self.task_ids}
        visits = {task_id: 0 for task_id in self.task_ids}
        for step in range(start_step, stop_step):
            for rank in range(sampler.world_size):
                for microtask, task_id in enumerate(
                    sampler.tasks_for_step(step, rank=rank)
                ):
                    demo_index = self.demo_for_task_visit(task_id, step)
                    digest.update(
                        struct.pack(
                            ">6q",
                            step,
                            rank,
                            microtask,
                            task_id,
                            step,
                            demo_index,
                        )
                    )
                    coverage[task_id].add(demo_index)
                    visits[task_id] += 1
        video_digest = digest.hexdigest()
        combined = hashlib.sha256(
            bytes.fromhex(query["identity_sha256"]) + bytes.fromhex(video_digest)
        ).hexdigest()
        visit_counts = tuple(visits.values())
        video_counts = tuple(len(value) for value in coverage.values())
        return {
            "query": query,
            "teacher_video_seed": self.seed,
            "videos_per_task_visit": 1,
            "teacher_video_identity_sha256": video_digest,
            "combined_identity_sha256": combined,
            "min_video_visits_per_task": min(visit_counts),
            "max_video_visits_per_task": max(visit_counts),
            "min_unique_videos_per_task": min(video_counts),
            "max_unique_videos_per_task": max(video_counts),
        }
