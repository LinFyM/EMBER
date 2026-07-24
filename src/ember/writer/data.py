"""Action-hidden full-video inputs and source-only functional-query data."""

from __future__ import annotations

import hashlib
import os
import struct
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


class MixedTaskBatchSampler:
    """Episode-balanced no-replacement task cycles with exact step resume.

    Across all ranks, every consecutive ``len(task_ids)`` global task slots is a
    permutation of the source tasks.  A task's sample positions walk a fixed,
    seeded permutation of its episodes cyclically, so any full episode-count
    window includes every declared episode.  Frame choices change on later
    episode cycles while remaining a pure function of the global step.
    """

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
    ) -> None:
        batch_cycle = tuple(
            int(value)
            for value in (per_rank_batch_cycle or (per_rank_batch_size,))
        )
        if (
            not task_ids
            or len(set(task_ids)) != len(task_ids)
            or per_rank_batch_size <= 0
            or not batch_cycle
            or any(value <= 0 or value > per_rank_batch_size for value in batch_cycle)
            or not 0 <= start_step <= stop_step
            or not 0 <= rank < world_size
        ):
            raise WriterModelError("invalid mixed-task sampler")
        episode_rows = dataset.task_episode_rows
        missing = set(task_ids) - set(episode_rows)
        if missing:
            raise WriterModelError(f"query rows missing for tasks: {sorted(missing)}")
        if any(
            not episodes or any(not rows for rows in episodes.values())
            for task, episodes in episode_rows.items()
            if task in task_ids
        ):
            raise WriterModelError("a task has an empty episode")
        episode_counts = {len(episode_rows[task]) for task in task_ids}
        if len(episode_counts) != 1:
            raise WriterModelError("source tasks must declare the same episode count")
        self.dataset = dataset
        self.task_ids = tuple(sorted(task_ids))
        self.per_rank_batch_size = per_rank_batch_size
        self.per_rank_batch_cycle = batch_cycle
        self.start_step = start_step
        self.stop_step = stop_step
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.episodes_per_task = episode_counts.pop()
        self.episode_rows = episode_rows
        self.episode_orders = {
            task_id: tuple(
                int(value)
                for value in np.random.default_rng(
                    np.random.SeedSequence([self.seed, task_id, 0xE91])
                ).permutation(tuple(sorted(episode_rows[task_id])))
            )
            for task_id in self.task_ids
        }
        self._query_prefixes = {task_id: [0] for task_id in self.task_ids}

    def __len__(self) -> int:
        return self.stop_step - self.start_step

    def _task_visit_for_global_slot(self, slot: int) -> tuple[int, int]:
        task_visit, offset = divmod(slot, len(self.task_ids))
        order = np.random.default_rng(
            np.random.SeedSequence([self.seed, task_visit])
        ).permutation(self.task_ids)
        return int(order[offset]), task_visit

    def task_visit_for_step(self, step: int) -> tuple[int, int]:
        """Return this rank's task and global task-cycle visit for one data step."""

        if not self.start_step <= step < self.stop_step:
            raise WriterModelError("data step is outside the sampler interval")
        return self._task_visit_for_global_slot(step * self.world_size + self.rank)

    def batch_size_for_step(self, step: int) -> int:
        if step < 0:
            raise WriterModelError("data step must be non-negative")
        return self.per_rank_batch_cycle[step % len(self.per_rank_batch_cycle)]

    def _global_slot_for_task_visit(self, task_id: int, task_visit: int) -> int:
        order = np.random.default_rng(
            np.random.SeedSequence([self.seed, task_visit])
        ).permutation(self.task_ids)
        offsets = np.flatnonzero(order == task_id)
        if offsets.size != 1:
            raise WriterModelError("task visit is outside the sampler authority")
        return task_visit * len(self.task_ids) + int(offsets[0])

    def _queries_before_task_visit(self, task_id: int, task_visit: int) -> int:
        prefixes = self._query_prefixes[task_id]
        while len(prefixes) <= task_visit:
            previous_visit = len(prefixes) - 1
            slot = self._global_slot_for_task_visit(task_id, previous_visit)
            step = slot // self.world_size
            prefixes.append(prefixes[-1] + self.batch_size_for_step(step))
        return prefixes[task_visit]

    def _episode_for_task_visit(
        self, task_id: int, task_visit: int, batch_offset: int
    ) -> tuple[int, int]:
        position = self._queries_before_task_visit(task_id, task_visit) + batch_offset
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
            for rank in range(self.world_size):
                slot = step * self.world_size + rank
                task_id, task_visit = self._task_visit_for_global_slot(slot)
                batch_size = self.batch_size_for_step(step)
                if batch_size >= self.episodes_per_task:
                    coverage[task_id].update(self.episode_orders[task_id])
                    continue
                for batch_offset in range(batch_size):
                    demo_index, _ = self._episode_for_task_visit(
                        task_id, task_visit, batch_offset
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
                slot = step * self.world_size + rank
                task_id, task_visit = self._task_visit_for_global_slot(slot)
                for batch_offset in range(self.batch_size_for_step(step)):
                    row = self._sample_for_task_visit(
                        task_id, task_visit, batch_offset
                    )
                    row_task, demo_index, frame = frame_index[row]
                    if row_task != task_id:
                        raise WriterModelError("sampler query crossed task authority")
                    digest.update(
                        struct.pack(
                            ">7q",
                            step,
                            rank,
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
            "global_examples": sum(
                self.batch_size_for_step(step)
                for step in range(start_step, stop_step)
            )
            * self.world_size,
            "unique_query_rows": len(unique_rows),
            "min_examples_per_task": min(counts),
            "max_examples_per_task": max(counts),
            "identity_sha256": digest.hexdigest(),
        }

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.start_step, self.stop_step):
            task_id, task_visit = self.task_visit_for_step(step)
            yield [
                self._sample_for_task_visit(task_id, task_visit, batch_offset)
                for batch_offset in range(self.batch_size_for_step(step))
            ]


class TeacherVideoSchedule:
    """Independent deterministic no-replacement cycles for one teaching video."""

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
        """Digest the joint action-query and independent one-video schedule."""

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
                slot = step * sampler.world_size + rank
                task_id, task_visit = sampler._task_visit_for_global_slot(slot)
                demo_index = self.demo_for_task_visit(task_id, task_visit)
                digest.update(
                    struct.pack(">5q", step, rank, task_id, task_visit, demo_index)
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
            "teacher_video_identity_sha256": video_digest,
            "combined_identity_sha256": combined,
            "min_video_visits_per_task": min(visit_counts),
            "max_video_visits_per_task": max(visit_counts),
            "min_unique_videos_per_task": min(video_counts),
            "max_unique_videos_per_task": max(video_counts),
        }


class WriterFlowNoiseSchedule:
    """Exact-resume Gaussian flow noise, shared by all frames of one visit."""

    _SEED_TAG = 0xF10A05

    def __init__(
        self,
        *,
        seed: int,
        action_horizon: int = 50,
        padded_action_dim: int = 32,
    ) -> None:
        if seed < 0 or min(action_horizon, padded_action_dim) <= 0:
            raise WriterModelError("invalid Writer flow-noise schedule")
        self.seed = int(seed)
        self.action_horizon = int(action_horizon)
        self.padded_action_dim = int(padded_action_dim)

    def _visit_seed(self, global_visit_slot: int) -> int:
        if global_visit_slot < 0:
            raise WriterModelError("Writer flow-noise visit must be non-negative")
        digest = hashlib.sha256(
            struct.pack(
                ">3Q",
                self.seed,
                int(global_visit_slot),
                self._SEED_TAG,
            )
        ).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def noise_for_visit(
        self,
        global_visit_slot: int,
        *,
        device: Any,
    ) -> Any:
        """Return one CPU-generated, device-independent ``[50,32]`` sample."""

        import torch

        generator = torch.Generator(device="cpu").manual_seed(
            self._visit_seed(global_visit_slot)
        )
        value = torch.randn(
            self.action_horizon,
            self.padded_action_dim,
            generator=generator,
            dtype=torch.float32,
        )
        return value.to(device=device, non_blocking=True)

    def identity_for_visits(
        self,
        start_global_visit: int,
        stop_global_visit: int,
    ) -> dict[str, Any]:
        if not 0 <= start_global_visit <= stop_global_visit:
            raise WriterModelError("invalid Writer flow-noise visit interval")
        digest = hashlib.sha256()
        for visit in range(start_global_visit, stop_global_visit):
            digest.update(
                struct.pack(
                    ">2Q",
                    visit,
                    self._visit_seed(visit),
                )
            )
        return {
            "writer_flow_noise_seed": self.seed,
            "start_global_visit": start_global_visit,
            "stop_global_visit": stop_global_visit,
            "action_horizon": self.action_horizon,
            "padded_action_dim": self.padded_action_dim,
            "identity_sha256": digest.hexdigest(),
        }
