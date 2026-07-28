"""Deterministic hierarchical mixed-task batches for shared Source-SFT."""

from __future__ import annotations

import hashlib
import struct
from functools import lru_cache
from typing import Any, Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler

from ember.writer.data import FunctionalQueryDataset
from ember.writer.model import WriterModelError


class CyclicSubsetMixedBatchSampler(Sampler[list[int]]):
    """Yield mixed-rank, partial-task updates in complete task cycles.

    Every optimizer update assigns a disjoint task subset to each rank, while
    every physical rank mixes the same number of tasks. Consecutive updates
    form a complete cycle that visits every declared task exactly once. A
    task's action rows follow deterministic no-replacement episode and chunk
    cycles indexed by its complete-cycle visit, so resume is sample exact.
    """

    _EPISODE_SEED_TAG = 0xE915
    _CHUNK_SEED_TAG = 0xC84A
    _TASK_CYCLE_SEED_TAG = 0x7A5C
    _BATCH_SEED_TAG = 0xBA7C

    def __init__(
        self,
        dataset: FunctionalQueryDataset,
        *,
        task_ids: Sequence[int],
        per_rank_batch_size: int,
        tasks_per_rank_per_update: int,
        start_step: int,
        stop_step: int,
        rank: int,
        world_size: int,
        seed: int,
    ) -> None:
        selected_tasks = tuple(sorted(int(value) for value in task_ids))
        global_tasks_per_update = int(world_size) * int(
            tasks_per_rank_per_update
        )
        episode_rows = dataset.task_episode_rows
        invalid = (
            not selected_tasks
            or len(set(selected_tasks)) != len(selected_tasks)
            or per_rank_batch_size <= 0
            or tasks_per_rank_per_update <= 1
            or per_rank_batch_size % tasks_per_rank_per_update != 0
            or global_tasks_per_update <= 0
            or len(selected_tasks) % global_tasks_per_update != 0
            or not 0 <= start_step <= stop_step
            or not 0 <= rank < world_size
            or world_size <= 0
            or seed < 0
            or set(selected_tasks) - set(episode_rows)
        )
        if invalid:
            raise WriterModelError("invalid cyclic subset mixed-task sampler")
        active = {task_id: episode_rows[task_id] for task_id in selected_tasks}
        if any(
            not episodes or any(not rows for rows in episodes.values())
            for episodes in active.values()
        ):
            raise WriterModelError("cyclic subset sampler found an empty episode")
        episode_counts = {len(episodes) for episodes in active.values()}
        if len(episode_counts) != 1:
            raise WriterModelError(
                "cyclic subset sampler requires equal episode counts per task"
            )

        self.dataset = dataset
        self.task_ids = selected_tasks
        self.per_rank_batch_size = int(per_rank_batch_size)
        self.tasks_per_rank_per_update = int(tasks_per_rank_per_update)
        self.global_tasks_per_update = global_tasks_per_update
        self.updates_per_complete_task_cycle = (
            len(self.task_ids) // self.global_tasks_per_update
        )
        self.samples_per_task_per_visit = (
            self.per_rank_batch_size // self.tasks_per_rank_per_update
        )
        self.start_step = int(start_step)
        self.stop_step = int(stop_step)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.episode_rows = active
        self.episodes_per_task = episode_counts.pop()
        if any(
            len(rows) < 2
            for episodes in self.episode_rows.values()
            for rows in episodes.values()
        ):
            raise WriterModelError(
                "cyclic subset sampler requires multiple chunks per episode"
            )

    def __len__(self) -> int:
        return self.stop_step - self.start_step

    def cycle_and_phase(self, step: int) -> tuple[int, int]:
        if step < 0:
            raise WriterModelError("cyclic subset step must be non-negative")
        return divmod(step, self.updates_per_complete_task_cycle)

    @lru_cache(maxsize=None)
    def _task_order(self, cycle: int) -> tuple[int, ...]:
        if cycle < 0:
            raise WriterModelError("cyclic subset cycle must be non-negative")
        order = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, cycle, self._TASK_CYCLE_SEED_TAG]
            )
        ).permutation(self.task_ids)
        return tuple(int(value) for value in order)

    def tasks_for_step(
        self,
        step: int,
        *,
        rank: int | None = None,
    ) -> tuple[int, ...]:
        selected_rank = self.rank if rank is None else int(rank)
        if step < 0 or not 0 <= selected_rank < self.world_size:
            raise WriterModelError("cyclic subset step or rank is invalid")
        cycle, phase = self.cycle_and_phase(step)
        start = phase * self.global_tasks_per_update
        selected = self._task_order(cycle)[
            start : start + self.global_tasks_per_update
        ]
        logical_rank = (selected_rank + step) % self.world_size
        group_start = logical_rank * self.tasks_per_rank_per_update
        group = selected[
            group_start : group_start + self.tasks_per_rank_per_update
        ]
        if len(group) != self.tasks_per_rank_per_update:
            raise WriterModelError("cyclic subset task assignment changed")
        return group

    @lru_cache(maxsize=None)
    def _episode_order(self, task_id: int, episode_cycle: int) -> tuple[int, ...]:
        declared = tuple(sorted(self.episode_rows[task_id]))
        order = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, task_id, episode_cycle, self._EPISODE_SEED_TAG]
            )
        ).permutation(declared)
        return tuple(int(value) for value in order)

    @lru_cache(maxsize=None)
    def _chunk_order(self, task_id: int, demo_index: int) -> tuple[int, ...]:
        rows = self.episode_rows[task_id][demo_index]
        order = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, task_id, demo_index, self._CHUNK_SEED_TAG]
            )
        ).permutation(rows)
        return tuple(int(value) for value in order)

    def _episode_for_global_offset(
        self, task_id: int, global_offset: int
    ) -> tuple[int, int]:
        if task_id not in self.episode_rows or global_offset < 0:
            raise WriterModelError("cyclic subset episode request is invalid")
        episode_cycle, episode_offset = divmod(
            global_offset, self.episodes_per_task
        )
        return self._episode_order(task_id, episode_cycle)[episode_offset], episode_cycle

    def _row_for_global_offset(self, task_id: int, global_offset: int) -> int:
        demo_index, episode_visit = self._episode_for_global_offset(
            task_id, global_offset
        )
        order = self._chunk_order(task_id, demo_index)
        _, chunk_offset = divmod(episode_visit, len(order))
        return order[chunk_offset]

    def row_for(
        self,
        *,
        task_id: int,
        step: int,
        rank: int,
        task_offset: int,
    ) -> int:
        if (
            step < 0
            or not 0 <= rank < self.world_size
            or task_id not in self.tasks_for_step(step, rank=rank)
            or not 0 <= task_offset < self.samples_per_task_per_visit
        ):
            raise WriterModelError("cyclic subset sample request is invalid")
        cycle, _ = self.cycle_and_phase(step)
        global_offset = (
            cycle * self.samples_per_task_per_visit + task_offset
        )
        return self._row_for_global_offset(task_id, global_offset)

    def batch_for_step(
        self, step: int, *, rank: int | None = None
    ) -> tuple[int, ...]:
        selected_rank = self.rank if rank is None else int(rank)
        tasks = self.tasks_for_step(step, rank=selected_rank)
        rows = [
            self.row_for(
                task_id=task_id,
                step=step,
                rank=selected_rank,
                task_offset=task_offset,
            )
            for task_id in tasks
            for task_offset in range(self.samples_per_task_per_visit)
        ]
        order = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, step, selected_rank, self._BATCH_SEED_TAG]
            )
        ).permutation(len(rows))
        result = tuple(rows[int(index)] for index in order)
        if len(result) != self.per_rank_batch_size:
            raise WriterModelError("cyclic subset physical batch changed size")
        return result

    def coverage_for_steps(
        self, start_step: int, stop_step: int
    ) -> dict[int, tuple[int, ...]]:
        if not 0 <= start_step <= stop_step:
            raise WriterModelError("invalid cyclic subset coverage range")
        coverage = {task_id: set() for task_id in self.task_ids}
        for step in range(start_step, stop_step):
            for rank in range(self.world_size):
                for task_id in self.tasks_for_step(step, rank=rank):
                    for task_offset in range(self.samples_per_task_per_visit):
                        row = self.row_for(
                            task_id=task_id,
                            step=step,
                            rank=rank,
                            task_offset=task_offset,
                        )
                        coverage[task_id].add(self.dataset.frame_index[row][1])
        return {
            task_id: tuple(sorted(demo_indices))
            for task_id, demo_indices in coverage.items()
        }

    def consumed_identity_summary(
        self, start_step: int, stop_step: int
    ) -> dict[str, Any]:
        if not 0 <= start_step <= stop_step:
            raise WriterModelError("invalid cyclic subset consumed range")
        digest = hashlib.sha256()
        frame_index = self.dataset.frame_index
        unique_rows: set[int] = set()
        task_examples = {task_id: 0 for task_id in self.task_ids}
        task_visits = {task_id: 0 for task_id in self.task_ids}
        for step in range(start_step, stop_step):
            for rank in range(self.world_size):
                for task_slot, task_id in enumerate(
                    self.tasks_for_step(step, rank=rank)
                ):
                    task_visits[task_id] += 1
                    for task_offset in range(self.samples_per_task_per_visit):
                        row = self.row_for(
                            task_id=task_id,
                            step=step,
                            rank=rank,
                            task_offset=task_offset,
                        )
                        row_task, demo_index, frame = frame_index[row]
                        if row_task != task_id:
                            raise WriterModelError(
                                "cyclic subset query crossed task authority"
                            )
                        digest.update(
                            struct.pack(
                                ">8q",
                                step,
                                rank,
                                task_slot,
                                task_offset,
                                row,
                                task_id,
                                demo_index,
                                frame,
                            )
                        )
                        unique_rows.add(row)
                        task_examples[task_id] += 1
        counts = tuple(task_examples.values())
        visits = tuple(task_visits.values())
        return {
            "start_step": start_step,
            "stop_step": stop_step,
            "global_examples": (
                (stop_step - start_step)
                * self.world_size
                * self.per_rank_batch_size
            ),
            "unique_query_rows": len(unique_rows),
            "min_examples_per_task": min(counts),
            "max_examples_per_task": max(counts),
            "min_task_visits": min(visits),
            "max_task_visits": max(visits),
            "tasks_per_rank_per_update": self.tasks_per_rank_per_update,
            "global_tasks_per_update": self.global_tasks_per_update,
            "updates_per_complete_task_cycle": (
                self.updates_per_complete_task_cycle
            ),
            "samples_per_task_per_visit": self.samples_per_task_per_visit,
            "identity_sha256": digest.hexdigest(),
        }

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.start_step, self.stop_step):
            yield list(self.batch_for_step(step))
