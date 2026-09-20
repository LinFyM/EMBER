"""Deterministic hierarchical mixed-task batches for shared Source-SFT."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any, Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler

from ember.writer.data import FunctionalQueryDataset
from ember.writer.errors import WriterModelError


def source_batch_sizes(global_batch: int, world_size: int, micro_batch: int,
                       accumulation: int) -> tuple[tuple[int, ...], ...]:
    """Partition one logical update, keeping every rank active for DDP."""
    if min(global_batch, world_size, micro_batch, accumulation) <= 0:
        raise WriterModelError("source batch dimensions must be positive")
    counts = [global_batch // world_size + (rank < global_batch % world_size)
              for rank in range(world_size)]
    if any(not micro_batch * (accumulation - 1) < count <= micro_batch * accumulation for count in counts):
        raise WriterModelError("source microbatches must cover the global batch with a nonempty final batch on every rank")
    return tuple(tuple(min(micro_batch, count - index * micro_batch) for index in range(accumulation))
                 for count in counts)


class SourceBaseBatchSampler(Sampler[list[int]]):
    """Pack the original source's 32-query task slots onto physical ranks.

    The global query stream is independent of microbatch and world size.
    With global256, each update therefore retains the original eight task
    slots, fixed episode permutations and deterministic frame choices.
    """

    def __init__(self, dataset: FunctionalQueryDataset, *, task_ids: Sequence[int],
                 per_rank_batch_size: int, logical_task_batch_size: int,
                 global_batch_size: int, gradient_accumulation_steps: int,
                 start_step: int, stop_step: int, rank: int, world_size: int, seed: int) -> None:
        tasks = tuple(sorted(task_ids))
        if (not tasks or len(set(tasks)) != len(tasks) or not 0 <= start_step <= stop_step
                or not 0 <= rank < world_size or min(per_rank_batch_size, logical_task_batch_size) <= 0
                or seed < 0 or set(tasks) - set(dataset.task_episode_rows)):
            raise WriterModelError("invalid source-base query stream")
        episodes = {task: dataset.task_episode_rows[task] for task in tasks}
        counts = {len(value) for value in episodes.values()}
        if len(counts) != 1 or not all(rows for value in episodes.values() for rows in value.values()):
            raise WriterModelError("source-base tasks require equal nonempty episode pools")
        self.task_ids, self.episode_rows = tasks, episodes
        self.episodes_per_task = counts.pop()
        if self.episodes_per_task <= 0:
            raise WriterModelError("source-base episode pool is empty")
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.per_rank_batch_size, self.logical_task_batch_size = per_rank_batch_size, logical_task_batch_size
        plan = source_batch_sizes(global_batch_size, world_size, per_rank_batch_size, gradient_accumulation_steps)
        self.global_batch_size, self.accumulation = global_batch_size, gradient_accumulation_steps
        self.batch_sizes = plan[rank]
        self.rank_offset = sum(sum(sizes) for sizes in plan[:rank])
        self.start_step, self.stop_step = start_step, stop_step
        self.episode_orders = {
            task: tuple(int(value) for value in np.random.default_rng(
                np.random.SeedSequence([seed, task, 0xE91])).permutation(tuple(sorted(episodes[task]))))
            for task in tasks
        }

    def __len__(self) -> int:
        return self.stop_step - self.start_step

    @lru_cache(maxsize=256)
    def _task_order(self, visit: int) -> tuple[int, ...]:
        return tuple(int(value) for value in np.random.default_rng(
            np.random.SeedSequence([self.seed, visit])).permutation(self.task_ids))

    def _row(self, position: int) -> int:
        task_slot, batch_offset = divmod(position, self.logical_task_batch_size)
        task_visit, task_offset = divmod(task_slot, len(self.task_ids))
        task = self._task_order(task_visit)[task_offset]
        episode_cycle, episode_offset = divmod(
            task_visit * self.logical_task_batch_size + batch_offset, self.episodes_per_task)
        demo = self.episode_orders[task][episode_offset]
        rows = self.episode_rows[task][demo]
        offset = int(np.random.default_rng(np.random.SeedSequence(
            [self.seed, task, demo, episode_cycle, 0xF4A])).integers(len(rows)))
        return rows[offset]

    def batch_for_step(self, step: int) -> list[int]:
        if not self.start_step <= step < self.stop_step:
            raise WriterModelError("source-base micro step is outside the sampler interval")
        update, micro = divmod(step, self.accumulation)
        start = update * self.global_batch_size + self.rank_offset + sum(self.batch_sizes[:micro])
        return [self._row(start + offset) for offset in range(self.batch_sizes[micro])]

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.start_step, self.stop_step):
            yield self.batch_for_step(step)


class HierarchicalMixedBatchSampler(Sampler[list[int]]):
    """Pack the historical all-task logical stream onto physical microbatches.

    A logical rank draws equal queries from every task, then shuffles its batch.
    The concatenated logical ranks retain the historical task/episode/chunk
    sequence regardless of physical world size or gradient accumulation.
    ``start_step`` and ``stop_step`` always count optimizer updates.
    """

    kind = "hierarchical_task_episode_chunk_mixed_v1"

    def __init__(
        self, dataset: FunctionalQueryDataset, *, task_ids: Sequence[int],
        logical_world_size: int, logical_per_rank_batch_size: int,
        per_rank_batch_size: int, gradient_accumulation_steps: int,
        start_step: int, stop_step: int, rank: int, world_size: int, seed: int,
        physical_packing: str = "contiguous",
    ) -> None:
        tasks = tuple(sorted(int(value) for value in task_ids))
        rows_by_task = dataset.task_episode_rows
        if (not tasks or len(set(tasks)) != len(tasks)
                or logical_world_size <= 0 or logical_per_rank_batch_size <= 0
                or logical_per_rank_batch_size % len(tasks)
                or not 0 <= start_step <= stop_step or not 0 <= rank < world_size
                or seed < 0 or set(tasks) - set(rows_by_task)
                or physical_packing not in {"contiguous", "task_striped"}):
            raise WriterModelError("invalid hierarchical mixed-task sampler")
        episodes = {task: rows_by_task[task] for task in tasks}
        counts = {len(value) for value in episodes.values()}
        if (len(counts) != 1 or 0 in counts
                or not all(rows for value in episodes.values() for rows in value.values())):
            raise WriterModelError("hierarchical sampler requires equal nonempty episode pools")
        self.dataset, self.task_ids, self.episode_rows = dataset, tasks, episodes
        self.episodes_per_task = counts.pop()
        self.logical_world_size = logical_world_size
        self.logical_per_rank_batch_size = logical_per_rank_batch_size
        self.samples_per_task_per_logical_rank = logical_per_rank_batch_size // len(tasks)
        self.global_samples_per_task = self.samples_per_task_per_logical_rank * logical_world_size
        self.global_batch_size = logical_per_rank_batch_size * logical_world_size
        self.batch_plan = source_batch_sizes(
            self.global_batch_size, world_size, per_rank_batch_size, gradient_accumulation_steps)
        self.batch_sizes = self.batch_plan[rank]
        self.rank_offset = sum(sum(sizes) for sizes in self.batch_plan[:rank])
        self.per_rank_batch_size, self.accumulation = per_rank_batch_size, gradient_accumulation_steps
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.physical_packing = physical_packing
        self.start_step, self.stop_step = start_step, stop_step

    def __len__(self) -> int:
        return (self.stop_step - self.start_step) * self.accumulation

    @lru_cache(maxsize=256)
    def _episode_order(self, task: int, cycle: int) -> tuple[int, ...]:
        return tuple(int(value) for value in np.random.default_rng(
            np.random.SeedSequence([self.seed, task, cycle, 0xE915])
        ).permutation(tuple(sorted(self.episode_rows[task]))))

    @lru_cache(maxsize=None)
    def _chunk_order(self, task: int, demo: int) -> tuple[int, ...]:
        return tuple(int(value) for value in np.random.default_rng(
            np.random.SeedSequence([self.seed, task, demo, 0xC84A])
        ).permutation(self.episode_rows[task][demo]))

    def _episode_for_offset(self, task: int, offset: int) -> tuple[int, int]:
        cycle, index = divmod(offset, self.episodes_per_task)
        return self._episode_order(task, cycle)[index], cycle

    def _row(self, task: int, offset: int) -> int:
        demo, visit = self._episode_for_offset(task, offset)
        rows = self._chunk_order(task, demo)
        return rows[visit % len(rows)]

    def logical_batch_for_step(self, step: int, logical_rank: int) -> tuple[int, ...]:
        if step < 0 or not 0 <= logical_rank < self.logical_world_size:
            raise WriterModelError("invalid hierarchical logical step or rank")
        start = step * self.global_samples_per_task + logical_rank * self.samples_per_task_per_logical_rank
        rows = [self._row(task, start + offset) for task in self.task_ids
                for offset in range(self.samples_per_task_per_logical_rank)]
        order = np.random.default_rng(np.random.SeedSequence(
            [self.seed, step, logical_rank, 0xBA7C])).permutation(len(rows))
        return tuple(rows[int(index)] for index in order)

    @lru_cache(maxsize=8)
    def global_rows_for_step(self, step: int) -> tuple[int, ...]:
        return tuple(row for rank in range(self.logical_world_size)
                     for row in self.logical_batch_for_step(step, rank))

    def batch_for_step(self, step: int, micro_index: int) -> list[int]:
        if not 0 <= micro_index < self.accumulation:
            raise WriterModelError("invalid hierarchical microbatch index")
        start = sum(self.batch_sizes[:micro_index])
        rows = self.physical_rows_for_step(step)
        return list(rows[start:start + self.batch_sizes[micro_index]])

    @lru_cache(maxsize=8)
    def physical_rows_for_step(self, step: int) -> tuple[int, ...]:
        rows = self.global_rows_for_step(step)
        if self.physical_packing == "contiguous":
            return rows[self.rank_offset:self.rank_offset + sum(self.batch_sizes)]
        by_task: dict[int, list[int]] = {task: [] for task in self.task_ids}
        for row in rows:
            by_task[self.dataset.frame_index[row][0]].append(row)
        physical: list[list[int]] = [[] for _ in range(self.world_size)]
        for task_index, task in enumerate(self.task_ids):
            for offset, row in enumerate(by_task[task]):
                physical[(task_index + offset) % self.world_size].append(row)
        if tuple(map(len, physical)) != tuple(sum(sizes) for sizes in self.batch_plan):
            raise WriterModelError("task-striped physical packing changed rank query counts")
        return tuple(physical[self.rank])

    def task_counts_for_step(self, step: int) -> dict[int, int]:
        rows = self.physical_rows_for_step(step)
        frame_index = self.dataset.frame_index
        return dict(Counter(frame_index[row][0] for row in rows))

    def resume_contract(self) -> dict[str, Any]:
        record = {
            "sampler_kind": self.kind, "sampler_seed": self.seed,
            "task_ids": list(self.task_ids), "world_size": self.world_size,
            "rank": self.rank, "per_rank_batch_size": self.per_rank_batch_size,
            "gradient_accumulation_steps": self.accumulation,
            "microbatch_sizes": list(self.batch_sizes),
            "logical_world_size": self.logical_world_size,
            "logical_per_rank_batch_size": self.logical_per_rank_batch_size,
            "global_batch_size": self.global_batch_size,
        }
        if self.physical_packing != "contiguous":
            record["physical_packing"] = self.physical_packing
        return record

    def coverage_for_steps(self, start_step: int, stop_step: int) -> dict[int, tuple[int, ...]]:
        if not 0 <= start_step <= stop_step:
            raise WriterModelError("invalid hierarchical coverage range")
        return {task: tuple(sorted({self._episode_for_offset(task, offset)[0]
                    for offset in range(start_step * self.global_samples_per_task,
                                        stop_step * self.global_samples_per_task)}))
                for task in self.task_ids}

    def consumed_summary(self, start_step: int, stop_step: int) -> dict[str, Any]:
        if not 0 <= start_step <= stop_step:
            raise WriterModelError("invalid hierarchical consumed range")
        visits = stop_step - start_step
        return {
            "start_step": start_step, "stop_step": stop_step,
            "global_examples": visits * self.global_batch_size,
            "min_examples_per_task": visits * self.global_samples_per_task,
            "max_examples_per_task": visits * self.global_samples_per_task,
            "min_task_visits": visits, "max_task_visits": visits,
            "global_samples_per_task_per_step": self.global_samples_per_task,
            "global_tasks_per_update": len(self.task_ids),
        }

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.start_step, self.stop_step):
            for micro_index in range(self.accumulation):
                yield self.batch_for_step(step, micro_index)
