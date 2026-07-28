"""Deterministic hierarchical mixed-task batches for shared Source-SFT."""

from __future__ import annotations

import hashlib
import struct
from typing import Any, Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler

from ember.writer.data import FunctionalQueryDataset
from ember.writer.model import WriterModelError


class HierarchicalMixedBatchSampler(Sampler[list[int]]):
    """Yield one exactly task-balanced mixed physical batch per optimizer step.

    Every rank receives the same number of samples from every task. Global sample
    identities are coordinated across ranks by a pure function of
    ``(step, rank, task, offset)``:

    ``uniform task -> no-replacement episode cycle -> no-replacement chunk cycle``.

    This makes the physical batch mixed-task, keeps tasks and episodes equally
    weighted, avoids cross-rank duplication within each episode cycle, and makes
    resume sample-exact without mutable sampler state.
    """

    _EPISODE_SEED_TAG = 0xE915
    _CHUNK_SEED_TAG = 0xC84A
    _BATCH_SEED_TAG = 0xBA7C

    def __init__(
        self,
        dataset: FunctionalQueryDataset,
        *,
        task_ids: Sequence[int],
        per_rank_batch_size: int,
        start_step: int,
        stop_step: int,
        rank: int,
        world_size: int,
        seed: int,
    ) -> None:
        selected_tasks = tuple(sorted(int(value) for value in task_ids))
        episode_rows = dataset.task_episode_rows
        invalid = (
            not selected_tasks
            or len(set(selected_tasks)) != len(selected_tasks)
            or per_rank_batch_size <= 0
            or per_rank_batch_size % len(selected_tasks) != 0
            or not 0 <= start_step <= stop_step
            or not 0 <= rank < world_size
            or world_size <= 0
            or seed < 0
            or set(selected_tasks) - set(episode_rows)
        )
        if invalid:
            raise WriterModelError("invalid hierarchical mixed-task sampler")
        active = {task_id: episode_rows[task_id] for task_id in selected_tasks}
        if any(
            not episodes or any(not rows for rows in episodes.values())
            for episodes in active.values()
        ):
            raise WriterModelError("hierarchical sampler found an empty episode")
        episode_counts = {len(episodes) for episodes in active.values()}
        if len(episode_counts) != 1:
            raise WriterModelError(
                "hierarchical sampler requires equal episode counts per task"
            )

        self.dataset = dataset
        self.task_ids = selected_tasks
        self.per_rank_batch_size = int(per_rank_batch_size)
        self.samples_per_task_per_rank = (
            self.per_rank_batch_size // len(self.task_ids)
        )
        self.global_samples_per_task = (
            self.samples_per_task_per_rank * int(world_size)
        )
        self.start_step = int(start_step)
        self.stop_step = int(stop_step)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.episode_rows = active
        self.episodes_per_task = episode_counts.pop()
        if (
            self.global_samples_per_task > self.episodes_per_task
            or any(
                len(rows) < 2
                for episodes in self.episode_rows.values()
                for rows in episodes.values()
            )
        ):
            raise WriterModelError(
                "hierarchical sampler cannot guarantee disjoint rank samples"
            )

    def __len__(self) -> int:
        return self.stop_step - self.start_step

    def _episode_for_global_offset(
        self, task_id: int, global_offset: int
    ) -> tuple[int, int]:
        if task_id not in self.episode_rows or global_offset < 0:
            raise WriterModelError("hierarchical episode request is invalid")
        episode_cycle, episode_offset = divmod(
            global_offset, self.episodes_per_task
        )
        declared = tuple(sorted(self.episode_rows[task_id]))
        order = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, task_id, episode_cycle, self._EPISODE_SEED_TAG]
            )
        ).permutation(declared)
        return int(order[episode_offset]), episode_cycle

    def _row_for_global_offset(self, task_id: int, global_offset: int) -> int:
        demo_index, episode_visit = self._episode_for_global_offset(
            task_id, global_offset
        )
        rows = self.episode_rows[task_id][demo_index]
        _, chunk_offset = divmod(episode_visit, len(rows))
        order = np.random.default_rng(
            np.random.SeedSequence(
                [
                    self.seed,
                    task_id,
                    demo_index,
                    self._CHUNK_SEED_TAG,
                ]
            )
        ).permutation(rows)
        return int(order[chunk_offset])

    def row_for(
        self,
        *,
        task_id: int,
        step: int,
        rank: int,
        task_offset: int,
    ) -> int:
        """Return one globally coordinated row identity."""

        if (
            step < 0
            or not 0 <= rank < self.world_size
            or not 0 <= task_offset < self.samples_per_task_per_rank
        ):
            raise WriterModelError("hierarchical sample request is invalid")
        global_offset = (
            step * self.global_samples_per_task
            + rank * self.samples_per_task_per_rank
            + task_offset
        )
        return self._row_for_global_offset(task_id, global_offset)

    def batch_for_step(
        self, step: int, *, rank: int | None = None
    ) -> tuple[int, ...]:
        """Return one mixed physical batch in deterministic shuffled order."""

        selected_rank = self.rank if rank is None else int(rank)
        if step < 0 or not 0 <= selected_rank < self.world_size:
            raise WriterModelError("hierarchical batch request is invalid")
        rows = [
            self.row_for(
                task_id=task_id,
                step=step,
                rank=selected_rank,
                task_offset=task_offset,
            )
            for task_id in self.task_ids
            for task_offset in range(self.samples_per_task_per_rank)
        ]
        order = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, step, selected_rank, self._BATCH_SEED_TAG]
            )
        ).permutation(len(rows))
        result = tuple(rows[int(index)] for index in order)
        if len(result) != self.per_rank_batch_size:
            raise WriterModelError("hierarchical physical batch changed size")
        return result

    def coverage_for_steps(
        self, start_step: int, stop_step: int
    ) -> dict[int, tuple[int, ...]]:
        """Return exact cross-rank episode coverage for a half-open step range."""

        if not 0 <= start_step <= stop_step:
            raise WriterModelError("invalid hierarchical coverage range")
        coverage = {task_id: set() for task_id in self.task_ids}
        for step in range(start_step, stop_step):
            for task_id in self.task_ids:
                for rank in range(self.world_size):
                    for task_offset in range(self.samples_per_task_per_rank):
                        global_offset = (
                            step * self.global_samples_per_task
                            + rank * self.samples_per_task_per_rank
                            + task_offset
                        )
                        demo_index, _ = self._episode_for_global_offset(
                            task_id, global_offset
                        )
                        coverage[task_id].add(demo_index)
        return {
            task_id: tuple(sorted(demo_indices))
            for task_id, demo_indices in coverage.items()
        }

    def consumed_identity_summary(
        self, start_step: int, stop_step: int
    ) -> dict[str, Any]:
        """Digest the exact cross-rank task/episode/chunk schedule."""

        if not 0 <= start_step <= stop_step:
            raise WriterModelError("invalid hierarchical consumed range")
        digest = hashlib.sha256()
        frame_index = self.dataset.frame_index
        unique_rows: set[int] = set()
        task_examples = {task_id: 0 for task_id in self.task_ids}
        for step in range(start_step, stop_step):
            for rank in range(self.world_size):
                for batch_offset, row in enumerate(
                    self.batch_for_step(step, rank=rank)
                ):
                    task_id, demo_index, frame = frame_index[row]
                    if task_id not in task_examples:
                        raise WriterModelError(
                            "hierarchical query crossed task authority"
                        )
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
            "global_examples": (
                (stop_step - start_step)
                * self.world_size
                * self.per_rank_batch_size
            ),
            "unique_query_rows": len(unique_rows),
            "min_examples_per_task": min(counts),
            "max_examples_per_task": max(counts),
            "samples_per_task_per_rank_per_step": self.samples_per_task_per_rank,
            "global_samples_per_task_per_step": self.global_samples_per_task,
            "identity_sha256": digest.hexdigest(),
        }

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.start_step, self.stop_step):
            yield list(self.batch_for_step(step))
