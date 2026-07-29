"""Exact task, action-query, and teacher-video schedules for AS-Writer."""

from __future__ import annotations

import hashlib
import struct
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from ember.writer.data import FunctionalQueryDataset
from ember.writer.model import WriterModelError


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
        or not (
            len(task_ids) == world_size * tasks_per_rank_per_update
            or (
                tasks_per_rank_per_update == 1
                and len(task_ids) % world_size == 0
            )
        )
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
    """Yield task-pure batches under one explicit multi-task update topology.

    Task-complete updates contain every task once globally and use cost-balanced
    long-first rank groups.  The historical update recipe instead assigns one
    task to each rank and rotates a seeded permutation of all tasks across
    consecutive global rank slots.  Both modes retain task-pure physical batches
    and exact task-visit cursors.
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
        self.update_topology = (
            "task_complete_all_tasks"
            if len(self.task_ids)
            == self.world_size * self.tasks_per_rank_per_update
            else "rank_rotating_one_task_per_rank"
        )
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
        if self.update_topology != "task_complete_all_tasks":
            raise WriterModelError(
                "cost-balanced groups require task-complete updates"
            )
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

    def _task_visit_for_global_slot(self, slot: int) -> tuple[int, int]:
        if slot < 0:
            raise WriterModelError("global task slot must be non-negative")
        task_visit, offset = divmod(slot, len(self.task_ids))
        order = np.random.default_rng(
            np.random.SeedSequence([self.seed, task_visit])
        ).permutation(self.task_ids)
        return int(order[offset]), task_visit

    def assignments_for_step(
        self,
        step: int,
    ) -> tuple[tuple[int, int, int, int], ...]:
        """Return ``(rank, microtask, task_id, task_visit)`` assignments."""

        if step < 0:
            raise WriterModelError("task assignment step must be non-negative")
        if self.update_topology == "rank_rotating_one_task_per_rank":
            return tuple(
                (
                    rank,
                    0,
                    *self._task_visit_for_global_slot(
                        step * self.world_size + rank
                    ),
                )
                for rank in range(self.world_size)
            )
        groups = self._cost_balanced_groups(step)
        assignments: list[tuple[int, int, int, int]] = []
        for rank in range(self.world_size):
            group = (rank + step) % self.world_size
            assignments.extend(
                (rank, microtask, task_id, step)
                for microtask, task_id in enumerate(groups[group])
            )
        return tuple(assignments)

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
        return tuple(
            task_id
            for rank, _, task_id, _ in self.assignments_for_step(step)
            if rank == selected_rank
        )

    def task_visit_for_step(self, step: int, microtask: int) -> tuple[int, int]:
        """Return one task and its visit index inside a macro update."""

        if (
            not self.start_step <= step < self.stop_step
            or not 0 <= microtask < self.tasks_per_rank_per_update
        ):
            raise WriterModelError("task-complete microtask is outside the sampler")
        matches = tuple(
            (task_id, task_visit)
            for rank, selected_microtask, task_id, task_visit
            in self.assignments_for_step(step)
            if rank == self.rank and selected_microtask == microtask
        )
        if len(matches) != 1:
            raise WriterModelError("task assignment is not unique")
        return matches[0]

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
            for _, _, task_id, task_visit in self.assignments_for_step(step):
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
            for (
                rank,
                microtask,
                task_id,
                task_visit,
            ) in self.assignments_for_step(step):
                for batch_offset in range(self.batch_size_for_step(step)):
                    row = self._sample_for_task_visit(
                        task_id, task_visit, batch_offset
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
                * self.world_size
                * self.tasks_per_rank_per_update
                * self.per_rank_batch_size
            ),
            "unique_query_rows": len(unique_rows),
            "min_examples_per_task": min(counts),
            "max_examples_per_task": max(counts),
            "identity_sha256": digest.hexdigest(),
        }

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.start_step, self.stop_step):
            for (
                rank,
                _,
                task_id,
                task_visit,
            ) in self.assignments_for_step(step):
                if rank != self.rank:
                    continue
                yield [
                    self._sample_for_task_visit(
                        task_id, task_visit, batch_offset
                    )
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
            for (
                rank,
                microtask,
                task_id,
                task_visit,
            ) in sampler.assignments_for_step(step):
                demo_index = self.demo_for_task_visit(task_id, task_visit)
                digest.update(
                    struct.pack(
                        ">6q",
                        step,
                        rank,
                        microtask,
                        task_id,
                        task_visit,
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
