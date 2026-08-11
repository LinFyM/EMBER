"""Exact task, action-query, and teacher-video schedules for AS-Writer."""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from ember.writer.data import FunctionalQueryDataset
from ember.writer.errors import WriterModelError


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
    optimizer_updates_per_task_cycle: int,
    dynamic_task_assignment: bool = False,
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
        or optimizer_updates_per_task_cycle <= 0
        or (
            not dynamic_task_assignment
            and len(task_ids)
            != (
                world_size
                * tasks_per_rank_per_update
                * optimizer_updates_per_task_cycle
            )
        )
        or (
            dynamic_task_assignment
            and (
                optimizer_updates_per_task_cycle != 1
                or world_size > len(task_ids)
                or tasks_per_rank_per_update
                != math.ceil(len(task_ids) / world_size)
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
    """Yield task-pure batches from one sealed complete-task assignment."""

    _GROUP_SEED_TAG = 0xC057
    _LATIN_GROUP_SEED_TAG = 0xC14C4
    _LATIN_TAIL_SEED_TAG = 0x7A11
    _LATIN_FORMAL_TASK_CYCLES = 200
    _LATIN_FULL_CYCLES = 198
    _PHASE_STRATUM_SEED_TAG = 0x57A7
    _PHASE_JITTER_SEED_TAG = 0x9177

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
        optimizer_updates_per_task_cycle: int = 1,
        video_schedule: TeacherVideoSchedule,
        task_video_costs: Mapping[int, Mapping[int, int]],
        assignment_strategy: str = "cost_balanced_long_first",
    ) -> None:
        dynamic_task_assignment = (
            assignment_strategy == "cost_balanced_long_first_dynamic_uneven"
        )
        batch_cycle = _task_complete_batch_cycle(
            task_ids=task_ids,
            per_rank_batch_size=per_rank_batch_size,
            per_rank_batch_cycle=per_rank_batch_cycle,
            start_step=start_step,
            stop_step=stop_step,
            rank=rank,
            world_size=world_size,
            tasks_per_rank_per_update=tasks_per_rank_per_update,
            optimizer_updates_per_task_cycle=optimizer_updates_per_task_cycle,
            dynamic_task_assignment=dynamic_task_assignment,
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
        self.optimizer_updates_per_task_cycle = int(
            optimizer_updates_per_task_cycle
        )
        self.assignment_strategy = str(assignment_strategy)
        self.dynamic_task_assignment = dynamic_task_assignment
        self.tasks_per_rank_per_cycle = (
            self.tasks_per_rank_per_update
            * self.optimizer_updates_per_task_cycle
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
        if self.assignment_strategy not in {
            "cost_balanced_long_first",
            "cost_balanced_long_first_dynamic_uneven",
            "randomized_latin_group4",
        }:
            raise WriterModelError("unsupported task assignment strategy")
        if self.assignment_strategy == "randomized_latin_group4" and (
            self.world_size != 4
            or self.tasks_per_rank_per_update != 1
            or self.optimizer_updates_per_task_cycle != 6
            or len(self.task_ids) != 24
            or self.stop_step
            > self._LATIN_FORMAL_TASK_CYCLES
            * self.optimizer_updates_per_task_cycle
        ):
            raise WriterModelError("invalid randomized Latin group4 sampler")
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
        if self.dynamic_task_assignment:
            return sum(
                len(self.tasks_for_step(step))
                for step in range(self.start_step, self.stop_step)
            )
        return (
            self.stop_step - self.start_step
        ) * self.tasks_per_rank_per_update

    def task_cycle_and_phase(self, step: int) -> tuple[int, int]:
        """Map one optimizer update to its complete-task cycle and phase."""

        if step < 0:
            raise WriterModelError("task assignment step must be non-negative")
        return divmod(step, self.optimizer_updates_per_task_cycle)

    def _cost_order_for_task_cycle(
        self, task_cycle: int
    ) -> tuple[dict[int, int], dict[int, int], tuple[int, ...]]:
        current_costs = {
            task_id: sum(
                self.task_video_costs[task_id][demo]
                for demo in self.video_schedule.demos_for_task_visit(
                    task_id,
                    task_cycle,
                    excluded=self.action_demo_indices_for_task_visit(
                        task_id, task_cycle
                    ),
                )
            )
            for task_id in self.task_ids
        }
        tie_order = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, task_cycle, self._GROUP_SEED_TAG]
            )
        ).permutation(self.task_ids)
        tie_rank = {int(task_id): index for index, task_id in enumerate(tie_order)}
        ordered = tuple(
            sorted(
                self.task_ids,
                key=lambda task_id: (-current_costs[task_id], tie_rank[task_id]),
            )
        )
        return current_costs, tie_rank, ordered

    def _cost_balanced_groups(self, task_cycle: int) -> tuple[tuple[int, ...], ...]:
        current_costs, tie_rank, ordered = self._cost_order_for_task_cycle(
            task_cycle
        )
        groups: list[list[int]] = [[] for _ in range(self.world_size)]
        loads = [0] * self.world_size
        rank_offset = (self.seed + task_cycle) % self.world_size
        for task_id in ordered:
            candidates = [
                rank
                for rank, group in enumerate(groups)
                if len(group) < self.tasks_per_rank_per_cycle
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
            any(len(group) != self.tasks_per_rank_per_cycle for group in result)
            or set().union(*(set(group) for group in result))
            != set(self.task_ids)
        ):
            raise WriterModelError("task-complete cost balancing failed")
        return result

    def _dynamic_cost_balanced_assignments(
        self, task_cycle: int
    ) -> tuple[tuple[int, int, int, int], ...]:
        """Assign train tasks once with deterministic unequal rank capacities."""

        current_costs, tie_rank, ordered = self._cost_order_for_task_cycle(
            task_cycle
        )
        base, remainder = divmod(len(self.task_ids), self.world_size)
        rank_offset = (self.seed + task_cycle) % self.world_size
        capacities = [base] * self.world_size
        for offset in range(remainder):
            capacities[(rank_offset + offset) % self.world_size] += 1
        groups: list[list[int]] = [[] for _ in range(self.world_size)]
        loads = [0] * self.world_size
        for task_id in ordered:
            candidates = [
                rank
                for rank, group in enumerate(groups)
                if len(group) < capacities[rank]
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
        assignments = tuple(
            (rank, microtask, task_id, task_cycle)
            for rank, group in enumerate(groups)
            for microtask, task_id in enumerate(
                sorted(
                    group,
                    key=lambda value: (-current_costs[value], tie_rank[value]),
                )
            )
        )
        if (
            len(assignments) != len(self.task_ids)
            or {task_id for _, _, task_id, _ in assignments} != set(self.task_ids)
            or tuple(
                sum(rank == selected for rank, _, _, _ in assignments)
                for selected in range(self.world_size)
            )
            != tuple(capacities)
        ):
            raise WriterModelError("dynamic task-complete cost balancing failed")
        return assignments

    def _latin_task_matrix(
        self, superblock: int, *, independent_tail: bool = False
    ) -> np.ndarray:
        """Return the pre-registered independent 6x4 task matrix."""

        if superblock < 0:
            raise WriterModelError("Latin superblock must be non-negative")
        entropy = [self.seed, superblock, self._LATIN_GROUP_SEED_TAG]
        if independent_tail:
            entropy.append(self._LATIN_TAIL_SEED_TAG)
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        return rng.permutation(self.task_ids).reshape(6, 4, order="C")

    def _randomized_latin_assignments(
        self,
        task_cycle: int,
        phase: int,
    ) -> tuple[tuple[int, int, int, int], ...]:
        """Assign one randomized, phase-balanced task to every rank."""

        if not 0 <= task_cycle < self._LATIN_FORMAL_TASK_CYCLES:
            raise WriterModelError("Latin task cycle is outside the sealed cell")
        if not 0 <= phase < 6:
            raise WriterModelError("Latin task phase is outside the sealed cell")

        if task_cycle < self._LATIN_FULL_CYCLES:
            superblock, row_cycle = divmod(task_cycle, 6)
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [self.seed, superblock, self._LATIN_GROUP_SEED_TAG]
                )
            )
            matrix = rng.permutation(self.task_ids).reshape(6, 4, order="C")
            positive_columns = {
                int(value) for value in rng.permutation(4)[:2]
            }

            def assigned_phase(row: int, column: int) -> int:
                sign = 1 if column in positive_columns else -1
                return (row + sign * row_cycle) % 6

        else:
            superblock = self._LATIN_FULL_CYCLES // 6
            matrix = self._latin_task_matrix(
                superblock,
                independent_tail=True,
            )
            reverse = task_cycle == self._LATIN_FULL_CYCLES + 1

            def assigned_phase(row: int, column: int) -> int:
                del column
                return 5 - row if reverse else row

        assignments = []
        for row in range(6):
            for column in range(4):
                if assigned_phase(row, column) != phase:
                    continue
                rank = (
                    column + task_cycle + (superblock % self.world_size)
                ) % self.world_size
                assignments.append(
                    (rank, 0, int(matrix[row, column]), task_cycle)
                )
        assignments.sort()
        if (
            len(assignments) != self.world_size
            or {rank for rank, _, _, _ in assignments}
            != set(range(self.world_size))
            or len({task_id for _, _, task_id, _ in assignments})
            != self.world_size
        ):
            raise WriterModelError("randomized Latin group4 assignment failed")
        return tuple(assignments)

    def assignments_for_step(
        self,
        step: int,
    ) -> tuple[tuple[int, int, int, int], ...]:
        """Return ``(rank, microtask, task_id, task_visit)`` assignments."""

        task_cycle, phase = self.task_cycle_and_phase(step)
        if self.assignment_strategy == "randomized_latin_group4":
            return self._randomized_latin_assignments(task_cycle, phase)
        if self.dynamic_task_assignment:
            if phase != 0:
                raise WriterModelError("dynamic task assignment gained a phase")
            return self._dynamic_cost_balanced_assignments(task_cycle)
        groups = self._cost_balanced_groups(task_cycle)
        assignments: list[tuple[int, int, int, int]] = []
        for rank in range(self.world_size):
            group = (rank + task_cycle) % self.world_size
            left = phase * self.tasks_per_rank_per_update
            right = left + self.tasks_per_rank_per_update
            assignments.extend(
                (rank, microtask, task_id, task_cycle)
                for microtask, task_id in enumerate(groups[group][left:right])
            )
        return tuple(assignments)

    def tasks_for_step(
        self,
        step: int,
        *,
        rank: int | None = None,
    ) -> tuple[int, ...]:
        """Return long-first selected tasks for one rank and optimizer update."""

        selected_rank = self.rank if rank is None else rank
        if step < 0 or not 0 <= selected_rank < self.world_size:
            raise WriterModelError("task-complete step or rank is invalid")
        return tuple(
            task_id
            for rank, _, task_id, _ in self.assignments_for_step(step)
            if rank == selected_rank
        )

    def task_queue_for_step(self, step: int) -> tuple[tuple[int, int], ...]:
        """Return the deterministic long-first jobs before physical rank ownership."""

        if not self.start_step <= step < self.stop_step:
            raise WriterModelError("task queue step is outside the sampler")
        task_cycle, phase = self.task_cycle_and_phase(step)
        if not self.dynamic_task_assignment or phase != 0:
            raise WriterModelError("task queue requires complete-task dynamic sampling")
        _, _, ordered = self._cost_order_for_task_cycle(task_cycle)
        return tuple((task_id, task_cycle) for task_id in ordered)

    def task_visit_for_step(self, step: int, microtask: int) -> tuple[int, int]:
        """Return one task and its visit index inside a macro update."""

        if (
            not self.start_step <= step < self.stop_step
            or not 0 <= microtask < len(self.tasks_for_step(step))
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
        if self.video_schedule.videos_per_visit == 1:
            teacher_demo = self.video_schedule.demos_for_task_visit(
                task_id,
                task_visit,
            )[0]
            selected: list[tuple[int, int]] = []
            cursor = task_visit * self.per_rank_batch_size
            while len(selected) <= batch_offset:
                episode_cycle, episode_offset = divmod(
                    cursor, self.episodes_per_task
                )
                demo_index = self.episode_orders[task_id][episode_offset]
                if demo_index != teacher_demo:
                    selected.append((demo_index, episode_cycle))
                cursor += 1
            return selected[batch_offset]
        position = task_visit * self.per_rank_batch_size + batch_offset
        episode_cycle, episode_offset = divmod(position, self.episodes_per_task)
        demo_index = self.episode_orders[task_id][episode_offset]
        return demo_index, episode_cycle

    def action_demo_indices_for_task_visit(
        self,
        task_id: int,
        task_visit: int,
    ) -> tuple[int, ...]:
        """Return the exact action-query episodes excluded from K-shot video use."""

        if task_id not in self.task_ids or task_visit < 0:
            raise WriterModelError("action-demo request is outside the sampler")
        return tuple(
            sorted(
                {
                    self._episode_for_task_visit(
                        task_id, task_visit, batch_offset
                    )[0]
                    for batch_offset in range(self.per_rank_batch_size)
                }
            )
        )

    def _phase_strata_for_task_visit(
        self, task_id: int, task_visit: int
    ) -> tuple[int, ...]:
        """Assign every normalized episode-progress stratum exactly once."""

        return tuple(
            int(value)
            for value in np.random.default_rng(
                np.random.SeedSequence(
                    [
                        self.seed,
                        task_id,
                        task_visit,
                        self._PHASE_STRATUM_SEED_TAG,
                    ]
                )
            ).permutation(self.per_rank_batch_size)
        )

    def _sample_for_task_visit(
        self,
        task_id: int,
        task_visit: int,
        batch_offset: int,
        *,
        phase_stratum: int | None = None,
    ) -> int:
        demo_index, _ = self._episode_for_task_visit(
            task_id, task_visit, batch_offset
        )
        rows = self.episode_rows[task_id][demo_index]
        stratum = (
            self._phase_strata_for_task_visit(task_id, task_visit)[batch_offset]
            if phase_stratum is None
            else phase_stratum
        )
        jitter = float(
            np.random.default_rng(
                np.random.SeedSequence(
                    [
                        self.seed,
                        task_id,
                        task_visit,
                        stratum,
                        self._PHASE_JITTER_SEED_TAG,
                    ]
                )
            ).random()
        )
        phase = (stratum + jitter) / self.per_rank_batch_size
        row_offset = min(int(np.floor(phase * len(rows))), len(rows) - 1)
        return rows[row_offset]

    def batch_indices_for_task_visit(
        self,
        step: int,
        task_id: int,
        task_visit: int,
    ) -> tuple[int, ...]:
        """Return the exact task-pure row indices independent of physical rank."""

        if not self.start_step <= step < self.stop_step:
            raise WriterModelError("task-addressable batch step is outside sampler")
        jobs = {
            selected_task: selected_visit
            for _, _, selected_task, selected_visit in self.assignments_for_step(step)
        }
        if jobs.get(task_id) != task_visit:
            raise WriterModelError("task-addressable batch is outside the schedule")
        phase_strata = self._phase_strata_for_task_visit(task_id, task_visit)
        return tuple(
            self._sample_for_task_visit(
                task_id,
                task_visit,
                batch_offset,
                phase_stratum=phase_strata[batch_offset],
            )
            for batch_offset in range(self.batch_size_for_step(step))
        )

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
                phase_strata = self._phase_strata_for_task_visit(
                    task_id, task_visit
                )
                for batch_offset in range(self.batch_size_for_step(step)):
                    row = self._sample_for_task_visit(
                        task_id,
                        task_visit,
                        batch_offset,
                        phase_stratum=phase_strata[batch_offset],
                    )
                    row_task, demo_index, frame = frame_index[row]
                    if row_task != task_id:
                        raise WriterModelError(
                            "sampler query crossed task authority"
                        )
                    unique_rows.add(row)
                    task_examples[task_id] += 1
        counts = tuple(task_examples.values())
        return {
            "start_step": start_step,
            "stop_step": stop_step,
            "global_examples": (
                (stop_step - start_step)
                * (
                    len(self.task_ids)
                    if self.dynamic_task_assignment
                    else self.world_size * self.tasks_per_rank_per_update
                )
                * self.per_rank_batch_size
            ),
            "unique_query_rows": len(unique_rows),
            "min_examples_per_task": min(counts),
            "max_examples_per_task": max(counts),
            "identity_evidence": "cursor_counts_and_dataset_row_coverage",
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
                yield list(
                    self.batch_indices_for_task_visit(step, task_id, task_visit)
                )


class TeacherVideoSchedule:
    """Deterministic K-video schedule with no replacement inside each set."""

    _SEED_TAG = 0x71DE0

    def __init__(
        self,
        *,
        task_ids: Sequence[int],
        demo_indices: Sequence[int],
        seed: int,
        videos_per_visit: int = 4,
    ) -> None:
        if (
            not task_ids
            or len(set(task_ids)) != len(task_ids)
            or not demo_indices
            or len(set(demo_indices)) != len(demo_indices)
            or seed < 0
            or not 1 <= videos_per_visit <= len(demo_indices)
        ):
            raise WriterModelError("invalid teacher-video schedule")
        self.task_ids = tuple(sorted(int(value) for value in task_ids))
        self.demo_indices = tuple(sorted(int(value) for value in demo_indices))
        self.seed = int(seed)
        self.videos_per_visit = int(videos_per_visit)

    def demos_for_task_visit(
        self,
        task_id: int,
        task_visit: int,
        *,
        excluded: Sequence[int] = (),
    ) -> tuple[int, ...]:
        if task_id not in self.task_ids or task_visit < 0:
            raise WriterModelError("teacher-video request is outside the schedule")
        excluded_set = {int(value) for value in excluded}
        if self.videos_per_visit == 1:
            cycle, offset = divmod(task_visit, len(self.demo_indices))
            order = np.random.default_rng(
                np.random.SeedSequence(
                    [self.seed, task_id, cycle, self._SEED_TAG]
                )
            ).permutation(self.demo_indices)
            selected = int(order[offset])
            if selected in excluded_set:
                raise WriterModelError(
                    "one-shot teacher video overlapped an action query"
                )
            return (selected,)
        candidates = [
            int(value) for value in self.demo_indices if int(value) not in excluded_set
        ]
        if len(candidates) < self.videos_per_visit:
            raise WriterModelError("action queries leave too few independent videos")
        order = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, task_id, task_visit, self._SEED_TAG]
            )
        ).permutation(candidates)
        return tuple(int(value) for value in order[: self.videos_per_visit])

    def identity_for_task_visits(
        self, task_id: int, start_visit: int, stop_visit: int
    ) -> dict[str, Any]:
        if not 0 <= start_visit <= stop_visit:
            raise WriterModelError("invalid teacher-video visit range")
        demo_sets = tuple(
            self.demos_for_task_visit(task_id, visit)
            for visit in range(start_visit, stop_visit)
        )
        return {
            "task_id": task_id,
            "start_visit": start_visit,
            "stop_visit": stop_visit,
            "demo_sets": demo_sets,
            "unique_demo_indices": tuple(
                sorted({demo for demos in demo_sets for demo in demos})
            ),
        }

    def consumed_identity_summary(
        self,
        sampler: MixedTaskBatchSampler,
        start_step: int,
        stop_step: int,
    ) -> dict[str, Any]:
        """Summarize action queries and K videos used by every task visit."""

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
        coverage = {task_id: set() for task_id in self.task_ids}
        visits = {task_id: 0 for task_id in self.task_ids}
        for step in range(start_step, stop_step):
            for (
                rank,
                microtask,
                task_id,
                task_visit,
            ) in sampler.assignments_for_step(step):
                demos = self.demos_for_task_visit(
                    task_id,
                    task_visit,
                    excluded=sampler.action_demo_indices_for_task_visit(
                        task_id, task_visit
                    ),
                )
                coverage[task_id].update(demos)
                visits[task_id] += 1
        visit_counts = tuple(visits.values())
        video_counts = tuple(len(value) for value in coverage.values())
        return {
            "query": query,
            "teacher_video_seed": self.seed,
            "videos_per_task_visit": self.videos_per_visit,
            "min_video_visits_per_task": min(visit_counts),
            "max_video_visits_per_task": max(visit_counts),
            "min_unique_videos_per_task": min(video_counts),
            "max_unique_videos_per_task": max(video_counts),
        }
