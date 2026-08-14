"""Deterministic dynamic-K teacher-video selection for AS-Writer."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ember.writer.errors import WriterModelError


class TeacherVideoSchedule:
    """Deterministic primary videos plus optional training-only companions."""

    _SEED_TAG = 0x71DE0
    _DYNAMIC_K_SEED_TAG = 0xD14A4C

    def __init__(
        self,
        *,
        task_ids: Sequence[int],
        demo_indices: Sequence[int],
        seed: int,
        videos_per_visit: int = 4,
        companion_videos_per_visit: int = 0,
        dynamic_k_max: int | None = None,
    ) -> None:
        if (
            not task_ids
            or len(set(task_ids)) != len(task_ids)
            or not demo_indices
            or len(set(demo_indices)) != len(demo_indices)
            or seed < 0
            or not 1 <= videos_per_visit <= len(demo_indices)
            or companion_videos_per_visit < 0
            or companion_videos_per_visit >= len(demo_indices)
            or (companion_videos_per_visit and videos_per_visit != 1)
            or dynamic_k_max not in {None, 4}
            or (dynamic_k_max is not None and videos_per_visit != dynamic_k_max)
            or (dynamic_k_max is not None and companion_videos_per_visit != 0)
            or (dynamic_k_max is not None and len(task_ids) % dynamic_k_max)
        ):
            raise WriterModelError("invalid teacher-video schedule")
        self.task_ids = tuple(sorted(int(value) for value in task_ids))
        self.demo_indices = tuple(sorted(int(value) for value in demo_indices))
        self.seed = int(seed)
        self.videos_per_visit = int(videos_per_visit)
        self.companion_videos_per_visit = int(companion_videos_per_visit)
        self.dynamic_k_max = dynamic_k_max
        dynamic_order = np.random.default_rng(
            np.random.SeedSequence([self.seed, self._DYNAMIC_K_SEED_TAG])
        ).permutation(self.task_ids)
        self._dynamic_k_position = {
            int(task_id): position for position, task_id in enumerate(dynamic_order)
        }

    def shot_count_for_task_visit(self, task_id: int, task_visit: int) -> int:
        """Return the sealed K for one task and macro visit."""

        if task_id not in self.task_ids or task_visit < 0:
            raise WriterModelError("teacher-video request is outside the schedule")
        if self.dynamic_k_max is None:
            return self.videos_per_visit
        return 1 + (
            (self._dynamic_k_position[task_id] + task_visit) % self.dynamic_k_max
        )

    def _one_shot_order(self, task_id: int, task_visit: int) -> tuple[int, ...]:
        cycle = task_visit // len(self.demo_indices)
        return tuple(
            int(value)
            for value in np.random.default_rng(
                np.random.SeedSequence(
                    [self.seed, task_id, cycle, self._SEED_TAG]
                )
            ).permutation(self.demo_indices)
        )

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
        count = self.shot_count_for_task_visit(task_id, task_visit)
        if count == 1 and self.dynamic_k_max is None:
            offset = task_visit % len(self.demo_indices)
            selected = self._one_shot_order(task_id, task_visit)[offset]
            if selected in excluded_set:
                raise WriterModelError(
                    "one-shot teacher video overlapped an action query"
                )
            return (selected,)
        candidates = [
            int(value) for value in self.demo_indices if int(value) not in excluded_set
        ]
        if len(candidates) < count:
            raise WriterModelError("action queries leave too few independent videos")
        order = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, task_id, task_visit, self._SEED_TAG]
            )
        ).permutation(candidates)
        return tuple(int(value) for value in order[:count])

    def cross_video_credit_demos_for_task_visit(
        self,
        task_id: int,
        task_visit: int,
        anchor_demos: Sequence[int],
        *,
        view_count: int = 4,
    ) -> tuple[tuple[int, ...], ...]:
        """Return one anchor plus three disjoint same-task K4 credit sets."""

        first = tuple(int(value) for value in anchor_demos)
        if view_count != 4 or len(first) != 4 or len(set(first)) != 4:
            raise WriterModelError("CV-CSD anchor condition is not one unique K4 set")
        views = [first]
        excluded = list(first)
        for offset in range(1, view_count):
            demos = self.demos_for_task_visit(
                task_id, task_visit + offset, excluded=excluded
            )
            if len(demos) != 4:
                raise WriterModelError("CV-CSD support condition is not K4")
            views.append(tuple(demos))
            excluded.extend(demos)
        if len(set(excluded)) != 4 * view_count:
            raise WriterModelError("CV-CSD credit conditions are not disjoint")
        return tuple(views)

    def companion_demos_for_task_visit(
        self,
        task_id: int,
        task_visit: int,
        *,
        excluded: Sequence[int] = (),
    ) -> tuple[int, ...]:
        """Take subsequent legal demos from the unchanged one-shot cycle."""

        if task_id not in self.task_ids or task_visit < 0:
            raise WriterModelError("companion-video request is outside the schedule")
        if not self.companion_videos_per_visit:
            return ()
        order = self._one_shot_order(task_id, task_visit)
        primary_offset = task_visit % len(order)
        primary = order[primary_offset]
        excluded_set = {primary, *(int(value) for value in excluded)}
        selected = []
        for distance in range(1, len(order)):
            candidate = order[(primary_offset + distance) % len(order)]
            if candidate not in excluded_set:
                selected.append(candidate)
                if len(selected) == self.companion_videos_per_visit:
                    return tuple(selected)
        raise WriterModelError("action queries leave too few companion videos")

    def training_demos_for_task_visit(
        self,
        task_id: int,
        task_visit: int,
        *,
        excluded: Sequence[int] = (),
    ) -> tuple[int, ...]:
        return self.demos_for_task_visit(
            task_id, task_visit, excluded=excluded
        ) + self.companion_demos_for_task_visit(
            task_id, task_visit, excluded=excluded
        )

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
        sampler: Any,
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
        primary_coverage = {task_id: set() for task_id in self.task_ids}
        companion_coverage = {task_id: set() for task_id in self.task_ids}
        visits = {task_id: 0 for task_id in self.task_ids}
        for step in range(start_step, stop_step):
            for _, _, task_id, task_visit in sampler.assignments_for_step(step):
                excluded = sampler.action_demo_indices_for_task_visit(
                    task_id, task_visit
                )
                demos = self.demos_for_task_visit(
                    task_id,
                    task_visit,
                    excluded=excluded,
                )
                companions = self.companion_demos_for_task_visit(
                    task_id, task_visit, excluded=excluded
                )
                primary_coverage[task_id].update(demos)
                companion_coverage[task_id].update(companions)
                visits[task_id] += 1
        visit_counts = tuple(visits.values())
        primary_counts = tuple(len(value) for value in primary_coverage.values())
        companion_counts = tuple(len(value) for value in companion_coverage.values())
        shot_counts = tuple(
            len(
                self.demos_for_task_visit(
                    task_id,
                    visit,
                    excluded=sampler.action_demo_indices_for_task_visit(
                        task_id, visit
                    ),
                )
            )
            for step in range(start_step, stop_step)
            for _, _, task_id, visit in sampler.assignments_for_step(step)
        )
        return {
            "query": query,
            "teacher_video_seed": self.seed,
            "videos_per_task_visit": self.videos_per_visit,
            "dynamic_k_max": self.dynamic_k_max,
            "min_videos_per_task_visit": min(shot_counts, default=0),
            "max_videos_per_task_visit": max(shot_counts, default=0),
            "training_companion_videos_per_task_visit": (
                self.companion_videos_per_visit
            ),
            "min_video_visits_per_task": min(visit_counts),
            "max_video_visits_per_task": max(visit_counts),
            "min_unique_videos_per_task": min(primary_counts),
            "max_unique_videos_per_task": max(primary_counts),
            "min_unique_companion_videos_per_task": min(companion_counts),
            "max_unique_companion_videos_per_task": max(companion_counts),
        }
