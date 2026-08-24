"""Exact task-local action-query sampling without cross-task mixing."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from ember.expert_manifold.contract import ExpertManifoldError


class TaskLocalEpochSampler:
    """Map each task/step to a deterministic stream of shuffled dataset rows."""

    def __init__(
        self,
        rows: Sequence[int],
        *,
        task_id: int,
        batch_size: int,
        seed: int,
    ) -> None:
        self.rows = tuple(int(value) for value in rows)
        self.task_id = int(task_id)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        if (
            not self.rows
            or len(set(self.rows)) != len(self.rows)
            or self.batch_size <= 0
            or self.task_id < 0
            or self.seed < 0
        ):
            raise ExpertManifoldError("invalid task-local expert sampler")

    @lru_cache(maxsize=8)
    def _epoch(self, epoch: int) -> tuple[int, ...]:
        if epoch < 0:
            raise ExpertManifoldError("task-local sampler epoch is negative")
        order = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.task_id, epoch, 0xE871])
        ).permutation(self.rows)
        return tuple(int(value) for value in order)

    def batch_for_step(self, step: int) -> tuple[int, ...]:
        if step < 0:
            raise ExpertManifoldError("task-local sampler step is negative")
        start = step * self.batch_size
        result: list[int] = []
        while len(result) < self.batch_size:
            epoch, offset = divmod(start, len(self.rows))
            order = self._epoch(epoch)
            take = min(self.batch_size - len(result), len(order) - offset)
            result.extend(order[offset : offset + take])
            start += take
        return tuple(result)


class BalancedTwoDomainSampler:
    """Draw an exact half-batch from each deterministic domain stream."""

    def __init__(
        self,
        domain_rows: Sequence[Sequence[int]],
        *,
        task_id: int,
        batch_size: int,
        seed: int,
    ) -> None:
        if len(domain_rows) != 2 or batch_size <= 0 or batch_size % 2:
            raise ExpertManifoldError(
                "balanced recovery sampler needs two half-batches"
            )
        half = batch_size // 2
        self.samplers = tuple(
            TaskLocalEpochSampler(
                rows,
                task_id=task_id * 2 + domain,
                batch_size=half,
                seed=seed,
            )
            for domain, rows in enumerate(domain_rows)
        )

    def batch_for_step(self, step: int) -> tuple[int, ...]:
        first, second = (sampler.batch_for_step(step) for sampler in self.samplers)
        return tuple(
            value for pair in zip(first, second, strict=True) for value in pair
        )
