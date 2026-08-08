from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pytest

from ember.source_sft.sampler import (
    CyclicSubsetMixedBatchSampler,
)
from ember.writer.errors import WriterModelError


@dataclass
class _DatasetStub:
    task_episode_rows: dict[int, dict[int, tuple[int, ...]]]
    frame_index: tuple[tuple[int, int, int], ...]


def _subset_dataset(
    task_ids: tuple[int, ...] = tuple(range(10, 90, 10)),
) -> _DatasetStub:
    episode_rows: dict[int, dict[int, tuple[int, ...]]] = {}
    frame_index: list[tuple[int, int, int]] = []
    flat = 0
    for task_id in task_ids:
        episode_rows[task_id] = {}
        for demo_index in range(5):
            rows = tuple(range(flat, flat + 3))
            episode_rows[task_id][demo_index] = rows
            frame_index.extend(
                (task_id, demo_index, frame) for frame in range(3)
            )
            flat += 3
    return _DatasetStub(episode_rows, tuple(frame_index))


def _subset_sampler(
    *,
    rank: int,
    start_step: int = 0,
    stop_step: int = 4,
) -> CyclicSubsetMixedBatchSampler:
    return CyclicSubsetMixedBatchSampler(
        _subset_dataset(),  # type: ignore[arg-type]
        task_ids=tuple(range(10, 90, 10)),
        per_rank_batch_size=8,
        tasks_per_rank_per_update=2,
        start_step=start_step,
        stop_step=stop_step,
        rank=rank,
        world_size=2,
        seed=20260728,
    )


def test_cyclic_subset_mixes_each_rank_and_completes_every_task_cycle() -> None:
    dataset = _subset_dataset()
    samplers = [_subset_sampler(rank=rank) for rank in range(2)]
    for cycle in range(2):
        selected_tasks = []
        for phase in range(2):
            step = cycle * 2 + phase
            rank_task_sets = []
            rank_rows = []
            for sampler in samplers:
                batch = sampler.batch_for_step(step)
                counts = Counter(dataset.frame_index[row][0] for row in batch)
                assert len(counts) == 2
                assert set(counts.values()) == {4}
                rank_task_sets.append(set(counts))
                rank_rows.append(set(batch))
            assert rank_task_sets[0].isdisjoint(rank_task_sets[1])
            assert rank_rows[0].isdisjoint(rank_rows[1])
            selected_tasks.extend(rank_task_sets[0] | rank_task_sets[1])
        assert Counter(selected_tasks) == Counter(range(10, 90, 10))


def test_cyclic_subset_resume_and_consumed_identity_are_exact() -> None:
    full = list(_subset_sampler(rank=1, start_step=0, stop_step=7))
    prefix = list(_subset_sampler(rank=1, start_step=0, stop_step=3))
    resumed = list(_subset_sampler(rank=1, start_step=3, stop_step=7))
    assert prefix + resumed == full

    first = _subset_sampler(rank=0)
    second = _subset_sampler(rank=1)
    summary = first.consumed_identity_summary(0, 4)
    assert summary == second.consumed_identity_summary(0, 4)
    assert summary["global_examples"] == 4 * 2 * 8
    assert summary["min_examples_per_task"] == 8
    assert summary["max_examples_per_task"] == 8
    assert summary["min_task_visits"] == 2
    assert summary["max_task_visits"] == 2
    assert summary["global_tasks_per_update"] == 4
    assert summary["updates_per_complete_task_cycle"] == 2
    assert summary["samples_per_task_per_visit"] == 4
    assert all(
        demos == (0, 1, 2, 3, 4)
        for demos in first.coverage_for_steps(0, 4).values()
    )


def test_cyclic_subset_requires_multiple_tasks_per_physical_batch() -> None:
    with pytest.raises(WriterModelError, match="cyclic subset"):
        CyclicSubsetMixedBatchSampler(
            _subset_dataset(),  # type: ignore[arg-type]
            task_ids=tuple(range(10, 90, 10)),
            per_rank_batch_size=8,
            tasks_per_rank_per_update=1,
            start_step=0,
            stop_step=1,
            rank=0,
            world_size=2,
            seed=7,
        )


def test_production_topology_preserves_full24_sample_clock() -> None:
    task_ids = tuple(range(24))
    dataset = _subset_dataset(task_ids)
    sampler = CyclicSubsetMixedBatchSampler(
        dataset,  # type: ignore[arg-type]
        task_ids=task_ids,
        per_rank_batch_size=144,
        tasks_per_rank_per_update=2,
        start_step=0,
        stop_step=3,
        rank=0,
        world_size=4,
        seed=20260728,
    )
    summary = sampler.consumed_identity_summary(0, 3)
    assert summary["global_examples"] == 3 * 4 * 144
    assert summary["global_tasks_per_update"] == 8
    assert summary["updates_per_complete_task_cycle"] == 3
    assert summary["samples_per_task_per_visit"] == 72
    assert summary["min_task_visits"] == summary["max_task_visits"] == 1
    assert summary["min_examples_per_task"] == 72
    assert summary["max_examples_per_task"] == 72
    assert all(
        len(sampler.tasks_for_step(step, rank=rank)) == 2
        for step in range(3)
        for rank in range(4)
    )
