from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pytest

from ember.source_sft.sampler import (
    HierarchicalMixedBatchSampler,
    SourceBaseBatchSampler,
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


def _sft_dataset():
    episode_rows, frame_index = {}, []
    for task in range(24):
        episode_rows[task] = {}
        for demo in range(50):
            start = len(frame_index)
            frame_index.extend((task, demo, frame) for frame in range(3 + demo % 3))
            episode_rows[task][demo] = tuple(range(start, len(frame_index)))
    return _DatasetStub(episode_rows, tuple(frame_index))


def _sft_sampler(dataset, world=4, micro=144, accumulation=1, rank=0, start=0, stop=4):
    return HierarchicalMixedBatchSampler(
        dataset, task_ids=tuple(range(24)), logical_world_size=4, logical_per_rank_batch_size=144,
        per_rank_batch_size=micro, gradient_accumulation_steps=accumulation,
        start_step=start, stop_step=stop, rank=rank, world_size=world, seed=20260723)


@pytest.mark.parametrize("world,micro,accumulation", [
    (1, 32, 18), (2, 32, 9), (3, 64, 3), (4, 16, 9), (4, 64, 3), (5, 32, 4), (6, 32, 3)])
def test_sft_physical_packing_preserves_all24_and_global576(world, micro, accumulation):
    dataset = _sft_dataset()
    samplers = [_sft_sampler(dataset, world, micro, accumulation, rank) for rank in range(world)]
    reference = _sft_sampler(dataset)
    for step in range(4):
        # Concatenation order is by rank then microbatch: the unchanged logical stream.
        physical = [row for sampler in samplers for index in range(accumulation)
                    for row in sampler.batch_for_step(step, index)]
        logical = [row for rank in range(4) for row in reference.logical_batch_for_step(step, rank)]
        assert physical == logical
        assert Counter(dataset.frame_index[row][0] for row in physical) == Counter({task: 24 for task in range(24)})
    prefix = list(_sft_sampler(dataset, world, micro, accumulation, stop=2))
    resumed = list(_sft_sampler(dataset, world, micro, accumulation, start=2))
    assert prefix + resumed == list(samplers[0])
    assert samplers[0].consumed_summary(0, 450)["global_examples"] == 259200
    assert samplers[0].consumed_summary(0, 450)["min_examples_per_task"] == 10800


def test_sft_episode_and_chunk_cycles_cover_without_replacement():
    dataset = _sft_dataset()
    sampler = _sft_sampler(dataset)
    for task in (0, 13, 23):
        for cycle in range(10):
            rows = [sampler._row(task, offset) for offset in range(cycle * 50, (cycle + 1) * 50)]
            assert sorted(dataset.frame_index[row][1] for row in rows) == list(range(50))
        for demo, expected in dataset.task_episode_rows[task].items():
            observed = [sampler._row(task, cycle * 50 + sampler._episode_order(task, cycle).index(demo))
                        for cycle in range(len(expected))]
            assert set(observed) == set(expected)
    assert all(len(demos) == 50 for demos in sampler.coverage_for_steps(0, 3).values())


def test_sft_packing_refuses_empty_microbatches():
    with pytest.raises(WriterModelError, match="microbatches"):
        _sft_sampler(_sft_dataset(), world=4, micro=64, accumulation=4)


@pytest.mark.parametrize("world", range(1, 7))
def test_task_striped_packing_preserves_36_task_logical_batch(world):
    dataset = _subset_dataset(tuple(range(36)))
    accumulation = (576 // world + 63) // 64
    samplers = [HierarchicalMixedBatchSampler(
        dataset, task_ids=tuple(range(36)), logical_world_size=4,
        logical_per_rank_batch_size=144, per_rank_batch_size=64,
        gradient_accumulation_steps=accumulation, start_step=0, stop_step=2,
        rank=rank, world_size=world, seed=20260920,
        physical_packing="task_striped",
    ) for rank in range(world)]
    for step in range(2):
        logical = samplers[0].global_rows_for_step(step)
        physical = [row for sampler in samplers for micro in range(accumulation)
                    for row in sampler.batch_for_step(step, micro)]
        assert Counter(physical) == Counter(logical)
        assert set(Counter(dataset.frame_index[row][0] for row in logical).values()) == {16}
        assert max(sum(sampler.task_counts_for_step(step).values()) for sampler in samplers) - min(
            sum(sampler.task_counts_for_step(step).values()) for sampler in samplers) <= 1
    if world == 4:
        assert all(set(sampler.task_counts_for_step(0).values()) == {4} for sampler in samplers)


@pytest.mark.parametrize("world,micro,accumulation", [(4, 4, 16), (4, 8, 8), (2, 8, 16),
    (3, 8, 11), (3, 16, 6), (4, 12, 6), (5, 8, 7), (6, 8, 6)])
def test_source_base_keeps_original_global256_task_episode_stream(world, micro, accumulation):
    dataset = _subset_dataset()

    def batches(ranks, batch, accumulation, start=0, stop=5):
        samplers = [SourceBaseBatchSampler(
            dataset, task_ids=tuple(dataset.task_episode_rows), per_rank_batch_size=batch,
            logical_task_batch_size=32, start_step=start * accumulation, stop_step=stop * accumulation,
            global_batch_size=256, gradient_accumulation_steps=accumulation,
            rank=rank, world_size=ranks, seed=20260721) for rank in range(ranks)]
        return [[row for step in range(update * accumulation, (update + 1) * accumulation)
                 for sampler in samplers for row in sampler.batch_for_step(step)]
                for update in range(start, stop)]

    original = batches(8, 32, 1)
    physical = batches(world, micro, accumulation)
    assert [sorted(rows) for rows in physical] == [sorted(rows) for rows in original]
    assert batches(world, micro, accumulation, stop=2) + batches(world, micro, accumulation, start=2) == physical
    for update in physical:
        tasks = [dataset.frame_index[row][0] for row in update]
        assert set(Counter(tasks).values()) == {32}
