from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pytest

from ember.source_sft.sampler import HierarchicalMixedBatchSampler
from ember.writer.model import WriterModelError


@dataclass
class _DatasetStub:
    task_episode_rows: dict[int, dict[int, tuple[int, ...]]]
    frame_index: tuple[tuple[int, int, int], ...]


def _dataset() -> _DatasetStub:
    episode_rows: dict[int, dict[int, tuple[int, ...]]] = {}
    frame_index: list[tuple[int, int, int]] = []
    flat = 0
    for task_id in (10, 20, 30, 40):
        episode_rows[task_id] = {}
        for demo_index in range(5):
            rows = tuple(range(flat, flat + 3))
            episode_rows[task_id][demo_index] = rows
            frame_index.extend(
                (task_id, demo_index, frame) for frame in range(3)
            )
            flat += 3
    return _DatasetStub(episode_rows, tuple(frame_index))


def _sampler(
    *,
    rank: int,
    start_step: int = 0,
    stop_step: int = 6,
) -> HierarchicalMixedBatchSampler:
    return HierarchicalMixedBatchSampler(
        _dataset(),  # type: ignore[arg-type]
        task_ids=(10, 20, 30, 40),
        per_rank_batch_size=8,
        start_step=start_step,
        stop_step=stop_step,
        rank=rank,
        world_size=2,
        seed=20260728,
    )


def test_each_physical_batch_mixes_every_task_with_equal_counts() -> None:
    dataset = _dataset()
    samplers = [_sampler(rank=rank) for rank in range(2)]
    for step in range(6):
        rank_rows = []
        for sampler in samplers:
            batch = sampler.batch_for_step(step)
            counts = Counter(dataset.frame_index[row][0] for row in batch)
            assert counts == {10: 2, 20: 2, 30: 2, 40: 2}
            rank_rows.append(set(batch))
        assert rank_rows[0].isdisjoint(rank_rows[1])


def test_hierarchical_mixed_sampler_resume_is_sample_exact() -> None:
    full = list(_sampler(rank=1, start_step=0, stop_step=9))
    prefix = list(_sampler(rank=1, start_step=0, stop_step=3))
    resumed = list(_sampler(rank=1, start_step=3, stop_step=9))
    assert prefix + resumed == full


def test_episode_and_chunk_cycles_are_balanced_without_replacement() -> None:
    dataset = _dataset()
    sampler = _sampler(rank=0)
    assert all(
        demos == (0, 1, 2, 3, 4)
        for demos in sampler.coverage_for_steps(0, 2).values()
    )

    selected_by_demo = {demo_index: set() for demo_index in range(5)}
    for global_offset in range(15):
        demo_index, _ = sampler._episode_for_global_offset(20, global_offset)
        selected_by_demo[demo_index].add(
            sampler._row_for_global_offset(20, global_offset)
        )
    assert all(
        rows == set(dataset.task_episode_rows[20][demo_index])
        for demo_index, rows in selected_by_demo.items()
    )


def test_consumed_digest_covers_all_ranks_and_is_task_balanced() -> None:
    first = _sampler(rank=0)
    second = _sampler(rank=1, start_step=3)
    summary = first.consumed_identity_summary(0, 6)
    assert summary == second.consumed_identity_summary(0, 6)
    assert summary["global_examples"] == 6 * 2 * 8
    assert summary["min_examples_per_task"] == 24
    assert summary["max_examples_per_task"] == 24
    assert summary["samples_per_task_per_rank_per_step"] == 2
    assert summary["global_samples_per_task_per_step"] == 4
    assert len(summary["identity_sha256"]) == 64


def test_hierarchical_sampler_requires_task_divisible_physical_batch() -> None:
    with pytest.raises(WriterModelError, match="hierarchical mixed-task"):
        HierarchicalMixedBatchSampler(
            _dataset(),  # type: ignore[arg-type]
            task_ids=(10, 20, 30, 40),
            per_rank_batch_size=7,
            start_step=0,
            stop_step=1,
            rank=0,
            world_size=2,
            seed=7,
        )
