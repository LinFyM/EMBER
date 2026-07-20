from __future__ import annotations

from dataclasses import dataclass

from ember.writer.data import MixedTaskBatchSampler


@dataclass
class _DatasetStub:
    task_episode_rows: dict[int, dict[int, tuple[int, ...]]]


def _dataset() -> tuple[_DatasetStub, dict[int, tuple[int, int]]]:
    rows: dict[int, dict[int, tuple[int, ...]]] = {}
    identity: dict[int, tuple[int, int]] = {}
    flat = 0
    for task_id in (10, 20, 30, 40):
        rows[task_id] = {}
        for demo_index in range(5):
            episode_rows = tuple(range(flat, flat + 3))
            rows[task_id][demo_index] = episode_rows
            identity.update(
                {row: (task_id, demo_index) for row in episode_rows}
            )
            flat += 3
    return _DatasetStub(rows), identity


def _sampler(
    dataset: _DatasetStub,
    *,
    rank: int,
    start_step: int,
    stop_step: int,
) -> MixedTaskBatchSampler:
    return MixedTaskBatchSampler(
        dataset,  # type: ignore[arg-type]
        task_ids=(10, 20, 30, 40),
        per_rank_batch_size=2,
        start_step=start_step,
        stop_step=stop_step,
        rank=rank,
        world_size=2,
        seed=20260720,
    )


def test_mixed_task_sampler_has_global_no_replacement_cycles() -> None:
    dataset, identity = _dataset()
    rank_batches = [list(_sampler(dataset, rank=rank, start_step=0, stop_step=6)) for rank in range(2)]

    global_tasks = []
    for step in range(6):
        for rank in range(2):
            batch_tasks = {identity[row][0] for row in rank_batches[rank][step]}
            assert len(batch_tasks) == 1
            global_tasks.append(batch_tasks.pop())

    for start in range(0, len(global_tasks), 4):
        assert set(global_tasks[start : start + 4]) == {10, 20, 30, 40}


def test_mixed_task_sampler_resume_is_sample_exact() -> None:
    dataset, _ = _dataset()
    full = list(_sampler(dataset, rank=1, start_step=0, stop_step=9))
    prefix = list(_sampler(dataset, rank=1, start_step=0, stop_step=3))
    resumed = list(_sampler(dataset, rank=1, start_step=3, stop_step=9))

    assert prefix + resumed == full


def test_mixed_task_sampler_covers_every_episode_in_each_full_window() -> None:
    dataset, _ = _dataset()
    sampler = _sampler(dataset, rank=0, start_step=0, stop_step=8)

    assert all(
        episodes == (0, 1, 2, 3, 4)
        for episodes in sampler.coverage_for_steps(2, 8).values()
    )
