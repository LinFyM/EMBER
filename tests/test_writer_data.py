from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pytest

from ember.writer.as_sampling import (
    MixedTaskBatchSampler,
    TeacherVideoSchedule,
)
from ember.writer.model import WriterModelError


@dataclass
class _DatasetStub:
    task_episode_rows: dict[int, dict[int, tuple[int, ...]]]
    frame_index: tuple[tuple[int, int, int], ...]


def _dataset() -> tuple[_DatasetStub, dict[int, tuple[int, int]]]:
    rows: dict[int, dict[int, tuple[int, ...]]] = {}
    identity: dict[int, tuple[int, int]] = {}
    frame_index: list[tuple[int, int, int]] = []
    flat = 0
    for task_id in (10, 20, 30, 40):
        rows[task_id] = {}
        for demo_index in range(5):
            episode_rows = tuple(range(flat, flat + 3))
            rows[task_id][demo_index] = episode_rows
            identity.update(
                {row: (task_id, demo_index) for row in episode_rows}
            )
            frame_index.extend(
                (task_id, demo_index, offset) for offset in range(3)
            )
            flat += 3
    return _DatasetStub(rows, tuple(frame_index)), identity


def _sampler(
    dataset: _DatasetStub,
    *,
    rank: int,
    start_step: int,
    stop_step: int,
) -> MixedTaskBatchSampler:
    schedule = TeacherVideoSchedule(
        task_ids=(10, 20, 30, 40),
        demo_indices=range(5),
        seed=19,
    )
    return MixedTaskBatchSampler(
        dataset,  # type: ignore[arg-type]
        task_ids=(10, 20, 30, 40),
        per_rank_batch_size=2,
        start_step=start_step,
        stop_step=stop_step,
        rank=rank,
        world_size=2,
        seed=20260720,
        tasks_per_rank_per_update=2,
        video_schedule=schedule,
        task_video_costs={
            task_id: {
                demo_index: task_id + 3 * demo_index
                for demo_index in range(5)
            }
            for task_id in (10, 20, 30, 40)
        },
    )


def _phase_dataset(
    *, rows_per_episode: int = 100
) -> tuple[_DatasetStub, dict[int, tuple[int, int]]]:
    rows: dict[int, dict[int, tuple[int, ...]]] = {10: {}}
    identity: dict[int, tuple[int, int]] = {}
    frame_index: list[tuple[int, int, int]] = []
    flat = 0
    for demo_index in range(50):
        episode_rows = tuple(range(flat, flat + rows_per_episode))
        rows[10][demo_index] = episode_rows
        identity.update(
            {
                row: (demo_index, offset)
                for offset, row in enumerate(episode_rows)
            }
        )
        frame_index.extend(
            (10, demo_index, offset) for offset in range(rows_per_episode)
        )
        flat += rows_per_episode
    return _DatasetStub(rows, tuple(frame_index)), identity


def _phase_sampler(
    dataset: _DatasetStub,
    *,
    seed: int = 20260801,
    start_step: int = 0,
    stop_step: int = 40,
) -> MixedTaskBatchSampler:
    schedule = TeacherVideoSchedule(
        task_ids=(10,),
        demo_indices=range(50),
        seed=19,
    )
    return MixedTaskBatchSampler(
        dataset,  # type: ignore[arg-type]
        task_ids=(10,),
        per_rank_batch_size=20,
        start_step=start_step,
        stop_step=stop_step,
        rank=0,
        world_size=1,
        seed=seed,
        tasks_per_rank_per_update=1,
        video_schedule=schedule,
        task_video_costs={10: {demo_index: 1 for demo_index in range(50)}},
    )


def test_task_complete_sampler_covers_every_task_and_runs_long_first() -> None:
    dataset, identity = _dataset()
    samplers = [
        _sampler(dataset, rank=rank, start_step=0, stop_step=6)
        for rank in range(2)
    ]
    rank_batches = [list(sampler) for sampler in samplers]
    for step in range(6):
        global_tasks = []
        for rank in range(2):
            tasks = samplers[rank].tasks_for_step(step)
            costs = [
                samplers[rank].task_video_costs[task_id][
                    samplers[rank].video_schedule.demo_for_task_visit(
                        task_id,
                        step,
                    )
                ]
                for task_id in tasks
            ]
            assert costs == sorted(costs, reverse=True)
            for microtask, task_id in enumerate(tasks):
                batch = rank_batches[rank][step * 2 + microtask]
                assert {identity[row][0] for row in batch} == {task_id}
                global_tasks.append(task_id)
        assert set(global_tasks) == {10, 20, 30, 40}


def test_mixed_task_sampler_resume_is_sample_exact() -> None:
    dataset, _ = _dataset()
    full = list(_sampler(dataset, rank=1, start_step=0, stop_step=9))
    prefix = list(_sampler(dataset, rank=1, start_step=0, stop_step=3))
    resumed = list(_sampler(dataset, rank=1, start_step=3, stop_step=9))

    assert prefix + resumed == full


def test_phase_strata_cover_batch_and_change_episode_mapping_by_visit() -> None:
    dataset, identity = _phase_dataset()
    sampler = _phase_sampler(dataset)

    strata_by_visit = []
    for task_visit in range(12):
        strata = sampler._phase_strata_for_task_visit(10, task_visit)
        demos = tuple(
            sampler._episode_for_task_visit(10, task_visit, batch_offset)[0]
            for batch_offset in range(20)
        )
        strata_by_visit.append(strata)
        assert set(strata) == set(range(20))
        assert len(set(demos)) == 20

    assert len(set(strata_by_visit)) == len(strata_by_visit)
    first_batch = list(_phase_sampler(dataset, stop_step=1))[0]
    assert len({identity[row][0] for row in first_batch}) == 20
    assert first_batch == [
        sampler._sample_for_task_visit(10, 0, batch_offset)
        for batch_offset in range(20)
    ]


def test_phase_stratified_schedule_is_seeded_and_exactly_resumable() -> None:
    dataset, _ = _phase_dataset()
    first = _phase_sampler(dataset, stop_step=30)
    second = _phase_sampler(dataset, stop_step=30)
    changed = _phase_sampler(dataset, seed=20260802, stop_step=30)

    full = list(first)
    assert full == list(second)
    assert full != list(changed)
    assert list(_phase_sampler(dataset, stop_step=11)) + list(
        _phase_sampler(dataset, start_step=11, stop_step=30)
    ) == full
    assert first.consumed_identity_summary(0, 30) == _phase_sampler(
        dataset,
        start_step=11,
        stop_step=30,
    ).consumed_identity_summary(0, 30)
    assert (
        first.consumed_identity_summary(0, 30)["identity_sha256"]
        != changed.consumed_identity_summary(0, 30)["identity_sha256"]
    )


def test_phase_stratified_episode_marginals_are_uniform_with_real_jitter() -> None:
    dataset, identity = _phase_dataset(rows_per_episode=100)
    sampler = _phase_sampler(dataset, stop_step=2_000)
    frames_by_demo: dict[int, list[int]] = defaultdict(list)

    for task_visit in range(2_000):
        for batch_offset in range(20):
            row = sampler._sample_for_task_visit(10, task_visit, batch_offset)
            demo_index, frame = identity[row]
            frames_by_demo[demo_index].append(frame)

    fixed_bin_centers = {5 * stratum + 2 for stratum in range(20)}
    for frames in frames_by_demo.values():
        assert len(frames) == 800
        quartiles = [
            sum(lower <= frame < lower + 25 for frame in frames) / len(frames)
            for lower in (0, 25, 50, 75)
        ]
        assert all(abs(fraction - 0.25) < 0.07 for fraction in quartiles)
        assert min(frames) < 5
        assert max(frames) >= 95
        assert any(frame not in fixed_bin_centers for frame in frames)


def test_task_complete_sampler_rejects_variable_batch_cycles() -> None:
    dataset, _ = _dataset()
    schedule = TeacherVideoSchedule(
        task_ids=(10, 20, 30, 40),
        demo_indices=range(5),
        seed=19,
    )
    with pytest.raises(WriterModelError, match="task-complete sampler"):
        MixedTaskBatchSampler(
            dataset,  # type: ignore[arg-type]
            task_ids=(10, 20, 30, 40),
            per_rank_batch_size=2,
            per_rank_batch_cycle=(2, 1),
            start_step=0,
            stop_step=8,
            rank=0,
            world_size=2,
            seed=20260720,
            tasks_per_rank_per_update=2,
            video_schedule=schedule,
            task_video_costs={
                task_id: {demo_index: 1 for demo_index in range(5)}
                for task_id in (10, 20, 30, 40)
            },
        )


def test_mixed_task_sampler_covers_every_episode_in_each_full_window() -> None:
    dataset, _ = _dataset()
    sampler = _sampler(dataset, rank=0, start_step=0, stop_step=8)

    assert all(
        episodes == (0, 1, 2, 3, 4)
        for episodes in sampler.coverage_for_steps(2, 8).values()
    )
    summary = sampler.consumed_identity_summary(0, 8)
    assert summary["global_examples"] == 8 * 4 * 2
    assert summary["unique_query_rows"] <= summary["global_examples"]
    assert summary["identity_sha256"] == _sampler(
        dataset, rank=1, start_step=3, stop_step=8
    ).consumed_identity_summary(0, 8)["identity_sha256"]


def test_teacher_video_schedule_is_independent_resumable_and_cycle_complete() -> None:
    schedule = TeacherVideoSchedule(
        task_ids=(10, 20, 30, 40), demo_indices=range(5), seed=19
    )
    full = [schedule.demo_for_task_visit(20, visit) for visit in range(13)]
    resumed = [schedule.demo_for_task_visit(20, visit) for visit in range(4, 13)]
    assert full[4:] == resumed
    assert all(set(full[start : start + 5]) == set(range(5)) for start in (0, 5))

    changed_video_seed = TeacherVideoSchedule(
        task_ids=(10, 20, 30, 40), demo_indices=range(5), seed=23
    )
    assert full != [changed_video_seed.demo_for_task_visit(20, visit) for visit in range(13)]

    dataset, _ = _dataset()
    assert list(_sampler(dataset, rank=0, start_step=0, stop_step=5)) == list(
        _sampler(dataset, rank=0, start_step=0, stop_step=5)
    )
    identity = schedule.identity_for_task_visits(20, 0, 10)
    assert identity["unique_demo_indices"] == tuple(range(5))
    assert len(identity["identity_sha256"]) == 64


def test_joint_writer_consumed_digest_covers_video_and_action_schedules() -> None:
    dataset, _ = _dataset()
    sampler = _sampler(dataset, rank=0, start_step=0, stop_step=8)
    first = TeacherVideoSchedule(
        task_ids=(10, 20, 30, 40), demo_indices=range(5), seed=19
    )
    second = TeacherVideoSchedule(
        task_ids=(10, 20, 30, 40), demo_indices=range(5), seed=23
    )
    summary = first.consumed_identity_summary(sampler, 0, 8)
    assert summary == first.consumed_identity_summary(sampler, 0, 8)
    assert summary["query"] == sampler.consumed_identity_summary(0, 8)
    assert summary["videos_per_task_visit"] == 1
    assert summary["min_video_visits_per_task"] == 8
    assert (
        summary["combined_identity_sha256"]
        != second.consumed_identity_summary(sampler, 0, 8)[
            "combined_identity_sha256"
        ]
    )
