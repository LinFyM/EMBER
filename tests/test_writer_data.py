from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest
from torch.utils.data import DataLoader

from ember.writer.as_sampling import MixedTaskBatchSampler, TeacherVideoSchedule
from ember.writer.data import RawTeacherVideoStore, WriterTaskAuthority
from ember.writer.errors import WriterModelError


@dataclass
class _DatasetStub:
    task_episode_rows: dict[int, dict[int, tuple[int, ...]]]
    frame_index: tuple[tuple[int, int, int], ...]

    def __len__(self) -> int:
        return len(self.frame_index)

    def __getitem__(self, item: int) -> dict[str, int]:
        return {"row": item}


def test_teacher_video_store_selects_the_declared_rgb_view(tmp_path: Path) -> None:
    path = tmp_path / "video.hdf5"
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("data/demo_0/obs")
        obs.create_dataset(
            "agentview_rgb", data=np.zeros((3, 2, 2, 3), dtype=np.uint8)
        )
        obs.create_dataset(
            "eye_in_hand_rgb", data=np.full((3, 2, 2, 3), 17, dtype=np.uint8)
        )
    authority = WriterTaskAuthority(
        task_id=7,
        language="task",
        path=path,
        expected_bytes=path.stat().st_size,
    )
    store = RawTeacherVideoStore(
        (authority,), frame_stride=1, camera_view="eye_in_hand"
    )

    video = store.load(7, 0)

    assert video.frames.shape == (3, 3, 2, 2)
    assert np.all(video.frames == 17)
    assert store.frame_counts(7, 0) == (3, 3)
    store.close()


def _dataset(
    task_ids: tuple[int, ...] = (10, 20, 30, 40),
    demos: int = 8,
    rows_per_demo: int = 6,
) -> tuple[_DatasetStub, dict[int, tuple[int, int]]]:
    rows: dict[int, dict[int, tuple[int, ...]]] = {}
    identity: dict[int, tuple[int, int]] = {}
    frame_index: list[tuple[int, int, int]] = []
    flat = 0
    for task_id in task_ids:
        rows[task_id] = {}
        for demo_index in range(demos):
            episode_rows = tuple(range(flat, flat + rows_per_demo))
            rows[task_id][demo_index] = episode_rows
            identity.update({row: (task_id, demo_index) for row in episode_rows})
            frame_index.extend(
                (task_id, demo_index, offset) for offset in range(rows_per_demo)
            )
            flat += rows_per_demo
    return _DatasetStub(rows, tuple(frame_index)), identity


def _sampler(
    dataset: _DatasetStub,
    *,
    rank: int,
    start_step: int,
    stop_step: int,
    seed: int = 20260720,
) -> MixedTaskBatchSampler:
    task_ids = tuple(sorted(dataset.task_episode_rows))
    demos = tuple(sorted(next(iter(dataset.task_episode_rows.values()))))
    schedule = TeacherVideoSchedule(
        task_ids=task_ids,
        demo_indices=demos,
        seed=19,
        videos_per_visit=4,
    )
    return MixedTaskBatchSampler(
        dataset,  # type: ignore[arg-type]
        task_ids=task_ids,
        per_rank_batch_size=2,
        start_step=start_step,
        stop_step=stop_step,
        rank=rank,
        world_size=2,
        seed=seed,
        tasks_per_rank_per_update=2,
        video_schedule=schedule,
        task_video_costs={
            task_id: {
                demo_index: task_id + 3 * demo_index
                for demo_index in demos
            }
            for task_id in task_ids
        },
    )


def test_task_complete_sampler_covers_every_task_and_uses_actual_k4_cost() -> None:
    dataset, identity = _dataset()
    samplers = [_sampler(dataset, rank=rank, start_step=0, stop_step=6) for rank in range(2)]
    rank_batches = [list(sampler) for sampler in samplers]
    for step in range(6):
        global_tasks = []
        for rank, sampler in enumerate(samplers):
            tasks = sampler.tasks_for_step(step)
            costs = []
            for task_id in tasks:
                excluded = sampler.action_demo_indices_for_task_visit(task_id, step)
                demos = sampler.video_schedule.demos_for_task_visit(
                    task_id, step, excluded=excluded
                )
                assert not set(demos) & set(excluded)
                costs.append(sum(sampler.task_video_costs[task_id][demo] for demo in demos))
            assert costs == sorted(costs, reverse=True)
            for microtask, task_id in enumerate(tasks):
                batch = rank_batches[rank][step * 2 + microtask]
                assert {identity[row][0] for row in batch} == {task_id}
                global_tasks.append(task_id)
        assert set(global_tasks) == {10, 20, 30, 40}


def test_sampler_resume_and_seed_are_sample_exact() -> None:
    dataset, _ = _dataset()
    full = list(_sampler(dataset, rank=1, start_step=0, stop_step=9))
    prefix = list(_sampler(dataset, rank=1, start_step=0, stop_step=3))
    resumed = list(_sampler(dataset, rank=1, start_step=3, stop_step=9))
    changed = list(
        _sampler(dataset, rank=1, start_step=0, stop_step=9, seed=20260721)
    )
    assert prefix + resumed == full
    assert changed != full


def test_dynamic_uneven_sampler_uses_all_five_ranks_and_covers_train24_once() -> None:
    task_ids = tuple(range(100, 124))
    dataset, _ = _dataset(task_ids=task_ids)
    schedule = TeacherVideoSchedule(
        task_ids=task_ids, demo_indices=range(8), seed=19, videos_per_visit=1
    )

    def sampler(rank: int, start: int, stop: int) -> MixedTaskBatchSampler:
        return MixedTaskBatchSampler(
            dataset,  # type: ignore[arg-type]
            task_ids=task_ids,
            per_rank_batch_size=2,
            start_step=start,
            stop_step=stop,
            rank=rank,
            world_size=5,
            seed=20260720,
            tasks_per_rank_per_update=5,
            video_schedule=schedule,
            task_video_costs={
                task_id: {demo: task_id + demo for demo in range(8)}
                for task_id in task_ids
            },
            assignment_strategy="cost_balanced_long_first_dynamic_uneven",
        )

    samplers = [sampler(rank, 0, 3) for rank in range(5)]
    for step in range(3):
        assignments = [value.tasks_for_step(step) for value in samplers]
        assert sorted(map(len, assignments)) == [4, 5, 5, 5, 5]
        assert {task for shard in assignments for task in shard} == set(task_ids)
        assert sum(map(len, assignments)) == 24
    full = list(sampler(4, 0, 3))
    resumed = list(sampler(4, 0, 1)) + list(sampler(4, 1, 3))
    assert resumed == full

    reference: dict[tuple[int, int], tuple[int, ...]] = {}
    for rank_sampler in samplers:
        batches = iter(rank_sampler)
        for step in range(3):
            for task_id in rank_sampler.tasks_for_step(step):
                reference[step, task_id] = tuple(next(batches))
    for step in range(3):
        queue = samplers[0].task_queue_for_step(step)
        assert len(queue) == 24
        assert {task_id for task_id, visit in queue if visit == step} == set(task_ids)
        for task_id, task_visit in queue:
            assert samplers[0].batch_indices_for_task_visit(
                step, task_id, task_visit
            ) == reference[step, task_id]


def test_dataloader_prefetch_preserves_exact_sampler_and_resume_rows() -> None:
    dataset, _ = _dataset()

    def rows(start: int, stop: int, workers: int) -> list[list[int]]:
        loader = DataLoader(
            dataset,
            batch_sampler=_sampler(
                dataset,
                rank=1,
                start_step=start,
                stop_step=stop,
            ),
            num_workers=workers,
            persistent_workers=workers > 0,
            prefetch_factor=2 if workers else None,
            multiprocessing_context="spawn" if workers else None,
        )
        return [batch["row"].tolist() for batch in loader]

    serial = rows(0, 9, 0)
    prefetched = rows(0, 9, 2)
    resumed = rows(0, 3, 2) + rows(3, 9, 2)
    assert prefetched == serial
    assert resumed == serial


def test_k4_teacher_schedule_is_unique_exclusion_safe_and_resumable() -> None:
    schedule = TeacherVideoSchedule(
        task_ids=(10, 20), demo_indices=range(8), seed=19, videos_per_visit=4
    )
    full = [schedule.demos_for_task_visit(20, visit, excluded=(0, 1)) for visit in range(13)]
    resumed = [schedule.demos_for_task_visit(20, visit, excluded=(0, 1)) for visit in range(4, 13)]
    assert full[4:] == resumed
    assert all(len(set(demos)) == 4 and not set(demos) & {0, 1} for demos in full)
    identity = schedule.identity_for_task_visits(20, 0, 20)
    assert identity["unique_demo_indices"] == tuple(range(8))


def test_cross_video_credit_schedule_keeps_anchor_and_uses_16_unique_demos() -> None:
    schedule = TeacherVideoSchedule(
        task_ids=(10,), demo_indices=range(50), seed=19, videos_per_visit=4
    )
    anchor = schedule.demos_for_task_visit(10, 3)
    views = schedule.cross_video_credit_demos_for_task_visit(10, 3, anchor)
    assert views[0] == anchor
    assert len(views) == 4
    assert all(len(view) == 4 and len(set(view)) == 4 for view in views)
    assert len({demo for view in views for demo in view}) == 16
    assert views == schedule.cross_video_credit_demos_for_task_visit(10, 3, anchor)


def test_one_shot_schedule_covers_all_videos_and_excludes_action_queries() -> None:
    dataset, _ = _dataset(task_ids=(10,), demos=50, rows_per_demo=20)
    schedule = TeacherVideoSchedule(
        task_ids=(10,), demo_indices=range(50), seed=19, videos_per_visit=1
    )
    sampler = MixedTaskBatchSampler(
        dataset,  # type: ignore[arg-type]
        task_ids=(10,),
        per_rank_batch_size=20,
        start_step=0,
        stop_step=50,
        rank=0,
        world_size=1,
        seed=20260720,
        tasks_per_rank_per_update=1,
        video_schedule=schedule,
        task_video_costs={10: {demo: 1 for demo in range(50)}},
    )
    full = []
    for visit in range(50):
        actions = sampler.action_demo_indices_for_task_visit(10, visit)
        teacher = schedule.demos_for_task_visit(10, visit, excluded=actions)
        assert len(actions) == 20
        assert len(teacher) == 1
        assert teacher[0] not in actions
        full.append(teacher[0])
    assert set(full) == set(range(50))
    assert full[17:] == [
        schedule.demos_for_task_visit(10, visit)[0]
        for visit in range(17, 50)
    ]


def test_one_shot_companion_keeps_primary_schedule_and_skips_queries() -> None:
    dataset, _ = _dataset(task_ids=(10,), demos=50, rows_per_demo=20)
    plain = TeacherVideoSchedule(
        task_ids=(10,), demo_indices=range(50), seed=19, videos_per_visit=1
    )
    schedule = TeacherVideoSchedule(
        task_ids=(10,),
        demo_indices=range(50),
        seed=19,
        videos_per_visit=1,
        companion_videos_per_visit=1,
    )
    sampler = MixedTaskBatchSampler(
        dataset,  # type: ignore[arg-type]
        task_ids=(10,),
        per_rank_batch_size=20,
        start_step=0,
        stop_step=50,
        rank=0,
        world_size=1,
        seed=20260720,
        tasks_per_rank_per_update=1,
        video_schedule=schedule,
        task_video_costs={10: {demo: 1 for demo in range(50)}},
    )
    for visit in range(50):
        excluded = sampler.action_demo_indices_for_task_visit(10, visit)
        primary = schedule.demos_for_task_visit(10, visit, excluded=excluded)
        companion = schedule.companion_demos_for_task_visit(
            10, visit, excluded=excluded
        )
        assert primary == plain.demos_for_task_visit(10, visit, excluded=excluded)
        assert len(companion) == 1
        assert companion[0] != primary[0]
        assert companion[0] not in excluded
        assert schedule.training_demos_for_task_visit(
            10, visit, excluded=excluded
        ) == primary + companion


def test_joint_consumed_summary_covers_k4_and_action_schedules() -> None:
    dataset, _ = _dataset()
    sampler = _sampler(dataset, rank=0, start_step=0, stop_step=8)
    schedule = sampler.video_schedule
    summary = schedule.consumed_identity_summary(sampler, 0, 8)
    assert summary["query"] == sampler.consumed_identity_summary(0, 8)
    assert summary["videos_per_task_visit"] == 4
    assert summary["min_video_visits_per_task"] == 8
    assert summary["min_unique_videos_per_task"] == 8


def test_action_phase_strata_cover_batch_and_full_episode_window() -> None:
    dataset, _ = _dataset(task_ids=(10,), demos=50, rows_per_demo=100)
    schedule = TeacherVideoSchedule(
        task_ids=(10,), demo_indices=range(50), seed=19, videos_per_visit=4
    )
    sampler = MixedTaskBatchSampler(
        dataset,  # type: ignore[arg-type]
        task_ids=(10,),
        per_rank_batch_size=20,
        start_step=0,
        stop_step=4,
        rank=0,
        world_size=1,
        seed=20260801,
        tasks_per_rank_per_update=1,
        video_schedule=schedule,
        task_video_costs={10: {demo: 1 for demo in range(50)}},
    )
    for visit in range(4):
        assert set(sampler._phase_strata_for_task_visit(10, visit)) == set(range(20))
        assert len(sampler.action_demo_indices_for_task_visit(10, visit)) == 20
    assert sampler.coverage_for_steps(0, 3)[10] == tuple(range(50))


def test_sampler_rejects_variable_batch_cycles() -> None:
    dataset, _ = _dataset()
    schedule = TeacherVideoSchedule(
        task_ids=(10, 20, 30, 40),
        demo_indices=range(8),
        seed=19,
        videos_per_visit=4,
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
            task_video_costs={task_id: {demo: 1 for demo in range(8)} for task_id in (10, 20, 30, 40)},
        )
