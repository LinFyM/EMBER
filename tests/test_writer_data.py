from __future__ import annotations

from dataclasses import dataclass

from ember.writer.data import MixedTaskBatchSampler, TeacherVideoSchedule


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
    summary = sampler.consumed_identity_summary(0, 8)
    assert summary["global_examples"] == 8 * 2 * 2
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
    assert summary["min_video_visits_per_task"] == 4
    assert (
        summary["combined_identity_sha256"]
        != second.consumed_identity_summary(sampler, 0, 8)[
            "combined_identity_sha256"
        ]
    )
