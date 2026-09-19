from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from ember.writer.data import FunctionalQueryDataset, RawTeacherVideoStore, WriterTaskAuthority
from ember.writer.functional import task_logical_batch_policy_rng_seed
from ember.writer.learning_data import EVENT_SCHEMA, LearningTask, WriterTrainingData


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


def test_dual_teacher_frames_share_stride_endpoint_and_camera_rotation(tmp_path: Path) -> None:
    path = tmp_path / "dual.hdf5"
    agent = np.arange(12 * 2 * 2 * 3, dtype=np.uint8).reshape(12, 2, 2, 3)
    wrist = 255 - agent
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("data/demo_0/obs")
        obs.create_dataset("agentview_rgb", data=agent)
        obs.create_dataset("eye_in_hand_rgb", data=wrist)
    authority = WriterTaskAuthority(0, "task", path, path.stat().st_size)
    store = RawTeacherVideoStore((authority,), frame_stride=5, camera_view="dual")
    video = store.load(0, 0)
    assert video.frame_indices.tolist() == [0, 5, 10, 11]
    assert video.frames.shape == (4, 2, 3, 2, 2)
    for index, raw in enumerate((agent, wrist)):
        np.testing.assert_array_equal(video.frames[:, index], raw[[0, 5, 10, 11], ::-1, ::-1].transpose(0, 3, 1, 2))
    assert store.frame_counts(0, 0) == (12, 4)
    store.close()


def test_dual_teacher_refuses_missing_or_unsynchronized_wrist(tmp_path: Path) -> None:
    import pytest
    from ember.writer.errors import WriterModelError

    for wrist_count in (None, 11):
        path = tmp_path / f"bad_{wrist_count}.hdf5"
        with h5py.File(path, "w") as handle:
            obs = handle.create_group("data/demo_0/obs")
            obs.create_dataset("agentview_rgb", data=np.zeros((12, 2, 2, 3), dtype=np.uint8))
            if wrist_count:
                obs.create_dataset("eye_in_hand_rgb", data=np.zeros((wrist_count, 2, 2, 3), dtype=np.uint8))
        authority = WriterTaskAuthority(0, "task", path, path.stat().st_size)
        store = RawTeacherVideoStore((authority,), frame_stride=5, camera_view="dual")
        for read in (store.load, store.frame_counts):
            with pytest.raises(WriterModelError, match="RGB view|synchronized"):
                read(0, 0)
        store.close()


def test_future_control_matches_next_observed_transition_and_has_no_terminal_query(tmp_path: Path) -> None:
    # A deterministic toy controller records observations AFTER applying a_i,
    # as LIBERO create_dataset does. The next displacement is the label oracle.
    path = tmp_path / "post_action.hdf5"
    actions = np.zeros((4, 7), dtype=np.float32)
    actions[:, 0] = [0.1, 0.2, 0.3, 0.4]
    actions[:, 6] = [-1, -1, 1, 1]
    ee = np.zeros((4, 6), dtype=np.float32)
    ee[:, 0] = np.cumsum(actions[:, 0])
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data/demo_16")
        demo.create_dataset("actions", data=actions)
        obs = demo.create_group("obs")
        obs.create_dataset("ee_states", data=ee)
        obs.create_dataset("gripper_states", data=np.zeros((4, 2), dtype=np.float32))
        for camera in ("agentview_rgb", "eye_in_hand_rgb"):
            obs.create_dataset(camera, data=np.zeros((4, 2, 2, 3), dtype=np.uint8))
    authority = WriterTaskAuthority(0, "task", path, path.stat().st_size)
    future = FunctionalQueryDataset([authority], demo_indices=[16], action_chunk_size=50, action_start_offset=1)
    try:
        assert future.frame_index == ((0, 16, 0), (0, 16, 1), (0, 16, 2))
        row = future[1]
        np.testing.assert_allclose(row["observation.state"][:3] + row["action"][0, :3], ee[2, :3])
        assert row["action"][0, 6] == 1  # next transition closes the gripper
        assert row["frame_index"] == 1 and row["action_start_index"] == 2
        terminal = future[2]
        assert terminal["action_is_pad"].tolist() == [False] + [True] * 49
        np.testing.assert_array_equal(terminal["action"], np.repeat(actions[-1:], 50, axis=0))
    finally:
        future.close()
    historical = FunctionalQueryDataset([authority], demo_indices=[16], action_chunk_size=50, action_start_offset=0)
    try:
        assert len(historical) == 4
        np.testing.assert_array_equal(historical[1]["action"][0], actions[1])
    finally:
        historical.close()


@pytest.fixture
def training_data_factory(tmp_path, monkeypatch):
    """Use real lazy HDF5 reads with small observations and metadata for train24."""
    path = tmp_path / "train.hdf5"
    lengths = tuple(31 + demo % 10 for demo in range(50))
    with h5py.File(path, "w") as handle:
        for demo, length in enumerate(lengths):
            group = handle.create_group(f"data/demo_{demo}")
            actions = np.full((length, 7), demo * 100, dtype=np.float32)
            actions[:, 0] += np.arange(length)
            group.create_dataset("actions", data=actions)
            obs = group.create_group("obs")
            obs.create_dataset("ee_states", data=np.zeros((length, 6), dtype=np.float32))
            obs.create_dataset("gripper_states", data=np.zeros((length, 2), dtype=np.float32))
            for camera in ("agentview_rgb", "eye_in_hand_rgb"):
                obs.create_dataset(camera, data=np.zeros((length, 2, 2, 3), dtype=np.uint8))
    def metadata(_asset_root, task_ids):
        return {task: LearningTask(WriterTaskAuthority(task, f"task{task}", path, path.stat().st_size),
                                   f"suite{task // 6}", task % 6, lengths) for task in task_ids}
    monkeypatch.setattr("ember.writer.learning_data.load_learning_tasks", metadata)
    opened = []
    def create(*, camera_view="dual", **changes):
        config = {"seed": 7, "sampler_seed": 20260721, "teacher_video_seed": 20260722,
                  "maximum_updates": 12, "teaching_seed": 20260919, "teaching_queries_per_task": 7,
                  "teaching_episode": "same_video", "grouping": "baseline", "event_schema_version": EVENT_SCHEMA,
                  "task_ids": list(range(24)), "tasks_per_update": 4, "conditions_per_task": 1,
                  "queries_per_task": 21, "cardinalities": [1], "video_demos": list(range(46)),
                  "action_demos": list(range(46)), "diagnostic_action_demos": list(range(46, 50)),
                  "held_video_demos": list(range(46, 50)), "action_start_offset": 1,
                  "query_alignment": "post_action_observation_future_control_v1", **changes}
        data = WriterTrainingData(tmp_path, config, camera_view=camera_view)
        opened.append(data)
        return data
    yield create
    for data in opened:
        data.close()


def test_registered_camera_modes_share_events_and_read_only_declared_rgb(training_data_factory, monkeypatch):
    single = training_data_factory(camera_view="agentview")
    dual = training_data_factory()
    assert single.event_plan() == dual.event_plan()
    assert single.next_iteration() == dual.next_iteration()
    reads = []
    original = h5py.Dataset.__getitem__
    def rgb_only(dataset, key):
        assert dataset.name.endswith(("/agentview_rgb", "/eye_in_hand_rgb"))
        reads.append(dataset.name.rsplit("/", 1)[-1])
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", rgb_only)
    frames_a, indices_a = single.load_videos(0, [0])
    assert reads == ["agentview_rgb"]
    frames_b, indices_b = dual.load_videos(0, [0])
    assert reads == ["agentview_rgb", "agentview_rgb", "eye_in_hand_rgb"]
    np.testing.assert_array_equal(frames_a[0], frames_b[0][:, 0])
    np.testing.assert_array_equal(indices_a[0], indices_b[0])
    assert frames_a[0].ndim == 4 and frames_b[0].ndim == 5
    with pytest.raises(ValueError, match="registered agentview or dual"):
        training_data_factory(camera_view="eye_in_hand")


def test_full_training_plan_has_balanced_rounds_and_cross_episode_events(training_data_factory):
    data = training_data_factory(maximum_updates=1200)
    plan = data.event_plan()
    assert json.loads(json.dumps(plan)) == plan
    assert len(plan["events"]) == 4800 and len(plan["groups"]) == 1200
    assert sorted(event_index for group in plan["groups"] for event_index in group) == list(range(4800))
    for occurrence in range(200):
        draws = [data.next_iteration() for _ in range(6)]
        tasks = [draw["task"] for group in draws for draw in group]
        expected = np.random.default_rng(np.random.SeedSequence([20260721, occurrence])).permutation(range(24))
        assert tasks == expected.tolist()
        assert all([draw["job_id"] for draw in group] == list(range(4)) for group in draws)
        assert all(draw["occurrence"] == occurrence for group in draws for draw in group)
    assert data.counts == dict.fromkeys(range(24), 200)
    with pytest.raises(StopIteration):
        data.next_iteration()
    for task in data.task_ids:
        events = [event for event in plan["events"] if event["task"] == task]
        teachers = [event["teacher_demo"] for event in events]
        for start in range(0, 184, 46):
            assert set(teachers[start:start + 46]) == set(range(46))
        exposure = Counter(demo for event in events for demo in event["action_demos"])
        assert set(exposure) == set(range(46)) and max(exposure.values()) - min(exposure.values()) <= 6
        for event in events:
            assert len(event["action_demos"]) == len(set(event["action_demos"])) == 21
            assert event["teacher_demo"] not in event["action_demos"]
            assert set(event["action_demos"]) <= set(range(46))
            assert all(0 <= frame < data.tasks[task].episode_lengths[demo] - 1
                       for demo, frame in zip(event["action_demos"], event["action_frames"], strict=True))
            assert event["action_start_indices"] == [frame + 1 for frame in event["action_frames"]]


def test_regrouping_and_json_resume_preserve_event_and_flow_identity(training_data_factory):
    baseline = training_data_factory()
    explicit = training_data_factory(grouping="explicit", event_groups=[list(range(start, start + 4))
                                    for _ in range(2) for start in range(0, 24, 4)])
    first, second = baseline.event_plan(), explicit.event_plan()
    assert first["groups"] != second["groups"]
    assert first["events"] == second["events"]
    for _ in range(7):
        baseline.next_iteration()
    saved = json.loads(json.dumps(baseline.sampler_state()))
    resumed = training_data_factory()
    resumed.restore_sampler(saved)
    assert baseline.next_iteration() == resumed.next_iteration()
    for event in first["events"]:
        assert event["policy_rng_seed"] == task_logical_batch_policy_rng_seed(
            optimization_seed=7, task_id=event["task"], task_visit=event["occurrence"],
            demo_indices=event["action_demos"], frame_indices=event["action_frames"],
        )
    with pytest.raises(ValueError, match="contract or grouping"):
        explicit.restore_sampler(saved)
    tampered = deepcopy(saved)
    tampered["task_occurrences"]["0"] += 1
    tampered["task_occurrences"]["1"] -= 1
    with pytest.raises(ValueError, match="exposure cursor"):
        resumed.restore_sampler(tampered)
    assert resumed.next_step == 8
    # Exported plans and checkpoint state cannot mutate the live event schedule.
    first["events"][0]["action_demos"][0] = 49
    saved["event_contract"]["groups"][0][0] = -1
    assert baseline.event_plan()["events"] == second["events"]


@pytest.mark.parametrize('relation', ['same_video', 'cross_episode'])
def test_completed_window_continuation_preserves_events_and_next_round(training_data_factory, relation):
    from ember.writer.continuation import require_extended_prefix

    parent = training_data_factory(maximum_updates=1500, teaching_episode=relation)
    child = training_data_factory(maximum_updates=2100, teaching_episode=relation)
    parent_plan, child_plan = parent.event_plan(), child.event_plan()
    require_extended_prefix(parent_plan, child_plan)
    for _ in range(1500):
        parent.next_iteration()
    saved = parent.sampler_state()
    with pytest.raises(ValueError, match='contract or grouping'):
        child.restore_sampler(saved)
    child.restore_sampler(saved, extend_completed=True)
    draws = child.next_iteration()
    expected = [child_plan['events'][i] for i in child_plan['groups'][1500]]
    assert [(r['task'], r['occurrence'], r['query_seed']) for r in draws] == [
        (r['task'], r['occurrence'], r['query_seed']) for r in expected]
    assert all(r['occurrence'] == 250 for r in draws)
    changed = deepcopy(parent_plan)
    changed['events'][0]['teaching']['policy_rng_seed'] += 1
    with pytest.raises(ValueError, match='every parent event'):
        require_extended_prefix(changed, child_plan)
    changed = deepcopy(saved)
    changed['next_step'] = 1200
    with pytest.raises(ValueError, match='completed parent1500'):
        child.restore_sampler(changed, extend_completed=True)


def test_event_batch_reads_only_selected_actions_and_keeps_full_batch_rng(training_data_factory, monkeypatch):
    reads = []
    original = h5py.Dataset.__getitem__
    def record(dataset, key):
        if dataset.name.endswith("/actions"):
            reads.append(int(dataset.name.split("/")[2].removeprefix("demo_")))
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", record)
    data = training_data_factory()
    plan = data.event_plan()
    assert reads == []  # Construction and the complete event export inspect metadata only.
    draw = data.next_iteration()[0]
    event = next(event for event in plan["events"]
                 if event["task"] == draw["task"] and event["occurrence"] == draw["occurrence"])
    # Simulate a worker loading only one physical slice, independently of scheduling order.
    batch, trace = data.action_batch(draw["task"], draw["occurrence"], draw["video_demos"],
                                   query_seed=draw["query_seed"], query_offset=5, query_count=7)
    assert reads == event["action_demos"][5:12]
    assert draw["video_demos"][0] not in reads and set(reads) <= set(range(46))
    assert trace["policy_random_batch_size"] == 21 and trace["query_offset"] == 5
    assert trace["policy_rng_seed"] == event["policy_rng_seed"]
    assert batch["demo_index"].tolist() == trace["action_demos"]
    assert batch["frame_index"].tolist() == trace["action_frames"]
    assert batch["action_start_index"].tolist() == trace["action_start_indices"]
    np.testing.assert_array_equal(batch["action"][:, 0, 0].numpy(),
                                  np.asarray(trace["action_demos"]) * 100 + trace["action_start_indices"])
    with pytest.raises(ValueError, match="registered training event"):
        data.action_batch(draw["task"], draw["occurrence"], (46,), query_seed=draw["query_seed"])
    with pytest.raises(ValueError, match="registered training event"):
        data.action_batch(draw["task"], draw["occurrence"], draw["video_demos"], query_seed=0)


def test_teaching_queries_use_five_real_next_actions_and_matched_ablation_noise(training_data_factory):
    same = training_data_factory()
    other = training_data_factory(teaching_episode="cross_episode")
    same_plan, other_plan = same.event_plan(), other.event_plan()
    for left, right in zip(same_plan["events"], other_plan["events"], strict=True):
        assert {k: v for k, v in left.items() if k != "teaching"} == {
            k: v for k, v in right.items() if k != "teaching"}
        a, b = left["teaching"], right["teaching"]
        assert a["policy_rng_seed"] == b["policy_rng_seed"] != left["policy_rng_seed"]
        assert set(a["action_demos"]) == {left["teacher_demo"]}
        assert left["teacher_demo"] not in b["action_demos"]
        for taught in (a, b):
            demo = taught["action_demos"][0]
            length = same.tasks[left["task"]].episode_lengths[demo]
            legal = list(range(0, length - 5, 5))
            assert set(taught["action_frames"]) <= set(legal)
            assert taught["sampling_with_replacement"] == (len(legal) < 7)
            if len(legal) >= 7:
                assert len(set(taught["action_frames"])) == 7
    draw = same.next_iteration()[0]
    batch, trace = same.action_batch(draw["task"], draw["occurrence"], draw["video_demos"],
                                    query_seed=draw["query_seed"], teaching=True)
    assert not batch["action_is_pad"][:, :5].any()
    expected = (np.asarray(trace["action_demos"])[:, None] * 100
                + np.asarray(trace["action_frames"])[:, None] + np.arange(1, 6)[None])
    np.testing.assert_array_equal(batch["action"][:, :5, 0].numpy(), expected)
    assert trace["policy_random_batch_size"] == 7
    changed = same.sampler_state()
    with pytest.raises(ValueError, match="contract or grouping"):
        other.restore_sampler(changed)


def test_frozen_diagnostics_use_held_actions_and_exclude_the_condition_video(training_data_factory):
    data = training_data_factory()
    before = data.sampler_state()
    batch, trace = data.diagnostic_batch(0, seed=31, count=13, teacher_demo=47)
    assert set(trace["action_demos"]) == {46, 48, 49}
    exposure = Counter(trace["action_demos"])
    assert max(exposure.values()) - min(exposure.values()) <= 1
    assert batch["demo_index"].tolist() == trace["action_demos"]
    assert data.diagnostic_batch(0, seed=31, count=13, teacher_demo=47)[1] == trace
    assert data.sampler_state() == before
    videos, indices = data.load_videos(0, (47,))
    assert len(videos) == 1 and videos[0].shape[1] == 2
    assert indices[0][-1] == data.tasks[0].episode_lengths[47] - 1


@pytest.mark.parametrize("change", [
    {"task_ids": list(range(23))}, {"maximum_updates": 7}, {"maximum_updates": 12006}, {"conditions_per_task": 2},
    {"action_demos": list(range(50))}, {"video_demos": list(range(50))},
    {"diagnostic_action_demos": list(range(42, 46))}, {"grouping": "suite_random"},
    {"grouping": "explicit", "event_groups": [[0, 1, 2, 2]] * 12},
    {"grouping": "explicit", "event_groups": [[0, 1, 2, 3]] * 12},
])
def test_training_plan_rejects_contract_and_round_changes(training_data_factory, change):
    with pytest.raises(ValueError):
        training_data_factory(**change)
