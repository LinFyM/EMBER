from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ember.writer.data import FunctionalQueryDataset, RawTeacherVideoStore, WriterTaskAuthority


def test_local_field_sampling_preserves_main_rng_and_only_true_future_actions():
    import random
    from types import SimpleNamespace
    import torch
    from ember.writer.learning_data import WriterTrainingData

    actions = np.arange(12 * 7, dtype=np.float32).reshape(12, 7)
    data = object.__new__(WriterTrainingData)
    data.tasks = {0: SimpleNamespace(episode_lengths=(12,) * 17)}
    data.video_pool = (16,)
    data.queries = SimpleNamespace(_handle=lambda task: {'data/demo_16/actions': actions})
    random.seed(81)
    python_rng, torch_rng = random.getstate(), torch.random.get_rng_state()
    arguments = dict(query_seed=913, positions_per_condition=4, future_horizon=15, seed=20260914)
    indices, labels, counts, trace = data.local_field_batch(0, 16, torch.tensor([0, 5, 10, 11]), **arguments)
    assert random.getstate() == python_rng
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    assert indices.tolist() == [0, 1, 2, 3] and counts.tolist() == [11, 6, 1, 0]
    assert trace['field_seed'] == 913 ^ 20260914
    assert trace['field_frame_positions'] == [0, 5, 10, 11]
    for row, position in enumerate(trace['field_frame_positions']):
        count = int(counts[row])
        np.testing.assert_array_equal(labels[row, :count], actions[position + 1:position + 1 + count])
        assert labels[row, count:].count_nonzero() == 0
    repeated = data.local_field_batch(0, 16, torch.tensor([0, 5, 10, 11]), **arguments)
    assert repeated[-1] == trace

    # A short video samples every available position, while a longer one draws
    # a subset from the entire real video, including the end region.
    data.tasks[0].episode_lengths = (12,) * 17
    seen = set()
    for seed in range(40):
        selected, _, _, row = data.local_field_batch(0, 16, torch.arange(12), **{**arguments, 'query_seed': seed})
        assert len(selected) == len(selected.unique()) == 4
        seen.update(row['field_frame_ordinals'])
    assert seen == set(range(12))


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
