from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ember.writer.data import RawTeacherVideoStore, WriterTaskAuthority


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
