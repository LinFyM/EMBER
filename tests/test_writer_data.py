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
