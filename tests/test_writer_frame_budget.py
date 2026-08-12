from __future__ import annotations

import numpy as np
import pytest

from ember.writer.data import RawTeacherVideo
from ember.writer.errors import WriterModelError
from ember.writer.frame_budget import (
    apply_condition_frame_budget,
    ordered_video_frame_indices,
)


def _video(length: int, marker: int) -> RawTeacherVideo:
    return RawTeacherVideo(
        frames=np.full((length, 3, 2, 2), marker, dtype=np.uint8),
        frame_indices=np.arange(length, dtype=np.int64) * 5,
        raw_frame_count=length * 5,
    )


def test_ordered_budget_keeps_endpoints_and_strict_order() -> None:
    selected = ordered_video_frame_indices(105, 16)
    assert selected.tolist() == sorted(set(selected.tolist()))
    assert (selected[0], selected[-1], len(selected)) == (0, 104, 16)
    assert ordered_video_frame_indices(4, 16).tolist() == [0, 1, 2, 3]


def test_condition_budget_is_equal_per_video_and_permutation_equivariant() -> None:
    videos = (_video(105, 1), _video(31, 2), _video(80, 3), _video(20, 4))
    natural = apply_condition_frame_budget(videos, 64)
    permuted = apply_condition_frame_budget(videos[::-1], 64)
    assert [len(video.frame_indices) for video in natural] == [16] * 4
    for left, right in zip(natural, permuted[::-1], strict=True):
        assert np.array_equal(left.frame_indices, right.frame_indices)
        assert np.array_equal(left.frames, right.frames)


def test_condition_budget_rejects_an_uncovered_video_set() -> None:
    with pytest.raises(WriterModelError, match="cover every video"):
        apply_condition_frame_budget((_video(2, 1), _video(2, 2)), 1)
