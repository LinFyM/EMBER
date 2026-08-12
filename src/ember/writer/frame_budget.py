"""Canonical ordered frame budget before the expensive joint backbone."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ember.writer.data import RawTeacherVideo
from ember.writer.errors import WriterModelError


def ordered_video_frame_indices(length: int, limit: int) -> np.ndarray:
    """Keep a deterministic, endpoint-preserving ordered view of one video."""

    if length <= 0 or limit <= 0:
        raise WriterModelError("invalid ordered video frame budget")
    if length <= limit:
        return np.arange(length, dtype=np.int64)
    selected = np.rint(np.linspace(0, length - 1, limit)).astype(np.int64)
    if (
        selected.shape != (limit,)
        or selected[0] != 0
        or selected[-1] != length - 1
        or np.any(selected[1:] <= selected[:-1])
    ):
        raise WriterModelError("ordered video frame budget lost its endpoints")
    return selected


def apply_condition_frame_budget(
    videos: Sequence[RawTeacherVideo],
    total_budget: int,
) -> tuple[RawTeacherVideo, ...]:
    """Allocate the same per-video cap to a permutation-invariant K-video set."""

    if not videos or total_budget < len(videos):
        raise WriterModelError("condition frame budget cannot cover every video")
    per_video = total_budget // len(videos)
    selected = []
    for video in videos:
        indices = ordered_video_frame_indices(len(video.frame_indices), per_video)
        selected.append(
            RawTeacherVideo(
                frames=video.frames[indices],
                frame_indices=video.frame_indices[indices],
                raw_frame_count=video.raw_frame_count,
            )
        )
    return tuple(selected)
