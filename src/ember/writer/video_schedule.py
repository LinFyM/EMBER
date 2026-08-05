"""Deterministic balanced K4 teacher-video schedules for Writer evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ember.pi05_target_data import SUITE_ORDER
from ember.writer.model import WriterModelError


SAME_TASK_OTHER_DEMO_OFFSET = 17
VIDEOS_PER_CONDITION = 4
K4_POSITION_OFFSETS = (0, 13, 27, 39)
WRITER_VIDEO_SAMPLING_MODES = {"with_replacement", "without_replacement"}
WRITER_VIDEO_SCHEDULE_WITH_REPLACEMENT = (
    "numeric_seeded_independent_k4_without_replacement_per_rollout"
)
WRITER_VIDEO_SCHEDULE_WITHOUT_REPLACEMENT = (
    "numeric_seeded_task_permutation_with_offsets_0_13_27_39; each demo appears "
    "once per shot lane in every 50-state block"
)


def writer_video_selection_seed(
    root_seed: int,
    suite: str,
    task_id: int,
    init_state_id: int,
    *,
    sampling_mode: str = "without_replacement",
) -> int:
    """Return a stable numeric RNG seed without content hashing."""

    if (
        root_seed < 0
        or suite not in SUITE_ORDER
        or not 0 <= task_id < 10
        or sampling_mode not in WRITER_VIDEO_SAMPLING_MODES
        or init_state_id < 0
    ):
        raise WriterModelError("invalid K4 evaluation video seed key")
    mode_tag = 1 if sampling_mode == "with_replacement" else 2
    state = np.random.SeedSequence(
        [root_seed, SUITE_ORDER.index(suite), task_id, init_state_id, mode_tag, 0x4B34]
    ).generate_state(2, dtype=np.uint32)
    return (int(state[0]) << 31 | int(state[1])) & ((1 << 63) - 1)


def writer_video_demo_indices(
    root_seed: int,
    suite: str,
    task_id: int,
    init_state_id: int,
    *,
    demo_count: int = 50,
    sampling_mode: str = "without_replacement",
) -> tuple[int, ...]:
    """Choose four unique demos with balanced no-replacement shot lanes."""

    if demo_count < VIDEOS_PER_CONDITION:
        raise WriterModelError("K4 evaluation has too few teacher demos")
    if sampling_mode == "with_replacement":
        rng = np.random.default_rng(
            writer_video_selection_seed(
                root_seed,
                suite,
                task_id,
                init_state_id,
                sampling_mode=sampling_mode,
            )
        )
        return tuple(
            int(value)
            for value in rng.choice(
                demo_count, size=VIDEOS_PER_CONDITION, replace=False
            )
        )
    if sampling_mode != "without_replacement":
        raise WriterModelError("invalid K4 evaluation sampling mode")
    block, position = divmod(init_state_id, demo_count)
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [root_seed, SUITE_ORDER.index(suite), task_id, block, 0x4B34]
        )
    )
    permutation = rng.permutation(demo_count)
    result = tuple(
        int(permutation[(position + offset) % demo_count])
        for offset in K4_POSITION_OFFSETS
    )
    if len(set(result)) != VIDEOS_PER_CONDITION:
        raise WriterModelError("K4 evaluation set contains duplicate videos")
    return result


def writer_condition_demo_indices(
    root_seed: int,
    suite: str,
    task_id: int,
    init_state_id: int,
    *,
    condition: str,
    valid_conditions: set[str],
    demo_count: int = 50,
    sampling_mode: str = "without_replacement",
) -> tuple[int, ...]:
    """Apply a paired condition-only transform to the complete K4 set."""

    if condition not in valid_conditions:
        raise WriterModelError("invalid K4 evaluation video condition")
    reference = writer_video_demo_indices(
        root_seed,
        suite,
        task_id,
        init_state_id,
        demo_count=demo_count,
        sampling_mode=sampling_mode,
    )
    if condition == "same_task_other":
        shifted = tuple(
            (value + SAME_TASK_OTHER_DEMO_OFFSET) % demo_count
            for value in reference
        )
        if len(set(shifted)) != VIDEOS_PER_CONDITION:
            raise WriterModelError("same-task K4 control contains duplicates")
        return shifted
    return reference


def writer_video_schedule_contract(
    sampling_mode: str | None,
    *,
    seed: int,
    demo_count: int,
) -> tuple[dict[str, Any], str, str]:
    """Build the one canonical K4 schedule contract."""

    effective = sampling_mode or "without_replacement"
    if effective not in WRITER_VIDEO_SAMPLING_MODES or demo_count != 50:
        raise WriterModelError("invalid K4 Writer video schedule")
    algorithm = (
        WRITER_VIDEO_SCHEDULE_WITH_REPLACEMENT
        if effective == "with_replacement"
        else WRITER_VIDEO_SCHEDULE_WITHOUT_REPLACEMENT
    )
    return (
        {
            "algorithm": algorithm,
            "seed": int(seed),
            "demo_count": int(demo_count),
            "videos_per_condition": VIDEOS_PER_CONDITION,
            "sampling_mode": effective,
            "queue_order_independent": True,
            "paired_between_all_video_conditions": True,
        },
        "ember_pi05_k4_writer_eval_pairing_v1",
        effective,
    )
