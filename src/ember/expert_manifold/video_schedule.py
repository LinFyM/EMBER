"""Strict paired one-shot teacher-video schedules for Expert-Manifold evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ember.expert_manifold.contract import ExpertManifoldError
from ember.pi05_target_data import SUITE_ORDER


SAME_TASK_OTHER_OFFSET = 17
VIDEO_CONDITIONS = {
    "correct",
    "same_task_other",
    "cross_suite_wrong",
    "shuffled",
    "shuffled_keep_first",
    "reversed",
    "no_video",
}
SAMPLING_MODES = {"with_replacement", "without_replacement"}


def video_selection_seed(
    root_seed: int,
    suite: str,
    task_id: int,
    init_state_id: int,
    *,
    sampling_mode: str,
) -> int:
    if (
        root_seed < 0
        or suite not in SUITE_ORDER
        or not 0 <= task_id < 10
        or init_state_id < 0
        or sampling_mode not in SAMPLING_MODES
    ):
        raise ExpertManifoldError("invalid one-shot video selection key")
    tag = 1 if sampling_mode == "with_replacement" else 2
    values = np.random.SeedSequence(
        [root_seed, SUITE_ORDER.index(suite), task_id, init_state_id, tag, 0xE901]
    ).generate_state(2, dtype=np.uint32)
    return (int(values[0]) << 31 | int(values[1])) & ((1 << 63) - 1)


def reference_demo_index(
    root_seed: int,
    suite: str,
    task_id: int,
    init_state_id: int,
    *,
    demo_count: int,
    sampling_mode: str,
) -> int:
    if demo_count != 50:
        raise ExpertManifoldError("one-shot evaluation requires all 50 teacher videos")
    if sampling_mode == "with_replacement":
        rng = np.random.default_rng(
            video_selection_seed(
                root_seed,
                suite,
                task_id,
                init_state_id,
                sampling_mode=sampling_mode,
            )
        )
        return int(rng.integers(0, demo_count))
    if sampling_mode != "without_replacement":
        raise ExpertManifoldError("invalid one-shot video sampling mode")
    block, position = divmod(init_state_id, demo_count)
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [root_seed, SUITE_ORDER.index(suite), task_id, block, 0xE901]
        )
    )
    return int(rng.permutation(demo_count)[position])


def condition_demo_index(
    root_seed: int,
    suite: str,
    task_id: int,
    init_state_id: int,
    *,
    condition: str,
    demo_count: int,
    sampling_mode: str,
) -> int:
    if condition not in VIDEO_CONDITIONS:
        raise ExpertManifoldError("invalid one-shot video condition")
    reference = reference_demo_index(
        root_seed,
        suite,
        task_id,
        init_state_id,
        demo_count=demo_count,
        sampling_mode=sampling_mode,
    )
    return (
        (reference + SAME_TASK_OTHER_OFFSET) % demo_count
        if condition == "same_task_other"
        else reference
    )


def frame_order_seed(
    root_seed: int,
    suite: str,
    task_id: int,
    demo_index: int,
) -> int:
    if (
        root_seed < 0
        or suite not in SUITE_ORDER
        or not 0 <= task_id < 10
        or not 0 <= demo_index < 50
    ):
        raise ExpertManifoldError("invalid one-shot frame-order key")
    values = np.random.SeedSequence(
        [root_seed, SUITE_ORDER.index(suite), task_id, demo_index, 0xE902]
    ).generate_state(2, dtype=np.uint32)
    return (int(values[0]) << 31 | int(values[1])) & ((1 << 63) - 1)


def video_schedule_contract(
    *, seed: int, demo_count: int, sampling_mode: str
) -> tuple[dict[str, Any], str]:
    if sampling_mode not in SAMPLING_MODES or demo_count != 50:
        raise ExpertManifoldError("invalid one-shot video schedule")
    algorithm = (
        "numeric_seeded_one_video_per_rollout"
        if sampling_mode == "with_replacement"
        else "numeric_seeded_task_permutation_one_video_per_state_block"
    )
    return (
        {
            "algorithm": algorithm,
            "seed": int(seed),
            "demo_count": demo_count,
            "videos_per_condition": 1,
            "sampling_mode": sampling_mode,
            "queue_order_independent": True,
            "paired_between_all_video_conditions": True,
        },
        "ember_pi05_expert_manifold_one_shot_pairing_v1",
    )
