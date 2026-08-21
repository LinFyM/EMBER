"""Strict paired one-shot teacher-video schedules for Expert-Manifold evaluation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ember.expert_manifold.contract import ExpertManifoldError
from ember.functional_adaptation.process_controls import (
    TEMPORAL_PROCESS_VIDEO_CONDITIONS,
    shuffled_frame_permutation,
)
from ember.pi05_target_data import SUITE_ORDER, target_global_task_id


SAME_TASK_OTHER_OFFSET = 17
VIDEO_SCHEDULE_SUITES = (*SUITE_ORDER, "libero_90")
VIDEO_CONDITIONS = {
    "correct",
    "same_task_other",
    "cross_suite_wrong",
    "no_video",
    *TEMPORAL_PROCESS_VIDEO_CONDITIONS,
}
_FUNCTIONAL_ONLY_CONDITIONS = {
    "wrong_task",
    "language_only",
    "video_only",
    "eye_in_hand_view",
}
_ALL_SCHEDULE_CONDITIONS = VIDEO_CONDITIONS | _FUNCTIONAL_ONLY_CONDITIONS
SAMPLING_MODES = {"with_replacement", "without_replacement"}


def _valid_video_task_key(suite: str, task_id: int) -> bool:
    return suite in VIDEO_SCHEDULE_SUITES and 0 <= task_id < (
        90 if suite == "libero_90" else 10
    )


def task_video_mapping(
    task_keys: Sequence[tuple[str, int]],
    task_roles: Mapping[tuple[str, int], str],
    condition: str,
) -> tuple[dict[str, Any], ...]:
    if condition not in _ALL_SCHEDULE_CONDITIONS or not task_keys:
        raise ExpertManifoldError("invalid Expert-Manifold video mapping")
    normalized = tuple((str(suite), int(task_id)) for suite, task_id in task_keys)
    if len(set(normalized)) != len(normalized):
        raise ExpertManifoldError("Expert-Manifold evaluation tasks are duplicated")
    selected = set(normalized)
    roles = sorted({str(task_roles.get(key, "")) for key in normalized})
    if not roles or "" in roles or set(task_roles) != selected:
        raise ExpertManifoldError("Expert-Manifold split-role mapping changed")
    result: list[dict[str, Any]] = []
    for role in roles:
        by_suite = {
            suite: tuple(
                sorted(
                    task_id
                    for name, task_id in normalized
                    if name == suite and task_roles[(name, task_id)] == role
                )
            )
            for suite in SUITE_ORDER
        }
        if condition == "cross_suite_wrong" and (
            any(not values for values in by_suite.values())
            or len({len(values) for values in by_suite.values()}) != 1
        ):
            raise ExpertManifoldError("cross-suite video control panel is unbalanced")
        for suite in SUITE_ORDER:
            for ordinal, task_id in enumerate(by_suite[suite]):
                video_suite, video_task_id = suite, task_id
                if condition == "cross_suite_wrong":
                    video_suite = SUITE_ORDER[
                        (SUITE_ORDER.index(suite) + 1) % len(SUITE_ORDER)
                    ]
                    video_task_id = by_suite[video_suite][ordinal]
                result.append(
                    {
                        "suite": suite,
                        "task_id": task_id,
                        "language_global_task_id": target_global_task_id(
                            suite, task_id
                        ),
                        "language_split_role": role,
                        "video_suite": video_suite,
                        "video_task_id": video_task_id,
                        "video_global_task_id": target_global_task_id(
                            video_suite, video_task_id
                        ),
                        "video_split_role": role,
                    }
                )
    return tuple(
        sorted(
            result,
            key=lambda row: (SUITE_ORDER.index(row["suite"]), row["task_id"]),
        )
    )


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
        or not _valid_video_task_key(suite, task_id)
        or init_state_id < 0
        or sampling_mode not in SAMPLING_MODES
    ):
        raise ExpertManifoldError("invalid one-shot video selection key")
    tag = 1 if sampling_mode == "with_replacement" else 2
    values = np.random.SeedSequence(
        [
            root_seed,
            VIDEO_SCHEDULE_SUITES.index(suite),
            task_id,
            init_state_id,
            tag,
            0xE901,
        ]
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
            [
                root_seed,
                VIDEO_SCHEDULE_SUITES.index(suite),
                task_id,
                block,
                0xE901,
            ]
        )
    )
    return int(rng.permutation(demo_count)[position])


def reference_demo_indices(
    root_seed: int,
    suite: str,
    task_id: int,
    init_state_id: int,
    *,
    demo_count: int,
    sampling_mode: str,
    video_count: int,
) -> tuple[int, ...]:
    """Return a nested, within-set unique prefix of the paired video schedule."""

    if video_count == 1:
        return (
            reference_demo_index(
                root_seed,
                suite,
                task_id,
                init_state_id,
                demo_count=demo_count,
                sampling_mode=sampling_mode,
            ),
        )
    if (
        demo_count != 50
        or not 1 < video_count <= demo_count
        or sampling_mode != "without_replacement"
    ):
        raise ExpertManifoldError(
            "multi-video evaluation requires the 50-video permutation schedule"
        )
    block, position = divmod(init_state_id, demo_count)
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                root_seed,
                VIDEO_SCHEDULE_SUITES.index(suite),
                task_id,
                block,
                0xE901,
            ]
        )
    )
    permutation = rng.permutation(demo_count)
    return tuple(
        int(permutation[(position + offset) % demo_count])
        for offset in range(video_count)
    )


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
    if condition not in _ALL_SCHEDULE_CONDITIONS:
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


def paired_condition_demo_indices(
    root_seed: int,
    suite: str,
    task_id: int,
    init_state_id: int,
    condition: str,
    demo_count: int,
    sampling_mode: str,
    video_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the reference and selected ordered K-video sets.

    K1 preserves the historical numeric demo offset.  For K>1, the
    same-task-other arm shifts the start position in the same deterministic
    task permutation, yielding a disjoint K-set without changing any other
    condition's reference videos.
    """

    if condition not in _ALL_SCHEDULE_CONDITIONS:
        raise ExpertManifoldError("invalid video-set condition")
    reference = reference_demo_indices(
        root_seed,
        suite,
        task_id,
        init_state_id,
        demo_count=demo_count,
        sampling_mode=sampling_mode,
        video_count=video_count,
    )
    if condition != "same_task_other":
        return reference, reference
    if video_count == 1:
        selected = ((reference[0] + SAME_TASK_OTHER_OFFSET) % demo_count,)
        return reference, selected
    if demo_count != 50 or sampling_mode != "without_replacement":
        raise ExpertManifoldError(
            "multi-video same-task-other requires the 50-video permutation schedule"
        )
    block, position = divmod(init_state_id, demo_count)
    permutation = np.random.default_rng(
        np.random.SeedSequence(
            [
                root_seed,
                VIDEO_SCHEDULE_SUITES.index(suite),
                task_id,
                block,
                0xE901,
            ]
        )
    ).permutation(demo_count)
    selected = tuple(
        int(permutation[(position + SAME_TASK_OTHER_OFFSET + offset) % demo_count])
        for offset in range(video_count)
    )
    if len(set(selected)) != video_count or set(selected) & set(reference):
        raise ExpertManifoldError("same-task-other video set is not disjoint")
    return reference, selected


def condition_demo_indices(
    root_seed: int,
    suite: str,
    task_id: int,
    init_state_id: int,
    *,
    condition: str,
    demo_count: int,
    sampling_mode: str,
    video_count: int,
) -> tuple[int, ...]:
    """Return only the selected side of a paired K-video schedule."""

    return paired_condition_demo_indices(
        root_seed,
        suite,
        task_id,
        init_state_id,
        condition=condition,
        demo_count=demo_count,
        sampling_mode=sampling_mode,
        video_count=video_count,
    )[1]


def frame_order_seed(
    root_seed: int,
    suite: str,
    task_id: int,
    demo_index: int,
) -> int:
    if (
        root_seed < 0
        or not _valid_video_task_key(suite, task_id)
        or not 0 <= demo_index < 50
    ):
        raise ExpertManifoldError("invalid one-shot frame-order key")
    values = np.random.SeedSequence(
        [
            root_seed,
            VIDEO_SCHEDULE_SUITES.index(suite),
            task_id,
            demo_index,
            0xE902,
        ]
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
