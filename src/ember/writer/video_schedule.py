"""Deterministic teacher-video pairing schedules for Writer evaluation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ember.pi05_target_data import SUITE_ORDER, target_global_task_id
from ember.writer.as_sampling import TeacherVideoSchedule
from ember.writer.model import WriterModelError


SAME_TASK_OTHER_DEMO_OFFSET = 17
WRITER_VIDEO_SAMPLING_MODES = {"with_replacement", "without_replacement"}
WRITER_VIDEO_SCHEDULE_WITH_REPLACEMENT = (
    "sha256 first 63 bits of canonical JSON "
    "[ember_pi05_writer_video_v1,seed,suite,task_id,init_state_id] modulo 50"
)
WRITER_VIDEO_SCHEDULE_WITHOUT_REPLACEMENT = (
    "TeacherVideoSchedule(seed, target_global_task_id, demos 0..49): each "
    "consecutive demo_count init-state block is a seeded no-replacement permutation"
)


def writer_video_selection_seed(
    root_seed: int,
    suite: str,
    task_id: int,
    init_state_id: int,
    *,
    sampling_mode: str = "with_replacement",
) -> int:
    if (
        root_seed < 0
        or suite not in SUITE_ORDER
        or not 0 <= task_id < 10
        or sampling_mode not in WRITER_VIDEO_SAMPLING_MODES
        or init_state_id < 0
    ):
        raise WriterModelError("invalid AS-Writer evaluation video seed key")
    namespace = (
        "ember_pi05_writer_video_v1"
        if sampling_mode == "with_replacement"
        else "ember_pi05_writer_video_permutation_v1"
    )
    encoded = json.dumps(
        [namespace, root_seed, suite, task_id, init_state_id],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def writer_video_demo_index(
    root_seed: int,
    suite: str,
    task_id: int,
    init_state_id: int,
    *,
    demo_count: int = 50,
    sampling_mode: str = "with_replacement",
) -> int:
    """Choose one demo independently of queue and worker execution order."""

    if demo_count <= 0 or sampling_mode not in WRITER_VIDEO_SAMPLING_MODES:
        raise WriterModelError("invalid AS-Writer evaluation video count")
    if sampling_mode == "with_replacement":
        return (
            writer_video_selection_seed(
                root_seed,
                suite,
                task_id,
                init_state_id,
                sampling_mode=sampling_mode,
            )
            % demo_count
        )
    global_task_id = target_global_task_id(suite, task_id)
    return TeacherVideoSchedule(
        task_ids=(global_task_id,),
        demo_indices=range(demo_count),
        seed=root_seed,
    ).demo_for_task_visit(global_task_id, init_state_id)


def writer_condition_demo_index(
    root_seed: int,
    suite: str,
    task_id: int,
    init_state_id: int,
    *,
    condition: str,
    valid_conditions: set[str],
    demo_count: int = 50,
    sampling_mode: str = "with_replacement",
) -> int:
    """Apply a deterministic condition-only transform to the paired demo."""

    if condition not in valid_conditions:
        raise WriterModelError("invalid AS-Writer evaluation video condition")
    reference = writer_video_demo_index(
        root_seed,
        suite,
        task_id,
        init_state_id,
        demo_count=demo_count,
        sampling_mode=sampling_mode,
    )
    if condition == "same_task_other":
        return (reference + SAME_TASK_OTHER_DEMO_OFFSET) % demo_count
    return reference


def writer_video_schedule_contract(
    sampling_mode: str | None,
    *,
    seed: int,
    demo_count: int,
) -> tuple[dict[str, Any], str, str]:
    """Build a backward-compatible sealed schedule and pairing schema."""

    legacy = sampling_mode is None
    effective = sampling_mode or "with_replacement"
    if effective not in WRITER_VIDEO_SAMPLING_MODES:
        raise WriterModelError("invalid PI05 Writer video sampling mode")
    algorithm = (
        WRITER_VIDEO_SCHEDULE_WITH_REPLACEMENT
        if effective == "with_replacement"
        else WRITER_VIDEO_SCHEDULE_WITHOUT_REPLACEMENT
    )
    schedule = {
        "algorithm": algorithm,
        "seed": seed,
        "demo_count": demo_count,
        "queue_order_independent": True,
        "paired_between_correct_and_wrong": True,
    }
    if not legacy:
        schedule["sampling_mode"] = effective
    pairing_schema = (
        "ember_pi05_writer_eval_pairing_v2"
        if legacy
        else "ember_pi05_writer_eval_pairing_v3"
    )
    return schedule, pairing_schema, effective
