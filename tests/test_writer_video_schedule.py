from __future__ import annotations

from ember.writer.inference import (
    WRITER_ADAPTER_SCHEMA,
    WRITER_EPISODE_EVIDENCE,
    _task_video_mapping,
    expected_writer_episode_evidence,
)
from ember.writer.video_schedule import writer_video_demo_indices


def _adapter(condition: str) -> dict:
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    keys = tuple((suite, 0) for suite in suites)
    mapping = list(_task_video_mapping(keys, {key: "train" for key in keys}, condition))
    return {
        "schema_version": WRITER_ADAPTER_SCHEMA,
        "kind": "as_writer",
        "writer_method": "as_writer",
        "arm": f"as_writer_{condition}_video",
        "video_condition": condition,
        "checkpoint": {"cursor": 12, "cursor_axis": "optimizer_step", "reference": "run:12"},
        "lora_contract": {"reference": "rank16:76tensors"},
        "video_schedule": {
            "seed": 7,
            "demo_count": 50,
            "videos_per_condition": 4,
            "sampling_mode": "without_replacement",
        },
        "task_video_mapping_reference": "next_suite_v1",
        "task_video_mapping": mapping,
        "pairing_reference": "paired_k4_v1",
    }


def test_k4_without_replacement_is_unique_balanced_and_resumable() -> None:
    first = [
        writer_video_demo_indices(
            7, "libero_spatial", 1, state, sampling_mode="without_replacement"
        )
        for state in range(50)
    ]
    resumed = [
        writer_video_demo_indices(
            7, "libero_spatial", 1, state, sampling_mode="without_replacement"
        )
        for state in reversed(range(50))
    ]
    assert list(reversed(resumed)) == first
    assert all(len(set(demos)) == 4 for demos in first)
    for lane in range(4):
        assert sorted(demos[lane] for demos in first) == list(range(50))


def test_episode_evidence_pairs_complete_k4_sets_and_order_seeds() -> None:
    correct = _adapter("correct")
    shuffled = _adapter("shuffled")
    correct_rows = [
        expected_writer_episode_evidence(
            correct,
            suite="libero_spatial",
            task_id=0,
            init_state_id=state,
            lora_reference=f"lora:{state}",
        )
        for state in range(50)
    ]
    shuffled_rows = [
        expected_writer_episode_evidence(
            shuffled,
            suite="libero_spatial",
            task_id=0,
            init_state_id=state,
            lora_reference=f"lora:{state}",
        )
        for state in range(50)
    ]
    assert {row["schema_version"] for row in correct_rows} == {WRITER_EPISODE_EVIDENCE}
    assert [row["teacher_demo_indices"] for row in shuffled_rows] == [
        row["teacher_demo_indices"] for row in correct_rows
    ]
    assert [row["teacher_video_order_seeds"] for row in shuffled_rows] == [
        row["teacher_video_order_seeds"] for row in correct_rows
    ]


def test_same_task_other_changes_all_four_videos_without_duplicates() -> None:
    correct = expected_writer_episode_evidence(
        _adapter("correct"),
        suite="libero_spatial",
        task_id=0,
        init_state_id=3,
        lora_reference="correct",
    )
    other = expected_writer_episode_evidence(
        _adapter("same_task_other"),
        suite="libero_spatial",
        task_id=0,
        init_state_id=3,
        lora_reference="other",
    )
    assert len(set(other["teacher_demo_indices"])) == 4
    assert set(correct["teacher_demo_indices"]).isdisjoint(other["teacher_demo_indices"])
