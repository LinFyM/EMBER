from __future__ import annotations

from ember.pi05_source_checkpoint import canonical_hash
from ember.writer.inference import (
    WRITER_ADAPTER_SCHEMA,
    _task_video_mapping,
    expected_writer_episode_evidence,
)
from ember.writer.video_schedule import writer_video_demo_index


def _adapter(condition: str) -> dict:
    keys = tuple(
        (suite, 0)
        for suite in (
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
        )
    )
    roles = {key: "train" for key in keys}
    mapping = list(_task_video_mapping(keys, roles, condition))
    return {
        "schema_version": WRITER_ADAPTER_SCHEMA,
        "kind": "as_writer",
        "writer_method": "as_writer",
        "arm": f"as_writer_{condition}_video",
        "video_condition": condition,
        "checkpoint": {
            "cursor": 12,
            "cursor_axis": "optimizer_step",
            "manifest_file_sha256": "3" * 64,
            "writer_state_sha256": "4" * 64,
        },
        "lora_contract_sha256": "5" * 64,
        "video_schedule": {
            "seed": 7,
            "demo_count": 50,
            "sampling_mode": "without_replacement",
        },
        "task_video_mapping_sha256": canonical_hash(mapping),
        "task_video_mapping": mapping,
        "pairing_sha256": "6" * 64,
    }


def test_without_replacement_schedule_is_complete_and_resumable() -> None:
    def cycle(start: int) -> list[int]:
        return [
            writer_video_demo_index(
                7,
                "libero_spatial",
                1,
                state,
                sampling_mode="without_replacement",
            )
            for state in range(start, start + 50)
        ]

    first = cycle(0)
    resumed = [
        writer_video_demo_index(
            7,
            "libero_spatial",
            1,
            state,
            sampling_mode="without_replacement",
        )
        for state in reversed(range(50))
    ]
    second = cycle(50)
    assert sorted(first) == sorted(second) == list(range(50))
    assert list(reversed(resumed)) == first
    assert first != second


def test_episode_evidence_seals_no_replacement_pairing() -> None:
    correct = _adapter("correct")
    shuffled = _adapter("shuffled")

    def rows(adapter: dict) -> list[dict]:
        return [
            expected_writer_episode_evidence(
                adapter,
                suite="libero_spatial",
                task_id=0,
                init_state_id=state,
                lora_sha256="7" * 64,
            )
            for state in range(50)
        ]

    correct_rows = rows(correct)
    shuffled_rows = rows(shuffled)
    assert {row["schema_version"] for row in correct_rows} == {
        "ember_pi05_semantic_program_grid_episode_evidence_v1"
    }
    assert {row["teacher_video_sampling_mode"] for row in correct_rows} == {
        "without_replacement"
    }
    assert sorted(row["teacher_demo_index"] for row in correct_rows) == list(range(50))
    assert [
        row["teacher_demo_index"] for row in shuffled_rows
    ] == [row["teacher_demo_index"] for row in correct_rows]
    assert [
        row["teacher_video_order_seed"] for row in shuffled_rows
    ] == [row["teacher_video_order_seed"] for row in correct_rows]
