from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.source_sft.contract import Pi05SourceSFTError
from ember.source_sft.validation import finalize_args as finalize_source_sft_args
from ember.writer.online_validation import OnlineWriterValidation, _online_summary
from ember.writer.validation_panel import (
    build_validation_loss_manifest,
    load_validation_loss_panel,
    summarize_validation_losses,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pi05_validation_functional_loss_panel_v1.json"


class _Dataset:
    def __init__(self, task_ids: tuple[int, ...]) -> None:
        self._frame_index: list[tuple[int, int, int]] = []
        self._episodes: dict[int, dict[int, tuple[int, ...]]] = {}
        for task_id in task_ids:
            task_episodes = {}
            for demo in range(50):
                rows = []
                for frame in range(3):
                    rows.append(len(self._frame_index))
                    self._frame_index.append((task_id, demo, frame))
                task_episodes[demo] = tuple(rows)
            self._episodes[task_id] = task_episodes

    @property
    def frame_index(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(self._frame_index)

    @property
    def task_episode_rows(self) -> dict[int, dict[int, tuple[int, ...]]]:
        return self._episodes


def test_validation_loss_panel_is_deterministic_balanced_and_unpaired() -> None:
    config = load_validation_loss_panel(CONFIG)
    task_ids = tuple(
        sorted(
            int(value)
            for value in __import__("json").loads(
                (
                    ROOT
                    / config["authorities"]["target_data_manifest"]["path"]
                ).read_text(encoding="utf-8")
            )["summary"]["roles"]["validation"]
        )
    )
    dataset = _Dataset(task_ids)
    left = build_validation_loss_manifest(dataset, config)
    right = build_validation_loss_manifest(dataset, config)
    assert left == right
    assert left["row_count"] == 512
    assert left["rows_per_task"] == 64
    assert all(
        row["teacher_demo_index"] != row["action_demo_index"]
        for row in left["rows"]
    )
    assert {
        task_id: sum(row["global_task_id"] == task_id for row in left["rows"])
        for task_id in task_ids
    } == {task_id: 64 for task_id in task_ids}


def test_validation_loss_summary_equal_weights_tasks() -> None:
    rows = [
        {"checkpoint_cursor": 100, "global_task_id": 1, "loss": 1.0},
        {"checkpoint_cursor": 100, "global_task_id": 1, "loss": 3.0},
        {"checkpoint_cursor": 100, "global_task_id": 2, "loss": 10.0},
    ]
    summary = summarize_validation_losses(rows)["checkpoints"][0]
    assert summary["per_task"]["1"]["mean_loss"] == 2.0
    assert summary["per_task"]["2"]["mean_loss"] == 10.0
    assert summary["task_balanced_mean_loss"] == 6.0


def test_source_sft_validation_panel_cannot_be_truncated_formally() -> None:
    values = {
        "panel_config": Path("panel.json"),
        "config": Path("source-sft.json"),
        "training_run": Path("training"),
        "checkpoints": [Path("checkpoint")],
        "source_run": Path("source"),
        "source_checkpoint": Path("source-checkpoint"),
        "tokenizer_path": Path("tokenizer"),
        "data_root": Path("data"),
        "output_dir": Path("output"),
    }
    with pytest.raises(Pi05SourceSFTError, match="cannot truncate"):
        finalize_source_sft_args(
            SimpleNamespace(
                **values,
                mode="formal",
                max_groups_per_task=1,
            )
        )

    profile = finalize_source_sft_args(
        SimpleNamespace(
            **values,
            mode="profile",
            max_groups_per_task=8,
        )
    )
    assert profile.max_groups_per_task == 8
    assert isinstance(profile.checkpoints, tuple)

    with pytest.raises(Pi05SourceSFTError, match="invalid"):
        finalize_source_sft_args(
            SimpleNamespace(
                **values,
                mode="profile",
                max_groups_per_task=9,
            )
        )


def test_online_validation_summary_records_checkpoint_slope(
    tmp_path: Path,
) -> None:
    validation = OnlineWriterValidation(
        panel={},
        manifest={},
        tasks=(),
        dataset=SimpleNamespace(),  # type: ignore[arg-type]
        store=SimpleNamespace(),  # type: ignore[arg-type]
        tokenizer=SimpleNamespace(),  # type: ignore[arg-type]
        output_dir=tmp_path,
        local_keys=(),
    )
    for cursor, loss in ((100, 2.0), (200, 1.5)):
        step = tmp_path / f"step_{cursor:08d}"
        step.mkdir()
        write_json_atomic(
            step / "rank_00_rows.json",
            {
                "rows": [
                    {
                        "ordinal": 0,
                        "checkpoint_cursor": cursor,
                        "global_task_id": 1,
                        "teacher_demo_index": 3,
                        "loss": loss,
                    }
                ]
            },
        )
        summary = _online_summary(validation, cursor, 1, 0.0)

    assert summary["previous_checkpoint_cursor"] == 100
    assert summary["loss_delta_from_previous"] == -0.5
    assert read_json(tmp_path / "step_00000200/summary.json") == summary
