from __future__ import annotations

from pathlib import Path

import torch

from ember.writer.data import WriterTaskAuthority
from ember.writer.feature_cache import (
    FeatureCacheTask,
    balanced_task_assignments,
    load_feature_cache_config,
    load_train_tasks,
    load_task_cache,
    pool_visual_tokens,
    save_task_cache,
    select_language_tokens,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _task(task_id: int, frames: int) -> FeatureCacheTask:
    authority = WriterTaskAuthority(task_id, f"task {task_id}", Path("unused"), 1)
    return FeatureCacheTask(task_id, authority.language, authority, "0" * 64, (frames,))


def test_config_and_lpt_schedule_cover_every_task_once() -> None:
    config = load_feature_cache_config(
        REPO_ROOT / "configs/writer_feature_cache_v1.json", REPO_ROOT
    )
    assert config["features"]["vision_feature_dim"] == 960
    tasks = tuple(_task(index, frames) for index, frames in enumerate((9, 8, 7, 6, 5, 4, 3, 2)))
    assignments = balanced_task_assignments(tasks, 3)
    assigned = [task.task_id for rank_tasks in assignments for task in rank_tasks]
    assert sorted(assigned) == list(range(8))
    loads = [sum(task.frame_count for task in rank_tasks) for rank_tasks in assignments]
    assert max(loads) - min(loads) <= max(task.frame_count for task in tasks)

    sealed_tasks = load_train_tasks(config, REPO_ROOT, Path("/not-read-by-schedule"))
    sealed_assignments = balanced_task_assignments(sealed_tasks, 8)
    sealed_loads = [
        sum(task.frame_count for task in rank_tasks) for rank_tasks in sealed_assignments
    ]
    assert len(sealed_tasks) == 70 and sum(sealed_loads) == 537_946
    assert max(sealed_loads) <= (sum(sealed_loads) / 8) * 1.03


def test_visual_pool_and_language_mask_match_sealed_math() -> None:
    visual = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    pooled = pool_visual_tokens(visual, expected_tokens=4, expected_dim=3)
    torch.testing.assert_close(pooled, visual.mean(dim=1) * (3**0.5))
    language = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    selected = select_language_tokens(
        language, torch.tensor([[True, True, False, False]]), expected_dim=3
    )
    torch.testing.assert_close(selected, language[0, :2] * (3**0.5))


def test_task_cache_roundtrip_preserves_episode_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "task.safetensors"
    record = save_task_cache(
        path,
        language_features=torch.randn(3, 5),
        video_features=torch.randn(7, 5),
        episode_offsets=torch.tensor([0, 2, 7]),
        demo_indices=torch.tensor([4, 9]),
        metadata={"schema_version": "test"},
    )
    cached = load_task_cache(path, expected_dim=5)
    assert record["frames"] == 7 and record["episodes"] == 2
    assert cached.video_features.dtype == torch.bfloat16
    assert cached.episode_offsets.tolist() == [0, 2, 7]
    assert cached.demo_indices.tolist() == [4, 9]
