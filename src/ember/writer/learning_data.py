"""Train24 sampling for fresh joint video-to-LoRA learning.

Task and episode identities are orchestration metadata. The model receives only
the returned RGB arrays, real frame indices and exact task language.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import default_collate

from ember.pi05_source_checkpoint import read_json
from ember.writer.data import FunctionalQueryDataset, RawTeacherVideoStore, WriterTaskAuthority
from ember.writer.functional import task_logical_batch_policy_rng_seed
from ember.writer.task_schedule import counted_task_group, task_occurrence_schedule


@dataclass(frozen=True)
class LearningTask:
    authority: WriterTaskAuthority
    suite: str
    suite_task_id: int
    episode_lengths: tuple[int, ...]


def load_learning_tasks(
    asset_root: Path, task_ids: Sequence[int], *, role: str = "train",
) -> dict[int, LearningTask]:
    if role not in {"train", "validation"}:
        raise ValueError("current development loader excludes Test")
    manifest = read_json(asset_root / "configs/pi05_target_data_v1/manifest.json")
    protocol = read_json(asset_root / "configs/libero_24_8_8_v1/protocol.json")
    selected = tuple(map(int, task_ids))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("learning tasks must be explicit and unique")
    rows = {int(row["global_task_id"]): row for row in manifest["tasks"]}
    output = {}
    data_root = asset_root / "data/datasets" / manifest["dataset"]["revision"]
    for task_id in selected:
        row = rows[task_id]
        suite, local = row["suite"], int(row["task_id"])
        if row["split_role"] != role or local not in protocol["split"]["suites"][suite][role]:
            raise ValueError("selected task crosses the fixed development split")
        authority = WriterTaskAuthority(
            task_id, str(row["language"]), data_root / row["hdf5"]["relative_path"], int(row["hdf5"]["bytes"]),
        )
        output[task_id] = LearningTask(
            authority, suite, local, tuple(map(int, row["demonstrations"]["episode_lengths"])),
        )
    return output


class JointTrainingData:
    """Stateless task occurrences and hierarchical cross-episode queries."""

    def __init__(self, asset_root: Path, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.seed = int(config["seed"])
        self.tasks = load_learning_tasks(asset_root, config["task_ids"])
        self.video_pool = tuple(map(int, config["video_demos"]))
        self.action_pool = tuple(map(int, config["action_demos"]))
        self.diagnostic_pool = tuple(map(int, config["diagnostic_action_demos"]))
        self.held_video_pool = tuple(map(int, config["held_video_demos"]))
        pools = (self.video_pool, self.action_pool, self.diagnostic_pool, self.held_video_pool)
        flat = [demo for pool in pools for demo in pool]
        if any(not pool for pool in pools) or len(set(flat)) != len(flat) or not set(flat) <= set(range(50)):
            raise ValueError("training video/query and diagnostic episode roles must be disjoint")
        if tuple(config["cardinalities"]) != (1, 2, 4) or min(len(self.video_pool), len(self.held_video_pool)) < 4:
            raise ValueError("dynamic K requires actual K1/2/4 and at least four unique videos")
        authorities = tuple(task.authority for task in self.tasks.values())
        self.videos = RawTeacherVideoStore(authorities, frame_stride=5)
        self.queries = FunctionalQueryDataset(authorities, demo_indices=self.action_pool, action_chunk_size=50)
        self.query_rows = self.queries.task_episode_rows
        self.groups = tuple(counted_task_group(
            (tuple(self.tasks),), (int(config["tasks_per_update"]),), step, seed=self.seed,
        ) for step in range(int(config["total_steps"])))
        self.occurrences = task_occurrence_schedule(self.groups)

    def video_demos(self, task: int, occurrence: int, *, cardinality: int | None = None) -> tuple[int, ...]:
        k = (1, 2, 4)[(occurrence + task) % 3] if cardinality is None else cardinality
        if k not in (1, 2, 4):
            raise ValueError("unsupported training cardinality")
        rng = random.Random(self.seed + 1000003 * task + 7919 * occurrence)
        return tuple(rng.sample(self.video_pool, k))

    def load_videos(self, task: int, demos: Sequence[int]):
        if len(set(demos)) != len(demos):
            raise ValueError("teaching condition repeats a video")
        videos = tuple(self.videos.load(task, demo) for demo in demos)
        return (
            tuple(torch.from_numpy(video.frames) for video in videos),
            tuple(torch.from_numpy(video.frame_indices) for video in videos),
        )

    def action_batch(self, task: int, occurrence: int, demos: Sequence[int]):
        if set(demos) & set(self.action_pool):
            raise ValueError("teaching video and action query episodes overlap")
        rng = random.Random(self.seed + 32452843 * task + 49999 * occurrence)
        rows = []
        for _ in range(int(self.config["queries_per_task"])):
            episode = rng.choice(self.action_pool)
            rows.append(self.queries[rng.choice(self.query_rows[task][episode])])
        seed = task_logical_batch_policy_rng_seed(
            optimization_seed=self.seed, task_id=task, task_visit=occurrence,
            demo_indices=[row["demo_index"] for row in rows],
            frame_indices=[row["frame_index"] for row in rows],
        )
        return default_collate(rows), {
            "action_demos": [row["demo_index"] for row in rows],
            "action_frames": [row["frame_index"] for row in rows],
            "policy_rng_seed": seed,
        }

    def step_costs(self, step: int) -> dict[int, int]:
        return {
            task: sum(self.videos.frame_counts(task, demo)[1] for demo in self.video_demos(task, self.occurrences[step][task]))
            for task in self.groups[step]
        }

    def sampler_state(self, next_step: int) -> dict[str, Any]:
        counts = {task: 0 for task in self.tasks}
        for group in self.groups[:next_step]:
            for task in group:
                counts[task] += 1
        return {"next_step": next_step, "task_occurrences": counts, "seed": self.seed}

    def close(self) -> None:
        self.videos.close()
        self.queries.close()
