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
    """Independent persisted sampling streams; rejected updates consume draws."""

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
        root = random.Random(self.seed)
        self.streams = {name: random.Random(root.getrandbits(63)) for name in (
            "task", "K", "video", "query", "initial_state", "environment", "flow", "exploration", "reservoir", "trust",
        )}
        self.suites = {suite: tuple(task for task in self.tasks if self.tasks[task].suite == suite)
                       for suite in sorted({task.suite for task in self.tasks.values()})}
        if len(self.suites) != 4 or int(config["tasks_per_update"]) != 4:
            raise ValueError("joint updates require one task from each of four suites")
        self.next_step = 0
        self.counts = {task: 0 for task in self.tasks}

    def next_iteration(self) -> tuple[dict[str, Any], ...]:
        draws = []
        for tasks in self.suites.values():
            task = self.streams["task"].choice(tasks)
            k = self.streams["K"].choice((1, 2, 4))
            demos = tuple(self.streams["video"].sample(self.video_pool, k))
            draws.append({
                "task": task, "occurrence": self.counts[task], "video_demos": demos,
                "query_seed": self.streams["query"].getrandbits(63),
                "episodes": tuple({
                    "initial_state": self.streams["initial_state"].randrange(32),
                    **{name: self.streams[name].getrandbits(31) for name in
                       ("environment", "flow", "exploration", "reservoir", "trust")},
                } for _ in range(4)),
                "frames": sum(self.videos.frame_counts(task, demo)[1] for demo in demos),
            })
            self.counts[task] += 1
        self.next_step += 1
        return tuple(draws)

    def load_videos(self, task: int, demos: Sequence[int]):
        if len(set(demos)) != len(demos):
            raise ValueError("teaching condition repeats a video")
        videos = tuple(self.videos.load(task, demo) for demo in demos)
        return (
            tuple(torch.from_numpy(video.frames) for video in videos),
            tuple(torch.from_numpy(video.frame_indices) for video in videos),
        )

    def action_batch(self, task: int, occurrence: int, demos: Sequence[int], *, query_seed: int):
        if set(demos) & set(self.action_pool):
            raise ValueError("teaching video and action query episodes overlap")
        rng = random.Random(query_seed)
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

    def sampler_state(self) -> dict[str, Any]:
        return {"next_step": self.next_step, "task_occurrences": dict(self.counts), "seed": self.seed,
                "streams": {name: rng.getstate() for name, rng in self.streams.items()}}

    def restore_sampler(self, state: Mapping[str, Any]) -> None:
        if state["seed"] != self.seed or set(state["streams"]) != set(self.streams):
            raise ValueError("sampling stream contract changed")
        self.next_step = int(state["next_step"])
        self.counts = {int(task): int(count) for task, count in state["task_occurrences"].items()}
        if set(self.counts) != set(self.tasks) or sum(self.counts.values()) != self.next_step * 4:
            raise ValueError("sampler exposure cursor changed")
        for name, rng in self.streams.items():
            rng.setstate(state["streams"][name])

    def close(self) -> None:
        self.videos.close()
        self.queries.close()
