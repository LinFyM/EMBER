"""Train24 sampling for supervised video-to-LoRA learning.

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


class WriterTrainingData:
    """Independent persisted task, video and action sampling streams."""

    def __init__(self, asset_root: Path, config: Mapping[str, Any], *, camera_view: str = "agentview") -> None:
        self.config = dict(config)
        self.seed = int(config["seed"])
        self.camera_view = str(camera_view)
        self.conditions_per_task = config.get("conditions_per_task")
        if type(self.conditions_per_task) is not int or self.conditions_per_task not in (1, 2):
            raise ValueError("conditions_per_task must explicitly be 1 or 2")
        self.tasks = load_learning_tasks(asset_root, config["task_ids"])
        self.video_pool = tuple(map(int, config["video_demos"]))
        self.action_pool = tuple(map(int, config["action_demos"]))
        self.diagnostic_pool = tuple(map(int, config["diagnostic_action_demos"]))
        self.held_video_pool = tuple(map(int, config["held_video_demos"]))
        pools = (self.video_pool, self.action_pool, self.diagnostic_pool, self.held_video_pool)
        flat = [demo for pool in pools for demo in pool]
        if any(not pool for pool in pools) or len(set(flat)) != len(flat) or not set(flat) <= set(range(50)):
            raise ValueError("training video/query and diagnostic episode roles must be disjoint")
        if tuple(config["cardinalities"]) != (1,):
            raise ValueError("the current supervised stage requires actual K=1 conditions")
        authorities = tuple(task.authority for task in self.tasks.values())
        self.videos = RawTeacherVideoStore(authorities, frame_stride=5, camera_view=camera_view)
        self.queries = FunctionalQueryDataset(authorities, demo_indices=self.action_pool, action_chunk_size=50)
        self.query_rows = self.queries.task_episode_rows
        self.diagnostic_queries = None
        root = random.Random(self.seed)
        self.streams = {name: random.Random(root.getrandbits(63)) for name in (
            "task", "video", "query",
        )}
        self.suites = {suite: tuple(task for task in self.tasks if self.tasks[task].suite == suite)
                       for suite in sorted({task.suite for task in self.tasks.values()})}
        if len(self.suites) != 4 or int(config["tasks_per_update"]) != 4:
            raise ValueError("supervised updates require one task from each of four suites")
        self.next_step = 0
        self.counts = {task: 0 for task in self.tasks}

    def next_iteration(self) -> tuple[dict[str, Any], ...]:
        draws = []
        for tasks in self.suites.values():
            task = self.streams["task"].choice(tasks)
            demos = self.streams["video"].sample(self.video_pool, self.conditions_per_task)
            query_seed = self.streams["query"].getrandbits(63)
            query_count = int(self.config["queries_per_task"]) // self.conditions_per_task
            for condition_index, demo in enumerate(demos):
                draws.append({
                    "job_id": len(draws), "condition_index": condition_index,
                    "task": task, "occurrence": self.counts[task], "video_demos": (demo,),
                    "query_seed": query_seed, "query_offset": condition_index * query_count,
                    "query_count": query_count, "frames": self.videos.frame_counts(task, demo)[1],
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

    def action_batch(self, task: int, occurrence: int, demos: Sequence[int], *, query_seed: int,
                     query_offset: int = 0, query_count: int | None = None):
        if set(demos) & set(self.action_pool):
            raise ValueError("teaching video and action query episodes overlap")
        return self._sample_actions(self.queries, self.action_pool, task, occurrence, query_seed,
                                    int(self.config["queries_per_task"]),
                                    query_offset=query_offset, query_count=query_count)

    def _diagnostic_dataset(self) -> FunctionalQueryDataset:
        if self.diagnostic_queries is None:
            self.diagnostic_queries = FunctionalQueryDataset(
                tuple(task.authority for task in self.tasks.values()),
                demo_indices=self.diagnostic_pool, action_chunk_size=50,
            )
        return self.diagnostic_queries

    def diagnostic_batch(self, task: int, *, seed: int, count: int):
        return self._sample_actions(self._diagnostic_dataset(), self.diagnostic_pool, task, 0, seed, count)

    def local_action_clip(
        self, task: int, occurrence: int, *, diagnostic: bool = False,
    ) -> tuple[tuple[torch.Tensor], tuple[torch.Tensor], torch.Tensor, dict[str, Any]]:
        """Sample a training-only RGB/action pair without consuming persisted streams."""
        if task not in self.tasks or type(occurrence) is not int or occurrence < 0:
            raise ValueError("local action task or occurrence is outside training authority")
        pool = self.diagnostic_pool if diagnostic else self.action_pool
        expected_pool = tuple(range(42, 46)) if diagnostic else tuple(range(16, 42))
        if pool != expected_pool or (diagnostic and occurrence >= 16):
            raise ValueError("local action episode roles or diagnostic occurrence changed")
        seed = 20260913 if diagnostic else self.seed
        rng = random.Random(f"local-action:{seed}:{task}:{occurrence}:{int(diagnostic)}")
        demo = pool[occurrence // 4] if diagnostic else rng.choice(pool)
        length = self.tasks[task].episode_lengths[demo]
        if length <= 15:
            raise ValueError("local action episode has no complete fifteen-step clip")
        start = rng.randrange(length - 15)
        dataset = self._diagnostic_dataset() if diagnostic else self.queries
        frames, indices, actions = dataset.local_action_clip(
            task, demo, start, camera_view=self.camera_view,
        )
        trace = {
            "local_action_demo": demo, "local_start_frame": start,
            "local_frame_indices": indices.tolist(),
            "local_action_start": start + 1, "local_action_stop": start + 16,
            "local_flow_seed": rng.getrandbits(63), "local_diagnostic": diagnostic,
        }
        return (torch.from_numpy(frames),), (torch.from_numpy(indices),), torch.from_numpy(actions), trace

    def _sample_actions(self, dataset, pool, task, occurrence, query_seed, count, *,
                        query_offset=0, query_count=None):
        query_count = count if query_count is None else query_count
        if not 0 <= query_offset < count or not 0 < query_count <= count - query_offset:
            raise ValueError("action query slice exceeds full task batch")
        rng = random.Random(query_seed)
        episode_rows = self.query_rows[task] if dataset is self.queries else dataset.task_episode_rows[task]
        selected = []
        for _ in range(count):
            episode = rng.choice(pool)
            # Dataset episode rows are ordered by frame. randrange uses the same
            # draw as choice(rows), preserving the original full task selection.
            frame = rng.randrange(len(episode_rows[episode]))
            selected.append((episode, frame, episode_rows[episode][frame]))
        seed = task_logical_batch_policy_rng_seed(
            optimization_seed=self.seed, task_id=task, task_visit=occurrence,
            demo_indices=[episode for episode, _, _ in selected],
            frame_indices=[frame for _, frame, _ in selected],
        )
        selected = selected[query_offset:query_offset + query_count]
        rows = [dataset[row] for _, _, row in selected]
        return default_collate(rows), {
            "action_demos": [episode for episode, _, _ in selected],
            "action_frames": [frame for _, frame, _ in selected],
            "policy_rng_seed": seed, "policy_random_batch_size": count,
            "query_offset": query_offset,
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
        if self.diagnostic_queries is not None:
            self.diagnostic_queries.close()
