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
        self.asset_root = asset_root
        self.config = dict(config)
        if (type(config.get("action_start_offset")) is not int or config["action_start_offset"] != 1
                or config.get("query_alignment") != "post_action_observation_future_control_v1"):
            raise ValueError("Writer requires post-action observations with future-control labels")
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
        if (self.video_pool != tuple(range(16, 42)) or self.action_pool != self.video_pool
                or self.diagnostic_pool != tuple(range(42, 46)) or self.held_video_pool != tuple(range(46, 50))):
            raise ValueError("native correction requires the registered training and held episode pools")
        if tuple(config["cardinalities"]) != (1,):
            raise ValueError("the current supervised stage requires actual K=1 conditions")
        authorities = tuple(task.authority for task in self.tasks.values())
        self.videos = RawTeacherVideoStore(authorities, frame_stride=5, camera_view=camera_view)
        self.queries = FunctionalQueryDataset(authorities, demo_indices=self.action_pool,
                                              action_chunk_size=50, action_start_offset=1)
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
        if not demos or len(set(demos)) != len(demos) or not set(demos) <= set(self.video_pool):
            raise ValueError("teaching video must belong to the registered training pool")
        query_pool = tuple(demo for demo in self.action_pool if demo not in demos)
        return self._sample_actions(self.queries, query_pool, task, occurrence, query_seed,
                                    int(self.config["queries_per_task"]),
                                    query_offset=query_offset, query_count=query_count)

    def local_field_batch(self, task, demo, frame_indices, *, query_seed,
                          positions_per_condition, future_horizon, seed):
        """Training-only real teacher futures; its local RNG never advances main sampling."""
        if (task not in self.tasks or demo not in self.video_pool or frame_indices.ndim != 1
                or len(frame_indices) == 0 or positions_per_condition <= 0 or future_horizon <= 0):
            raise ValueError("local correction positions require a registered train teacher video")
        positions = frame_indices.detach().cpu().long()
        actions = self.queries._handle(task)[f"data/demo_{demo}/actions"]
        if (actions.shape != (self.tasks[task].episode_lengths[demo], 7)
                or int(positions[0]) != 0 or int(positions[-1]) != len(actions) - 1
                or not bool((positions[1:] > positions[:-1]).all())):
            raise ValueError("local correction labels must retain the full real video including its final frame")
        field_seed = int(query_seed) ^ int(seed)
        selected = sorted(random.Random(field_seed).sample(range(len(positions)), min(positions_per_condition, len(positions))))
        ordinals = torch.tensor(selected, dtype=torch.long)
        selected_positions = positions[ordinals].tolist()
        raw = torch.zeros(len(selected), future_horizon, 7, dtype=torch.float32)
        counts = torch.tensor([min(future_horizon, len(actions) - p - 1) for p in selected_positions])
        for row, (position, count) in enumerate(zip(selected_positions, counts.tolist(), strict=True)):
            if count:
                raw[row, :count] = torch.as_tensor(actions[position + 1:position + 1 + count])
        return ordinals, raw, counts, {
            "field_seed": field_seed, "field_frame_ordinals": selected,
            "field_frame_positions": selected_positions, "field_real_future_counts": counts.tolist(),
            "field_action_start_indices": [position + 1 for position in selected_positions],
        }

    def _diagnostic_dataset(self) -> FunctionalQueryDataset:
        if self.diagnostic_queries is None:
            self.diagnostic_queries = FunctionalQueryDataset(
                tuple(task.authority for task in self.tasks.values()),
                demo_indices=self.diagnostic_pool, action_chunk_size=50, action_start_offset=1,
            )
        return self.diagnostic_queries

    def diagnostic_batch(self, task: int, *, seed: int, count: int):
        return self._sample_actions(self._diagnostic_dataset(), self.diagnostic_pool, task, 0, seed, count)

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
            "action_start_indices": [frame + 1 for _, frame, _ in selected],
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
