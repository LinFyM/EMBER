"""Formal 36-task operator teaching/query event owner; no held action dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from ember.writer.data import FunctionalQueryDataset, RawTeacherVideoStore
from ember.writer.functional import task_logical_batch_policy_rng_seed


TASKS = (0, 1, 2, 4, 5, 7, 12, 13, 14, 15, 17, 19, 20, 21, 22, 25, 28, 29,
         32, 34, 35, 36, 37, 38, 42, 43, 51, 55, 56, 62, 64, 73, 95, 96, 97, 101)
CHECKPOINTS = (90, 180, 270)
UPDATES = 270
class FormalData:
    """Only action-hidden RGB is teaching; own HDF5 state/action is query label."""

    def __init__(self, asset_root: Path, spec: dict, *, query_labels: bool = True,
                 task_ids: tuple[int, ...] = TASKS, role: str = "train") -> None:
        from ember.writer.learning_data import load_learning_tasks

        self.tasks = load_learning_tasks(asset_root, task_ids, role=role,
                                         protocol_path=spec["source"]["data_protocol"])
        authorities = tuple(row.authority for row in self.tasks.values())
        self.videos = RawTeacherVideoStore(authorities, frame_stride=5, camera_view="dual")
        self.queries = (FunctionalQueryDataset(authorities, demo_indices=tuple(range(50)),
                                              action_chunk_size=50, action_start_offset=1)
                        if query_labels else None)
        self.rows = self.queries.task_episode_rows if self.queries is not None else None
        self.seed = int(spec["events"]["seed"])
        self.next_step = 0

    def tasks_for_step(self, step: int) -> tuple[int, ...]:
        if step not in range(UPDATES):
            raise ValueError("formal macro step is outside 270 updates")
        visit, slot = divmod(step, 9)
        order = np.random.default_rng(np.random.SeedSequence([self.seed, 0, visit])).permutation(TASKS)
        return tuple(int(task) for task in order[4 * slot:4 * (slot + 1)])

    def event(self, step: int, task: int) -> dict:
        if task not in self.tasks_for_step(step):
            raise ValueError("task is outside this formal four-condition macro")
        visit = step // 9
        teacher_order = np.random.default_rng(np.random.SeedSequence([self.seed, 1, task])).permutation(50)
        teacher = int(teacher_order[visit])
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 2, task, visit]))
        demos = rng.choice(np.asarray([i for i in range(50) if i != teacher]),
                           size=28, replace=False)
        lengths = self.tasks[task].episode_lengths
        frames = [int(rng.integers(lengths[int(demo)] - 1)) for demo in demos]
        demos = [int(demo) for demo in demos]
        flow_seed = task_logical_batch_policy_rng_seed(
            optimization_seed=7, task_id=task, task_visit=visit,
            demo_indices=demos, frame_indices=frames)
        return {"update": step + 1, "visit": visit, "task": task, "teacher_demo": teacher,
                "queries": [{"demo": demo, "frame": frame}
                            for demo, frame in zip(demos, frames, strict=True)],
                "flow_seed": flow_seed, "query_offset": 1}

    def sampler_state(self) -> dict:
        return {"schema_version": "ember_operator_read_write_events_v2", "next_step": self.next_step,
                "seed": self.seed, "tasks": list(TASKS), "teacher_pool": list(range(30)),
                "demo_pool": [0, 49], "query_offset": 1, "queries_per_task": 28}

    def restore(self, state: dict) -> None:
        expected = self.sampler_state()
        if (type(state.get("next_step")) is not int or state["next_step"] not in (0, *CHECKPOINTS)
                or {k: v for k, v in state.items() if k != "next_step"}
                != {k: v for k, v in expected.items() if k != "next_step"}):
            raise ValueError("operator sampler identity or registered ECP cursor changed")
        self.next_step = state["next_step"]

    def condition(self, runtime, task: int, demo: int) -> tuple[tuple, int, int]:
        video = self.videos.load(task, demo)
        pixels = torch.from_numpy(video.frames).to(runtime.device, non_blocking=True)
        indices = torch.from_numpy(video.frame_indices).to(runtime.device, non_blocking=True)
        tokens, mask, _ = runtime.tokenizer([self.tasks[task].authority.language])
        return (pixels, indices, tokens, mask), video.raw_frame_count, len(pixels)

    def batch(self, event: dict) -> dict:
        if self.queries is None or self.rows is None:
            raise ValueError("deployment/video-only path has no action-query dataset")
        rows = [self.queries[self.rows[event["task"]][row["demo"]][row["frame"]]]
                for row in event["queries"]]
        if len(rows) != 28 or any(row["demo_index"] == event["teacher_demo"] for row in rows):
            raise ValueError("FM query touched its teacher episode")
        return default_collate(rows)

    def close(self) -> None:
        self.videos.close()
        if self.queries is not None:
            self.queries.close()
