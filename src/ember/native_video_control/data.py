"""Matched fit32 events and action-hidden video conditions for M/V."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ember.operator_writer.data import FormalData
from ember.writer.data import FunctionalQueryDataset, RawTeacherVideoStore
from ember.writer.functional import task_logical_batch_policy_rng_seed
from ember.writer.learning_data import load_learning_tasks

from .specification import CHECKPOINTS, EVENT_SCHEMA, FIT32, HELDOUTS, SMOKE_CHECKPOINTS, TASKS, UPDATES


class NativeData(FormalData):
    """Reuse the original RGB/query consumers; replace only the bounded schedule."""

    def __init__(self, asset_root: Path, spec: dict, *, query_labels: bool = True) -> None:
        event = spec["events"]
        if (event["schema_version"] != EVENT_SCHEMA or event["seed"] != 20261007
                or event["task_ids"] != list(FIT32) or event["new_training_heldout_ids"] != list(HELDOUTS)
                or spec["execution"]["updates_per_mode"] != UPDATES
                or spec["execution"]["checkpoints"] != list(CHECKPOINTS)
                or any(event[key] != value for key, value in
                       (("queries_per_task", 28), ("tasks_per_update", 4), ("query_action_offset", 1),
                        ("visits_per_task", 16), ("macros_per_visit", 8)))):
            raise ValueError("native video diagnostic event contract changed")
        self.spec, self.role = spec, "train"
        self.tasks = load_learning_tasks(Path(asset_root), FIT32 if query_labels else TASKS,
                                        role=self.role, protocol_path=spec["source"]["data_protocol"])
        if any(len(row.episode_lengths) != 50 or min(row.episode_lengths) < 2 for row in self.tasks.values()):
            raise ValueError("native diagnostic requires fifty valid episodes per task")
        authorities = tuple(row.authority for row in self.tasks.values())
        self.videos = RawTeacherVideoStore(authorities, frame_stride=5, camera_view="dual")
        self.queries = (FunctionalQueryDataset(authorities, demo_indices=tuple(range(50)),
                                              action_chunk_size=50, action_start_offset=1)
                        if query_labels else None)
        self.rows = self.queries.task_episode_rows if self.queries is not None else None
        self.seed, self.event_schema, self.updates = 20261007, EVENT_SCHEMA, UPDATES
        self.checkpoints, self.next_step = CHECKPOINTS, 0

    def tasks_for_step(self, step: int) -> tuple[int, ...]:
        if type(step) is not int or step not in range(UPDATES):
            raise ValueError("native diagnostic macro is outside its registered updates")
        visit, slot = divmod(step, 8)
        order = np.random.default_rng(np.random.SeedSequence([self.seed, 0, visit])).permutation(FIT32)
        return tuple(int(task) for task in order[4 * slot:4 * (slot + 1)])

    def event(self, step: int, task: int) -> dict:
        if task not in self.tasks_for_step(step):
            raise ValueError("task is outside this native four-condition macro")
        visit = step // 8
        teacher_order = np.random.default_rng(np.random.SeedSequence([self.seed, 1, task])).permutation(50)
        teacher = int(teacher_order[visit])
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 2, task, visit]))
        demos = [int(demo) for demo in rng.choice([i for i in range(50) if i != teacher], size=28, replace=False)]
        lengths = self.tasks[task].episode_lengths
        frames = [int(rng.integers(lengths[demo] - 1)) for demo in demos]
        flow_seed = task_logical_batch_policy_rng_seed(
            optimization_seed=7, task_id=task, task_visit=visit, demo_indices=demos, frame_indices=frames)
        return {"update": step + 1, "visit": visit, "task": task, "teacher_demo": teacher,
                "queries": [{"demo": demo, "frame": frame} for demo, frame in zip(demos, frames, strict=True)],
                "flow_seed": flow_seed, "query_offset": 1}

    def sampler_state(self) -> dict:
        return {
            "schema_version": EVENT_SCHEMA, "next_step": self.next_step, "seed": self.seed,
            "tasks": list(FIT32), "new_training_heldout_ids": list(HELDOUTS),
            "teacher_demo_pool": list(range(50)), "teacher_permutation_seed": [self.seed, 1, "task"],
            "visits_per_task": 16, "macros_per_visit": 8, "tasks_per_update": 4,
            "demo_pool": [0, 49], "query_offset": 1, "queries_per_task": 28,
            "optimization_seed": 7, "flow_seed_owner": "task_logical_batch_policy_rng_seed",
        }

    def restore(self, state: dict) -> None:
        expected = self.sampler_state()
        if (type(state.get("next_step")) is not int
                or state["next_step"] not in (*CHECKPOINTS, *SMOKE_CHECKPOINTS)
                or {k: v for k, v in state.items() if k != "next_step"}
                != {k: v for k, v in expected.items() if k != "next_step"}):
            raise ValueError("native sampler identity or complete-checkpoint cursor changed")
        self.next_step = state["next_step"]

    def condition(self, runtime, task: int, demo: int) -> tuple[tuple, int, int]:
        video = self.videos.load(task, demo)
        pixels = torch.from_numpy(video.frames).to(runtime.device, non_blocking=True)
        indices = torch.from_numpy(video.frame_indices).to(runtime.device, non_blocking=True)
        tokens, mask, _ = runtime.tokenizer([self.tasks[task].authority.language])
        return (pixels, indices, tokens, mask), video.raw_frame_count, len(pixels)

    # FormalData.batch/close preserve the real 50-action chunk, final-action
    # padding and the independent query RGB/state labels; teacher remains RGB.
