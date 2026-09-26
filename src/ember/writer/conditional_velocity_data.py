"""The registered 36-task, cross-episode event stream for conditional velocity."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from torch.utils.data import default_collate

from ember.writer.data import FunctionalQueryDataset, RawTeacherVideoStore
from ember.writer.functional import task_logical_batch_policy_rng_seed
from ember.writer.learning_data import load_learning_tasks


EVENT_SCHEMA = "ember_conditional_velocity_events_v1"
COVERAGE_TASKS = (0, 1, 2, 4, 5, 7, 12, 13, 14, 15, 17, 19, 20, 21, 22, 25,
                  28, 29, 32, 34, 35, 36, 37, 38, 42, 43, 51, 55, 56, 62, 64,
                  73, 95, 96, 97, 101)


class VelocityEvents:
    def __init__(self, lengths: Mapping[int, tuple[int, ...]], *, seed: int = 20260927,
                 query_count: int = 28) -> None:
        self.lengths = {int(task): tuple(map(int, values)) for task, values in lengths.items()}
        self.tasks = tuple(sorted(self.lengths))
        if (self.tasks != COVERAGE_TASKS or any(len(row) != 50 or min(row) < 2 for row in self.lengths.values())
                or seed != 20260927 or query_count != 28):
            raise ValueError("registered 36-task/50-demo event authority changed")
        self.seed, self.query_count, self.next_step = seed, query_count, 0

    def event(self, update: int) -> tuple[dict, ...]:
        if type(update) is not int or update < 0:
            raise ValueError("invalid zero-based velocity update")
        round_index, slot = divmod(update, 9)
        task_rng = np.random.default_rng(np.random.SeedSequence(
            [self.seed, round_index, 0x5441534B],
        ))
        tasks = task_rng.permutation(self.tasks)[4 * slot:4 * (slot + 1)]
        jobs = []
        for task_value in tasks:
            task = int(task_value)
            teacher_rng = np.random.default_rng(np.random.SeedSequence(
                [self.seed, task, round_index // 50, 0x564944],
            ))
            teacher = int(teacher_rng.permutation(50)[round_index % 50])
            others = np.asarray([demo for demo in range(50) if demo != teacher])
            query_rng = np.random.default_rng(np.random.SeedSequence(
                [self.seed, task, round_index, 0x515259],
            ))
            demos = [int(demo) for demo in query_rng.choice(others, 28, replace=False)]
            frames = [int(query_rng.integers(self.lengths[task][demo] - 1)) for demo in demos]
            jobs.append({
                "task": task, "visit": round_index, "teacher_demo": teacher,
                "action_demos": demos, "action_frames": frames,
                "action_start_indices": [frame + 1 for frame in frames],
                "policy_rng_seed": task_logical_batch_policy_rng_seed(
                    optimization_seed=7, task_id=task, task_visit=round_index,
                    demo_indices=demos, frame_indices=frames,
                ),
            })
        return tuple(jobs)

    def sampler_state(self) -> dict:
        return {"schema_version": EVENT_SCHEMA, "next_step": self.next_step,
                "tasks": list(self.tasks), "seed": self.seed, "queries_per_task": self.query_count,
                "teacher_pool": [0, 49], "action_start_offset": 1}

    def restore(self, state: Mapping, *, maximum: int = 270) -> None:
        expected = self.sampler_state()
        cursor = state.get("next_step")
        if (type(cursor) is not int or not 0 <= cursor <= maximum
                or {key: value for key, value in state.items() if key != "next_step"}
                != {key: value for key, value in expected.items() if key != "next_step"}):
            raise ValueError("velocity event identity or cursor changed")
        self.next_step = cursor


class VelocityTrainingData:
    """Reuse the native video/query owners; task identities remain outside the model."""

    def __init__(self, asset_root: Path, spec: Mapping) -> None:
        self.tasks = load_learning_tasks(asset_root, spec["train_tasks"], role="train",
                                         protocol_path=spec["protocol"])
        if tuple(sorted(self.tasks)) != tuple(spec["train_tasks"]):
            raise ValueError("coverage36 task authority changed")
        self.events = VelocityEvents({task: row.episode_lengths for task, row in self.tasks.items()},
                                     seed=spec["sampler_seed"], query_count=spec["query_count_per_task"])
        authorities = tuple(row.authority for row in self.tasks.values())
        self.videos = RawTeacherVideoStore(authorities, frame_stride=5, camera_view="agentview")
        self.queries = FunctionalQueryDataset(authorities, demo_indices=tuple(range(50)),
                                              action_chunk_size=50, action_start_offset=1)
        self.rows = self.queries.task_episode_rows

    def batch(self, event: Mapping):
        task = event["task"]
        return default_collate([
            self.queries[self.rows[task][demo][frame]]
            for demo, frame in zip(event["action_demos"], event["action_frames"], strict=True)
        ])

    def close(self) -> None:
        self.videos.close()
        self.queries.close()
