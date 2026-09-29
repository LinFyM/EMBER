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
CONTINUATION_CHECKPOINTS = (360, 450, 540, 630, 720, 810, 900)
CONTINUATION_UPDATES = 900
CONTINUATION1350_CHECKPOINTS = tuple(range(990, 1351, 90))
CONTINUATION1350_UPDATES = 1350
CONTINUATION1800_CHECKPOINTS = tuple(range(1440, 1801, 90))
CONTINUATION1800_UPDATES = 1800
PILOT_CHECKPOINTS = (1890,)
PILOT_UPDATES = 1890
CONTINUATION2340_CHECKPOINTS = tuple(range(1980, 2341, 90))
CONTINUATION2340_UPDATES = 2340


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
        self.event_schema = spec["events"]["schema_version"]
        self.updates = int(spec["execution"]["updates_per_mode"])
        self.checkpoints = tuple(spec["execution"]["checkpoints"])
        self.next_step = 0

    def tasks_for_step(self, step: int) -> tuple[int, ...]:
        if step not in range(self.updates):
            raise ValueError("formal macro step is outside registered updates")
        visit, slot = divmod(step, 9)
        order = np.random.default_rng(np.random.SeedSequence([self.seed, 0, visit])).permutation(TASKS)
        return tuple(int(task) for task in order[4 * slot:4 * (slot + 1)])

    def event(self, step: int, task: int) -> dict:
        if task not in self.tasks_for_step(step):
            raise ValueError("task is outside this formal four-condition macro")
        visit = step // 9
        teacher_round = visit // 50
        allowed_rounds = {"ember_operator_read_write_events_v2": 1,
                          "ember_operator_read_write_events_v3": 2,
                          "ember_operator_read_write_events_v4": 3,
                          "ember_operator_read_write_events_v5": 4,
                          "ember_operator_read_write_events_v6": 5,
                          "ember_operator_read_write_events_v7": 6}
        if teacher_round >= allowed_rounds.get(self.event_schema, 0):
            raise ValueError("teacher round is outside the registered event contract")
        teacher_seed = ([self.seed, 1, task] if teacher_round == 0 else
                        [self.seed, 1, task, teacher_round])
        teacher_order = np.random.default_rng(np.random.SeedSequence(teacher_seed)).permutation(50)
        teacher = int(teacher_order[visit % 50])
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
        common = {"next_step": self.next_step, "seed": self.seed, "tasks": list(TASKS),
                  "demo_pool": [0, 49], "query_offset": 1, "queries_per_task": 28}
        if self.event_schema == "ember_operator_read_write_events_v2":
            return {"schema_version": self.event_schema, **common, "teacher_pool": list(range(30))}
        rounds = {"ember_operator_read_write_events_v3": 2,
                  "ember_operator_read_write_events_v4": 3,
                  "ember_operator_read_write_events_v5": 4,
                  "ember_operator_read_write_events_v6": 5,
                  "ember_operator_read_write_events_v7": 6}.get(self.event_schema)
        if rounds is not None:
            return {"schema_version": self.event_schema, **common,
                    "teacher_rounds": [[self.seed, 1, "task"]] +
                                      [[self.seed, 1, "task", round_index]
                                       for round_index in range(1, rounds)],
                    "teacher_visits_per_round": 50, "teacher_demo_pool": list(range(50))}
        raise ValueError("operator event schema changed")

    def restore(self, state: dict, *, migrate_sealed_270: bool = False,
                migrate_continuation_900: bool = False,
                migrate_continuation_1350: bool = False,
                migrate_continuation_1800: bool = False,
                migrate_pilot_1890: bool = False) -> dict | None:
        expected = self.sampler_state()
        if migrate_sealed_270:
            legacy = {"schema_version": "ember_operator_read_write_events_v2", "next_step": 270,
                      "seed": self.seed, "tasks": list(TASKS), "teacher_pool": list(range(30)),
                      "demo_pool": [0, 49], "query_offset": 1, "queries_per_task": 28}
            if self.event_schema != "ember_operator_read_write_events_v3" or state != legacy:
                raise ValueError("sealed 270 sampler migration source changed")
            self.next_step = 270
            return {"from_schema": legacy["schema_version"], "to_schema": self.event_schema,
                    "cursor": 270, "historical_teacher_pool_meaning": "visits_0_to_29_not_demo_ids"}
        if (migrate_continuation_900 or migrate_continuation_1350
                or migrate_continuation_1800 or migrate_pilot_1890):
            if sum((migrate_continuation_900, migrate_continuation_1350,
                    migrate_continuation_1800, migrate_pilot_1890)) != 1:
                raise ValueError("operator sampler migration must name one parent")
            parent = (900 if migrate_continuation_900 else
                      1350 if migrate_continuation_1350 else
                      1800 if migrate_continuation_1800 else 1890)
            source_schema = ("ember_operator_read_write_events_v6" if migrate_pilot_1890 else
                             f"ember_operator_read_write_events_v{parent // 450 + 1}")
            target_schema = ("ember_operator_read_write_events_v7" if migrate_pilot_1890 else
                             f"ember_operator_read_write_events_v{parent // 450 + 2}")
            legacy = {**expected, "schema_version": source_schema,
                      "next_step": parent,
                      "teacher_rounds": expected["teacher_rounds"][:-1]}
            if self.event_schema != target_schema or state != legacy:
                raise ValueError(f"{parent} sampler migration source changed")
            self.next_step = parent
            return {"from_schema": legacy["schema_version"], "to_schema": self.event_schema,
                    "cursor": parent,
                    "appended_teacher_round": [self.seed, 1, "task", 5 if migrate_pilot_1890
                                               else parent // 450]}
        if (type(state.get("next_step")) is not int or state["next_step"] not in (0, *self.checkpoints)
                or {k: v for k, v in state.items() if k != "next_step"}
                != {k: v for k, v in expected.items() if k != "next_step"}):
            raise ValueError("operator sampler identity or registered ECP cursor changed")
        self.next_step = state["next_step"]
        return None

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
