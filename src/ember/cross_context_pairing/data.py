"""Bounded cross-context query pairing; no policy randomness is drawn here."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import default_collate

from ember.operator_writer.data import FormalData, TASKS as OLD_TASKS
from ember.pi05_source_checkpoint import read_json
from ember.writer.data import FunctionalQueryDataset, RawTeacherVideoStore, WriterTaskAuthority
from ember.writer.functional import task_logical_batch_policy_rng_seed
from ember.writer.learning_data import LearningTask, load_learning_tasks


GOAL_GROUPS = ((113, 118, 121), (114, 119, 122), (115, 120, 123))
GOAL_NAMES = ("front", "left", "right")
NEW_TASKS = tuple(sorted(task for group in GOAL_GROUPS for task in group))
TASKS = (*OLD_TASKS, *NEW_TASKS)
UPDATES, CHECKPOINTS, ROOT_SEED, OLD_SEED = 216, (0, 108, 216), 20261008, 20260928
EVENT_SCHEMA = "cross_context_pairing_events_v1"
PARENT = Path("/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928"
              "/continuation2340/T/train/attempts/continuation/checkpoints/macro_00002340")


def pairing_tasks(asset_root: Path, protocol_path: str) -> dict[int, LearningTask]:
    """Preserve the target protocol; add only the nine audited source authorities."""
    asset_root = Path(asset_root)
    tasks = load_learning_tasks(asset_root, OLD_TASKS, role="train", protocol_path=protocol_path)
    source = read_json(asset_root / "configs/pi05_source_corpus_v1/source_manifest.json")
    rows = {int(row["task_index"]): row for row in source["tasks"]}
    dataset = asset_root / "data/datasets" / source["dataset"]["revision"] / source["dataset"]["subdir"]
    for task in NEW_TASKS:
        local = task - 40
        if local not in rows or rows[local]["split"] != "train":
            raise ValueError(f"pairing task {task} is outside the audited Source-71 train authority")
        row = rows[local]
        authority = WriterTaskAuthority(task, row["language"], dataset / row["hdf5"]["filename"],
                                        int(row["hdf5"]["bytes"]))
        tasks[task] = LearningTask(authority, "libero_90", local,
                                  tuple(map(int, row["demonstrations"]["episode_lengths"])))
    if any(len(task.episode_lengths) != 50 or min(task.episode_lengths) < 2 for task in tasks.values()):
        raise ValueError("cross-context pairing requires fifty valid episodes for every allowed task")
    if any(len({tasks[task].authority.language for task in group}) != 1 for group in GOAL_GROUPS):
        raise ValueError("cross-context query group crosses exact task language")
    return tasks


class PairingData(FormalData):
    """Keep each query's original random panel while changing its teacher owner."""

    def __init__(self, asset_root: Path, spec: dict, arm: str = "Within", *,
                 query_labels: bool = True, parent_provenance: dict | None = None) -> None:
        if arm not in ("Within", "Product"):
            raise ValueError("pairing arm must be Within or Product")
        self.spec, self.role, self.arm = spec, "train", arm
        self.tasks = pairing_tasks(asset_root, spec["source"]["data_protocol"])
        authorities = tuple(task.authority for task in self.tasks.values())
        self.videos = RawTeacherVideoStore(authorities, frame_stride=5, camera_view="dual")
        self.queries = (FunctionalQueryDataset(authorities, demo_indices=tuple(range(50)),
                                              action_chunk_size=50, action_start_offset=1)
                        if query_labels else None)
        self.rows = self.queries.task_episode_rows if self.queries is not None else None
        if self.rows is not None:
            for task, metadata in self.tasks.items():
                for demo, length in enumerate(metadata.episode_lengths):
                    if len(self.rows[task][demo]) != length - 1:
                        raise ValueError(f"actual query length differs from its manifest: task={task}, demo={demo}")
        self.seed, self.event_schema, self.updates, self.next_step = ROOT_SEED, EVENT_SCHEMA, UPDATES, 0
        self.checkpoints, self.parent_macro = CHECKPOINTS, 2340
        self.parent_provenance = dict(parent_provenance or {})
        self._old = FormalData.__new__(FormalData)
        self._old.tasks, self._old.seed = self.tasks, OLD_SEED
        self._old.event_schema = "ember_operator_read_write_events_v7"
        self._old_schedule = tuple(
            (int(task), visit, 9 * visit + position // 4)
            for visit in range(260, 278)
            for position, task in enumerate(np.random.default_rng(
                np.random.SeedSequence([OLD_SEED, 0, visit])).permutation(OLD_TASKS)))

    def _old_for_step(self, step: int) -> tuple:
        if type(step) is not int or step not in range(UPDATES):
            raise ValueError("pairing macro is outside the registered 216 updates")
        offset = 9 * (step // 3) + (0, 1, 5)[step % 3]
        count = 1 if step % 3 == 0 else 4
        return self._old_schedule[offset:offset + count]

    def tasks_for_step(self, step: int) -> tuple[int, ...]:
        old = tuple(row[0] for row in self._old_for_step(step))
        return (*GOAL_GROUPS[(step // 3) % 3], *old) if step % 3 == 0 else old

    @staticmethod
    def _records(event: dict) -> list[dict]:
        return [{"task": event["task"], "origin_task": event["task"],
                 "demo": row["demo"], "frame": row["frame"], "origin_visit": event["visit"],
                 "origin_flow_seed": event["flow_seed"], "origin_offset": offset}
                for offset, row in enumerate(event["queries"])]

    def _new_panel(self, task: int, visit: int) -> dict:
        order = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, 1, task])).permutation(50)
        teacher = int(order[visit])
        rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, 2, task, visit]))
        demos = [int(demo) for demo in rng.choice([demo for demo in range(50) if demo != teacher],
                                                size=28, replace=False)]
        frames = [int(rng.integers(self.tasks[task].episode_lengths[demo] - 1)) for demo in demos]
        seed = task_logical_batch_policy_rng_seed(optimization_seed=7, task_id=task, task_visit=visit,
                                                  demo_indices=demos, frame_indices=frames)
        return {"task": task, "visit": visit, "teacher_demo": teacher, "flow_seed": seed,
                "query_offset": 1, "queries": [{"demo": demo, "frame": frame}
                                               for demo, frame in zip(demos, frames, strict=True)]}

    def _triplet_events(self, step: int) -> list[dict]:
        goal, visit = (step // 3) % 3, step // 9
        panels = [self._new_panel(task, visit) for task in GOAL_GROUPS[goal]]
        records = [self._records(panel) for panel in panels]
        if self.arm == "Product":
            shuffled = [[records[scene][int(index)] for index in np.random.default_rng(
                np.random.SeedSequence([ROOT_SEED, 3, goal, visit, scene])).permutation(28)]
                        for scene in range(3)]
            cursors, assigned = [0, 0, 0], [[], [], []]
            for teacher_scene in range(3):
                for query_scene in range(3):
                    count = 9 + int(query_scene == (teacher_scene + visit) % 3)
                    start = cursors[query_scene]
                    assigned[teacher_scene].extend(shuffled[query_scene][start:start + count])
                    cursors[query_scene] += count
            records = assigned
        return [{**panel, "queries": records[scene], "update": step + 1,
                 "pairing_arm": self.arm, "group": "new", "goal_index": goal,
                 "goal": GOAL_NAMES[goal], "teacher_scene": scene}
                for scene, panel in enumerate(panels)]

    def event(self, step: int, task: int) -> dict:
        if task not in self.tasks_for_step(step):
            raise ValueError("task is outside this registered four-condition macro")
        if task in NEW_TASKS:
            return next(event for event in self._triplet_events(step) if event["task"] == task)
        _, visit, source_macro = next(row for row in self._old_for_step(step) if row[0] == task)
        event = self._old._event_for_visit(source_macro, task, visit)
        return {**event, "queries": self._records(event), "origin_update": event["update"],
                "update": step + 1, "pairing_arm": self.arm, "group": "old"}

    def batch(self, event: dict) -> dict:
        if self.queries is None or self.rows is None:
            raise ValueError("video-only pairing data has no action-query dataset")
        queries = event["queries"]
        if len(queries) != 28:
            raise ValueError("pairing condition must retain 28 actual query records")
        rows = []
        for query in queries:
            if (query["task"] != query["origin_task"]
                    or (query["task"] == event["task"] and query["demo"] == event["teacher_demo"])):
                raise ValueError("pairing query identity changed or touched its teacher episode")
            row = self.queries[self.rows[query["task"]][query["demo"]][query["frame"]]]
            if ((row["task_id"], row["demo_index"], row["frame_index"])
                    != (query["task"], query["demo"], query["frame"])
                    or row["task"] != self.tasks[event["task"]].authority.language):
                raise ValueError("actual query task/episode/frame or exact goal language changed")
            rows.append(row)
        return default_collate(rows)

    def sampler_state(self) -> dict:
        return {"schema_version": EVENT_SCHEMA, "arm": self.arm, "next_step": self.next_step,
                "tasks": list(TASKS), "goal_groups": [list(group) for group in GOAL_GROUPS],
                "new_seed": ROOT_SEED, "new_visits_per_task": 24,
                "old_seed": OLD_SEED, "old_first_visit": 260, "old_visits_per_task": 18,
                "query_offset": 1, "queries_per_condition": 28, "logical_queries_per_macro": 112,
                "query_randomness_owner": "original_task_logical_batch_policy_rng_seed_28_panel",
                "product_permutation_seed": [ROOT_SEED, 3, "goal_index", "visit", "query_scene"],
                "extra_cell_column": "(teacher_scene+visit)%3",
                "parent": {"checkpoint": str(PARENT), "macro": 2340,
                           "event_schema": "ember_operator_read_write_events_v7", "event_seed": OLD_SEED},
                "parent_provenance": self.parent_provenance}

    def restore(self, state: dict, *, smoke: bool = False) -> None:
        expected = self.sampler_state()
        cursors = (0, 2, 4) if smoke else CHECKPOINTS
        if (type(state.get("next_step")) is not int or state["next_step"] not in cursors
                or {key: value for key, value in state.items() if key != "next_step"}
                != {key: value for key, value in expected.items() if key != "next_step"}):
            raise ValueError("pairing sampler arm, parent identity or complete-checkpoint cursor changed")
        self.next_step = state["next_step"]

    # Original condition/close keep teaching action-hidden RGB and exact language.
    # Product's top-level flow_seed is its teacher's ORIGINAL Q_c seed; it must
    # never resample reassigned records. The consumer uses each record's origin
    # seed/offset from the complete canonical 28-query noise/time panel instead.
