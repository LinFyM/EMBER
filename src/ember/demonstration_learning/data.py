"""Outcome-blind P/I events and lazy old/new cross-episode FM queries."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import default_collate

from ember.pi05_processing import quat2axisangle
from ember.pi05_source_checkpoint import read_json
from ember.writer.data import FunctionalQueryDataset, RawTeacherVideoStore
from ember.writer.learning_data import load_learning_tasks


TASKS = (0, 1, 2, 4, 5, 7, 12, 13, 14, 15, 17, 19, 20, 21, 22, 25,
         28, 29, 32, 34, 35, 36, 37, 38, 42, 43, 51, 55, 56, 62, 64,
         73, 95, 96, 97, 101)
SUPPORTED = (0, 1, 2, 4, 7, 13, 15, 17, 19, 21, 28, 29, 34, 35, 37,
             38, 55, 95, 96, 97)
SCHEMA = "ember_demonstration_transfer_learning_events_v1"


def _rng(seed: int, *parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), *map(int, parts)]))


def _flow_seed(spec: Mapping, task: int, visit: int) -> int:
    return int(np.random.SeedSequence([
        spec["data"]["event_seeds"]["flow"], task, visit,
    ]).generate_state(1, dtype=np.uint64)[0] & ((1 << 63) - 1))


class PairedEvents:
    """One 288-update plan; only the video reference differs between arms."""

    def __init__(self, spec: Mapping, lengths: Mapping[int, tuple[int, ...]],
                 support: Mapping[int, Mapping]) -> None:
        self.spec = spec
        self.lengths = {int(k): tuple(map(int, v)) for k, v in lengths.items()}
        self.support = support
        if (tuple(sorted(self.lengths)) != TASKS or tuple(sorted(support)) != SUPPORTED
                or any(len(row) != 50 or min(row) < 2 for row in self.lengths.values())):
            raise ValueError("fixed 36-task source/20-task support changed")
        self.next_step = 0
        self._cells = {}
        for task in SUPPORTED:
            cells = [(c, d) for c in range(4) for d in range(4)]
            order = _rng(spec["data"]["event_seeds"]["new_pair_order"], task).permutation(16)
            self._cells[task] = tuple(cells[int(i)] for i in order)

    def _old(self, task: int, visit: int) -> dict:
        seeds = self.spec["data"]["event_seeds"]
        cycle, index = divmod(visit, 46)
        teacher = int(_rng(seeds["old_teacher_order"], task, cycle).permutation(46)[index])
        other = [demo for demo in range(46) if demo != teacher]
        rng = _rng(seeds["query"], task, visit, 0)
        demos = [int(v) for v in rng.choice(other, size=28, replace=False)]
        frames = [int(rng.integers(self.lengths[task][demo] - 1)) for demo in demos]
        return {"kind": "old", "teacher_demo": teacher,
                "queries": [{"demo": d, "frame": f} for d, f in zip(demos, frames, strict=True)]}

    def _new(self, task: int, visit: int) -> dict:
        index = visit // 2
        c, d = self._cells[task][index]
        states = self.support[task]["common_success_states"]
        rng = _rng(self.spec["data"]["event_seeds"]["query"], task, visit, c)
        queries = []
        for _ in range(28):
            state = int(rng.choice(states))
            path = self.support[task]["paths"][(c, state)]
            steps = self.support[task]["rgb_steps"][(c, state)]
            ordinal = int(rng.integers(len(steps)))
            queries.append({"source": c, "state": state, "rgb_ordinal": ordinal,
                            "frame": steps[ordinal], "path": path})
        return {"kind": "new", "source": c, "independent_reference": d,
                "teacher_demo": c, "queries": queries}

    def event(self, update: int, arm: str) -> tuple[dict, ...]:
        if arm not in ("P", "I") or not 0 <= update < 288:
            raise ValueError("event is outside the bounded P/I plan")
        visit, slot = divmod(update, 9)
        seed = self.spec["data"]["event_seeds"]["task_order"]
        order = _rng(seed, visit).permutation(TASKS)
        jobs = []
        for task_value in order[4 * slot:4 * (slot + 1)]:
            task = int(task_value)
            row = self._new(task, visit) if task in SUPPORTED and visit % 2 else self._old(task, visit)
            if row["kind"] == "new" and arm == "I":
                row["teacher_demo"] = row["independent_reference"]
            jobs.append({"task": task, "visit": visit, "update": update + 1,
                         "flow_seed": _flow_seed(self.spec, task, visit), **row})
        return tuple(jobs)

    def sampler_state(self) -> dict:
        return {"schema_version": SCHEMA, "next_step": self.next_step,
                "tasks": list(TASKS), "supported": list(SUPPORTED),
                "event_seeds": self.spec["data"]["event_seeds"],
                "old_query_offset": 1, "new_query_offset": 0,
                "queries_per_condition": 28, "tasks_per_update": 4}

    def restore(self, state: Mapping) -> None:
        expected = self.sampler_state()
        cursor = state.get("next_step")
        if (type(cursor) is not int or cursor not in (2, 4)
                or {k: v for k, v in state.items() if k != "next_step"}
                != {k: v for k, v in expected.items() if k != "next_step"}):
            raise ValueError("paired event identity or resume cursor changed")
        self.next_step = cursor


def audit_full_cycle(events: PairedEvents) -> dict:
    """Check the complete precommitted support/marginal plan without model work."""
    visits = {task: [] for task in TASKS}
    new_queries = old_queries = 0
    for update in range(288):
        p, i = events.event(update, "P"), events.event(update, "I")
        if len(p) != 4 or len({row["task"] for row in p}) != 4:
            raise ValueError("macro update lost four distinct tasks")
        for left, right in zip(p, i, strict=True):
            _check_pair(left, right)
            visits[left["task"]].append(left)
            new_queries += 28 * (left["kind"] == "new")
            old_queries += 28 * (left["kind"] == "old")
    for task, rows in visits.items():
        _check_task_cycle(task, rows)
    if (new_queries, old_queries) != (8960, 23296):
        raise ValueError("full paired cycle query allocation changed")
    return {"updates": 288, "task_visits_each": 32,
            "new_queries": new_queries, "old_queries": old_queries,
            "queries_total": new_queries + old_queries,
            "supported": list(SUPPORTED), "crossing_cells_per_supported_task": 16}


def _check_pair(left: Mapping, right: Mapping) -> None:
    if ({k: v for k, v in left.items() if k != "teacher_demo"}
            != {k: v for k, v in right.items() if k != "teacher_demo"}):
        raise ValueError("P/I query, time or noise event changed")
    if len(left["queries"]) != 28:
        raise ValueError("paired event lost its 28 true queries")


def _check_task_cycle(task: int, rows: list[dict]) -> None:
    if len(rows) != 32 or [row["visit"] for row in rows] != list(range(32)):
        raise ValueError("coverage task lost one visit per round")
    newer = [row for row in rows if row["kind"] == "new"]
    if task in SUPPORTED:
        cells = {(row["source"], row["independent_reference"]) for row in newer}
        if len(newer) != 16 or cells != {(c, d) for c in range(4) for d in range(4)}:
            raise ValueError("supported task lost the full independent 4x4 crossing")
    elif newer:
        raise ValueError("unsupported task received new query data")


class TransferData:
    """Read source HDF5 via its owner and only sealed successful new NPZ traces."""

    def __init__(self, asset_root: Path, spec: Mapping) -> None:
        self.tasks = load_learning_tasks(asset_root, spec["data"]["task_ids"], role="train",
                                         protocol_path=spec["data"]["protocol"])
        table = read_json(Path(spec["data"]["new_support_table"]))
        if (table.get("schema_version") != "ember_demonstration_transfer_training_support_table_v1"
                or len(table.get("tasks", [])) != 20 or sum(len(t["conditions"]) for t in table["tasks"]) != 296):
            raise ValueError("sealed 296-row support table changed")
        support = {}
        for task_row in table["tasks"]:
            task, paths, steps = int(task_row["task"]), {}, {}
            states = tuple(map(int, task_row["common_success_states"]))
            for item in task_row["conditions"]:
                if (int(item["task"]) != task or int(item["state"]) not in states
                        or int(item["demo"]) not in range(4)):
                    raise ValueError("support crossing identity changed")
                key = (int(item["demo"]), int(item["state"]))
                path = Path(item["query_trace"])
                if key in paths or not path.is_file():
                    raise ValueError("support trace missing or duplicated")
                with np.load(path, allow_pickle=False) as sample:
                    action_count = int(sample["actions"].shape[0])
                    saved = tuple(map(int, sample["rgb_steps"].tolist()))
                    indices = tuple(step for step in saved if step < action_count)
                    if (not indices or min(saved) < 0 or max(saved) > action_count
                            or any(b <= a for a, b in zip(saved, saved[1:]))
                            or saved[:len(indices)] != indices):
                        raise ValueError("new query RGB/action offset contract changed")
                paths[key], steps[key] = str(path), indices
            if set(paths) != {(c, s) for c in range(4) for s in states}:
                raise ValueError("new task lacks a complete four-source common-init crossing")
            support[task] = {"common_success_states": states, "paths": paths, "rgb_steps": steps}
        self.events = PairedEvents(spec, {t: v.episode_lengths for t, v in self.tasks.items()}, support)
        authorities = tuple(row.authority for row in self.tasks.values())
        self.videos = RawTeacherVideoStore(authorities, frame_stride=5, camera_view="agentview")
        self.old = FunctionalQueryDataset(authorities, demo_indices=tuple(range(46)),
                                          action_chunk_size=50, action_start_offset=1)
        self.old_rows = self.old.task_episode_rows
        self._new_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def _trace(self, path: str) -> dict[str, np.ndarray]:
        if path not in self._new_cache:
            with np.load(path, allow_pickle=False) as sample:
                self._new_cache[path] = {key: sample[key] for key in
                    ("actions", "rgb_steps", "rgb_canonical180", "eef_pos", "eef_quat", "gripper_qpos")}
            if len(self._new_cache) > 8:
                self._new_cache.popitem(last=False)
        self._new_cache.move_to_end(path)
        return self._new_cache[path]

    def _new_row(self, task: int, query: Mapping) -> dict:
        trace = self._trace(query["path"])
        ordinal, step = int(query["rgb_ordinal"]), int(query["frame"])
        if int(trace["rgb_steps"][ordinal]) != step or step >= len(trace["actions"]):
            raise ValueError("new query action/RGB alignment changed")
        action = trace["actions"][step:step + 50].astype(np.float32)
        padded = np.repeat(action[-1:], 50, axis=0)
        padded[:len(action)] = action
        image = trace["rgb_canonical180"][ordinal]
        state = np.concatenate((trace["eef_pos"][step].astype(np.float32),
                                quat2axisangle(trace["eef_quat"][step]),
                                trace["gripper_qpos"][step].astype(np.float32)))
        if image.shape != (2, 256, 256, 3) or state.shape != (8,):
            raise ValueError("new own observation changed")
        return {"observation.images.camera1": torch.from_numpy(image[0].transpose(2, 0, 1).copy()),
                "observation.images.camera2": torch.from_numpy(image[1].transpose(2, 0, 1).copy()),
                "observation.state": torch.from_numpy(state), "action": torch.from_numpy(padded),
                "task": self.tasks[task].authority.language}

    def batch(self, event: Mapping) -> dict:
        task = int(event["task"])
        if event["kind"] == "new":
            rows = [self._new_row(task, query) for query in event["queries"]]
        else:
            rows = [self.old[self.old_rows[task][q["demo"]][q["frame"]]]
                    for q in event["queries"]]
        return default_collate(rows)

    def close(self) -> None:
        self.videos.close()
        self.old.close()
        self._new_cache.clear()
