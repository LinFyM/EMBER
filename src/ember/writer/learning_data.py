"""Train24 sampling for supervised video-to-LoRA learning.

Task and episode identities are orchestration metadata. The model receives only
the returned RGB arrays, real frame indices and exact task language.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
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
    """Load task metadata only; training callers retain the fixed train default."""
    if role not in {"train", "validation", "test"}:
        raise ValueError("task metadata requires a registered target split")
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
            raise ValueError("selected task crosses the fixed target split")
        authority = WriterTaskAuthority(
            task_id, str(row["language"]), data_root / row["hdf5"]["relative_path"], int(row["hdf5"]["bytes"]),
        )
        output[task_id] = LearningTask(
            authority, suite, local, tuple(map(int, row["demonstrations"]["episode_lengths"])),
        )
    return output


EVENT_SCHEMA = "v52_full_video_cross_episode_events_v1"


def _episode_queries(task, lengths, order, *, seed, cursor, count, teacher_demo=None):
    """Walk an episode permutation; skip the teacher without consuming labels."""
    demos, frames = [], []
    while len(demos) < count:
        cycle, offset = divmod(cursor, len(order))
        cursor += 1
        demo = order[offset]
        if demo == teacher_demo:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, task, demo, cycle, 0xF4A]))
        demos.append(demo)
        frames.append(int(rng.integers(lengths[demo] - 1)))
    return demos, frames, cursor


class WriterTrainingData:
    """Fixed task/video/query events, grouped independently of device ownership."""

    def __init__(self, asset_root: Path, config: Mapping[str, Any], *, camera_view: str = "dual") -> None:
        self.asset_root, self.config = asset_root, deepcopy(dict(config))
        self._validate_config()
        if camera_view != "dual":
            raise ValueError("v5.2 fresh training requires synchronized dual-camera video")
        self.seed = config["seed"]
        self.sampler_seed = config["sampler_seed"]
        self.teacher_video_seed = config["teacher_video_seed"]
        self.maximum_updates = config["maximum_updates"]
        self.camera_view, self.conditions_per_task = camera_view, 1
        self.tasks = load_learning_tasks(asset_root, config["task_ids"])
        self.task_ids = tuple(sorted(self.tasks))
        if len(self.task_ids) != 24 or len({task.suite for task in self.tasks.values()}) != 4:
            raise ValueError("training events require the complete fixed train24 split")
        self.video_pool = tuple(config["video_demos"])
        self.action_pool = tuple(config["action_demos"])
        self.diagnostic_pool = tuple(config["diagnostic_action_demos"])
        self.held_video_pool = tuple(config["held_video_demos"])
        self._groups = self._build_groups()
        self._events = self._build_events()
        authorities = tuple(task.authority for task in self.tasks.values())
        self.videos = RawTeacherVideoStore(authorities, frame_stride=5, camera_view=camera_view)
        self.queries = FunctionalQueryDataset(authorities, demo_indices=self.action_pool,
                                              action_chunk_size=50, action_start_offset=1)
        self.query_rows = self.queries.task_episode_rows
        self.diagnostic_queries = None
        if any(len(self.query_rows[task][demo]) != self.tasks[task].episode_lengths[demo] - 1
               for task in self.task_ids for demo in self.action_pool):
            self.close()
            raise ValueError("training event metadata differs from the functional-query lengths")
        self.next_step = 0
        self.counts = dict.fromkeys(self.task_ids, 0)

    def _validate_config(self) -> None:
        config = self.config
        if (type(config.get("action_start_offset")) is not int or config["action_start_offset"] != 1
                or config.get("query_alignment") != "post_action_observation_future_control_v1"):
            raise ValueError("Writer requires post-action observations with future-control labels")
        if config.get("event_schema_version") != EVENT_SCHEMA:
            raise ValueError("training event schema must be explicitly registered")
        for name in ("seed", "sampler_seed", "teacher_video_seed", "maximum_updates"):
            if type(config.get(name)) is not int or config[name] < 0:
                raise ValueError(f"training event {name} must be a non-negative integer")
        if not 0 < config["maximum_updates"] <= 1200 or config["maximum_updates"] % 6:
            raise ValueError("training events require complete six-update rounds, at most 1200 updates")
        if (config.get("tasks_per_update") != 4 or config.get("conditions_per_task") != 1
                or config.get("queries_per_task") != 21 or tuple(config["cardinalities"]) != (1,)):
            raise ValueError("training events require four tasks, one video and 21 queries per task")
        if any(tuple(config[name]) != tuple(range(46)) for name in ("video_demos", "action_demos")):
            raise ValueError("training video/action pools must be episodes 0 through 45")
        if any(tuple(config[name]) != tuple(range(46, 50))
               for name in ("diagnostic_action_demos", "held_video_demos")):
            raise ValueError("episodes 46 through 49 are reserved for frozen diagnostics")

    def _build_groups(self) -> tuple[tuple[int, ...], ...]:
        grouping = self.config.get("grouping")
        if grouping == "baseline" and "event_groups" not in self.config:
            groups = []
            for occurrence in range(self.maximum_updates // 6):
                order = np.random.default_rng(
                    np.random.SeedSequence([self.sampler_seed, occurrence]),
                ).permutation(self.task_ids)
                groups.extend(order[start:start + 4].tolist() for start in range(0, 24, 4))
        elif grouping == "explicit":
            groups = self.config.get("event_groups", ())
        else:
            raise ValueError("grouping must be baseline or explicitly supplied event_groups")
        if (len(groups) != self.maximum_updates or any(
                len(group) != 4 or any(type(task) is not int for task in group)
                or len(set(group)) != 4 for group in groups)):
            raise ValueError("event_groups require exactly four distinct tasks per update")
        for start in range(0, self.maximum_updates, 6):
            if sorted(task for group in groups[start:start + 6] for task in group) != list(self.task_ids):
                raise ValueError("each round must contain every train24 event exactly once")
        task_offset = {task: offset for offset, task in enumerate(self.task_ids)}
        return tuple(tuple((step // 6) * 24 + task_offset[task] for task in group)
                     for step, group in enumerate(groups))

    def _build_events(self) -> tuple[dict[str, Any], ...]:
        events = []
        for task in self.task_ids:
            lengths = self.tasks[task].episode_lengths
            if len(lengths) != 50 or min(lengths) < 2:
                raise ValueError("training events require 50 episodes with future-control support")
            order = np.random.default_rng(
                np.random.SeedSequence([self.sampler_seed, task, 0xE91]),
            ).permutation(self.action_pool).tolist()
            cursor = 0
            for occurrence in range(self.maximum_updates // 6):
                cycle, offset = divmod(occurrence, len(self.video_pool))
                teacher = int(np.random.default_rng(np.random.SeedSequence(
                    [self.teacher_video_seed, task, cycle, 0x71DE0],
                )).permutation(self.video_pool)[offset])
                demos, frames, cursor = _episode_queries(
                    task, lengths, order, seed=self.sampler_seed, cursor=cursor,
                    count=21, teacher_demo=teacher,
                )
                query_seed = int(np.random.SeedSequence(
                    [self.sampler_seed, task, occurrence, 0x51555259],
                ).generate_state(1, dtype=np.uint64)[0]) & ((1 << 63) - 1)
                events.append({
                    "task": task, "occurrence": occurrence, "teacher_demo": teacher,
                    "query_seed": query_seed, "action_demos": demos, "action_frames": frames,
                    "action_start_indices": [frame + 1 for frame in frames],
                    "policy_rng_seed": task_logical_batch_policy_rng_seed(
                        optimization_seed=self.seed, task_id=task, task_visit=occurrence,
                        demo_indices=demos, frame_indices=frames,
                    ), "policy_random_batch_size": 21,
                    "frames": (lengths[teacher] - 1) // 5 + 1 + bool((lengths[teacher] - 1) % 5),
                })
        events.sort(key=lambda event: (event["occurrence"], event["task"]))
        return tuple(events)

    def _event_contract(self) -> dict[str, Any]:
        return {"schema_version": EVENT_SCHEMA, "seed": self.seed, "sampler_seed": self.sampler_seed,
                "teacher_video_seed": self.teacher_video_seed, "maximum_updates": self.maximum_updates,
                "grouping": self.config["grouping"], "groups": [list(group) for group in self._groups],
                "task_ids": list(self.task_ids), "video_demos": list(self.video_pool),
                "action_demos": list(self.action_pool), "queries_per_task": 21,
                "action_start_offset": 1, "query_alignment": self.config["query_alignment"],
                "episode_lengths": [list(self.tasks[task].episode_lengths) for task in self.task_ids]}

    def event_plan(self) -> dict[str, Any]:
        """Return all events in round/task order; groups reference zero-based event indices."""
        return {**self._event_contract(), "group_reference": "zero_based_event_index",
                "event_order": "occurrence_then_sorted_task", "events": deepcopy(list(self._events))}

    def next_iteration(self) -> tuple[dict[str, Any], ...]:
        if self.next_step >= self.maximum_updates:
            raise StopIteration("the registered training event plan is exhausted")
        draws = []
        for event_index in self._groups[self.next_step]:
            event = self._events[event_index]
            task = event["task"]
            draws.append({"job_id": len(draws), "condition_index": 0, "task": task,
                          "occurrence": event["occurrence"], "video_demos": (event["teacher_demo"],),
                          "query_seed": event["query_seed"], "query_offset": 0, "query_count": 21,
                          "frames": event["frames"]})
            self.counts[task] += 1
        self.next_step += 1
        return tuple(draws)

    def load_videos(self, task: int, demos: Sequence[int]):
        if len(demos) != 1:
            raise ValueError("the registered teaching condition requires exactly one video")
        videos = tuple(self.videos.load(task, demo) for demo in demos)
        return (
            tuple(torch.from_numpy(video.frames) for video in videos),
            tuple(torch.from_numpy(video.frame_indices) for video in videos),
        )

    def action_batch(self, task: int, occurrence: int, demos: Sequence[int], *, query_seed: int,
                     query_offset: int = 0, query_count: int | None = None):
        if (task not in self.tasks or type(occurrence) is not int
                or not 0 <= occurrence < self.maximum_updates // 6):
            raise ValueError("action query is outside the registered task/visit events")
        event = self._events[occurrence * 24 + self.task_ids.index(task)]
        if tuple(demos) != (event["teacher_demo"],) or query_seed != event["query_seed"]:
            raise ValueError("action query differs from its registered training event")
        return self._collate_event(self.queries, event, query_offset=query_offset, query_count=query_count)

    def _diagnostic_dataset(self) -> FunctionalQueryDataset:
        if self.diagnostic_queries is None:
            self.diagnostic_queries = FunctionalQueryDataset(
                tuple(task.authority for task in self.tasks.values()),
                demo_indices=self.diagnostic_pool, action_chunk_size=50, action_start_offset=1,
            )
            rows = self.diagnostic_queries.task_episode_rows
            if any(len(rows[task][demo]) != self.tasks[task].episode_lengths[demo] - 1
                   for task in self.task_ids for demo in self.diagnostic_pool):
                self.diagnostic_queries.close()
                self.diagnostic_queries = None
                raise ValueError("frozen diagnostic metadata differs from its functional-query lengths")
        return self.diagnostic_queries

    def diagnostic_batch(self, task: int, *, seed: int, count: int, teacher_demo: int | None = None):
        if (task not in self.tasks or type(seed) is not int or seed < 0
                or type(count) is not int or count <= 0
                or (teacher_demo is not None and teacher_demo not in range(50))):
            raise ValueError("invalid frozen training-action diagnostic")
        order = np.random.default_rng(np.random.SeedSequence([seed, task, 0xE91])).permutation(
            self.diagnostic_pool,
        ).tolist()
        demos, frames, _ = _episode_queries(
            task, self.tasks[task].episode_lengths, order, seed=seed, cursor=0,
            count=count, teacher_demo=teacher_demo,
        )
        event = {"task": task, "action_demos": demos, "action_frames": frames,
                 "action_start_indices": [frame + 1 for frame in frames],
                 "policy_random_batch_size": count, "policy_rng_seed": task_logical_batch_policy_rng_seed(
                     optimization_seed=self.seed, task_id=task, task_visit=0,
                     demo_indices=demos, frame_indices=frames,
                 )}
        return self._collate_event(self._diagnostic_dataset(), event)

    def _collate_event(self, dataset, event, *, query_offset=0, query_count=None):
        count = event["policy_random_batch_size"]
        query_count = count if query_count is None else query_count
        if (type(query_offset) is not int or type(query_count) is not int
                or not 0 <= query_offset < count or not 0 < query_count <= count - query_offset):
            raise ValueError("action query slice exceeds full task batch")
        chosen = slice(query_offset, query_offset + query_count)
        trace = {key: event[key][chosen] for key in ("action_demos", "action_frames", "action_start_indices")}
        episode_rows = self.query_rows if dataset is self.queries else dataset.task_episode_rows
        rows = [dataset[episode_rows[event["task"]][demo][frame]]
                for demo, frame in zip(trace["action_demos"], trace["action_frames"], strict=True)]
        return default_collate(rows), {**trace, "policy_rng_seed": event["policy_rng_seed"],
                                      "policy_random_batch_size": count, "query_offset": query_offset}

    def sampler_state(self) -> dict[str, Any]:
        return {"next_step": self.next_step, "task_occurrences": dict(self.counts),
                "event_contract": self._event_contract()}

    def restore_sampler(self, state: Mapping[str, Any]) -> None:
        if state.get("event_contract") != self._event_contract():
            raise ValueError("sampling event contract or grouping changed")
        step = state.get("next_step")
        if type(step) is not int or not 0 <= step <= self.maximum_updates:
            raise ValueError("sampler step is outside the registered training plan")
        counts = {int(task): count for task, count in state["task_occurrences"].items()}
        expected = dict.fromkeys(self.task_ids, 0)
        for group in self._groups[:step]:
            for event_index in group:
                expected[self._events[event_index]["task"]] += 1
        if counts != expected or any(type(count) is not int for count in counts.values()):
            raise ValueError("sampler exposure cursor changed")
        self.next_step, self.counts = step, counts

    def close(self) -> None:
        self.videos.close()
        self.queries.close()
        if self.diagnostic_queries is not None:
            self.diagnostic_queries.close()
