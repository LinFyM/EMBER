"""Fixed target and audited non-held sampling for supervised Writer learning.

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
        if task_id not in rows or not 0 <= task_id < 40:
            raise ValueError("selected task crosses the fixed development split")
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


def validate_training_data_config(config: Mapping[str, Any]) -> None:
    """Only the registered target-only and half-target/half-meta recipes."""
    meta = config.get("extra_meta_tasks")
    mixed = isinstance(meta, list) and bool(meta)
    conditions = config.get("conditions_per_task")
    expected = {"tasks_per_update": 8 if mixed else 4, "queries_per_task": 64,
                "cardinalities": [1], "frame_stride": 5, "include_last_frame": True}
    if (not isinstance(meta, list) or type(conditions) is not int
            or conditions not in ((1,) if mixed else (1, 2))
            or any(config.get(key) != value for key, value in expected.items())):
        raise ValueError("supervised data scientific contract changed")
    meta_fields = {"meta_task_id_offset": 40, "meta_seed": 20260911, "meta_tasks_per_update": 4,
                   "meta_source_manifest": "configs/pi05_source_corpus_v1/source_manifest.json",
                   "meta_overlap_audit": "configs/pi05_source_corpus_v1/overlap_audit.json"}
    if mixed:
        if (len(meta) != 71 or any(type(task) is not int for task in meta)
                or len(set(meta)) != 71 or any(not 40 <= task < 130 for task in meta)
                or config.get("seed") != 7
                or any(config.get(key) != value for key, value in meta_fields.items())):
            raise ValueError("audited meta scientific contract changed")
    elif any(key in config for key in meta_fields):
        raise ValueError("meta authority requires the explicit audited allowlist")


def load_meta_learning_tasks(asset_root: Path, config: Mapping[str, Any]) -> dict[int, LearningTask]:
    """Check the sealed specification authority before opening source actions."""
    validate_training_data_config(config)
    if not config["extra_meta_tasks"]:
        return {}
    manifest = read_json(asset_root / config["meta_source_manifest"])
    audit = read_json(asset_root / config["meta_overlap_audit"])
    target = read_json(asset_root / "configs/pi05_target_data_v1/manifest.json")
    rows = {int(row["task_index"]): row for row in manifest["tasks"]}
    source = {int(row["task_id"]): row for row in audit["source_tasks"]}
    active = {task for task, row in source.items() if row["decision"] == "active" and row["exact_match"] is None}
    excluded = {task for task, row in source.items() if row["decision"] == "exclude" and row["exact_match"] is not None}
    target_identity = lambda row: (row["suite"], int(row["task_id"]), row["language"])
    if (manifest["schema_version"] != "ember_pi05_source_manifest_v1"
            or audit["schema_version"] != "ember_pi05_source_overlap_v1"
            or len(rows) != len(manifest["tasks"]) or len(source) != len(audit["source_tasks"])
            or set(source) != set(range(90)) or len(active) != 71 or len(excluded) != 19
            or active | excluded != set(range(90)) or set(rows) != active
            or set(config["extra_meta_tasks"]) != {40 + task for task in active}
            or len(audit["target_tasks"]) != 40 or len(target["tasks"]) != 40
            or {target_identity(row) for row in audit["target_tasks"]}
            != {target_identity(row) for row in target["tasks"]}
            or {int(row["global_task_id"]) for row in target["tasks"]} != set(range(40))
            or {task for row in audit["equivalences"] for task in row["source_task_ids"]} != excluded):
        raise ValueError("meta manifest/audit/allowlist crosses the fixed target authority")
    for document in (manifest, audit):
        if (set(document["summary"]["active_source_task_ids"]) != active
                or set(document["summary"]["excluded_source_task_ids"]) != excluded):
            raise ValueError("meta manifest/audit summary disagrees with task authority")
    dataset = manifest["dataset"]
    if dataset != {"repo_id": "yifengzhu-hf/LIBERO-datasets", "subdir": "libero_90",
                   "revision": "f13aa24a3da8c43c7225569f28c562979fa0e35a"}:
        raise ValueError("meta source dataset authority changed")
    data_root = asset_root / "data/datasets" / dataset["revision"] / dataset["subdir"]
    output = {}
    for task in sorted(active):
        row, authority = rows[task], source[task]
        filename = row["hdf5"]["filename"]
        lengths = tuple(map(int, row["demonstrations"]["episode_lengths"]))
        if (row["language"] != authority["language"] or row["task_name"] != authority["task_name"]
                or authority["suite"] != "libero_90" or authority["problem_folder"] != "libero_90"
                or filename != authority["task_name"] + "_demo.hdf5" or Path(filename).name != filename
                or len(lengths) != 50 or min(lengths) <= 0 or int(row["hdf5"]["bytes"]) <= 0):
            raise ValueError("meta task identity or episode authority changed")
        output[40 + task] = LearningTask(
            WriterTaskAuthority(40 + task, row["language"], data_root / filename, int(row["hdf5"]["bytes"])),
            "libero_90", task, lengths,
        )
    return output


class WriterTrainingData:
    """Independent persisted task, video and action sampling streams."""

    def __init__(self, asset_root: Path, config: Mapping[str, Any], *, camera_view: str = "agentview") -> None:
        self.config = dict(config)
        validate_training_data_config(config)
        self.seed = int(config["seed"])
        self.conditions_per_task = config.get("conditions_per_task")
        if type(self.conditions_per_task) is not int or self.conditions_per_task not in (1, 2):
            raise ValueError("conditions_per_task must explicitly be 1 or 2")
        self.tasks = load_learning_tasks(asset_root, config["task_ids"])
        self.target_tasks = tuple(self.tasks)
        meta_tasks = load_meta_learning_tasks(asset_root, config)
        self.meta_tasks = tuple(meta_tasks)
        self.tasks.update(meta_tasks)
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
        if self.meta_tasks:
            meta_root = random.Random(config["meta_seed"])
            self.streams.update({f"meta_{name}": random.Random(meta_root.getrandbits(63))
                                 for name in ("task", "video", "query")})
        self.suites = {suite: tuple(task for task in self.target_tasks if self.tasks[task].suite == suite)
                       for suite in sorted({self.tasks[task].suite for task in self.target_tasks})}
        if len(self.suites) != 4 or len(self.target_tasks) != 24:
            raise ValueError("supervised updates require one task from each of four suites")
        self.next_step = 0
        self.counts = {task: 0 for task in self.tasks}

    def next_iteration(self) -> tuple[dict[str, Any], ...]:
        draws = []
        for tasks in self.suites.values():
            task = self.streams["task"].choice(tasks)
            self._append_task(draws, task, stream_prefix="")
        if self.meta_tasks:
            for task in self.streams["meta_task"].sample(self.meta_tasks, self.config["meta_tasks_per_update"]):
                self._append_task(draws, task, stream_prefix="meta_")
        self.next_step += 1
        return tuple(draws)

    def _append_task(self, draws, task, *, stream_prefix):
        demos = self.streams[stream_prefix + "video"].sample(self.video_pool, self.conditions_per_task)
        query_seed = self.streams[stream_prefix + "query"].getrandbits(63)
        query_count = int(self.config["queries_per_task"]) // self.conditions_per_task
        for condition_index, demo in enumerate(demos):
            draws.append({
                "job_id": len(draws), "condition_index": condition_index,
                "task": task, "occurrence": self.counts[task], "video_demos": (demo,),
                "query_seed": query_seed, "query_offset": condition_index * query_count,
                "query_count": query_count, "frames": self.videos.frame_counts(task, demo)[1],
            })
        self.counts[task] += 1

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

    def diagnostic_batch(self, task: int, *, seed: int, count: int):
        if task not in self.target_tasks:
            raise ValueError("action diagnostics require fixed train24 tasks")
        if self.diagnostic_queries is None:
            self.diagnostic_queries = FunctionalQueryDataset(
                tuple(self.tasks[task].authority for task in self.target_tasks),
                demo_indices=self.diagnostic_pool, action_chunk_size=50,
            )
        return self._sample_actions(self.diagnostic_queries, self.diagnostic_pool, task, 0, seed, count)

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
                **({"meta_seed": self.config["meta_seed"]} if self.meta_tasks else {}),
                "streams": {name: rng.getstate() for name, rng in self.streams.items()}}

    def restore_sampler(self, state: Mapping[str, Any]) -> None:
        if (state["seed"] != self.seed or set(state["streams"]) != set(self.streams)
                or state.get("meta_seed") != self.config.get("meta_seed")):
            raise ValueError("sampling stream contract changed")
        next_step = int(state["next_step"])
        counts = {int(task): int(count) for task, count in state["task_occurrences"].items()}
        if (next_step < 0 or set(counts) != set(self.tasks) or any(count < 0 for count in counts.values())
                or any(sum(counts[task] for task in tasks) != next_step for tasks in self.suites.values())
                or sum(counts[task] for task in self.meta_tasks) != next_step * (4 if self.meta_tasks else 0)
                or any(counts[task] > next_step for task in self.meta_tasks)):
            raise ValueError("sampler exposure cursor changed")
        self.next_step, self.counts = next_step, counts
        for name, rng in self.streams.items():
            rng.setstate(state["streams"][name])

    def close(self) -> None:
        self.videos.close()
        self.queries.close()
        if self.diagnostic_queries is not None:
            self.diagnostic_queries.close()
