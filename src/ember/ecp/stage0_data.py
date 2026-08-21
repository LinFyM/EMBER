"""Audited non-held plus train-fold data for ECP Stage 0."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ember.pi05_source_checkpoint import read_json
from ember.writer.data import RawTeacherVideoStore, WriterTaskAuthority


@dataclass(frozen=True)
class ECPStage0Task:
    authority_id: int
    domain: str
    domain_task_id: int
    language: str
    path: Path
    expected_bytes: int
    episode_lengths: tuple[int, ...]

    def writer_authority(self) -> WriterTaskAuthority:
        return WriterTaskAuthority(
            task_id=self.authority_id,
            language=self.language,
            path=self.path,
            expected_bytes=self.expected_bytes,
        )


@dataclass(frozen=True)
class ECPStage0Pair:
    video_demos: tuple[int, int]
    action_demos: tuple[int, int]
    speed_factors: tuple[int, int]


@dataclass(frozen=True)
class PackedStage0Pair:
    frames: torch.Tensor
    video_offsets: torch.Tensor
    frame_condition_ids: torch.Tensor
    frame_action_targets: torch.Tensor
    metrics: dict[str, Any]


def load_stage0_tasks(
    *,
    source_manifest: Path,
    target_manifest: Path,
    data_root: Path,
    held_target_ids: Sequence[int],
) -> tuple[ECPStage0Task, ...]:
    """Build one collision-free sampler namespace from two audited datasets."""

    source = read_json(source_manifest)
    target = read_json(target_manifest)
    held = set(map(int, held_target_ids))
    rows: list[tuple[str, int, str, Path, int, tuple[int, ...]]] = []
    for record in sorted(source["tasks"], key=lambda row: int(row["task_index"])):
        task_id = int(record["task_index"])
        rows.append(
            (
                "libero90_nonheld",
                task_id,
                str(record["language"]),
                data_root / "libero_90" / str(record["hdf5"]["filename"]),
                int(record["hdf5"]["bytes"]),
                tuple(map(int, record["demonstrations"]["episode_lengths"])),
            )
        )
    for record in target["tasks"]:
        task_id = int(record["global_task_id"])
        if record["split_role"] != "train" or task_id in held:
            continue
        rows.append(
            (
                "target_train_fit",
                task_id,
                str(record["language"]),
                data_root / str(record["hdf5"]["relative_path"]),
                int(record["hdf5"]["bytes"]),
                tuple(map(int, record["demonstrations"]["episode_lengths"])),
            )
        )
    tasks = tuple(
        ECPStage0Task(index, *row) for index, row in enumerate(rows)
    )
    if len(tasks) != 90 or sum(task.domain == "target_train_fit" for task in tasks) != 19:
        raise ValueError("ECP Stage 0 task roles differ from 71 non-held + fit19")
    for task in tasks:
        if (
            not task.path.is_file()
            or task.path.stat().st_size != task.expected_bytes
            or len(task.episode_lengths) != 50
        ):
            raise ValueError(f"ECP Stage 0 task authority changed: {task.authority_id}")
    return tasks


class ECPStage0Schedule:
    """Deterministic cross-episode pairs, speed views, and cost-balanced shards."""

    def __init__(
        self,
        tasks: Sequence[ECPStage0Task],
        *,
        seed: int,
        frame_stride: int = 5,
    ) -> None:
        self.tasks = tuple(tasks)
        self.by_id = {task.authority_id: task for task in tasks}
        self.seed = seed
        self.frame_stride = frame_stride

    def pair(self, authority_id: int, task_visit: int) -> ECPStage0Pair:
        order = np.random.default_rng(
            np.random.SeedSequence([self.seed, authority_id, task_visit])
        ).permutation(50)
        speeds = (1, 2) if (authority_id + task_visit) % 2 == 0 else (2, 1)
        return ECPStage0Pair(
            video_demos=(int(order[0]), int(order[1])),
            action_demos=(int(order[2]), int(order[3])),
            speed_factors=speeds,
        )

    def _stride5_count(self, raw_frames: int) -> int:
        count = (raw_frames - 1) // self.frame_stride + 1
        return count + int((raw_frames - 1) % self.frame_stride != 0)

    def pair_cost(self, authority_id: int, task_visit: int) -> int:
        task = self.by_id[authority_id]
        pair = self.pair(authority_id, task_visit)
        cost = 0
        for demo, speed in zip(
            pair.video_demos, pair.speed_factors, strict=True
        ):
            available = self._stride5_count(task.episode_lengths[demo])
            cost += (available - 1) // speed + 1
            cost += int((available - 1) % speed != 0)
        return cost

    def assignments(
        self, task_visit: int, world_size: int
    ) -> tuple[tuple[int, ...], ...]:
        if world_size <= 0 or len(self.tasks) % world_size:
            raise ValueError("ECP Stage 0 tasks must divide evenly across ranks")
        capacity = len(self.tasks) // world_size
        tie = np.random.default_rng(
            np.random.SeedSequence([self.seed, task_visit, 0xC057])
        ).permutation([task.authority_id for task in self.tasks])
        tie_rank = {int(task_id): index for index, task_id in enumerate(tie)}
        ordered = sorted(
            self.by_id,
            key=lambda task_id: (
                -self.pair_cost(task_id, task_visit), tie_rank[task_id]
            ),
        )
        groups: list[list[int]] = [[] for _ in range(world_size)]
        loads = [0] * world_size
        for task_id in ordered:
            rank = min(
                (index for index, group in enumerate(groups) if len(group) < capacity),
                key=lambda index: (loads[index], index),
            )
            groups[rank].append(task_id)
            loads[rank] += self.pair_cost(task_id, task_visit)
        return tuple(tuple(group) for group in groups)


def _speed_view(video: Any, factor: int) -> tuple[np.ndarray, np.ndarray]:
    selected = list(range(0, int(video.frames.shape[0]), factor))
    if selected[-1] != video.frames.shape[0] - 1:
        selected.append(int(video.frames.shape[0] - 1))
    return video.frames[selected], video.frame_indices[selected]


def pack_stage0_pair(
    *,
    store: RawTeacherVideoStore,
    action_store: Any,
    schedule: ECPStage0Schedule,
    authority_id: int,
    task_visit: int,
    device: torch.device,
) -> PackedStage0Pair:
    pair = schedule.pair(authority_id, task_visit)
    views = [
        _speed_view(store.load(authority_id, demo), speed)
        for demo, speed in zip(
            pair.video_demos, pair.speed_factors, strict=True
        )
    ]
    counts = [len(frames) for frames, _ in views]
    frames = torch.from_numpy(np.concatenate([row[0] for row in views])).to(
        device=device, non_blocking=True
    )
    frame_indices = torch.from_numpy(np.concatenate([row[1] for row in views])).to(
        device=device, non_blocking=True
    )
    offsets = torch.tensor(
        [0, counts[0], sum(counts)], dtype=torch.long, device=device
    )
    targets = action_store.phase_targets(
        task_id=authority_id,
        video_demos=pair.video_demos,
        action_demos=pair.action_demos,
        frame_indices=frame_indices,
        video_offsets=offsets,
        device=device,
    )
    padded_targets = targets.new_zeros(2, max(counts), *targets.shape[1:])
    for row, (start, stop) in enumerate(zip(offsets.tolist(), offsets.tolist()[1:])):
        padded_targets[row, : stop - start] = targets[start:stop]
    return PackedStage0Pair(
        frames=frames,
        video_offsets=offsets,
        frame_condition_ids=torch.zeros(
            frames.shape[0], dtype=torch.long, device=device
        ),
        frame_action_targets=padded_targets,
        metrics={
            "video_demos": list(pair.video_demos),
            "action_demos": list(pair.action_demos),
            "speed_factors": list(pair.speed_factors),
            "sampled_frames": counts,
        },
    )
