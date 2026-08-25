"""Fold-correct K-video data for G2 Natural Program."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ember.ecp.natural_program_labels import NaturalProgramLabelStore
from ember.meta_protocol import load_meta_protocol, meta_task_split
from ember.pi05_source_checkpoint import read_json
from ember.writer.data import RawTeacherVideo, RawTeacherVideoStore, WriterTaskAuthority


@dataclass(frozen=True)
class NaturalProgramTask:
    authority_id: int
    domain: str
    domain_task_id: int
    role: str
    suite: str
    language: str
    task_name: str
    problem_folder: str
    bddl_file: str
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
class NaturalProgramSample:
    video_demos: tuple[int, ...]
    action_demos: tuple[int, ...]
    k: int
    robustness_view: str


@dataclass(frozen=True)
class PackedNaturalProgramCondition:
    frames: torch.Tensor
    frame_indices: torch.Tensor
    raw_frame_counts: torch.Tensor
    video_offsets: torch.Tensor
    video_set_offsets: torch.Tensor
    frame_condition_ids: torch.Tensor
    query_times: torch.Tensor
    action_targets: torch.Tensor
    progress_targets: torch.Tensor
    rising_targets: torch.Tensor
    contact_targets: torch.Tensor
    contact_mask: torch.Tensor
    predicate_targets: torch.Tensor
    predicate_mask: torch.Tensor
    metrics: dict[str, Any]


@dataclass(frozen=True)
class _PackedSupervision:
    query_times: torch.Tensor
    action_targets: torch.Tensor
    progress_targets: torch.Tensor
    rising_targets: torch.Tensor
    contact_targets: torch.Tensor
    contact_mask: torch.Tensor
    predicate_targets: torch.Tensor
    predicate_mask: torch.Tensor


def load_natural_program_tasks(
    *,
    meta_protocol_path: Path,
    source_manifest_path: Path,
    target_manifest_path: Path,
    data_root: Path,
    target_fit_ids: Sequence[int],
    target_held_ids: Sequence[int],
    held_meta_fold: int = 0,
) -> tuple[NaturalProgramTask, ...]:
    protocol = load_meta_protocol(meta_protocol_path)
    split = meta_task_split(protocol, held_out_fold=held_meta_fold)
    meta_roles = {task.task_id: "meta_fit" for task in split.train}
    meta_roles.update({task.task_id: "meta_held" for task in split.validation})
    source = read_json(source_manifest_path)
    source_by_id = {int(row["task_index"]): row for row in source["tasks"]}
    rows: list[NaturalProgramTask] = []
    for domain_task_id in map(int, protocol["active_source_task_ids"]):
        record = source_by_id[domain_task_id]
        task_name = str(record["task_name"])
        rows.append(
            NaturalProgramTask(
                authority_id=len(rows),
                domain="libero90_nonheld",
                domain_task_id=domain_task_id,
                role=meta_roles[domain_task_id],
                suite="libero_90",
                language=str(record["language"]),
                task_name=task_name,
                problem_folder="libero_90",
                bddl_file=f"{task_name}.bddl",
                path=data_root / "libero_90" / str(record["hdf5"]["filename"]),
                expected_bytes=int(record["hdf5"]["bytes"]),
                episode_lengths=tuple(
                    map(int, record["demonstrations"]["episode_lengths"])
                ),
            )
        )

    fit = set(map(int, target_fit_ids))
    held = set(map(int, target_held_ids))
    if len(fit) != 19 or len(held) != 5 or fit & held:
        raise ValueError("Natural Program target fold is not 19 fit / 5 held")
    target = read_json(target_manifest_path)
    target_rows = sorted(
        (
            row
            for row in target["tasks"]
            if row["split_role"] == "train"
            and int(row["global_task_id"]) in fit | held
        ),
        key=lambda row: int(row["global_task_id"]),
    )
    for record in target_rows:
        domain_task_id = int(record["global_task_id"])
        rows.append(
            NaturalProgramTask(
                authority_id=len(rows),
                domain="target_train24",
                domain_task_id=domain_task_id,
                role="target_fit" if domain_task_id in fit else "target_held",
                suite=str(record["suite"]),
                language=str(record["language"]),
                task_name=str(record["task_name"]),
                problem_folder=str(record["problem_folder"]),
                bddl_file=str(record["bddl"]["filename"]),
                path=data_root / str(record["hdf5"]["relative_path"]),
                expected_bytes=int(record["hdf5"]["bytes"]),
                episode_lengths=tuple(
                    map(int, record["demonstrations"]["episode_lengths"])
                ),
            )
        )
    counts = {role: sum(task.role == role for task in rows) for role in {
        "meta_fit", "meta_held", "target_fit", "target_held"
    }}
    if counts != {
        "meta_fit": 56,
        "meta_held": 15,
        "target_fit": 19,
        "target_held": 5,
    }:
        raise ValueError(f"Natural Program fold roles changed: {counts}")
    for task in rows:
        if (
            not task.path.is_file()
            or task.path.stat().st_size != task.expected_bytes
            or len(task.episode_lengths) != 50
        ):
            raise ValueError(
                f"Natural Program task authority changed: {task.authority_id}"
            )
    return tuple(rows)


class NaturalProgramSchedule:
    """Role-balanced macros and deterministic, disjoint K-video queries."""

    def __init__(
        self,
        tasks: Sequence[NaturalProgramTask],
        *,
        seed: int,
        query_points: int,
    ) -> None:
        self.tasks = tuple(tasks)
        self.by_id = {task.authority_id: task for task in tasks}
        self.meta_fit = tuple(
            task.authority_id for task in tasks if task.role == "meta_fit"
        )
        self.target_fit = tuple(
            task.authority_id for task in tasks if task.role == "target_fit"
        )
        self.seed = int(seed)
        self.query_points = int(query_points)
        if len(self.by_id) != 95 or self.query_points < 2:
            raise ValueError("invalid Natural Program schedule")
        self._meta_order = tuple(
            int(value)
            for value in np.random.default_rng(
                np.random.SeedSequence([self.seed, 0x4D455441])
            ).permutation(self.meta_fit)
        )

    def training_task_ids(self, macro: int) -> tuple[int, ...]:
        start = (int(macro) * len(self.target_fit)) % len(self.meta_fit)
        meta = tuple(
            self._meta_order[(start + offset) % len(self.meta_fit)]
            for offset in range(len(self.target_fit))
        )
        return tuple(self.target_fit) + meta

    def sample(self, authority_id: int, visit: int) -> NaturalProgramSample:
        if authority_id not in self.by_id or visit < 0:
            raise ValueError("invalid Natural Program task visit")
        k_values = (1, 2, 4)
        k = k_values[(authority_id + visit) % len(k_values)]
        order = np.random.default_rng(
            np.random.SeedSequence([self.seed, authority_id, visit, 0x4B564944])
        ).permutation(50)
        robustness = "speed2" if (authority_id + visit) % 2 == 0 else "crop80"
        return NaturalProgramSample(
            video_demos=tuple(map(int, order[:k])),
            action_demos=tuple(map(int, order[k : 2 * k])),
            k=k,
            robustness_view=robustness,
        )

    def contrastive_task_ids(
        self, authority_id: int, visit: int, *, count: int
    ) -> tuple[int, ...]:
        """Choose role-balanced fit-language negatives independent of rank."""

        if count <= 0 or count % 2:
            raise ValueError("invalid Natural Program contrastive negative count")
        rows = []
        for role, pool in enumerate((self.target_fit, self.meta_fit)):
            candidates = tuple(
                task_id for task_id in pool if task_id != authority_id
            )
            if count // 2 > len(candidates):
                raise ValueError("too few role-balanced contrastive tasks")
            order = np.random.default_rng(
                np.random.SeedSequence(
                    [self.seed, authority_id, visit, role, 0x4E4547]
                )
            ).permutation(len(candidates))
            rows.extend(candidates[int(index)] for index in order[: count // 2])
        return tuple(rows)

    def _sample_cost(self, authority_id: int, visit: int) -> int:
        task = self.by_id[authority_id]
        sample = self.sample(authority_id, visit)
        return sum(
            (task.episode_lengths[demo] - 1) // 5 + 2
            for demo in sample.video_demos
        )

    def _assign_tasks(
        self, tasks: Sequence[int], *, macro: int, world_size: int
    ) -> tuple[tuple[int, ...], ...]:
        if not 1 <= world_size <= 6:
            raise ValueError("Natural Program world size is outside 1..6")
        ordered = sorted(tasks, key=lambda task: (-self._sample_cost(task, macro), task))
        groups: list[list[int]] = [[] for _ in range(world_size)]
        loads = [0] * world_size
        for task in ordered:
            rank = min(range(world_size), key=lambda row: (loads[row], row))
            groups[rank].append(task)
            loads[rank] += self._sample_cost(task, macro)
        return tuple(tuple(group) for group in groups)

    def optimizer_task_groups(
        self, macro: int, *, tasks_per_role: int
    ) -> tuple[tuple[int, ...], ...]:
        """Split one task-equal macro into role-balanced optimizer updates."""

        if tasks_per_role <= 0:
            raise ValueError("G2 optimizer tasks per role must be positive")
        tasks = self.training_task_ids(macro)
        by_role = {
            role: tuple(task for task in tasks if self.by_id[task].role == role)
            for role in ("target_fit", "meta_fit")
        }
        if len(by_role["target_fit"]) != len(by_role["meta_fit"]):
            raise ValueError("G2 optimizer roles lost equal macro mass")
        offset = int(macro) % len(by_role["target_fit"])
        by_role = {
            role: values[offset:] + values[:offset]
            for role, values in by_role.items()
        }
        return tuple(
            by_role["target_fit"][start : start + tasks_per_role]
            + by_role["meta_fit"][start : start + tasks_per_role]
            for start in range(0, len(by_role["target_fit"]), tasks_per_role)
        )

    def optimizer_assignments(
        self, macro: int, world_size: int, *, tasks_per_role: int
    ) -> tuple[tuple[tuple[int, ...], ...], ...]:
        return tuple(
            self._assign_tasks(tasks, macro=macro, world_size=world_size)
            for tasks in self.optimizer_task_groups(
                macro, tasks_per_role=tasks_per_role
            )
        )

    def assignments(self, macro: int, world_size: int) -> tuple[tuple[int, ...], ...]:
        return self._assign_tasks(
            self.training_task_ids(macro), macro=macro, world_size=world_size
        )


def _video_view(video: RawTeacherVideo, view: str) -> tuple[np.ndarray, np.ndarray]:
    count = int(video.frames.shape[0])
    if view == "full":
        selected = np.arange(count)
    elif view == "endpoints":
        selected = np.asarray([0, count - 1])
    elif view == "speed2":
        selected = np.arange(0, count, 2)
        if selected[-1] != count - 1:
            selected = np.concatenate((selected, [count - 1]))
    elif view == "crop80":
        left = max(0, int(round(0.1 * (count - 1))))
        right = min(count - 1, int(round(0.9 * (count - 1))))
        selected = np.arange(left, right + 1)
    else:
        raise ValueError(f"unsupported Natural Program video view: {view}")
    return video.frames[selected], video.frame_indices[selected]


def _query_indices(raw_count: int, points: int) -> np.ndarray:
    return np.rint(np.linspace(0.0, raw_count - 1, points)).astype(np.int64)


def _windowed_rising(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Preserve any transition between adjacent sparse query locations."""

    result = np.zeros(indices.shape, dtype=np.float32)
    previous = -1
    for row, current in enumerate(map(int, indices)):
        result[row] = float(np.asarray(values[previous + 1 : current + 1]).max())
        previous = current
    return result


def pack_ordered_teacher_videos(
    videos: Sequence[RawTeacherVideo],
    *,
    view: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    viewed = tuple(_video_view(video, view) for video in videos)
    counts = [int(frames.shape[0]) for frames, _ in viewed]
    frames = torch.from_numpy(np.concatenate([row[0] for row in viewed])).to(
        device=device, non_blocking=True
    )
    frame_indices = torch.from_numpy(np.concatenate([row[1] for row in viewed])).to(
        device=device, non_blocking=True
    )
    raw_counts = torch.tensor(
        [video.raw_frame_count for video in videos],
        dtype=torch.long,
        device=device,
    )
    offsets = torch.tensor(
        [0, *np.cumsum(counts).tolist()], dtype=torch.long, device=device
    )
    return frames, frame_indices, raw_counts, offsets, counts


def _pack_cross_episode_supervision(
    *,
    task: NaturalProgramTask,
    sample: NaturalProgramSample,
    action_store: Any,
    label_store: NaturalProgramLabelStore,
    query_points: int,
    predicate_slots: int,
    device: torch.device,
) -> _PackedSupervision:
    # This single action-episode grid owns action and every dynamic label.
    # phase_targets maps it identically because each row includes that action
    # episode's final index, avoiding a second rounding through video length.
    action_query_rows = [
        _query_indices(task.episode_lengths[demo], query_points)
        for demo in sample.action_demos
    ]
    action_indices = torch.from_numpy(np.concatenate(action_query_rows)).to(
        device=device, non_blocking=True
    )
    action_offsets = torch.arange(
        0,
        (sample.k + 1) * query_points,
        query_points,
        dtype=torch.long,
        device=device,
    )
    action = action_store.phase_targets(
        task_id=task.authority_id,
        video_demos=sample.video_demos,
        action_demos=sample.action_demos,
        frame_indices=action_indices,
        video_offsets=action_offsets,
        device=device,
    ).reshape(sample.k, query_points, -1, 7).mean(0)

    progress = []
    rising = []
    contact = []
    contact_masks = []
    predicates = []
    masks = []
    for demo, indices in zip(
        sample.action_demos, action_query_rows, strict=True
    ):
        labels = label_store.load(task.authority_id, demo)
        progress.append(labels.progress[indices])
        rising.append(_windowed_rising(labels.rising, indices))
        contact.append(labels.contact[indices])
        contact_masks.append(labels.contact_mask[indices])
        predicates.append(labels.predicates[indices])
        masks.append(labels.predicate_mask)
    predicate_mask = np.stack(masks).all(0)
    if predicate_mask.shape != (predicate_slots,):
        raise ValueError("Natural Program predicate label width changed")
    contact_values = np.stack(contact).astype(np.float32, copy=False)
    contact_valid = np.stack(contact_masks).astype(np.float32, copy=False)
    contact_mass = contact_valid.sum(0)
    contact_mean = (contact_values * contact_valid).sum(0) / np.maximum(
        contact_mass, 1.0
    )
    tensor = {"device": device, "dtype": torch.float32}
    return _PackedSupervision(
        query_times=torch.linspace(0.0, 1.0, query_points, **tensor)[None],
        action_targets=action[None],
        progress_targets=torch.from_numpy(np.stack(progress).mean(0)).to(**tensor)[
            None
        ],
        rising_targets=torch.from_numpy(np.stack(rising).mean(0)).to(**tensor)[None],
        contact_targets=torch.from_numpy(contact_mean).to(**tensor)[None],
        contact_mask=torch.from_numpy(contact_mass > 0).to(
            device=device, dtype=torch.bool
        )[None],
        predicate_targets=torch.from_numpy(np.stack(predicates).mean(0)).to(
            **tensor
        )[None],
        predicate_mask=torch.from_numpy(predicate_mask).to(
            device=device, dtype=torch.bool
        )[None],
    )


def pack_natural_program_condition(
    *,
    task: NaturalProgramTask,
    sample: NaturalProgramSample,
    video_store: RawTeacherVideoStore,
    action_store: Any,
    label_store: NaturalProgramLabelStore,
    query_points: int,
    predicate_slots: int,
    device: torch.device,
    view: str = "full",
) -> PackedNaturalProgramCondition:
    videos = tuple(
        video_store.load(task.authority_id, demo) for demo in sample.video_demos
    )
    frames, frame_indices, raw_counts, offsets, counts = pack_ordered_teacher_videos(
        videos, view=view, device=device
    )
    supervision = _pack_cross_episode_supervision(
        task=task,
        sample=sample,
        action_store=action_store,
        label_store=label_store,
        query_points=query_points,
        predicate_slots=predicate_slots,
        device=device,
    )
    return PackedNaturalProgramCondition(
        frames=frames,
        frame_indices=frame_indices,
        raw_frame_counts=raw_counts,
        video_offsets=offsets,
        video_set_offsets=torch.tensor(
            [0, len(videos)], dtype=torch.long, device=device
        ),
        frame_condition_ids=torch.zeros(
            frames.shape[0], dtype=torch.long, device=device
        ),
        query_times=supervision.query_times,
        action_targets=supervision.action_targets,
        progress_targets=supervision.progress_targets,
        rising_targets=supervision.rising_targets,
        contact_targets=supervision.contact_targets,
        contact_mask=supervision.contact_mask,
        predicate_targets=supervision.predicate_targets,
        predicate_mask=supervision.predicate_mask,
        metrics={
            "K": sample.k,
            "video_demos": list(sample.video_demos),
            "action_demos": list(sample.action_demos),
            "view": view,
            "sampled_frames": counts,
            "raw_frame_counts": [video.raw_frame_count for video in videos],
        },
    )
