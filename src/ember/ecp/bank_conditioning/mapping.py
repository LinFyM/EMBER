"""Pre-registered G3 mapping split, schedule, and paired-update credit."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.shared_compiler import SharedCompilerOutput
from ember.ecp.shared_compiler_native_teacher import (
    NativeTeacherFactors,
    native_teacher_supervision_loss,
)
from ember.pi05_source_checkpoint import read_json


@dataclass(frozen=True)
class MappingCondition:
    authority_id: int
    role: str
    video_demo: int
    sampled_frames: int


@dataclass(frozen=True)
class SharedCompilerMappingSplit:
    fit: tuple[MappingCondition, ...]
    video_held: tuple[MappingCondition, ...]
    task_held: tuple[MappingCondition, ...]
    member_names: Mapping[int, tuple[str, ...]]

    @property
    def fit_by_task(self) -> dict[int, tuple[MappingCondition, ...]]:
        return _by_task(self.fit)

    @property
    def video_held_by_task(self) -> dict[int, tuple[MappingCondition, ...]]:
        return _by_task(self.video_held)


@dataclass(frozen=True)
class MappingLoss:
    total: torch.Tensor
    input_subspace: torch.Tensor
    output_subspace: torch.Tensor
    update_direction: torch.Tensor
    member_distances: torch.Tensor
    responsibilities: torch.Tensor
    family_recovery: torch.Tensor
    best_family_recovery: torch.Tensor
    best_member: int


@dataclass(frozen=True)
class MappingConsistencyLoss:
    total: torch.Tensor
    predicted_family_distance: torch.Tensor
    allowed_family_distance: torch.Tensor


def _by_task(
    conditions: Sequence[MappingCondition],
) -> dict[int, tuple[MappingCondition, ...]]:
    rows: dict[int, list[MappingCondition]] = {}
    for condition in conditions:
        rows.setdefault(condition.authority_id, []).append(condition)
    return {
        task: tuple(sorted(values, key=lambda row: row.video_demo))
        for task, values in rows.items()
    }


def _task_teacher_authority(
    record: Mapping[str, Any],
) -> tuple[str, tuple[str, ...], dict[int, int]]:
    path = Path(str(record.get("manifest", ""))).resolve()
    if not path.is_file() or path.stat().st_size != int(
        record.get("manifest_bytes", -1)
    ):
        raise ValueError("G3 mapping task teacher manifest changed")
    manifest = read_json(path)
    role = str(manifest.get("task", {}).get("role", ""))
    members = tuple(map(str, manifest.get("member_names", ())))
    videos = tuple(map(int, manifest.get("video_demos", ())))
    frame_counts: dict[int, int] = {}
    for row in manifest.get("teachers", ()):
        video = int(row.get("video_demo", -1))
        sampled = tuple(
            map(
                int,
                row.get("provenance", {})
                .get("video_capture", {})
                .get("sampled_frames", ()),
            )
        )
        if len(sampled) != 1 or sampled[0] <= 0:
            raise ValueError("G3 mapping video cost authority changed")
        previous = frame_counts.setdefault(video, sampled[0])
        if previous != sampled[0]:
            raise ValueError("G3 mapping member video costs disagree")
    if (
        role not in {"meta_fit", "target_fit"}
        or not members
        or set(frame_counts) != set(videos)
        or len(videos) != int(record.get("video_count", -1))
        or len(members) != int(record.get("member_count", -1))
    ):
        raise ValueError("G3 mapping task teacher authority changed")
    return role, members, frame_counts


def load_mapping_split(
    config: Mapping[str, Any], *, asset_root: Path
) -> SharedCompilerMappingSplit:
    """Resolve the compact pre-registration against all 451 sealed videos."""

    split = config.get("mapping_split", {})
    root_path = Path(str(config["authorities"]["native_teacher_manifest"]))
    if not root_path.is_absolute():
        root_path = asset_root / root_path
    root = read_json(root_path.resolve())
    records = tuple(root.get("records", ()))
    by_id = {int(row.get("authority_id", -1)): row for row in records}
    task_fit = tuple(map(int, split.get("task_fit_ids", ())))
    task_held = tuple(map(int, split.get("task_holdout_ids", ())))
    task_fit_set = set(task_fit)
    task_held_set = set(task_held)
    held_video = {
        int(task): int(video)
        for task, video in split.get("held_video_by_fit_task", {}).items()
    }
    preregistered = all(
        (
            root.get("status") == "complete",
            int(root.get("task_count", -1)) == 50,
            int(root.get("video_count", -1)) == 451,
            len(task_fit) == len(task_fit_set) == 40,
            len(task_held) == len(task_held_set) == 10,
            not task_fit_set.intersection(task_held_set),
            task_fit_set.union(task_held_set) == set(by_id),
            set(held_video) == task_fit_set,
            split.get("selection_uses_outcomes") is False,
        )
    )
    if not preregistered:
        raise ValueError("G3 mapping split pre-registration changed")

    fit: list[MappingCondition] = []
    video_holdout: list[MappingCondition] = []
    task_holdout: list[MappingCondition] = []
    members: dict[int, tuple[str, ...]] = {}
    for task_id in sorted(by_id):
        role, member_names, frame_counts = _task_teacher_authority(by_id[task_id])
        members[task_id] = member_names
        for video, frames in sorted(frame_counts.items()):
            condition = MappingCondition(task_id, role, video, frames)
            if task_id in task_held_set:
                task_holdout.append(condition)
            elif video == held_video[task_id]:
                video_holdout.append(condition)
            else:
                fit.append(condition)
    roles = {
        role: len({row.authority_id for row in fit if row.role == role})
        for role in ("meta_fit", "target_fit")
    }
    resolved = all(
        (
            len(fit) == 329,
            len(video_holdout) == 40,
            len(task_holdout) == 82,
            roles == {"meta_fit": 25, "target_fit": 15},
            all(len(rows) >= 2 for rows in _by_task(fit).values()),
            split.get("counts")
            == {
                "fit_conditions": 329,
                "video_holdout_conditions": 40,
                "task_holdout_conditions": 82,
                "total_conditions": 451,
            },
        )
    )
    if not resolved:
        raise ValueError("G3 mapping split resolved to a different authority")
    return SharedCompilerMappingSplit(
        fit=tuple(fit),
        video_held=tuple(video_holdout),
        task_held=tuple(task_holdout),
        member_names=members,
    )


class SharedCompilerMappingSchedule:
    """Fixed six-task role-balanced updates independent of world size."""

    def __init__(
        self, split: SharedCompilerMappingSplit, *, seed: int
    ) -> None:
        self.split = split
        self.seed = int(seed)
        self.fit_by_task = split.fit_by_task
        self.meta = tuple(
            sorted(
                task
                for task, rows in self.fit_by_task.items()
                if rows[0].role == "meta_fit"
            )
        )
        self.target = tuple(
            sorted(
                task
                for task, rows in self.fit_by_task.items()
                if rows[0].role == "target_fit"
            )
        )
        if len(self.meta) != 25 or len(self.target) != 15:
            raise ValueError("G3 mapping role schedule changed")

    def task_groups(self, macro: int) -> tuple[tuple[int, ...], ...]:
        if macro < 0:
            raise ValueError("G3 mapping macro is negative")
        generator = random.Random(self.seed + 1009 * macro)
        meta = list(self.meta)
        target = list(self.target)
        generator.shuffle(meta)
        generator.shuffle(target)
        selected_meta = meta[:15]
        return tuple(
            tuple((*selected_meta[start : start + 3], *target[start : start + 3]))
            for start in range(0, 15, 3)
        )

    def condition(
        self, task_id: int, *, macro: int, update: int
    ) -> MappingCondition:
        rows = self.fit_by_task[task_id]
        generator = random.Random(
            self.seed + 1000003 * macro + 1009 * update + 17 * task_id
        )
        return rows[generator.randrange(len(rows))]

    def companion(
        self, primary: MappingCondition, *, macro: int, update: int
    ) -> MappingCondition:
        rows = self.fit_by_task[primary.authority_id]
        candidates = tuple(row for row in rows if row.video_demo != primary.video_demo)
        generator = random.Random(
            self.seed + 2000003 * macro + 2017 * update + 31 * primary.authority_id
        )
        return candidates[generator.randrange(len(candidates))]

    @staticmethod
    def assignments(
        group: Sequence[int],
        conditions: Mapping[int, MappingCondition],
        world_size: int,
    ) -> tuple[tuple[int, ...], ...]:
        if len(group) != 6 or not 1 <= world_size <= 6:
            raise ValueError("G3 mapping global batch or world size changed")
        rows: list[list[int]] = [[] for _ in range(world_size)]
        loads = [0] * world_size
        maximum_tasks = math.ceil(len(group) / world_size)
        for task in sorted(
            group,
            key=lambda value: (-conditions[value].sampled_frames, value),
        ):
            eligible = [
                rank
                for rank in range(world_size)
                if len(rows[rank]) < maximum_tasks
            ]
            rank = min(eligible, key=lambda value: (loads[value], value))
            rows[rank].append(task)
            loads[rank] += conditions[task].sampled_frames
        return tuple(tuple(row) for row in rows)


def _factor_inner(
    a1: torch.Tensor,
    b1: torch.Tensor,
    a2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    return (
        (b1.float().transpose(0, 1) @ b2.float())
        * (a1.float() @ a2.float().transpose(0, 1))
    ).sum()


def _update_cosine(
    a1: torch.Tensor,
    b1: torch.Tensor,
    a2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    dot = _factor_inner(a1, b1, a2, b2)
    left = _factor_inner(a1, b1, a1, b1).clamp_min(0).sqrt()
    right = _factor_inner(a2, b2, a2, b2).clamp_min(0).sqrt()
    return (dot / (left * right).clamp_min(1e-12)).clamp(-1.0, 1.0)


def _student_pairs(
    output: SharedCompilerOutput, *, detach: bool
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    values = []
    for target, (a, b) in enumerate(
        zip(output.input_directions, output.output_directions, strict=True)
    ):
        scale = output.residual.scales[target].detach()
        pair = (a, b.transpose(0, 1) * scale[None])
        values.append(tuple(value.detach() for value in pair) if detach else pair)
    return tuple(values)


def _teacher_pairs(
    teacher: NativeTeacherFactors,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (
            a,
            b.transpose(0, 1) * teacher.scales[target][None],
        )
        for target, (a, b) in enumerate(zip(teacher.a, teacher.b, strict=True))
    )


def _family_cosines(
    left: Sequence[tuple[torch.Tensor, torch.Tensor]],
    right: Sequence[tuple[torch.Tensor, torch.Tensor]],
    owners: Sequence[TargetOwner],
) -> torch.Tensor:
    target = torch.stack(
        [
            _update_cosine(a1, b1, a2.to(a1), b2.to(b1))
            for (a1, b1), (a2, b2) in zip(left, right, strict=True)
        ]
    )
    return torch.stack(
        [
            target[
                torch.tensor(
                    [owner.family is family for owner in owners],
                    device=target.device,
                    dtype=torch.bool,
                )
            ].mean()
            for family in TargetFamily
        ]
    )


def paired_mapping_loss(
    *,
    output: SharedCompilerOutput,
    teachers: Sequence[NativeTeacherFactors],
    owners: Sequence[TargetOwner],
    temperature: float,
) -> MappingLoss:
    """Use one set-valued, four-family paired-update objective.

    Input/output subspaces remain diagnostics.  They are bank-dependent
    projections of the same functional update and therefore cannot be an
    equal-weight mapping target across videos.
    """

    if not teachers or temperature <= 0:
        raise ValueError("G3 mapping teacher set or temperature changed")
    student = _student_pairs(output, detach=False)
    family = torch.stack(
        [
            _family_cosines(student, _teacher_pairs(teacher), owners)
            for teacher in teachers
        ]
    )
    distances = 1.0 - family.mean(-1)
    log_prior = -math.log(len(teachers))
    logits = log_prior - distances / float(temperature)
    responsibilities = logits.softmax(0)
    credit = native_teacher_supervision_loss(
        student_a_directions=output.input_directions,
        student_b_directions=output.output_directions,
        student_scales=output.residual.scales,
        teachers=teachers,
        owners=owners,
        member_weights=responsibilities.detach(),
        selection_weight=1.0,
        spectrum_weight=0.0,
    )
    best = int(distances.detach().argmin())
    return MappingLoss(
        total=credit.update_direction,
        input_subspace=credit.input_subspace,
        output_subspace=credit.output_subspace,
        update_direction=credit.update_direction,
        member_distances=distances,
        responsibilities=responsibilities,
        family_recovery=(responsibilities.detach()[:, None] * family).sum(0),
        best_family_recovery=family[best],
        best_member=best,
    )


def cross_video_consistency_loss(
    *,
    primary_output: SharedCompilerOutput,
    companion_output: SharedCompilerOutput,
    primary_teachers: Sequence[NativeTeacherFactors],
    companion_teachers: Sequence[NativeTeacherFactors],
    owners: Sequence[TargetOwner],
    responsibilities: torch.Tensor,
    margin: float,
) -> MappingConsistencyLoss:
    """Bound student dispersion by same-member teacher dispersion."""

    primary_by_name = {row.member_name: row for row in primary_teachers}
    companion_by_name = {row.member_name: row for row in companion_teachers}
    names = tuple(row.member_name for row in primary_teachers)
    if (
        set(primary_by_name) != set(companion_by_name)
        or responsibilities.shape != (len(names),)
        or margin < 0
    ):
        raise ValueError("G3 mapping consistency member set changed")
    predicted = 1.0 - _family_cosines(
        _student_pairs(primary_output, detach=True),
        _student_pairs(companion_output, detach=False),
        owners,
    )
    teacher = torch.stack(
        [
            1.0
            - _family_cosines(
                _teacher_pairs(primary_by_name[name]),
                _teacher_pairs(companion_by_name[name]),
                owners,
            )
            for name in names
        ]
    )
    allowed = (responsibilities.detach()[:, None].to(teacher) * teacher).sum(0)
    total = torch.relu(predicted - allowed - float(margin)).mean()
    return MappingConsistencyLoss(
        total=total,
        predicted_family_distance=predicted,
        allowed_family_distance=allowed,
    )
