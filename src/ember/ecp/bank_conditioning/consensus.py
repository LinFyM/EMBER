"""Fit-only stable functional teachers for G3 mapping acquisition."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import torch

from ember.ecp.bank_conditioning.mapping import SharedCompilerMappingSplit
from ember.ecp.shared_compiler_native_teacher import (
    NativeTeacherFactors,
    NativeTeacherStore,
)
from ember.lora import LoRAContract


def truncated_mean_update(
    pairs: Sequence[tuple[torch.Tensor, torch.Tensor]], *, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rank-truncated factors of the mean low-rank update.

    Each pair is ``A[rank,in]`` and already-scaled ``B[out,rank]``.  The
    calculation stays in the concatenated small core and never materializes
    an ``out x in`` update.
    """

    rows = tuple(pairs)
    if (
        not rows
        or rank <= 0
        or any(a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1] for a, b in rows)
    ):
        raise ValueError("functional consensus factor topology changed")
    a = torch.cat(tuple(value[0].float() for value in rows), dim=0)
    b = torch.cat(tuple(value[1].float() for value in rows), dim=1) / len(rows)
    q_b, r_b = torch.linalg.qr(b, mode="reduced")
    q_a, r_a = torch.linalg.qr(a.transpose(0, 1), mode="reduced")
    left, singular, right = torch.linalg.svd(
        r_b @ r_a.transpose(0, 1), full_matrices=False
    )
    if singular.numel() < rank or not bool(torch.isfinite(singular).all()):
        raise ValueError("functional consensus rank is unsupported")
    root = singular[:rank].clamp_min(0).sqrt()
    consensus_a = root[:, None] * (right[:rank] @ q_a.transpose(0, 1))
    consensus_b = (q_b @ left[:, :rank]) * root[None]
    return consensus_a, consensus_b


def _direction_scale(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a_scale = a.square().mean(-1).sqrt().clamp_min(1e-12)
    b_rows = b.transpose(0, 1)
    b_scale = b_rows.square().mean(-1).sqrt().clamp_min(1e-12)
    return a / a_scale[:, None], b_rows / b_scale[:, None], a_scale * b_scale


class FitConsensusTeacherStore:
    """Cache one cross-video functional target per fit task/member.

    Only videos in ``mapping_split.fit`` enter the mean.  The pre-registered
    held video remains a read-only evaluation target and is not used here.
    """

    def __init__(
        self,
        native_teachers: NativeTeacherStore,
        mapping_split: SharedCompilerMappingSplit,
        contract: LoRAContract,
    ) -> None:
        if contract.rank != 4:
            raise ValueError("functional consensus requires rank four")
        self.native_teachers = native_teachers
        self.fit_by_task = mapping_split.fit_by_task
        self.contract = contract
        self.cache: dict[int, dict[str, NativeTeacherFactors]] = {}

    def lookup_members(
        self,
        *,
        authority_id: int,
        video_demo: int,
        member_names: Sequence[str],
    ) -> tuple[NativeTeacherFactors, ...]:
        if authority_id not in self.cache:
            self.cache[authority_id] = self._build_task(authority_id, member_names)
        rows = self.cache[authority_id]
        expected = tuple(map(str, member_names))
        if set(rows) != set(expected):
            raise ValueError("functional consensus member authority changed")
        return tuple(
            replace(rows[name], video_demo=int(video_demo)) for name in expected
        )

    def _build_task(
        self, authority_id: int, member_names: Sequence[str]
    ) -> dict[str, NativeTeacherFactors]:
        conditions = self.fit_by_task.get(int(authority_id), ())
        videos = tuple(sorted({int(row.video_demo) for row in conditions}))
        if len(videos) < 2:
            raise ValueError("functional consensus needs two fit videos")
        result = {}
        for member in map(str, member_names):
            teachers = tuple(
                self.native_teachers.lookup(
                    authority_id=authority_id,
                    k=1,
                    video_demo=video,
                    member_name=member,
                )
                for video in videos
            )
            if any(teacher is None for teacher in teachers):
                raise ValueError("functional consensus lost a fit teacher")
            a_rows = []
            b_rows = []
            scales = []
            for target in range(len(self.contract.targets)):
                pairs = tuple(
                    (
                        teacher.a[target],
                        teacher.b[target].transpose(0, 1)
                        * teacher.scales[target][None],
                    )
                    for teacher in teachers
                    if teacher is not None
                )
                a, b = truncated_mean_update(pairs, rank=self.contract.rank)
                a_direction, b_direction, scale = _direction_scale(a, b)
                a_rows.append(a_direction)
                b_rows.append(b_direction)
                scales.append(scale)
            prototype = NativeTeacherFactors(
                authority_id=int(authority_id),
                video_demo=videos[0],
                member_name=member,
                a=tuple(a_rows),
                b=tuple(b_rows),
                scales=torch.stack(scales),
                provenance={
                    "kind": "fit_video_rank4_truncated_mean_update",
                    "fit_video_demos": list(videos),
                    "held_video_used": False,
                    "member_name": member,
                },
            )
            prototype.lora_state(self.contract)
            result[member] = prototype
        return result
