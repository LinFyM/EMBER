"""Task-equal video visits and temporal controls for code inference training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from ember.functional_adaptation.process_controls import frame_control
from ember.writer.teacher_video_schedule import TeacherVideoSchedule
from ember.writer.video_program import LanguageAxialProcessFeatures


@dataclass(frozen=True)
class MetaCodeTaskVisit:
    task_id: int
    demos: tuple[int, ...]
    action_demos: tuple[int, ...]
    frame_cost: int


@dataclass(frozen=True)
class ControlledProcessInput:
    features: LanguageAxialProcessFeatures
    frame_condition_ids: torch.Tensor
    frame_positions: torch.Tensor
    video_offsets: torch.Tensor


class MetaCodeTrainingSchedule:
    """Assign every task exactly once per macro while balancing video frames."""

    def __init__(
        self,
        *,
        task_ids: Sequence[int],
        demo_indices: Sequence[int],
        sampled_frame_counts: Mapping[int, Mapping[int, int]],
        world_size: int,
        seed: int,
        dynamic_k_max: int | None,
        temporal_controls: Sequence[str],
    ) -> None:
        self.task_ids = tuple(sorted(int(value) for value in task_ids))
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.controls = tuple(str(value) for value in temporal_controls)
        self.frame_counts = {
            int(task_id): {int(demo): int(count) for demo, count in rows.items()}
            for task_id, rows in sampled_frame_counts.items()
        }
        if (
            not self.task_ids
            or self.world_size <= 0
            or set(self.frame_counts) != set(self.task_ids)
            or not self.controls
        ):
            raise ValueError("invalid meta-code training schedule")
        self.videos = TeacherVideoSchedule(
            task_ids=self.task_ids,
            demo_indices=demo_indices,
            seed=seed,
            videos_per_visit=4 if dynamic_k_max is None else dynamic_k_max,
            dynamic_k_max=dynamic_k_max,
        )

    def control_for_macro(self, macro: int) -> str:
        if macro < 0:
            raise ValueError("meta-code macro must be non-negative")
        return self.controls[macro % len(self.controls)]

    def assignments(self, macro: int) -> tuple[tuple[MetaCodeTaskVisit, ...], ...]:
        visits = []
        for task_id in self.task_ids:
            demos = self.videos.demos_for_task_visit(task_id, macro)
            action_demos = self.videos.demos_for_task_visit(
                task_id,
                macro,
                excluded=demos,
            )
            if len(action_demos) != len(demos) or set(action_demos) & set(demos):
                raise ValueError("meta-code video/action episodes are not disjoint")
            visits.append(
                MetaCodeTaskVisit(
                    task_id=task_id,
                    demos=demos,
                    action_demos=action_demos,
                    frame_cost=sum(self.frame_counts[task_id][demo] for demo in demos),
                )
            )
        tie_order = np.random.default_rng(
            np.random.SeedSequence([self.seed, macro, 0xC0DE])
        ).permutation(self.task_ids)
        tie_rank = {int(task_id): rank for rank, task_id in enumerate(tie_order)}
        ordered = sorted(
            visits, key=lambda row: (-row.frame_cost, tie_rank[row.task_id])
        )
        groups: list[list[MetaCodeTaskVisit]] = [[] for _ in range(self.world_size)]
        loads = [0] * self.world_size
        rank_offset = (self.seed + macro) % self.world_size
        for visit in ordered:
            rank = min(
                range(self.world_size),
                key=lambda value: (
                    loads[value],
                    (value - rank_offset) % self.world_size,
                ),
            )
            groups[rank].append(visit)
            loads[rank] += visit.frame_cost
        flattened = [row.task_id for group in groups for row in group]
        if len(flattened) != len(self.task_ids) or set(flattened) != set(self.task_ids):
            raise ValueError("meta-code macro lost task-equal ownership")
        return tuple(tuple(group) for group in groups)


def controlled_process_input(
    *,
    features: LanguageAxialProcessFeatures,
    frame_condition_ids: torch.Tensor,
    frame_positions: torch.Tensor,
    video_offsets: torch.Tensor,
    condition: str,
    order_seed: int,
) -> ControlledProcessInput:
    """Reorder real per-frame features and rerun the complete posterior path."""

    offsets = video_offsets.detach().cpu().tolist()
    content_parts = []
    position_parts = []
    lengths = []
    for video, (start, stop) in enumerate(zip(offsets, offsets[1:])):
        control = frame_control(
            stop - start,
            condition=condition,
            order_seed=order_seed + video,
        )
        content_parts.append(control.content.to(frame_positions.device) + start)
        position_parts.append(
            frame_positions[start:stop].index_select(
                0, control.positions.to(frame_positions.device)
            )
        )
        lengths.append(int(control.content.numel()))
    content = torch.cat(content_parts)
    positions = torch.cat(position_parts)
    offsets_out = torch.tensor([0, *np.cumsum(lengths).tolist()], dtype=torch.long)
    return ControlledProcessInput(
        features=LanguageAxialProcessFeatures(
            text_queries=features.text_queries,
            frame_evidence=features.frame_evidence.index_select(0, content),
            patch_evidence=features.patch_evidence.index_select(0, content),
            visual_patch_tokens=features.visual_patch_tokens.index_select(0, content),
            action_probe_tokens=features.action_probe_tokens.index_select(0, content),
            valid_task_tokens=features.valid_task_tokens,
        ),
        frame_condition_ids=frame_condition_ids.index_select(0, content),
        frame_positions=positions,
        video_offsets=offsets_out,
    )
