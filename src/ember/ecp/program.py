"""Shared Program schema and visible anchors for EMBER-ECP."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.stage0 import ECPVideoEncoderOutput


@dataclass(frozen=True)
class ECPProgram:
    """One fixed-capacity, dynamically occupied policy Program distribution."""

    language: torch.Tensor
    scene: torch.Tensor
    process: torch.Tensor
    presence: torch.Tensor
    uncertainty: torch.Tensor

    def prior_only(self) -> "ECPProgram":
        return ECPProgram(
            language=self.language,
            scene=self.scene,
            process=torch.zeros_like(self.process),
            presence=torch.zeros_like(self.presence),
            uncertainty=torch.zeros_like(self.uncertainty),
        )


def _owner_coordinates(
    owners: tuple[TargetOwner, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    families = tuple(TargetFamily)
    family_ids = torch.tensor(
        [families.index(owner.family) for owner in owners], dtype=torch.long
    )
    layer_ids = torch.tensor(
        [
            owner.layer
            if owner.layer is not None
            else (17 if owner.family is TargetFamily.ACTION_OUT else 0)
            for owner in owners
        ],
        dtype=torch.long,
    )
    return family_ids, layer_ids


class VisibleProgramProjector(torch.nn.Module):
    """Build language, scene, and ordered process anchors before privileged input."""

    def __init__(
        self,
        owners: tuple[TargetOwner, ...],
        *,
        width: int = 128,
        event_slots: int = 8,
    ) -> None:
        super().__init__()
        self.owner_count = len(owners)
        self.width = width
        self.event_slots = event_slots
        family_ids, layer_ids = _owner_coordinates(owners)
        self.register_buffer("family_ids", family_ids, persistent=False)
        self.register_buffer("layer_ids", layer_ids, persistent=False)
        self.owner_embedding = torch.nn.Embedding(self.owner_count, width)
        self.family_embedding = torch.nn.Embedding(len(TargetFamily), width)
        self.layer_embedding = torch.nn.Embedding(18, width)
        self.language_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, width),
        )
        self.scene_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(3 * width),
            torch.nn.Linear(3 * width, 2 * width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * width, width),
        )
        self.language_norm = torch.nn.LayerNorm(width)
        self.scene_norm = torch.nn.LayerNorm(width)
        self.process_norm = torch.nn.LayerNorm(width)

    def _owner_bias(self) -> torch.Tensor:
        owner_ids = torch.arange(self.owner_count, device=self.family_ids.device)
        return (
            self.owner_embedding(owner_ids)
            + self.family_embedding(self.family_ids)
            + self.layer_embedding(self.layer_ids)
        )

    @staticmethod
    def _groups(
        video_group_ids: torch.Tensor, group_count: int | None
    ) -> tuple[torch.Tensor, ...]:
        count = (
            int(video_group_ids.max().item()) + 1
            if group_count is None
            else int(group_count)
        )
        groups = tuple(video_group_ids == index for index in range(count))
        if count <= 0 or any(not bool(mask.any()) for mask in groups):
            raise ValueError("every ECP Program group must contain at least one video")
        return groups

    def forward(
        self,
        encoded: ECPVideoEncoderOutput,
        video_group_ids: torch.Tensor,
        *,
        group_count: int | None = None,
    ) -> ECPProgram:
        videos = encoded.process.shape[0]
        if (
            video_group_ids.shape != (videos,)
            or encoded.process.shape[1:3]
            != (self.event_slots, self.owner_count)
            or encoded.process.shape[-1] != self.width
            or encoded.language_summary.shape != (videos, self.width)
            or encoded.scene_transition.shape != (videos, 3 * self.width)
        ):
            raise ValueError("visible ECP Program tensors changed shape")

        bias = self._owner_bias()
        language = self.language_norm(
            self.language_projection(encoded.language_summary)[:, None] + bias
        )
        scene = self.scene_norm(
            self.scene_projection(encoded.scene_transition)[:, None] + bias
        )
        process = self.process_norm(encoded.process + bias[None, None])

        language_rows = []
        scene_rows = []
        process_rows = []
        presence_rows = []
        uncertainty_rows = []
        for mask in self._groups(video_group_ids, group_count):
            group_presence = encoded.presence[mask].float()
            weights = group_presence / group_presence.sum(0, keepdim=True).clamp_min(
                1e-6
            )
            mean = (weights[:, :, None, None] * process[mask]).sum(0)
            variance = (
                weights[:, :, None, None]
                * (
                    encoded.uncertainty[mask].float().square()
                    + (process[mask].float() - mean.float()).square()
                )
            ).sum(0)
            language_rows.append(language[mask].mean(0))
            scene_rows.append(scene[mask].mean(0))
            process_rows.append(mean)
            presence_rows.append(group_presence.mean(0))
            uncertainty_rows.append(variance.clamp_min(1e-4).sqrt())
        return ECPProgram(
            language=torch.stack(language_rows),
            scene=torch.stack(scene_rows),
            process=torch.stack(process_rows),
            presence=torch.stack(presence_rows),
            uncertainty=torch.stack(uncertainty_rows),
        )
