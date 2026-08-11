"""Persistent stable-success condition keys for PCUG consolidation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ember.expert_manifold.contract import ExpertManifoldError


@dataclass(frozen=True)
class PersistedSuccessKeyPlan:
    """Snapshot used by the provisional blind solve and later bank commit."""

    features: torch.Tensor
    persisted_before_mask: torch.Tensor


@dataclass(frozen=True)
class SuccessKeyBankUpdateSummary:
    """Deterministic first-stable-success transition at a macro boundary."""

    current_stable_success_count: int
    current_stable_success_ordinals: tuple[int, ...]
    persisted_before_count: int
    persisted_before_ordinals: tuple[int, ...]
    newly_stored_count: int
    newly_stored_ordinals: tuple[int, ...]
    persisted_after_count: int
    persisted_after_ordinals: tuple[int, ...]


def _ordinals(mask: torch.Tensor) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in torch.nonzero(mask, as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )


class SuccessKeyAnchorBank:
    """Replicated first-stable-success key bank with one slot per train task."""

    TASK_COUNT = 24

    def __init__(
        self,
        task_global_ids: Sequence[int],
        *,
        feature_width: int,
        device: torch.device,
    ) -> None:
        ids = tuple(int(value) for value in task_global_ids)
        if (
            len(ids) != self.TASK_COUNT
            or len(set(ids)) != self.TASK_COUNT
            or min(ids) < 0
            or feature_width <= 0
        ):
            raise ExpertManifoldError("invalid PCUG success-key bank authority")
        self.features = torch.zeros(
            self.TASK_COUNT,
            feature_width,
            dtype=torch.float32,
            device=device,
        )
        self.present = torch.zeros(
            self.TASK_COUNT,
            dtype=torch.bool,
            device=device,
        )
        self.task_global_ids = torch.tensor(ids, dtype=torch.long, device=device)

    @property
    def feature_width(self) -> int:
        return int(self.features.shape[1])

    def _validate_macro_inputs(
        self,
        correct_features: torch.Tensor,
        stable_success_mask: torch.Tensor,
    ) -> None:
        if (
            correct_features.shape != self.features.shape
            or correct_features.dtype != torch.float32
            or correct_features.device != self.features.device
            or stable_success_mask.shape != (self.TASK_COUNT,)
            or stable_success_mask.dtype != torch.bool
            or stable_success_mask.device != self.features.device
            or not bool(torch.isfinite(correct_features).all())
            or bool((correct_features.square().sum(dim=1) <= 0).any())
        ):
            raise ExpertManifoldError("invalid PCUG macro success-key inputs")
        self.validate()

    def persisted_plan(self) -> PersistedSuccessKeyPlan:
        """Freeze the only equality rows allowed to affect provisional ``D0``."""

        self.validate()
        persisted = self.present.clone()
        return PersistedSuccessKeyPlan(
            features=self.features[persisted].contiguous(),
            persisted_before_mask=persisted,
        )

    @torch.no_grad()
    def commit_first_stable_successes_(
        self,
        correct_features: torch.Tensor,
        stable_success_mask: torch.Tensor,
        plan: PersistedSuccessKeyPlan,
    ) -> SuccessKeyBankUpdateSummary:
        """Persist stable-success keys; update-specific harmful keys stay ephemeral."""

        self._validate_macro_inputs(correct_features, stable_success_mask)
        if (
            plan.features.device != self.features.device
            or plan.features.dtype != torch.float32
            or plan.persisted_before_mask.shape != self.present.shape
            or plan.persisted_before_mask.dtype != torch.bool
            or plan.persisted_before_mask.device != self.features.device
            or not torch.equal(plan.persisted_before_mask, self.present)
            or plan.features.shape
            != (int(self.present.sum()), self.feature_width)
            or not torch.equal(plan.features, self.features[self.present])
        ):
            raise ExpertManifoldError("PCUG success-key plan changed before commit")
        newly_stored = stable_success_mask & ~self.present
        self.features[newly_stored] = correct_features[newly_stored]
        self.present.logical_or_(newly_stored)
        self.validate()
        return SuccessKeyBankUpdateSummary(
            current_stable_success_count=int(stable_success_mask.sum()),
            current_stable_success_ordinals=_ordinals(stable_success_mask),
            persisted_before_count=int(plan.persisted_before_mask.sum()),
            persisted_before_ordinals=_ordinals(plan.persisted_before_mask),
            newly_stored_count=int(newly_stored.sum()),
            newly_stored_ordinals=_ordinals(newly_stored),
            persisted_after_count=int(self.present.sum()),
            persisted_after_ordinals=_ordinals(self.present),
        )

    @torch.no_grad()
    def restore_(
        self,
        *,
        features: torch.Tensor,
        present: torch.Tensor,
        task_global_ids: torch.Tensor,
    ) -> None:
        if (
            features.shape != self.features.shape
            or features.dtype != torch.float32
            or present.shape != self.present.shape
            or present.dtype != torch.bool
            or task_global_ids.shape != self.task_global_ids.shape
            or task_global_ids.dtype != torch.long
            or not bool(torch.isfinite(features).all())
            or not torch.equal(
                task_global_ids.detach().cpu(), self.task_global_ids.detach().cpu()
            )
            or bool((features[~present] != 0).any())
            or bool((features[present].square().sum(dim=1) <= 0).any())
        ):
            raise ExpertManifoldError("invalid PCUG success-key checkpoint state")
        self.features.copy_(features.to(device=self.features.device))
        self.present.copy_(present.to(device=self.present.device))
        self.validate()

    def validate(self) -> None:
        if (
            self.features.shape != (self.TASK_COUNT, self.feature_width)
            or self.features.dtype != torch.float32
            or self.present.shape != (self.TASK_COUNT,)
            or self.present.dtype != torch.bool
            or self.task_global_ids.shape != (self.TASK_COUNT,)
            or self.task_global_ids.dtype != torch.long
            or not (
                self.features.device
                == self.present.device
                == self.task_global_ids.device
            )
            or not bool(torch.isfinite(self.features).all())
            or bool((self.features[~self.present] != 0).any())
            or bool((self.features[self.present].square().sum(dim=1) <= 0).any())
            or self.task_global_ids.unique().numel() != self.TASK_COUNT
        ):
            raise ExpertManifoldError("PCUG success-key bank became invalid")
