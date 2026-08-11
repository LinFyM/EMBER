"""Outcome-only success-key anchors for SKNC Program consolidation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ember.expert_manifold.contract import ExpertManifoldError


@dataclass(frozen=True)
class SuccessKeyConstraintPlan:
    """One macro's current and persisted equality-constraint rows."""

    features: torch.Tensor
    current_all_success_mask: torch.Tensor
    persisted_before_mask: torch.Tensor


@dataclass(frozen=True)
class SuccessKeyBankUpdateSummary:
    """Deterministic first-success bank transition at one macro boundary."""

    current_all_success_count: int
    current_all_success_ordinals: tuple[int, ...]
    persisted_before_count: int
    persisted_before_ordinals: tuple[int, ...]
    constraint_row_count: int
    newly_stored_count: int
    newly_stored_ordinals: tuple[int, ...]
    persisted_after_count: int
    persisted_after_ordinals: tuple[int, ...]


class SuccessKeyAnchorBank:
    """Replicated, training-only first-4/4 key bank with one slot per task."""

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
            raise ExpertManifoldError("invalid SKNC success-key bank authority")
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

    def _validated_macro_inputs(
        self,
        correct_features: torch.Tensor,
        success_counts: torch.Tensor,
    ) -> torch.Tensor:
        if (
            correct_features.shape != self.features.shape
            or correct_features.dtype != torch.float32
            or correct_features.device != self.features.device
            or success_counts.shape != (self.TASK_COUNT,)
            or success_counts.device != self.features.device
            or success_counts.dtype not in {torch.int32, torch.int64}
            or not bool(torch.isfinite(correct_features).all())
            or bool((success_counts < 0).any())
            or bool((success_counts > 4).any())
            or bool((correct_features.square().sum(dim=1) <= 0).any())
        ):
            raise ExpertManifoldError("invalid SKNC macro success-key inputs")
        self.validate()
        return success_counts == 4

    def constraint_plan(
        self,
        correct_features: torch.Tensor,
        success_counts: torch.Tensor,
    ) -> SuccessKeyConstraintPlan:
        """Build persisted rows first, then current 4/4 rows in task order."""

        current = self._validated_macro_inputs(correct_features, success_counts)
        persisted = self.present.clone()
        rows = torch.cat(
            (self.features[persisted], correct_features[current]),
            dim=0,
        ).contiguous()
        return SuccessKeyConstraintPlan(
            features=rows,
            current_all_success_mask=current,
            persisted_before_mask=persisted,
        )

    @torch.no_grad()
    def commit_first_successes_(
        self,
        correct_features: torch.Tensor,
        success_counts: torch.Tensor,
        plan: SuccessKeyConstraintPlan,
    ) -> SuccessKeyBankUpdateSummary:
        """Persist only a task's first deterministic all-success key."""

        current = self._validated_macro_inputs(correct_features, success_counts)
        if (
            plan.features.device != self.features.device
            or plan.features.dtype != torch.float32
            or plan.current_all_success_mask.dtype != torch.bool
            or plan.persisted_before_mask.dtype != torch.bool
            or plan.current_all_success_mask.device != self.features.device
            or plan.persisted_before_mask.device != self.features.device
            or not torch.equal(plan.current_all_success_mask, current)
            or not torch.equal(plan.persisted_before_mask, self.present)
            or plan.features.shape
            != (int(self.present.sum() + current.sum()), self.feature_width)
        ):
            raise ExpertManifoldError("SKNC success-key plan changed before commit")
        newly_stored = current & ~self.present
        self.features[newly_stored] = correct_features[newly_stored]
        self.present.logical_or_(newly_stored)
        self.validate()

        def ordinals(mask: torch.Tensor) -> tuple[int, ...]:
            return tuple(
                int(value)
                for value in torch.nonzero(mask, as_tuple=False)
                .flatten()
                .detach()
                .cpu()
                .tolist()
            )

        return SuccessKeyBankUpdateSummary(
            current_all_success_count=int(current.sum()),
            current_all_success_ordinals=ordinals(current),
            persisted_before_count=int(plan.persisted_before_mask.sum()),
            persisted_before_ordinals=ordinals(plan.persisted_before_mask),
            constraint_row_count=int(plan.features.shape[0]),
            newly_stored_count=int(newly_stored.sum()),
            newly_stored_ordinals=ordinals(newly_stored),
            persisted_after_count=int(self.present.sum()),
            persisted_after_ordinals=ordinals(self.present),
        )

    @torch.no_grad()
    def restore_(
        self,
        *,
        features: torch.Tensor,
        present: torch.Tensor,
        task_global_ids: torch.Tensor,
    ) -> None:
        """Restore exact-resume state after validating task-slot ownership."""

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
            raise ExpertManifoldError("invalid SKNC success-key checkpoint state")
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
            raise ExpertManifoldError("SKNC success-key bank became invalid")
