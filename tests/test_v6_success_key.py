from __future__ import annotations

import pytest
import torch

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_success_key import SuccessKeyAnchorBank


def _bank() -> SuccessKeyAnchorBank:
    return SuccessKeyAnchorBank(
        range(100, 124), feature_width=6, device=torch.device("cpu")
    )


def _features() -> torch.Tensor:
    generator = torch.Generator().manual_seed(19)
    return torch.randn(24, 6, generator=generator)


def test_current_all_success_keys_constrain_before_first_commit() -> None:
    bank = _bank()
    features = _features()
    success = torch.tensor([4, 3, 4, *([0] * 21)], dtype=torch.long)
    plan = bank.constraint_plan(features, success)
    torch.testing.assert_close(plan.features, features[[0, 2]])
    assert plan.current_all_success_mask.nonzero().flatten().tolist() == [0, 2]
    assert not bool(plan.persisted_before_mask.any())
    summary = bank.commit_first_successes_(features, success, plan)
    assert summary.constraint_row_count == 2
    assert summary.newly_stored_ordinals == (0, 2)
    assert summary.persisted_after_ordinals == (0, 2)


def test_first_success_key_is_never_replaced_but_current_key_is_constrained() -> None:
    bank = _bank()
    first = _features()
    success = torch.tensor([4, *([0] * 23)], dtype=torch.long)
    first_plan = bank.constraint_plan(first, success)
    bank.commit_first_successes_(first, success, first_plan)
    stored = bank.features[0].clone()

    second = _features().roll(1, dims=1)
    second_plan = bank.constraint_plan(second, success)
    assert second_plan.features.shape == (2, 6)
    torch.testing.assert_close(second_plan.features[0], stored)
    torch.testing.assert_close(second_plan.features[1], second[0])
    summary = bank.commit_first_successes_(second, success, second_plan)
    assert summary.newly_stored_count == 0
    torch.testing.assert_close(bank.features[0], stored)


def test_bank_restore_preserves_slot_authority() -> None:
    bank = _bank()
    features = _features()
    present = torch.zeros(24, dtype=torch.bool)
    present[[1, 7]] = True
    checkpoint_features = torch.zeros_like(features)
    checkpoint_features[present] = features[present]
    bank.restore_(
        features=checkpoint_features,
        present=present,
        task_global_ids=torch.arange(100, 124),
    )
    assert bank.present.nonzero().flatten().tolist() == [1, 7]
    with pytest.raises(ExpertManifoldError):
        bank.restore_(
            features=checkpoint_features,
            present=present,
            task_global_ids=torch.arange(101, 125),
        )


def test_stale_constraint_plan_cannot_commit() -> None:
    bank = _bank()
    features = _features()
    success = torch.tensor([4, *([0] * 23)], dtype=torch.long)
    plan = bank.constraint_plan(features, success)
    other = success.roll(1)
    with pytest.raises(ExpertManifoldError, match="plan changed"):
        bank.commit_first_successes_(features, other, plan)
