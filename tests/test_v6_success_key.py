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
    return torch.randn(24, 6, generator=torch.Generator().manual_seed(19))


def test_current_stable_keys_do_not_enter_provisional_plan_before_commit() -> None:
    bank = _bank()
    features = _features()
    stable = torch.zeros(24, dtype=torch.bool)
    stable[[0, 2]] = True
    plan = bank.persisted_plan()
    assert plan.features.shape == (0, 6)
    assert not bool(plan.persisted_before_mask.any())
    summary = bank.commit_first_stable_successes_(features, stable, plan)
    assert summary.current_stable_success_ordinals == (0, 2)
    assert summary.newly_stored_ordinals == (0, 2)
    assert summary.persisted_after_ordinals == (0, 2)


def test_first_stable_key_is_never_replaced_and_harmful_is_not_persisted() -> None:
    bank = _bank()
    first = _features()
    stable = torch.zeros(24, dtype=torch.bool)
    stable[0] = True
    bank.commit_first_stable_successes_(first, stable, bank.persisted_plan())
    stored = bank.features[0].clone()

    second = _features().roll(1, dims=1)
    plan = bank.persisted_plan()
    assert plan.features.shape == (1, 6)
    torch.testing.assert_close(plan.features[0], stored)
    no_stable = torch.zeros(24, dtype=torch.bool)
    summary = bank.commit_first_stable_successes_(second, no_stable, plan)
    assert summary.newly_stored_count == 0
    torch.testing.assert_close(bank.features[0], stored)
    assert not bool(bank.present[1])


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


def test_stale_persisted_plan_cannot_commit() -> None:
    bank = _bank()
    features = _features()
    stale = bank.persisted_plan()
    stable = torch.zeros(24, dtype=torch.bool)
    stable[0] = True
    bank.commit_first_stable_successes_(features, stable, stale)
    with pytest.raises(ExpertManifoldError, match="plan changed"):
        bank.commit_first_stable_successes_(features, stable, stale)
