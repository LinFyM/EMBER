from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_contract import load_v6_prior_config
from ember.expert_manifold.v6_prior_policy_batch import (
    LOGICAL_POLICY_BATCH_SIZE,
    POSITIVE_POLICY_RANDOMNESS,
    policy_rng_seed_for_logical_batch,
)


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/pi05_v6_counterfactual_null_condition_kernel_program_residual_v1.json"
)


def _batch(count: int = LOGICAL_POLICY_BATCH_SIZE) -> dict[str, torch.Tensor]:
    return {
        "demo_index": torch.arange(count, dtype=torch.long) % 50,
        "frame_index": torch.arange(count, dtype=torch.long) * 3,
    }


def test_residual_policy_batch_is_fixed_b20_over_two_physical_b10_forwards() -> None:
    config = load_v6_prior_config(CONFIG)
    assert config["data"]["action_queries_per_task"] == 20
    assert config["optimization"]["functional_policy_microbatch_size"] == 10
    assert config["optimization"]["physical_policy_forwards_per_task"] == 2
    assert config["optimization"]["extra_negative_policy_forwards_per_task"] == 0
    assert config["writer"]["effective_runtime_activation_checkpointing"] is False
    assert config["objective"]["positive_policy_randomness"] == POSITIVE_POLICY_RANDOMNESS


def test_policy_rng_is_logical_batch_keyed_and_requires_exact_b20() -> None:
    config = load_v6_prior_config(CONFIG)
    batch = _batch()
    seed = policy_rng_seed_for_logical_batch(
        config,
        batch,
        task_id=7,
        task_visit=11,
    )
    assert seed == policy_rng_seed_for_logical_batch(
        config,
        {name: value.clone() for name, value in batch.items()},
        task_id=7,
        task_visit=11,
    )
    assert seed != policy_rng_seed_for_logical_batch(
        config,
        batch,
        task_id=7,
        task_visit=12,
    )
    with pytest.raises(ExpertManifoldError, match="randomness changed"):
        policy_rng_seed_for_logical_batch(
            config,
            _batch(19),
            task_id=7,
            task_visit=11,
        )
