"""Logical policy-batch and physical microbatch contract for v6-prior."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from ember.expert_manifold.contract import ExpertManifoldError
from ember.writer.functional import (
    INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
    INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    TASK_LOGICAL_BATCH_POLICY_RNG_SCHEME,
    task_logical_batch_policy_rng_seed,
)


LOGICAL_POLICY_BATCH_SIZE = 20
POSITIVE_POLICY_RANDOMNESS = {
    "scope": "one_independent_flow_noise_and_time_per_action_query",
    "seed_scheme": TASK_LOGICAL_BATCH_POLICY_RNG_SCHEME,
    "flow_time_sampling_scheme": INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
    "flow_noise_sampling_scheme": INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
}


def positive_policy_randomness_matches(value: Any) -> bool:
    return value == POSITIVE_POLICY_RANDOMNESS


def policy_rng_seed_for_logical_batch(
    config: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    task_id: int,
    task_visit: int,
) -> int:
    randomness = config["objective"]["positive_policy_randomness"]
    demo_indices = batch.get("demo_index")
    frame_indices = batch.get("frame_index")
    if (
        not positive_policy_randomness_matches(randomness)
        or not isinstance(demo_indices, torch.Tensor)
        or not isinstance(frame_indices, torch.Tensor)
        or demo_indices.ndim != 1
        or frame_indices.shape != demo_indices.shape
        or demo_indices.numel() != LOGICAL_POLICY_BATCH_SIZE
    ):
        raise ExpertManifoldError("v6-prior action query randomness changed")
    return task_logical_batch_policy_rng_seed(
        optimization_seed=int(config["optimization"]["seed"]),
        task_id=task_id,
        task_visit=task_visit,
        demo_indices=demo_indices.detach().cpu().tolist(),
        frame_indices=frame_indices.detach().cpu().tolist(),
    )
