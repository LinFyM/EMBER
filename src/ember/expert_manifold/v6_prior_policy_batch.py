"""Logical policy-batch and physical microbatch contract for v6-prior."""

from __future__ import annotations

import math
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
FUNCTIONAL_POLICY_MICROBATCH_CANDIDATES = frozenset((10, 16))
POLICY_RUNTIME_SELECTION_KEYS = (
    "logical_policy_batch_size",
    "functional_policy_microbatch_size",
    "physical_policy_forwards_per_task",
    "policy_gradient_checkpointing",
    "writer_activation_checkpointing",
)
POSITIVE_POLICY_RANDOMNESS = {
    "scope": "one_independent_flow_noise_and_time_per_action_query",
    "seed_scheme": TASK_LOGICAL_BATCH_POLICY_RNG_SCHEME,
    "flow_time_sampling_scheme": INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
    "flow_noise_sampling_scheme": INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
}


def positive_policy_randomness_matches(value: Any) -> bool:
    return value == POSITIVE_POLICY_RANDOMNESS


def optimization_policy_microbatch_matches(value: Mapping[str, Any]) -> bool:
    try:
        return int(value.get("functional_policy_microbatch_size", -1)) in (
            FUNCTIONAL_POLICY_MICROBATCH_CANDIDATES
        )
    except (TypeError, ValueError):
        return False


def policy_batch_config_matches(config: Mapping[str, Any]) -> bool:
    try:
        optimization = config["optimization"]
        gradient = config["gradient_profile"]
        microbatch = int(optimization["functional_policy_microbatch_size"])
        valid = (
            microbatch in FUNCTIONAL_POLICY_MICROBATCH_CANDIDATES
            and int(config["data"]["action_queries_per_task"])
            == LOGICAL_POLICY_BATCH_SIZE
            and int(gradient["logical_policy_batch_size"]) == LOGICAL_POLICY_BATCH_SIZE
            and int(gradient["functional_policy_microbatch_size"]) == microbatch
            and int(gradient["physical_policy_forwards_per_task"])
            == math.ceil(LOGICAL_POLICY_BATCH_SIZE / microbatch)
        )
        for section_name in ("gradient_profile", "profile_run"):
            evidence = config[section_name].get("artifact_evidence")
            valid = valid and (
                evidence is None
                or int(
                    evidence["runtime_selection"]["functional_policy_microbatch_size"]
                )
                == microbatch
            )
        return valid
    except (KeyError, TypeError, ValueError):
        return False


def policy_runtime_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    logical_batch = int(config["data"]["action_queries_per_task"])
    microbatch = int(config["optimization"]["functional_policy_microbatch_size"])
    return {
        "logical_policy_batch_size": logical_batch,
        "functional_policy_microbatch_size": microbatch,
        "physical_policy_forwards_per_task": math.ceil(logical_batch / microbatch),
        "policy_gradient_checkpointing": False,
        "writer_activation_checkpointing": bool(
            config["writer"]["activation_checkpointing"]
        ),
    }


def policy_runtime_selection_matches(value: Mapping[str, Any]) -> bool:
    try:
        logical_batch = int(value["logical_policy_batch_size"])
        microbatch = int(value["functional_policy_microbatch_size"])
        return (
            logical_batch == LOGICAL_POLICY_BATCH_SIZE
            and microbatch in FUNCTIONAL_POLICY_MICROBATCH_CANDIDATES
            and int(value["physical_policy_forwards_per_task"])
            == math.ceil(logical_batch / microbatch)
            and value.get("policy_gradient_checkpointing") is False
            and value.get("writer_activation_checkpointing") is True
        )
    except (KeyError, TypeError, ValueError):
        return False


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
