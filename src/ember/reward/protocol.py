"""Order-independent task, video, environment, and PI05 flow-noise schedules."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


SUITE_HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


class RewardProtocolError(RuntimeError):
    """Raised when a reward-adaptation authority or cursor changes."""


@dataclass(frozen=True, order=True)
class RewardTask:
    suite: str
    task_id: int
    global_task_id: int
    split_role: str
    language: str
    problem_folder: str
    bddl_file: str
    bddl_bytes: int
    # Historical ledgers carried a BDDL content hash.  New reward-credit runs
    # deliberately bind only the sealed path and byte count: recomputing a
    # cryptographic digest in every process does not improve the scientific
    # contract and is forbidden by the current throughput policy.
    bddl_sha256: str | None
    horizon: int

    def __post_init__(self) -> None:
        if (
            self.suite not in SUITE_HORIZONS
            or not 0 <= self.task_id < 10
            or not 0 <= self.global_task_id < 40
            or self.split_role not in {"train", "validation", "test"}
            or not self.language.strip()
            or not self.problem_folder
            or not self.bddl_file.endswith(".bddl")
            or self.bddl_bytes <= 0
            or self.horizon != SUITE_HORIZONS[self.suite]
        ):
            raise RewardProtocolError("invalid PI05 reward task authority")
        if self.bddl_sha256 is not None and len(self.bddl_sha256) != 64:
            raise RewardProtocolError("reward task BDDL hash is not SHA-256")


_UINT64_MASK = (1 << 64) - 1
_SEED_TAGS = {
    "environment": 0x243F6A8885A308D3,
    "policy_noise": 0x13198A2E03707344,
    "update": 0xA4093822299F31D0,
    "task_video": 0x082EFA98EC4E6C89,
}
_SUITE_IDS = {
    "libero_spatial": 0,
    "libero_object": 1,
    "libero_goal": 2,
    "libero_10": 3,
}


def _splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def _component_id(value: int | str) -> int:
    if isinstance(value, int):
        if value < 0:
            raise RewardProtocolError("invalid reward seed request")
        return value
    try:
        return _SUITE_IDS[value]
    except (KeyError, TypeError):
        raise RewardProtocolError("invalid reward seed component") from None


def mixed_seed(tag: str, *components: int | str) -> int:
    """Mix a small sealed integer identity without content hashing.

    The schedule needs deterministic, order-independent seeds, not collision
    resistance.  SplitMix64 is substantially cheaper than serializing and
    hashing every policy replan while retaining the required statistical seed
    quality.
    """

    if tag not in _SEED_TAGS:
        raise RewardProtocolError("invalid reward seed request")
    state = _SEED_TAGS[tag]
    for ordinal, component in enumerate(components):
        identity = _component_id(component)
        state = _splitmix64(state ^ _splitmix64(identity + ordinal * 0x9E3779B9))
    return state & ((1 << 63) - 1)


def _historical_seed(tag: str, *components: Any) -> int:
    """Preserve already-published reward schedules for retired callers.

    Active Reward-Credit code uses the explicitly versioned integer-mixer APIs
    below.  Keeping this function avoids silently changing old artifacts while
    ensuring no cryptographic hashing occurs in the new hot path.
    """

    if not tag or any(isinstance(value, int) and value < 0 for value in components):
        raise RewardProtocolError("invalid reward seed request")
    payload = json.dumps(
        [tag, *components],
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def environment_seed(
    root_seed: int,
    suite: str,
    task_id: int,
    adaptation_seed: int,
    rollout_cursor: int,
) -> int:
    """Seed one official BDDL reset; arm/rank/order are deliberately absent."""

    # LIBERO forwards this value to NumPy's legacy RandomState.seed, whose
    # accepted domain is uint32 rather than PyTorch's wider seed domain.
    return _historical_seed(
        "ember_pi05_reward_environment_v1",
        root_seed,
        suite,
        task_id,
        adaptation_seed,
        rollout_cursor,
    ) % (1 << 32)


def policy_noise_seed(
    root_seed: int,
    suite: str,
    task_id: int,
    adaptation_seed: int,
    rollout_cursor: int,
    replan_index: int,
) -> int:
    """Seed explicit PI05 flow noise independently of scheduling and adapter arm."""

    return _historical_seed(
        "ember_pi05_reward_policy_noise_v1",
        root_seed,
        suite,
        task_id,
        adaptation_seed,
        rollout_cursor,
        replan_index,
    )


def update_seed(
    root_seed: int,
    suite: str,
    task_id: int,
    adaptation_seed: int,
    optimizer_update_cursor: int,
) -> int:
    return _historical_seed(
        "ember_pi05_reward_update_v1",
        root_seed,
        suite,
        task_id,
        adaptation_seed,
        optimizer_update_cursor,
    )


def task_local_video_demo(
    root_seed: int,
    global_task_id: int,
    adaptation_seed: int,
    *,
    demo_count: int = 50,
) -> int:
    """Choose the one video shared by AS/RL Writer arms for a whole adaptation."""

    if demo_count <= 0:
        raise RewardProtocolError("task-local video schedule has no demonstrations")
    return (
        _historical_seed(
            "ember_pi05_task_local_video_v1",
            root_seed,
            global_task_id,
            adaptation_seed,
        )
        % demo_count
    )


def reward_credit_environment_seed(
    root_seed: int,
    suite: str,
    task_id: int,
    adaptation_seed: int,
    rollout_cursor: int,
) -> int:
    """Hash-free environment schedule owned only by the new reward schema."""

    return mixed_seed(
        "environment",
        root_seed,
        suite,
        task_id,
        adaptation_seed,
        rollout_cursor,
    ) % (1 << 32)


def reward_credit_policy_noise_seed(
    root_seed: int,
    suite: str,
    task_id: int,
    adaptation_seed: int,
    rollout_cursor: int,
    replan_index: int,
) -> int:
    """Hash-free PI05 flow-noise schedule owned by Reward-Credit v1."""

    return mixed_seed(
        "policy_noise",
        root_seed,
        suite,
        task_id,
        adaptation_seed,
        rollout_cursor,
        replan_index,
    )


def reward_preference_flow_seed(
    root_seed: int,
    *,
    cycle: int,
    global_task_id: int,
    mc_index: int,
) -> int:
    """Key one reward-CFM sample by scientific identity, never rank or order."""

    return mixed_seed(
        "update",
        root_seed,
        cycle,
        global_task_id,
        mc_index,
    )
