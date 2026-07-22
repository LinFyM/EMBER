"""Order-independent task, video, environment, and PI05 flow-noise schedules."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence


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
    bddl_sha256: str
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
        if len(self.bddl_sha256) != 64:
            raise RewardProtocolError("reward task BDDL hash is not SHA-256")


def _seed(tag: str, *components: Any) -> int:
    if not tag or any(isinstance(value, int) and value < 0 for value in components):
        raise RewardProtocolError("invalid reward seed request")
    payload = json.dumps(
        [tag, *components], sort_keys=False, separators=(",", ":"), ensure_ascii=False
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
    return _seed(
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

    return _seed(
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
    return _seed(
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
    return _seed(
        "ember_pi05_task_local_video_v1",
        root_seed,
        global_task_id,
        adaptation_seed,
    ) % demo_count


def rank_strided_assignments(
    values: Sequence[Any], world_size: int
) -> tuple[tuple[Any, ...], ...]:
    if world_size <= 0 or not values or len(set(values)) != len(values):
        raise RewardProtocolError("invalid reward rank assignment")
    return tuple(tuple(values[rank::world_size]) for rank in range(world_size))
