"""Immutable random-reset interaction ledgers and exact cursor accounting."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ember.pi05_source_checkpoint import (
    canonical_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.reward.protocol import RewardProtocolError


_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class InteractionCursors:
    rollout: int
    environment_actions: int
    optimizer_updates: int

    def __post_init__(self) -> None:
        if min(self.rollout, self.environment_actions, self.optimizer_updates) < 0:
            raise RewardProtocolError("reward interaction cursor is negative")
        if self.optimizer_updates > self.rollout:
            raise RewardProtocolError("reward optimizer cursor exceeds rollout cursor")

    def to_dict(self) -> dict[str, int]:
        return {
            "rollout_cursor": self.rollout,
            "environment_action_cursor": self.environment_actions,
            "optimizer_update_cursor": self.optimizer_updates,
        }


def validate_random_reset_row(row: Mapping[str, Any]) -> None:
    seeds = row.get("policy_noise_seeds")
    initial = str(row.get("initial_observation_sha256", ""))
    valid = row.get("valid_action_steps")
    retained = bool(row.get("failure_replay_retained", False))
    if (
        row.get("official_random_reset") is not True
        or row.get("fixed_init_state_id", "sentinel") is not None
        or row.get("set_init_state_called") is not False
        or int(row.get("dummy_settling_steps", -1)) != 10
        or int(row.get("rollout_cursor", -1)) < 0
        or int(row.get("env_seed", -1)) < 0
        or int(row.get("policy_seed_root", -1)) < 0
        or not isinstance(seeds, list)
        or any(int(seed) < 0 for seed in seeds)
        or _SHA256.fullmatch(initial) is None
        or not isinstance(valid, list)
        or any(not 0 < int(value) <= 5 for value in valid)
        or int(row.get("action_chunk_count", -1)) != len(valid)
        or (
            (bool(row.get("success")) or retained)
            and len(valid) != len(seeds)
        )
        or (not bool(row.get("success")) and not retained and valid)
        or int(row.get("steps", -1)) < 0
    ):
        raise RewardProtocolError("random-reset reward ledger row changed")


def _ledger_path(root: Path, unit_key: str, rollout_cursor: int) -> Path:
    if not unit_key or "/" in unit_key or rollout_cursor < 0:
        raise RewardProtocolError("invalid reward ledger identity")
    return root / "rollouts" / unit_key / f"rollout_{rollout_cursor:08d}.json"


def write_rollout_once(
    root: Path, unit_key: str, row: Mapping[str, Any]
) -> Path:
    validate_random_reset_row(row)
    cursor = int(row["rollout_cursor"])
    path = _ledger_path(root, unit_key, cursor)
    payload = dict(row)
    if path.is_file():
        if canonical_hash(read_json(path)) != canonical_hash(payload):
            raise RewardProtocolError(
                f"replayed reward interaction changed: {unit_key}/{cursor}"
            )
        return path
    write_json_atomic(path, payload)
    return path


def ledger_prefix_summary(
    root: Path, unit_key: str, rollout_cursor: int
) -> dict[str, Any]:
    if rollout_cursor < 0:
        raise RewardProtocolError("invalid reward ledger prefix cursor")
    digest = hashlib.sha256()
    successes = 0
    environment_actions = 0
    reward_sum = 0.0
    for cursor in range(rollout_cursor):
        path = _ledger_path(root, unit_key, cursor)
        if not path.is_file():
            raise RewardProtocolError(
                f"reward ledger prefix has a gap: {unit_key}/{cursor}"
            )
        row = read_json(path)
        validate_random_reset_row(row)
        if int(row["rollout_cursor"]) != cursor:
            raise RewardProtocolError("reward ledger cursor differs from its path")
        file_sha = sha256_file(path)
        digest.update(bytes.fromhex(file_sha))
        successes += int(bool(row["success"]))
        environment_actions += int(row["steps"])
        reward_sum += float(row["reward_sum"])
    return {
        "rollout_cursor": rollout_cursor,
        "environment_action_cursor": environment_actions,
        "successes": successes,
        "reward_sum": reward_sum,
        "ledger_prefix_sha256": digest.hexdigest(),
    }
