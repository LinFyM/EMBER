"""Matched task-arm assignments, seeds, and selection for task-local LoRA RL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.source_base_checkpoint import read_json, sha256_file
from ember.writer.model import WriterModelError


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, order=True)
class TaskArm:
    task_id: int
    arm: str

    @property
    def key(self) -> str:
        return f"task_{self.task_id:03d}_{self.arm}"


def load_task_local_rl_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "ember_task_local_lora_rl_v1":
        raise WriterModelError("unsupported task-local LoRA RL config")
    protocol = config.get("protocol", {})
    for key in (
        "manifest",
        "evaluation_config",
        "source_selection",
        "writer_config",
        "lora_contract",
    ):
        authority = REPO_ROOT / str(protocol.get(key, ""))
        if not authority.is_file() or sha256_file(authority) != protocol.get(
            f"{key}_sha256"
        ):
            raise WriterModelError(f"task-local RL authority changed: {key}")
    manifest = read_json(REPO_ROOT / protocol["manifest"])
    if (
        manifest.get("protocol_references", {}).get("split_sha256")
        != protocol.get("split_sha256")
    ):
        raise WriterModelError("task-local RL manifest and split disagree")
    role = config.get("role", {})
    task_ids = tuple(int(value) for value in role.get("task_ids", []))
    split_by_id = {
        int(record["task_index"]): record["split"] for record in manifest["tasks"]
    }
    if (
        len(task_ids) != 10
        or len(set(task_ids)) != 10
        or any(split_by_id.get(task_id) != "validation" for task_id in task_ids)
        or tuple(config.get("arms", [])) != ("identity", "writer")
    ):
        raise WriterModelError("task-local RL validation role or arms changed")
    algorithm = config.get("algorithm", {})
    environment = config.get("environment", {})
    if (
        algorithm.get("name")
        != "on_policy_success_weighted_flow_regression"
        or algorithm.get("reward") != "binary_task_success"
        or int(algorithm.get("rollouts_per_update", 0)) <= 0
        or int(algorithm.get("replay_epochs", 0)) != 1
        or algorithm.get("teacher_actions") is not False
        or environment.get("official_random_reset") is not True
        or environment.get("fixed_pruned_init_states") is not False
        or int(environment.get("dummy_settling_steps", -1)) != 10
        or int(environment.get("max_horizon", -1)) != 400
        or environment.get("terminate_on_success") is not True
        or int(config.get("policy", {}).get("action_execution_horizon", -1))
        != 50
        or int(config.get("parallel", {}).get("world_size", 0)) != 8
        or int(config.get("parallel", {}).get("policy_processes_per_gpu", 0))
        != 1
    ):
        raise WriterModelError("task-local RL information or topology wall changed")
    if (
        int(config.get("profile", {}).get("task_count", -1)) != 4
        or config.get("checkpoint_selection", {}).get("metric")
        != "preceding_segment_adaptation_success_rate"
        or config.get("checkpoint_selection", {}).get(
            "fresh_fixed_state_evaluation_is_separate"
        )
        is not True
    ):
        raise WriterModelError("task-local RL profile or selection rule changed")
    return config


def task_arms(config: Mapping[str, Any]) -> tuple[TaskArm, ...]:
    return tuple(
        TaskArm(int(task_id), str(arm))
        for task_id in config["role"]["task_ids"]
        for arm in config["arms"]
    )


def rank_assignments(
    units: Sequence[TaskArm], world_size: int
) -> tuple[tuple[TaskArm, ...], ...]:
    if world_size <= 0 or not units or len(set(units)) != len(units):
        raise WriterModelError("invalid task-local RL assignment")
    return tuple(tuple(units[rank::world_size]) for rank in range(world_size))


def rollout_seed(base: int, task_id: int, update: int, rollout: int) -> int:
    if min(base, task_id, update, rollout) < 0 or rollout >= 100:
        raise WriterModelError("invalid matched task-local rollout seed")
    return base + task_id * 1_000_000 + update * 100 + rollout


def optimizer_seed(base: int, task_id: int, update: int) -> int:
    if min(base, task_id, update) < 0:
        raise WriterModelError("invalid task-local optimizer seed")
    return base + task_id * 1_000_000 + update


def select_adaptation_checkpoint(
    candidates: Sequence[Mapping[str, int]],
) -> Mapping[str, int]:
    """Maximize exact segment success fraction, breaking ties by early cursor."""

    if not candidates:
        raise WriterModelError("task-local checkpoint selection is empty")
    normalized = []
    for candidate in candidates:
        successes = int(candidate.get("segment_successes", -1))
        rollouts = int(candidate.get("segment_rollouts", 0))
        cursor = int(candidate.get("interaction_cursor", -1))
        if successes < 0 or not successes <= rollouts or cursor <= 0:
            raise WriterModelError("invalid task-local checkpoint selection row")
        normalized.append((candidate, successes, rollouts, cursor))

    def better(left: tuple[Any, int, int, int], right: tuple[Any, int, int, int]) -> bool:
        _, left_successes, left_rollouts, left_cursor = left
        _, right_successes, right_rollouts, right_cursor = right
        cross_left = left_successes * right_rollouts
        cross_right = right_successes * left_rollouts
        return cross_left > cross_right or (
            cross_left == cross_right and left_cursor < right_cursor
        )

    selected = normalized[0]
    for candidate in normalized[1:]:
        if better(candidate, selected):
            selected = candidate
    return selected[0]
