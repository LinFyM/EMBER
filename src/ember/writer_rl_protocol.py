"""Sealed source-task schedule and seed rules for Writer-only RL."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ember.source_base_checkpoint import read_json, sha256_file
from ember.writer.model import WriterModelError


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_writer_rl_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "ember_writer_only_rl_v1":
        raise WriterModelError("unsupported Writer-only RL config")
    protocol = config.get("protocol", {})
    for key in ("manifest", "writer_config", "lora_contract"):
        authority = REPO_ROOT / str(protocol.get(key, ""))
        if not authority.is_file() or sha256_file(authority) != protocol.get(
            f"{key}_sha256"
        ):
            raise WriterModelError(f"Writer-only RL authority changed: {key}")
    manifest = read_json(REPO_ROOT / protocol["manifest"])
    if (
        manifest.get("protocol_references", {}).get("split_sha256")
        != protocol.get("split_sha256")
    ):
        raise WriterModelError("Writer-only RL manifest and split disagree")
    algorithm = config.get("algorithm", {})
    environment = config.get("environment", {})
    if (
        algorithm.get("name")
        != "on_policy_success_weighted_flow_regression"
        or algorithm.get("reward") != "binary_task_success"
        or int(algorithm.get("rollouts_per_task_cycle", 0)) <= 0
        or int(algorithm.get("replay_epochs", 0)) != 1
        or algorithm.get("teacher_actions") is not False
        or environment.get("official_random_reset") is not True
        or environment.get("fixed_pruned_init_states") is not False
    ):
        raise WriterModelError("Writer-only RL information wall changed")
    if (
        int(config.get("parallel", {}).get("world_size", 0)) != 8
        or int(config.get("parallel", {}).get("policy_processes_per_gpu", 0))
        != 1
    ):
        raise WriterModelError("Writer-only RL requires eight symmetric ranks")
    return config


def source_task_ids(config: dict[str, Any]) -> tuple[int, ...]:
    manifest = read_json(REPO_ROOT / config["protocol"]["manifest"])
    task_ids = tuple(
        int(record["task_index"])
        for record in manifest.get("tasks", [])
        if record.get("split") == "train"
    )
    if len(task_ids) != 70 or len(set(task_ids)) != 70:
        raise WriterModelError("Writer-only RL requires exactly 70 source tasks")
    return task_ids


def rank_task_assignments(
    task_ids: Sequence[int], world_size: int
) -> tuple[tuple[int, ...], ...]:
    if world_size <= 0 or not task_ids or len(set(task_ids)) != len(task_ids):
        raise WriterModelError("invalid Writer-only RL task assignment request")
    return tuple(tuple(task_ids[rank::world_size]) for rank in range(world_size))


def updates_per_cycle(task_ids: Sequence[int], world_size: int) -> int:
    return max(map(len, rank_task_assignments(task_ids, world_size)))


def task_for_update(
    task_ids: Sequence[int], world_size: int, rank: int, update: int
) -> tuple[int, int, int] | None:
    """Return (task_id, cycle, slot), or None for a no-rollout padding slot."""

    if not 0 <= rank < world_size or update < 0:
        raise WriterModelError("invalid Writer-only RL schedule cursor")
    assignments = rank_task_assignments(task_ids, world_size)
    slots = max(map(len, assignments))
    cycle, slot = divmod(update, slots)
    if slot >= len(assignments[rank]):
        return None
    return assignments[rank][slot], cycle, slot


def environment_seed(base: int, cycle: int, task_id: int, rollout: int) -> int:
    if min(base, cycle, task_id, rollout) < 0 or rollout >= 100:
        raise WriterModelError("invalid Writer-only RL environment seed request")
    return base + cycle * 100_000 + task_id * 100 + rollout


def policy_seed(base: int, cycle: int, task_id: int, rollout: int) -> int:
    if min(base, cycle, task_id, rollout) < 0 or rollout >= 100:
        raise WriterModelError("invalid Writer-only RL policy seed request")
    return base + cycle * 100_000 + task_id * 100 + rollout


def update_seed(base: int, update: int, rank: int) -> int:
    if min(base, update, rank) < 0 or rank >= 100:
        raise WriterModelError("invalid Writer-only RL update seed request")
    return base + update * 100 + rank


def schedule_summary(
    task_ids: Sequence[int],
    world_size: int,
    next_update: int,
    rollouts_per_task: int,
) -> dict[str, Any]:
    if next_update < 0 or rollouts_per_task <= 0:
        raise WriterModelError("invalid Writer-only RL coverage request")
    counts = {int(task_id): 0 for task_id in task_ids}
    for update in range(next_update):
        for rank in range(world_size):
            scheduled = task_for_update(task_ids, world_size, rank, update)
            if scheduled is not None:
                counts[scheduled[0]] += rollouts_per_task
    slots = updates_per_cycle(task_ids, world_size)
    return {
        "next_update": next_update,
        "completed_full_task_cycles": next_update // slots,
        "cycle_slot_cursor": next_update % slots,
        "declared_task_count": len(counts),
        "tasks_with_interactions": sum(value > 0 for value in counts.values()),
        "min_rollouts_per_task": min(counts.values()),
        "max_rollouts_per_task": max(counts.values()),
        "total_rollouts": sum(counts.values()),
    }


def rank_rollout_count(
    task_ids: Sequence[int],
    world_size: int,
    rank: int,
    next_update: int,
    rollouts_per_task: int,
) -> int:
    if next_update < 0 or rollouts_per_task <= 0:
        raise WriterModelError("invalid Writer-only RL rank coverage request")
    scheduled = sum(
        task_for_update(task_ids, world_size, rank, update) is not None
        for update in range(next_update)
    )
    return scheduled * rollouts_per_task
