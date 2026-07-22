"""Test-only PI05 three-arm task-local adaptation authority."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.pi05_source_checkpoint import read_json, sha256_file
from ember.reward.protocol import (
    RewardProtocolError,
    RewardTask,
    SUITE_HORIZONS,
    rank_strided_assignments,
    task_local_video_demo,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_LOCAL_CONFIG_SCHEMA = "ember_pi05_task_local_rl_v1"
TASK_LOCAL_ARMS = ("identity", "as_writer", "rl_writer")


@dataclass(frozen=True, order=True)
class TaskLocalUnit:
    global_task_id: int
    suite: str
    task_id: int
    adaptation_seed: int
    arm: str

    @property
    def key(self) -> str:
        return (
            f"task_{self.global_task_id:03d}_seed_{self.adaptation_seed:010d}_{self.arm}"
        )


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    return REPO_ROOT / str(config["authorities"][name]["path"])


def _validate_role_and_arms(config: Mapping[str, Any]) -> None:
    role = config.get("role", {})
    arms = config.get("arms", {})
    initialization = config.get("initialization", {})
    if (
        role.get("split_role") != "test"
        or role.get("task_global_ids") != [6, 8, 10, 17, 24, 27, 30, 33]
        or int(role.get("task_count", -1)) != 8
        or int(role.get("test_actions_read_before_or_during_adaptation", -1)) != 0
        or tuple(arms) != TASK_LOCAL_ARMS
        or "source_sft" not in config.get("forbidden_arms", [])
        or initialization.get("writer_arms_share_video") is not True
        or initialization.get("writer_lora_generated_once") is not True
        or initialization.get("writer_lora_fixed_during_adaptation") is not True
    ):
        raise RewardProtocolError("task-local RL role or initialization wall changed")


def _validate_execution_contract(config: Mapping[str, Any]) -> None:
    algorithm = config.get("algorithm", {})
    environment = config.get("environment", {})
    policy = config.get("policy", {})
    if (
        algorithm.get("name")
        != "on_policy_binary_success_filtered_executed_prefix_flow_regression"
        or algorithm.get("executed_action_prefix_only") is not True
        or algorithm.get("teacher_actions") is not False
        or environment.get("official_random_bddl_reset") is not True
        or environment.get("fixed_pruned_init_states") is not False
        or int(environment.get("dummy_settling_steps", -1)) != 10
        or environment.get("horizons") != SUITE_HORIZONS
        or int(policy.get("chunk_size", -1)) != 50
        or int(policy.get("num_inference_steps", -1)) != 10
        or int(policy.get("replan_steps", -1)) != 5
        or config.get("checkpoint_selection", {}).get(
            "fixed_50_evaluation_is_separate"
        )
        is not True
        or int(config.get("parallel", {}).get("world_size", -1)) != 8
    ):
        raise RewardProtocolError("task-local RL execution wall changed")


def load_task_local_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    expected_authorities = {
        "target_data_manifest",
        "evaluation_config",
        "as_writer_config",
        "rl_writer_config",
        "feature_cache_config",
        "lora_contract",
        "source_base_config",
    }
    if (
        config.get("schema_version") != TASK_LOCAL_CONFIG_SCHEMA
        or config.get("sealed_stage") != "test"
        or set(config.get("authorities", {})) != expected_authorities
    ):
        raise RewardProtocolError("unsupported PI05 task-local RL config")
    for name, authority in config["authorities"].items():
        artifact = REPO_ROOT / str(authority.get("path", ""))
        if not artifact.is_file() or sha256_file(artifact) != authority.get("sha256"):
            raise RewardProtocolError(f"task-local RL authority changed: {name}")
    _validate_role_and_arms(config)
    _validate_execution_contract(config)
    tasks = test_tasks(config)
    if tuple(task.global_task_id for task in tasks) != tuple(
        config["role"]["task_global_ids"]
    ):
        raise RewardProtocolError("task-local RL test task identities changed")
    return config


def test_tasks(config: Mapping[str, Any]) -> tuple[RewardTask, ...]:
    expected_ids = set(map(int, config["role"]["task_global_ids"]))
    manifest = read_json(authority_path(config, "target_data_manifest"))
    tasks = []
    for row in manifest.get("tasks", []):
        if int(row["global_task_id"]) not in expected_ids:
            continue
        bddl = row["bddl"]
        tasks.append(
            RewardTask(
                suite=str(row["suite"]),
                task_id=int(row["task_id"]),
                global_task_id=int(row["global_task_id"]),
                split_role=str(row["split_role"]),
                language=str(row["language"]),
                problem_folder=str(row["problem_folder"]),
                bddl_file=str(bddl["filename"]),
                bddl_bytes=int(bddl["bytes"]),
                bddl_sha256=str(bddl["sha256"]),
                horizon=SUITE_HORIZONS[str(row["suite"])],
            )
        )
    tasks.sort(key=lambda task: task.global_task_id)
    if (
        len(tasks) != 8
        or {task.global_task_id for task in tasks} != expected_ids
        or any(task.split_role != "test" for task in tasks)
    ):
        raise RewardProtocolError("task-local RL role is not the sealed eight test tasks")
    return tuple(tasks)


def task_local_units(
    config: Mapping[str, Any], *, rl_writer_available: bool
) -> tuple[TaskLocalUnit, ...]:
    arms = TASK_LOCAL_ARMS if rl_writer_available else TASK_LOCAL_ARMS[:2]
    seeds = tuple(int(value) for value in config["initialization"]["adaptation_seeds"])
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise RewardProtocolError("task-local adaptation seed set changed")
    return tuple(
        TaskLocalUnit(
            global_task_id=task.global_task_id,
            suite=task.suite,
            task_id=task.task_id,
            adaptation_seed=seed,
            arm=arm,
        )
        for task in test_tasks(config)
        for seed in seeds
        for arm in arms
    )


def unit_assignments(
    units: Sequence[TaskLocalUnit], world_size: int
) -> tuple[tuple[TaskLocalUnit, ...], ...]:
    assigned = rank_strided_assignments(units, world_size)
    if len({len(values) for values in assigned}) != 1:
        raise RewardProtocolError("task-local task-arm units are not rank-balanced")
    return assigned


def cohort_video_demo(
    config: Mapping[str, Any], unit: TaskLocalUnit, *, demo_count: int = 50
) -> int:
    return task_local_video_demo(
        int(config["initialization"]["teacher_video_seed"]),
        unit.global_task_id,
        unit.adaptation_seed,
        demo_count=demo_count,
    )


def select_adaptation_checkpoint(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not candidates:
        raise RewardProtocolError("task-local checkpoint selection is empty")
    normalized = []
    for candidate in candidates:
        successes = int(candidate.get("segment_successes", -1))
        rollouts = int(candidate.get("segment_rollouts", 0))
        actions = int(candidate.get("environment_action_cursor", -1))
        if successes < 0 or not successes <= rollouts or rollouts <= 0 or actions < 0:
            raise RewardProtocolError("invalid task-local checkpoint selection row")
        normalized.append((candidate, successes, rollouts, actions))

    def better(left: tuple[Any, int, int, int], right: tuple[Any, int, int, int]) -> bool:
        _, left_successes, left_rollouts, left_actions = left
        _, right_successes, right_rollouts, right_actions = right
        left_cross = left_successes * right_rollouts
        right_cross = right_successes * left_rollouts
        return left_cross > right_cross or (
            left_cross == right_cross and left_actions < right_actions
        )

    selected = normalized[0]
    for candidate in normalized[1:]:
        if better(candidate, selected):
            selected = candidate
    return selected[0]
