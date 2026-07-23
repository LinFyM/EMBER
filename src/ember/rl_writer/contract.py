"""Authorities and deterministic task schedule for PI05 RL-Writer."""

from __future__ import annotations

import argparse
import importlib.metadata
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist

from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import DistributedContext, write_json_atomic
from ember.pi05_source_checkpoint import canonical_hash, read_json, sha256_file
from ember.pi05_source_contract import append_jsonl
from ember.reward.protocol import RewardProtocolError, RewardTask, SUITE_HORIZONS
from ember.writer.data import TeacherVideoSchedule


REPO_ROOT = Path(__file__).resolve().parents[3]
RL_WRITER_CONFIG_SCHEMA = "ember_pi05_rl_writer_v1"
RL_WRITER_LAUNCH_SCHEMA = "ember_pi05_rl_writer_launch_v1"
RL_WRITER_BRANCHES = {"zero_as_warmup", "micro_as_warmup"}
_SCHEDULE_TAG = 0xA17E


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    return REPO_ROOT / str(config["authorities"][name]["path"])


def _validate_authorities(config: Mapping[str, Any]) -> None:
    expected = {
        "target_data_manifest",
        "evaluation_config",
        "as_writer_config",
        "feature_cache_config",
        "lora_contract",
        "source_base_config",
        "tokenizer_manifest",
    }
    if set(config.get("authorities", {})) != expected:
        raise RewardProtocolError("RL-Writer authority set changed")
    for name, authority in config["authorities"].items():
        path = REPO_ROOT / str(authority.get("path", ""))
        if not path.is_file() or sha256_file(path) != authority.get("sha256"):
            raise RewardProtocolError(f"RL-Writer authority changed: {name}")


def _validate_information_wall(config: Mapping[str, Any]) -> None:
    wall = config.get("information_wall", {})
    branches = config.get("branches", {})
    algorithm = config.get("algorithm", {})
    environment = config.get("environment", {})
    policy = config.get("policy", {})
    stage = str(config.get("sealed_stage", ""))
    expected_roles = (
        ["train"] if stage == "development" else ["train", "validation"]
    )
    role_fields = (
        ("development_reward_split_roles", "development_video_split_roles")
        if stage == "development"
        else ("final_reward_split_roles", "final_video_split_roles")
    )
    expected_micro_queries = 24 if stage == "development" else 32
    zero_read_fields = (
        (
            "validation_reward_reads",
            "validation_action_reads",
            "test_reward_reads",
            "test_action_reads",
            "fixed_pruned_init_reads",
        )
        if stage == "development"
        else (
            "validation_action_reads",
            "test_reward_reads",
            "test_action_reads",
            "fixed_pruned_init_reads",
        )
    )
    if (
        stage not in {"development", "final"}
        or wall.get("writer_input")
        != "pure task language plus exactly one action-hidden teacher video"
        or wall.get(role_fields[0]) != expected_roles
        or wall.get(role_fields[1]) != expected_roles
        or any(int(wall.get(name, -1)) != 0 for name in zero_read_fields)
        or set(branches) != RL_WRITER_BRANCHES
        or int(branches["zero_as_warmup"].get("teacher_action_queries", -1)) != 0
        or int(branches["micro_as_warmup"].get("teacher_action_queries", -1))
        != expected_micro_queries
        or "fresh same-seed" not in branches["micro_as_warmup"].get(
            "writer_initialization", ""
        )
        or algorithm.get("name")
        != "on_policy_binary_success_filtered_executed_prefix_flow_regression"
        or algorithm.get("executed_action_prefix_only") is not True
        or int(algorithm.get("reward_replay_chunk_batch_size", -1)) != 8
        or algorithm.get("gradient_synchronization")
        != "ordered_manual_sum_after_local_backward"
        or algorithm.get("teacher_actions", False) is not False
        or environment.get("official_random_bddl_reset") is not True
        or environment.get("fixed_pruned_init_states") is not False
        or int(environment.get("dummy_settling_steps", -1)) != 10
        or environment.get("horizons") != SUITE_HORIZONS
        or int(policy.get("chunk_size", -1)) != 50
        or int(policy.get("num_inference_steps", -1)) != 10
        or int(policy.get("replan_steps", -1)) != 5
        or int(config.get("parallel", {}).get("world_size", -1)) != 8
        or int(config.get("parallel", {}).get("gpu0_extra_cuda_processes", -1)) != 0
    ):
        raise RewardProtocolError("RL-Writer information or execution wall changed")


def load_rl_writer_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    stage = str(config.get("sealed_stage", ""))
    if config.get("schema_version") != RL_WRITER_CONFIG_SCHEMA or stage not in {
        "development",
        "final",
    }:
        raise RewardProtocolError("unsupported PI05 RL-Writer config")
    _validate_authorities(config)
    _validate_information_wall(config)
    tasks = reward_tasks(config, stage=stage)
    expected_roles = (
        {"train"} if stage == "development" else {"train", "validation"}
    )
    if len(tasks) != (24 if stage == "development" else 32) or {
        task.split_role for task in tasks
    } != expected_roles:
        raise RewardProtocolError("RL-Writer sealed source role changed")
    return config


def reward_tasks(config: Mapping[str, Any], *, stage: str) -> tuple[RewardTask, ...]:
    if stage not in {"development", "final"}:
        raise RewardProtocolError("unsupported RL-Writer stage")
    roles = {"train"} if stage == "development" else {"train", "validation"}
    manifest = read_json(authority_path(config, "target_data_manifest"))
    tasks = []
    for row in manifest.get("tasks", []):
        if row.get("split_role") not in roles:
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
    expected = 24 if stage == "development" else 32
    if len(tasks) != expected or len(set(tasks)) != expected:
        raise RewardProtocolError("RL-Writer task role is not 24/32")
    return tuple(tasks)


def updates_per_cycle(tasks: Sequence[RewardTask], world_size: int) -> int:
    if world_size <= 0 or not tasks or len(tasks) % world_size:
        raise RewardProtocolError("RL-Writer full task cycle is not rank-balanced")
    return len(tasks) // world_size


def task_for_update(
    tasks: Sequence[RewardTask],
    *,
    world_size: int,
    rank: int,
    update: int,
    seed: int,
) -> tuple[RewardTask, int, int]:
    """Return task, full-cycle index, and task visit for one rank/update."""

    if not 0 <= rank < world_size or update < 0 or seed < 0:
        raise RewardProtocolError("invalid RL-Writer schedule cursor")
    slots = updates_per_cycle(tasks, world_size)
    cycle, slot = divmod(update, slots)
    order = np.random.default_rng(
        np.random.SeedSequence([seed, cycle, _SCHEDULE_TAG])
    ).permutation(len(tasks))
    task = tasks[int(order[slot * world_size + rank])]
    return task, cycle, cycle


def schedule_summary(
    tasks: Sequence[RewardTask],
    *,
    world_size: int,
    next_update: int,
    seed: int,
    rollouts_per_task_update: int,
    video_schedule: TeacherVideoSchedule,
) -> dict[str, Any]:
    if next_update < 0 or rollouts_per_task_update <= 0:
        raise RewardProtocolError("invalid RL-Writer coverage cursor")
    counts = {task.global_task_id: 0 for task in tasks}
    videos = {task.global_task_id: set() for task in tasks}
    digest_rows = []
    for update in range(next_update):
        for rank in range(world_size):
            task, cycle, visit = task_for_update(
                tasks,
                world_size=world_size,
                rank=rank,
                update=update,
                seed=seed,
            )
            demo = video_schedule.demo_for_task_visit(task.global_task_id, visit)
            counts[task.global_task_id] += rollouts_per_task_update
            videos[task.global_task_id].add(demo)
            digest_rows.append([update, rank, task.global_task_id, cycle, visit, demo])
    values = tuple(counts.values())
    video_counts = tuple(len(value) for value in videos.values())
    slots = updates_per_cycle(tasks, world_size)
    return {
        "next_update": next_update,
        "completed_full_task_cycles": next_update // slots,
        "cycle_slot_cursor": next_update % slots,
        "declared_task_count": len(tasks),
        "tasks_with_interactions": sum(value > 0 for value in values),
        "min_rollouts_per_task": min(values),
        "max_rollouts_per_task": max(values),
        "total_rollouts": sum(values),
        "min_unique_videos_per_task": min(video_counts),
        "max_unique_videos_per_task": max(video_counts),
        "schedule_sha256": canonical_hash(digest_rows),
    }


def _checkpoint_updates(value: str | Sequence[int], total_updates: int) -> tuple[int, ...]:
    raw = value.split(",") if isinstance(value, str) else value
    try:
        result = tuple(sorted({int(item) for item in raw}))
    except (TypeError, ValueError) as error:
        raise RewardProtocolError("invalid RL-Writer checkpoint updates") from error
    if not result or result[-1] != total_updates or any(item <= 0 for item in result):
        raise RewardProtocolError("RL-Writer checkpoints must end at total updates")
    return result


def resolve_runtime(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> tuple[int, tuple[int, ...]]:
    if args.stage != config.get("sealed_stage"):
        raise RewardProtocolError("RL-Writer stage requires its own immutable config")
    if args.branch not in RL_WRITER_BRANCHES:
        raise RewardProtocolError("unsupported RL-Writer branch")
    if args.branch == "micro_as_warmup":
        raise RewardProtocolError(
            "micro-AS RL-Writer remains blocked until zero-branch no-signal evidence"
        )
    if context.world_size != 8:
        raise RewardProtocolError("RL-Writer requires eight symmetric CUDA ranks")
    if args.mode == "formal":
        formal = config["formal_run"]
        if formal.get("status") != "sealed":
            raise RewardProtocolError("formal RL-Writer is pending a real source/profile seal")
        source = formal
    else:
        source = config["profile_defaults"]
    total = int(args.total_updates or source["total_updates"])
    checkpoints = _checkpoint_updates(
        args.checkpoint_updates or source["checkpoint_updates"], total
    )
    stop = int(args.stop_after_update or total)
    if not 0 < stop <= total or stop not in checkpoints:
        raise RewardProtocolError("RL-Writer segment must end at a checkpoint")
    cycle_updates = updates_per_cycle(
        reward_tasks(config, stage=args.stage), context.world_size
    )
    if any(value % cycle_updates for value in checkpoints):
        raise RewardProtocolError("RL-Writer checkpoints need complete source-task cycles")
    if args.mode == "formal":
        expected = (
            int(source["expected_world_size"]),
            int(source["total_updates"]),
            tuple(int(value) for value in source["checkpoint_updates"]),
        )
        if expected != (context.world_size, total, checkpoints):
            raise RewardProtocolError("formal RL-Writer launch differs from its seal")
        state = git_state(REPO_ROOT)
        if state["dirty_paths"]:
            raise RewardProtocolError("formal RL-Writer requires a clean worktree")
        if args.resume is None and state["commit"] != state["origin_main"]:
            raise RewardProtocolError("fresh formal RL-Writer must already be pushed")
        if context.numa_node is None or not context.cpu_affinity:
            raise RewardProtocolError("formal RL-Writer requires GPU-local NUMA affinity")
    args.stop_after_update = stop
    return total, checkpoints


def build_contract(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    source: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    feature_cache: Mapping[str, Any],
    tasks: Sequence[RewardTask],
    trainable: Mapping[str, Any],
    total_updates: int,
    checkpoint_updates: Sequence[int],
    libero_paths: Mapping[str, str],
) -> dict[str, Any]:
    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "device": str(context.device),
        "numa_node": context.numa_node,
        "cpu_affinity": list(context.cpu_affinity or ()),
    }
    topology: list[Any] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(topology, local)
    else:
        topology[0] = local
    return {
        "schema_version": RL_WRITER_LAUNCH_SCHEMA,
        "mode": args.mode,
        "stage": args.stage,
        "branch": args.branch,
        "git": {
            key: value
            for key, value in git_state(REPO_ROOT).items()
            if key in {"branch", "commit"}
        },
        "host": socket.gethostname(),
        "config_sha256": sha256_file(args.config.resolve()),
        "authorities": dict(config["authorities"]),
        "source": dict(source),
        "tokenizer": dict(tokenizer),
        "feature_cache": dict(feature_cache),
        "information_wall": dict(config["information_wall"]),
        "branch_contract": dict(config["branches"][args.branch]),
        "algorithm": dict(config["algorithm"]),
        "environment": dict(config["environment"]),
        "policy": dict(config["policy"]),
        "data": dict(config["data"]),
        "optimization": dict(config["optimization"]),
        "rng": dict(config["rng"]),
        "tasks": [
            {
                "suite": task.suite,
                "task_id": task.task_id,
                "global_task_id": task.global_task_id,
                "split_role": task.split_role,
                "language": task.language,
                "problem_folder": task.problem_folder,
                "bddl_file": task.bddl_file,
                "bddl_bytes": task.bddl_bytes,
                "bddl_sha256": task.bddl_sha256,
                "horizon": task.horizon,
            }
            for task in tasks
        ],
        "trainable": dict(trainable),
        "runtime": {
            "world_size": context.world_size,
            "one_policy_cuda_process_per_gpu": True,
            "gpu0_extra_cuda_processes": 0,
            "topology": topology,
            "persistent_policy": True,
            "persistent_task_environment_pool": True,
            "total_updates": total_updates,
            "checkpoint_updates": list(checkpoint_updates),
            "updates_per_full_task_cycle": updates_per_cycle(tasks, context.world_size),
            "rollouts_per_task_update": int(
                config["algorithm"]["rollouts_per_task_update"]
            ),
            "fixed_init_state_sampling": False,
            "explicit_per_replan_flow_noise": True,
            "executed_prefix_mask": True,
        },
        "libero_paths": dict(libero_paths),
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("lerobot", "peft", "safetensors")
            },
        },
    }


def publish_contract(
    *,
    output_dir: Path,
    contract: Mapping[str, Any],
    resume: Path | None,
    context: DistributedContext,
) -> str:
    digest = canonical_hash(contract)
    digests: list[Any] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(digests, digest)
    else:
        digests[0] = digest
    payload: list[Any] = [None]
    if context.is_main:
        try:
            if len(set(digests)) != 1:
                raise RewardProtocolError("RL-Writer rank launch contracts differ")
            path = output_dir / "run_contract.json"
            if resume is not None:
                if not path.is_file() or canonical_hash(read_json(path)) != digest:
                    raise RewardProtocolError("RL-Writer resume launch contract changed")
            else:
                existing = set(output_dir.iterdir()) if output_dir.exists() else set()
                allowed = {output_dir / "libero_config"}
                if existing - allowed:
                    raise RewardProtocolError(
                        "fresh RL-Writer output directory is not empty"
                    )
                output_dir.mkdir(parents=True, exist_ok=True)
                write_json_atomic(path, dict(contract))
            append_jsonl(
                output_dir / "invocations.jsonl",
                {
                    "argv": sys.argv,
                    "host": socket.gethostname(),
                    "resume": str(resume) if resume else None,
                    "started_unix": time.time(),
                },
            )
            payload[0] = {"digest": digest}
        except Exception as error:
            payload[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=context.device)
    if payload[0].get("error"):
        raise RewardProtocolError(payload[0]["error"])
    return str(payload[0]["digest"])
