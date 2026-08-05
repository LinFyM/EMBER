"""Sealed contracts and full-24 schedules for antithetic Program-Credit Writer."""

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
from ember.pi05_source_checkpoint import (
    DistributedContext,
    canonical_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.pi05_source_contract import append_jsonl
from ember.reward.protocol import RewardProtocolError, RewardTask, SUITE_HORIZONS
from ember.writer.as_contract import load_writer_config
from ember.writer.as_sampling import TeacherVideoSchedule
from ember.writer.topology import visible_physical_cuda_index


REPO_ROOT = Path(__file__).resolve().parents[3]
RL_WRITER_CONFIG_SCHEMA = "ember_pi05_antithetic_program_credit_writer_v2"
RL_WRITER_LAUNCH_SCHEMA = "ember_pi05_antithetic_program_credit_launch_v1"
_SCHEDULE_TAG = 0xF10C0ED


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    return REPO_ROOT / str(config["authorities"][name]["path"])


def load_coldstart_writer_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the sealed non-parameter Writer runtime used by RL and evaluation."""

    resolved = load_writer_config(authority_path(config, "as_writer_config"))
    override = config.get("coldstart_writer_runtime", {})
    if set(override) != {"max_frames_per_encoder_call"}:
        raise RewardProtocolError("cold-start Writer runtime override changed")
    resolved = dict(resolved)
    resolved["writer"] = dict(resolved["writer"])
    resolved["writer"]["max_frames_per_encoder_call"] = int(
        override["max_frames_per_encoder_call"]
    )
    return resolved


def _validate_authorities(config: Mapping[str, Any]) -> None:
    expected = {
        "target_data_manifest",
        "evaluation_config",
        "as_writer_config",
        "lora_contract",
        "source_base_config",
        "tokenizer_manifest",
    }
    if set(config.get("authorities", {})) != expected:
        raise RewardProtocolError("Program-Credit authority set changed")
    for name, authority in config["authorities"].items():
        path = REPO_ROOT / str(authority.get("path", ""))
        if not path.is_file() or sha256_file(path) != authority.get("sha256"):
            raise RewardProtocolError(f"Program-Credit authority changed: {name}")


def _validate_information_wall(config: Mapping[str, Any]) -> None:
    wall = config.get("information_wall", {})
    algorithm = config.get("algorithm", {})
    environment = config.get("environment", {})
    policy = config.get("policy", {})
    parallel = config.get("parallel", {})
    observed = {
        "stage": config.get("sealed_stage"),
        "writer_input": wall.get("writer_input"),
        "reward_roles": wall.get("development_reward_split_roles"),
        "video_roles": wall.get("development_video_split_roles"),
        "observer_input": wall.get("progress_observer_input"),
        "algorithm": algorithm.get("name"),
        "rollouts": int(algorithm.get("rollouts_per_task_condition", -1)),
        "pairs": int(algorithm.get("antithetic_pairs_per_task", -1)),
        "program_shape": algorithm.get("program_shape"),
        "program_sigma": float(algorithm.get("program_sigma", -1)),
        "pair_credit": algorithm.get("pair_credit"),
        "rollout_schema": algorithm.get("rollout_schema"),
        "pair_randomness": algorithm.get("pair_randomness"),
        "freeze_observer": algorithm.get("semantic_encoder_frozen_after_coldstart"),
        "freeze_decoder": algorithm.get(
            "factor_head_decoder_frozen_after_coldstart"
        ),
        "coldstart_frame_chunk": config.get("coldstart_writer_runtime", {}).get(
            "max_frames_per_encoder_call"
        ),
        "source_backward": algorithm.get("source_policy_backward"),
        "functional_loss": algorithm.get("functional_action_loss"),
        "replay": algorithm.get("executed_action_replay"),
        "gradient_sync": algorithm.get("gradient_synchronization"),
        "teacher_actions": algorithm.get("teacher_actions", True),
        "critic": algorithm.get("critic", True),
        "random_reset": environment.get("official_random_bddl_reset"),
        "fixed_states": environment.get("fixed_pruned_init_states"),
        "environment_lanes": int(
            environment.get("paired_persistent_environment_lanes", -1)
        ),
        "pair_reset": environment.get("pair_reset_semantics"),
        "settling": int(environment.get("dummy_settling_steps", -1)),
        "horizons": environment.get("horizons"),
        "chunk": int(policy.get("chunk_size", -1)),
        "inference_steps": int(policy.get("num_inference_steps", -1)),
        "replan": int(policy.get("replan_steps", -1)),
        "max_world": int(parallel.get("maximum_world_size", -1)),
        "global_tasks": int(parallel.get("global_tasks_per_outer_cycle", -1)),
        "credit_ready": parallel.get("credit_collective_readiness"),
    }
    expected = {
        "stage": "development",
        "writer_input": "pure task language plus exactly one action-hidden teacher video",
        "reward_roles": ["train"],
        "video_roles": ["train"],
        "observer_input": "pure task language plus teacher and rollout agentview RGB only",
        "algorithm": "antithetic_program_credit_writer_v1",
        "rollouts": 4,
        "pairs": 2,
        "program_shape": [320, 256],
        "program_sigma": 0.05,
        "pair_credit": "binary_discordant_sign_paired_success_zero_paired_failure_half_semantic_difference",
        "rollout_schema": "ember_pi05_antithetic_program_credit_rollout_v2",
        "pair_randomness": "plus_minus_use_lockstep_persistent_environment_lanes_and_share_environment_seed_and_policy_noise_prefix",
        "freeze_observer": True,
        "freeze_decoder": True,
        "coldstart_frame_chunk": 16,
        "source_backward": False,
        "functional_loss": False,
        "replay": False,
        "gradient_sync": "full24_equal_task_manual_sum_after_local_program_backward",
        "teacher_actions": False,
        "critic": False,
        "random_reset": True,
        "fixed_states": False,
        "environment_lanes": 2,
        "pair_reset": "independent_lane_same_seed_same_reset_index",
        "settling": 10,
        "horizons": SUITE_HORIZONS,
        "chunk": 50,
        "inference_steps": 10,
        "replan": 5,
        "max_world": 6,
        "global_tasks": 24,
        "credit_ready": "launch_unique_atomic_rank_markers_after_cuda_complete_before_each_nccl_gradient_sum",
    }
    zero_reads = (
        "teacher_action_reads_after_coldstart",
        "validation_reward_reads",
        "validation_action_reads",
        "test_reward_reads",
        "test_action_reads",
        "fixed_pruned_init_reads",
    )
    if observed != expected or any(int(wall.get(name, -1)) != 0 for name in zero_reads):
        raise RewardProtocolError("Program-Credit information or execution wall changed")


def load_rl_writer_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != RL_WRITER_CONFIG_SCHEMA:
        raise RewardProtocolError("unsupported antithetic Program-Credit config")
    _validate_authorities(config)
    _validate_information_wall(config)
    load_coldstart_writer_config(config)
    progress = config.get("progress_credit", {})
    if (
        progress.get("observer")
        != "coldstart_frozen_writer_semantic_encoder_task_patch_plus_fixed_action_expert_interaction"
        or progress.get("teacher_frames") != "real_first_and_real_last"
        or progress.get("rollout_frames")
        != "post_settling_start_and_true_terminal"
        or progress.get("normalization") != "per_component_rms"
        or progress.get("projection")
        != "teacher_change_energy_weighted_cosine_times_clipped_relative_magnitude"
        or progress.get("episode_utility")
        != "terminal_start_relative_content_projection"
        or progress.get("usage") != "paired_failure_tie_break_only"
        or min(
            float(progress.get("normalization_epsilon", 0)),
            float(progress.get("projection_epsilon", 0)),
        )
        <= 0
    ):
        raise RewardProtocolError("task-grounded progress credit contract changed")
    tasks = reward_tasks(config)
    if len(tasks) != 24 or {task.split_role for task in tasks} != {"train"}:
        raise RewardProtocolError("Program-Credit development task split changed")
    return config


def reward_tasks(config: Mapping[str, Any]) -> tuple[RewardTask, ...]:
    manifest = read_json(authority_path(config, "target_data_manifest"))
    tasks = []
    for row in manifest.get("tasks", []):
        if row.get("split_role") != "train":
            continue
        bddl = row["bddl"]
        tasks.append(
            RewardTask(
                suite=str(row["suite"]),
                task_id=int(row["task_id"]),
                global_task_id=int(row["global_task_id"]),
                split_role="train",
                language=str(row["language"]),
                problem_folder=str(row["problem_folder"]),
                bddl_file=str(bddl["filename"]),
                bddl_bytes=int(bddl["bytes"]),
                bddl_sha256=str(bddl["sha256"]),
                horizon=SUITE_HORIZONS[str(row["suite"])],
            )
        )
    tasks.sort(key=lambda task: task.global_task_id)
    if len(tasks) != 24 or len(set(tasks)) != 24:
        raise RewardProtocolError("Program-Credit requires the sealed train24 split")
    return tuple(tasks)


def cycle_assignments(
    tasks: Sequence[RewardTask], *, world_size: int, cycle: int, seed: int
) -> tuple[tuple[RewardTask, ...], ...]:
    """Horizon-balanced full-24 LPT assignment with seeded tie rotation."""

    if (
        len(tasks) != 24
        or world_size <= 0
        or world_size > 6
        or len(tasks) % world_size
        or cycle < 0
        or seed < 0
    ):
        raise RewardProtocolError("invalid Program-Credit full-24 topology")
    capacity = len(tasks) // world_size
    permutation = np.random.default_rng(
        np.random.SeedSequence([seed, cycle, _SCHEDULE_TAG])
    ).permutation(len(tasks))
    priority = {int(index): ordinal for ordinal, index in enumerate(permutation)}
    ordered = sorted(
        enumerate(tasks), key=lambda pair: (-pair[1].horizon, priority[pair[0]])
    )
    bins: list[list[RewardTask]] = [[] for _ in range(world_size)]
    costs = [0] * world_size
    for _, task in ordered:
        eligible = [rank for rank in range(world_size) if len(bins[rank]) < capacity]
        rank = min(eligible, key=lambda value: (costs[value], len(bins[value]), value))
        bins[rank].append(task)
        costs[rank] += task.horizon
    result = tuple(tuple(values) for values in bins)
    flattened = [task.global_task_id for values in result for task in values]
    if len(flattened) != 24 or len(set(flattened)) != 24:
        raise RewardProtocolError("Program-Credit assignment lost a task")
    return result


def schedule_summary(
    tasks: Sequence[RewardTask],
    *,
    world_size: int,
    next_cycle: int,
    seed: int,
    rollouts_per_task: int,
    video_schedule: TeacherVideoSchedule,
) -> dict[str, Any]:
    if next_cycle < 0 or rollouts_per_task <= 0:
        raise RewardProtocolError("invalid Program-Credit coverage cursor")
    counts = {task.global_task_id: 0 for task in tasks}
    videos = {task.global_task_id: set() for task in tasks}
    rows = []
    for cycle in range(next_cycle):
        for rank, assigned in enumerate(
            cycle_assignments(tasks, world_size=world_size, cycle=cycle, seed=seed)
        ):
            for task in assigned:
                demo = video_schedule.demo_for_task_visit(task.global_task_id, cycle)
                counts[task.global_task_id] += rollouts_per_task
                videos[task.global_task_id].add(demo)
                rows.append([cycle, rank, task.global_task_id, demo])
    values = tuple(counts.values())
    video_counts = tuple(len(value) for value in videos.values())
    return {
        "next_cycle": next_cycle,
        "completed_full24_cycles": next_cycle,
        "declared_task_count": len(tasks),
        "tasks_with_interactions": sum(value > 0 for value in values),
        "min_rollouts_per_task": min(values),
        "max_rollouts_per_task": max(values),
        "total_rollouts": sum(values),
        "min_unique_videos_per_task": min(video_counts),
        "max_unique_videos_per_task": max(video_counts),
        "schedule_sha256": canonical_hash(rows),
    }


def _checkpoint_cycles(value: str | Sequence[int], total: int) -> tuple[int, ...]:
    raw = value.split(",") if isinstance(value, str) else value
    try:
        result = tuple(sorted({int(item) for item in raw}))
    except (TypeError, ValueError) as error:
        raise RewardProtocolError("invalid Program-Credit checkpoint cycles") from error
    if not result or result[-1] != total or any(item <= 0 for item in result):
        raise RewardProtocolError("Program-Credit checkpoints must end at total cycles")
    return result


def resolve_runtime(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> tuple[int, tuple[int, ...], int]:
    if args.stage != "development":
        raise RewardProtocolError("only development Program-Credit is authorized")
    if context.world_size not in {1, 2, 3, 4, 6}:
        raise RewardProtocolError("Program-Credit world size must divide train24 and be <=6")
    sources = {"profile": config["profile_defaults"], "formal": config["formal_run"]}
    source = sources[args.mode]
    if source.get("status") != "sealed":
        message = (
            "Program-Credit profile awaits focused CPU contracts"
            if args.mode == "profile"
            else "formal Program-Credit awaits the live profile seal"
        )
        raise RewardProtocolError(message)
    total = int(args.total_cycles or source["total_cycles"])
    checkpoints = _checkpoint_cycles(
        args.checkpoint_cycles or source["checkpoint_cycles"], total
    )
    stop = int(args.stop_after_cycle or total)
    epochs = int(args.learning_epochs or source["learning_epochs"])
    if not 0 < stop <= total or stop not in checkpoints or epochs != 1:
        raise RewardProtocolError("invalid Program-Credit segment or update count")
    if args.mode == "formal":
        expected = (
            int(source["expected_world_size"]),
            int(source["total_cycles"]),
            tuple(int(value) for value in source["checkpoint_cycles"]),
            int(source["learning_epochs"]),
        )
        if expected != (context.world_size, total, checkpoints, epochs):
            raise RewardProtocolError("formal Program-Credit launch differs from its seal")
        state = git_state(REPO_ROOT)
        if state["dirty_paths"]:
            raise RewardProtocolError("formal Program-Credit requires a clean worktree")
        if args.resume is None and state["commit"] != state["origin_main"]:
            raise RewardProtocolError("fresh formal Program-Credit must already be pushed")
        if context.numa_node is None or not context.cpu_affinity:
            raise RewardProtocolError("formal Program-Credit requires GPU-local NUMA affinity")
    args.stop_after_cycle = stop
    return total, checkpoints, epochs


def build_contract(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    source: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    coldstart: Mapping[str, Any],
    video_data: Mapping[str, Any],
    tasks: Sequence[RewardTask],
    trainable: Mapping[str, Any],
    total_cycles: int,
    checkpoint_cycles: Sequence[int],
    learning_epochs: int,
    libero_paths: Mapping[str, str],
) -> dict[str, Any]:
    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "physical_gpu": visible_physical_cuda_index(context.local_rank),
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
        "stage": "development",
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
        "coldstart": dict(coldstart),
        "video_data": dict(video_data),
        "information_wall": dict(config["information_wall"]),
        "algorithm": {**dict(config["algorithm"]), "learning_epochs": learning_epochs},
        "environment": dict(config["environment"]),
        "policy": dict(config["policy"]),
        "data": dict(config["data"]),
        "optimization": dict(config["optimization"]),
        "rng": dict(config["rng"]),
        "progress_credit": dict(config["progress_credit"]),
        "tasks": [task.__dict__ for task in tasks],
        "trainable": dict(trainable),
        "runtime": {
            "world_size": context.world_size,
            "tasks_per_rank_per_cycle": len(tasks) // context.world_size,
            "global_tasks_per_cycle": len(tasks),
            "topology": topology,
            "persistent_policy": True,
            "persistent_task_environment_pool": True,
            "persistent_environment_lanes_per_task": int(
                config["environment"]["paired_persistent_environment_lanes"]
            ),
            "total_cycles": total_cycles,
            "checkpoint_cycles": list(checkpoint_cycles),
            "rollouts_per_task_condition": int(
                config["algorithm"]["rollouts_per_task_condition"]
            ),
            "antithetic_pairs_per_task": 2,
            "program_shape": [320, 256],
            "program_sigma": float(config["algorithm"]["program_sigma"]),
            "optimizer_updates_per_cycle": 1,
            "source_policy_backward_calls": 0,
            "functional_action_loss_calls": 0,
        },
        "libero_paths": dict(libero_paths),
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("lerobot", "peft", "safetensors", "h5py")
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
                raise RewardProtocolError("Program-Credit rank launch contracts differ")
            path = output_dir / "run_contract.json"
            if resume is not None:
                if not path.is_file() or canonical_hash(read_json(path)) != digest:
                    raise RewardProtocolError("Program-Credit resume contract changed")
            else:
                existing = set(output_dir.iterdir()) if output_dir.exists() else set()
                allowed = {output_dir / "libero_config"}
                if existing - allowed:
                    raise RewardProtocolError("fresh Program-Credit output is not empty")
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
