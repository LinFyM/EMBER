"""Runtime and immutable launch authority for matched task-local LoRA RL."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import random
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ember.direct_lora_protocol import policy_files
from ember.lora import (
    canonical_contract_sha256,
    load_lora_contract,
    task_lora_state_dict,
)
from ember.source_base_checkpoint import (
    DistributedContext,
    canonical_hash,
    git_state,
    parse_checkpoint_steps,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.task_local_rl_protocol import (
    TaskArm,
    load_task_local_rl_config,
    rank_assignments,
    task_arms,
)
from ember.writer.inference import FrozenWriterTaskAdapter
from ember.writer.model import WriterModelError
from ember.writer_rl_runtime import _build_policy_runtime, _prepare_libero_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def barrier(context: DistributedContext) -> None:
    if context.world_size > 1:
        dist.barrier()


@dataclass
class TaskLocalRLRuntime:
    config: dict[str, Any]
    task_ids: tuple[int, ...]
    units: tuple[TaskArm, ...]
    assignments: tuple[tuple[TaskArm, ...], ...]
    languages: dict[int, str]
    checkpoint_updates: tuple[int, ...]
    policy: torch.nn.Module
    preprocessor: Any
    postprocessor: Any
    env_preprocessor: Any
    env_postprocessor: Any
    adapter: FrozenWriterTaskAdapter
    lora_contract: Any
    source_policy_files: dict[str, str]
    contract: dict[str, Any]
    contract_sha256: str


def initialize_distributed() -> DistributedContext:
    if not torch.cuda.is_available():
        raise WriterModelError("task-local LoRA RL requires CUDA")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 8 or not 0 <= local_rank < torch.cuda.device_count():
        raise WriterModelError(
            "task-local LoRA RL requires eight symmetric CUDA ranks"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
    )


def _seed_everything(seed: int, context: DistributedContext) -> None:
    rank_seed = seed + context.rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _selected_task_ids(config: dict[str, Any], mode: str) -> tuple[int, ...]:
    task_ids = tuple(int(value) for value in config["role"]["task_ids"])
    if mode == "profile":
        task_ids = task_ids[: int(config["profile"]["task_count"])]
    if not task_ids:
        raise WriterModelError("task-local RL selected no tasks")
    return task_ids


def _validate_launch(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    context: DistributedContext,
    task_ids: tuple[int, ...],
    checkpoint_updates: tuple[int, ...],
) -> None:
    if (
        context.world_size != int(config["parallel"]["world_size"])
        or not 0 < args.stop_after_update <= args.total_updates
        or args.stop_after_update not in checkpoint_updates
    ):
        raise WriterModelError("invalid task-local RL launch segment or topology")
    if args.mode == "profile" and 2 * len(task_ids) != context.world_size:
        raise WriterModelError(
            "task-local profile must assign exactly one arm to every CUDA rank"
        )
    if args.mode == "formal":
        formal = config["formal_run"]
        expected = (
            formal.get("status"),
            int(formal.get("expected_world_size", -1)),
            int(formal.get("task_count", -1)),
            int(formal.get("total_updates", -1)),
            tuple(int(value) for value in formal.get("checkpoint_updates", [])),
        )
        actual = (
            "sealed",
            context.world_size,
            len(task_ids),
            args.total_updates,
            checkpoint_updates,
        )
        if expected != actual or args.stop_after_update != args.total_updates:
            raise WriterModelError("formal task-local RL run is not profile-sealed")
        if git_state()["dirty_paths"]:
            raise WriterModelError("formal task-local RL requires a clean worktree")


def _build_contract(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    context: DistributedContext,
    task_ids: tuple[int, ...],
    assignments: tuple[tuple[TaskArm, ...], ...],
    checkpoint_updates: tuple[int, ...],
    source_files: dict[str, str],
    adapter: FrozenWriterTaskAdapter,
    trainable: dict[str, Any],
    libero_paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "ember_task_local_lora_rl_launch_v1",
        "mode": args.mode,
        "git": git_state(),
        "config_sha256": sha256_file(args.config),
        "protocol": config["protocol"],
        "role": {**config["role"], "task_ids": list(task_ids)},
        "arms": config["arms"],
        "algorithm": config["algorithm"],
        "checkpoint_selection": config["checkpoint_selection"],
        "environment": config["environment"],
        "policy": config["policy"],
        "optimization": config["optimization"],
        "rng": config["rng"],
        "source_policy_files": source_files,
        "writer_initialization": adapter.evidence,
        "assignments": [
            [unit.key for unit in rank_units] for rank_units in assignments
        ],
        "runtime": {
            "world_size": context.world_size,
            "one_policy_cuda_process_per_gpu": True,
            "gpu0_extra_cuda_processes": 0,
            "envs_per_rank": 1,
            "env_worker_processes_per_rank": 0,
            "independent_task_arm_optimizers": True,
            "total_updates_per_arm": args.total_updates,
            "checkpoint_updates_per_arm": list(checkpoint_updates),
            "rollouts_per_update": int(
                config["algorithm"]["rollouts_per_update"]
            ),
            "fixed_init_state_sampling": False,
            "matched_seed_schedule_excludes_arm": True,
        },
        "trainable": trainable,
        "libero_paths": libero_paths,
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "lerobot": importlib.metadata.version("lerobot"),
        },
    }


def prepare_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> TaskLocalRLRuntime:
    config = load_task_local_rl_config(args.config)
    task_ids = _selected_task_ids(config, args.mode)
    selected = set(task_ids)
    units = tuple(unit for unit in task_arms(config) if unit.task_id in selected)
    assignments = rank_assignments(units, context.world_size)
    checkpoint_updates = parse_checkpoint_steps(
        args.checkpoint_updates, args.total_updates
    )
    _validate_launch(
        args=args,
        config=config,
        context=context,
        task_ids=task_ids,
        checkpoint_updates=checkpoint_updates,
    )
    _seed_everything(int(config["rng"]["training_seed"]), context)

    if context.is_main:
        if not args.resume and args.output_dir.exists() and any(
            args.output_dir.iterdir()
        ):
            raise WriterModelError(
                f"task-local RL output directory is not empty: {args.output_dir}"
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _prepare_libero_config(args.output_dir)
    barrier(context)
    os.environ["LIBERO_CONFIG_PATH"] = str(
        (args.output_dir / "libero_config").resolve()
    )
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(context.local_rank)

    selection = read_json(REPO_ROOT / config["protocol"]["source_selection"])
    source_files = policy_files(args.policy_path, selection)
    policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor = (
        _build_policy_runtime(
            config=config,
            policy_path=args.policy_path,
            task_ids=task_ids,
            device=context.device,
        )
    )
    adapter = FrozenWriterTaskAdapter(
        policy=policy,
        policy_files=source_files,
        writer_config_path=REPO_ROOT / config["protocol"]["writer_config"],
        writer_checkpoint=args.writer_checkpoint,
        feature_cache=args.feature_cache,
        task_ids=task_ids,
        device=context.device,
        require_formal=True,
        writer_rl_config_path=args.writer_rl_config,
    )
    lora_contract = load_lora_contract(
        REPO_ROOT / config["protocol"]["lora_contract"]
    )
    lora_state = task_lora_state_dict(policy)
    for parameter in lora_state.values():
        parameter.requires_grad_(True)
    trainable_names = sorted(
        name for name, value in policy.named_parameters() if value.requires_grad
    )
    if set(trainable_names) != set(lora_state) or any(
        parameter.requires_grad for parameter in adapter.writer.parameters()
    ):
        raise WriterModelError("task-local RL left shared state trainable")
    trainable = {
        "object": "task_local_lora_only",
        "parameter_count_per_arm": sum(
            value.numel() for value in lora_state.values()
        ),
        "name_count": len(trainable_names),
        "names_sha256": canonical_hash(trainable_names),
        "lora_contract_sha256": canonical_contract_sha256(lora_contract),
        "shared_base_parameter_count": 0,
        "writer_parameter_count": 0,
        "critic_parameter_count": 0,
    }
    libero_paths = read_json(args.output_dir / "libero_config" / "config.yaml")
    contract = _build_contract(
        args=args,
        config=config,
        context=context,
        task_ids=task_ids,
        assignments=assignments,
        checkpoint_updates=checkpoint_updates,
        source_files=source_files,
        adapter=adapter,
        trainable=trainable,
        libero_paths=libero_paths,
    )
    contract_sha256 = canonical_hash(contract)
    if context.is_main:
        contract_path = args.output_dir / "run_contract.json"
        if args.resume:
            if (
                not contract_path.is_file()
                or canonical_hash(read_json(contract_path)) != contract_sha256
            ):
                raise WriterModelError("task-local RL resume launch changed")
        else:
            write_json_atomic(contract_path, contract)
            write_json_atomic(
                args.output_dir / "runtime_paths.json",
                {
                    "host": socket.gethostname(),
                    "source_policy": str(args.policy_path),
                    "writer_checkpoint": str(args.writer_checkpoint),
                    "writer_feature_cache": str(args.feature_cache),
                    "writer_rl_config": (
                        str(args.writer_rl_config)
                        if args.writer_rl_config is not None
                        else None
                    ),
                },
            )
    barrier(context)
    manifest = read_json(REPO_ROOT / config["protocol"]["manifest"])
    languages = {
        int(record["task_index"]): str(record["language"])
        for record in manifest["tasks"]
        if int(record["task_index"]) in selected
    }
    if set(languages) != selected:
        raise WriterModelError("task-local RL validation languages changed")
    torch.cuda.reset_peak_memory_stats(context.device)
    return TaskLocalRLRuntime(
        config=config,
        task_ids=task_ids,
        units=units,
        assignments=assignments,
        languages=languages,
        checkpoint_updates=checkpoint_updates,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        adapter=adapter,
        lora_contract=lora_contract,
        source_policy_files=source_files,
        contract=contract,
        contract_sha256=contract_sha256,
    )
