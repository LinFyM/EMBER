"""Runtime construction and immutable launch contract for Writer-only RL."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import random
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from lerobot.configs import PreTrainedConfig
from lerobot.envs import make_env_pre_post_processors
from lerobot.envs.configs import LiberoEnv
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from torch.nn.parallel import DistributedDataParallel

from ember.lora import load_lora_contract
from ember.source_base_checkpoint import (
    DistributedContext,
    barrier,
    canonical_hash,
    git_state,
    parse_checkpoint_steps,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.writer.inference import FrozenWriterTaskAdapter
from ember.writer.model import CompleteLoRAWriter, WriterModelError
from ember.writer_rl_checkpoint import load_writer_rl_checkpoint, restore_rng
from ember.writer_rl_protocol import (
    load_writer_rl_config,
    source_task_ids,
    updates_per_cycle,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_FILE_NAMES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


@dataclass
class WriterRLRuntime:
    config: dict[str, Any]
    task_ids: tuple[int, ...]
    languages: dict[int, str]
    checkpoint_updates: tuple[int, ...]
    policy: torch.nn.Module
    preprocessor: Any
    postprocessor: Any
    env_preprocessor: Any
    env_postprocessor: Any
    adapter: FrozenWriterTaskAdapter
    writer: CompleteLoRAWriter
    wrapped_writer: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    lora_contract: Any
    contract: dict[str, Any]
    contract_sha256: str
    next_update: int
    optimizer_updates: int
    local_counters: dict[str, int]


def initialize_distributed() -> DistributedContext:
    if not torch.cuda.is_available():
        raise WriterModelError("Writer-only RL requires CUDA")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 8 or not 0 <= local_rank < torch.cuda.device_count():
        raise WriterModelError("Writer-only RL requires eight symmetric CUDA ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
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


def _policy_files(policy_path: Path) -> dict[str, str]:
    missing = [
        name for name in POLICY_FILE_NAMES if not (policy_path / name).is_file()
    ]
    if missing:
        raise WriterModelError(f"source policy is incomplete: {missing}")
    return {name: sha256_file(policy_path / name) for name in POLICY_FILE_NAMES}


def _prepare_libero_config(output_dir: Path) -> dict[str, str]:
    package = importlib.util.find_spec("libero")
    if package is None or package.origin is None:
        raise WriterModelError("installed LIBERO package cannot be located")
    benchmark_root = Path(package.origin).resolve().parent / "libero"
    paths = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(benchmark_root.parent / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    for name in ("benchmark_root", "bddl_files", "assets"):
        if not Path(paths[name]).exists():
            raise WriterModelError(f"LIBERO {name} path is missing: {paths[name]}")
    config_dir = output_dir / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(config_dir / "config.yaml", paths)
    return paths


def _validate_launch(
    args: argparse.Namespace,
    config: dict[str, Any],
    context: DistributedContext,
    checkpoint_updates: tuple[int, ...],
) -> None:
    if not 0 < args.stop_after_update <= args.total_updates:
        raise WriterModelError("invalid Writer-only RL update segment")
    if context.world_size != int(config["parallel"]["world_size"]):
        raise WriterModelError("Writer-only RL launch world size changed")
    if args.mode == "formal":
        formal = config["formal_run"]
        expected = (
            formal.get("status"),
            int(formal.get("total_updates", -1)),
            tuple(int(value) for value in formal.get("checkpoint_updates", [])),
            int(formal.get("expected_world_size", -1)),
        )
        actual = (
            "sealed",
            args.total_updates,
            checkpoint_updates,
            context.world_size,
        )
        if expected != actual or args.stop_after_update != args.total_updates:
            raise WriterModelError("formal Writer-only RL run is not sealed")
        cycle_updates = updates_per_cycle(source_task_ids(config), context.world_size)
        if any(value % cycle_updates for value in checkpoint_updates):
            raise WriterModelError("formal reward checkpoints must end full task cycles")
        if git_state()["dirty_paths"]:
            raise WriterModelError("formal Writer-only RL requires a clean worktree")


def _build_policy_runtime(
    *,
    config: dict[str, Any],
    policy_path: Path,
    task_ids: tuple[int, ...],
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Any, Any, Any]:
    environment = config["environment"]
    rename_map = dict(config["policy"]["rename_map"])
    policy_config = PreTrainedConfig.from_pretrained(policy_path)
    if not isinstance(policy_config, SmolVLAConfig):
        raise WriterModelError("source checkpoint is not SmolVLA")
    policy_config.device = str(device)
    policy_config.pretrained_path = policy_path
    policy_config.use_amp = config["policy"]["precision"] == "bfloat16"
    if policy_config.n_action_steps != int(
        config["policy"]["action_execution_horizon"]
    ):
        raise WriterModelError("Writer-only RL policy is not using execution h50")
    feature_env = LiberoEnv(
        task=environment["suite"],
        task_ids=[task_ids[0]],
        episode_length=int(environment["max_horizon"]),
        obs_type="pixels_agent_pos",
        camera_name=environment["camera_name"],
        init_states=False,
        observation_height=int(environment["observation_height"]),
        observation_width=int(environment["observation_width"]),
        control_mode=environment["control_mode"],
    )
    policy = make_policy(policy_config, env_cfg=feature_env, rename_map=rename_map)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(policy_path),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=feature_env, policy_cfg=policy_config
    )
    return policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor


def _build_contract(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    context: DistributedContext,
    checkpoint_updates: tuple[int, ...],
    policy_files: dict[str, str],
    adapter: FrozenWriterTaskAdapter,
    trainable: dict[str, Any],
    libero_paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "ember_writer_only_rl_launch_v1",
        "mode": args.mode,
        "git": git_state(),
        "config_sha256": sha256_file(args.config.resolve()),
        "protocol": config["protocol"],
        "algorithm": config["algorithm"],
        "environment": config["environment"],
        "policy": config["policy"],
        "optimization": config["optimization"],
        "rng": config["rng"],
        "source_policy_files": policy_files,
        "cold_start_writer": adapter.evidence,
        "runtime": {
            "world_size": context.world_size,
            "one_policy_cuda_process_per_rank": True,
            "gpu0_extra_cuda_processes": 0,
            "envs_per_rank": 1,
            "env_worker_processes_per_rank": 0,
            "total_updates": args.total_updates,
            "updates_per_full_task_cycle": updates_per_cycle(
                source_task_ids(config), context.world_size
            ),
            "checkpoint_updates": list(checkpoint_updates),
            "rollouts_per_task_cycle": config["algorithm"][
                "rollouts_per_task_cycle"
            ],
            "ddp_writer_only": True,
            "generated_lora_in_place_updates": False,
            "fixed_init_state_sampling": False,
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
) -> WriterRLRuntime:
    config = load_writer_rl_config(args.config.resolve())
    task_ids = source_task_ids(config)
    checkpoint_updates = parse_checkpoint_steps(
        args.checkpoint_updates, args.total_updates
    )
    _validate_launch(args, config, context, checkpoint_updates)
    _seed_everything(int(config["rng"]["training_seed"]), context)

    if context.is_main:
        if args.resume is None and args.output_dir.exists() and any(
            args.output_dir.iterdir()
        ):
            raise WriterModelError(
                f"Writer-only RL output directory is not empty: {args.output_dir}"
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

    policy_path = args.policy_path.resolve()
    policy_files = _policy_files(policy_path)
    policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor = (
        _build_policy_runtime(
            config=config,
            policy_path=policy_path,
            task_ids=task_ids,
            device=context.device,
        )
    )
    adapter = FrozenWriterTaskAdapter(
        policy=policy,
        policy_files=policy_files,
        writer_config_path=(REPO_ROOT / config["protocol"]["writer_config"]),
        writer_checkpoint=args.writer_checkpoint.resolve(),
        feature_cache=args.feature_cache.resolve(),
        task_ids=task_ids,
        device=context.device,
        require_formal=True,
    )
    writer = adapter.writer
    for parameter in writer.parameters():
        parameter.requires_grad_(True)
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise WriterModelError("Writer-only RL left source policy trainable")
    trainable_names = sorted(name for name, _ in writer.named_parameters())
    trainable = {
        "object": "shared_writer_only",
        "parameter_count": sum(value.numel() for value in writer.parameters()),
        "name_count": len(trainable_names),
        "names_sha256": canonical_hash(trainable_names),
        "lora_contract_sha256": adapter.evidence["lora_contract_sha256"],
        "source_policy_parameter_count": 0,
        "generated_lora_in_place_parameter_count": 0,
        "critic_parameter_count": 0,
    }
    optimizer = torch.optim.AdamW(
        writer.parameters(),
        lr=float(config["optimization"]["learning_rate"]),
        betas=tuple(config["optimization"]["betas"]),
        eps=float(config["optimization"]["eps"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    libero_paths = read_json(args.output_dir / "libero_config" / "config.yaml")
    contract = _build_contract(
        args=args,
        config=config,
        context=context,
        checkpoint_updates=checkpoint_updates,
        policy_files=policy_files,
        adapter=adapter,
        trainable=trainable,
        libero_paths=libero_paths,
    )
    contract_sha256 = canonical_hash(contract)
    if context.is_main:
        contract_path = args.output_dir / "run_contract.json"
        if args.resume is not None and (
            not contract_path.is_file()
            or canonical_hash(read_json(contract_path)) != contract_sha256
        ):
            raise WriterModelError("Writer-only RL resume launch contract changed")
        write_json_atomic(contract_path, contract)
        write_json_atomic(
            args.output_dir / "runtime_paths.json",
            {
                "host": socket.gethostname(),
                "source_policy": str(policy_path),
                "cold_start_writer": str(args.writer_checkpoint.resolve()),
                "feature_cache": str(args.feature_cache.resolve()),
            },
        )
    barrier(context)

    next_update = 0
    optimizer_updates = 0
    local_counters = {
        "rollouts": 0,
        "successes": 0,
        "env_steps": 0,
        "wall_nanoseconds": 0,
    }
    resume_rng = None
    if args.resume is not None:
        next_update, optimizer_updates, local_counters, resume_rng = (
            load_writer_rl_checkpoint(
                checkpoint=args.resume.resolve(),
                context=context,
                writer=writer,
                optimizer=optimizer,
                scheduler=scheduler,
                contract_sha256=contract_sha256,
            )
        )
        if not 0 <= next_update < args.stop_after_update:
            raise WriterModelError("Writer-only RL resume cursor is outside segment")

    wrapped_writer: torch.nn.Module = DistributedDataParallel(
        writer,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=False,
        static_graph=True,
    )
    writer.train()
    lora_contract = load_lora_contract(
        REPO_ROOT / config["protocol"]["lora_contract"]
    )
    torch.cuda.reset_peak_memory_stats(context.device)
    barrier(context)
    if resume_rng is not None:
        restore_rng(resume_rng, context)
    manifest = read_json(REPO_ROOT / config["protocol"]["manifest"])
    languages = {
        int(record["task_index"]): str(record["language"])
        for record in manifest["tasks"]
        if record["split"] == "train"
    }
    return WriterRLRuntime(
        config=config,
        task_ids=task_ids,
        languages=languages,
        checkpoint_updates=checkpoint_updates,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        adapter=adapter,
        writer=writer,
        wrapped_writer=wrapped_writer,
        optimizer=optimizer,
        scheduler=scheduler,
        lora_contract=lora_contract,
        contract=contract,
        contract_sha256=contract_sha256,
        next_update=next_update,
        optimizer_updates=optimizer_updates,
        local_counters=local_counters,
    )


def writer_inputs(
    runtime: WriterRLRuntime, task_id: int, device: torch.device
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    cached = runtime.adapter.store.load(task_id)
    if cached.demo_indices.tolist() != list(range(50)):
        raise WriterModelError("Writer-only RL context does not contain all 50 videos")
    return (
        cached,
        cached.language_features.to(device),
        cached.video_features.to(device),
        cached.episode_offsets.to(device),
    )
