"""Canonical eight-rank Writer cold-start training on sealed source tasks."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from lerobot.configs import PreTrainedConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ember.lora import canonical_contract_sha256, load_lora_contract
from ember.source_base_checkpoint import (
    DistributedContext,
    barrier,
    canonical_hash,
    git_state,
    parse_checkpoint_steps,
    read_json,
    restore_rng,
    sha256_file,
    write_json_atomic,
)
from ember.writer.checkpoint import load_writer_checkpoint, save_writer_checkpoint
from ember.writer.data import FunctionalQueryDataset, MixedTaskBatchSampler
from ember.writer.feature_cache import (
    WriterFeatureStore,
    load_feature_cache_config,
    load_train_tasks,
)
from ember.writer.functional import (
    prepare_frozen_writer_policy,
    writer_functional_action_loss,
)
from ember.writer.model import (
    CompleteLoRAWriter,
    WriterModelError,
    build_lora_tensor_specs,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class WriterRuntime:
    config: dict[str, Any]
    checkpoint_steps: tuple[int, ...]
    dataset: FunctionalQueryDataset
    task_ids: tuple[int, ...]
    sampler: MixedTaskBatchSampler
    iterator: Any
    feature_store: WriterFeatureStore
    policy: SmolVLAPolicy
    preprocessor: Any
    writer: CompleteLoRAWriter
    wrapped_writer: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    contract: dict[str, Any]
    contract_sha256: str
    resume_step: int


def initialize_distributed() -> DistributedContext:
    if not torch.cuda.is_available():
        raise WriterModelError("Writer training requires CUDA")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not 0 <= local_rank < torch.cuda.device_count():
        raise WriterModelError("LOCAL_RANK is outside visible CUDA devices")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
    )


def seed_everything(seed: int, context: DistributedContext) -> None:
    rank_seed = seed + context.rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)
    torch.backends.cuda.matmul.allow_tf32 = True


def load_writer_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "ember_writer_cold_start_v1":
        raise WriterModelError("unsupported Writer cold-start config")
    protocol = config.get("protocol", {})
    for key in ("manifest", "feature_cache_config", "lora_contract"):
        authority = REPO_ROOT / str(protocol.get(key, ""))
        if not authority.is_file() or sha256_file(authority) != protocol.get(
            f"{key}_sha256"
        ):
            raise WriterModelError(f"sealed Writer authority changed: {key}")
    manifest = read_json(REPO_ROOT / protocol["manifest"])
    if (
        manifest.get("protocol_references", {}).get("split_sha256")
        != protocol.get("split_sha256")
    ):
        raise WriterModelError("Writer manifest and split disagree")
    return config


def _policy_files(policy_path: Path) -> dict[str, str]:
    names = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    missing = [name for name in names if not (policy_path / name).is_file()]
    if missing:
        raise WriterModelError(f"source policy is incomplete: {missing}")
    return {name: sha256_file(policy_path / name) for name in names}


def _validate_cache(
    cache_root: Path,
    *,
    policy_files: dict[str, str],
    task_ids: tuple[int, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_contract = read_json(cache_root / "run_contract.json")
    manifest = read_json(cache_root / "cache_manifest.json")
    if (
        run_contract.get("schema_version")
        != "ember_writer_feature_cache_launch_v1"
        or run_contract.get("mode") != "formal"
        or run_contract.get("policy_files") != {
            name: policy_files[name] for name in ("config.json", "model.safetensors")
        }
        or tuple(run_contract.get("task_ids", [])) != task_ids
        or tuple(run_contract.get("demo_indices", [])) != tuple(range(50))
        or manifest.get("schema_version")
        != "ember_writer_feature_cache_manifest_v1"
        or manifest.get("contract_sha256") != run_contract.get("contract_sha256")
        or int(manifest.get("task_count", -1)) != 70
        or int(manifest.get("episode_count", -1)) != 3500
    ):
        raise WriterModelError("formal Writer feature cache changed")
    return run_contract, manifest


def _build_policy(
    policy_path: Path, device: torch.device
) -> tuple[SmolVLAPolicy, Any]:
    policy_config = PreTrainedConfig.from_pretrained(policy_path)
    if not isinstance(policy_config, SmolVLAConfig):
        raise WriterModelError("source checkpoint is not SmolVLA")
    policy_config.device = str(device)
    policy_config.pretrained_path = policy_path
    policy_config.use_amp = False
    policy = SmolVLAPolicy.from_pretrained(policy_path, config=policy_config)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor


def _build_writer(
    config: dict[str, Any], policy: SmolVLAPolicy
) -> tuple[CompleteLoRAWriter, dict[str, Any]]:
    contract = load_lora_contract(REPO_ROOT / config["protocol"]["lora_contract"])
    template = prepare_frozen_writer_policy(policy, contract)
    writer_config = config["writer"]
    writer = CompleteLoRAWriter(
        build_lora_tensor_specs(template),
        template_state=template,
        **writer_config,
    )
    trainable_names = sorted(
        name for name, value in writer.named_parameters() if value.requires_grad
    )
    trainable = {
        "parameter_count": sum(value.numel() for value in writer.parameters()),
        "name_count": len(trainable_names),
        "names_sha256": canonical_hash(trainable_names),
        "lora_contract_sha256": canonical_contract_sha256(contract),
        "generated_lora_parameter_count": contract.parameter_count,
        "generated_lora_tensor_count": contract.state_tensor_count,
    }
    return writer, trainable


def _build_contract(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    context: DistributedContext,
    checkpoint_steps: tuple[int, ...],
    policy_files: dict[str, str],
    cache_contract: dict[str, Any],
    cache_manifest: dict[str, Any],
    trainable: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "ember_writer_cold_start_launch_v1",
        "mode": args.mode,
        "git": git_state(),
        "config_sha256": sha256_file(args.config.resolve()),
        "protocol": config["protocol"],
        "source_policy_files": policy_files,
        "feature_cache": {
            "contract_sha256": cache_contract["contract_sha256"],
            "extraction_sha256": cache_contract["extraction_sha256"],
            "manifest_sha256": sha256_file(args.feature_cache / "cache_manifest.json"),
            "task_count": cache_manifest["task_count"],
            "episode_count": cache_manifest["episode_count"],
            "frame_count": cache_manifest["frame_count"],
        },
        "writer": config["writer"],
        "data": config["data"],
        "optimization": config["optimization"],
        "runtime": {
            "world_size": context.world_size,
            "one_policy_cuda_process_per_rank": True,
            "per_rank_batch_size": args.batch_size,
            "effective_query_batch_size": context.world_size * args.batch_size,
            "total_steps": args.total_steps,
            "checkpoint_steps": list(checkpoint_steps),
            "num_workers_per_rank": args.num_workers,
            "ddp_writer_only": True,
            "ddp_broadcast_buffers": False,
            "ddp_static_graph": context.world_size > 1,
        },
        "trainable": trainable,
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "lerobot": importlib.metadata.version("lerobot"),
        },
    }


def _validate_launch(
    args: argparse.Namespace,
    config: dict[str, Any],
    context: DistributedContext,
    checkpoint_steps: tuple[int, ...],
) -> None:
    if not 0 < args.stop_after_step <= args.total_steps or args.batch_size <= 0:
        raise WriterModelError("invalid Writer step or batch request")
    if args.mode == "formal":
        formal = config["formal_run"]
        if formal.get("status") != "sealed":
            raise WriterModelError("formal Writer run is not sealed after profiling")
        expected = (
            int(formal["expected_world_size"]),
            int(formal["per_rank_batch_size"]),
            int(formal["total_steps"]),
            tuple(int(value) for value in formal["checkpoint_steps"]),
        )
        actual = (
            context.world_size,
            args.batch_size,
            args.total_steps,
            checkpoint_steps,
        )
        if actual != expected or args.stop_after_step != args.total_steps:
            raise WriterModelError("formal Writer launch differs from sealed profile")
        if git_state()["dirty_paths"]:
            raise WriterModelError("formal Writer launch requires a clean worktree")
    if context.world_size != 8:
        raise WriterModelError("Writer training requires exactly eight symmetric ranks")


def _reduce(value: float, context: DistributedContext, operation: dist.ReduceOp) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=context.device)
    if context.world_size > 1:
        dist.all_reduce(tensor, op=operation)
    return float(tensor.item())


def prepare_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> WriterRuntime:
    config = load_writer_config(args.config.resolve())
    checkpoint_steps = parse_checkpoint_steps(args.checkpoint_steps, args.total_steps)
    _validate_launch(args, config, context, checkpoint_steps)
    seed_everything(int(config["data"]["sampler_seed"]), context)

    cache_config = load_feature_cache_config(
        REPO_ROOT / config["protocol"]["feature_cache_config"], REPO_ROOT
    )
    tasks = load_train_tasks(cache_config, REPO_ROOT, args.data_root.resolve())
    task_ids = tuple(task.task_id for task in tasks)
    first_demo, last_demo = config["data"]["demo_indices"]
    dataset = FunctionalQueryDataset(
        [task.authority for task in tasks],
        demo_indices=range(first_demo, last_demo + 1),
        action_chunk_size=50,
        max_open_files_per_worker=int(config["data"]["max_open_files_per_worker"]),
    )

    resume_step = 0
    if args.resume is not None:
        preview = torch.load(
            args.resume / "trainer_state.pt", map_location="cpu", weights_only=False
        )
        resume_step = int(preview["next_step"])
        if not 0 <= resume_step < args.stop_after_step:
            raise WriterModelError("Writer resume step is outside this segment")
    sampler = MixedTaskBatchSampler(
        dataset,
        task_ids=task_ids,
        per_rank_batch_size=args.batch_size,
        start_step=resume_step,
        stop_step=args.stop_after_step,
        rank=context.rank,
        world_size=context.world_size,
        seed=int(config["data"]["sampler_seed"]),
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=config["loader"]["prefetch_factor"] if args.num_workers else None,
    )
    iterator = iter(dataloader)

    policy_files = _policy_files(args.policy_path.resolve())
    cache_contract, cache_manifest = _validate_cache(
        args.feature_cache.resolve(), policy_files=policy_files, task_ids=task_ids
    )
    policy, preprocessor = _build_policy(args.policy_path.resolve(), context.device)
    writer, trainable = _build_writer(config, policy)
    writer.to(context.device)
    optimizer = torch.optim.AdamW(
        writer.parameters(),
        lr=float(config["optimization"]["peak_lr"]),
        betas=tuple(config["optimization"]["betas"]),
        eps=float(config["optimization"]["eps"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    scheduler = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=int(
            config["optimization"]["scheduler_reference_warmup_steps"]
        ),
        num_decay_steps=int(
            config["optimization"]["scheduler_reference_decay_steps"]
        ),
        peak_lr=float(config["optimization"]["peak_lr"]),
        decay_lr=float(config["optimization"]["decay_lr"]),
    ).build(optimizer, args.total_steps)
    contract = _build_contract(
        args=args,
        config=config,
        context=context,
        checkpoint_steps=checkpoint_steps,
        policy_files=policy_files,
        cache_contract=cache_contract,
        cache_manifest=cache_manifest,
        trainable=trainable,
    )
    contract_sha256 = canonical_hash(contract)

    if context.is_main:
        if args.resume is None and args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise WriterModelError(f"Writer output directory is not empty: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        contract_path = args.output_dir / "run_contract.json"
        if args.resume is not None and (
            not contract_path.is_file()
            or canonical_hash(read_json(contract_path)) != contract_sha256
        ):
            raise WriterModelError("Writer resume launch contract changed")
        write_json_atomic(contract_path, contract)
        write_json_atomic(
            args.output_dir / "runtime_paths.json",
            {
                "host": socket.gethostname(),
                "source_policy": str(args.policy_path.resolve()),
                "feature_cache": str(args.feature_cache.resolve()),
                "data_root": str(args.data_root.resolve()),
            },
        )
    barrier(context)

    resume_rng = None
    if args.resume is not None:
        loaded_step, resume_rng = load_writer_checkpoint(
            checkpoint=args.resume.resolve(),
            context=context,
            writer=writer,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler_seed=int(config["data"]["sampler_seed"]),
            per_rank_batch_size=args.batch_size,
            contract_sha256=contract_sha256,
        )
        if loaded_step != resume_step:
            raise WriterModelError("Writer resume preview and checkpoint disagree")

    wrapped_writer: torch.nn.Module = writer
    if context.world_size > 1:
        wrapped_writer = DistributedDataParallel(
            writer,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
            static_graph=True,
        )
    wrapped_writer.train()
    feature_store = WriterFeatureStore(
        args.feature_cache.resolve(),
        task_ids=task_ids,
        expected_extraction_sha256=str(cache_contract["extraction_sha256"]),
        max_cached_tasks=int(config["data"]["feature_lru_tasks_per_rank"]),
        expected_dim=int(config["writer"]["vision_feature_dim"]),
    )
    torch.cuda.reset_peak_memory_stats(context.device)
    barrier(context)
    if resume_rng is not None:
        restore_rng(resume_rng, context)
    return WriterRuntime(
        config,
        checkpoint_steps,
        dataset,
        task_ids,
        sampler,
        iterator,
        feature_store,
        policy,
        preprocessor,
        writer,
        wrapped_writer,
        optimizer,
        scheduler,
        contract,
        contract_sha256,
        resume_step,
    )


def _task_id(batch: dict[str, Any]) -> int:
    values = batch.get("task_id")
    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise WriterModelError("functional query batch lost task identity")
    unique = values.unique()
    if unique.numel() != 1:
        raise WriterModelError("one Writer rank received multiple tasks")
    return int(unique.item())


def run_steps(
    args: argparse.Namespace, context: DistributedContext, runtime: WriterRuntime
) -> None:
    lora_contract = load_lora_contract(
        REPO_ROOT / runtime.config["protocol"]["lora_contract"]
    )
    step = runtime.resume_step
    while step < args.stop_after_step:
        tick = time.perf_counter()
        batch = next(runtime.iterator)
        data_seconds = time.perf_counter() - tick
        task_id = _task_id(batch)
        cached = runtime.feature_store.load(task_id)
        if cached.demo_indices.tolist() != list(range(50)):
            raise WriterModelError("Writer context does not contain all 50 episodes")
        language = cached.language_features.to(context.device)
        video = cached.video_features.to(context.device)
        for camera in ("observation.images.camera1", "observation.images.camera2"):
            batch[camera] = batch[camera].to(dtype=torch.float32).div_(255.0)
        batch = runtime.preprocessor(batch)

        runtime.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = writer_functional_action_loss(
                runtime.wrapped_writer,  # type: ignore[arg-type]
                runtime.policy,
                lora_contract,
                language_features=language,
                video_features=video,
                episode_offsets=cached.episode_offsets,
                batch=batch,
            )
        if not bool(torch.isfinite(loss).detach()):
            raise WriterModelError(f"non-finite Writer loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            runtime.writer.parameters(),
            float(runtime.config["optimization"]["grad_clip_norm"]),
        )
        runtime.optimizer.step()
        runtime.scheduler.step()
        torch.cuda.synchronize(context.device)
        step += 1
        step_seconds = time.perf_counter() - tick

        if step % args.log_every == 0 or step == args.stop_after_step:
            mean_loss = _reduce(float(loss.detach()), context, dist.ReduceOp.SUM)
            mean_loss /= context.world_size
            slowest_step = _reduce(step_seconds, context, dist.ReduceOp.MAX)
            slowest_data = _reduce(data_seconds, context, dist.ReduceOp.MAX)
            peak_allocated = _reduce(
                torch.cuda.max_memory_allocated(context.device) / 2**30,
                context,
                dist.ReduceOp.MAX,
            )
            peak_reserved = _reduce(
                torch.cuda.max_memory_reserved(context.device) / 2**30,
                context,
                dist.ReduceOp.MAX,
            )
            if context.is_main:
                print(
                    json.dumps(
                        {
                            "event": "train",
                            "step": step,
                            "loss": mean_loss,
                            "grad_norm_rank0": float(grad_norm),
                            "lr": runtime.scheduler.get_last_lr()[0],
                            "step_seconds_max": slowest_step,
                            "data_seconds_max": slowest_data,
                            "global_queries_per_second": context.world_size
                            * args.batch_size
                            / slowest_step,
                            "peak_allocated_gib_max": peak_allocated,
                            "peak_reserved_gib_max": peak_reserved,
                        }
                    ),
                    flush=True,
                )
        if step in runtime.checkpoint_steps:
            save_writer_checkpoint(
                output_dir=args.output_dir,
                step=step,
                context=context,
                writer=runtime.writer,
                optimizer=runtime.optimizer,
                scheduler=runtime.scheduler,
                sampler=runtime.sampler,
                contract=runtime.contract,
                mode=args.mode,
            )
    if context.is_main:
        print(json.dumps({"event": "complete", "step": step}), flush=True)


def train(args: argparse.Namespace) -> None:
    context = initialize_distributed()
    try:
        runtime = prepare_runtime(args, context)
        if context.is_main:
            print(
                json.dumps(
                    {
                        "event": "start",
                        "mode": args.mode,
                        "contract_sha256": runtime.contract_sha256,
                        "resume_step": runtime.resume_step,
                        "stop_after_step": args.stop_after_step,
                        "dataset_frames": len(runtime.dataset),
                        "tasks": len(runtime.task_ids),
                        "trainable": runtime.contract["trainable"],
                    }
                ),
                flush=True,
            )
        run_steps(args, context, runtime)
        runtime.dataset.close()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/writer_cold_start_v1.json"
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--checkpoint-steps", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    config = load_writer_config(args.config.resolve())
    if args.stop_after_step is None:
        args.stop_after_step = args.total_steps
    if args.num_workers is None:
        args.num_workers = int(config["loader"]["num_workers_per_rank"])
    return args
