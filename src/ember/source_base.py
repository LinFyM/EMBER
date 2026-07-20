"""Canonical distributed source-embodiment training for sealed LIBERO-90 data."""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ember.source_base_checkpoint import (
    DistributedContext,
    SourceBaseError,
    barrier,
    build_contract,
    canonical_hash,
    git_state,
    parse_checkpoint_steps,
    read_json,
    restore_rng,
    save_checkpoint,
    sha256_file,
    validate_launch,
    write_json_atomic,
)
from ember.writer.data import (
    FunctionalQueryDataset,
    MixedTaskBatchSampler,
    WriterTaskAuthority,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainingRuntime:
    config: dict[str, Any]
    checkpoint_steps: tuple[int, ...]
    dataset: FunctionalQueryDataset
    task_ids: tuple[int, ...]
    policy: SmolVLAPolicy
    trainable: dict[str, Any]
    preprocessor: Any
    postprocessor: Any
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    contract: dict[str, Any]
    contract_hash: str
    resume_step: int
    sampler: MixedTaskBatchSampler
    iterator: Any
    wrapped: torch.nn.Module


def _initialize_distributed() -> DistributedContext:
    if not torch.cuda.is_available():
        raise SourceBaseError("source-base training requires CUDA")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not 0 <= local_rank < torch.cuda.device_count():
        raise SourceBaseError("LOCAL_RANK is outside visible CUDA devices")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
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


def _load_launch_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "ember_source_base_v1":
        raise SourceBaseError("unsupported source-base config schema")
    protocol = config.get("protocol", {})
    for key in ("manifest", "normalization"):
        authority = REPO_ROOT / protocol[key]
        expected = protocol[f"{key}_sha256"]
        if sha256_file(authority) != expected:
            raise SourceBaseError(f"sealed {key} hash changed")
    return config


def _load_train_authorities(
    config: dict[str, Any], data_root: Path
) -> tuple[list[WriterTaskAuthority], dict[int, str]]:
    manifest = read_json(REPO_ROOT / config["protocol"]["manifest"])
    if manifest.get("protocol_references", {}).get("split_sha256") != config["protocol"]["split_sha256"]:
        raise SourceBaseError("manifest and split authorities disagree")
    authorities: list[WriterTaskAuthority] = []
    expected_hashes: dict[int, str] = {}
    for record in manifest.get("tasks", []):
        if record.get("split") != "train":
            continue
        task_id = int(record["task_index"])
        hdf5 = record["hdf5"]
        authorities.append(
            WriterTaskAuthority(
                task_id=task_id,
                language=str(record["language"]),
                path=data_root / hdf5["filename"],
                expected_bytes=int(hdf5["bytes"]),
                expected_sha256=None,
            )
        )
        expected_hashes[task_id] = str(hdf5["sha256"])
    authorities.sort(key=lambda item: item.task_id)
    if len(authorities) != 70 or len({item.task_id for item in authorities}) != 70:
        raise SourceBaseError("source base requires exactly 70 unique train tasks")
    return authorities, expected_hashes


def _load_stats(config: dict[str, Any], task_ids: Sequence[int]) -> dict[str, dict[str, np.ndarray]]:
    raw = read_json(REPO_ROOT / config["protocol"]["normalization"])
    authority = raw.get("authority", {})
    if authority.get("split") != "train" or authority.get("task_indices") != list(task_ids):
        raise SourceBaseError("normalization authority is not the sealed train split")
    stats: dict[str, dict[str, np.ndarray]] = {}
    for feature in ("observation.state", "action"):
        stats[feature] = {
            name: np.asarray(raw[feature][name], dtype=np.float32)
            for name in ("mean", "std")
        }
    return stats


def _build_policy_config(
    config: dict[str, Any], vlm_path: Path, device: torch.device
) -> SmolVLAConfig:
    camera_shape = tuple(config["features"]["camera_shape"])
    return SmolVLAConfig(
        input_features={
            "observation.state": PolicyFeature(
                type=FeatureType.STATE, shape=(config["features"]["state_dim"],)
            ),
            "observation.images.camera1": PolicyFeature(
                type=FeatureType.VISUAL, shape=camera_shape
            ),
            "observation.images.camera2": PolicyFeature(
                type=FeatureType.VISUAL, shape=camera_shape
            ),
        },
        output_features={
            "action": PolicyFeature(
                type=FeatureType.ACTION, shape=(config["features"]["action_dim"],)
            )
        },
        device=str(device),
        use_amp=False,
        push_to_hub=False,
        chunk_size=config["features"]["chunk_size"],
        n_action_steps=config["features"]["chunk_size"],
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=True,
        optimizer_lr=config["optimization"]["peak_lr"],
        optimizer_betas=tuple(config["optimization"]["betas"]),
        optimizer_eps=config["optimization"]["eps"],
        optimizer_weight_decay=config["optimization"]["weight_decay"],
        optimizer_grad_clip_norm=config["optimization"]["grad_clip_norm"],
        scheduler_warmup_steps=config["optimization"]["scheduler_reference_warmup_steps"],
        scheduler_decay_steps=config["optimization"]["scheduler_reference_decay_steps"],
        scheduler_decay_lr=config["optimization"]["decay_lr"],
        vlm_model_name=str(vlm_path),
        load_vlm_weights=True,
        attention_mode="cross_attn",
        prefix_length=0,
        pad_language_to="max_length",
        num_expert_layers=0,
        num_vlm_layers=16,
        self_attn_every_n_layers=2,
        expert_width_multiplier=0.75,
    )


def _build_policy(
    config: dict[str, Any],
    model_path: Path,
    vlm_path: Path,
    device: torch.device,
) -> SmolVLAPolicy:
    policy_config = _build_policy_config(config, vlm_path, device)
    policy = SmolVLAPolicy.from_pretrained(
        model_path,
        config=policy_config,
        local_files_only=True,
        strict=True,
    )
    policy.train()
    return policy


def _trainable_contract(policy: SmolVLAPolicy) -> dict[str, Any]:
    allowed = (
        "model.vlm_with_expert.lm_expert.",
        "model.state_proj.",
        "model.action_in_proj.",
        "model.action_out_proj.",
        "model.action_time_mlp_in.",
        "model.action_time_mlp_out.",
    )
    names = sorted(name for name, value in policy.named_parameters() if value.requires_grad)
    forbidden = [name for name in names if not name.startswith(allowed)]
    if forbidden or not names:
        raise SourceBaseError(f"unexpected trainable parameters: {forbidden[:5]}")
    if any(value.requires_grad for name, value in policy.named_parameters() if ".vlm." in name):
        raise SourceBaseError("VLM must remain frozen during source-base training")
    return {
        "parameter_count": sum(value.numel() for value in policy.parameters() if value.requires_grad),
        "total_parameter_count": sum(value.numel() for value in policy.parameters()),
        "name_count": len(names),
        "names_sha256": canonical_hash(names),
        "allowed_prefixes": list(allowed),
    }


def _reduce_scalar(value: float, context: DistributedContext, operation: dist.ReduceOp) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=context.device)
    if context.world_size > 1:
        dist.all_reduce(tensor, op=operation)
    return float(tensor.item())


def _build_stage_components(
    args: argparse.Namespace,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[
    FunctionalQueryDataset,
    tuple[int, ...],
    SmolVLAPolicy,
    dict[str, Any],
    Any,
    Any,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    dict[int, str],
]:
    authorities, expected_hashes = _load_train_authorities(config, args.data_root)
    task_ids = tuple(item.task_id for item in authorities)
    first_demo, last_demo = config["data"]["demo_indices"]
    dataset = FunctionalQueryDataset(
        authorities,
        demo_indices=range(first_demo, last_demo + 1),
        action_chunk_size=config["features"]["chunk_size"],
        max_open_files_per_worker=config["data"]["max_open_files_per_worker"],
    )
    model_path = args.foundation_path if args.resume is None else args.resume / "policy"
    policy = _build_policy(config, model_path, args.vlm_path, device)
    trainable = _trainable_contract(policy)
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        policy.config, dataset_stats=_load_stats(config, task_ids)
    )
    optimizer = torch.optim.AdamW(
        (value for value in policy.parameters() if value.requires_grad),
        lr=config["optimization"]["peak_lr"],
        betas=tuple(config["optimization"]["betas"]),
        eps=config["optimization"]["eps"],
        weight_decay=config["optimization"]["weight_decay"],
    )
    scheduler = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=config["optimization"]["scheduler_reference_warmup_steps"],
        num_decay_steps=config["optimization"]["scheduler_reference_decay_steps"],
        peak_lr=config["optimization"]["peak_lr"],
        decay_lr=config["optimization"]["decay_lr"],
    ).build(optimizer, args.total_steps)
    return (
        dataset,
        task_ids,
        policy,
        trainable,
        preprocessor,
        postprocessor,
        optimizer,
        scheduler,
        expected_hashes,
    )


def _persist_launch(
    args: argparse.Namespace,
    context: DistributedContext,
    contract: dict[str, Any],
    expected_hashes: dict[int, str],
) -> str:
    contract_hash = canonical_hash(contract)
    if context.is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        contract_path = args.output_dir / "run_contract.json"
        if contract_path.exists() and canonical_hash(read_json(contract_path)) != contract_hash:
            raise SourceBaseError("output directory belongs to a different launch contract")
        write_json_atomic(contract_path, contract)
        write_json_atomic(
            args.output_dir / "runtime_paths.json",
            {
                "data_root": str(args.data_root.resolve()),
                "foundation_path": str(args.foundation_path.resolve()),
                "vlm_path": str(args.vlm_path.resolve()),
                "host": socket.gethostname(),
                "expected_train_hdf5_sha256": expected_hashes,
            },
        )
    barrier(context)
    return contract_hash


def _load_trainer_resume(
    args: argparse.Namespace,
    context: DistributedContext,
    contract_hash: str,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> int:
    if args.resume is None:
        return 0
    state = torch.load(
        args.resume / "trainer_state.pt", map_location=context.device, weights_only=False
    )
    if state["contract_sha256"] != contract_hash:
        raise SourceBaseError("resume checkpoint belongs to a different launch contract")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state["next_step"])


def _load_rank_resume(
    args: argparse.Namespace,
    context: DistributedContext,
    resume_step: int,
) -> dict[str, Any] | None:
    if args.resume is None:
        return None
    state = torch.load(
        args.resume / f"rank_{context.rank:02d}_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    expected = (resume_step, context.rank, context.world_size, args.batch_size)
    actual = (
        state["next_step"],
        state["rank"],
        state["world_size"],
        state["per_rank_batch_size"],
    )
    if actual != expected:
        raise SourceBaseError("rank resume state does not match this topology")
    return state["rng"]


def _build_data_and_ddp(
    args: argparse.Namespace,
    context: DistributedContext,
    config: dict[str, Any],
    dataset: FunctionalQueryDataset,
    task_ids: tuple[int, ...],
    policy: SmolVLAPolicy,
    resume_step: int,
) -> tuple[MixedTaskBatchSampler, Any, torch.nn.Module]:
    sampler = MixedTaskBatchSampler(
        dataset,
        task_ids=task_ids,
        per_rank_batch_size=args.batch_size,
        start_step=resume_step,
        stop_step=args.stop_after_step,
        rank=context.rank,
        world_size=context.world_size,
        seed=config["data"]["sampler_seed"],
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
    policy.to(context.device)
    wrapped: torch.nn.Module = policy
    if context.world_size > 1:
        wrapped = DistributedDataParallel(
            policy,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
            static_graph=True,
        )
    wrapped.train()
    return sampler, iterator, wrapped


def _prepare_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> TrainingRuntime:
    config_path = args.config.resolve()
    config = _load_launch_config(config_path)
    checkpoint_steps = parse_checkpoint_steps(args.checkpoint_steps, args.total_steps)
    validate_launch(config, args, context, checkpoint_steps)
    _seed_everything(config["data"]["sampler_seed"], context)
    current_git = git_state()
    if args.mode == "formal" and current_git["dirty_paths"]:
        raise SourceBaseError("formal launch requires a clean committed worktree")
    components = _build_stage_components(args, config, context.device)
    dataset, task_ids, policy, trainable = components[:4]
    preprocessor, postprocessor, optimizer, scheduler, expected_hashes = components[4:]
    contract = build_contract(
        config_path=config_path,
        config=config,
        args=args,
        context=context,
        checkpoint_steps=checkpoint_steps,
        trainable=trainable,
        git=current_git,
    )
    contract_hash = _persist_launch(args, context, contract, expected_hashes)
    resume_step = _load_trainer_resume(
        args, context, contract_hash, optimizer, scheduler
    )
    if not 0 <= resume_step <= args.stop_after_step <= args.total_steps:
        raise SourceBaseError("invalid resume or stop step")
    sampler, iterator, wrapped = _build_data_and_ddp(
        args, context, config, dataset, task_ids, policy, resume_step
    )
    resume_rng = _load_rank_resume(args, context, resume_step)
    torch.cuda.reset_peak_memory_stats(context.device)
    barrier(context)
    if resume_rng is not None:
        restore_rng(resume_rng, context)
    return TrainingRuntime(
        config,
        checkpoint_steps,
        dataset,
        task_ids,
        policy,
        trainable,
        preprocessor,
        postprocessor,
        optimizer,
        scheduler,
        contract,
        contract_hash,
        resume_step,
        sampler,
        iterator,
        wrapped,
    )


def _log_step(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    runtime: TrainingRuntime,
    step: int,
    loss: torch.Tensor,
    grad_norm: torch.Tensor,
    step_seconds: float,
    data_seconds: float,
) -> None:
    mean_loss = _reduce_scalar(float(loss.detach()), context, dist.ReduceOp.SUM)
    mean_loss /= context.world_size
    slowest_step = _reduce_scalar(step_seconds, context, dist.ReduceOp.MAX)
    slowest_data = _reduce_scalar(data_seconds, context, dist.ReduceOp.MAX)
    peak_allocated = _reduce_scalar(
        torch.cuda.max_memory_allocated(context.device) / 2**30,
        context,
        dist.ReduceOp.MAX,
    )
    peak_reserved = _reduce_scalar(
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
                    "global_samples_per_second": context.world_size * args.batch_size / slowest_step,
                    "peak_allocated_gib_max": peak_allocated,
                    "peak_reserved_gib_max": peak_reserved,
                }
            ),
            flush=True,
        )


def _run_steps(
    args: argparse.Namespace, context: DistributedContext, runtime: TrainingRuntime
) -> None:
    step = runtime.resume_step
    while step < args.stop_after_step:
        tick = time.perf_counter()
        batch = next(runtime.iterator)
        data_seconds = time.perf_counter() - tick
        for camera in ("observation.images.camera1", "observation.images.camera2"):
            batch[camera] = batch[camera].to(dtype=torch.float32).div_(255.0)
        batch = runtime.preprocessor(batch)
        runtime.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = runtime.wrapped(batch)
        if not torch.isfinite(loss):
            raise SourceBaseError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (value for value in runtime.policy.parameters() if value.requires_grad),
            runtime.config["optimization"]["grad_clip_norm"],
        )
        runtime.optimizer.step()
        runtime.scheduler.step()
        torch.cuda.synchronize(context.device)
        step += 1
        step_seconds = time.perf_counter() - tick
        if step % args.log_every == 0 or step == args.stop_after_step:
            _log_step(
                args=args,
                context=context,
                runtime=runtime,
                step=step,
                loss=loss,
                grad_norm=grad_norm,
                step_seconds=step_seconds,
                data_seconds=data_seconds,
            )
        if step in runtime.checkpoint_steps:
            save_checkpoint(
                output_dir=args.output_dir,
                step=step,
                context=context,
                policy=runtime.policy,
                optimizer=runtime.optimizer,
                scheduler=runtime.scheduler,
                preprocessor=runtime.preprocessor,
                postprocessor=runtime.postprocessor,
                sampler=runtime.sampler,
                contract=runtime.contract,
                mode=args.mode,
            )
    if context.is_main:
        print(json.dumps({"event": "complete", "step": step}), flush=True)


def train(args: argparse.Namespace) -> None:
    context = _initialize_distributed()
    try:
        runtime = _prepare_runtime(args, context)
        if context.is_main:
            print(
                json.dumps(
                    {
                        "event": "start",
                        "mode": args.mode,
                        "contract_sha256": runtime.contract_hash,
                        "resume_step": runtime.resume_step,
                        "stop_after_step": args.stop_after_step,
                        "dataset_frames": len(runtime.dataset),
                        "tasks": len(runtime.task_ids),
                        "trainable": runtime.trainable,
                    }
                ),
                flush=True,
            )
        _run_steps(args, context, runtime)
        runtime.dataset.close()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/source_base_v1.json")
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--foundation-path", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--checkpoint-steps", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = _load_launch_config(args.config.resolve())
    if args.num_workers is None:
        args.num_workers = config["loader"]["num_workers_per_rank"]
    if args.stop_after_step is None:
        args.stop_after_step = args.total_steps
    if args.total_steps <= 0 or args.batch_size <= 0 or args.num_workers < 0 or args.log_every <= 0:
        parser.error("steps, batch size, workers, and log frequency are invalid")
    train(args)


if __name__ == "__main__":
    main()
