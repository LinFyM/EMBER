"""Matched action-supervised task-local LoRA training on validation tasks."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
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
from torch.utils.data import DataLoader

from ember.direct_lora_checkpoint import (
    DirectLoRACheckpointError,
    load_direct_lora_checkpoint,
    restore_task_rng,
    save_direct_lora_checkpoint,
    verify_checkpoint_files,
)
from ember.direct_lora_protocol import (
    REPO_ROOT,
    DirectContext,
    DirectLoRAError,
    build_run_contract,
    load_direct_lora_config,
    load_tasks,
    persist_run_contract,
    policy_files,
    task_assignments,
    validate_launch,
)
from ember.lora import (
    SmolVLALoRAContract,
    canonical_contract_sha256,
    initialize_identity_lora_,
    inject_task_lora,
    load_lora_contract,
    task_lora_state_dict,
)
from ember.source_base_checkpoint import (
    canonical_hash,
    parse_checkpoint_steps,
    read_json,
    write_json_atomic,
)
from ember.writer.data import FunctionalQueryDataset, MixedTaskBatchSampler
from ember.writer.feature_cache import (
    FeatureCacheTask,
)


def _initialize_distributed() -> DirectContext:
    if not torch.cuda.is_available():
        raise DirectLoRAError("direct-LoRA training requires CUDA")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not 0 <= local_rank < torch.cuda.device_count():
        raise DirectLoRAError("LOCAL_RANK is outside visible CUDA devices")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    return DirectContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
    )


def _seed_task(seed: int, task_id: int, device: torch.device) -> None:
    task_seed = seed + task_id * 10_000
    random.seed(task_seed)
    np.random.seed(task_seed)
    torch.manual_seed(task_seed)
    torch.cuda.manual_seed(task_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats(device)


def _build_policy(
    policy_path: Path,
    device: torch.device,
    contract: SmolVLALoRAContract,
) -> tuple[SmolVLAPolicy, Any, dict[str, Any]]:
    policy_config = PreTrainedConfig.from_pretrained(policy_path)
    if not isinstance(policy_config, SmolVLAConfig):
        raise DirectLoRAError("selected source checkpoint is not SmolVLA")
    policy_config.device = str(device)
    policy_config.pretrained_path = policy_path
    policy_config.use_amp = False
    policy = SmolVLAPolicy.from_pretrained(policy_path, config=policy_config)
    inject_task_lora(policy, contract)
    state = task_lora_state_dict(policy)
    trainable_names = sorted(
        name for name, value in policy.named_parameters() if value.requires_grad
    )
    if set(trainable_names) != set(state):
        raise DirectLoRAError("direct SFT left trainable state outside task LoRA")
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    trainable = {
        "parameter_count": sum(value.numel() for value in state.values()),
        "name_count": len(trainable_names),
        "names_sha256": canonical_hash(trainable_names),
        "lora_contract_sha256": canonical_contract_sha256(contract),
    }
    return policy, preprocessor, trainable


def _latest_checkpoint(task_dir: Path) -> Path | None:
    latest_path = task_dir / "latest_checkpoint.json"
    if not latest_path.is_file():
        return None
    latest = read_json(latest_path)
    checkpoint = Path(str(latest.get("path", "")))
    verify_checkpoint_files(checkpoint)
    return checkpoint


def _train_task(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    context: DirectContext,
    task: FeatureCacheTask,
    policy: SmolVLAPolicy,
    preprocessor: Any,
    lora_contract: SmolVLALoRAContract,
    run_contract_sha256: str,
    checkpoint_steps: tuple[int, ...],
) -> dict[str, Any]:
    started = time.monotonic()
    task_dir = args.output_dir / "tasks" / f"task_{task.task_id:03d}"
    task_contract = {
        "schema_version": "ember_direct_lora_task_v1",
        "run_contract_sha256": run_contract_sha256,
        "task_id": task.task_id,
        "language": task.language,
        "expected_hdf5_sha256": task.expected_hdf5_sha256,
    }
    task_contract_sha256 = canonical_hash(task_contract)
    task_contract_path = task_dir / "task_contract.json"
    if args.resume:
        if task_contract_path.is_file() and (
            canonical_hash(read_json(task_contract_path)) != task_contract_sha256
        ):
            raise DirectLoRAError(f"task {task.task_id} resume contract changed")
        if not task_contract_path.is_file():
            if task_dir.exists() and any(task_dir.iterdir()):
                raise DirectLoRAError(
                    f"task {task.task_id} has state without a task contract"
                )
            write_json_atomic(task_contract_path, task_contract)
    else:
        if task_dir.exists() and any(task_dir.iterdir()):
            raise DirectLoRAError(f"task output is not empty: {task_dir}")
        write_json_atomic(task_contract_path, task_contract)

    _seed_task(int(config["data"]["sampler_seed"]), task.task_id, context.device)
    initialize_identity_lora_(policy, lora_contract)
    policy.zero_grad(set_to_none=True)
    policy.train()
    optimizer = torch.optim.AdamW(
        task_lora_state_dict(policy).values(),
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
    first_demo, last_demo = config["data"]["demo_indices"]
    dataset = FunctionalQueryDataset(
        [task.authority],
        demo_indices=range(first_demo, last_demo + 1),
        action_chunk_size=int(config["data"]["action_chunk_size"]),
        max_open_files_per_worker=int(
            config["data"]["max_open_files_per_worker"]
        ),
    )
    sampler_seed = int(config["data"]["sampler_seed"]) + task.task_id
    resume_checkpoint = _latest_checkpoint(task_dir) if args.resume else None
    resume_step = 0
    if resume_checkpoint is not None:
        preview = torch.load(
            resume_checkpoint / "trainer_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        resume_step = int(preview["next_step"])
    if not 0 <= resume_step <= args.stop_after_step:
        raise DirectLoRAError(f"task {task.task_id} resume step is invalid")
    if resume_step == args.stop_after_step:
        dataset.close()
        return {
            "task_id": task.task_id,
            "rank": context.rank,
            "step": resume_step,
            "status": (
                "complete"
                if resume_step == args.total_steps
                else "already_at_segment_end"
            ),
            "wall_seconds": time.monotonic() - started,
        }
    sampler = MixedTaskBatchSampler(
        dataset,
        task_ids=(task.task_id,),
        per_rank_batch_size=args.batch_size,
        start_step=resume_step,
        stop_step=args.stop_after_step,
        rank=0,
        world_size=1,
        seed=sampler_seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=(
            int(config["loader"]["prefetch_factor"]) if args.num_workers else None
        ),
    )
    iterator = iter(dataloader)
    if resume_checkpoint is not None:
        loaded_step, resume_rng = load_direct_lora_checkpoint(
            checkpoint=resume_checkpoint,
            task_id=task.task_id,
            policy=policy,
            contract=lora_contract,
            optimizer=optimizer,
            scheduler=scheduler,
            task_contract_sha256=task_contract_sha256,
            per_rank_batch_size=args.batch_size,
            sampler_seed=sampler_seed,
            device=context.device,
        )
        if loaded_step != resume_step:
            raise DirectLoRAError("direct-LoRA resume previews disagree")
        restore_task_rng(resume_rng, context.device)

    step = resume_step
    last_loss = float("nan")
    while step < args.stop_after_step:
        tick = time.perf_counter()
        batch = next(iterator)
        data_seconds = time.perf_counter() - tick
        if set(int(value) for value in batch["task_id"].tolist()) != {task.task_id}:
            raise DirectLoRAError("direct-LoRA batch crossed task authority")
        for camera in ("observation.images.camera1", "observation.images.camera2"):
            batch[camera] = batch[camera].to(dtype=torch.float32).div_(255.0)
        batch = preprocessor(batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = policy(batch)
        if not bool(torch.isfinite(loss).detach()):
            raise DirectLoRAError(f"non-finite direct-LoRA loss for task {task.task_id}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            task_lora_state_dict(policy).values(),
            float(config["optimization"]["grad_clip_norm"]),
        )
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize(context.device)
        step += 1
        last_loss = float(loss.detach())
        step_seconds = time.perf_counter() - tick
        if step % args.log_every == 0 or step == args.stop_after_step:
            print(
                json.dumps(
                    {
                        "event": "train",
                        "rank": context.rank,
                        "task_id": task.task_id,
                        "step": step,
                        "loss": last_loss,
                        "grad_norm": float(grad_norm),
                        "lr": scheduler.get_last_lr()[0],
                        "step_seconds": step_seconds,
                        "data_seconds": data_seconds,
                        "queries_per_second": args.batch_size / step_seconds,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(
                            context.device
                        )
                        / 2**30,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved(
                            context.device
                        )
                        / 2**30,
                    }
                ),
                flush=True,
            )
        if step in checkpoint_steps:
            checkpoint = save_direct_lora_checkpoint(
                task_dir=task_dir,
                task_id=task.task_id,
                step=step,
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                task_contract_sha256=task_contract_sha256,
                device=context.device,
                formal=args.mode == "formal",
            )
            print(
                json.dumps(
                    {
                        "event": "checkpoint",
                        "rank": context.rank,
                        "task_id": task.task_id,
                        "step": step,
                        "path": str(checkpoint),
                    }
                ),
                flush=True,
            )
    dataset.close()
    result = {
        "task_id": task.task_id,
        "rank": context.rank,
        "step": step,
        "status": "complete" if step == args.total_steps else "segment_complete",
        "last_loss": last_loss,
        "wall_seconds": time.monotonic() - started,
        "latest_checkpoint": str(_latest_checkpoint(task_dir)),
    }
    write_json_atomic(task_dir / "task_result.json", result)
    return result


def train(args: argparse.Namespace) -> None:
    context = _initialize_distributed()
    try:
        config_path = args.config.resolve()
        config = load_direct_lora_config(config_path)
        all_tasks = load_tasks(config, args.data_root.resolve())
        tasks = all_tasks
        if args.mode == "profile":
            tasks = tasks[: int(config["profile"]["task_count"])]
        task_ids = tuple(task.task_id for task in tasks)
        checkpoint_steps = parse_checkpoint_steps(
            args.checkpoint_steps, args.total_steps
        )
        validate_launch(
            args=args,
            config=config,
            context=context,
            task_ids=task_ids,
            checkpoint_steps=checkpoint_steps,
        )
        selection = read_json(REPO_ROOT / config["protocol"]["source_selection"])
        source_policy_files = policy_files(args.policy_path.resolve(), selection)
        lora_contract = load_lora_contract(
            REPO_ROOT / config["protocol"]["lora_contract"]
        )
        policy, preprocessor, trainable = _build_policy(
            args.policy_path.resolve(), context.device, lora_contract
        )
        assignments = task_assignments(task_ids, context.world_size)
        run_contract = build_run_contract(
            args=args,
            config_path=config_path,
            config=config,
            context=context,
            tasks=tasks,
            assignments=assignments,
            checkpoint_steps=checkpoint_steps,
            source_policy_files=source_policy_files,
            trainable=trainable,
        )
        run_contract_sha256 = persist_run_contract(
            args=args, context=context, contract=run_contract
        )
        by_id = {task.task_id: task for task in tasks}
        results = []
        for task_id in assignments[context.rank]:
            results.append(
                _train_task(
                    args=args,
                    config=config,
                    context=context,
                    task=by_id[task_id],
                    policy=policy,
                    preprocessor=preprocessor,
                    lora_contract=lora_contract,
                    run_contract_sha256=run_contract_sha256,
                    checkpoint_steps=checkpoint_steps,
                )
            )
        rank_result = {
            "rank": context.rank,
            "local_rank": context.local_rank,
            "device": str(context.device),
            "assigned_task_ids": list(assignments[context.rank]),
            "results": results,
        }
        write_json_atomic(
            args.output_dir / f"rank_{context.rank:02d}.json", rank_result
        )
        dist.barrier()
        if context.rank == 0:
            rank_results = [
                read_json(args.output_dir / f"rank_{rank:02d}.json")
                for rank in range(context.world_size)
            ]
            completed = [
                result
                for rank_result in rank_results
                for result in rank_result["results"]
            ]
            summary = {
                "schema_version": "ember_direct_lora_sft_summary_v1",
                "mode": args.mode,
                "run_contract_sha256": run_contract_sha256,
                "task_ids": list(task_ids),
                "stop_after_step": args.stop_after_step,
                "results": sorted(completed, key=lambda item: item["task_id"]),
            }
            if args.mode == "formal" and args.stop_after_step == args.total_steps:
                if (
                    len(completed) != len(task_ids)
                    or any(result["status"] != "complete" for result in completed)
                ):
                    raise DirectLoRAError("formal direct-LoRA task set is incomplete")
            write_json_atomic(args.output_dir / "run_summary.json", summary)
            print(
                json.dumps(
                    {
                        "event": "complete",
                        "tasks": len(completed),
                        "stop_after_step": args.stop_after_step,
                    }
                ),
                flush=True,
            )
        dist.barrier()
    except DirectLoRACheckpointError as error:
        raise DirectLoRAError(str(error)) from error
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/direct_lora_sft_v1.json"
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--checkpoint-steps", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_direct_lora_config(args.config.resolve())
    if args.stop_after_step is None:
        args.stop_after_step = args.total_steps
    if args.num_workers is None:
        args.num_workers = int(config["loader"]["num_workers_per_rank"])
    train(args)
