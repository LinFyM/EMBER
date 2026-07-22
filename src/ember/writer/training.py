"""Canonical eight-rank PI05 Action-Supervised Writer training.

Only the shared Writer is trainable.  It receives pure task language plus one
action-hidden teacher video and generates the complete sealed PI05 task LoRA;
an independently sampled action query from the same development-train task is
used only by the frozen policy's functional behavior loss.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.distributed as dist
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ember.pi05_eval_contract import (
    inspect_source_checkpoint,
    inspect_tokenizer,
    load_evaluation_authorities,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import (
    DistributedContext,
    barrier,
    canonical_hash,
    restore_rng,
    write_json_atomic,
)
from ember.pi05_source_contract import append_jsonl, reconcile_metrics
from ember.pi05_source_setup import (
    initialize_distributed,
    load_policy,
    load_stats,
    reduce_max,
    reduce_mean,
    seed_everything,
)
from ember.writer.checkpoint import load_writer_checkpoint, save_writer_checkpoint
from ember.writer.as_contract import (
    REPO_ROOT,
    authority_path,
    build_contract,
    inspect_feature_cache,
    load_training_data,
    load_writer_config,
    publish_contract,
    resolve_runtime,
    resume_step,
    writer_trainable_contract,
)
from ember.writer.data import (
    FunctionalQueryDataset,
    MixedTaskBatchSampler,
    TeacherVideoSchedule,
)
from ember.writer.feature_cache import (
    WriterFeatureStore,
)
from ember.writer.functional import (
    functional_lora_loss_gradient,
    prepare_frozen_writer_policy,
)
from ember.writer.model import (
    CompleteLoRAWriter,
    WriterModelError,
    build_lora_tensor_specs,
)


@dataclass
class WriterRuntime:
    args: argparse.Namespace
    context: DistributedContext
    config: dict[str, Any]
    dataset: FunctionalQueryDataset
    task_ids: tuple[int, ...]
    sampler: MixedTaskBatchSampler
    video_schedule: TeacherVideoSchedule
    video_partner: dict[int, int]
    iterator: Iterator[dict[str, Any]]
    feature_store: WriterFeatureStore
    processor: Pi05LiberoProcessor
    policy: torch.nn.Module
    writer: CompleteLoRAWriter
    wrapped_writer: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    lora_contract: Any
    contract: dict[str, Any]
    contract_sha256: str
    total_steps: int
    batch_size: int
    checkpoint_steps: tuple[int, ...]
    resume_step: int
    metrics_path: Path
    metrics_rows: int


def _build_writer(
    config: Mapping[str, Any], policy: torch.nn.Module
) -> tuple[CompleteLoRAWriter, Any, dict[str, Any]]:
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    if hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
    if hasattr(policy, "config"):
        policy.config.gradient_checkpointing = False
    template = prepare_frozen_writer_policy(policy, lora)
    writer_config = {
        key: value
        for key, value in config["writer"].items()
        if key != "generated_adapter"
    }
    writer = CompleteLoRAWriter(
        build_lora_tensor_specs(template),
        template_state=template,
        **writer_config,
    )
    return writer, lora, writer_trainable_contract(writer, policy, lora)


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    return CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=int(config["warmup_steps"]),
        num_decay_steps=int(config["decay_steps"]),
        peak_lr=float(config["peak_lr"]),
        decay_lr=float(config["decay_lr"]),
    ).build(optimizer, total_steps)


def _build_trainable_models(
    *,
    config: Mapping[str, Any],
    context: DistributedContext,
    source: Mapping[str, Any],
    source_config: Mapping[str, Any],
    total_steps: int,
) -> tuple[
    torch.nn.Module,
    CompleteLoRAWriter,
    Any,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    dict[str, Any],
]:
    policy = load_policy(Path(source["model_path"]), source_config, context.device)
    writer, lora, trainable = _build_writer(config, policy)
    writer.to(context.device)
    optimizer_config = config["optimization"]["optimizer"]
    optimizer = torch.optim.AdamW(
        writer.parameters(),
        lr=float(config["optimization"]["scheduler"]["peak_lr"]),
        betas=tuple(optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler = _make_scheduler(
        optimizer, config["optimization"]["scheduler"], total_steps
    )
    return policy, writer, lora, optimizer, scheduler, trainable


def _restore_training_state(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    config: Mapping[str, Any],
    writer: CompleteLoRAWriter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    batch_size: int,
    contract_sha256: str,
    initial_step: int,
) -> tuple[dict[str, Any] | None, int]:
    if args.resume is None:
        return None, 0
    loaded, rng, metrics_rows = load_writer_checkpoint(
        checkpoint=args.resume.resolve(),
        context=context,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler_seed=int(config["data"]["sampler_seed"]),
        teacher_video_seed=int(config["data"]["teacher_video_seed"]),
        per_rank_batch_size=batch_size,
        contract_sha256=contract_sha256,
    )
    if loaded != initial_step:
        raise WriterModelError("AS-Writer resume path and state disagree")
    return rng, metrics_rows


def _build_sampler_and_loader(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    config: Mapping[str, Any],
    dataset: FunctionalQueryDataset,
    task_ids: tuple[int, ...],
    batch_size: int,
    initial_step: int,
) -> tuple[MixedTaskBatchSampler, TeacherVideoSchedule, DataLoader[Any]]:
    sampler = MixedTaskBatchSampler(
        dataset,
        task_ids=task_ids,
        per_rank_batch_size=batch_size,
        start_step=initial_step,
        stop_step=args.stop_after_step,
        rank=context.rank,
        world_size=context.world_size,
        seed=int(config["data"]["sampler_seed"]),
    )
    first_demo, last_demo = map(int, config["data"]["demo_indices"])
    schedule = TeacherVideoSchedule(
        task_ids=task_ids,
        demo_indices=range(first_demo, last_demo + 1),
        seed=int(config["data"]["teacher_video_seed"]),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=int(config["loader"]["prefetch_factor"]) if args.num_workers else None,
        generator=torch.Generator().manual_seed(
            int(config["optimization"]["seed"]) + context.rank + 0xA55A
        ),
    )
    return sampler, schedule, loader


def _wrap_writer(writer: CompleteLoRAWriter, context: DistributedContext) -> torch.nn.Module:
    if context.world_size == 1:
        return writer
    return DistributedDataParallel(
        writer,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
        static_graph=True,
    )


def _metrics_cursor(
    path: Path,
    *,
    context: DistributedContext,
    initial_step: int,
    expected_rows: int,
) -> int:
    count = reconcile_metrics(path, initial_step, expected_rows) if context.is_main else 0
    rows = torch.tensor(count, dtype=torch.int64, device=context.device)
    if context.world_size > 1:
        dist.broadcast(rows, src=0)
    return int(rows.item())


def _build_condition_inputs(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    cache: Mapping[str, Any],
    authorities: Any,
    context: DistributedContext,
    task_ids: tuple[int, ...],
) -> tuple[WriterFeatureStore, Pi05LiberoProcessor]:
    store = WriterFeatureStore(
        args.feature_cache.resolve(),
        task_ids=task_ids,
        expected_extraction_sha256=str(cache["extraction_sha256"]),
        max_cached_tasks=int(config["data"]["feature_lru_tasks_per_rank"]),
        expected_dim=int(config["writer"]["vision_feature_dim"]),
        expected_spatial_tokens=int(config["writer"]["vision_spatial_tokens"]),
        expected_run_contract_file_sha256=str(cache["run_contract_file_sha256"]),
        expected_manifest_file_sha256=str(cache["cache_manifest_file_sha256"]),
    )
    processor = Pi05LiberoProcessor(
        load_stats(
            authorities.source_base_config,
            authorities.source_base_config["data"]["active_task_ids"],
        ),
        args.tokenizer_path,
        int(authorities.source_base_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    return store, processor


def _video_partner_map(
    config: Mapping[str, Any], task_ids: tuple[int, ...]
) -> dict[int, int]:
    pairs = tuple(
        tuple(int(value) for value in pair)
        for pair in config["conditioning_training"]["video_task_pairs"]
    )
    result = {
        source: target
        for left, right in pairs
        for source, target in ((left, right), (right, left))
    }
    if set(result) != set(task_ids):
        raise WriterModelError("video-forced task pairs do not cover the train tasks")
    return result


def prepare_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> WriterRuntime:
    config = load_writer_config(args.config.resolve())
    total_steps, batch_size, checkpoint_steps = resolve_runtime(args, config, context)
    initial_step = resume_step(args.resume)
    if not 0 <= initial_step < args.stop_after_step:
        raise WriterModelError("AS-Writer resume cursor is outside this segment")
    seed_everything(int(config["optimization"]["seed"]), context)

    dataset, tasks, data_validation = load_training_data(args, config, context)
    task_ids = tuple(task.task_id for task in tasks)
    authorities = load_evaluation_authorities(
        authority_path(config, "evaluation_config"), REPO_ROOT
    )
    source = inspect_source_checkpoint(
        authorities, args.source_run, args.checkpoint, evaluation_mode="formal"
    )
    tokenizer = inspect_tokenizer(authorities, args.tokenizer_path)
    cache = inspect_feature_cache(
        args.feature_cache.resolve(), config, source, task_ids
    )
    policy, writer, lora_contract, optimizer, scheduler, trainable = (
        _build_trainable_models(
            config=config,
            context=context,
            source=source,
            source_config=authorities.source_base_config,
            total_steps=total_steps,
        )
    )
    contract = build_contract(
        args=args,
        config=config,
        context=context,
        source=source,
        tokenizer=tokenizer,
        cache=cache,
        data_validation=data_validation,
        task_ids=task_ids,
        trainable=trainable,
        total_steps=total_steps,
        batch_size=batch_size,
        checkpoint_steps=checkpoint_steps,
    )
    contract_sha256 = canonical_hash(contract)
    publish_contract(args, context, contract, contract_sha256)

    resume_rng, expected_metrics_rows = _restore_training_state(
        args=args,
        context=context,
        config=config,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=batch_size,
        contract_sha256=contract_sha256,
        initial_step=initial_step,
    )
    sampler, video_schedule, loader = _build_sampler_and_loader(
        args=args,
        context=context,
        config=config,
        dataset=dataset,
        task_ids=task_ids,
        batch_size=batch_size,
        initial_step=initial_step,
    )
    wrapped = _wrap_writer(writer, context)
    writer.train()
    feature_store, processor = _build_condition_inputs(
        args=args,
        config=config,
        cache=cache,
        authorities=authorities,
        context=context,
        task_ids=task_ids,
    )
    metrics_path = args.output_dir / "metrics.jsonl"
    metrics_rows = _metrics_cursor(
        metrics_path,
        context=context,
        initial_step=initial_step,
        expected_rows=expected_metrics_rows,
    )
    torch.cuda.reset_peak_memory_stats(context.device)
    barrier(context)
    if resume_rng is not None:
        restore_rng(resume_rng, context)
    return WriterRuntime(
        args=args,
        context=context,
        config=config,
        dataset=dataset,
        task_ids=task_ids,
        sampler=sampler,
        video_schedule=video_schedule,
        video_partner=_video_partner_map(config, task_ids),
        iterator=iter(loader),
        feature_store=feature_store,
        processor=processor,
        policy=policy,
        writer=writer,
        wrapped_writer=wrapped,
        optimizer=optimizer,
        scheduler=scheduler,
        lora_contract=lora_contract,
        contract=contract,
        contract_sha256=contract_sha256,
        total_steps=total_steps,
        batch_size=batch_size,
        checkpoint_steps=checkpoint_steps,
        resume_step=initial_step,
        metrics_path=metrics_path,
        metrics_rows=metrics_rows,
    )


def _batch_task_id(batch: Mapping[str, Any]) -> int:
    values = batch.get("task_id")
    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise WriterModelError("AS-Writer action batch lost task identity")
    unique = values.unique()
    if unique.numel() != 1:
        raise WriterModelError("one AS-Writer rank received multiple tasks")
    return int(unique.item())


def _pack_writer_conditions(
    language: torch.Tensor,
    correct_video: torch.Tensor,
    wrong_video: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if language.ndim != 2 or correct_video.ndim != 3 or wrong_video.ndim != 3:
        raise WriterModelError("video-forced Writer condition tensors changed shape")
    null_language = torch.zeros(
        1, language.shape[1], dtype=language.dtype, device=language.device
    )
    languages = torch.cat((language, null_language, null_language), dim=0)
    videos = torch.cat((correct_video, correct_video, wrong_video), dim=0)
    language_offsets = torch.tensor(
        [0, language.shape[0], language.shape[0] + 1, language.shape[0] + 2],
        dtype=torch.int64,
    )
    video_offsets = torch.tensor(
        [
            0,
            correct_video.shape[0],
            2 * correct_video.shape[0],
            2 * correct_video.shape[0] + wrong_video.shape[0],
        ],
        dtype=torch.int64,
    )
    return languages, videos, language_offsets, video_offsets


def _generic_policy_batch(
    runtime: WriterRuntime, batch: Mapping[str, Any]
) -> dict[str, Any]:
    tasks = batch.get("task")
    if not isinstance(tasks, (list, tuple)) or not tasks:
        raise WriterModelError("video-forced policy batch lost language rows")
    value = dict(batch)
    prompt = str(runtime.config["conditioning_training"]["generic_policy_prompt"])
    value["task"] = [prompt] * len(tasks)
    return runtime.processor.training_batch(value)


def _adapter_state_at(
    generated: Mapping[str, torch.Tensor], index: int
) -> dict[str, torch.Tensor]:
    result = {}
    for name, value in generated.items():
        if value.ndim != 3 or value.shape[0] != 3:
            raise WriterModelError("video-forced Writer did not generate three adapters")
        result[name] = value[index]
    return result


def _matching_objective(
    losses: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    normal, forced_correct, forced_wrong = losses
    normal_weight = float(config["normal_loss_weight"])
    forced_weight = float(config["video_forced_loss_weight"])
    matching_weight = float(config["matching_loss_weight"])
    margin = float(config["matching_margin"])
    temperature = float(config["matching_temperature"])
    z = (margin + forced_correct - forced_wrong) / temperature
    probability = torch.sigmoid(z)
    objective = (
        normal_weight * normal
        + forced_weight * forced_correct
        + matching_weight * temperature * torch.nn.functional.softplus(z)
    )
    coefficients = (
        torch.as_tensor(normal_weight, device=normal.device),
        torch.as_tensor(forced_weight, device=normal.device) + matching_weight * probability,
        -matching_weight * probability,
    )
    return objective, coefficients, probability


def _differentiate_condition_batch(
    runtime: WriterRuntime,
    packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    policy_batch: Mapping[str, Any],
    forced_policy_batch: Mapping[str, Any],
) -> tuple[torch.Tensor, list[torch.Tensor], list[Mapping[str, Any]], torch.Tensor]:
    runtime.optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated = runtime.wrapped_writer(
            packed[0], packed[1], packed[3], language_offsets=packed[2]
        )
        values: list[torch.Tensor] = []
        gradients: list[dict[str, torch.Tensor]] = []
        details: list[Mapping[str, Any]] = []
        for index, policy_value in enumerate(
            (policy_batch, forced_policy_batch, forced_policy_batch)
        ):
            value, detail, gradient = functional_lora_loss_gradient(
                runtime.policy,
                _adapter_state_at(generated, index),
                runtime.lora_contract,
                batch=policy_value,
            )
            values.append(value)
            details.append(detail)
            gradients.append(gradient)
        loss, coefficients, probability = _matching_objective(
            (values[0], values[1], values[2]),
            runtime.config["conditioning_training"],
        )
    names = tuple(generated)
    gradient_tensors = tuple(
        torch.stack(
            [coefficients[index].to(gradients[index][name]) * gradients[index][name]
             for index in range(3)],
            dim=0,
        )
        for name in names
    )
    torch.autograd.backward(tuple(generated[name] for name in names), gradient_tensors)
    return loss, values, details, probability


def _one_step(
    runtime: WriterRuntime,
    step: int,
    started: float,
) -> dict[str, Any]:
    tick = time.monotonic()
    batch = next(runtime.iterator)
    data_seconds = time.monotonic() - tick
    task_id, task_visit = runtime.sampler.task_visit_for_step(step)
    if _batch_task_id(batch) != task_id:
        raise WriterModelError("AS-Writer sampler and action batch disagree")
    demo_index = runtime.video_schedule.demo_for_task_visit(task_id, task_visit)
    partner_id = runtime.video_partner[task_id]
    wrong_demo_index = runtime.video_schedule.demo_for_task_visit(partner_id, task_visit)
    teacher = runtime.feature_store.load_one_video(
        language_task_id=task_id,
        video_task_id=task_id,
        demo_index=demo_index,
    )
    wrong_teacher = runtime.feature_store.load_one_video(
        language_task_id=task_id,
        video_task_id=partner_id,
        demo_index=wrong_demo_index,
    )
    language = teacher.language_features.to(runtime.context.device)
    video = teacher.video_features.to(runtime.context.device)
    wrong_video = wrong_teacher.video_features.to(runtime.context.device)
    policy_batch = runtime.processor.training_batch(batch)
    forced_policy_batch = _generic_policy_batch(runtime, batch)
    packed = _pack_writer_conditions(language, video, wrong_video)

    loss, values, details, matching_probability = _differentiate_condition_batch(
        runtime, packed, policy_batch, forced_policy_batch
    )
    if not bool(torch.isfinite(loss)):
        raise WriterModelError(f"non-finite AS-Writer loss at step {step}")
    if any(parameter.grad is not None for parameter in runtime.policy.parameters()):
        raise WriterModelError("frozen PI05 source policy accumulated gradients")
    grad_norm = torch.nn.utils.clip_grad_norm_(
        runtime.writer.parameters(),
        float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"]),
    )
    if not bool(torch.isfinite(grad_norm).detach()):
        raise WriterModelError(f"non-finite AS-Writer gradient at step {step}")
    applied_lr = float(runtime.optimizer.param_groups[0]["lr"])
    runtime.optimizer.step()
    runtime.scheduler.step()
    completed = step + 1
    step_seconds = reduce_max(time.monotonic() - tick, runtime.context)
    return {
        "optimizer_step": completed,
        "mean_functional_action_loss": reduce_mean(float(loss), runtime.context),
        "mean_normal_action_loss": reduce_mean(float(values[0]), runtime.context),
        "mean_video_forced_correct_loss": reduce_mean(float(values[1]), runtime.context),
        "mean_video_forced_wrong_loss": reduce_mean(float(values[2]), runtime.context),
        "mean_matching_probability": reduce_mean(float(matching_probability), runtime.context),
        "gradient_norm_before_clip_max": reduce_max(float(grad_norm), runtime.context),
        "applied_lr": applied_lr,
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "global_action_queries": completed
        * runtime.context.world_size
        * runtime.batch_size,
        "global_writer_invocations": completed * runtime.context.world_size * 3,
        "global_functional_query_evaluations": completed
        * runtime.context.world_size
        * runtime.batch_size
        * 3,
        "rank0_task_id": task_id,
        "rank0_teacher_demo_index": demo_index,
        "rank0_wrong_video_task_id": partner_id,
        "rank0_wrong_teacher_demo_index": wrong_demo_index,
        "rank0_policy_loss_detail": [value.get("loss") for value in details],
        "data_seconds_max": reduce_max(data_seconds, runtime.context),
        "step_seconds_max": step_seconds,
        "global_action_queries_per_second": runtime.context.world_size
        * runtime.batch_size
        / step_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": int(
            reduce_max(
                torch.cuda.max_memory_allocated(runtime.context.device), runtime.context
            )
        ),
        "max_cuda_reserved_bytes": int(
            reduce_max(
                torch.cuda.max_memory_reserved(runtime.context.device), runtime.context
            )
        ),
    }


def run_steps(runtime: WriterRuntime) -> None:
    started = time.monotonic()
    for step in range(runtime.resume_step, runtime.args.stop_after_step):
        row = _one_step(runtime, step, started)
        completed = int(row["optimizer_step"])
        if runtime.context.is_main:
            append_jsonl(runtime.metrics_path, row)
            runtime.metrics_rows += 1
            if completed == 1 or completed % runtime.args.log_every == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
        if completed in runtime.checkpoint_steps:
            save_writer_checkpoint(
                output_dir=runtime.args.output_dir,
                step=completed,
                context=runtime.context,
                writer=runtime.writer,
                optimizer=runtime.optimizer,
                scheduler=runtime.scheduler,
                sampler=runtime.sampler,
                video_schedule=runtime.video_schedule,
                contract=runtime.contract,
                mode=runtime.args.mode,
                metrics_rows=runtime.metrics_rows,
            )
    barrier(runtime.context)
    if runtime.context.is_main:
        stop = runtime.args.stop_after_step
        write_json_atomic(
            runtime.args.output_dir / "run_summary.json",
            {
                "schema_version": "ember_pi05_as_writer_run_summary_v1",
                "contract_sha256": runtime.contract_sha256,
                "completed_optimizer_steps": stop,
                "requested_optimizer_steps": runtime.total_steps,
                "stopped_early_for_profile": stop < runtime.total_steps,
                "metrics_rows": runtime.metrics_rows,
                "wall_seconds": time.monotonic() - started,
                "final_checkpoint": str(
                    runtime.args.output_dir / "checkpoints" / f"step_{stop:08d}"
                )
                if stop in runtime.checkpoint_steps
                else None,
                "train_tasks": len(runtime.task_ids),
                "teacher_action_episodes_available": len(runtime.task_ids) * 50,
                "validation_action_reads": 0,
                "test_action_reads": 0,
                "test_video_value_reads": 0,
            },
        )


def train(args: argparse.Namespace) -> None:
    context = initialize_distributed(require_numa=args.mode == "formal")
    runtime: WriterRuntime | None = None
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
                        "tasks": len(runtime.task_ids),
                        "trainable": runtime.contract["trainable"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        run_steps(runtime)
    finally:
        if runtime is not None:
            runtime.dataset.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_as_writer_v2.json",
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--checkpoint-steps", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--skip-data-sha", action="store_true")
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    config = load_writer_config(args.config.resolve())
    if args.num_workers is None:
        args.num_workers = int(config["loader"]["num_workers_per_rank"])
    if args.num_workers < 0 or args.log_every <= 0:
        raise WriterModelError("invalid AS-Writer loader or logging request")
    for name in (
        "config",
        "source_run",
        "checkpoint",
        "tokenizer_path",
        "feature_cache",
        "data_root",
        "output_dir",
        "resume",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    return args
