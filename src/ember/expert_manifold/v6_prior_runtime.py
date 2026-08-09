"""Asset loading and exact-resume runtime for the residual Writer."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import h5py
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from ember.expert_manifold.contract import (
    ExpertManifoldError,
    ExpertTask,
    load_task_expert_config,
    load_train_tasks,
)
from ember.expert_manifold.v6_prior import (
    V6PriorOwnership,
    V6PriorWarmStart,
    freeze_v6_prior_writer,
    load_v6_prior_warm_start_,
)
from ember.expert_manifold.v6_prior_checkpoint import load_v6_prior_checkpoint
from ember.expert_manifold.v6_prior_contract import (
    REPO_ROOT,
    authority_path,
    load_v6_prior_config,
    runtime_for_mode,
)
from ember.expert_manifold.v6_prior_run_contract import (
    build_run_contract,
    checkpoint_contract,
    cursor_contract,
    publish_contract,
    residual_git_state,
)
from ember.lora import LoRAContract
from ember.pi05_eval_contract import (
    inspect_source_checkpoint,
    inspect_tokenizer,
    load_evaluation_authorities,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import DistributedContext, read_json
from ember.pi05_source_contract import reconcile_metrics
from ember.pi05_source_setup import (
    initialize_deferred_process_group,
    load_policy,
    load_stats,
    seed_everything,
)
from ember.writer.architecture import LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS
from ember.writer.as_sampling import MixedTaskBatchSampler, TeacherVideoSchedule
from ember.writer.condition_update import (
    FrozenV6ConditionResidualWriter,
    validate_frozen_v6_residual_writer,
)
from ember.writer.data import FunctionalQueryDataset, RawTeacherVideoStore
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.model import CompleteLoRAWriter, build_lora_tensor_specs


_RESUME_NAME = re.compile(r"macro_([0-9]{8})")


@dataclass(frozen=True)
class RuntimeSegment:
    total_macros: int
    checkpoint_macros: tuple[int, ...]
    start_macro: int
    stop_macro: int
    schedule_origin: int

    @property
    def schedule_start_macro(self) -> int:
        return self.schedule_origin + self.start_macro

    @property
    def schedule_stop_macro(self) -> int:
        return self.schedule_origin + self.stop_macro


@dataclass
class V6PriorRuntime:
    args: argparse.Namespace
    context: DistributedContext
    config: dict[str, Any]
    segment: RuntimeSegment
    source: dict[str, Any]
    tokenizer: dict[str, Any]
    tasks: tuple[ExpertTask, ...]
    task_by_global_id: dict[int, ExpertTask]
    dataset: FunctionalQueryDataset
    sampler: MixedTaskBatchSampler
    video_schedule: TeacherVideoSchedule
    iterator: Iterator[dict[str, Any]]
    video_store: RawTeacherVideoStore
    language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    processor: Pi05LiberoProcessor
    policy: torch.nn.Module
    writer: FrozenV6ConditionResidualWriter
    lora_contract: LoRAContract
    warm_start: V6PriorWarmStart
    ownership: V6PriorOwnership
    run_contract: dict[str, Any]
    checkpoint_contract: dict[str, Any]
    metrics_path: Path


def _resume_macro(path: Path | None) -> int:
    if path is None:
        return 0
    match = _RESUME_NAME.fullmatch(path.name)
    if match is None or path.parent.name != "checkpoints":
        raise ExpertManifoldError("residual Writer resume path is not a macro checkpoint")
    return int(match.group(1))


def _resolve_segment(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> RuntimeSegment:
    total, checkpoints, schedule_origin = runtime_for_mode(config, args.mode)
    start = _resume_macro(args.resume)
    stop = int(args.stop_after_macro or total)
    selected = (
        config["profile_run"] if args.mode == "mechanism-profile" else config["formal_run"]
    )
    profile_valid = (
        args.mode != "mechanism-profile"
        or (args.resume is None and args.stop_after_macro in {None, 1} and stop == 1)
    )
    formal_valid = (
        args.mode != "formal"
        or (
            args.stop_after_macro is not None
            and (start, stop) in {(0, 10), (10, 25), (25, 50)}
        )
    )
    state = residual_git_state(REPO_ROOT)
    if args.resume is None:
        git_valid = state["commit"] == state["authority_commit"]
    else:
        try:
            stored = read_json(args.resume.parent.parent / "run_contract.json")
            resume_commit = stored["git"]["commit"]
        except Exception as error:
            raise ExpertManifoldError(
                "residual Writer resume lacks its original Git authority"
            ) from error
        git_valid = (
            isinstance(resume_commit, str)
            and bool(resume_commit)
            and state["commit"] == resume_commit
            and state.get("authority_contains_commit") is True
        )
    valid = (
        context.world_size == int(selected["expected_world_size"])
        and 24 % context.world_size == 0
        and 24 // context.world_size == int(selected["tasks_per_rank"])
        and args.num_workers == int(selected["num_workers_per_rank"])
        and 0 <= start < stop <= total
        and profile_valid
        and formal_valid
        and not state["dirty_paths"]
        and git_valid
    )
    if not valid:
        raise ExpertManifoldError("residual Writer runtime differs from its sealed segment")
    return RuntimeSegment(
        total_macros=total,
        checkpoint_macros=checkpoints,
        start_macro=start,
        stop_macro=stop,
        schedule_origin=schedule_origin,
    )


def _validate_collective_environment(context: DistributedContext) -> None:
    if context.world_size <= 1:
        return
    expected = {
        "NCCL_P2P_DISABLE": "1",
        "NCCL_ALGO": "Ring",
        "NCCL_PROTO": "Simple",
    }
    if {name: os.environ.get(name) for name in expected} != expected:
        raise ExpertManifoldError("residual Writer collective environment changed")


def _load_source(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    authorities = load_evaluation_authorities(
        authority_path(config, "evaluation_config"), REPO_ROOT
    )
    source = inspect_source_checkpoint(
        authorities,
        args.source_run,
        args.checkpoint,
        evaluation_mode="formal",
    )
    tokenizer = inspect_tokenizer(authorities, args.tokenizer_path)
    return authorities, source, tokenizer


def _sampled_video_cost(raw_frames: int, stride: int) -> int:
    if raw_frames <= 0 or stride <= 0:
        raise ExpertManifoldError("invalid teacher video length")
    count = len(range(0, raw_frames, stride))
    return count if (raw_frames - 1) % stride == 0 else count + 1


def _video_costs(
    tasks: Sequence[ExpertTask],
    *,
    demo_indices: Sequence[int],
    frame_stride: int,
) -> dict[int, dict[int, int]]:
    costs: dict[int, dict[int, int]] = {}
    for task in tasks:
        rows = {}
        with h5py.File(task.authority.path, "r") as handle:
            for demo_index in demo_indices:
                pixels = handle.get(f"data/demo_{demo_index}/obs/agentview_rgb")
                if not isinstance(pixels, h5py.Dataset) or pixels.ndim != 4:
                    raise ExpertManifoldError("teacher video metadata changed")
                rows[int(demo_index)] = _sampled_video_cost(
                    int(pixels.shape[0]), frame_stride
                )
        costs[task.global_task_id] = rows
    return costs


def _build_data(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    segment: RuntimeSegment,
) -> tuple[
    tuple[ExpertTask, ...],
    FunctionalQueryDataset,
    MixedTaskBatchSampler,
    TeacherVideoSchedule,
    DataLoader[Any],
]:
    expert_config = load_task_expert_config(
        authority_path(config, "task_expert_config")
    )
    tasks = load_train_tasks(expert_config, args.data_root)
    data = config["data"]
    first, last = map(int, data["demo_indices"])
    demos = tuple(range(first, last + 1))
    dataset = FunctionalQueryDataset(
        [task.authority for task in tasks],
        demo_indices=demos,
        action_chunk_size=int(data["action_chunk_size"]),
        max_open_files_per_worker=8,
    )
    task_ids = tuple(task.global_task_id for task in tasks)
    schedule = TeacherVideoSchedule(
        task_ids=task_ids,
        demo_indices=demos,
        seed=int(data["teacher_video_seed"]),
        videos_per_visit=1,
    )
    costs = _video_costs(
        tasks,
        demo_indices=demos,
        frame_stride=int(config["writer"]["frame_stride"]),
    )
    sampler = MixedTaskBatchSampler(
        dataset,
        task_ids=task_ids,
        per_rank_batch_size=int(data["action_queries_per_task"]),
        start_step=segment.schedule_start_macro,
        stop_step=segment.schedule_stop_macro,
        rank=context.rank,
        world_size=context.world_size,
        seed=int(data["sampler_seed"]),
        tasks_per_rank_per_update=24 // context.world_size,
        video_schedule=schedule,
        task_video_costs=costs,
        assignment_strategy="cost_balanced_long_first",
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers else None,
        multiprocessing_context="spawn" if args.num_workers else None,
        generator=torch.Generator().manual_seed(
            int(config["optimization"]["seed"]) + context.rank + 0xA55A
        ),
    )
    return tasks, dataset, sampler, schedule, loader


def _build_policy_writer(
    *,
    config: Mapping[str, Any],
    context: DistributedContext,
    source: Mapping[str, Any],
    source_config: Mapping[str, Any],
) -> tuple[
    torch.nn.Module,
    FrozenV6ConditionResidualWriter,
    LoRAContract,
    V6PriorWarmStart,
    V6PriorOwnership,
]:
    policy = load_policy(Path(str(source["model_path"])), source_config, context.device)
    if hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
    if hasattr(policy, "config"):
        policy.config.gradient_checkpointing = False
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    template = prepare_frozen_writer_policy(policy, lora)
    bridge = policy.model.paligemma_with_expert
    writer_config = {
        name: value
        for name, value in config["writer"].items()
        if name in LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS
    }
    base = CompleteLoRAWriter(
        build_lora_tensor_specs(template),
        template_state=template,
        paligemma_model=bridge.paligemma.model.language_model,
        expert_model=bridge.gemma_expert.model,
        **writer_config,
    )
    base_checkpoint = (
        REPO_ROOT / str(config["initialization"]["checkpoint"])
    ).resolve()
    warm_start = load_v6_prior_warm_start_(base, base_checkpoint)
    if any(
        not torch.equal(
            value.detach().cpu(), template[name].detach().cpu().to(value.dtype)
        )
        for name, value in base.template_state().items()
    ):
        raise ExpertManifoldError("historical v6 load changed physical identity")
    ownership = freeze_v6_prior_writer(base)
    feature = config["condition_feature"]
    writer = FrozenV6ConditionResidualWriter(
        base,
        feature_width=int(feature["feature_width"]),
        feature_seed=int(feature["projection_seed"]),
    ).to(context.device)
    validate_frozen_v6_residual_writer(writer, require_zero_memory=True)
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise ExpertManifoldError("residual Writer source policy is not frozen")
    return policy, writer, lora, warm_start, ownership


def _build_language_inputs(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    source_config: Mapping[str, Any],
    tasks: Sequence[ExpertTask],
) -> tuple[
    RawTeacherVideoStore,
    Pi05LiberoProcessor,
    dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    max_length = int(source_config["features"]["tokenizer_max_length"])
    processor = Pi05LiberoProcessor(
        load_stats(source_config, source_config["data"]["active_task_ids"]),
        args.tokenizer_path,
        max_length,
        str(context.device),
    )
    tokenizer = Pi05TeacherPrefixTokenizer(
        args.tokenizer_path, max_length, str(context.device)
    )
    language = {task.global_task_id: tokenizer((task.language,)) for task in tasks}
    store = RawTeacherVideoStore(
        [task.authority for task in tasks],
        frame_stride=int(config["writer"]["frame_stride"]),
        max_open_files=4,
    )
    return store, processor, language


def _reconcile_metrics_cursor(
    path: Path,
    *,
    context: DistributedContext,
    expected_rows: int,
) -> int:
    payload: list[Any] = [None]
    if context.is_main:
        try:
            payload[0] = {
                "rows": reconcile_metrics(
                    path,
                    expected_rows,
                    expected_rows,
                    cursor_key="macro",
                )
            }
        except Exception as error:
            payload[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=context.device)
    result = payload[0]
    if (
        not isinstance(result, Mapping)
        or "error" in result
        or type(result.get("rows")) is not int
    ):
        raise ExpertManifoldError(
            "residual Writer metrics differ from resume cursor: "
            f"{result}"
        )
    return int(result["rows"])


def _restore_resume(
    runtime_args: argparse.Namespace,
    config: Mapping[str, Any],
    segment: RuntimeSegment,
    context: DistributedContext,
    writer: FrozenV6ConditionResidualWriter,
    checkpoint_contract_value: Mapping[str, Any],
) -> None:
    if runtime_args.resume is None:
        return
    loaded, rows = load_v6_prior_checkpoint(
        checkpoint=runtime_args.resume,
        memory=writer.program_memory,
        context=context,
        expected_cursor_contract=cursor_contract(config, segment.start_macro),
        expected_checkpoint_contract=checkpoint_contract_value,
    )
    if loaded != segment.start_macro or rows != segment.start_macro:
        raise ExpertManifoldError("residual Writer resume cursor changed")
    validate_frozen_v6_residual_writer(writer, require_zero_memory=False)


def _prepare_runtime(
    args: argparse.Namespace,
    context: DistributedContext,
) -> V6PriorRuntime:
    config = load_v6_prior_config(args.config)
    segment = _resolve_segment(args, config, context)
    seed_everything(int(config["optimization"]["seed"]), context)
    authorities, source, tokenizer = _load_source(args, config)
    tasks, dataset, sampler, schedule, loader = _build_data(
        args=args,
        config=config,
        context=context,
        segment=segment,
    )
    policy, writer, lora, warm_start, ownership = _build_policy_writer(
        config=config,
        context=context,
        source=source,
        source_config=authorities.source_base_config,
    )
    video_store, processor, language = _build_language_inputs(
        args=args,
        config=config,
        context=context,
        source_config=authorities.source_base_config,
        tasks=tasks,
    )
    _validate_collective_environment(context)
    initialize_deferred_process_group(
        context,
        rendezvous_root=args.output_dir.parent,
    )
    contract = build_run_contract(
        args=args,
        config=config,
        context=context,
        segment=segment,
        source=source,
        tokenizer=tokenizer,
        tasks=tasks,
        sampler=sampler,
        video_schedule=schedule,
        warm_start=warm_start,
        ownership=ownership,
        writer=writer,
        repo_root=REPO_ROOT,
    )
    checkpoint_contract_value = checkpoint_contract(contract)
    publish_contract(args, contract, context)
    _restore_resume(
        args,
        config,
        segment,
        context,
        writer,
        checkpoint_contract_value,
    )
    metrics_path = args.output_dir / "metrics.jsonl"
    expected_rows = segment.start_macro if args.mode == "formal" else 0
    if _reconcile_metrics_cursor(
        metrics_path,
        context=context,
        expected_rows=expected_rows,
    ) != expected_rows:
        raise ExpertManifoldError("residual Writer metrics differ from resume cursor")
    iterator = iter(loader)
    torch.cuda.reset_peak_memory_stats(context.device)
    if context.world_size > 1:
        dist.barrier(device_ids=[context.local_rank])
    return V6PriorRuntime(
        args=args,
        context=context,
        config=config,
        segment=segment,
        source=dict(source),
        tokenizer=dict(tokenizer),
        tasks=tasks,
        task_by_global_id={task.global_task_id: task for task in tasks},
        dataset=dataset,
        sampler=sampler,
        video_schedule=schedule,
        iterator=iterator,
        video_store=video_store,
        language_tokens=language,
        processor=processor,
        policy=policy,
        writer=writer,
        lora_contract=lora,
        warm_start=warm_start,
        ownership=ownership,
        run_contract=contract,
        checkpoint_contract=checkpoint_contract_value,
        metrics_path=metrics_path,
    )
