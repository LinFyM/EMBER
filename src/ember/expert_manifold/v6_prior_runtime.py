"""Asset loading, launch contracts, and exact-resume runtime for v6-prior."""

from __future__ import annotations

import argparse
import math
import os
import re
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import h5py
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from ember.expert_manifold.contract import (
    ExpertManifoldError,
    ExpertTask,
    load_task_expert_config,
    load_train_tasks,
)
from ember.expert_manifold.evaluation import inspect_task_expert_bank
from ember.expert_manifold.v6_prior import (
    V6PriorOwnership,
    V6PriorWarmStart,
    configure_v6_prior_trainability,
    load_v6_prior_warm_start_,
)
from ember.expert_manifold.v6_prior_checkpoint import load_v6_prior_checkpoint
from ember.expert_manifold.v6_prior_contract import (
    REPO_ROOT,
    V6_PRIOR_CONFIG_SCHEMA,
    V6_PRIOR_RUN_SCHEMA,
    authority_path,
    git_commit_is_strict_ancestor,
    load_v6_prior_config,
    runtime_for_mode,
)
from ember.expert_manifold.v6_prior_policy_batch import policy_runtime_fields
from ember.lora import LoRAContract, validate_lora_state
from ember.pi05_eval_contract import (
    git_state,
    inspect_source_checkpoint,
    inspect_tokenizer,
    load_evaluation_authorities,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import (
    DistributedContext,
    read_json,
    write_json_atomic,
)
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import (
    initialize_deferred_process_group,
    load_policy,
    load_stats,
    seed_everything,
)
from ember.writer.architecture import LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS
from ember.writer.as_sampling import MixedTaskBatchSampler, TeacherVideoSchedule
from ember.writer.data import FunctionalQueryDataset, RawTeacherVideoStore
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.model import CompleteLoRAWriter, build_lora_tensor_specs
from ember.writer.topology import visible_physical_cuda_index


_RESUME_NAME = re.compile(r"macro_([0-9]{8})")


@dataclass(frozen=True)
class RuntimeSegment:
    total_macros: int
    checkpoint_macros: tuple[int, ...]
    start_macro: int
    stop_macro: int
    schedule_start_macro: int
    schedule_stop_macro: int


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
    writer: CompleteLoRAWriter
    lora_contract: LoRAContract
    expert_targets: dict[str, torch.Tensor]
    expert_bank: dict[str, Any]
    warm_start: V6PriorWarmStart
    ownership: V6PriorOwnership
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    trainable_names: tuple[str, ...]
    trainable_parameters: tuple[torch.nn.Parameter, ...]
    run_contract: dict[str, Any]
    checkpoint_contract: dict[str, Any]
    metrics_path: Path


def _resume_macro(path: Path | None) -> int:
    if path is None:
        return 0
    match = _RESUME_NAME.fullmatch(path.name)
    if match is None or path.parent.name != "checkpoints":
        raise ExpertManifoldError("v6-prior resume path is not a macro checkpoint")
    return int(match.group(1))


def _resolve_segment(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> RuntimeSegment:
    total, checkpoints = runtime_for_mode(config, args.mode)
    start = _resume_macro(args.resume)
    stop = int(args.stop_after_macro or total)
    selected = (
        config["gradient_profile"]
        if args.mode == "gradient-profile"
        else config["profile_run" if args.mode == "profile" else "formal_run"]
    )
    valid = (
        context.world_size == int(selected["expected_world_size"])
        and 24 // context.world_size == int(selected["tasks_per_rank"])
        and 24 % context.world_size == 0
        and 0 <= start < stop <= total
    )
    if args.mode == "gradient-profile":
        valid = valid and args.num_workers == int(selected["num_workers_per_rank"])
    elif args.mode == "profile":
        valid = valid and args.num_workers in tuple(
            int(value) for value in selected["allowed_num_workers_per_rank"]
        )
    else:
        valid = valid and args.num_workers == int(
            config["profile_run"]["artifact_evidence"]["runtime_selection"][
                "num_workers_per_rank"
            ]
        )
    if args.mode == "gradient-profile":
        valid = valid and args.resume is None and stop == total == 1
    else:
        valid = valid and stop in checkpoints and (start == 0 or start in checkpoints)
    state = git_state(REPO_ROOT)
    phase_lineage: tuple[tuple[str, str], ...] = ()
    if args.mode == "profile":
        phase_lineage = (
            (
                str(config["gradient_profile"]["artifact_evidence"]["git"]["commit"]),
                str(state["commit"]),
            ),
        )
    elif args.mode == "formal":
        resume_evidence = config["profile_run"]["artifact_evidence"]
        gradient_commit = str(resume_evidence["gradient_commit"])
        profile_commit = str(resume_evidence["profile_git"]["commit"])
        phase_lineage = (
            (gradient_commit, profile_commit),
            (profile_commit, str(state["commit"])),
        )
    valid = (
        valid
        and not state["dirty_paths"]
        and state["commit"] == state["upstream_commit"]
        and all(
            git_commit_is_strict_ancestor(ancestor, descendant)
            for ancestor, descendant in phase_lineage
        )
    )
    if not valid:
        raise ExpertManifoldError("v6-prior runtime differs from its sealed segment")
    schedule_start = (
        int(config["gradient_profile"]["schedule_macro"])
        if args.mode == "gradient-profile"
        else start
    )
    schedule_stop = schedule_start + (stop - start)
    return RuntimeSegment(
        total_macros=total,
        checkpoint_macros=checkpoints,
        start_macro=start,
        stop_macro=stop,
        schedule_start_macro=schedule_start,
        schedule_stop_macro=schedule_stop,
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
        raise ExpertManifoldError("v6-prior collective environment changed")


def _scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
) -> torch.optim.lr_scheduler.LambdaLR:
    schedule = config["optimization"]["scheduler"]
    peak_lr = float(config["optimization"]["optimizer"]["peak_lr"])
    decay_lr = float(schedule["decay_lr"])
    total = int(schedule["total_macros"])
    warmup = int(schedule["warmup_macros"])

    def factor(macro: int) -> float:
        if macro < warmup:
            return (macro + 1) / warmup
        progress = min(1.0, (macro - warmup) / (total - warmup))
        value = decay_lr + 0.5 * (peak_lr - decay_lr) * (
            1.0 + math.cos(math.pi * progress)
        )
        return value / peak_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _load_source(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    authorities = load_evaluation_authorities(
        authority_path(config, "evaluation_config"),
        REPO_ROOT,
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
    video_costs = _video_costs(
        tasks,
        demo_indices=demos,
        frame_stride=int(config["writer"]["frame_stride"]),
    )
    declared_maximum = int(config["gradient_profile"]["longest_video_sampled_frames"])
    if max(max(rows.values()) for rows in video_costs.values()) != declared_maximum:
        raise ExpertManifoldError("v6-prior longest sampled video changed")
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
        task_video_costs=video_costs,
        assignment_strategy="cost_balanced_long_first",
    )
    if args.mode == "gradient-profile":
        selected_costs = []
        for _, _, task_id, task_visit in sampler.assignments_for_step(
            segment.schedule_start_macro
        ):
            demo = schedule.demos_for_task_visit(
                task_id,
                task_visit,
                excluded=sampler.action_demo_indices_for_task_visit(
                    task_id, task_visit
                ),
            )[0]
            selected_costs.append(video_costs[task_id][demo])
        if max(selected_costs) != declared_maximum:
            raise ExpertManifoldError(
                "v6-prior gradient profile lost the longest sampled video"
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
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    source: Mapping[str, Any],
    source_config: Mapping[str, Any],
) -> tuple[
    torch.nn.Module,
    CompleteLoRAWriter,
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
    writer = CompleteLoRAWriter(
        build_lora_tensor_specs(template),
        template_state=template,
        paligemma_model=bridge.paligemma.model.language_model,
        expert_model=bridge.gemma_expert.model,
        **writer_config,
    )
    warm_start = load_v6_prior_warm_start_(writer, args.warm_start)
    if any(
        not torch.equal(
            value.detach().cpu(),
            template[name].detach().cpu().to(value.dtype),
        )
        for name, value in writer.template_state().items()
    ):
        raise ExpertManifoldError("v6-prior warm start changed physical identity")
    ownership = configure_v6_prior_trainability(writer)
    writer.to(context.device)
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise ExpertManifoldError("v6-prior source policy is not frozen")
    return policy, writer, lora, warm_start, ownership


def _load_expert_targets(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    tasks: Sequence[ExpertTask],
    lora: LoRAContract,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    expert = inspect_task_expert_bank(
        config_path=authority_path(config, "task_expert_config"),
        bank_root=args.expert_bank_root,
        step=int(config["expert_basis"]["expert_step"]),
        source=source,
        task_keys=tuple((task.suite, task.task_id) for task in tasks),
        evaluation_role="development_train",
        require_formal=True,
    )
    rows = sorted(expert["tasks"], key=lambda value: int(value["ordinal"]))
    states = []
    for ordinal, row in enumerate(rows):
        if (
            int(row["ordinal"]) != ordinal
            or int(row["global_task_id"]) != tasks[ordinal].global_task_id
        ):
            raise ExpertManifoldError("v6-prior expert task ordering changed")
        state = load_file(
            str(Path(str(row["checkpoint"])) / "adapter.safetensors"),
            device="cpu",
        )
        validate_lora_state(state, lora)
        states.append(state)
    names = tuple(states[0])
    targets = {
        name: torch.stack([state[name] for state in states]).to(device)
        for name in names
    }
    return targets, expert


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
        args.tokenizer_path,
        max_length,
        str(context.device),
    )
    language = {task.global_task_id: tokenizer((task.language,)) for task in tasks}
    store = RawTeacherVideoStore(
        [task.authority for task in tasks],
        frame_stride=int(config["writer"]["frame_stride"]),
        max_open_files=4,
    )
    return store, processor, language


def _synchronize_writer(
    writer: CompleteLoRAWriter,
    context: DistributedContext,
) -> None:
    if context.world_size <= 1:
        return
    with torch.no_grad():
        for value in (*writer.parameters(), *writer.buffers()):
            dist.broadcast(value, src=0)


def _rank_topology(context: DistributedContext) -> list[dict[str, Any]]:
    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_name": torch.cuda.get_device_name(context.device),
        "physical_gpu": visible_physical_cuda_index(context.local_rank),
        "device": str(context.device),
        "numa_node": context.numa_node,
        "cpu_affinity": list(context.cpu_affinity or ()),
    }
    rows: list[Any] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(rows, local)
    else:
        rows[0] = local
    return rows


def _run_contract(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    segment: RuntimeSegment,
    source: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    tasks: Sequence[ExpertTask],
    sampler: MixedTaskBatchSampler,
    video_schedule: TeacherVideoSchedule,
    expert: Mapping[str, Any],
    warm_start: V6PriorWarmStart,
    ownership: V6PriorOwnership,
    trainable_names: Sequence[str],
) -> dict[str, Any]:
    state = git_state(REPO_ROOT)
    schedule_start = (
        segment.schedule_start_macro if args.mode == "gradient-profile" else 0
    )
    schedule_stop = schedule_start + segment.total_macros
    return {
        "schema_version": V6_PRIOR_RUN_SCHEMA,
        "mode": args.mode,
        "git": dict(state),
        "config": {
            "path": str(args.config),
            "schema": V6_PRIOR_CONFIG_SCHEMA,
            "bytes": args.config.stat().st_size,
        },
        "source": dict(source),
        "tokenizer": dict(tokenizer),
        "initialization": {
            "mode": "historical_v6_macro400_load_only",
            "checkpoint": str(warm_start.checkpoint),
            "writer_state_tensor_count": warm_start.state_tensor_count,
            "writer_state_value_count": warm_start.state_value_count,
            "optimizer": "fresh",
            "scheduler": "fresh",
            "rng": "fresh_seed",
        },
        "expert_bank": {
            "root": str(args.expert_bank_root),
            "step": int(config["expert_basis"]["expert_step"]),
            "training_commit": expert["training_commit"],
            "tasks": [dict(row) for row in expert["tasks"]],
            "deployment_read": False,
        },
        "data": {
            "root": str(args.data_root),
            "tasks": [
                {
                    "ordinal": task.ordinal,
                    "global_task_id": task.global_task_id,
                    "suite": task.suite,
                    "task_id": task.task_id,
                    "language": task.language,
                    "path": str(task.authority.path),
                    "bytes": task.authority.expected_bytes,
                }
                for task in tasks
            ],
            **dict(config["data"]),
            "consumed_schedule": video_schedule.consumed_identity_summary(
                sampler,
                schedule_start,
                schedule_stop,
            ),
        },
        "method": dict(config["method"]),
        "information_wall": dict(config["information_wall"]),
        "writer": dict(config["writer"]),
        "expert_basis": dict(config["expert_basis"]),
        "objective": dict(config["objective"]),
        "optimization": dict(config["optimization"]),
        "ownership": {
            "frozen_parameter_count": ownership.frozen_parameter_count,
            "trainable_parameter_count": ownership.trainable_parameter_count,
            "frozen_tensor_count": ownership.frozen_tensor_count,
            "trainable_tensor_count": ownership.trainable_tensor_count,
            "trainable_tensor_names": list(trainable_names),
            "source_policy_trainable_parameter_count": 0,
        },
        "runtime": {
            "host": socket.gethostname(),
            "device": torch.cuda.get_device_name(context.device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "world_size": context.world_size,
            "tasks_per_rank": 24 // context.world_size,
            "rank_topology": _rank_topology(context),
            "total_macros": segment.total_macros,
            "gradient_profile_schedule_macro": (
                segment.schedule_start_macro
                if args.mode == "gradient-profile"
                else None
            ),
            "checkpoint_macros": list(segment.checkpoint_macros),
            "num_workers_per_rank": args.num_workers,
            "action_loader_prefetch_factor": 2 if args.num_workers else None,
            "action_loader_persistent_workers": args.num_workers > 0,
            **policy_runtime_fields(config),
            "distributed_model_wrapper": "none",
            "gradient_reduction": (
                "single_flat_parameter_ordered_allreduce_mean_after_local_task_mean"
            ),
            "deferred_process_group": True,
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
            "nccl_algo": os.environ.get("NCCL_ALGO"),
            "nccl_proto": os.environ.get("NCCL_PROTO"),
            "cuda_allocator_conf_observed": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        },
        "content_hash_policy": "disabled_by_owner",
    }


def _checkpoint_contract(run_contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_schema": run_contract["schema_version"],
        "mode": run_contract["mode"],
        "git_commit": run_contract["git"]["commit"],
        "config": run_contract["config"],
        "source": run_contract["source"],
        "initialization": run_contract["initialization"],
        "expert_bank_root": run_contract["expert_bank"]["root"],
        "expert_step": run_contract["expert_bank"]["step"],
        "objective": run_contract["objective"],
        "ownership": run_contract["ownership"],
        "world_size": run_contract["runtime"]["world_size"],
    }


def _cursor_contract(config: Mapping[str, Any], macro: int) -> dict[str, Any]:
    data = config["data"]
    return {
        "next_macro": macro,
        "task_visits_per_task": macro,
        "sampler_seed": int(data["sampler_seed"]),
        "teacher_video_seed": int(data["teacher_video_seed"]),
        "counterfactual_seed": int(data["counterfactual_seed"]),
        "counterfactual_phase": macro % 3,
        "videos_per_task_visit": 1,
        "action_queries_per_task": int(data["action_queries_per_task"]),
    }


def _publish_contract(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    context: DistributedContext,
) -> None:
    path = args.output_dir / "run_contract.json"
    if context.is_main:
        if args.resume is None:
            if args.output_dir.exists() and any(args.output_dir.iterdir()):
                raise ExpertManifoldError("fresh v6-prior output is not empty")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(path, dict(contract))
        elif (
            args.resume.parent.parent.resolve() != args.output_dir
            or not path.is_file()
            or read_json(path) != dict(contract)
        ):
            raise ExpertManifoldError("v6-prior resume run contract changed")
        append_jsonl(
            args.output_dir / "invocations.jsonl",
            {
                "argv": os.sys.argv,
                "started_unix": time.time(),
                "resume": str(args.resume) if args.resume else None,
                "requested_stop_after_macro": args.stop_after_macro,
            },
        )
    if context.world_size > 1:
        dist.barrier(device_ids=[context.local_rank])


def _metrics_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(bool(line) for line in path.read_text(encoding="utf-8").splitlines())


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
        args=args,
        config=config,
        context=context,
        source=source,
        source_config=authorities.source_base_config,
    )
    expert_targets, expert = _load_expert_targets(
        args=args,
        config=config,
        source=source,
        tasks=tasks,
        lora=lora,
        device=context.device,
    )
    video_store, processor, language = _build_language_inputs(
        args=args,
        config=config,
        context=context,
        source_config=authorities.source_base_config,
        tasks=tasks,
    )
    trainable_rows = tuple(
        (name, parameter)
        for name, parameter in writer.named_parameters()
        if parameter.requires_grad
    )
    trainable_names = tuple(name for name, _ in trainable_rows)
    trainable_parameters = tuple(parameter for _, parameter in trainable_rows)
    optimizer_config = config["optimization"]["optimizer"]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(optimizer_config["peak_lr"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler = _scheduler(optimizer, config)
    _validate_collective_environment(context)
    initialize_deferred_process_group(
        context,
        rendezvous_root=args.output_dir.parent,
    )
    _synchronize_writer(writer, context)
    contract = _run_contract(
        args=args,
        config=config,
        context=context,
        segment=segment,
        source=source,
        tokenizer=tokenizer,
        tasks=tasks,
        sampler=sampler,
        video_schedule=schedule,
        expert=expert,
        warm_start=warm_start,
        ownership=ownership,
        trainable_names=trainable_names,
    )
    checkpoint_contract = _checkpoint_contract(contract)
    _publish_contract(args, contract, context)
    iterator = iter(loader)
    if args.resume is not None:
        loaded, rows = load_v6_prior_checkpoint(
            checkpoint=args.resume,
            writer=writer,
            optimizer=optimizer,
            scheduler=scheduler,
            context=context,
            expected_cursor_contract=_cursor_contract(config, segment.start_macro),
            expected_checkpoint_contract=checkpoint_contract,
        )
        if loaded != segment.start_macro or rows != segment.start_macro:
            raise ExpertManifoldError("v6-prior resume cursor changed")
    metrics_path = args.output_dir / "metrics.jsonl"
    if _metrics_rows(metrics_path) != segment.start_macro:
        raise ExpertManifoldError("v6-prior metrics differ from resume cursor")
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
        expert_targets=expert_targets,
        expert_bank=expert,
        warm_start=warm_start,
        ownership=ownership,
        optimizer=optimizer,
        scheduler=scheduler,
        trainable_names=trainable_names,
        trainable_parameters=trainable_parameters,
        run_contract=contract,
        checkpoint_contract=checkpoint_contract,
        metrics_path=metrics_path,
    )
