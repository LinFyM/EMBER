"""Formal and profile orchestration for ECP Stage 1 realizability training."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.ecp.contracts import build_target_owners
from ember.ecp.observer_authority import FrozenObserverAuthority, load_frozen_observer_authority
from ember.ecp.stage0_training import load_stage0_config, stage0_source_authority
from ember.ecp.stage1 import ECPStage1Model
from ember.ecp.stage1_data import (
    ECPStage1EvidenceBank,
    ECPStage1Task,
    build_stage1_schedule,
    build_stage1_video_store,
    load_stage1_evidence_bank,
    load_stage1_tasks,
    gauge_canonicalize_lora_state,
    tokenize_stage1_languages,
)
from ember.ecp.stage1_panels import ECPStage1FunctionalPanel, cache_stage1_functional_panels
from ember.ecp.stage1_train_step import run_stage1_update
from ember.lora import LoRAContract, validate_lora_state
from ember.pi05_eval_contract import git_state, git_state_is_clean_pushed_or_frozen_authority
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import DistributedContext, barrier, read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl, reconcile_metrics
from ember.pi05_source_setup import (
    initialize_deferred_process_group,
    initialize_distributed,
    load_config,
    load_policy,
    seed_everything,
)
from ember.writer.data import RawTeacherVideoStore
from ember.writer.functional import prepare_frozen_writer_policy


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_SCHEMA = "ember_ecp_stage1_privileged_absolute_compiler_run_v2"
STAGE = "stage1_privileged_absolute_compiler"


@dataclass
class ECPStage1Runtime:
    args: argparse.Namespace
    config: dict[str, Any]
    context: DistributedContext
    tasks: tuple[ECPStage1Task, ...]
    task_by_ordinal: dict[int, ECPStage1Task]
    schedule: tuple[tuple[int, int], ...]
    video_store: RawTeacherVideoStore
    language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor]]
    evidence_bank: ECPStage1EvidenceBank
    functional_panels: dict[int, tuple[ECPStage1FunctionalPanel, ...]]
    functional_start_task_visits: int
    policy: torch.nn.Module
    observer: FrozenObserverAuthority
    contract: LoRAContract
    prior_state: Mapping[str, torch.Tensor]
    model: ECPStage1Model
    trainable_parameters: tuple[torch.nn.Parameter, ...]
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    start_task_visits: int
    stop_after_task_visits: int
    total_task_visits: int
    checkpoint_task_visits: tuple[int, ...]
    metrics_rows: int


@dataclass(frozen=True)
class ECPStage1Authorities:
    source: dict[str, Any]
    source_config: dict[str, Any]
    policy: torch.nn.Module
    contract: LoRAContract
    identity_state: Mapping[str, torch.Tensor]
    observer: FrozenObserverAuthority
    prior_state: Mapping[str, torch.Tensor]
    model: ECPStage1Model


@dataclass(frozen=True)
class ECPStage1Inputs:
    tasks: tuple[ECPStage1Task, ...]
    evidence_bank: ECPStage1EvidenceBank
    video_store: RawTeacherVideoStore
    schedule: tuple[tuple[int, int], ...]


def stage1_repo_authority(config: Mapping[str, Any], name: str) -> Path:
    path = Path(str(config["authorities"][name]))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def stage1_asset_authority(
    config: Mapping[str, Any], name: str, asset_root: Path
) -> Path:
    path = Path(str(config["authorities"][name]))
    if path.is_absolute():
        return path
    return asset_root / path


def load_stage1_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if (
        config.get("schema_version")
        != "ember_ecp_stage1_privileged_absolute_compiler_v2"
        or config.get("status") != "active_stage1_absolute_warmstart"
        or config.get("model", {}).get("hard_rank_partition") is not False
        or config.get("information_wall", {}).get("validation_action_or_reward_reads")
        != 0
        or config.get("information_wall", {}).get("test_action_or_reward_reads")
        != 0
    ):
        raise ValueError("unsupported ECP Stage 1 contract")
    return config


def _runtime_limits(
    args: argparse.Namespace, config: Mapping[str, Any], context: DistributedContext
) -> tuple[int, int, tuple[int, ...]]:
    if args.mode == "formal":
        expected_world = int(config["optimization"]["world_size"])
        total = int(config["optimization"]["total_task_visits"])
        checkpoints = tuple(
            int(value)
            for value in config["optimization"]["checkpoint_task_visits"]
        )
        stop = int(args.stop_after_task_visits or total)
        if (
            context.world_size != expected_world
            or stop
            not in set(config["optimization"]["stage_stop_task_visits"])
            or os.environ.get("NCCL_P2P_DISABLE") != "1"
            or not git_state_is_clean_pushed_or_frozen_authority(git_state(REPO_ROOT))
        ):
            raise ValueError("formal ECP Stage 1 launch authority changed")
    else:
        if context.world_size != int(config["profile_defaults"]["world_size"]):
            raise ValueError("ECP Stage 1 profile must use one GPU")
        total = int(args.stop_after_task_visits or config["profile_defaults"]["task_visits"])
        stop = total
        checkpoints = ()
    if total % context.world_size or stop % context.world_size:
        raise ValueError("ECP Stage 1 cursor must align to world size")
    return total, stop, checkpoints


def _scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    total_updates: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    cell = config["optimization"]["scheduler"]
    warmup = int(cell["warmup_updates"])
    peak = float(config["optimization"]["optimizer"]["learning_rate"])
    floor = float(cell["decay_lr"]) / peak

    def scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_updates - warmup, 1)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _needed_functional_members(
    *,
    schedule: tuple[tuple[int, int], ...],
    evidence: ECPStage1EvidenceBank,
    rank: int,
    world_size: int,
    start: int,
    stop: int,
    functional_start: int,
) -> set[int]:
    result = set()
    for cursor in range(start, stop, world_size):
        if cursor + world_size <= functional_start:
            continue
        ordinal, task_visit = schedule[cursor + rank]
        indices = evidence.member_indices(ordinal)
        result.add(indices[task_visit % len(indices)])
    return result


def _build_contract(
    runtime: ECPStage1Runtime,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    local = {
        "rank": runtime.context.rank,
        "local_rank": runtime.context.local_rank,
        "device": str(runtime.context.device),
        "numa_node": runtime.context.numa_node,
        "cpu_affinity": list(runtime.context.cpu_affinity or ()),
    }
    topology: list[Any] = [None] * runtime.context.world_size
    if runtime.context.world_size > 1:
        dist.all_gather_object(topology, local)
    else:
        topology[0] = local
    state = git_state(REPO_ROOT)
    return {
        "schema_version": RUN_SCHEMA,
        "stage": STAGE,
        "mode": runtime.args.mode,
        "command": list(sys.argv),
        "git": {"branch": state["branch"], "commit": state["commit"]},
        "host": socket.gethostname(),
        "config": {"path": str(runtime.args.config), "bytes": runtime.args.config.stat().st_size},
        "source": dict(source),
        "asset_root": str(runtime.args.asset_root),
        "data_root": str(runtime.args.data_root),
        "tokenizer": {"path": str(runtime.args.tokenizer_path), "bytes": runtime.args.tokenizer_path.stat().st_size},
        "observer_authority": {
            "native": str(runtime.observer.native_checkpoint),
            "action_meta": str(runtime.observer.action_meta_checkpoint),
            "frozen": True,
        },
        "tasks": [
            {
                "ordinal": task.ordinal,
                "global_task_id": task.global_task_id,
                "suite": task.suite,
                "task_id": task.task_id,
                "fold_role": task.fold_role,
            }
            for task in runtime.tasks
        ],
        "successful_members": len(runtime.evidence_bank.members),
        "model": dict(runtime.config["model"]),
        "data": dict(runtime.config["data"]),
        "objective": dict(runtime.config["objective"]),
        "optimization": dict(runtime.config["optimization"]),
        "information_wall": dict(runtime.config["information_wall"]),
        "runtime": {
            "world_size": runtime.context.world_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
            "topology": topology,
            "start_task_visits": runtime.start_task_visits,
            "stop_after_task_visits": runtime.stop_after_task_visits,
            "functional_start_task_visits": runtime.functional_start_task_visits,
            "loaded_functional_members_by_rank": len(runtime.functional_panels),
        },
        "trainable_parameters": sum(value.numel() for value in runtime.trainable_parameters),
        "content_hash_policy": "disabled_by_owner",
    }


def load_stage1_authorities(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> ECPStage1Authorities:
    source = stage0_source_authority(args)
    source_config = load_config(stage1_repo_authority(config, "source_base_config"))
    policy = load_policy(Path(source["model_path"]), source_config, context.device)
    contract = load_pi05_lora_contract(stage1_repo_authority(config, "lora_contract"))
    identity = prepare_frozen_writer_policy(policy, contract)
    owners = build_target_owners(contract)
    stage0_config = load_stage0_config(stage1_repo_authority(config, "stage0_config"))
    observer = load_frozen_observer_authority(
        stage0_config=stage0_config,
        owners=owners,
        policy=policy,
        native_checkpoint=stage1_asset_authority(
            config, "native_observer_checkpoint", args.asset_root
        ),
        action_meta_checkpoint=stage1_asset_authority(
            config, "action_meta_checkpoint", args.asset_root
        ),
        device=context.device,
        max_frames_per_call=args.max_frames_per_call,
    )
    prior = load_file(
        str(stage1_asset_authority(config, "stable_shared_prior", args.asset_root)),
        device=str(context.device),
    )
    validate_lora_state(prior, contract)
    prior = gauge_canonicalize_lora_state(prior, contract)
    model = ECPStage1Model(
        owners,
        contract,
        prior,
        program_width=int(config["model"]["program_width"]),
        compiler_width=int(config["model"]["compiler_width"]),
        event_slots=int(config["model"]["event_slots"]),
        phase_width=int(config["model"]["phase_response_width"]),
        factor_head_init=config["model"]["factor_head_init_std"],
    ).to(context.device)
    return ECPStage1Authorities(
        source=source,
        source_config=source_config,
        policy=policy,
        contract=contract,
        identity_state=identity,
        observer=observer,
        prior_state=prior,
        model=model,
    )


def _load_inputs(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    contract: LoRAContract,
    *,
    total_task_visits: int,
) -> ECPStage1Inputs:
    tasks = load_stage1_tasks(
        target_manifest=stage1_repo_authority(config, "target_manifest"),
        selection_path=stage1_repo_authority(config, "successful_member_selection"),
        data_root=args.data_root,
    )
    evidence = load_stage1_evidence_bank(
        selection_path=stage1_repo_authority(config, "successful_member_selection"),
        phase_analysis_path=stage1_asset_authority(
            config, "phase_analysis", args.asset_root
        ),
        phase_code_root=stage1_asset_authority(
            config, "phase_code_root", args.asset_root
        ),
        asset_root=args.asset_root,
        contract=contract,
        device=context.device,
    )
    return ECPStage1Inputs(
        tasks=tasks,
        evidence_bank=evidence,
        video_store=build_stage1_video_store(
            tasks, frame_stride=int(config["data"]["frame_stride"])
        ),
        schedule=build_stage1_schedule(
            config=config,
            tasks=tasks,
            world_size=context.world_size,
            total_task_visits=total_task_visits,
            mode=args.mode,
        ),
    )


def _prepare_optimization(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    model: ECPStage1Model,
    *,
    total_task_visits: int,
    stop_after_task_visits: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int, int]:
    cell = config["optimization"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cell["learning_rate"]),
        betas=tuple(float(value) for value in cell["betas"]),
        eps=float(cell["eps"]),
        weight_decay=float(cell["weight_decay"]),
    )
    scheduler = _scheduler(
        optimizer, config, total_task_visits // context.world_size
    )
    initialize_deferred_process_group(context, rendezvous_root=args.output_dir.parent)
    if context.world_size > 1:
        for value in model.state_dict().values():
            dist.broadcast(value, src=0)
    start = 0
    expected_metrics = 0
    if args.resume is not None:
        start, expected_metrics = load_ecp_checkpoint(
            checkpoint=args.resume,
            stage=STAGE,
            context=context,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            run_contract_schema=RUN_SCHEMA,
        )
    if not 0 <= start < stop_after_task_visits or start % context.world_size:
        raise ValueError("ECP Stage 1 resume cursor is outside this segment")
    return optimizer, scheduler, start, expected_metrics


def _load_functional_panels(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    authorities: ECPStage1Authorities,
    inputs: ECPStage1Inputs,
    *,
    start: int,
    stop: int,
    functional_start: int,
) -> dict[int, tuple[ECPStage1FunctionalPanel, ...]]:
    needed = _needed_functional_members(
        schedule=inputs.schedule,
        evidence=inputs.evidence_bank,
        rank=context.rank,
        world_size=context.world_size,
        start=start,
        stop=stop,
        functional_start=functional_start,
    )
    if not needed:
        return {}
    return cache_stage1_functional_panels(
        policy=authorities.policy,
        identity_state=authorities.identity_state,
        evidence_bank=inputs.evidence_bank,
        contract=authorities.contract,
        device=context.device,
        policy_seed=int(config["objective"]["train_policy_seed"]),
        fit_only=True,
        member_indices=needed,
    )


def _initialize_run_contract(
    runtime: ECPStage1Runtime, source: Mapping[str, Any]
) -> None:
    contract = _build_contract(runtime, source)
    if runtime.context.is_main and runtime.args.resume is None:
        runtime.args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(runtime.args.output_dir / "run_contract.json", contract)
    elif runtime.context.is_main:
        existing = read_json(runtime.args.output_dir / "run_contract.json")
        if (
            existing.get("schema_version") != RUN_SCHEMA
            or existing.get("stage") != STAGE
            or existing.get("git", {}).get("commit")
            != contract["git"]["commit"]
            or existing.get("config", {}).get("bytes")
            != contract["config"]["bytes"]
            or existing.get("source", {}).get("checkpoint")
            != contract["source"]["checkpoint"]
            or existing.get("runtime", {}).get("world_size")
            != runtime.context.world_size
        ):
            raise ValueError("ECP Stage 1 resume run contract changed")
    barrier(runtime.context)


def prepare_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> ECPStage1Runtime:
    config = load_stage1_config(args.config)
    total, stop, checkpoints = _runtime_limits(args, config, context)
    seed_everything(int(config["optimization"]["seed"]), context)
    authorities = load_stage1_authorities(args, config, context)
    inputs = _load_inputs(
        args,
        config=config,
        context=context,
        contract=authorities.contract,
        total_task_visits=total,
    )
    optimizer, scheduler, start, expected_metrics = _prepare_optimization(
        args,
        config,
        context,
        authorities.model,
        total_task_visits=total,
        stop_after_task_visits=stop,
    )
    functional_start = (
        0
        if args.mode == "profile" and args.profile_functional
        else int(config["objective"]["functional_start_task_visits"])
    )
    panels = _load_functional_panels(
        args,
        config,
        context,
        authorities,
        inputs,
        start=start,
        stop=stop,
        functional_start=functional_start,
    )
    language = tokenize_stage1_languages(
        inputs.tasks,
        tokenizer_path=args.tokenizer_path,
        max_length=int(
            authorities.source_config["features"]["tokenizer_max_length"]
        ),
        device=context.device,
    )
    metrics_rows = (
        reconcile_metrics(
            args.output_dir / "metrics.jsonl",
            start,
            expected_metrics,
            cursor_key="task_visits",
        )
        if context.is_main and args.resume is not None
        else 0
    )
    runtime = ECPStage1Runtime(
        args=args,
        config=config,
        context=context,
        tasks=inputs.tasks,
        task_by_ordinal={task.ordinal: task for task in inputs.tasks},
        schedule=inputs.schedule,
        video_store=inputs.video_store,
        language_tokens=language,
        evidence_bank=inputs.evidence_bank,
        functional_panels=panels,
        functional_start_task_visits=functional_start,
        policy=authorities.policy,
        observer=authorities.observer,
        contract=authorities.contract,
        prior_state=authorities.prior_state,
        model=authorities.model,
        trainable_parameters=tuple(authorities.model.parameters()),
        optimizer=optimizer,
        scheduler=scheduler,
        start_task_visits=start,
        stop_after_task_visits=stop,
        total_task_visits=total,
        checkpoint_task_visits=checkpoints,
        metrics_rows=metrics_rows,
    )
    _initialize_run_contract(runtime, authorities.source)
    authorities.model.train()
    torch.cuda.reset_peak_memory_stats(context.device)
    return runtime


def train(args: argparse.Namespace) -> None:
    if args.resume is None and args.output_dir.exists():
        raise ValueError("fresh ECP Stage 1 output directory already exists")
    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    runtime: ECPStage1Runtime | None = None
    try:
        runtime = prepare_runtime(args, context)
        started = time.monotonic()
        for cursor in range(
            runtime.start_task_visits,
            runtime.stop_after_task_visits,
            context.world_size,
        ):
            row = run_stage1_update(runtime, cursor=cursor, run_started=started)
            if context.is_main:
                append_jsonl(args.output_dir / "metrics.jsonl", row)
                runtime.metrics_rows += 1
                if row["optimizer_update"] % args.log_every == 0:
                    print(json.dumps(row, sort_keys=True), flush=True)
            if row["task_visits"] in runtime.checkpoint_task_visits:
                save_ecp_checkpoint(
                    output_dir=args.output_dir,
                    macro=int(row["task_visits"]),
                    stage=STAGE,
                    context=context,
                    model=runtime.model,
                    optimizer=runtime.optimizer,
                    scheduler=runtime.scheduler,
                    run_contract_schema=RUN_SCHEMA,
                    metrics_rows=runtime.metrics_rows,
                )
        if context.is_main:
            result = {
                "stage": STAGE,
                "completed_task_visits": runtime.stop_after_task_visits,
                "total_task_visits": runtime.total_task_visits,
            }
            write_json_atomic(args.output_dir / "segment_completion.json", result)
            if runtime.stop_after_task_visits == runtime.total_task_visits:
                write_json_atomic(args.output_dir / "completion.json", result)
    finally:
        if runtime is not None:
            runtime.video_store.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/pi05_ecp_stage1_privileged_absolute_compiler_v2.json",
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-task-visits", type=int)
    parser.add_argument("--max-frames-per-call", type=int)
    parser.add_argument("--profile-functional", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
        "asset_root",
        "source_run",
        "checkpoint",
        "tokenizer_path",
        "data_root",
        "output_dir",
        "resume",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.log_every <= 0:
        raise ValueError("ECP Stage 1 log interval must be positive")
    return args
