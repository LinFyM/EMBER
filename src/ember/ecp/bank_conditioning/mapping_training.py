"""Distributed G3 mapping acquisition for the frozen-Program compiler."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist

from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.ecp.contracts import TargetOwner, build_target_owners
from ember.ecp.natural_program import NaturalProgramModel
from ember.ecp.natural_program_data import (
    NaturalProgramTask,
    load_natural_program_tasks,
)
from ember.ecp.shared_compiler import SharedNativeFactorCompiler
from ember.ecp.shared_compiler_assets import (
    G3_CONFIG_SCHEMA,
    SharedCompilerRankAssets,
    authority_path,
    build_frozen_g2_program,
    load_shared_compiler_config,
    load_shared_rank_assets,
)
from ember.ecp.shared_compiler_authority import (
    MAPPING_RUN_SCHEMA,
    build_mapping_run_contract,
    publish_shared_compiler_run_contract,
)
from ember.ecp.bank_conditioning.mapping import (
    SharedCompilerMappingSchedule,
    SharedCompilerMappingSplit,
    load_mapping_split,
)
from ember.ecp.bank_conditioning.consensus import FitConsensusTeacherStore
from ember.ecp.bank_conditioning.mapping_step import run_mapping_optimizer_step
from ember.ecp.shared_compiler_native_teacher import NativeTeacherStore
from ember.ecp.stage0_training import stage0_source_authority, tokenize_stage0_languages
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import derive_pi05_lora_rank
from ember.pi05_source_checkpoint import DistributedContext, read_json, write_json_atomic
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


REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class MappingRuntime:
    args: argparse.Namespace
    config: dict[str, Any]
    context: DistributedContext
    tasks: tuple[NaturalProgramTask, ...]
    task_by_id: dict[int, NaturalProgramTask]
    mapping_split: SharedCompilerMappingSplit
    schedule: SharedCompilerMappingSchedule
    video_store: RawTeacherVideoStore
    language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor]]
    policy: torch.nn.Module
    program: NaturalProgramModel
    compiler: SharedNativeFactorCompiler
    owners: tuple[TargetOwner, ...]
    ranks: SharedCompilerRankAssets
    rank4_contract: Any
    native_teachers: NativeTeacherStore
    consensus_teachers: FitConsensusTeacherStore
    query_points: int
    trainable_parameters: tuple[torch.nn.Parameter, ...]
    frozen_parameters: tuple[torch.nn.Parameter, ...]
    optimizer: torch.optim.Optimizer
    scheduler_lr: torch.optim.lr_scheduler.LRScheduler
    optimizer_steps_per_macro: int
    optimizer_steps: int
    total_macros: int
    stop_after_macro: int
    checkpoint_macros: tuple[int, ...]
    start_macro: int
    metrics_rows: int
    run_contract: dict[str, Any]

    def close(self) -> None:
        self.video_store.close()


@dataclass
class _TrainingAssets:
    source: dict[str, Any]
    source_config: dict[str, Any]
    policy: torch.nn.Module
    ranks: SharedCompilerRankAssets
    rank4_contract: Any
    owners: tuple[TargetOwner, ...]
    program: NaturalProgramModel
    compiler: SharedNativeFactorCompiler
    trainable: tuple[torch.nn.Parameter, ...]
    native_teachers: NativeTeacherStore
    video_store: RawTeacherVideoStore
    language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor]]
    query_points: int
    frozen: tuple[torch.nn.Parameter, ...]


def _tasks(
    config: Mapping[str, Any], data_root: Path, asset_root: Path
) -> tuple[NaturalProgramTask, ...]:
    fold = config["fold"]
    return load_natural_program_tasks(
        meta_protocol_path=authority_path(
            config, "meta_protocol", asset_root=asset_root
        ),
        source_manifest_path=authority_path(
            config, "source_manifest", asset_root=asset_root
        ),
        target_manifest_path=authority_path(
            config, "target_manifest", asset_root=asset_root
        ),
        data_root=data_root,
        target_fit_ids=fold["target_fit_task_ids"],
        target_held_ids=fold["target_held_task_ids"],
        held_meta_fold=int(fold["meta_held_fold"]),
    )


def _resolve_runtime(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> tuple[int, int, tuple[int, ...], int]:
    cell = config["formal_run" if args.mode == "formal" else "profile_defaults"]
    if context.world_size not in cell["allowed_world_sizes"]:
        raise ValueError("G3 mapping world size is outside its launch contract")
    total = int(cell["total_macros"])
    stop = int(args.stop_after_macro or total)
    checkpoints = tuple(map(int, cell["checkpoint_macros"]))
    steps = 5 if args.mode == "formal" else 1
    if not 0 < stop <= total:
        raise ValueError("G3 mapping stop is outside its segment")
    if args.mode == "formal":
        if stop not in set(map(int, cell["stage_stop_macros"])):
            raise ValueError("formal G3 mapping stop is not pre-registered")
        state = git_state(REPO_ROOT)
        if (
            not git_state_is_clean_pushed_or_frozen_authority(state)
            or state.get("branch") != ""
            or state.get("upstream") is not None
        ):
            raise ValueError(
                "formal G3 mapping requires clean detached origin/main authority"
            )
    return total, stop, checkpoints, steps


def _optimizer(
    parameters: tuple[torch.nn.Parameter, ...], config: Mapping[str, Any]
) -> torch.optim.AdamW:
    cell = config["optimization"]["optimizer"]
    return torch.optim.AdamW(
        parameters,
        lr=float(cell["peak_lr"]),
        betas=tuple(cell["betas"]),
        eps=float(cell["eps"]),
        weight_decay=float(cell["weight_decay"]),
    )


def _scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    cell = config["optimization"]["scheduler"]
    warmup = int(cell["warmup_optimizer_steps"])
    peak = float(config["optimization"]["optimizer"]["peak_lr"])
    floor = float(cell["decay_lr"]) / peak

    def scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return floor + (1.0 - floor) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _resume_cursor(
    runtime_args: argparse.Namespace,
    context: DistributedContext,
    *,
    stage: str,
    compiler: SharedNativeFactorCompiler,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    optimizer_steps_per_macro: int,
    stop: int,
) -> tuple[int, int]:
    start = 0
    expected_metrics = 0
    if runtime_args.resume is not None:
        start, expected_metrics = load_ecp_checkpoint(
            checkpoint=runtime_args.resume,
            stage=stage,
            context=context,
            model=compiler,
            optimizer=optimizer,
            scheduler=scheduler,
            run_contract_schema=MAPPING_RUN_SCHEMA,
        )
        if int(scheduler.last_epoch) != start * optimizer_steps_per_macro:
            raise ValueError("G3 mapping resume optimizer cursor changed")
    if not 0 <= start < stop:
        raise ValueError("G3 mapping resume cursor is outside this segment")
    rows = (
        reconcile_metrics(
            runtime_args.output_dir / "metrics.jsonl",
            start,
            expected_metrics,
            cursor_key="macro",
        )
        if context.is_main
        else 0
    )
    return start, rows


def _load_training_assets(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    tasks: tuple[NaturalProgramTask, ...],
) -> _TrainingAssets:
    expected_checkpoint = authority_path(
        config, "source_checkpoint", asset_root=args.asset_root
    )
    expected_tokenizer = authority_path(
        config, "tokenizer", asset_root=args.asset_root
    )
    source_matches = all(
        (
            args.checkpoint == expected_checkpoint,
            args.source_run == expected_checkpoint.parent.parent,
            args.tokenizer_path == expected_tokenizer,
        )
    )
    if not source_matches:
        raise ValueError("G3 mapping source or tokenizer authority changed")
    source = stage0_source_authority(args)
    source_config = load_config(
        authority_path(config, "source_base_config", asset_root=args.asset_root)
    )
    policy = load_policy(Path(source["model_path"]), source_config, context.device)
    policy.requires_grad_(False).eval()
    ranks = load_shared_rank_assets(
        config,
        asset_root=args.asset_root,
        held_global_ids=set(map(int, config["fold"]["target_held_task_ids"])),
        device=context.device,
    )
    owners = build_target_owners(ranks.contract)
    rank4_contract = derive_pi05_lora_rank(ranks.contract, rank=4)
    program = build_frozen_g2_program(
        config, asset_root=args.asset_root, owners=owners, device=context.device
    )
    prepare_frozen_writer_policy(policy, ranks.contract)
    compiler = SharedNativeFactorCompiler(
        owners,
        program_width=int(config["model"]["program_width"]),
        event_slots=int(config["model"]["event_slots"]),
        anchor_width=int(config["model"]["anchor_width"]),
        relative_eigenvalue_floor=float(
            config["model"]["relative_eigenvalue_floor"]
        ),
    ).to(context.device)
    compiler.scale_head.requires_grad_(False)
    compiler.train()
    trainable = tuple(value for value in compiler.parameters() if value.requires_grad)
    if set(map(id, trainable)) != set(map(id, compiler.anchor_scorer.parameters())):
        raise ValueError("G3 mapping trainable ownership escaped anchor scorer")
    teacher_path = authority_path(
        config, "native_teacher_manifest", asset_root=args.asset_root
    )
    teacher_root = read_json(teacher_path)
    native_teachers = NativeTeacherStore(
        teacher_path,
        contract=rank4_contract,
        expected_fit_task_ids=set(map(int, teacher_root["coverage"]["task_ids"])),
        expected_full_fit_task_ids=set(
            map(int, teacher_root["fit_authority_task_ids"])
        ),
        device=context.device,
    )
    video_store = RawTeacherVideoStore(
        tuple(task.writer_authority() for task in tasks),
        frame_stride=int(config["data"]["frame_stride"]),
        max_open_files=8,
    )
    language_tokens = tokenize_stage0_languages(
        tasks,
        tokenizer_path=args.tokenizer_path,
        max_length=int(source_config["features"]["tokenizer_max_length"]),
        device=context.device,
    )
    g2 = read_json(authority_path(config, "g2_config", asset_root=args.asset_root))
    return _TrainingAssets(
        source=source,
        source_config=source_config,
        policy=policy,
        ranks=ranks,
        rank4_contract=rank4_contract,
        owners=owners,
        program=program,
        compiler=compiler,
        trainable=trainable,
        native_teachers=native_teachers,
        video_store=video_store,
        language_tokens=language_tokens,
        query_points=int(g2["data"]["query_points"]),
        frozen=(
            *tuple(policy.parameters()),
            *tuple(program.parameters()),
            *tuple(compiler.scale_head.parameters()),
        ),
    )


def prepare_mapping_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> MappingRuntime:
    config = load_shared_compiler_config(args.config)
    if config.get("schema_version") != G3_CONFIG_SCHEMA:
        raise ValueError("G3 mapping requires the active bank-conditioned config")
    total, stop, checkpoints, steps_per_macro = _resolve_runtime(
        args, config, context
    )
    seed_everything(int(config["optimization"]["seed"]), context)
    tasks = _tasks(config, args.data_root, args.asset_root)
    task_by_id = {task.authority_id: task for task in tasks}
    mapping_split = load_mapping_split(config, asset_root=args.asset_root)
    schedule = SharedCompilerMappingSchedule(
        mapping_split, seed=int(config["optimization"]["seed"])
    )
    assets = _load_training_assets(args, config, context, tasks)
    consensus_teachers = FitConsensusTeacherStore(
        assets.native_teachers, mapping_split, assets.rank4_contract
    )

    initialize_deferred_process_group(
        context, rendezvous_root=args.output_dir.parent
    )
    if context.world_size > 1:
        for value in assets.compiler.state_dict().values():
            dist.broadcast(value, src=0)
    optimizer = _optimizer(assets.trainable, config)
    scheduler_lr = _scheduler(
        optimizer, config, total * steps_per_macro
    )
    contract = build_mapping_run_contract(
        args=args,
        config=config,
        context=context,
        source=assets.source,
        policy=assets.policy,
        program=assets.program,
        compiler=assets.compiler,
        native_teacher_store=assets.native_teachers,
        owners=assets.owners,
        mapping_split=mapping_split,
        total_macros=total,
        checkpoint_macros=checkpoints,
        repo_root=REPO_ROOT,
    )
    publish_shared_compiler_run_contract(args, context, contract)
    stage = f"g3_mapping_{args.phase}"
    start, metrics_rows = _resume_cursor(
        args,
        context,
        stage=stage,
        compiler=assets.compiler,
        optimizer=optimizer,
        scheduler=scheduler_lr,
        optimizer_steps_per_macro=steps_per_macro,
        stop=stop,
    )
    torch.cuda.reset_peak_memory_stats(context.device)
    return MappingRuntime(
        args=args,
        config=config,
        context=context,
        tasks=tasks,
        task_by_id=task_by_id,
        mapping_split=mapping_split,
        schedule=schedule,
        video_store=assets.video_store,
        language_tokens=assets.language_tokens,
        policy=assets.policy,
        program=assets.program,
        compiler=assets.compiler,
        owners=assets.owners,
        ranks=assets.ranks,
        rank4_contract=assets.rank4_contract,
        native_teachers=assets.native_teachers,
        consensus_teachers=consensus_teachers,
        query_points=assets.query_points,
        trainable_parameters=assets.trainable,
        frozen_parameters=assets.frozen,
        optimizer=optimizer,
        scheduler_lr=scheduler_lr,
        optimizer_steps_per_macro=steps_per_macro,
        optimizer_steps=start * steps_per_macro,
        total_macros=total,
        stop_after_macro=stop,
        checkpoint_macros=checkpoints,
        start_macro=start,
        metrics_rows=metrics_rows,
        run_contract=contract,
    )


def _macro_groups(
    runtime: MappingRuntime, macro: int
) -> tuple[tuple[int, ...], ...]:
    if runtime.args.mode == "formal":
        return runtime.schedule.task_groups(macro)
    return (tuple(map(int, runtime.config["profile_defaults"]["task_group"])),)


def run_mapping_macro(
    runtime: MappingRuntime, macro: int, run_started: float
) -> dict[str, Any]:
    tick = time.monotonic()
    records = []
    updates = []
    for update, group in enumerate(_macro_groups(runtime, macro)):
        step_records, step = run_mapping_optimizer_step(
            runtime,
            macro=macro,
            update=update,
            task_group=group,
        )
        records.extend(step_records)
        updates.append(step)
    expected = 30 if runtime.args.mode == "formal" else 6
    if len(records) != expected:
        raise RuntimeError("G3 mapping macro lost its fixed logical task count")
    recoveries = [
        float(row["mean_best_recovery"])
        for row in records
        for _ in (0,)
    ] + [
        float(row["companion"]["mean_best_recovery"])
        for row in records
    ]
    return {
        "macro": macro + 1,
        "rank": runtime.context.rank,
        "phase": runtime.args.phase,
        "logical_task_count": len(records),
        "K1_condition_count": 2 * len(records),
        "role_counts": {
            role: sum(row["role"] == role for row in records)
            for role in ("meta_fit", "target_fit")
        },
        "mean_mapping_loss": sum(float(row["mapping_loss"]) for row in records)
        / len(records),
        "mean_input_subspace_loss": sum(
            float(row["input_subspace_loss"]) for row in records
        )
        / len(records),
        "mean_output_subspace_loss": sum(
            float(row["output_subspace_loss"]) for row in records
        )
        / len(records),
        "mean_update_direction_loss": sum(
            float(row["update_direction_loss"]) for row in records
        )
        / len(records),
        "mean_companion_mapping_loss": sum(
            float(row["companion"]["mapping_loss"]) for row in records
        )
        / len(records),
        "mean_cross_video_loss": sum(
            float(row["cross_video_loss"]) for row in records
        )
        / len(records),
        "mean_best_recovery": sum(recoveries) / len(recoveries),
        "optimizer_step_end": runtime.optimizer_steps,
        "optimizer_updates": updates,
        "conditions": records,
        "native_teacher_tensor_reads": runtime.native_teachers.tensor_reads,
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
            runtime.context.device
        ),
        "max_cuda_reserved_bytes": torch.cuda.max_memory_reserved(
            runtime.context.device
        ),
        "macro_seconds": time.monotonic() - tick,
        "elapsed_seconds": time.monotonic() - run_started,
    }


def train(args: argparse.Namespace) -> None:
    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    runtime: MappingRuntime | None = None
    try:
        runtime = prepare_mapping_runtime(args, context)
        started = time.monotonic()
        stage = f"g3_mapping_{args.phase}"
        for macro in range(runtime.start_macro, runtime.stop_after_macro):
            row = run_mapping_macro(runtime, macro, started)
            if context.is_main:
                append_jsonl(args.output_dir / "metrics.jsonl", row)
                runtime.metrics_rows += 1
                if (macro + 1) % args.log_every == 0:
                    print(json.dumps(row, sort_keys=True), flush=True)
            if macro + 1 in runtime.checkpoint_macros:
                save_ecp_checkpoint(
                    output_dir=args.output_dir,
                    macro=macro + 1,
                    stage=stage,
                    context=context,
                    model=runtime.compiler,
                    optimizer=runtime.optimizer,
                    scheduler=runtime.scheduler_lr,
                    run_contract_schema=MAPPING_RUN_SCHEMA,
                    metrics_rows=runtime.metrics_rows,
                )
        if context.is_main:
            completion = {
                "stage": stage,
                "completed_macros": runtime.stop_after_macro,
                "total_macros": runtime.total_macros,
            }
            write_json_atomic(args.output_dir / "segment_completion.json", completion)
            if runtime.stop_after_macro == runtime.total_macros:
                write_json_atomic(args.output_dir / "completion.json", completion)
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_shared_compiler_g3_v4.json",
    )
    parser.add_argument("--phase", choices=("f3",), required=True)
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-macro", type=int)
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
        raise ValueError("G3 mapping log interval must be positive")
    return args
