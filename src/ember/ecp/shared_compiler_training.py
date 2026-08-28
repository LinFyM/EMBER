"""Distributed task-equal training for the frozen-Program G3 compiler."""

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

from ember.batched_lora import BatchedLoRAInference
from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.ecp.contracts import TargetOwner, build_target_owners
from ember.ecp.natural_program import NaturalProgramModel
from ember.ecp.natural_program_data import (
    NaturalProgramSchedule,
    NaturalProgramTask,
    load_natural_program_tasks,
)
from ember.ecp.shared_compiler import SharedNativeFactorCompiler
from ember.ecp.shared_compiler_assets import (
    G3_CONFIG_SCHEMA,
    SharedCompilerRankAssets,
    SharedTaskMembers,
    authority_path,
    build_frozen_g2_program,
    load_shared_compiler_config,
    load_shared_rank_assets,
    load_shared_task_members,
)
from ember.ecp.shared_compiler_authority import (
    RUN_SCHEMA,
    build_shared_compiler_run_contract,
    publish_shared_compiler_run_contract,
)
from ember.ecp.shared_compiler_effects import SharedEffectBankStore
from ember.ecp.shared_compiler_native_teacher import (
    G3_NATIVE_TEACHER_FORMAL_MACROS,
    NativeTeacherStore,
)
from ember.ecp.shared_compiler_train_step import (
    SharedCompilerTaskLoss,
    run_shared_compiler_optimizer_step,
)
from ember.ecp.stage0_training import stage0_source_authority, tokenize_stage0_languages
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import derive_pi05_lora_rank
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import DistributedContext, read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl, reconcile_metrics
from ember.pi05_source_setup import (
    initialize_deferred_process_group,
    initialize_distributed,
    load_config,
    load_policy,
    load_stats,
    seed_everything,
)
from ember.writer.data import FunctionalQueryDataset, RawTeacherVideoStore
from ember.writer.functional import prepare_frozen_writer_policy


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE = "g3_shared_compiler"


@dataclass
class SharedCompilerRuntime:
    args: argparse.Namespace
    config: dict[str, Any]
    context: DistributedContext
    tasks: tuple[NaturalProgramTask, ...]
    task_by_id: dict[int, NaturalProgramTask]
    members: tuple[SharedTaskMembers, ...]
    schedule: NaturalProgramSchedule
    video_store: RawTeacherVideoStore
    query_dataset: FunctionalQueryDataset
    query_processor: Pi05LiberoProcessor
    language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor]]
    policy: torch.nn.Module
    program: NaturalProgramModel
    compiler: SharedNativeFactorCompiler
    owners: tuple[TargetOwner, ...]
    ranks: SharedCompilerRankAssets
    rank4_contract: Any
    lora: BatchedLoRAInference
    effect_banks: SharedEffectBankStore
    native_teachers: NativeTeacherStore | None
    query_points: int
    selection_parameters: tuple[torch.nn.Parameter, ...]
    scale_video_parameters: tuple[torch.nn.Parameter, ...]
    trainable_parameters: tuple[torch.nn.Parameter, ...]
    frozen_parameters: tuple[torch.nn.Parameter, ...]
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    optimizer_steps_per_macro: int
    optimizer_steps: int
    total_macros: int
    stop_after_macro: int
    checkpoint_macros: tuple[int, ...]
    start_macro: int
    metrics_rows: int
    run_contract: dict[str, Any]

    def close(self) -> None:
        self.lora.close()
        self.video_store.close()
        self.query_dataset.close()


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
) -> tuple[int, int, tuple[int, ...]]:
    cell = config["formal_run" if args.mode == "formal" else "profile_defaults"]
    if context.world_size not in cell["allowed_world_sizes"]:
        raise ValueError("G3 world size is outside its launch contract")
    total = int(cell["total_macros"])
    stop = int(args.stop_after_macro or cell.get("stop_after_macro", total))
    checkpoints = tuple(map(int, cell["checkpoint_macros"]))
    if not 0 < stop <= total:
        raise ValueError("G3 stop macro is outside its segment")
    if args.mode == "formal":
        if stop not in set(map(int, cell["stage_stop_macros"])):
            raise ValueError("formal G3 stop is not pre-registered")
        state = git_state(REPO_ROOT)
        if (
            not git_state_is_clean_pushed_or_frozen_authority(state)
            or state.get("branch") != ""
            or state.get("upstream") is not None
        ):
            raise ValueError("formal G3 requires a clean detached origin/main authority")
    return total, stop, checkpoints


def _trainable_groups(
    compiler: SharedNativeFactorCompiler,
) -> tuple[tuple[torch.nn.Parameter, ...], tuple[torch.nn.Parameter, ...]]:
    ordinary = []
    scale_and_video = []
    for name, parameter in compiler.named_parameters():
        destination = scale_and_video if name.startswith("scale_head.") else ordinary
        destination.append(parameter)
    if not ordinary or not scale_and_video:
        raise ValueError("G3 optimizer parameter ownership changed")
    if len({id(value) for value in (*ordinary, *scale_and_video)}) != len(
        tuple(compiler.parameters())
    ):
        raise ValueError("G3 optimizer parameter groups overlap or omit parameters")
    return tuple(ordinary), tuple(scale_and_video)


def _optimizer(
    selection: tuple[torch.nn.Parameter, ...],
    scale_and_video: tuple[torch.nn.Parameter, ...],
    config: Mapping[str, Any],
) -> torch.optim.AdamW:
    cell = config["optimization"]["optimizer"]
    return torch.optim.AdamW(
        [
            {"params": selection, "lr": float(cell["peak_lr"])},
            {
                "params": scale_and_video,
                "lr": float(cell["scale_and_video_lr"]),
            },
        ],
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
    args: argparse.Namespace,
    context: DistributedContext,
    *,
    compiler: SharedNativeFactorCompiler,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    optimizer_steps_per_macro: int,
    stop: int,
) -> tuple[int, int]:
    start = 0
    expected_metrics = 0
    if args.resume is not None:
        start, expected_metrics = load_ecp_checkpoint(
            checkpoint=args.resume,
            stage=STAGE,
            context=context,
            model=compiler,
            optimizer=optimizer,
            scheduler=scheduler,
            run_contract_schema=RUN_SCHEMA,
        )
        if int(scheduler.last_epoch) != start * optimizer_steps_per_macro:
            raise ValueError("G3 resume optimizer-step cursor changed")
    if not 0 <= start < stop:
        raise ValueError("G3 resume cursor is outside this segment")
    rows = (
        reconcile_metrics(
            args.output_dir / "metrics.jsonl",
            start,
            expected_metrics,
            cursor_key="macro",
        )
        if context.is_main
        else 0
    )
    return start, rows


def _native_teacher_task_ids(schedule: NaturalProgramSchedule) -> set[int]:
    output = set()
    for macro in range(G3_NATIVE_TEACHER_FORMAL_MACROS):
        for task_id in schedule.training_task_ids(macro):
            if schedule.sample(task_id, macro).k == 1:
                output.add(task_id)
    if len(output) != 50:
        raise ValueError("G3 formal K1 teacher task coverage changed")
    return output


def prepare_runtime(
    args: argparse.Namespace,
    context: DistributedContext,
    *,
    load_native_teachers: bool = False,
) -> SharedCompilerRuntime:
    config = load_shared_compiler_config(args.config)
    if load_native_teachers and config.get("schema_version") != G3_CONFIG_SCHEMA:
        raise ValueError("G3 training requires the active native-teacher config")
    total, stop, checkpoints = _resolve_runtime(args, config, context)
    seed_everything(int(config["optimization"]["seed"]), context)
    tasks = _tasks(config, args.data_root, args.asset_root)
    members = load_shared_task_members(config, tasks, asset_root=args.asset_root)
    schedule = NaturalProgramSchedule(
        tasks,
        seed=int(config["optimization"]["seed"]),
        query_points=2,
    )
    formal_steps = len(schedule.optimizer_task_groups(0, tasks_per_role=1))
    optimizer_steps_per_macro = formal_steps if args.mode == "formal" else 1

    expected_checkpoint = authority_path(
        config, "source_checkpoint", asset_root=args.asset_root
    )
    expected_tokenizer = authority_path(
        config, "tokenizer", asset_root=args.asset_root
    )
    expected_effect_bank = authority_path(
        config, "shared_effect_bank", asset_root=args.asset_root
    )
    if (
        args.checkpoint != expected_checkpoint
        or args.source_run != expected_checkpoint.parent.parent
        or args.tokenizer_path != expected_tokenizer
        or args.effect_bank_root != expected_effect_bank
    ):
        raise ValueError("G3 source, tokenizer, or effect-bank authority changed")
    source = stage0_source_authority(args)
    source_config = load_config(
        authority_path(config, "source_base_config", asset_root=args.asset_root)
    )
    policy = load_policy(Path(source["model_path"]), source_config, context.device)
    policy.requires_grad_(False).eval()
    rank_assets = load_shared_rank_assets(
        config,
        asset_root=args.asset_root,
        held_global_ids=set(map(int, config["fold"]["target_held_task_ids"])),
        device=context.device,
    )
    owners = build_target_owners(rank_assets.contract)
    rank4_contract = derive_pi05_lora_rank(rank_assets.contract, rank=4)
    program = build_frozen_g2_program(
        config,
        asset_root=args.asset_root,
        owners=owners,
        device=context.device,
    )
    prepare_frozen_writer_policy(policy, rank_assets.contract)
    lora = BatchedLoRAInference(policy, rank_assets.contract)
    compiler = SharedNativeFactorCompiler(
        owners,
        program_width=int(config["model"]["program_width"]),
        event_slots=int(config["model"]["event_slots"]),
        key_width=int(config["model"]["key_width"]),
        maximum_video_correction=float(
            config["model"]["maximum_video_correction"]
        ),
        video_score_bound=float(config["model"]["video_score_bound"]),
    ).to(context.device)
    compiler.train()

    initialize_deferred_process_group(
        context, rendezvous_root=args.output_dir.parent
    )
    if context.world_size > 1:
        for value in compiler.state_dict().values():
            dist.broadcast(value, src=0)
    selection_parameters, scale_video_parameters = _trainable_groups(compiler)
    trainable = (*selection_parameters, *scale_video_parameters)
    frozen = tuple(policy.parameters()) + tuple(program.parameters())
    optimizer = _optimizer(selection_parameters, scale_video_parameters, config)
    scheduler = _scheduler(
        optimizer, config, total * optimizer_steps_per_macro
    )
    fit_ids = {
        task.authority_id
        for task in tasks
        if task.role in {"meta_fit", "target_fit"}
    }
    effect_banks = SharedEffectBankStore(
        args.effect_bank_root,
        contract=rank_assets.contract,
        owners=owners,
        expected_task_ids=fit_ids,
        device=context.device,
    )
    native_teachers = None
    if load_native_teachers:
        native_teachers = NativeTeacherStore(
            authority_path(
                config, "native_teacher_manifest", asset_root=args.asset_root
            ),
            contract=rank4_contract,
            expected_fit_task_ids=_native_teacher_task_ids(schedule),
            expected_full_fit_task_ids=fit_ids,
            device=context.device,
        )
    video_store = RawTeacherVideoStore(
        tuple(task.writer_authority() for task in tasks),
        frame_stride=int(config["data"]["frame_stride"]),
        max_open_files=8,
    )
    fit_authorities = tuple(
        task.writer_authority()
        for task in tasks
        if task.role in {"meta_fit", "target_fit"}
    )
    query_dataset = FunctionalQueryDataset(
        fit_authorities,
        demo_indices=range(50),
        action_chunk_size=int(config["data"]["action_chunk_size"]),
        max_open_files_per_worker=8,
    )
    query_processor = Pi05LiberoProcessor(
        load_stats(source_config, source_config["data"]["active_task_ids"]),
        args.tokenizer_path,
        int(source_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    language_tokens = tokenize_stage0_languages(
        tasks,
        tokenizer_path=args.tokenizer_path,
        max_length=int(source_config["features"]["tokenizer_max_length"]),
        device=context.device,
    )
    g2 = read_json(authority_path(config, "g2_config", asset_root=args.asset_root))
    query_points = int(g2["data"]["query_points"])
    contract = build_shared_compiler_run_contract(
        args=args,
        config=config,
        context=context,
        tasks=tasks,
        members=members,
        source=source,
        policy=policy,
        program=program,
        compiler=compiler,
        native_teacher_store=native_teachers,
        owners=owners,
        total_macros=total,
        checkpoint_macros=checkpoints,
        optimizer_steps_per_macro=optimizer_steps_per_macro,
        repo_root=REPO_ROOT,
    )
    publish_shared_compiler_run_contract(args, context, contract)
    start, metrics_rows = _resume_cursor(
        args,
        context,
        compiler=compiler,
        optimizer=optimizer,
        scheduler=scheduler,
        optimizer_steps_per_macro=optimizer_steps_per_macro,
        stop=stop,
    )
    torch.cuda.reset_peak_memory_stats(context.device)
    return SharedCompilerRuntime(
        args=args,
        config=config,
        context=context,
        tasks=tasks,
        task_by_id={task.authority_id: task for task in tasks},
        members=members,
        schedule=schedule,
        video_store=video_store,
        query_dataset=query_dataset,
        query_processor=query_processor,
        language_tokens=language_tokens,
        policy=policy,
        program=program,
        compiler=compiler,
        owners=owners,
        ranks=rank_assets,
        rank4_contract=rank4_contract,
        lora=lora,
        effect_banks=effect_banks,
        native_teachers=native_teachers,
        query_points=query_points,
        selection_parameters=selection_parameters,
        scale_video_parameters=scale_video_parameters,
        trainable_parameters=trainable,
        frozen_parameters=frozen,
        optimizer=optimizer,
        scheduler=scheduler,
        optimizer_steps_per_macro=optimizer_steps_per_macro,
        optimizer_steps=start * optimizer_steps_per_macro,
        total_macros=total,
        stop_after_macro=stop,
        checkpoint_macros=checkpoints,
        start_macro=start,
        metrics_rows=metrics_rows,
        run_contract=contract,
    )


def _macro_assignments(
    runtime: SharedCompilerRuntime, macro: int
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    assignments = runtime.schedule.optimizer_assignments(
        macro, runtime.context.world_size, tasks_per_role=1
    )
    if runtime.args.mode == "formal":
        return assignments
    pair = tuple(map(int, runtime.config["profile_defaults"]["task_pairs"][macro]))
    roles = {runtime.task_by_id[task_id].role for task_id in pair}
    if len(pair) != 2 or roles != {"meta_fit", "target_fit"}:
        raise RuntimeError("G3 profile pair lost its fit-role contract")
    profile_assignment = (
        (pair,) if runtime.context.world_size == 1 else ((pair[0],), (pair[1],))
    )
    return (profile_assignment,)


def run_shared_compiler_macro(
    runtime: SharedCompilerRuntime, macro: int, run_started: float
) -> dict[str, Any]:
    tick = time.monotonic()
    step_start = runtime.optimizer_steps
    records = []
    updates = []
    local_task_ids = []
    for assignments in _macro_assignments(runtime, macro):
        step_records, update, local = run_shared_compiler_optimizer_step(
            runtime, macro=macro, assignments=assignments
        )
        records.extend(step_records)
        updates.append(update)
        local_task_ids.extend(local)
    records.sort(key=lambda row: int(row["authority_id"]))
    expected = 38 if runtime.args.mode == "formal" else 2
    if len(records) != expected:
        raise RuntimeError("G3 macro lost task-equal coverage")
    mean_names = tuple(SharedCompilerTaskLoss.__dataclass_fields__)
    means = {
        name: sum(float(row[name]) for row in records) / len(records)
        for name in mean_names
    }
    return {
        "macro": macro + 1,
        "rank": runtime.context.rank,
        "global_task_count": len(records),
        "role_counts": {
            role: sum(row["role"] == role for row in records)
            for role in ("meta_fit", "target_fit")
        },
        "K_counts": {
            str(k): sum(int(row["K"]) == k for row in records)
            for k in (1, 2, 4)
        },
        "global_means": means,
        "optimizer_step_start": step_start,
        "optimizer_step_end": runtime.optimizer_steps,
        "optimizer_steps_this_macro": len(updates),
        "optimizer_updates": updates,
        "local_task_ids": local_task_ids,
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
            runtime.context.device
        ),
        "macro_seconds": time.monotonic() - tick,
        "elapsed_seconds": time.monotonic() - run_started,
        "conditions": records,
    }


def train(args: argparse.Namespace) -> None:
    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    runtime: SharedCompilerRuntime | None = None
    try:
        runtime = prepare_runtime(args, context, load_native_teachers=True)
        started = time.monotonic()
        for macro in range(runtime.start_macro, runtime.stop_after_macro):
            row = run_shared_compiler_macro(runtime, macro, started)
            if context.is_main:
                append_jsonl(args.output_dir / "metrics.jsonl", row)
                runtime.metrics_rows += 1
                if (macro + 1) % args.log_every == 0:
                    print(json.dumps(row, sort_keys=True), flush=True)
            if macro + 1 in runtime.checkpoint_macros:
                save_ecp_checkpoint(
                    output_dir=args.output_dir,
                    macro=macro + 1,
                    stage=STAGE,
                    context=context,
                    model=runtime.compiler,
                    optimizer=runtime.optimizer,
                    scheduler=runtime.scheduler,
                    run_contract_schema=RUN_SCHEMA,
                    metrics_rows=runtime.metrics_rows,
                )
        if context.is_main:
            completion = {
                "stage": STAGE,
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
        default=REPO_ROOT / "configs/pi05_ecp_shared_compiler_g3_v2.json",
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--effect-bank-root", type=Path, required=True)
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
        "effect_bank_root",
        "output_dir",
        "resume",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.log_every <= 0:
        raise ValueError("G3 log interval must be positive")
    return args
