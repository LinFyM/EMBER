"""Distributed task-equal training for G2 Natural Program (Pass A)."""

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
from ember.ecp.behavior.codes import (
    BehaviorCodeAuthority,
    load_behavior_code_authority,
    load_program_model_initialization,
)
from ember.ecp.contracts import build_target_owners
from ember.ecp.natural_program import NaturalProgramModel
from ember.ecp.natural_program_authority import (
    RUN_SCHEMA,
    build_natural_program_run_contract,
    publish_natural_program_run_contract,
)
from ember.ecp.natural_program_data import (
    NaturalProgramSchedule,
    NaturalProgramTask,
    load_natural_program_tasks,
)
from ember.ecp.natural_program_labels import NaturalProgramLabelStore
from ember.ecp.natural_program_objective import NaturalProgramLoss
from ember.ecp.natural_program_train_step import (
    run_natural_program_optimizer_step,
)
from ember.ecp.observer_authority import load_frozen_native_observer
from ember.ecp.stage0_training import (
    build_stage0_optimizer,
    load_stage0_config,
    stage0_source_authority,
    tokenize_stage0_languages,
)
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import (
    DistributedContext,
    read_json,
    write_json_atomic,
)
from ember.pi05_source_contract import append_jsonl, reconcile_metrics
from ember.pi05_source_setup import (
    initialize_deferred_process_group,
    initialize_distributed,
    load_config,
    load_policy,
    load_stats,
    seed_everything,
)
from ember.privileged_actions import PrivilegedMetaActionStore
from ember.writer.data import RawTeacherVideoStore


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE = "g2_natural_program"


@dataclass
class NaturalProgramRuntime:
    args: argparse.Namespace
    config: dict[str, Any]
    context: DistributedContext
    tasks: tuple[NaturalProgramTask, ...]
    task_by_id: dict[int, NaturalProgramTask]
    schedule: NaturalProgramSchedule
    video_store: RawTeacherVideoStore
    action_store: PrivilegedMetaActionStore
    label_store: NaturalProgramLabelStore
    language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor]]
    policy: torch.nn.Module
    model: NaturalProgramModel
    trainable_parameters: tuple[torch.nn.Parameter, ...]
    frozen_parameters: tuple[torch.nn.Parameter, ...]
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    tasks_per_rank: int | None
    tasks_per_role_per_optimizer_step: int
    optimizer_steps_per_macro: int
    optimizer_steps: int
    total_macros: int
    stop_after_macro: int
    checkpoint_macros: tuple[int, ...]
    start_macro: int
    metrics_rows: int
    run_contract: dict[str, Any]
    behavior_codes: BehaviorCodeAuthority | None
    initialization: dict[str, Any] | None

    def close(self) -> None:
        self.video_store.close()
        self.action_store.close()
        self.label_store.close()


def _common_config_signature(config: Mapping[str, Any]) -> tuple[Any, ...]:
    data = config.get("data", {})
    model = config.get("model", {})
    objective = config.get("objective", {})
    negatives = int(objective.get("contrastive_negative_languages", 0))
    return (
        data.get("K_values"),
        data.get("video_weights"),
        data.get("robustness_for_every_task"),
        model.get("target_owners"),
        model.get("event_slots"),
        model.get("program_width"),
        model.get("process_fusion_inputs"),
        model.get("temporal_head_inputs"),
        model.get("native_observer_training"),
        model.get("temporal_owner_readout"),
        model.get("canonical_alignment"),
        config.get("gate", {}).get("shuffled_or_reversed_use"),
        negatives > 0 and negatives % 2 == 0,
        objective.get("temporal_residual_mode"),
    )


def _behavior_config_signature(config: Mapping[str, Any]) -> tuple[Any, ...]:
    behavior = config.get("behavior_alignment")
    if not isinstance(behavior, Mapping):
        return ()
    return (
        behavior.get("selected_targets"),
        behavior.get("families"),
        int(behavior.get("dimension", 0)),
        behavior.get("kind"),
        behavior.get("program_blocks"),
        int(behavior.get("internal_fold", -1)),
        int(config["optimization"]["tasks_per_role_per_optimizer_step"]),
        float(config["objective"]["weights"].get("behavior_alignment", 0.0)) > 0.0,
        config.get("initialization", {}).get("kind"),
    )


def load_natural_program_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    schema = config.get("schema_version")
    common = (
        [1, 2, 4],
        "uniform_beta_1_over_K",
        True,
        38,
        8,
        128,
        ["native_process", "native_uncertainty"],
        ["P_process", "rho", "tau"],
        "frozen_stage0_v3",
        "fixed_owner_specific_linear_v1",
        "boundary_anchored_forward_only_dp_v2",
        False,
        True,
        "query_centered_mse_v1",
    )
    if _common_config_signature(config) != common:
        raise ValueError("unsupported G2 Natural Program config")
    tasks_per_role = int(
        config.get("optimization", {}).get("tasks_per_role_per_optimizer_step", 0)
    )
    if schema == "ember_ecp_natural_program_g2_v1":
        if config.get("behavior_alignment") is not None or tasks_per_role != 2:
            raise ValueError("legacy G2 config unexpectedly enables behavior alignment")
    elif schema == "ember_ecp_natural_program_g2_behavior_kernel_v3":
        expected_behavior = (
            [0, 16, 34, 1, 17, 35, 36, 37],
            ["q", "q", "q", "v", "v", "v", "action_in", "action_out"],
            16,
            "decoder_free_program_kernel_v1",
            [
                "P_lang",
                "P_scene",
                "sqrt_rho_P_process",
                "sqrt_rho_sigma",
                "rho",
                "tau",
            ],
            0,
            5,
            True,
            "qualified_g2_model_only_strict_fresh_optimizer",
        )
        if _behavior_config_signature(config) != expected_behavior:
            raise ValueError("unsupported behavior-aligned G2 config")
    else:
        raise ValueError("unsupported G2 Natural Program config")
    return config


def _authority(config: Mapping[str, Any], name: str) -> Path:
    path = REPO_ROOT / str(config["authorities"][name])
    if not path.is_file():
        raise FileNotFoundError(f"G2 authority is missing: {name}")
    return path


def _asset_authority(
    args: argparse.Namespace, config: Mapping[str, Any], name: str
) -> Path:
    path = args.asset_root / str(config["authorities"][name])
    if not path.exists():
        raise FileNotFoundError(f"G2 retained asset is missing: {name}")
    return path


def _resolve_runtime(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> tuple[int, int, tuple[int, ...]]:
    cell = config["formal_run" if args.mode == "formal" else "profile_defaults"]
    if context.world_size not in cell["allowed_world_sizes"]:
        raise ValueError("G2 world size is outside its launch contract")
    total = int(cell["total_macros"])
    stop = int(args.stop_after_macro or cell.get("stop_after_macro", total))
    checkpoints = tuple(map(int, cell["checkpoint_macros"]))
    if not 0 < stop <= total:
        raise ValueError("G2 stop macro is outside its segment")
    if args.mode == "formal":
        if stop not in set(map(int, cell["stage_stop_macros"])):
            raise ValueError("formal G2 stop is not pre-registered")
        state = git_state(REPO_ROOT)
        if (
            not git_state_is_clean_pushed_or_frozen_authority(state)
            or state.get("branch") != ""
            or state.get("upstream") is not None
        ):
            raise ValueError("formal G2 requires a clean detached origin/main authority")
    return total, stop, checkpoints


def _scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    total_optimizer_steps: int,
    warmup_optimizer_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    cell = config["optimization"]["scheduler"]
    floor = float(cell["decay_lr"]) / float(cell["peak_lr"])

    def scale(step: int) -> float:
        if step < warmup_optimizer_steps:
            return (step + 1) / warmup_optimizer_steps
        progress = (step - warmup_optimizer_steps) / max(
            total_optimizer_steps - warmup_optimizer_steps, 1
        )
        return floor + (1.0 - floor) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _tasks(config: Mapping[str, Any], data_root: Path) -> tuple[NaturalProgramTask, ...]:
    fold = config["fold"]
    return load_natural_program_tasks(
        meta_protocol_path=_authority(config, "meta_protocol"),
        source_manifest_path=_authority(config, "source_manifest"),
        target_manifest_path=_authority(config, "target_manifest"),
        data_root=data_root,
        target_fit_ids=fold["target_fit_task_ids"],
        target_held_ids=fold["target_held_task_ids"],
        held_meta_fold=int(fold["meta_held_fold"]),
    )


def _load_program_model(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> tuple[dict[str, Any], dict[str, Any], torch.nn.Module, NaturalProgramModel]:
    source = stage0_source_authority(args)
    source_config = load_config(_authority(config, "source_base_config"))
    policy = load_policy(Path(source["model_path"]), source_config, context.device)
    policy.requires_grad_(False).eval()
    owners = build_target_owners(
        load_pi05_lora_contract(_authority(config, "lora_contract"))
    )
    native = load_frozen_native_observer(
        stage0_config=load_stage0_config(_authority(config, "stage0_config")),
        owners=owners,
        native_checkpoint=_asset_authority(
            args, config, "native_observer_checkpoint"
        ),
        device=context.device,
        max_frames_per_call=int(config["model"]["max_frames_per_call"]),
    )
    model = NaturalProgramModel(
        native.encoder,
        prefix_width=int(config["model"]["prefix_width"]),
        width=int(config["model"]["program_width"]),
        owners=int(config["model"]["target_owners"]),
        event_slots=int(config["model"]["event_slots"]),
        action_phases=int(config["model"]["action_phases"]),
        predicate_slots=int(config["model"]["predicate_slots"]),
    ).to(context.device)
    model.requires_grad_(True)
    # Stage 0 v3 is an established observer authority.  G2 qualifies the new
    # Program readers and alignment on top of that fixed evidence instead of
    # relearning (and potentially erasing) the observer's event grounding.
    model.encoder.requires_grad_(False).eval()
    return source, source_config, policy, model


def _open_program_data(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    tasks: tuple[NaturalProgramTask, ...],
    source_config: Mapping[str, Any],
    context: DistributedContext,
) -> tuple[
    RawTeacherVideoStore,
    PrivilegedMetaActionStore,
    NaturalProgramLabelStore,
    dict[int, tuple[torch.Tensor, torch.Tensor]],
]:
    authorities = tuple(task.writer_authority() for task in tasks)
    stats = load_stats(source_config, source_config["data"]["active_task_ids"])
    return (
        RawTeacherVideoStore(
            authorities,
            frame_stride=int(config["data"]["frame_stride"]),
            max_open_files=8,
        ),
        PrivilegedMetaActionStore(
            authorities,
            action_q01=stats["action"]["q01"],
            action_q99=stats["action"]["q99"],
            phase_count=int(config["model"]["action_phases"]),
            max_open_files=8,
        ),
        NaturalProgramLabelStore(
            args.label_root,
            tasks=tasks,
            predicate_slots=int(config["model"]["predicate_slots"]),
            max_open_tasks=8,
        ),
        tokenize_stage0_languages(
            tasks,
            tokenizer_path=args.tokenizer_path,
            max_length=int(source_config["features"]["tokenizer_max_length"]),
            device=context.device,
        ),
    )


def _resume_cursor(
    args: argparse.Namespace,
    context: DistributedContext,
    *,
    model: NaturalProgramModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    optimizer_steps_per_macro: int,
    stop: int,
) -> tuple[int, int]:
    start_macro = 0
    expected_metrics = 0
    if args.resume is not None:
        start_macro, expected_metrics = load_ecp_checkpoint(
            checkpoint=args.resume,
            stage=STAGE,
            context=context,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            run_contract_schema=RUN_SCHEMA,
        )
        expected_steps = start_macro * optimizer_steps_per_macro
        if int(scheduler.last_epoch) != expected_steps:
            raise ValueError("G2 resume optimizer-step cursor changed")
    if not 0 <= start_macro < stop:
        raise ValueError("G2 resume cursor is outside this segment")
    metrics_rows = (
        reconcile_metrics(
            args.output_dir / "metrics.jsonl",
            start_macro,
            expected_metrics,
            cursor_key="macro",
        )
        if context.is_main
        else 0
    )
    return start_macro, metrics_rows


def _behavior_alignment_assets(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    model: NaturalProgramModel,
    device: torch.device,
) -> tuple[BehaviorCodeAuthority | None, dict[str, Any] | None]:
    behavior = config.get("behavior_alignment")
    if behavior is None:
        return None, None
    authority = load_behavior_code_authority(
        _asset_authority(args, config, "behavior_codes"),
        asset_root=args.asset_root,
        device=device,
    )
    if authority.selected_targets != tuple(map(int, behavior["selected_targets"])):
        raise ValueError("G2 behavior-code target order changed")
    initialization = load_program_model_initialization(
        model,
        _asset_authority(args, config, "initial_program_checkpoint"),
        device=device,
        allowed_new_prefix=config["initialization"].get("allowed_new_prefix"),
        expected_macro=int(config["initialization"]["checkpoint_macro"]),
    )
    return authority, initialization


def _program_optimizer(
    model: NaturalProgramModel,
    config: Mapping[str, Any],
    *,
    total_optimizer_steps: int,
    warmup_optimizer_steps: int,
) -> tuple[
    tuple[torch.nn.Parameter, ...],
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = build_stage0_optimizer(trainable, config["optimization"])
    scheduler = _scheduler(
        optimizer,
        config,
        total_optimizer_steps,
        warmup_optimizer_steps,
    )
    return trainable, optimizer, scheduler


def _training_schedule(
    *,
    config: Mapping[str, Any],
    mode: str,
    tasks: tuple[NaturalProgramTask, ...],
    behavior_codes: BehaviorCodeAuthority | None,
) -> tuple[NaturalProgramSchedule, int, int]:
    schedule = NaturalProgramSchedule(
        tasks,
        seed=int(config["data"]["pair_seed"]),
        query_points=int(config["data"]["query_points"]),
        gradient_task_ids=(
            behavior_codes.fit_task_ids if behavior_codes is not None else None
        ),
    )
    formal_tasks_per_role = int(
        config["optimization"]["tasks_per_role_per_optimizer_step"]
    )
    tasks_per_role = int(
        formal_tasks_per_role
        if mode == "formal"
        else config["profile_defaults"]["tasks_per_role_per_optimizer_step"]
    )
    formal_steps = len(
        schedule.optimizer_task_groups(0, tasks_per_role=formal_tasks_per_role)
    )
    return schedule, tasks_per_role, formal_steps if mode == "formal" else 1


def prepare_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> NaturalProgramRuntime:
    config = load_natural_program_config(args.config)
    total, stop, checkpoints = _resolve_runtime(args, config, context)
    seed_everything(int(config["optimization"]["seed"]), context)
    tasks = _tasks(config, args.data_root)
    source, source_config, policy, model = _load_program_model(
        args, config, context
    )
    behavior_codes, initialization = _behavior_alignment_assets(
        args, config, model, context.device
    )
    (
        schedule,
        tasks_per_role_per_optimizer_step,
        optimizer_steps_per_macro,
    ) = _training_schedule(
        config=config,
        mode=args.mode,
        tasks=tasks,
        behavior_codes=behavior_codes,
    )
    initialize_deferred_process_group(
        context, rendezvous_root=args.output_dir.parent
    )
    if context.world_size > 1:
        for value in model.state_dict().values():
            dist.broadcast(value, src=0)
    frozen_parameters = tuple(policy.parameters()) + tuple(model.encoder.parameters())
    trainable_parameters, optimizer, scheduler = _program_optimizer(
        model,
        config,
        total_optimizer_steps=total * optimizer_steps_per_macro,
        warmup_optimizer_steps=(
            int(config["optimization"]["scheduler"]["warmup_macros"])
            * optimizer_steps_per_macro
        ),
    )

    video_store, action_store, label_store, language_tokens = _open_program_data(
        args, config, tasks, source_config, context
    )
    native_checkpoint = _asset_authority(
        args, config, "native_observer_checkpoint"
    )
    contract = build_natural_program_run_contract(
        runtime_args=args,
        config=config,
        context=context,
        tasks=tasks,
        source=source,
        policy=policy,
        model=model,
        total_macros=total,
        checkpoint_macros=checkpoints,
        optimizer_steps_per_macro=optimizer_steps_per_macro,
        tasks_per_role_per_optimizer_step=tasks_per_role_per_optimizer_step,
        repo_root=REPO_ROOT,
        native_checkpoint=native_checkpoint,
        behavior_codes=behavior_codes,
        initialization=initialization,
    )
    publish_natural_program_run_contract(args, context, contract)

    start_macro, metrics_rows = _resume_cursor(
        args,
        context,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        optimizer_steps_per_macro=optimizer_steps_per_macro,
        stop=stop,
    )
    model.train()
    model.encoder.eval()
    torch.cuda.reset_peak_memory_stats(context.device)
    return NaturalProgramRuntime(
        args=args,
        config=config,
        context=context,
        tasks=tasks,
        task_by_id={task.authority_id: task for task in tasks},
        schedule=schedule,
        video_store=video_store,
        action_store=action_store,
        label_store=label_store,
        language_tokens=language_tokens,
        policy=policy,
        model=model,
        trainable_parameters=trainable_parameters,
        frozen_parameters=frozen_parameters,
        optimizer=optimizer,
        scheduler=scheduler,
        tasks_per_rank=(
            None
            if args.mode == "formal"
            else int(config["profile_defaults"]["tasks_per_rank_per_macro"])
        ),
        tasks_per_role_per_optimizer_step=tasks_per_role_per_optimizer_step,
        optimizer_steps_per_macro=optimizer_steps_per_macro,
        optimizer_steps=start_macro * optimizer_steps_per_macro,
        total_macros=total,
        stop_after_macro=stop,
        checkpoint_macros=checkpoints,
        start_macro=start_macro,
        metrics_rows=metrics_rows,
        run_contract=contract,
        behavior_codes=behavior_codes,
        initialization=initialization,
    )


def _macro_optimizer_assignments(
    runtime: NaturalProgramRuntime, macro: int
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    assignments = runtime.schedule.optimizer_assignments(
        macro,
        runtime.context.world_size,
        tasks_per_role=runtime.tasks_per_role_per_optimizer_step,
    )
    if runtime.tasks_per_rank is None:
        return assignments
    return (
        tuple(group[: runtime.tasks_per_rank] for group in assignments[0]),
    )


def run_natural_program_macro(
    runtime: NaturalProgramRuntime, macro: int, run_started: float
) -> dict[str, Any]:
    tick = time.monotonic()
    optimizer_step_start = runtime.optimizer_steps
    global_records: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    local_task_ids: list[int] = []
    for assignments in _macro_optimizer_assignments(runtime, macro):
        records, update, local = run_natural_program_optimizer_step(
            runtime, macro=macro, assignments=assignments
        )
        global_records.extend(records)
        updates.append(update)
        local_task_ids.extend(local)
    global_records.sort(key=lambda row: int(row["authority_id"]))
    if runtime.tasks_per_rank is None and [
        int(row["authority_id"]) for row in global_records
    ] != sorted(runtime.schedule.training_task_ids(macro)):
        raise RuntimeError("G2 distributed macro ownership changed")
    mean_names = (
        *NaturalProgramLoss.__dataclass_fields__.keys(),
        "mean_active_events",
        "one_event_fraction",
        "mean_presence_sum",
        "mean_cross_video_sigma",
    )
    means = {
        name: sum(float(row[name]) for row in global_records) / len(global_records)
        for name in mean_names
    }
    return {
        "macro": macro + 1,
        "rank": runtime.context.rank,
        "global_task_count": len(global_records),
        "role_counts": {
            role: sum(row["role"] == role for row in global_records)
            for role in ("meta_fit", "target_fit")
        },
        "K_counts": {
            str(k): sum(int(row["K"]) == k for row in global_records)
            for k in (1, 2, 4)
        },
        "local_task_ids": local_task_ids,
        "global_means": means,
        "optimizer_step_start": optimizer_step_start,
        "optimizer_step_end": runtime.optimizer_steps,
        "optimizer_steps_this_macro": len(updates),
        "optimizer_updates": updates,
        "owner_query_gradient_norm_before_clip": updates[-1][
            "owner_query_gradient_norm_before_clip"
        ],
        "behavior_kernel_gradient_norm": updates[-1][
            "behavior_kernel_gradient_norm"
        ],
        "behavior_program_gradient_norm_before_clip": updates[-1][
            "behavior_program_gradient_norm_before_clip"
        ],
        "gradient_norm_before_clip": updates[-1]["gradient_norm_before_clip"],
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "macro_seconds": time.monotonic() - tick,
        "elapsed_seconds": time.monotonic() - run_started,
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
            runtime.context.device
        ),
        "conditions": global_records,
    }


def train(args: argparse.Namespace) -> None:
    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    runtime: NaturalProgramRuntime | None = None
    try:
        runtime = prepare_runtime(args, context)
        started = time.monotonic()
        for macro in range(runtime.start_macro, runtime.stop_after_macro):
            row = run_natural_program_macro(runtime, macro, started)
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
                    model=runtime.model,
                    optimizer=runtime.optimizer,
                    scheduler=runtime.scheduler,
                    run_contract_schema=RUN_SCHEMA,
                    metrics_rows=runtime.metrics_rows,
                )
                if args.mode == "formal":
                    from ember.ecp.natural_program_gate import (
                        evaluate_natural_program_gate,
                    )

                    gate = evaluate_natural_program_gate(runtime, macro + 1)
                    if context.is_main:
                        print(
                            json.dumps(
                                {
                                    "g2_gate_macro": macro + 1,
                                    "passed": gate["passed"],
                                    "metrics": gate["metrics"],
                                },
                                sort_keys=True,
                            ),
                            flush=True,
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
        default=(
            REPO_ROOT
            / "configs/pi05_ecp_natural_program_g2_behavior_kernel_v3.json"
        ),
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
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
        "label_root",
        "output_dir",
        "resume",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.log_every <= 0:
        raise ValueError("G2 log interval must be positive")
    return args
