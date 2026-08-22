"""Runtime for OCPB Stage 1 task-equal closed-loop calibration."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.ecp.checkpoint import (
    ECP_CHECKPOINT_SCHEMA,
    checkpoint_macro,
    load_ecp_checkpoint,
    save_ecp_checkpoint,
)
from ember.ecp.stage1 import ECPStage1Model
from ember.ecp.stage1_outcome_config import (
    RUN_SCHEMA,
    STAGE,
    load_outcome_config,
    outcome_asset_authority,
    outcome_repo_authority,
)
from ember.ecp.stage1_outcome_contract import publish_outcome_run_contract
from ember.ecp.stage1_data import (
    ECPStage1EvidenceBank,
    ECPStage1Task,
    build_stage1_video_store,
    load_stage1_evidence_bank,
    load_stage1_tasks,
    tokenize_stage1_languages,
)
from ember.ecp.stage1_support import (
    CachedPolicySupportPanel,
    PolicySupportBank,
    cache_policy_support_panels,
    load_policy_support_bank,
)
from ember.ecp.stage1_training import (
    REPO_ROOT,
    load_stage1_authorities,
    load_stage1_config,
    stage1_asset_authority,
    stage1_repo_authority,
)
from ember.lora import LoRAContract
from ember.pi05_assets import prepare_libero_config
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import (
    DistributedContext,
    read_json,
    write_json_atomic,
)
from ember.pi05_source_contract import append_jsonl, reconcile_metrics
from ember.pi05_source_setup import (
    initialize_deferred_process_group,
    initialize_distributed,
    load_stats,
    seed_everything,
)
from ember.reward.protocol import RewardTask, SUITE_HORIZONS
from ember.reward.rollout import RandomResetEnvironmentPool
from ember.writer.data import RawTeacherVideoStore
from ember.writer.topology import visible_physical_cuda_index


@dataclass
class ECPStage1OutcomeRuntime:
    args: argparse.Namespace
    config: dict[str, Any]
    base_config: dict[str, Any]
    context: DistributedContext
    tasks: tuple[ECPStage1Task, ...]
    local_tasks: tuple[ECPStage1Task, ...]
    task_by_ordinal: dict[int, ECPStage1Task]
    video_store: RawTeacherVideoStore
    language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor]]
    evidence_bank: ECPStage1EvidenceBank
    support_bank: PolicySupportBank
    support_panels: dict[tuple[int, int], CachedPolicySupportPanel]
    reward_tasks: dict[int, RewardTask]
    policy: torch.nn.Module
    processor: Pi05LiberoProcessor
    observer: Any
    contract: LoRAContract
    identity_state: Mapping[str, torch.Tensor]
    prior_state: Mapping[str, torch.Tensor]
    model: ECPStage1Model
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    trainable_parameters: tuple[torch.nn.Parameter, ...]
    env_pool: RandomResetEnvironmentPool | None
    assignments: tuple[tuple[int, ...], ...]
    start_macro: int
    stop_macro: int
    total_macros: int
    checkpoint_macros: tuple[int, ...]
    metrics_rows: int


def _configure_egl(context: DistributedContext) -> None:
    expected = {
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_EGL_DEVICE_ID": str(
            visible_physical_cuda_index(context.local_rank)
        ),
    }
    for name, value in expected.items():
        observed = os.environ.get(name)
        if observed not in {None, value}:
            raise ValueError(f"outcome calibration {name} mapping changed")
        os.environ[name] = value


def _runtime_limits(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> tuple[int, int, int, tuple[int, ...]]:
    cell = (
        config["optimization"]
        if args.mode == "formal"
        else config["profile_defaults"]
    )
    total = int(cell["total_macros"])
    checkpoints = tuple(int(value) for value in cell["checkpoint_macros"])
    start = checkpoint_macro(args.resume)
    if args.stop_after_macro is not None:
        stop = int(args.stop_after_macro)
    elif args.mode == "formal":
        stop = min(value for value in checkpoints if value > start)
    else:
        stop = total
    if (
        context.world_size != int(cell["world_size"])
        or not 0 <= start < stop <= total
        or args.mode == "formal"
        and (
            stop not in set(int(value) for value in config["optimization"]["stage_stop_macros"])
            or not git_state_is_clean_pushed_or_frozen_authority(
                git_state(REPO_ROOT)
            )
            or os.environ.get("NCCL_P2P_DISABLE") != "1"
        )
    ):
        raise ValueError("ECP Stage 1 outcome launch authority changed")
    return start, stop, total, checkpoints


def _task_cost(task: ECPStage1Task, frame_stride: int) -> float:
    sampled = [
        (int(length) - 1) // frame_stride + 1 for length in task.episode_lengths
    ]
    return sum(sampled) / len(sampled)


def _task_assignments(
    tasks: Sequence[ECPStage1Task], world_size: int, frame_stride: int
) -> tuple[tuple[int, ...], ...]:
    bins: list[list[int]] = [[] for _ in range(world_size)]
    loads = [0.0] * world_size
    ordered = sorted(
        tasks,
        key=lambda task: (-_task_cost(task, frame_stride), task.ordinal),
    )
    for task in ordered:
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        bins[rank].append(task.ordinal)
        loads[rank] += _task_cost(task, frame_stride)
    return tuple(tuple(sorted(values)) for values in bins)


def _reward_tasks(
    base_config: Mapping[str, Any], tasks: Sequence[ECPStage1Task]
) -> dict[int, RewardTask]:
    manifest = read_json(stage1_repo_authority(base_config, "target_manifest"))
    rows = {int(row["global_task_id"]): row for row in manifest["tasks"]}
    result = {}
    for task in tasks:
        row = rows[task.global_task_id]
        bddl = row["bddl"]
        result[task.ordinal] = RewardTask(
            suite=task.suite,
            task_id=task.task_id,
            global_task_id=task.global_task_id,
            split_role="train",
            language=task.language,
            problem_folder=str(row["problem_folder"]),
            bddl_file=str(bddl["filename"]),
            bddl_bytes=int(bddl["bytes"]),
            bddl_sha256=None,
            horizon=SUITE_HORIZONS[task.suite],
        )
    return result


def _load_initialization(
    *,
    checkpoint: Path,
    model: ECPStage1Model,
    device: torch.device,
    initialization: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    weights = checkpoint / "ecp.safetensors"
    run_contract = read_json(checkpoint.parent.parent / "run_contract.json")
    expected_stage = str(initialization["stage"])
    expected_schema = str(initialization["run_contract_schema"])
    expected_macro = int(initialization["checkpoint_macro"])
    if (
        manifest.get("schema_version") != ECP_CHECKPOINT_SCHEMA
        or manifest.get("stage") != expected_stage
        or manifest.get("run_contract_schema") != expected_schema
        or int(manifest.get("next_macro", -1)) != expected_macro
        or checkpoint_macro(checkpoint) != expected_macro
        or int(manifest.get("world_size", -1)) != 6
        or run_contract.get("schema_version") != expected_schema
        or run_contract.get("stage") != expected_stage
        or not weights.is_file()
        or weights.stat().st_size
        != int(manifest.get("files", {}).get(weights.name, {}).get("bytes", -1))
    ):
        raise ValueError("OCPB initialization authority changed")
    model.load_state_dict(load_file(str(weights), device=str(device)), strict=True)
    return {
        "checkpoint": str(checkpoint.resolve()),
        "weights": str(weights.resolve()),
        "weights_bytes": weights.stat().st_size,
        "training_commit": str(run_contract["git"]["commit"]),
        "stage": expected_stage,
        "run_contract_schema": expected_schema,
        "checkpoint_macro": expected_macro,
        "restore_optimizer_and_rank_rng_in_formal": bool(
            initialization["restore_optimizer_and_rank_rng_in_formal"]
        ),
    }


def _optimizer(
    model: ECPStage1Model, config: Mapping[str, Any]
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    cell = config["optimization"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cell["learning_rate"]),
        betas=tuple(float(value) for value in cell["betas"]),
        eps=float(cell["eps"]),
        weight_decay=float(cell["weight_decay"]),
    )
    return optimizer, torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda _step: 1.0
    )


def _prepare_libero_paths(
    output_dir: Path, context: DistributedContext
) -> dict[str, str]:
    payload: list[Any] = [None]
    if context.is_main:
        try:
            payload[0] = prepare_libero_config(output_dir / "libero_config")
        except Exception as error:
            payload[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=context.device)
    result = payload[0]
    if not isinstance(result, Mapping) or result.get("error"):
        raise ValueError(f"outcome LIBERO preparation failed: {result}")
    os.environ["LIBERO_CONFIG_PATH"] = str(
        (output_dir / "libero_config").resolve()
    )
    return {str(name): str(value) for name, value in result.items()}


def prepare_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> ECPStage1OutcomeRuntime:
    _configure_egl(context)
    config = load_outcome_config(args.config)
    base_config = load_stage1_config(
        outcome_repo_authority(config, "base_stage1_config")
    )
    start, stop, total, checkpoints = _runtime_limits(args, config, context)
    seed_everything(int(config["optimization"]["seed"]), context)
    authorities = load_stage1_authorities(args, base_config, context)
    initialization = _load_initialization(
        checkpoint=outcome_asset_authority(
            config, "initialization_checkpoint", args.asset_root
        ),
        model=authorities.model,
        device=context.device,
        initialization=config["initialization"],
    )
    tasks = load_stage1_tasks(
        target_manifest=stage1_repo_authority(base_config, "target_manifest"),
        selection_path=stage1_repo_authority(
            base_config, "successful_member_selection"
        ),
        data_root=args.data_root,
    )
    fit = tuple(task for task in tasks if task.fold_role == "fit")
    expected_fit = tuple(int(value) for value in config["roles"]["fit_task_ordinals"])
    expected_held = tuple(int(value) for value in config["roles"]["held_task_ordinals"])
    if (
        tuple(task.ordinal for task in fit) != expected_fit
        or tuple(task.ordinal for task in tasks if task.fold_role != "fit")
        != expected_held
    ):
        raise ValueError("OCPB fit19/held5 roles changed")
    task_count = int(
        config["optimization" if args.mode == "formal" else "profile_defaults"][
            "task_count"
        ]
    )
    selected = fit[:task_count]
    assignments = _task_assignments(
        selected,
        context.world_size,
        int(base_config["data"]["frame_stride"]),
    )
    local_ordinals = set(assignments[context.rank])
    local_tasks = tuple(task for task in selected if task.ordinal in local_ordinals)
    evidence = load_stage1_evidence_bank(
        selection_path=stage1_repo_authority(
            base_config, "successful_member_selection"
        ),
        phase_analysis_path=stage1_asset_authority(
            base_config, "phase_analysis", args.asset_root
        ),
        phase_code_root=stage1_asset_authority(
            base_config, "phase_code_root", args.asset_root
        ),
        asset_root=args.asset_root,
        contract=authorities.contract,
        device=context.device,
    )
    support = load_policy_support_bank(
        manifest_path=outcome_asset_authority(
            config, "policy_support_bank", args.asset_root
        ),
        evidence_bank=evidence,
        task_ordinals=local_ordinals,
        device=context.device,
        require_owner_responses=True,
    )
    support_root = int(config["outcome_calibration"]["support_visit_root"])
    credit_offset = int(
        config["outcome_calibration"]["credit_macro_offset"]
    )
    requests = {
        (
            task.ordinal,
            support.task(task.ordinal)
            .panel_for_visit(support_root + macro + credit_offset)
            .panel_id,
        )
        for task in local_tasks
        for macro in range(start, stop)
    }
    panels = cache_policy_support_panels(
        bank=support, requests=requests, device=context.device
    )
    languages = tokenize_stage1_languages(
        local_tasks,
        tokenizer_path=args.tokenizer_path,
        max_length=int(authorities.source_config["features"]["tokenizer_max_length"]),
        device=context.device,
    )
    video_store = build_stage1_video_store(
        local_tasks, frame_stride=int(base_config["data"]["frame_stride"])
    )
    stats = load_stats(
        authorities.source_config,
        authorities.source_config["data"]["active_task_ids"],
    )
    processor = Pi05LiberoProcessor(
        stats,
        args.tokenizer_path,
        int(authorities.source_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    optimizer, scheduler = _optimizer(authorities.model, config)
    initialize_deferred_process_group(
        context, rendezvous_root=args.output_dir.parent
    )
    expected_metrics = 0
    if (
        args.resume is None
        and args.mode == "formal"
        and initialization["restore_optimizer_and_rank_rng_in_formal"]
    ):
        loaded, _ = load_ecp_checkpoint(
            checkpoint=outcome_asset_authority(
                config, "initialization_checkpoint", args.asset_root
            ),
            stage=str(initialization["stage"]),
            context=context,
            model=authorities.model,
            optimizer=optimizer,
            scheduler=scheduler,
            run_contract_schema=str(initialization["run_contract_schema"]),
        )
        if loaded != int(initialization["checkpoint_macro"]):
            raise ValueError("OCPB fork initialization cursor changed")
    elif args.resume is not None:
        loaded, expected_metrics = load_ecp_checkpoint(
            checkpoint=args.resume,
            stage=STAGE,
            context=context,
            model=authorities.model,
            optimizer=optimizer,
            scheduler=scheduler,
            run_contract_schema=RUN_SCHEMA,
        )
        if loaded != start:
            raise ValueError("OCPB resume cursor changed")
    metrics_rows = (
        reconcile_metrics(
            args.output_dir / "metrics.jsonl",
            start,
            expected_metrics,
            cursor_key="macro",
        )
        if context.is_main and args.resume is not None
        else 0
    )
    runtime = ECPStage1OutcomeRuntime(
        args=args,
        config=config,
        base_config=base_config,
        context=context,
        tasks=selected,
        local_tasks=local_tasks,
        task_by_ordinal={task.ordinal: task for task in selected},
        video_store=video_store,
        language_tokens=languages,
        evidence_bank=evidence,
        support_bank=support,
        support_panels=panels,
        reward_tasks=_reward_tasks(base_config, selected),
        policy=authorities.policy.requires_grad_(False).eval(),
        processor=processor,
        observer=authorities.observer,
        contract=authorities.contract,
        identity_state=authorities.identity_state,
        prior_state=authorities.prior_state,
        model=authorities.model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainable_parameters=tuple(authorities.model.parameters()),
        env_pool=None,
        assignments=assignments,
        start_macro=start,
        stop_macro=stop,
        total_macros=total,
        checkpoint_macros=checkpoints,
        metrics_rows=metrics_rows,
    )
    publish_outcome_run_contract(
        runtime,
        source=authorities.source,
        initialization=initialization,
    )
    paths = _prepare_libero_paths(args.output_dir, context)
    runtime.env_pool = RandomResetEnvironmentPool(
        bddl_root=Path(paths["bddl_files"]),
        assets_root=Path(paths["assets"]),
        render_resolution=int(config["environment"]["render_resolution"]),
    )
    runtime.model.train()
    torch.cuda.reset_peak_memory_stats(context.device)
    return runtime


def train(args: argparse.Namespace) -> None:
    from ember.ecp.stage1_outcome_train_step import run_outcome_macro

    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    runtime: ECPStage1OutcomeRuntime | None = None
    try:
        runtime = prepare_runtime(args, context)
        started = time.monotonic()
        for macro in range(runtime.start_macro, runtime.stop_macro):
            row = run_outcome_macro(runtime, macro=macro, run_started=started)
            if context.is_main:
                append_jsonl(args.output_dir / "metrics.jsonl", row)
                runtime.metrics_rows += 1
                summary = {name: row[name] for name in (
                    "macro",
                    "credit_macro",
                    "coordinate",
                    "plus_successes",
                    "minus_successes",
                    "nonzero_advantage_tasks",
                    "mean_functional_total",
                    "mean_owner_response",
                    "mean_outcome_surrogate",
                    "gradient_norm_before_clip",
                    "elapsed_seconds",
                )}
                print(json.dumps(summary, sort_keys=True), flush=True)
            if context.world_size > 1:
                rows = torch.tensor(
                    runtime.metrics_rows,
                    device=context.device,
                    dtype=torch.long,
                )
                dist.broadcast(rows, src=0)
                runtime.metrics_rows = int(rows.item())
            completed = macro + 1
            if completed in runtime.checkpoint_macros:
                save_ecp_checkpoint(
                    output_dir=args.output_dir,
                    macro=completed,
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
                "completed_macro": runtime.stop_macro,
                "total_macros": runtime.total_macros,
            }
            write_json_atomic(args.output_dir / "segment_completion.json", result)
            if runtime.stop_macro == runtime.total_macros:
                write_json_atomic(args.output_dir / "completion.json", result)
    finally:
        if runtime is not None:
            if runtime.env_pool is not None:
                runtime.env_pool.close()
            runtime.video_store.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_stage1_outcome_binding_v14.json",
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-macro", type=int)
    parser.add_argument("--max-frames-per-call", type=int)
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
    return args
