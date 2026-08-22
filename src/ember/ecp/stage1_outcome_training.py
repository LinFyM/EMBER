"""Runtime for OCPB v18 task-equal action-guided closed-loop binding."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.ecp.checkpoint import (
    checkpoint_macro,
    load_ecp_checkpoint,
    save_ecp_checkpoint,
)
from ember.ecp.contracts import TargetOwner, build_target_owners
from ember.ecp.stage1 import ECPStage1Model
from ember.ecp.stage1_config import (
    REPO_ROOT,
    RUN_SCHEMA,
    STAGE,
    load_stage1_config,
    load_stage1_initialization,
    stage1_asset_authority,
    stage1_repo_authority,
)
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
    PolicySupportPanel,
    cache_policy_support_panels,
    load_policy_support_bank,
)
from ember.ecp.stage1_training import load_stage1_authorities
from ember.lora import LoRAContract
from ember.pi05_assets import prepare_libero_config
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import (
    DistributedContext,
    barrier,
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
    context: DistributedContext
    tasks: tuple[ECPStage1Task, ...]
    local_tasks: tuple[ECPStage1Task, ...]
    task_by_ordinal: dict[int, ECPStage1Task]
    assignments: tuple[tuple[int, ...], ...]
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
    owners: tuple[TargetOwner, ...]
    identity_state: Mapping[str, torch.Tensor]
    prior_state: Mapping[str, torch.Tensor]
    model: ECPStage1Model
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    trainable_parameters: tuple[torch.nn.Parameter, ...]
    env_pool: RandomResetEnvironmentPool | None
    initialization: Mapping[str, Any]
    start_macro: int
    stop_macro: int
    total_macros: int
    checkpoint_macros: tuple[int, ...]
    metrics_rows: int


def successful_panel_for_visit(
    task: Any, visit: int
) -> PolicySupportPanel:
    successful = tuple(panel for panel in task.panels if panel.kind == "successful")
    if not successful:
        raise ValueError("outcome proposal task has no successful panel")
    return successful[visit % len(successful)]


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
            raise ValueError(f"Stage 1 outcome {name} mapping changed")
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
            stop
            not in set(int(value) for value in config["optimization"]["stage_stop_macros"])
            or os.environ.get("NCCL_P2P_DISABLE") != "1"
            or not git_state_is_clean_pushed_or_frozen_authority(
                git_state(REPO_ROOT)
            )
        )
    ):
        raise ValueError("ECP Stage 1 outcome launch authority changed")
    return start, stop, total, checkpoints


def _task_cost(task: ECPStage1Task, frame_stride: int) -> float:
    sampled_frames = sum(
        (int(length) - 1) // frame_stride + 1 for length in task.episode_lengths
    ) / len(task.episode_lengths)
    return float(SUITE_HORIZONS[task.suite]) + sampled_frames


def _task_assignments(
    tasks: Sequence[ECPStage1Task], world_size: int, frame_stride: int
) -> tuple[tuple[int, ...], ...]:
    bins: list[list[int]] = [[] for _ in range(world_size)]
    loads = [0.0] * world_size
    for task in sorted(
        tasks, key=lambda value: (-_task_cost(value, frame_stride), value.ordinal)
    ):
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        bins[rank].append(task.ordinal)
        loads[rank] += _task_cost(task, frame_stride)
    return tuple(tuple(sorted(values)) for values in bins)


def _reward_tasks(
    config: Mapping[str, Any], tasks: Sequence[ECPStage1Task]
) -> dict[int, RewardTask]:
    manifest = read_json(stage1_repo_authority(config, "target_manifest"))
    rows = {int(row["global_task_id"]): row for row in manifest["tasks"]}
    result = {}
    for task in tasks:
        row = rows[task.global_task_id]
        result[task.ordinal] = RewardTask(
            suite=task.suite,
            task_id=task.task_id,
            global_task_id=task.global_task_id,
            split_role="train",
            language=task.language,
            problem_folder=str(row["problem_folder"]),
            bddl_file=str(row["bddl"]["filename"]),
            bddl_bytes=int(row["bddl"]["bytes"]),
            bddl_sha256=None,
            horizon=SUITE_HORIZONS[task.suite],
        )
    return result


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
        raise ValueError(f"Stage 1 LIBERO preparation failed: {result}")
    os.environ["LIBERO_CONFIG_PATH"] = str(
        (output_dir / "libero_config").resolve()
    )
    return {str(name): str(value) for name, value in result.items()}


def _run_contract(
    runtime: ECPStage1OutcomeRuntime, source: Mapping[str, Any]
) -> dict[str, Any]:
    repository = git_state(REPO_ROOT)
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
    return {
        "schema_version": RUN_SCHEMA,
        "stage": STAGE,
        "mode": runtime.args.mode,
        "command": list(sys.argv),
        "git": {"branch": repository["branch"], "commit": repository["commit"]},
        "host": socket.gethostname(),
        "config": {
            "path": str(runtime.args.config),
            "bytes": runtime.args.config.stat().st_size,
        },
        "source": dict(source),
        "initialization": dict(runtime.initialization),
        "asset_root": str(runtime.args.asset_root),
        "data_root": str(runtime.args.data_root),
        "tokenizer": {
            "path": str(runtime.args.tokenizer_path),
            "bytes": runtime.args.tokenizer_path.stat().st_size,
        },
        "tasks": {
            "role": "train24_fit19_action_guided_outcome_binding",
            "count": len(runtime.tasks),
            "ordinals": [task.ordinal for task in runtime.tasks],
            "global_task_ids": [task.global_task_id for task in runtime.tasks],
            "equal_weight_per_macro": True,
            "assignments": [list(values) for values in runtime.assignments],
            "held_zero_gradient_ordinals": list(
                runtime.config["roles"]["held_task_ordinals"]
            ),
        },
        "model": dict(runtime.config["model"]),
        "data": dict(runtime.config["data"]),
        "objective": dict(runtime.config["objective"]),
        "outcome_binding": dict(runtime.config["outcome_binding"]),
        "environment": dict(runtime.config["environment"]),
        "optimization": dict(runtime.config["optimization"]),
        "information_wall": dict(runtime.config["information_wall"]),
        "runtime": {
            "world_size": runtime.context.world_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
            "topology": topology,
        },
        "trainable_parameters": sum(
            parameter.numel() for parameter in runtime.trainable_parameters
        ),
        "source_policy_trainable_parameters": 0,
        "observer_trainable_parameters": 0,
        "content_hash_policy": "disabled_by_owner",
    }


def _publish_run_contract(
    runtime: ECPStage1OutcomeRuntime, source: Mapping[str, Any]
) -> None:
    contract = _run_contract(runtime, source)
    path = runtime.args.output_dir / "run_contract.json"
    if runtime.context.is_main:
        if runtime.args.resume is None:
            if runtime.args.output_dir.exists():
                raise ValueError("fresh Stage 1 outcome output already exists")
            runtime.args.output_dir.mkdir(parents=True)
            write_json_atomic(path, contract)
        else:
            existing = read_json(path)
            if (
                existing.get("schema_version") != RUN_SCHEMA
                or existing.get("stage") != STAGE
                or existing.get("git", {}).get("commit")
                != contract["git"]["commit"]
                or existing.get("config", {}).get("bytes")
                != contract["config"]["bytes"]
                or existing.get("source", {}).get("checkpoint")
                != contract["source"].get("checkpoint")
                or existing.get("runtime", {}).get("world_size")
                != contract["runtime"]["world_size"]
                or existing.get("outcome_binding")
                != contract["outcome_binding"]
            ):
                raise ValueError("Stage 1 outcome resume run contract changed")
    barrier(runtime.context)


def prepare_outcome_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> ECPStage1OutcomeRuntime:
    _configure_egl(context)
    config = load_stage1_config(args.config)
    start, stop, total, checkpoints = _runtime_limits(args, config, context)
    seed_everything(int(config["optimization"]["seed"]), context)
    authorities = load_stage1_authorities(args, config, context)
    initialization = load_stage1_initialization(
        checkpoint=stage1_asset_authority(
            config, "initialization_checkpoint", args.asset_root
        ),
        model=authorities.model,
        device=context.device,
        initialization=config["initialization"],
    )
    tasks = load_stage1_tasks(
        target_manifest=stage1_repo_authority(config, "target_manifest"),
        selection_path=stage1_repo_authority(
            config, "successful_member_selection"
        ),
        data_root=args.data_root,
    )
    fit = tuple(task for task in tasks if task.fold_role == "fit")
    expected_fit = tuple(int(value) for value in config["roles"]["fit_task_ordinals"])
    expected_held = tuple(
        int(value) for value in config["roles"]["held_task_ordinals"]
    )
    if (
        tuple(task.ordinal for task in fit) != expected_fit
        or tuple(task.ordinal for task in tasks if task.fold_role != "fit")
        != expected_held
    ):
        raise ValueError("ECP fit19/held5 roles changed")
    cell = (
        config["optimization"]
        if args.mode == "formal"
        else config["profile_defaults"]
    )
    selected = fit[: int(cell["task_count"])]
    assignments = _task_assignments(
        selected, context.world_size, int(config["data"]["frame_stride"])
    )
    local_ordinals = set(assignments[context.rank])
    local_tasks = tuple(
        task for task in selected if task.ordinal in local_ordinals
    )
    evidence = load_stage1_evidence_bank(
        selection_path=stage1_repo_authority(
            config, "successful_member_selection"
        ),
        phase_analysis_path=stage1_asset_authority(
            config, "phase_analysis", args.asset_root
        ),
        phase_code_root=stage1_asset_authority(
            config, "phase_code_root", args.asset_root
        ),
        asset_root=args.asset_root,
        contract=authorities.contract,
        device=context.device,
    )
    support = load_policy_support_bank(
        manifest_path=stage1_asset_authority(
            config, "policy_support_bank", args.asset_root
        ),
        evidence_bank=evidence,
        task_ordinals=local_ordinals,
        device=context.device,
        contract=authorities.contract,
    )
    outcome = config["outcome_binding"]
    proposal_root = int(outcome["proposal_visit_root"])
    anchor_root = int(outcome["support_visit_root"])
    requests = set()
    for task in local_tasks:
        support_task = support.task(task.ordinal)
        for macro in range(start, stop):
            requests.add(
                (
                    task.ordinal,
                    successful_panel_for_visit(
                        support_task, proposal_root + macro
                    ).panel_id,
                )
            )
            requests.add(
                (
                    task.ordinal,
                    support_task.panel_for_visit(anchor_root + macro).panel_id,
                )
            )
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
        local_tasks, frame_stride=int(config["data"]["frame_stride"])
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
    if context.world_size > 1:
        for value in authorities.model.state_dict().values():
            dist.broadcast(value, src=0)
    expected_metrics = 0
    if args.resume is not None:
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
            raise ValueError("ECP Stage 1 outcome resume cursor changed")
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
        context=context,
        tasks=selected,
        local_tasks=local_tasks,
        task_by_ordinal={task.ordinal: task for task in selected},
        assignments=assignments,
        video_store=video_store,
        language_tokens=languages,
        evidence_bank=evidence,
        support_bank=support,
        support_panels=panels,
        reward_tasks=_reward_tasks(config, selected),
        policy=authorities.policy.requires_grad_(False).eval(),
        processor=processor,
        observer=authorities.observer,
        contract=authorities.contract,
        owners=build_target_owners(authorities.contract),
        identity_state=authorities.identity_state,
        prior_state=authorities.prior_state,
        model=authorities.model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainable_parameters=tuple(authorities.model.parameters()),
        env_pool=None,
        initialization=initialization,
        start_macro=start,
        stop_macro=stop,
        total_macros=total,
        checkpoint_macros=checkpoints,
        metrics_rows=metrics_rows,
    )
    _publish_run_contract(runtime, authorities.source)
    paths = _prepare_libero_paths(args.output_dir, context)
    runtime.env_pool = RandomResetEnvironmentPool(
        bddl_root=Path(paths["bddl_files"]),
        assets_root=Path(paths["assets"]),
        render_resolution=int(config["environment"]["render_resolution"]),
    )
    runtime.model.train()
    torch.cuda.reset_peak_memory_stats(context.device)
    return runtime


def train_action_guided_outcome(args: argparse.Namespace) -> None:
    from ember.ecp.stage1_outcome_train_step import run_outcome_macro

    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    runtime: ECPStage1OutcomeRuntime | None = None
    try:
        runtime = prepare_outcome_runtime(args, context)
        started = time.monotonic()
        for macro in range(runtime.start_macro, runtime.stop_macro):
            row = run_outcome_macro(runtime, macro=macro, run_started=started)
            if context.is_main:
                append_jsonl(args.output_dir / "metrics.jsonl", row)
                runtime.metrics_rows += 1
                if row["macro"] % args.log_every == 0:
                    summary = {
                        name: row[name]
                        for name in (
                            "macro",
                            "plus_successes",
                            "minus_successes",
                            "nonzero_advantage_tasks",
                            "mean_action_policy_loss",
                            "mean_structural_total",
                            "mean_outcome_leaf_gradient_norm",
                            "gradient_norm_before_clip",
                            "factor_head_gradient_norm_before_clip",
                            "elapsed_seconds",
                        )
                    }
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
