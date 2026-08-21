"""Fit19 functional warm start followed by train-task closed-loop outer credit."""

from __future__ import annotations

import argparse
import math
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.expert_manifold.contract import (
    ExpertTask,
    load_task_expert_config,
    load_train_tasks,
)
from ember.functional_adaptation.action_alignment import PrivilegedMetaActionStore
from ember.functional_adaptation.code_checkpoint import (
    OUTER_RUN_SCHEMA,
    load_code_writer_checkpoint,
)
from ember.functional_adaptation.code_schedule import MetaCodeTrainingSchedule
from ember.functional_adaptation.code_writer import (
    FunctionalCodeWarmStart,
    FunctionalCodeWriter,
    load_inference_warm_start_,
)
from ember.functional_adaptation.decoder_training import (
    authority_path,
    load_functional_adapter_config,
)
from ember.functional_adaptation.phase_decoder_codes import (
    load_phase_decoder_code_authority,
)
from ember.functional_adaptation.phase_decoder_training import (
    load_config as load_phase_decoder_config,
)
from ember.lora import LoRAContract
from ember.pi05_assets import prepare_libero_config
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
    inspect_source_checkpoint,
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
    initialize_distributed,
    load_policy,
    load_stats,
    seed_everything,
)
from ember.reward.protocol import (
    RewardTask,
    SUITE_HORIZONS,
)
from ember.reward.rollout import (
    RandomResetEnvironmentPool,
)
from ember.writer.data import RawTeacherVideoStore
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.topology import visible_physical_cuda_index


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class OuterCreditRuntime:
    args: argparse.Namespace
    context: DistributedContext
    config: dict[str, Any]
    settings: Mapping[str, Any]
    mode: Mapping[str, Any]
    source: Mapping[str, Any]
    tasks: Mapping[int, ExpertTask]
    reward_tasks: Mapping[int, RewardTask]
    target_codes: Mapping[int, torch.Tensor]
    schedule: MetaCodeTrainingSchedule
    video_store: RawTeacherVideoStore
    action_store: PrivilegedMetaActionStore | None
    language: Mapping[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    policy: torch.nn.Module
    processor: Pi05LiberoProcessor
    writer: FunctionalCodeWriter
    lora_contract: LoRAContract
    identity_state: Mapping[str, torch.Tensor]
    env_pool: RandomResetEnvironmentPool | None
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    trainable: tuple[torch.nn.Parameter, ...]
    warm_start: FunctionalCodeWarmStart
    start_macro: int
    stop_macro: int
    checkpoint_macros: tuple[int, ...]
    metrics_rows: int
    metrics_path: Path


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
            raise ValueError(f"outer-credit {name} mapping changed")
        os.environ[name] = value


def _resume_macro(path: Path | None) -> int:
    if path is None:
        return 0
    try:
        macro = int(path.name.removeprefix("macro_"))
    except ValueError as error:
        raise ValueError("outer-credit resume checkpoint changed name") from error
    if macro <= 0 or path.parent.name != "checkpoints":
        raise ValueError("outer-credit resume checkpoint changed location")
    return macro


def _resolve_segment(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> tuple[Mapping[str, Any], Mapping[str, Any], int, int, tuple[int, ...]]:
    settings = config["outer_credit"]
    mode = settings["modes"][args.mode]
    expected_world = int(mode["world_size"])
    if context.world_size != expected_world:
        raise ValueError("outer-credit world size changed")
    repository = git_state(REPO_ROOT)
    if args.mode == "formal" and not git_state_is_clean_pushed_or_frozen_authority(
        repository
    ):
        raise ValueError("formal outer credit requires a clean pushed commit")
    start = _resume_macro(args.resume)
    stop = int(args.stop_after_macro or mode["default_stop_macro"])
    checkpoints = tuple(int(value) for value in mode["checkpoint_macros"])
    total = int(mode["total_macros"])
    boundaries = (0, *checkpoints)
    if (
        checkpoints[-1] != total
        or stop not in checkpoints
        or start >= stop
        or args.mode == "formal"
        and (start, stop) not in set(zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("outer-credit segment is not a declared checkpoint interval")
    return settings, mode, start, stop, checkpoints


def _optimizer(
    writer: FunctionalCodeWriter,
    settings: Mapping[str, Any],
    total_macros: int,
    warmstart_macros: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    values = settings["optimizer"]
    optimizer = torch.optim.AdamW(
        (parameter for parameter in writer.parameters() if parameter.requires_grad),
        lr=float(values["learning_rate"]),
        betas=tuple(float(value) for value in values["betas"]),
        eps=float(values["eps"]),
        weight_decay=float(values["weight_decay"]),
    )

    def factor(step: int) -> float:
        warmup = int(values.get("warmup_macros", warmstart_macros))
        if not 1 <= warmup <= warmstart_macros:
            raise ValueError("outer-credit optimizer warmup changed")
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_macros - warmup))
        return 0.05 + 0.475 * (1.0 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _sampled_frame_count(raw_count: int, stride: int) -> int:
    count = (raw_count - 1) // stride + 1
    return count + int((raw_count - 1) % stride != 0)


def _frame_costs(
    config: Mapping[str, Any],
    tasks: Sequence[ExpertTask],
    demos: Sequence[int],
    stride: int,
) -> dict[int, dict[int, int]]:
    manifest = read_json(authority_path(config, "target_data_manifest", REPO_ROOT))
    rows = {int(row["global_task_id"]): row for row in manifest["tasks"]}
    return {
        task.global_task_id: {
            int(demo): _sampled_frame_count(
                int(rows[task.global_task_id]["demonstrations"]["episode_lengths"][demo]),
                stride,
            )
            for demo in demos
        }
        for task in tasks
    }


def _reward_tasks(
    config: Mapping[str, Any], tasks: Sequence[ExpertTask]
) -> dict[int, RewardTask]:
    manifest = read_json(authority_path(config, "target_data_manifest", REPO_ROOT))
    rows = {int(row["global_task_id"]): row for row in manifest["tasks"]}
    result = {}
    for task in tasks:
        row = rows[task.global_task_id]
        bddl = row["bddl"]
        result[task.global_task_id] = RewardTask(
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


def _phase_targets(
    args: argparse.Namespace,
    tasks: Sequence[ExpertTask],
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], tuple[int, ...]]:
    phase_config = load_phase_decoder_config(args.phase_decoder_config)
    codes = load_phase_decoder_code_authority(
        args.code_artifact,
        config=phase_config,
        config_path=Path(str(phase_config["_base_config_path"])),
        device=device,
    )
    by_ordinal = {task.ordinal: task for task in tasks}
    targets = {
        by_ordinal[ordinal].global_task_id: codes.fit_task_codes[index]
        for index, ordinal in enumerate(codes.fit_ordinals)
    }
    return targets, codes.held_ordinals


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
        raise ValueError(f"outer-credit LIBERO preparation failed: {result}")
    os.environ["LIBERO_CONFIG_PATH"] = str((output_dir / "libero_config").resolve())
    return {str(name): str(value) for name, value in result.items()}


def _run_contract(
    runtime: OuterCreditRuntime,
    *,
    held_ordinals: Sequence[int],
) -> dict[str, Any]:
    repository = git_state(REPO_ROOT)
    tasks = tuple(runtime.tasks.values())
    weights = runtime.settings["warmstart_loss_weights"]
    action_alignment = float(weights["action_alignment"]) > 0
    process_controls = any(
        float(weights[name]) > 0
        for name in ("control_confidence", "control_update")
    )
    return {
        "schema_version": OUTER_RUN_SCHEMA,
        "mode": runtime.args.mode,
        "method": "fixed_decoder_fit19_train_task_antithetic_outer_credit",
        "candidate_method": str(runtime.settings["method"]),
        "repository": {key: repository[key] for key in ("branch", "commit")},
        "source": {
            "run": str(runtime.args.source_run),
            "checkpoint": str(runtime.args.checkpoint),
            "model_path": str(runtime.source["model_path"]),
        },
        "decoder_profile": str(runtime.args.decoder_profile_root),
        "phase_code_artifact": str(runtime.args.code_artifact),
        "config": str(runtime.args.config),
        "data_root": str(runtime.args.data_root),
        "tokenizer": str(runtime.args.tokenizer_path),
        "initialization": {
            **asdict(runtime.warm_start),
            "checkpoint": str(runtime.warm_start.checkpoint),
            "skipped_shape_tensors": list(
                runtime.warm_start.skipped_shape_tensors
            ),
            "decoder_state_loaded_from_historical_writer": False,
        },
        "tasks": {
            "role": (
                "train24_fit19_outer_credit"
                if len(tasks) == 19
                else "train24_fit_subset_outer_credit"
            ),
            "count": len(tasks),
            "global_task_ids": [task.global_task_id for task in tasks],
            "ordinals": [task.ordinal for task in tasks],
            "held_zero_reward_ordinals": [int(value) for value in held_ordinals],
            "equal_weight_per_macro": True,
        },
        "runtime": {
            "host": socket.gethostname(),
            "world_size": runtime.context.world_size,
            "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
            "total_macros": int(runtime.mode["total_macros"]),
            "warmstart_macros": int(runtime.mode["warmstart_macros"]),
        },
        "trainable": {
            "writer_parameter_count": sum(
                int(parameter.numel()) for parameter in runtime.trainable
            ),
            "fixed_decoder_trainable_parameters": 0,
            "source_policy_trainable_parameters": 0,
        },
        "outer_objective": dict(runtime.settings["objective"]),
        "warmstart_objective": {
            "loss_weights": dict(weights),
            "temporal_controls": list(
                runtime.settings.get("warmstart_temporal_controls", ("reversed",))
            ),
            "dynamic_k_max": int(
                runtime.config["code_inference"]["training"]["dynamic_k_max"]
            ),
            "cross_episode_action_phase_alignment": action_alignment,
            "process_controls_train_writer": process_controls,
        },
        "deployment": {
            "inputs": ["exact language", "action-hidden ordered teacher videos"],
            "writer_runs_once_before_rollout": True,
            "output": "one complete 38-target rank16 LoRA",
            "task_id_route": False,
            "decoder_frozen": True,
        },
        "training_privileged": {
            "fit19_train_task_reward_and_BDDL_progress": True,
            "fit19_cross_episode_teacher_action_phase_alignment": action_alignment,
            "held5_reward_reads": 0,
            "held5_teacher_action_reads": 0,
            "validation_reward_reads": 0,
            "test_reward_reads": 0,
            "deployment_reward_reads": 0,
            "teacher_actions_read_during_process_warmstart": action_alignment,
            "teacher_actions_read_during_outer_credit": 0,
        },
        "content_hash_policy": "disabled_by_owner",
    }


def _publish_contract(runtime: OuterCreditRuntime, contract: Mapping[str, Any]) -> None:
    path = runtime.args.output_dir / "run_contract.json"
    if runtime.context.is_main:
        if runtime.args.resume is None:
            if runtime.args.output_dir.exists() and any(runtime.args.output_dir.iterdir()):
                raise ValueError("fresh outer-credit output directory is not empty")
            runtime.args.output_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(path, dict(contract))
        elif not path.is_file() or read_json(path) != dict(contract):
            raise ValueError("outer-credit resume contract changed")
        append_jsonl(
            runtime.args.output_dir / "invocations.jsonl",
            {
                "argv": sys.argv,
                "started_unix": time.time(),
                "resume": str(runtime.args.resume) if runtime.args.resume else None,
            },
        )
    if runtime.context.world_size > 1:
        dist.barrier(device_ids=[runtime.context.local_rank])


def _synchronize_or_resume(runtime: OuterCreditRuntime) -> None:
    if runtime.args.resume is not None:
        macro, rows = load_code_writer_checkpoint(
            checkpoint=runtime.args.resume,
            rank=runtime.context.rank,
            world_size=runtime.context.world_size,
            writer=runtime.writer,
            optimizer=runtime.optimizer,
            scheduler=runtime.scheduler,
        )
        if macro != runtime.start_macro or rows != macro:
            raise ValueError("outer-credit resume cursor changed")
        runtime.metrics_rows = rows
        return
    if runtime.context.world_size > 1:
        for value in runtime.writer.state_dict().values():
            dist.broadcast(value, src=0)


def prepare_runtime(args: argparse.Namespace) -> OuterCreditRuntime:
    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    _configure_egl(context)
    config = load_functional_adapter_config(args.config, REPO_ROOT)
    settings, mode, start, stop, checkpoints = _resolve_segment(
        args, config, context
    )
    seed_everything(int(settings["seed"]), context)
    expert_config = load_task_expert_config(
        authority_path(config, "train24_experts", REPO_ROOT)
    )
    all_tasks = load_train_tasks(expert_config, args.data_root)
    target_codes, held_ordinals = _phase_targets(args, all_tasks, context.device)
    all_fit_tasks = tuple(
        task for task in all_tasks if task.global_task_id in target_codes
    )
    if (
        len(all_fit_tasks) != 19
        or tuple(task.ordinal for task in all_fit_tasks)
        != tuple(int(value) for value in settings["fit_task_ordinals"])
        or tuple(int(value) for value in settings["held_task_ordinals"])
        != tuple(held_ordinals)
    ):
        raise ValueError("outer-credit fit19/held5 split changed")
    fit_tasks = all_fit_tasks[: int(mode["task_count"])]
    if not fit_tasks:
        raise ValueError("outer-credit mode selected no fit tasks")
    authorities = load_evaluation_authorities(
        authority_path(config, "evaluation_config", REPO_ROOT), REPO_ROOT
    )
    source = inspect_source_checkpoint(
        authorities,
        args.source_run,
        args.checkpoint,
        evaluation_mode="formal",
    )
    policy = load_policy(
        Path(str(source["model_path"])), authorities.source_base_config, context.device
    )
    lora = load_pi05_lora_contract(
        authority_path(config, "lora_contract", REPO_ROOT)
    )
    identity = prepare_frozen_writer_policy(policy, lora)
    policy.requires_grad_(False).eval()
    writer = FunctionalCodeWriter.from_policy(
        policy=policy,
        config=config,
        contract=lora,
        decoder_checkpoint=args.decoder_profile_root / "decoder.safetensors",
        device=context.device,
    )
    warm_start = load_inference_warm_start_(writer, args.warm_start)
    optimizer, scheduler = _optimizer(
        writer,
        settings,
        int(mode["total_macros"]),
        int(mode["warmstart_macros"]),
    )
    trainable = tuple(
        parameter for parameter in writer.parameters() if parameter.requires_grad
    )
    first, last = (int(value) for value in settings["train_demo_indices"])
    demos = tuple(range(first, last + 1))
    stride = int(config["code_inference"]["training"]["frame_stride"])
    schedule = MetaCodeTrainingSchedule(
        task_ids=[task.global_task_id for task in fit_tasks],
        demo_indices=demos,
        sampled_frame_counts=_frame_costs(config, fit_tasks, demos, stride),
        world_size=context.world_size,
        seed=int(settings["video_schedule_seed"]),
        dynamic_k_max=None,
        temporal_controls=tuple(
            str(value)
            for value in settings.get(
                "warmstart_temporal_controls", ("reversed",)
            )
        ),
    )
    task_authorities = tuple(task.authority for task in fit_tasks)
    video_store = RawTeacherVideoStore(
        task_authorities,
        frame_stride=stride,
        max_open_files=int(settings["video_open_files_per_rank"]),
    )
    tokenizer = Pi05TeacherPrefixTokenizer(
        args.tokenizer_path,
        int(authorities.source_base_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    language = {task.global_task_id: tokenizer([task.language]) for task in fit_tasks}
    stats = load_stats(
        authorities.source_base_config,
        authorities.source_base_config["data"]["active_task_ids"],
    )
    action_store = None
    if float(settings["warmstart_loss_weights"]["action_alignment"]) > 0:
        action_store = PrivilegedMetaActionStore(
            task_authorities,
            action_q01=stats["action"]["q01"],
            action_q99=stats["action"]["q99"],
            phase_count=int(config["code_inference"]["phase_queries"]),
            max_open_files=int(settings["video_open_files_per_rank"]),
        )
    processor = Pi05LiberoProcessor(
        stats,
        args.tokenizer_path,
        int(authorities.source_base_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    initialize_deferred_process_group(context, rendezvous_root=args.output_dir.parent)
    runtime = OuterCreditRuntime(
        args=args,
        context=context,
        config=config,
        settings=settings,
        mode=mode,
        source=source,
        tasks={task.global_task_id: task for task in fit_tasks},
        reward_tasks=_reward_tasks(config, fit_tasks),
        target_codes={
            task.global_task_id: target_codes[task.global_task_id]
            for task in fit_tasks
        },
        schedule=schedule,
        video_store=video_store,
        action_store=action_store,
        language=language,
        policy=policy,
        processor=processor,
        writer=writer,
        lora_contract=lora,
        identity_state=identity,
        env_pool=None,
        optimizer=optimizer,
        scheduler=scheduler,
        trainable=trainable,
        warm_start=warm_start,
        start_macro=start,
        stop_macro=stop,
        checkpoint_macros=checkpoints,
        metrics_rows=start,
        metrics_path=args.output_dir / "metrics.jsonl",
    )
    contract = _run_contract(runtime, held_ordinals=held_ordinals)
    _publish_contract(runtime, contract)
    paths = _prepare_libero_paths(args.output_dir, context)
    runtime.env_pool = RandomResetEnvironmentPool(
        bddl_root=Path(paths["bddl_files"]),
        assets_root=Path(paths["assets"]),
        render_resolution=int(settings["environment"]["render_resolution"]),
    )
    _synchronize_or_resume(runtime)
    writer.train()
    writer.fixed_decoder.eval()
    torch.cuda.reset_peak_memory_stats(context.device)
    return runtime


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--mode", choices=("smoke", "profile", "formal"), required=True)
    result.add_argument("--source-run", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--decoder-profile-root", type=Path, required=True)
    result.add_argument("--phase-decoder-config", type=Path, required=True)
    result.add_argument("--code-artifact", type=Path, required=True)
    result.add_argument("--warm-start", type=Path, required=True)
    result.add_argument("--tokenizer-path", type=Path, required=True)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--stop-after-macro", type=int)
    result.add_argument("--resume", type=Path)
    return result


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
        "source_run",
        "checkpoint",
        "decoder_profile_root",
        "phase_decoder_config",
        "code_artifact",
        "warm_start",
        "tokenizer_path",
        "data_root",
    ):
        value = getattr(args, name).resolve()
        if not value.exists():
            raise ValueError(f"missing outer-credit path: {value}")
        setattr(args, name, value)
    args.output_dir = args.output_dir.resolve()
    args.resume = args.resume.resolve() if args.resume else None
    return args
