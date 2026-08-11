"""Assets, paired K2 environments, and exact-resume Work-Queue runtime."""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import torch
import torch.distributed as dist

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
from ember.expert_manifold.v6_success_key import SuccessKeyAnchorBank
from ember.lora import LoRAContract
from ember.pi05_assets import prepare_libero_config
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
from ember.reward.protocol import RewardTask, SUITE_HORIZONS
from ember.reward.rollout import RandomResetEnvironmentPool
from ember.writer.architecture import LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS
from ember.writer.as_sampling import MixedTaskBatchSampler, TeacherVideoSchedule
from ember.writer.condition_update import (
    FrozenV6ConditionResidualWriter,
    validate_frozen_v6_residual_writer,
)
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
    reward_task_by_global_id: dict[int, RewardTask]
    dataset: FunctionalQueryDataset
    sampler: MixedTaskBatchSampler
    video_schedule: TeacherVideoSchedule
    video_store: RawTeacherVideoStore
    language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    processor: Pi05LiberoProcessor
    policy: torch.nn.Module
    writer: FrozenV6ConditionResidualWriter
    success_key_bank: SuccessKeyAnchorBank
    identity_state: dict[str, torch.Tensor]
    env_pool: RandomResetEnvironmentPool
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
        raise ExpertManifoldError(
            "residual Writer resume path is not a macro checkpoint"
        )
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
        config["profile_run"]
        if args.mode == "mechanism-profile"
        else config["formal_run"]
    )
    profile_valid = args.mode != "mechanism-profile" or (
        args.resume is None and args.stop_after_macro in {None, 1} and stop == 1
    )
    formal_boundaries = (0, *checkpoints)
    formal_segments = set(zip(formal_boundaries, formal_boundaries[1:]))
    formal_valid = args.mode != "formal" or (
        args.stop_after_macro is not None
        and checkpoints
        and checkpoints[-1] == total
        and (start, stop) in formal_segments
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
        context.world_size
        in tuple(int(value) for value in selected["allowed_world_sizes"])
        and context.world_size <= int(selected["maximum_world_size"])
        and selected["task_assignment"]
        == "host_local_atomic_completion_driven_long_first_train24"
        and args.num_workers == int(selected["num_workers_per_rank"])
        and 0 <= start < stop <= total
        and profile_valid
        and formal_valid
        and not state["dirty_paths"]
        and git_valid
    )
    if not valid:
        raise ExpertManifoldError(
            "residual Writer runtime differs from its sealed segment"
        )
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
            raise ExpertManifoldError(f"PCUG {name} mapping changed")
        os.environ[name] = value


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
        tasks_per_rank_per_update=math.ceil(24 / context.world_size),
        video_schedule=schedule,
        task_video_costs=costs,
        assignment_strategy="cost_balanced_long_first_dynamic_uneven",
    )
    return tasks, dataset, sampler, schedule


def _build_reward_tasks(
    tasks: Sequence[ExpertTask],
    config: Mapping[str, Any],
) -> dict[int, RewardTask]:
    manifest = read_json(authority_path(config, "target_data_manifest"))
    rows = {
        int(row["global_task_id"]): row
        for row in manifest.get("tasks", [])
        if row.get("split_role") == "train"
    }
    if len(rows) != 24:
        raise ExpertManifoldError("PCUG target manifest lost train24")
    reward_tasks = {}
    for task in tasks:
        row = rows.get(task.global_task_id)
        if (
            not isinstance(row, Mapping)
            or row.get("suite") != task.suite
            or int(row.get("task_id", -1)) != task.task_id
            or row.get("language") != task.language
            or Path(str(row.get("hdf5", {}).get("relative_path", ""))).name
            != task.authority.path.name
        ):
            raise ExpertManifoldError("PCUG HDF5 and task manifest disagree")
        bddl = row.get("bddl")
        if not isinstance(bddl, Mapping):
            raise ExpertManifoldError("PCUG train task lost BDDL authority")
        reward_tasks[task.global_task_id] = RewardTask(
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
    return reward_tasks


def _build_policy_writer(
    *,
    config: Mapping[str, Any],
    context: DistributedContext,
    source: Mapping[str, Any],
    source_config: Mapping[str, Any],
) -> tuple[
    torch.nn.Module,
    FrozenV6ConditionResidualWriter,
    dict[str, torch.Tensor],
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
    identity = {name: value.detach().clone() for name, value in template.items()}
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
        innovation_width=int(feature["innovation_width"]),
        phase_slots=int(feature["phase_slots"]),
        max_frames_per_encoder_call=int(
            config["writer"]["max_frames_per_encoder_call"]
        ),
        image_width=int(config["writer"]["image_width"]),
        expert_width=int(config["writer"]["expert_width"]),
        action_horizon=int(config["writer"]["action_horizon"]),
        padded_action_dim=int(config["writer"]["padded_action_dim"]),
        innovation_seed=int(config["writer"]["initialization_seed"]),
    ).to(context.device)
    validate_frozen_v6_residual_writer(writer, require_zero_memory=True)
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise ExpertManifoldError("residual Writer source policy is not frozen")
    return policy, writer, identity, lora, warm_start, ownership


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
            "residual Writer metrics differ from resume cursor: " f"{result}"
        )
    return int(result["rows"])


def _prepare_libero_paths(
    args: argparse.Namespace,
    context: DistributedContext,
) -> dict[str, str]:
    payload: list[Any] = [None]
    if context.is_main:
        try:
            payload[0] = prepare_libero_config(args.output_dir / "libero_config")
        except Exception as error:
            payload[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=context.device)
    paths = payload[0]
    if not isinstance(paths, Mapping) or paths.get("error"):
        raise ExpertManifoldError(f"PCUG LIBERO path preparation failed: {paths}")
    os.environ["LIBERO_CONFIG_PATH"] = str(
        (args.output_dir / "libero_config").resolve()
    )
    return {str(name): str(value) for name, value in paths.items()}


def _restore_resume(
    runtime_args: argparse.Namespace,
    config: Mapping[str, Any],
    segment: RuntimeSegment,
    context: DistributedContext,
    writer: FrozenV6ConditionResidualWriter,
    success_key_bank: SuccessKeyAnchorBank,
    checkpoint_contract_value: Mapping[str, Any],
) -> None:
    if runtime_args.resume is None:
        return
    loaded, rows = load_v6_prior_checkpoint(
        checkpoint=runtime_args.resume,
        memory=writer.program_memory,
        success_key_bank=success_key_bank,
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
    _configure_egl(context)
    authorities, source, tokenizer = _load_source(args, config)
    tasks, dataset, sampler, schedule = _build_data(
        args=args,
        config=config,
        context=context,
        segment=segment,
    )
    reward_tasks = _build_reward_tasks(tasks, config)
    policy, writer, identity, lora, warm_start, ownership = _build_policy_writer(
        config=config,
        context=context,
        source=source,
        source_config=authorities.source_base_config,
    )
    success_key_bank = SuccessKeyAnchorBank(
        [task.global_task_id for task in tasks],
        feature_width=int(config["condition_feature"]["feature_width"]),
        device=context.device,
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
        success_key_bank=success_key_bank,
        repo_root=REPO_ROOT,
    )
    checkpoint_contract_value = checkpoint_contract(contract)
    publish_contract(args, contract, context)
    paths = _prepare_libero_paths(args, context)
    env_pool = RandomResetEnvironmentPool(
        bddl_root=Path(paths["bddl_files"]),
        assets_root=Path(paths["assets"]),
        render_resolution=int(config["environment"]["render_resolution"]),
    )
    try:
        _restore_resume(
            args,
            config,
            segment,
            context,
            writer,
            success_key_bank,
            checkpoint_contract_value,
        )
        metrics_path = args.output_dir / "metrics.jsonl"
        expected_rows = segment.start_macro if args.mode == "formal" else 0
        if (
            _reconcile_metrics_cursor(
                metrics_path,
                context=context,
                expected_rows=expected_rows,
            )
            != expected_rows
        ):
            raise ExpertManifoldError(
                "residual Writer metrics differ from resume cursor"
            )
    except Exception:
        env_pool.close()
        raise
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
        reward_task_by_global_id=reward_tasks,
        dataset=dataset,
        sampler=sampler,
        video_schedule=schedule,
        video_store=video_store,
        language_tokens=language,
        processor=processor,
        policy=policy,
        writer=writer,
        success_key_bank=success_key_bank,
        identity_state=identity,
        env_pool=env_pool,
        lora_contract=lora,
        warm_start=warm_start,
        ownership=ownership,
        run_contract=contract,
        checkpoint_contract=checkpoint_contract_value,
        metrics_path=metrics_path,
    )
