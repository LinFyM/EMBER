"""Train the shared ordered-Procedure Writer from train24 on-policy reward."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.lora import copy_task_lora_state_
from ember.pi05_assets import prepare_libero_config
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import (
    DistributedContext,
    barrier,
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
    reward_credit_environment_seed,
)
from ember.reward.rollout import (
    RandomResetEnvironmentPool,
    RewardTrajectory,
    collect_randomized_reward_trajectories,
    complete_trajectory_batch,
)
from ember.writer.as_config import authority_path
from ember.writer.as_contract import load_run_authorities, writer_trainable_contract
from ember.writer.as_step import (
    ParameterSlice,
    accumulate_flat_gradient,
    assign_flat_gradient,
    gather_full24_records,
    parameter_layout,
    reduce_full24_gradient,
)
from ember.writer.data import (
    RawTeacherVideoStore,
    WriterTaskAuthority,
    pack_teacher_condition,
)
from ember.writer.errors import WriterModelError
from ember.writer.reward_checkpoint import (
    checkpoint_cycle,
    load_reward_checkpoint,
    save_reward_checkpoint,
)
from ember.writer.reward_config import (
    REWARD_CONFIG,
    REWARD_LAUNCH_SCHEMA,
    load_reward_config,
    require_reward_mode,
)
from ember.writer.reward_preference import (
    RewardPreferenceSummary,
    backpropagate_reward_preference,
    functional_reward_lora_gradient,
)
from ember.writer.teacher_video_schedule import TeacherVideoSchedule
from ember.writer.training import build_writer


@dataclass(frozen=True)
class RewardProbe:
    global_task_id: int
    suite: str
    shared_core: torch.Tensor
    per_video_procedure: torch.Tensor
    condition_video_offsets: torch.Tensor
    before_lora: Mapping[str, torch.Tensor]
    query: Mapping[str, torch.Tensor]
    before_action: torch.Tensor
    policy_noise_seed: int


@dataclass
class RewardRuntime:
    args: argparse.Namespace
    context: DistributedContext
    config: dict[str, Any]
    base_config: dict[str, Any]
    source_config: dict[str, Any]
    tasks: tuple[RewardTask, ...]
    writer_tasks: tuple[WriterTaskAuthority, ...]
    video_store: RawTeacherVideoStore
    video_schedule: TeacherVideoSchedule
    language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    processor: Pi05LiberoProcessor
    env_pool: RandomResetEnvironmentPool
    policy: torch.nn.Module
    writer: torch.nn.Module
    identity_state: Mapping[str, torch.Tensor]
    lora_contract: Any
    optimizer: torch.optim.Optimizer
    trainable_parameters: tuple[torch.nn.Parameter, ...]
    gradient_layout: tuple[ParameterSlice, ...]
    contract: dict[str, Any]
    start_cycle: int
    stop_cycle: int
    metrics_path: Path


def _load_tasks(
    *, data_root: Path, base_config: Mapping[str, Any]
) -> tuple[tuple[RewardTask, ...], tuple[WriterTaskAuthority, ...]]:
    manifest = read_json(authority_path(base_config, "target_data_manifest"))
    reward_tasks, writer_tasks = [], []
    for row in manifest["tasks"]:
        if row["split_role"] != "train":
            continue
        hdf5 = row["hdf5"]
        path = (data_root / str(hdf5["relative_path"])).resolve()
        writer_tasks.append(
            WriterTaskAuthority(
                task_id=int(row["global_task_id"]),
                language=str(row["language"]),
                path=path,
                expected_bytes=int(hdf5["bytes"]),
            )
        )
        bddl = row["bddl"]
        reward_tasks.append(
            RewardTask(
                suite=str(row["suite"]),
                task_id=int(row["task_id"]),
                global_task_id=int(row["global_task_id"]),
                split_role="train",
                language=str(row["language"]),
                problem_folder=str(row["problem_folder"]),
                bddl_file=str(bddl["filename"]),
                bddl_bytes=int(bddl["bytes"]),
                bddl_sha256=None,
                horizon=SUITE_HORIZONS[str(row["suite"])],
            )
        )
    reward_tasks.sort(key=lambda task: task.global_task_id)
    writer_tasks.sort(key=lambda task: task.task_id)
    if len(reward_tasks) != 24 or [task.global_task_id for task in reward_tasks] != [
        task.task_id for task in writer_tasks
    ]:
        raise WriterModelError("reward preference lost train24 task authority")
    return tuple(reward_tasks), tuple(writer_tasks)


def _optimizer(writer: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.AdamW:
    cell = config["optimization"]["optimizer"]
    return torch.optim.AdamW(
        (value for value in writer.parameters() if value.requires_grad),
        lr=float(cell["lr"]),
        betas=tuple(cell["betas"]),
        eps=float(cell["eps"]),
        weight_decay=float(cell["weight_decay"]),
    )


def _publish_contract(
    runtime_args: argparse.Namespace,
    context: DistributedContext,
    contract: Mapping[str, Any],
) -> None:
    if context.is_main:
        path = runtime_args.output_dir / "run_contract.json"
        if runtime_args.resume is None:
            if runtime_args.output_dir.exists() and any(
                runtime_args.output_dir.iterdir()
            ):
                raise WriterModelError("fresh reward output is not empty")
            runtime_args.output_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(path, dict(contract))
        elif not path.is_file() or read_json(path) != dict(contract):
            raise WriterModelError("reward exact-resume launch contract changed")
        append_jsonl(
            runtime_args.output_dir / "invocations.jsonl",
            {
                "argv": list(os.sys.argv),
                "host": socket.gethostname(),
                "resume": str(runtime_args.resume) if runtime_args.resume else None,
                "stop_after_cycle": runtime_args.stop_after_cycle,
                "started_unix": time.time(),
            },
        )
    barrier(context)


def _contract(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    config: Mapping[str, Any],
    base_config: Mapping[str, Any],
    source: Mapping[str, Any],
    trainable: Mapping[str, Any],
) -> dict[str, Any]:
    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "device": str(context.device),
        "numa_node": context.numa_node,
        "cpu_affinity": list(context.cpu_affinity or ()),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    topology: list[Any] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(topology, local)
    else:
        topology[0] = local
    state = git_state(Path(__file__).resolve().parents[3])
    return {
        "schema_version": REWARD_LAUNCH_SCHEMA,
        "mode": args.mode,
        "git": {"branch": state["branch"], "commit": state["commit"]},
        "config_path": str(args.config),
        "base_as_config_path": config["resolved_base_as_config"],
        "base_as_schema": base_config["schema_version"],
        "initialization": {
            **dict(config["initialization"]),
            "checkpoint": config["resolved_cold_start"],
        },
        "source": dict(source),
        "information_wall": dict(config["information_wall"]),
        "writer": dict(base_config["writer"]),
        "data": dict(config["data"]),
        "environment": dict(config["environment"]),
        "objective": dict(config["objective"]),
        "rng": dict(config["rng"]),
        "optimization": dict(config["optimization"]),
        "formal_run": dict(config["formal_run"]),
        "task_ids": [
            task.global_task_id
            for task in _load_tasks(data_root=args.data_root, base_config=base_config)[
                0
            ]
        ],
        "runtime": {
            "world_size": context.world_size,
            "rank_topology": topology,
            "total_cycles": int(config["formal_run"]["total_cycles"]),
            "task_assignment": config["data"]["task_queue"],
        },
        "trainable": dict(trainable),
    }


def prepare_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> RewardRuntime:
    config, base_config = load_reward_config(args.config)
    require_reward_mode(config, args.mode)
    if args.mode == "smoke" and context.world_size != 1:
        raise WriterModelError("reward smoke uses one GPU")
    allowed = config["formal_run"]["allowed_world_sizes"]
    if context.world_size not in allowed:
        raise WriterModelError("reward world size is outside 1--6")
    if args.mode == "formal":
        state = git_state(Path(__file__).resolve().parents[3])
        if not git_state_is_clean_pushed_or_frozen_authority(state):
            raise WriterModelError("formal reward training requires clean pushed Git")
    seed_everything(int(config["rng"]["optimizer_seed"]), context)
    authorities, source, _ = load_run_authorities(args, base_config)
    policy = load_policy(
        Path(source["model_path"]), authorities.source_base_config, context.device
    )
    writer, lora = build_writer(
        base_config,
        policy,
        asset_root=args.source_run.resolve().parents[2],
    )
    writer.to(context.device)
    writer.load_state_dict(
        load_file(
            str(Path(config["resolved_cold_start"]) / "writer.safetensors"),
            device=str(context.device),
        ),
        strict=True,
    )
    writer.train()
    trainable = writer_trainable_contract(writer, policy, lora)
    trainable["object"] = "ordered_procedure_reward_preference_writer_only"
    optimizer = _optimizer(writer, config)
    initialize_deferred_process_group(context, rendezvous_root=args.output_dir.parent)
    contract = _contract(
        args=args,
        context=context,
        config=config,
        base_config=base_config,
        source=source,
        trainable=trainable,
    )
    _publish_contract(args, context, contract)
    start_cycle = checkpoint_cycle(args.resume)
    if args.resume is not None:
        loaded, _ = load_reward_checkpoint(
            checkpoint=args.resume,
            context=context,
            writer=writer,
            optimizer=optimizer,
            contract=contract,
        )
        if loaded != start_cycle:
            raise WriterModelError("reward resume cursor changed")
    stop_cycle = (
        1
        if args.mode == "smoke"
        else int(args.stop_after_cycle or config["formal_run"]["stage_stop_cycles"][0])
    )
    if args.mode == "formal" and (
        stop_cycle not in config["formal_run"]["stage_stop_cycles"]
        or not start_cycle < stop_cycle
    ):
        raise WriterModelError("reward formal stop boundary changed")
    tasks, writer_tasks = _load_tasks(data_root=args.data_root, base_config=base_config)
    source_config = authorities.source_base_config
    length = int(source_config["features"]["tokenizer_max_length"])
    processor = Pi05LiberoProcessor(
        load_stats(source_config, source_config["data"]["active_task_ids"]),
        args.tokenizer_path,
        length,
        str(context.device),
    )
    tokenizer = Pi05TeacherPrefixTokenizer(
        args.tokenizer_path, length, str(context.device)
    )
    language = {task.task_id: tokenizer([task.language]) for task in writer_tasks}
    store = RawTeacherVideoStore(
        writer_tasks,
        frame_stride=int(base_config["writer"]["frame_stride"]),
        max_open_files=int(base_config["data"]["video_open_files_per_rank"]),
    )
    first, last = config["data"]["demo_indices"]
    schedule = TeacherVideoSchedule(
        task_ids=[task.global_task_id for task in tasks],
        demo_indices=range(first, last + 1),
        seed=int(config["data"]["teacher_video_seed"]),
        videos_per_visit=4,
    )
    libero_paths = prepare_libero_config(
        args.output_dir / f".libero_config_rank_{context.rank:02d}"
    )
    env_pool = RandomResetEnvironmentPool(
        bddl_root=Path(libero_paths["bddl_files"]),
        assets_root=Path(libero_paths["assets"]),
        render_resolution=int(config["environment"]["render_resolution"]),
    )
    torch.cuda.reset_peak_memory_stats(context.device)
    return RewardRuntime(
        args=args,
        context=context,
        config=config,
        base_config=base_config,
        source_config=source_config,
        tasks=tasks,
        writer_tasks=writer_tasks,
        video_store=store,
        video_schedule=schedule,
        language_tokens=language,
        processor=processor,
        env_pool=env_pool,
        policy=policy,
        writer=writer,
        identity_state=writer.template_state(),
        lora_contract=lora,
        optimizer=optimizer,
        trainable_parameters=tuple(
            value for value in writer.parameters() if value.requires_grad
        ),
        gradient_layout=parameter_layout(writer),
        contract=contract,
        start_cycle=start_cycle,
        stop_cycle=stop_cycle,
        metrics_path=args.output_dir / "metrics.jsonl",
    )


def _claim_task(
    runtime: RewardRuntime, queue: Path, ordered: tuple[RewardTask, ...]
) -> RewardTask | None:
    with queue.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        value = int(handle.read().strip())
        if value >= len(ordered):
            return None
        handle.seek(0)
        handle.truncate()
        handle.write(str(value + 1))
        handle.flush()
    return ordered[value]


def _task_order(runtime: RewardRuntime, cycle: int) -> tuple[RewardTask, ...]:
    del cycle
    return tuple(
        sorted(runtime.tasks, key=lambda task: (-task.horizon, task.global_task_id))
    )


def _trajectory_row(value: RewardTrajectory) -> dict[str, Any]:
    return {
        "rollout_cursor": value.rollout_cursor,
        "environment_seed": value.env_seed,
        "success": value.success,
        "steps": value.steps,
        "reward_sum": value.reward_sum,
        "replay_chunks": len(value.valid_action_steps),
        "valid_action_steps": list(value.valid_action_steps),
    }


def _task_gradient(
    runtime: RewardRuntime,
    task: RewardTask,
    cycle: int,
    flat: torch.Tensor,
    probe: RewardProbe | None,
) -> tuple[dict[str, Any], RewardProbe | None]:
    visit = cycle - 1
    demos = runtime.video_schedule.demos_for_task_visit(task.global_task_id, visit)
    packed, video_metrics = pack_teacher_condition(
        runtime.video_store,
        task_id=task.global_task_id,
        demos=demos,
        language=runtime.language_tokens[task.global_task_id],
        device=runtime.context.device,
    )
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=runtime.context.device.type,
            dtype=torch.bfloat16,
            enabled=runtime.context.device.type == "cuda",
        ),
    ):
        encoded = runtime.writer.encode_program(*packed, policy=runtime.policy)
        rollout_lora = runtime.writer.decode_program(encoded.program)
    copy_task_lora_state_(runtime.policy, rollout_lora, runtime.lora_contract)
    rollout_cursors = tuple(visit * 4 + lane for lane in range(4))
    env_seeds = tuple(
        reward_credit_environment_seed(
            int(runtime.config["rng"]["environment_seed_root"]),
            task.suite,
            task.task_id,
            int(runtime.config["rng"]["optimizer_seed"]),
            cursor,
        )
        for cursor in rollout_cursors
    )
    started = time.monotonic()
    environment = runtime.config["environment"]
    trajectories = collect_randomized_reward_trajectories(
        envs=tuple(runtime.env_pool.get(task, lane=lane) for lane in range(4)),
        policy=runtime.policy,
        preprocess=runtime.processor,
        postprocess=runtime.processor.unnormalize_action,
        suite=task.suite,
        task_id=task.task_id,
        global_task_id=task.global_task_id,
        language=task.language,
        adaptation_seed=int(runtime.config["rng"]["optimizer_seed"]),
        rollout_cursors=rollout_cursors,
        env_seeds=env_seeds,
        policy_seed_root=int(runtime.config["rng"]["policy_noise_seed_root"]),
        device=runtime.context.device,
        max_horizon=task.horizon,
        dummy_settling_steps=int(environment["dummy_settling_steps"]),
        dummy_action=environment["dummy_action"],
        action_execution_horizon=int(environment["action_execution_horizon"]),
        num_inference_steps=int(environment["num_inference_steps"]),
    )
    rollout_seconds = time.monotonic() - started
    batch, episode_ids, successes = complete_trajectory_batch(
        trajectories, torch.device("cpu")
    )
    copy_task_lora_state_(runtime.policy, runtime.identity_state, runtime.lora_contract)
    mixed = not bool((successes == successes[0]).all())
    summary = RewardPreferenceSummary(
        objective=0.0,
        successes=int(successes.sum()),
        replay_chunks=int(episode_ids.numel()),
        executed_action_steps=int(batch["executed_action_steps"].sum()),
        functional_policy_forwards=0,
        lora_gradient_rms=0.0,
    )
    credit_seconds = 0.0
    if mixed:
        credit_started = time.monotonic()
        with torch.autocast(
            device_type=runtime.context.device.type,
            dtype=torch.bfloat16,
            enabled=runtime.context.device.type == "cuda",
        ):
            recompiled = runtime.writer.compile_readouts(
                encoded.diagnostics.shared_core_slots,
                encoded.diagnostics.per_video_procedure_slots,
                packed[3],
            )
            generated = runtime.writer.decode_program(recompiled.program)
            lora_gradients, summary = functional_reward_lora_gradient(
                runtime.policy,
                generated,
                runtime.lora_contract,
                batch,
                episode_ids,
                successes,
                mc_samples=int(runtime.config["objective"]["flow_mc_samples"]),
                physical_microbatch_size=int(
                    runtime.config["optimization"]["reward_replay_chunk_batch_size"]
                ),
                flow_seed_root=int(runtime.config["rng"]["flow_credit_seed_root"]),
                cycle=cycle,
                global_task_id=task.global_task_id,
                device=runtime.context.device,
            )
            backpropagate_reward_preference(generated, lora_gradients)
        gradients = tuple(item.parameter.grad for item in runtime.gradient_layout)
        accumulate_flat_gradient(flat, gradients, runtime.gradient_layout)
        for item in runtime.gradient_layout:
            item.parameter.grad = None
        credit_seconds = time.monotonic() - credit_started
        if probe is None:
            first = trajectories[0]
            probe = RewardProbe(
                global_task_id=task.global_task_id,
                suite=task.suite,
                shared_core=encoded.diagnostics.shared_core_slots.detach(),
                per_video_procedure=encoded.diagnostics.per_video_procedure_slots.detach(),
                condition_video_offsets=packed[3],
                before_lora={
                    name: value.detach().clone() for name, value in generated.items()
                },
                query={
                    name: value.clone() for name, value in first.observations[0].items()
                },
                before_action=first.action_chunks[0].clone(),
                policy_noise_seed=first.policy_noise_seeds[0],
            )
    row = {
        "task_id": task.global_task_id,
        "suite": task.suite,
        "local_task_id": task.task_id,
        "cycle": cycle,
        "mixed": mixed,
        **asdict(summary),
        **video_metrics,
        "rollouts": 4,
        "trajectory_rows": [_trajectory_row(value) for value in trajectories],
        "rollout_seconds": rollout_seconds,
        "credit_seconds": credit_seconds,
        "target_dataset_action_reads": 0,
        "teacher_action_reads": 0,
    }
    return row, probe


def _lora_response(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    sums = {"lora_a": 0.0, "lora_b": 0.0, "effective_ba": 0.0}
    counts = {name: 0 for name in sums}
    for name, before_a in before.items():
        if not name.endswith(".lora_A.default.weight"):
            continue
        b_name = name.replace(".lora_A.default.weight", ".lora_B.default.weight")
        after_a, before_b, after_b = after[name], before[b_name], after[b_name]
        values = {
            "lora_a": after_a.float() - before_a.float(),
            "lora_b": after_b.float() - before_b.float(),
            "effective_ba": after_b.float() @ after_a.float()
            - before_b.float() @ before_a.float(),
        }
        for key, value in values.items():
            sums[key] += float(value.square().sum())
            counts[key] += value.numel()
    return {f"{key}_response_rms": math.sqrt(sums[key] / counts[key]) for key in sums}


@torch.inference_mode()
def _probe_after_update(
    runtime: RewardRuntime, probe: RewardProbe | None
) -> dict[str, Any] | None:
    if probe is None:
        return None
    with torch.autocast(
        device_type=runtime.context.device.type,
        dtype=torch.bfloat16,
        enabled=runtime.context.device.type == "cuda",
    ):
        encoded = runtime.writer.compile_readouts(
            probe.shared_core,
            probe.per_video_procedure,
            probe.condition_video_offsets,
        )
        after = runtime.writer.decode_program(encoded.program)
    response = _lora_response(probe.before_lora, after)
    copy_task_lora_state_(runtime.policy, after, runtime.lora_contract)
    generator = torch.Generator(device="cpu").manual_seed(probe.policy_noise_seed)
    noise = torch.randn(
        (
            1,
            int(runtime.policy.config.chunk_size),
            int(runtime.policy.config.max_action_dim),
        ),
        generator=generator,
    ).to(runtime.context.device)
    query = {
        name: value.to(runtime.context.device, non_blocking=True)
        for name, value in probe.query.items()
    }
    with torch.autocast(
        device_type=runtime.context.device.type,
        dtype=torch.bfloat16,
        enabled=runtime.context.device.type == "cuda",
    ):
        action = runtime.policy.predict_action_chunk(query, noise=noise, num_steps=10)
    copy_task_lora_state_(runtime.policy, runtime.identity_state, runtime.lora_contract)
    response["fixed_action_response_rms"] = float(
        (action.cpu().float() - probe.before_action.float()).square().mean().sqrt()
    )
    return {"task_id": probe.global_task_id, "suite": probe.suite, **response}


def _run_cycle(runtime: RewardRuntime, cycle: int) -> dict[str, Any]:
    started = time.monotonic()
    runtime.optimizer.zero_grad(set_to_none=True)
    flat = torch.zeros(
        runtime.gradient_layout[-1].stop,
        dtype=torch.float32,
        device=runtime.context.device,
    )
    records, probe = [], None
    if runtime.args.mode == "smoke":
        task_id = int(
            runtime.args.smoke_task_id or runtime.config["smoke_run"]["task_global_id"]
        )
        task = next(task for task in runtime.tasks if task.global_task_id == task_id)
        row, probe = _task_gradient(runtime, task, cycle, flat, probe)
        records.append(row)
        divisor = 1
    else:
        ordered = _task_order(runtime, cycle)
        queue = runtime.args.output_dir / f".cycle_{cycle:08d}_task_cursor"
        if runtime.context.is_main:
            queue.write_text("0", encoding="utf-8")
        barrier(runtime.context)
        while task := _claim_task(runtime, queue, ordered):
            row, probe = _task_gradient(runtime, task, cycle, flat, probe)
            records.append(row)
        barrier(runtime.context)
        if runtime.context.is_main:
            queue.unlink(missing_ok=True)
        divisor = 24
    if runtime.context.world_size > 1:
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(divisor)
    if not bool(torch.count_nonzero(flat)):
        raise WriterModelError("reward cycle produced zero shared gradient")
    assign_flat_gradient(flat, runtime.gradient_layout)
    clip = float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"])
    grad_norm = torch.nn.utils.clip_grad_norm_(runtime.trainable_parameters, clip)
    before = {
        name: value.detach().clone()
        for name, value in runtime.writer.procedure_set.named_parameters()
    }
    runtime.optimizer.step()
    parameter_delta = {
        name: float((value.detach() - before[name]).float().square().mean().sqrt())
        for name, value in runtime.writer.procedure_set.named_parameters()
    }
    probe_row = _probe_after_update(runtime, probe)
    if runtime.args.mode == "formal":
        global_records = gather_full24_records(
            records,
            world_size=runtime.context.world_size,
            task_ids=[task.global_task_id for task in runtime.tasks],
        )
    else:
        global_records = records
    probes: list[Any] = [None] * runtime.context.world_size
    if runtime.context.world_size > 1:
        dist.all_gather_object(probes, probe_row)
    else:
        probes[0] = probe_row
    probes = [value for value in probes if value is not None]
    torch.cuda.synchronize(runtime.context.device)
    elapsed = torch.tensor(
        time.monotonic() - started, dtype=torch.float64, device=runtime.context.device
    )
    if runtime.context.world_size > 1:
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return {
        "cycle": cycle,
        "cycle_semantics": (
            "one_complete_train24_reward_preference_update"
            if runtime.args.mode == "formal"
            else "one_task_live_smoke"
        ),
        "tasks": len(global_records),
        "rollouts": 4 * len(global_records),
        "successes": sum(int(row["successes"]) for row in global_records),
        "mixed_tasks": sum(bool(row["mixed"]) for row in global_records),
        "mixed_suites": sorted(
            {row["suite"] for row in global_records if row["mixed"]}
        ),
        "replay_chunks": sum(int(row["replay_chunks"]) for row in global_records),
        "executed_action_steps": sum(
            int(row["executed_action_steps"]) for row in global_records
        ),
        "reward_objective_mean": math.fsum(
            float(row["objective"]) for row in global_records
        )
        / len(global_records),
        "writer_gradient_norm_before_clip": float(grad_norm),
        "parameter_delta_rms": parameter_delta,
        "deployment_response_probes": probes,
        "task_records": global_records,
        "cycle_seconds": float(elapsed),
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
            runtime.context.device
        ),
        "max_cuda_reserved_bytes": torch.cuda.max_memory_reserved(
            runtime.context.device
        ),
        "target_dataset_action_reads": 0,
        "teacher_action_reads": 0,
        "validation_action_or_reward_reads": 0,
        "test_action_or_reward_reads": 0,
    }


def train(args: argparse.Namespace) -> None:
    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    runtime: RewardRuntime | None = None
    try:
        runtime = prepare_runtime(args, context)
        if context.is_main:
            print(
                json.dumps(
                    {
                        "event": "start",
                        "mode": args.mode,
                        "world_size": context.world_size,
                        "start_cycle": runtime.start_cycle,
                        "stop_cycle": runtime.stop_cycle,
                        "trainable_parameters": runtime.gradient_layout[-1].stop,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        for cycle in range(runtime.start_cycle + 1, runtime.stop_cycle + 1):
            row = _run_cycle(runtime, cycle)
            if context.is_main:
                append_jsonl(runtime.metrics_path, row)
                print(json.dumps(row, sort_keys=True), flush=True)
            if args.mode == "formal":
                save_reward_checkpoint(
                    output_dir=args.output_dir,
                    cycle=cycle,
                    context=context,
                    writer=runtime.writer,
                    optimizer=runtime.optimizer,
                    contract=runtime.contract,
                    metrics_rows=cycle,
                )
        if context.is_main:
            write_json_atomic(
                args.output_dir / "completion.json",
                {
                    "schema_version": "ember_pi05_v6_ordered_procedure_on_policy_preference_completion_v1",
                    "mode": args.mode,
                    "completed_cycle": runtime.stop_cycle,
                    "strict400_required": args.mode == "formal",
                },
            )
    finally:
        if runtime is not None:
            runtime.env_pool.close()
            runtime.video_store.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REWARD_CONFIG)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-cycle", type=int)
    parser.add_argument("--smoke-task-id", type=int)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
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
    for name in ("config", "source_run", "checkpoint", "tokenizer_path", "data_root"):
        if not getattr(args, name).exists():
            raise WriterModelError(
                f"missing reward training path: {getattr(args, name)}"
            )
    return args
