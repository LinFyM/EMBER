"""One successful-expert occupancy cycle for the shared CFMG Writer."""

from __future__ import annotations

import fcntl
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.lora import copy_task_lora_state_
from ember.pi05_source_checkpoint import barrier
from ember.reward.occupancy_panel import (
    complete_successful_expert_occupancy_batch,
    empty_successful_expert_occupancy_credit,
    successful_expert_occupancy_cycle_metrics,
)
from ember.reward.protocol import RewardTask, reward_credit_environment_seed
from ember.reward.rollout import (
    RewardTrajectory,
    collect_paired_reward_arm_trajectories,
    query_successful_expert_occupancy_actions,
)
from ember.writer.as_step import accumulate_flat_gradient, gather_full24_records
from ember.writer.data import pack_teacher_condition
from ember.writer.reward_gradient_update import (
    AppliedStep,
    RewardCreditView,
    RewardProbe,
    apply_direct_reward_step,
    make_reward_probe,
    probes_after_update,
)
from ember.writer.reward_preference import (
    SuccessfulExpertOccupancyCreditSummary,
    backpropagate_lora_cotangent,
    cross_video_gradient_geometry,
    functional_successful_expert_occupancy_gradient,
    mean_cross_video_task_gradient,
)

if TYPE_CHECKING:
    from ember.writer.model import WriterConditioningState
    from ember.writer.reward_training import RewardRuntime

TaskCreditResult = tuple[
    dict[str, Any],
    float,
    torch.Tensor,
    Mapping[str, torch.Tensor],
    torch.Tensor,
    tuple[RewardCreditView, ...],
]


def _claim_task(queue: Path, ordered: tuple[RewardTask, ...]) -> RewardTask | None:
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


def _encode_candidate_condition(
    runtime: RewardRuntime, task: RewardTask, demos: Sequence[int]
) -> tuple[
    tuple[Any, ...],
    Mapping[str, Any],
    WriterConditioningState,
    Mapping[str, torch.Tensor],
]:
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
        state = runtime.writer.encode_conditioning_state(*packed, policy=runtime.policy)
        encoded = runtime.writer.compile_conditioning_state(
            state, packed[3], use_query_delta=True
        )
        candidate = runtime.writer.decode_output(encoded)
    return packed, video_metrics, state, candidate


def _encode_task(runtime: RewardRuntime, task: RewardTask, cycle: int) -> tuple[
    int,
    tuple[int, ...],
    tuple[Any, ...],
    Mapping[str, Any],
    WriterConditioningState,
    Mapping[str, torch.Tensor],
]:
    visit = cycle - 1
    demos = runtime.video_schedule.demos_for_task_visit(task.global_task_id, visit)
    packed, video_metrics, state, candidate = _encode_candidate_condition(
        runtime, task, demos
    )
    return visit, tuple(demos), packed, video_metrics, state, candidate


def _collect_expert_trajectories(
    runtime: RewardRuntime,
    task: RewardTask,
    visit: int,
) -> tuple[tuple[RewardTrajectory, ...], float]:
    rollout_cursors = tuple(visit * 2 + lane for lane in range(2))
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
    environment = runtime.config["environment"]
    started = time.monotonic()
    try:
        copy_task_lora_state_(
            runtime.policy,
            runtime.expert_loras[task.global_task_id],
            runtime.lora_contract,
        )
        trajectories = collect_paired_reward_arm_trajectories(
            envs=tuple(runtime.env_pool.get(task, lane=lane) for lane in range(2)),
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
    finally:
        copy_task_lora_state_(
            runtime.policy, runtime.identity_state, runtime.lora_contract
        )
    return trajectories, time.monotonic() - started


def _differentiate_credit_view(
    runtime: RewardRuntime,
    packed: tuple[Any, ...],
    state: WriterConditioningState,
    candidate_lora: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    trajectory_ids: torch.Tensor,
    gradient_template: torch.Tensor,
) -> tuple[torch.Tensor, SuccessfulExpertOccupancyCreditSummary]:
    with torch.autocast(
        device_type=runtime.context.device.type,
        dtype=torch.bfloat16,
        enabled=runtime.context.device.type == "cuda",
    ):
        lora_gradient, summary = functional_successful_expert_occupancy_gradient(
            runtime.policy,
            candidate_lora,
            runtime.lora_contract,
            batch,
            trajectory_ids,
            endpoint_action_batch_size=int(
                runtime.config["optimization"]["endpoint_action_batch_size"]
            ),
            num_inference_steps=int(
                runtime.config["environment"]["num_inference_steps"]
            ),
            device=runtime.context.device,
        )
        recompiled = runtime.writer.compile_conditioning_state(
            state, packed[3], use_query_delta=True
        )
        generated = runtime.writer.decode_output(recompiled)
        backpropagate_lora_cotangent(generated, lora_gradient)
    flat = torch.zeros_like(gradient_template)
    gradients = tuple(item.parameter.grad for item in runtime.gradient_layout)
    accumulate_flat_gradient(flat, gradients, runtime.gradient_layout)
    for item in runtime.gradient_layout:
        item.parameter.grad = None
    return flat, summary


def _differentiate_credit_views(
    runtime: RewardRuntime,
    task: RewardTask,
    visit: int,
    anchor_demos: Sequence[int],
    packed: tuple[Any, ...],
    state: WriterConditioningState,
    candidate_lora: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    trajectory_ids: torch.Tensor,
    gradient_template: torch.Tensor,
) -> tuple[
    list[torch.Tensor],
    list[dict[str, Any]],
    tuple[RewardCreditView, ...],
    tuple[tuple[int, ...], ...],
]:
    demo_sets = runtime.video_schedule.cross_video_credit_demos_for_task_visit(
        task.global_task_id,
        visit,
        anchor_demos,
        view_count=int(runtime.config["data"]["credit_views_per_active_task"]),
    )
    view_gradients, view_rows, credit_views = [], [], []
    for view_index, demos in enumerate(demo_sets):
        if view_index == 0:
            view_packed, view_state, view_lora = packed, state, candidate_lora
            view_metrics: Mapping[str, Any] = {}
        else:
            view_packed, view_metrics, view_state, view_lora = (
                _encode_candidate_condition(runtime, task, demos)
            )
        flat, summary = _differentiate_credit_view(
            runtime,
            view_packed,
            view_state,
            view_lora,
            batch,
            trajectory_ids,
            gradient_template,
        )
        view_gradients.append(flat)
        credit_views.append(
            RewardCreditView(
                conditioning_state=view_state,
                condition_video_offsets=view_packed[3],
                before_credit_objective=summary.objective,
            )
        )
        view_rows.append(
            {
                "view_index": view_index,
                "demo_indices": list(demos),
                "parameter_grid_gradient_rms": float(flat.square().mean().sqrt()),
                **asdict(summary),
                **view_metrics,
            }
        )
    return view_gradients, view_rows, tuple(credit_views), demo_sets


def _differentiate_task_credit(
    runtime: RewardRuntime,
    task: RewardTask,
    visit: int,
    anchor_demos: Sequence[int],
    packed: tuple[Any, ...],
    state: WriterConditioningState,
    candidate_lora: Mapping[str, torch.Tensor],
    successful: Sequence[RewardTrajectory],
    gradient_template: torch.Tensor,
) -> TaskCreditResult:
    started = time.monotonic()
    actions_by_policy, action_metrics = query_successful_expert_occupancy_actions(
        policy=runtime.policy,
        lora_contract=runtime.lora_contract,
        identity_state=runtime.identity_state,
        trajectories=successful,
        expert_lora=runtime.expert_loras[task.global_task_id],
        student_lora=candidate_lora,
        device=runtime.context.device,
        microbatch_size=int(runtime.config["optimization"]["matched_action_batch_size"]),
        num_inference_steps=int(runtime.config["environment"]["num_inference_steps"]),
    )
    batch, trajectory_ids, selection_metrics = (
        complete_successful_expert_occupancy_batch(
            successful,
            actions_by_policy,
            strata_per_trajectory=int(
                runtime.config["objective"]["occupancy_strata_per_trajectory"]
            ),
            device=torch.device("cpu"),
        )
    )
    view_gradients, view_rows, credit_views, demo_sets = (
        _differentiate_credit_views(
            runtime,
            task,
            visit,
            anchor_demos,
            packed,
            state,
            candidate_lora,
            batch,
            trajectory_ids,
            gradient_template,
        )
    )
    task_gradient = mean_cross_video_task_gradient(view_gradients)
    first = view_rows[0]
    result = {
        "objective": math.fsum(float(row["objective"]) for row in view_rows) / 4,
        "expert_action_distance": math.fsum(
            float(row["expert_action_distance"]) for row in view_rows
        )
        / 4,
        "successful_trajectories": int(first["successful_trajectories"]),
        "selected_credit_states": int(first["selected_credit_states"]),
        "replay_rows": int(first["replay_rows"]),
        "successful_action_steps": int(first["successful_action_steps"]),
        "matched_expert_student_action_rms": float(
            first["matched_expert_student_action_rms"]
        ),
        "functional_policy_forwards": sum(
            int(row["functional_policy_forwards"]) for row in view_rows
        ),
        "functional_policy_backwards": sum(
            int(row["functional_policy_backwards"]) for row in view_rows
        ),
        "lora_gradient_rms": math.fsum(
            float(row["lora_gradient_rms"]) for row in view_rows
        )
        / 4,
        "credit_conditions": 4,
        "credit_unique_video_count": len(
            {demo for demos in demo_sets for demo in demos}
        ),
        "credit_view_records": view_rows,
        "cross_video_gradient_geometry": cross_video_gradient_geometry(view_gradients),
        **action_metrics,
        **selection_metrics,
    }
    return (
        result,
        time.monotonic() - started,
        task_gradient,
        batch,
        trajectory_ids,
        credit_views,
    )


def _task_gradient(
    runtime: RewardRuntime,
    task: RewardTask,
    cycle: int,
    gradient_template: torch.Tensor,
) -> tuple[dict[str, Any], RewardProbe | None, int, torch.Tensor | None]:
    visit, anchor_demos, packed, video_metrics, state, candidate_lora = _encode_task(
        runtime, task, cycle
    )
    expert, rollout_seconds = _collect_expert_trajectories(runtime, task, visit)
    successful = tuple(value for value in expert if value.success)
    credit = empty_successful_expert_occupancy_credit()
    credit_seconds = 0.0
    task_gradient = None
    probe = None
    active = int(bool(successful))
    if successful:
        (
            credit,
            credit_seconds,
            task_gradient,
            credit_batch,
            credit_trajectory_ids,
            credit_views,
        ) = _differentiate_task_credit(
            runtime,
            task,
            visit,
            anchor_demos,
            packed,
            state,
            candidate_lora,
            successful,
            gradient_template,
        )
        probe = make_reward_probe(
            runtime,
            task,
            state,
            packed[3],
            candidate_lora,
            successful,
            credit_batch,
            credit_trajectory_ids,
            float(credit["credit_view_records"][0]["objective"]),
            credit_views,
        )
    row = {
        "task_id": task.global_task_id,
        "rank": runtime.context.rank,
        "suite": task.suite,
        "local_task_id": task.task_id,
        "cycle": cycle,
        "active_credit": bool(active),
        "expert_successes": len(successful),
        "expert_failures": len(expert) - len(successful),
        **credit,
        **video_metrics,
        "anchor_demo_indices": list(anchor_demos),
        "expert_states": 2,
        "rollouts": 2,
        "expert_trajectory_rows": [_trajectory_row(value) for value in expert],
        "expert_rollout_seconds": rollout_seconds,
        "credit_seconds": credit_seconds,
        "target_dataset_action_reads": 0,
        "teacher_action_reads": 0,
        "privileged_expert_action_scope": "development_train24_success_only",
    }
    return row, probe, active, task_gradient


def _collect_cycle_tasks(
    runtime: RewardRuntime,
    cycle: int,
    gradient_template: torch.Tensor,
) -> tuple[list[dict[str, Any]], list[RewardProbe], int, dict[int, torch.Tensor]]:
    if runtime.args.mode == "smoke":
        task_id = int(runtime.args.smoke_task_ids[runtime.context.local_rank])
        task = next(task for task in runtime.tasks if task.global_task_id == task_id)
        row, probe, active, task_gradient = _task_gradient(
            runtime, task, cycle, gradient_template
        )
        gradients = (
            {task.global_task_id: task_gradient} if task_gradient is not None else {}
        )
        return [row], [probe] if probe is not None else [], active, gradients
    ordered = tuple(
        sorted(runtime.tasks, key=lambda task: (-task.horizon, task.global_task_id))
    )
    queue = runtime.args.output_dir / f".cycle_{cycle:08d}_task_cursor"
    if runtime.context.is_main:
        queue.write_text("0", encoding="utf-8")
    barrier(runtime.context)
    records: list[dict[str, Any]] = []
    task_gradients: dict[int, torch.Tensor] = {}
    probes: list[RewardProbe] = []
    active = 0
    while task := _claim_task(queue, ordered):
        row, task_probe, task_active, task_gradient = _task_gradient(
            runtime, task, cycle, gradient_template
        )
        records.append(row)
        active += task_active
        if task_probe is not None:
            probes.append(task_probe)
        if task_gradient is not None:
            task_gradients[task.global_task_id] = task_gradient
    barrier(runtime.context)
    if runtime.context.is_main:
        queue.unlink(missing_ok=True)
    return records, probes, active, task_gradients


def _gather_cycle_evidence(
    runtime: RewardRuntime,
    records: list[dict[str, Any]],
    local_probes: Sequence[RewardProbe],
    credit_rows: Sequence[Mapping[str, Any]],
    started: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    expected_task_ids = (
        [task.global_task_id for task in runtime.tasks]
        if runtime.args.mode == "formal"
        else list(runtime.args.smoke_task_ids)
    )
    global_records = gather_full24_records(
        records,
        world_size=runtime.context.world_size,
        task_ids=expected_task_ids,
        expected_count=len(expected_task_ids),
    )
    local_probe_rows = probes_after_update(runtime, local_probes, credit_rows)
    probe_shards: list[Any] = [None] * runtime.context.world_size
    if runtime.context.world_size > 1:
        dist.all_gather_object(probe_shards, local_probe_rows)
    else:
        probe_shards[0] = local_probe_rows
    probes = sorted(
        (dict(value) for shard in probe_shards for value in shard),
        key=lambda row: int(row["task_id"]),
    )
    if runtime.context.device.type == "cuda":
        torch.cuda.synchronize(runtime.context.device)
    elapsed = torch.tensor(
        time.monotonic() - started,
        dtype=torch.float64,
        device=runtime.context.device,
    )
    if runtime.context.world_size > 1:
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return global_records, probes, float(elapsed)


def _cycle_metrics(
    runtime: RewardRuntime,
    cycle: int,
    records: list[dict[str, Any]],
    probes: list[dict[str, Any]],
    step: AppliedStep,
    elapsed: float,
) -> dict[str, Any]:
    active_records = [row for row in records if row["active_credit"]]
    return {
        "cycle": cycle,
        "cycle_semantics": (
            "one_complete_train24_successful_expert_occupancy_distillation"
            if runtime.args.mode == "formal"
            else "four_suite_successful_expert_occupancy_live_smoke"
        ),
        "tasks": len(records),
        "expert_states": 2 * len(records),
        "rollouts": 2 * len(records),
        "expert_successes": sum(int(row["expert_successes"]) for row in records),
        "expert_failures": sum(int(row["expert_failures"]) for row in records),
        "active_tasks": step.active_tasks,
        "active_suites": sorted({row["suite"] for row in active_records}),
        **successful_expert_occupancy_cycle_metrics(records, active_records),
        "writer_gradient_norm_before_clip": step.gradient_norm,
        "writer_gradient_rms": step.gradient_rms,
        "gradient_coexistence": step.gradient_coexistence,
        "commitment_geometry": step.commitment_geometry,
        "parameter_delta_rms": step.parameter_delta_rms,
        "deployment_response_probes": probes,
        "task_records": records,
        "cycle_seconds": elapsed,
        "max_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(runtime.context.device)
            if runtime.context.device.type == "cuda"
            else 0
        ),
        "max_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved(runtime.context.device)
            if runtime.context.device.type == "cuda"
            else 0
        ),
        "target_dataset_action_reads": 0,
        "teacher_action_reads": 0,
        "privileged_train_expert_actions": True,
        "validation_action_or_reward_reads": 0,
        "test_action_or_reward_reads": 0,
    }


def run_cycle(runtime: RewardRuntime, cycle: int) -> dict[str, Any]:
    started = time.monotonic()
    runtime.optimizer.zero_grad(set_to_none=True)
    gradient_template = torch.zeros(
        runtime.gradient_layout[-1].stop,
        dtype=torch.float32,
        device=runtime.context.device,
    )
    records, local_probes, local_active, task_gradients = _collect_cycle_tasks(
        runtime, cycle, gradient_template
    )
    step = apply_direct_reward_step(
        runtime,
        local_active,
        task_gradients,
        local_probes,
    )
    records, probes, elapsed = _gather_cycle_evidence(
        runtime, records, local_probes, step.commitment_credit_rows, started
    )
    return _cycle_metrics(runtime, cycle, records, probes, step, elapsed)
