"""One matched-batch stratified occupancy cycle for direct factor heads."""

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
from ember.reward.protocol import RewardTask, reward_credit_environment_seed
from ember.reward.occupancy_panel import (
    complete_matched_stratified_occupancy_batch,
    empty_matched_occupancy_credit,
    occupancy_cycle_metrics,
)
from ember.reward.rollout import (
    RewardTrajectory,
    capture_paired_initial_states,
    collect_paired_reward_arm_trajectories,
    query_matched_occupancy_actions,
)
from ember.writer.as_step import accumulate_flat_gradient, gather_full24_records
from ember.writer.data import pack_teacher_condition
from ember.writer.errors import WriterModelError
from ember.writer.model import WriterConditioningState, WriterProgramOutput
from ember.writer.reward_preference import (
    MatchedStratifiedOccupancyCreditSummary,
    backpropagate_lora_cotangent,
    cross_video_gradient_geometry,
    functional_matched_stratified_occupancy_endpoint_gradient,
    mean_cross_video_task_gradient,
)
from ember.writer.reward_gradient_update import (
    AppliedStep,
    RewardPreferenceView,
    RewardProbe,
    apply_monotone_reward_step,
    make_reward_probe,
    probe_after_update,
)

if TYPE_CHECKING:
    from ember.writer.reward_training import RewardRuntime

TaskCreditResult = tuple[
    dict[str, Any],
    float,
    torch.Tensor,
    Mapping[str, torch.Tensor],
    torch.Tensor,
    tuple[RewardPreferenceView, ...],
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
    WriterProgramOutput,
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
    return packed, video_metrics, state, encoded, candidate


def _encode_pair(runtime: RewardRuntime, task: RewardTask, cycle: int) -> tuple[
    int,
    tuple[int, ...],
    tuple[Any, ...],
    Mapping[str, Any],
    WriterConditioningState,
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
]:
    visit = cycle - 1
    demos = runtime.video_schedule.demos_for_task_visit(task.global_task_id, visit)
    packed, video_metrics, state, encoded, candidate = _encode_candidate_condition(
        runtime, task, demos
    )
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=runtime.context.device.type,
            dtype=torch.bfloat16,
            enabled=runtime.context.device.type == "cuda",
        ),
    ):
        if encoded.reference_program is None:
            raise WriterModelError("direct-factor candidate lost AS139 reference")
        reference = runtime.writer.decode_program(encoded.reference_program)
    return visit, tuple(demos), packed, video_metrics, state, reference, candidate


def _collect_arm(
    runtime: RewardRuntime,
    task: RewardTask,
    visit: int,
    initial_states: Sequence[Any],
) -> tuple[RewardTrajectory, ...]:
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
    return collect_paired_reward_arm_trajectories(
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
        initial_states=initial_states,
    )


def _collect_paired_arms(
    runtime: RewardRuntime,
    task: RewardTask,
    visit: int,
    reference_lora: Mapping[str, torch.Tensor],
    candidate_lora: Mapping[str, torch.Tensor],
) -> tuple[tuple[RewardTrajectory, ...], tuple[RewardTrajectory, ...], float, float]:
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
    initial_states = capture_paired_initial_states(
        tuple(runtime.env_pool.get(task, lane=lane) for lane in range(2)),
        env_seeds,
        dummy_action=environment["dummy_action"],
        dummy_settling_steps=int(environment["dummy_settling_steps"]),
    )
    reference_started = time.monotonic()
    try:
        copy_task_lora_state_(runtime.policy, reference_lora, runtime.lora_contract)
        reference = _collect_arm(runtime, task, visit, initial_states)
        reference_seconds = time.monotonic() - reference_started
        candidate_started = time.monotonic()
        copy_task_lora_state_(runtime.policy, candidate_lora, runtime.lora_contract)
        candidate = _collect_arm(runtime, task, visit, initial_states)
        candidate_seconds = time.monotonic() - candidate_started
    finally:
        copy_task_lora_state_(
            runtime.policy, runtime.identity_state, runtime.lora_contract
        )
    return reference, candidate, reference_seconds, candidate_seconds


def _same_pair_identifiers(
    reference: RewardTrajectory, candidate: RewardTrajectory
) -> bool:
    names = (
        "suite",
        "task_id",
        "global_task_id",
        "adaptation_seed",
        "rollout_cursor",
        "env_seed",
        "policy_seed_root",
        "dummy_settling_steps",
    )
    if any(getattr(reference, name) != getattr(candidate, name) for name in names):
        return False
    shared = min(len(reference.policy_noise_seeds), len(candidate.policy_noise_seeds))
    return (
        shared > 0
        and reference.policy_noise_seeds[:shared]
        == candidate.policy_noise_seeds[:shared]
    )


def select_discordant_trajectory_pairs(
    reference: Sequence[RewardTrajectory],
    candidate: Sequence[RewardTrajectory],
) -> tuple[tuple[tuple[RewardTrajectory, RewardTrajectory], ...], tuple[str, ...]]:
    """Return winner/loser pairs only when the exact two arms disagree."""

    if len(reference) != 2 or len(candidate) != 2:
        raise WriterModelError("direct-factor credit requires two paired states")
    pairs: list[tuple[RewardTrajectory, RewardTrajectory]] = []
    labels: list[str] = []
    for reference_row, candidate_row in zip(reference, candidate, strict=True):
        if not _same_pair_identifiers(reference_row, candidate_row):
            raise WriterModelError("direct-factor arm pairing changed")
        if candidate_row.success and not reference_row.success:
            pairs.append((candidate_row, reference_row))
            labels.append("candidate")
        elif reference_row.success and not candidate_row.success:
            pairs.append((reference_row, candidate_row))
            labels.append("reference")
        elif reference_row.success:
            labels.append("both_success")
        else:
            labels.append("both_failure")
    return tuple(pairs), tuple(labels)


def _differentiate_credit_view(
    runtime: RewardRuntime,
    task: RewardTask,
    cycle: int,
    packed: tuple[Any, ...],
    state: WriterConditioningState,
    candidate_lora: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    trajectory_ids: torch.Tensor,
    gradient_template: torch.Tensor,
) -> tuple[torch.Tensor, MatchedStratifiedOccupancyCreditSummary]:
    with torch.autocast(
        device_type=runtime.context.device.type,
        dtype=torch.bfloat16,
        enabled=runtime.context.device.type == "cuda",
    ):
        lora_gradient, summary = (
            functional_matched_stratified_occupancy_endpoint_gradient(
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
    cycle: int,
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
    tuple[RewardPreferenceView, ...],
    tuple[tuple[int, ...], ...],
]:
    demo_sets = runtime.video_schedule.cross_video_credit_demos_for_task_visit(
        task.global_task_id,
        visit,
        anchor_demos,
        view_count=int(runtime.config["data"]["credit_views_per_active_task"]),
    )
    view_gradients, view_rows, preference_views = [], [], []
    for view_index, demos in enumerate(demo_sets):
        if view_index == 0:
            view_packed, view_state, view_lora = packed, state, candidate_lora
            view_metrics: Mapping[str, Any] = {}
        else:
            view_packed, view_metrics, view_state, _, view_lora = (
                _encode_candidate_condition(runtime, task, demos)
            )
        flat, summary = _differentiate_credit_view(
            runtime,
            task,
            cycle,
            view_packed,
            view_state,
            view_lora,
            batch,
            trajectory_ids,
            gradient_template,
        )
        view_gradients.append(flat)
        if runtime.args.mode == "smoke" or view_index == 0:
            preference_views.append(
                RewardPreferenceView(
                    conditioning_state=view_state,
                    condition_video_offsets=view_packed[3],
                    before_preference_margin=summary.preference_margin,
                )
            )
        view_rows.append(
            {
                "view_index": view_index,
                "demo_indices": list(demos),
                "factor_commitment_gradient_rms": float(flat.square().mean().sqrt()),
                **asdict(summary),
                **view_metrics,
            }
        )
    return view_gradients, view_rows, tuple(preference_views), demo_sets


def _differentiate_task_credit(
    runtime: RewardRuntime,
    task: RewardTask,
    cycle: int,
    visit: int,
    anchor_demos: Sequence[int],
    packed: tuple[Any, ...],
    state: WriterConditioningState,
    candidate_lora: Mapping[str, torch.Tensor],
    reference_lora: Mapping[str, torch.Tensor],
    pairs: Sequence[tuple[RewardTrajectory, RewardTrajectory]],
    active_labels: Sequence[str],
    gradient_sum: torch.Tensor,
) -> TaskCreditResult:
    started = time.monotonic()
    actions_by_arm, action_metrics = query_matched_occupancy_actions(
        policy=runtime.policy,
        lora_contract=runtime.lora_contract,
        identity_state=runtime.identity_state,
        pairs=pairs,
        active_labels=active_labels,
        reference_lora=reference_lora,
        candidate_lora=candidate_lora,
        device=runtime.context.device,
        microbatch_size=int(
            runtime.config["optimization"]["matched_action_batch_size"]
        ),
        num_inference_steps=int(runtime.config["environment"]["num_inference_steps"]),
    )
    batch, trajectory_ids, selection_metrics = (
        complete_matched_stratified_occupancy_batch(
            pairs,
            active_labels,
            actions_by_arm,
            strata_per_trajectory=int(
                runtime.config["objective"]["occupancy_strata_per_trajectory"]
            ),
            device=torch.device("cpu"),
        )
    )
    view_gradients, view_rows, preference_views, demo_sets = (
        _differentiate_credit_views(
            runtime,
            task,
            cycle,
            visit,
            anchor_demos,
            packed,
            state,
            candidate_lora,
            batch,
            trajectory_ids,
            gradient_sum,
        )
    )
    view_gradient_geometry = cross_video_gradient_geometry(view_gradients)
    task_gradient = mean_cross_video_task_gradient(view_gradients)
    gradient_sum.add_(task_gradient)
    first = view_rows[0]
    result = {
        "objective": math.fsum(float(row["objective"]) for row in view_rows) / 4,
        "preference_margin": math.fsum(
            float(row["preference_margin"]) for row in view_rows
        )
        / 4,
        "winner_action_distance": math.fsum(
            float(row["winner_action_distance"]) for row in view_rows
        )
        / 4,
        "loser_action_distance": math.fsum(
            float(row["loser_action_distance"]) for row in view_rows
        )
        / 4,
        "discordant_trajectories": int(first["discordant_trajectories"]),
        "selected_credit_pairs": int(first["selected_credit_pairs"]),
        "replay_rows": int(first["replay_rows"]),
        "successful_action_steps": int(first["successful_action_steps"]),
        "matched_winner_loser_action_rms": float(
            first["matched_winner_loser_action_rms"]
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
        "cross_video_gradient_geometry": view_gradient_geometry,
        **action_metrics,
        **selection_metrics,
    }
    return (
        result,
        time.monotonic() - started,
        task_gradient,
        batch,
        trajectory_ids,
        tuple(preference_views),
    )


def _task_gradient(
    runtime: RewardRuntime,
    task: RewardTask,
    cycle: int,
    gradient_sum: torch.Tensor,
    probe: RewardProbe | None,
) -> tuple[dict[str, Any], RewardProbe | None, int, torch.Tensor | None]:
    (
        visit,
        anchor_demos,
        packed,
        video_metrics,
        state,
        reference_lora,
        candidate_lora,
    ) = _encode_pair(runtime, task, cycle)
    reference, candidate, reference_seconds, candidate_seconds = _collect_paired_arms(
        runtime,
        task,
        visit,
        reference_lora,
        candidate_lora,
    )
    pairs, labels = select_discordant_trajectory_pairs(reference, candidate)
    credit = empty_matched_occupancy_credit()
    credit_seconds = 0.0
    task_gradient = None
    active = int(bool(pairs))
    if pairs:
        active_labels = tuple(
            label for label in labels if label in {"candidate", "reference"}
        )
        (
            credit,
            credit_seconds,
            task_gradient,
            preference_batch,
            preference_trajectory_ids,
            preference_views,
        ) = _differentiate_task_credit(
            runtime,
            task,
            cycle,
            visit,
            anchor_demos,
            packed,
            state,
            candidate_lora,
            reference_lora,
            pairs,
            active_labels,
            gradient_sum,
        )
        if probe is None:
            probe = make_reward_probe(
                runtime,
                task,
                state,
                packed[3],
                candidate_lora,
                candidate,
                labels,
                preference_batch,
                preference_trajectory_ids,
                float(credit["credit_view_records"][0]["preference_margin"]),
                preference_views,
            )
    row = {
        "task_id": task.global_task_id,
        "rank": runtime.context.rank,
        "suite": task.suite,
        "local_task_id": task.task_id,
        "cycle": cycle,
        "active_credit": bool(active),
        "pair_labels": list(labels),
        "reference_successes": sum(value.success for value in reference),
        "candidate_successes": sum(value.success for value in candidate),
        "candidate_gains": labels.count("candidate"),
        "reference_gains": labels.count("reference"),
        "both_success": labels.count("both_success"),
        "both_failure": labels.count("both_failure"),
        **credit,
        **video_metrics,
        "anchor_demo_indices": list(anchor_demos),
        "paired_states": 2,
        "rollouts": 4,
        "reference_trajectory_rows": [_trajectory_row(value) for value in reference],
        "candidate_trajectory_rows": [_trajectory_row(value) for value in candidate],
        "reference_rollout_seconds": reference_seconds,
        "candidate_rollout_seconds": candidate_seconds,
        "credit_seconds": credit_seconds,
        "target_dataset_action_reads": 0,
        "teacher_action_reads": 0,
    }
    return row, probe, active, task_gradient


def _collect_cycle_tasks(
    runtime: RewardRuntime,
    cycle: int,
    gradient_sum: torch.Tensor,
) -> tuple[
    list[dict[str, Any]],
    RewardProbe | None,
    int,
    dict[int, torch.Tensor],
]:
    if runtime.args.mode == "smoke":
        task_id = int(
            runtime.args.smoke_task_id or runtime.config["smoke_run"]["task_global_id"]
        )
        task = next(task for task in runtime.tasks if task.global_task_id == task_id)
        row, probe, active, task_gradient = _task_gradient(
            runtime, task, cycle, gradient_sum, None
        )
        gradients = (
            {task.global_task_id: task_gradient} if task_gradient is not None else {}
        )
        return [row], probe, active, gradients
    ordered = tuple(
        sorted(runtime.tasks, key=lambda task: (-task.horizon, task.global_task_id))
    )
    queue = runtime.args.output_dir / f".cycle_{cycle:08d}_task_cursor"
    if runtime.context.is_main:
        queue.write_text("0", encoding="utf-8")
    barrier(runtime.context)
    records: list[dict[str, Any]] = []
    task_gradients: dict[int, torch.Tensor] = {}
    probe = None
    active = 0
    while task := _claim_task(queue, ordered):
        row, probe, task_active, task_gradient = _task_gradient(
            runtime, task, cycle, gradient_sum, probe
        )
        records.append(row)
        active += task_active
        if task_gradient is not None:
            task_gradients[task.global_task_id] = task_gradient
    barrier(runtime.context)
    if runtime.context.is_main:
        queue.unlink(missing_ok=True)
    return records, probe, active, task_gradients


def _gather_cycle_evidence(
    runtime: RewardRuntime,
    records: list[dict[str, Any]],
    probe: RewardProbe | None,
    preference_rows: Sequence[Mapping[str, Any]],
    started: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    if runtime.args.mode == "formal":
        global_records = gather_full24_records(
            records,
            world_size=runtime.context.world_size,
            task_ids=[task.global_task_id for task in runtime.tasks],
        )
    else:
        global_records = records
    probe_row = probe_after_update(runtime, probe, preference_rows)
    probes: list[Any] = [None] * runtime.context.world_size
    if runtime.context.world_size > 1:
        dist.all_gather_object(probes, probe_row)
    else:
        probes[0] = probe_row
    if runtime.context.device.type == "cuda":
        torch.cuda.synchronize(runtime.context.device)
    elapsed = torch.tensor(
        time.monotonic() - started,
        dtype=torch.float64,
        device=runtime.context.device,
    )
    if runtime.context.world_size > 1:
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return (
        global_records,
        [value for value in probes if value is not None],
        float(elapsed),
    )


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
            "one_complete_train24_native_endpoint_action_preference"
            if runtime.args.mode == "formal"
            else "one_task_native_endpoint_action_preference_live_smoke"
        ),
        "tasks": len(records),
        "paired_states": 2 * len(records),
        "rollouts": 4 * len(records),
        "reference_successes": sum(int(row["reference_successes"]) for row in records),
        "candidate_successes": sum(int(row["candidate_successes"]) for row in records),
        "candidate_gains": sum(int(row["candidate_gains"]) for row in records),
        "reference_gains": sum(int(row["reference_gains"]) for row in records),
        "both_success": sum(int(row["both_success"]) for row in records),
        "both_failure": sum(int(row["both_failure"]) for row in records),
        "discordant_pairs": sum(
            int(row["candidate_gains"]) + int(row["reference_gains"]) for row in records
        ),
        "active_tasks": step.active_tasks,
        "active_suites": sorted({row["suite"] for row in active_records}),
        **occupancy_cycle_metrics(records, active_records),
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
        "validation_action_or_reward_reads": 0,
        "test_action_or_reward_reads": 0,
    }


def run_cycle(runtime: RewardRuntime, cycle: int) -> dict[str, Any]:
    started = time.monotonic()
    runtime.optimizer.zero_grad(set_to_none=True)
    gradient_sum = torch.zeros(
        runtime.gradient_layout[-1].stop,
        dtype=torch.float32,
        device=runtime.context.device,
    )
    records, probe, local_active, task_gradients = _collect_cycle_tasks(
        runtime, cycle, gradient_sum
    )
    step = apply_monotone_reward_step(
        runtime,
        gradient_sum,
        local_active,
        task_gradients,
        probe,
    )
    records, probes, elapsed = _gather_cycle_evidence(
        runtime, records, probe, step.commitment_preference_rows, started
    )
    return _cycle_metrics(runtime, cycle, records, probes, step, elapsed)
