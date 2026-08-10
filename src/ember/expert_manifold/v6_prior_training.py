"""Full24 K4 Reward-Credit Program cotangents with exact full48 writes."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.expert_manifold.contract import ExpertManifoldError, ExpertTask
from ember.expert_manifold.v6_prior import (
    counterfactual_kind,
    cross_suite_wrong_task,
)
from ember.expert_manifold.v6_prior_checkpoint import save_v6_prior_checkpoint
from ember.expert_manifold.v6_prior_contract import (
    V6_PRIOR_CANONICAL_CONFIG,
    V6_PRIOR_COMPLETION_SCHEMA,
    V6_PRIOR_MODES,
    V6_PRIOR_PROFILE_SCHEMA,
    load_v6_prior_config,
)
from ember.expert_manifold.v6_prior_profile import (
    FixedActionProfilePanel,
    base_versions,
    profile_action_panel,
    profile_credit_motion,
    profile_lora_response,
    profile_passes,
)
from ember.expert_manifold.v6_prior_runtime import V6PriorRuntime, _prepare_runtime
from ember.expert_manifold.v6_prior_run_contract import cursor_contract
from ember.expert_manifold.v6_prior_step import (
    GeneratedConditionGraph,
    generate_condition_graph,
)
from ember.expert_manifold.v6_reward_credit import (
    RewardProgramCreditSummary,
    reward_credit_is_finite,
    reward_program_cotangent,
)
from ember.lora import copy_task_lora_state_
from ember.pi05_source_checkpoint import DistributedContext, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import initialize_distributed
from ember.reward.protocol import reward_credit_environment_seed
from ember.reward.rollout import (
    RewardTrajectory,
    collect_randomized_reward_trajectories,
    complete_trajectory_batch,
)
from ember.writer.condition_update import (
    AnchoredReconciliationUpdateSummary,
    ProgramDeltaApplicationSummary,
    anchored_reconciliation_program_delta,
    apply_anchored_reconciliation_update_,
    program_residual_delta_application_evidence,
)


@dataclass(frozen=True)
class TaskObjective:
    task: ExpertTask
    task_visit: int
    teacher_demo: int
    counterfactual_kind: str
    counterfactual_task: ExpertTask | None
    counterfactual_demo: int | None
    correct_feature: torch.Tensor
    negative_feature: torch.Tensor
    program_cotangent: torch.Tensor
    credit: RewardProgramCreditSummary
    trajectory_rows: tuple[Mapping[str, Any], ...]
    correct_raw_frames: int
    correct_sampled_frames: int
    negative_raw_frames: int
    negative_sampled_frames: int
    rollout_seconds: float
    credit_seconds: float
    program_before: torch.Tensor | None = None
    correct_lora_before: Mapping[str, torch.Tensor] | None = None
    fixed_policy_panel: FixedActionProfilePanel | None = None


def _collect_task_replay(
    runtime: V6PriorRuntime,
    *,
    schedule_macro: int,
    task: ExpertTask,
    graph: GeneratedConditionGraph,
) -> tuple[
    tuple[RewardTrajectory, ...],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    FixedActionProfilePanel | None,
    float,
]:
    copy_task_lora_state_(runtime.policy, graph.correct_lora, runtime.lora_contract)
    reward_task = runtime.reward_task_by_global_id[task.global_task_id]
    rollout_cursors = tuple(schedule_macro * 4 + lane for lane in range(4))
    environment_seeds = tuple(
        reward_credit_environment_seed(
            int(runtime.config["rng"]["environment_seed_root"]),
            reward_task.suite,
            reward_task.task_id,
            int(runtime.config["optimization"]["seed"]),
            cursor,
        )
        for cursor in rollout_cursors
    )
    started = time.monotonic()
    environment = runtime.config["environment"]
    trajectories = collect_randomized_reward_trajectories(
        envs=tuple(runtime.env_pool.get(reward_task, lane=lane) for lane in range(4)),
        policy=runtime.policy,
        preprocess=runtime.processor,
        postprocess=runtime.processor.unnormalize_action,
        suite=reward_task.suite,
        task_id=reward_task.task_id,
        global_task_id=reward_task.global_task_id,
        language=reward_task.language,
        adaptation_seed=int(runtime.config["optimization"]["seed"]),
        rollout_cursors=rollout_cursors,
        env_seeds=environment_seeds,
        policy_seed_root=int(runtime.config["rng"]["policy_noise_seed_root"]),
        device=runtime.context.device,
        max_horizon=reward_task.horizon,
        dummy_settling_steps=int(environment["dummy_settling_steps"]),
        dummy_action=environment["dummy_action"],
        action_execution_horizon=int(environment["action_execution_horizon"]),
        num_inference_steps=int(environment["num_inference_steps"]),
        retain_failure_replay=True,
    )
    seconds = time.monotonic() - started
    batch, episode_ids, successes = complete_trajectory_batch(
        trajectories, torch.device("cpu")
    )
    fixed_panel = None
    if runtime.args.mode == "mechanism-profile":
        fixed_panel = profile_action_panel(trajectories)
    return (
        trajectories,
        batch,
        episode_ids,
        successes,
        fixed_panel,
        seconds,
    )


def _task_objective(
    runtime: V6PriorRuntime,
    *,
    schedule_macro: int,
    task: ExpertTask,
) -> TaskObjective:
    teacher_demo = runtime.video_schedule.demos_for_task_visit(
        task.global_task_id, schedule_macro
    )[0]
    correct_video = runtime.video_store.load(task.global_task_id, teacher_demo)
    kind = counterfactual_kind(task.ordinal, schedule_macro)
    negative_task = None
    negative_demo = None
    negative_video = None
    if kind == "wrong":
        negative_task = cross_suite_wrong_task(
            runtime.tasks,
            task_ordinal=task.ordinal,
            task_visit=schedule_macro,
        )
        negative_demo = runtime.video_schedule.demos_for_task_visit(
            negative_task.global_task_id, schedule_macro
        )[0]
        negative_video = runtime.video_store.load(
            negative_task.global_task_id, negative_demo
        )
    copy_task_lora_state_(runtime.policy, runtime.identity_state, runtime.lora_contract)
    graph = generate_condition_graph(
        writer=runtime.writer,
        policy=runtime.policy,
        correct_video=correct_video,
        counterfactual_video=negative_video,
        language_tokens=runtime.language_tokens[task.global_task_id],
        kind=kind,
        counterfactual_seed=int(runtime.config["data"]["counterfactual_seed"]),
        task_ordinal=task.ordinal,
        task_visit=schedule_macro,
        teacher_demo=teacher_demo,
        device=runtime.context.device,
    )
    replay = _collect_task_replay(
        runtime, schedule_macro=schedule_macro, task=task, graph=graph
    )
    (
        trajectories,
        batch,
        episode_ids,
        successes,
        fixed_panel,
        rollout_seconds,
    ) = replay
    copy_task_lora_state_(runtime.policy, runtime.identity_state, runtime.lora_contract)
    credit_started = time.monotonic()
    with torch.autocast(
        device_type=runtime.context.device.type,
        dtype=torch.bfloat16,
        enabled=runtime.context.device.type == "cuda",
    ):
        cotangent, credit = reward_program_cotangent(
            graph,
            policy=runtime.policy,
            contract=runtime.lora_contract,
            batch=batch,
            episode_ids=episode_ids,
            successes=successes,
            mc_samples=int(runtime.config["objective"]["flow_mc_samples"]),
            physical_microbatch_size=int(
                runtime.config["optimization"]["reward_replay_chunk_batch_size"]
            ),
            flow_seed_root=int(runtime.config["rng"]["flow_credit_seed_root"]),
            cycle=schedule_macro,
            global_task_id=task.global_task_id,
            device=runtime.context.device,
        )
    credit_seconds = time.monotonic() - credit_started
    if not reward_credit_is_finite(credit):
        raise ExpertManifoldError("Reward-Credit task summary became non-finite")
    if profile := runtime.args.mode == "mechanism-profile":
        if (fixed_panel is not None) is not bool(credit.mixed):
            raise ExpertManifoldError(
                "Reward-Credit action panel and mixed credit partition disagree"
            )
    counters = runtime.rank_counters
    counters["rollouts"] = int(counters["rollouts"]) + 4
    counters["environment_actions"] = int(counters["environment_actions"]) + sum(
        value.steps for value in trajectories
    )
    counters["successes"] = int(counters["successes"]) + sum(
        value.success for value in trajectories
    )
    counters["reward_sum"] = float(counters["reward_sum"]) + math.fsum(
        value.reward_sum for value in trajectories
    )
    trajectory_rows = tuple(_trajectory_record(value) for value in trajectories)
    return TaskObjective(
        task=task,
        task_visit=schedule_macro,
        teacher_demo=teacher_demo,
        counterfactual_kind=kind,
        counterfactual_task=negative_task,
        counterfactual_demo=negative_demo,
        correct_feature=graph.correct_feature.detach(),
        negative_feature=graph.negative_feature.detach(),
        program_cotangent=cotangent,
        credit=credit,
        trajectory_rows=trajectory_rows,
        correct_raw_frames=graph.correct_raw_frames,
        correct_sampled_frames=graph.correct_sampled_frames,
        negative_raw_frames=graph.negative_raw_frames,
        negative_sampled_frames=graph.negative_sampled_frames,
        rollout_seconds=rollout_seconds,
        credit_seconds=credit_seconds,
        program_before=(graph.program_input_before[0].detach() if profile else None),
        correct_lora_before=(
            {name: value.detach() for name, value in graph.correct_lora.items()}
            if profile
            else None
        ),
        fixed_policy_panel=fixed_panel,
    )


def _trajectory_record(value: RewardTrajectory) -> dict[str, Any]:
    return {
        "rollout_cursor": value.rollout_cursor,
        "environment_seed": value.env_seed,
        "policy_noise_seeds": list(value.policy_noise_seeds),
        "success": value.success,
        "steps": value.steps,
        "reward_sum": value.reward_sum,
        "replay_chunks": len(value.valid_action_steps),
        "valid_action_steps": list(value.valid_action_steps),
        "failure_replay_retained": not value.success,
    }


def _task_record(value: TaskObjective) -> dict[str, Any]:
    correct_norm = torch.linalg.vector_norm(value.correct_feature)
    negative_norm = torch.linalg.vector_norm(value.negative_feature)
    cosine = torch.dot(value.correct_feature, value.negative_feature) / (
        correct_norm * negative_norm
    ).clamp_min(torch.finfo(torch.float32).tiny)
    correct_norm_value, negative_norm_value, cosine_value = (
        torch.stack((correct_norm, negative_norm, cosine)).detach().cpu().tolist()
    )
    return {
        "task_ordinal": value.task.ordinal,
        "global_task_id": value.task.global_task_id,
        "suite": value.task.suite,
        "task_id": value.task.task_id,
        "task_visit": value.task_visit,
        "teacher_demo": value.teacher_demo,
        "videos": 1,
        "counterfactual_kind": value.counterfactual_kind,
        "counterfactual_global_task_id": (
            value.counterfactual_task.global_task_id
            if value.counterfactual_task is not None
            else None
        ),
        "counterfactual_demo": value.counterfactual_demo,
        **asdict(value.credit),
        "rollouts": len(value.trajectory_rows),
        "trajectory_rows": [dict(row) for row in value.trajectory_rows],
        "correct_feature_norm": correct_norm_value,
        "negative_feature_norm": negative_norm_value,
        "correct_negative_feature_cosine": cosine_value,
        "correct_raw_frames": value.correct_raw_frames,
        "correct_sampled_frames": value.correct_sampled_frames,
        "negative_raw_frames": value.negative_raw_frames,
        "negative_sampled_frames": value.negative_sampled_frames,
        "rollout_seconds": value.rollout_seconds,
        "credit_seconds": value.credit_seconds,
        "teacher_action_reads": 0,
        "source_action_reads": 0,
        "old_policy_forwards": 0,
        "negative_policy_forwards": 0,
        "writer_video_encodes": 2 if value.counterfactual_kind == "wrong" else 1,
    }


def _all_gather_fixed(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if context.world_size == 1:
        return value
    output = torch.empty(
        (context.world_size * value.shape[0], *value.shape[1:]),
        dtype=value.dtype,
        device=value.device,
    )
    dist.all_gather_into_tensor(output, value.contiguous())
    return output


def _credit_ready_rendezvous(
    runtime: V6PriorRuntime, *, macro: int
) -> tuple[Any, Path] | tuple[None, None]:
    torch.cuda.synchronize(runtime.context.device)
    if runtime.context.world_size <= 1:
        return None, None
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "")
    master_port = os.environ.get("MASTER_PORT", "")
    if not run_id or not master_port:
        raise ExpertManifoldError("Reward-Credit rendezvous lacks torchrun identity")
    token = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{run_id}-{master_port}")
    path = runtime.args.output_dir / (
        f".rank-local-reward-credit-ready-{token}-macro-{macro:08d}"
    )
    store = dist.FileStore(str(path), runtime.context.world_size)
    store.set(f"rank-{runtime.context.rank}", b"cuda-complete")
    store.wait(
        [f"rank-{rank}" for rank in range(runtime.context.world_size)],
        timedelta(minutes=30),
    )
    return store, path


def _gather_full48(
    runtime: V6PriorRuntime,
    local: Sequence[TaskObjective],
    *,
    macro: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expected_local = 24 // runtime.context.world_size
    if len(local) != expected_local:
        raise ExpertManifoldError("Reward-Credit local task coverage changed")
    payload = torch.empty(
        expected_local,
        1 + 2 * 256,
        dtype=torch.float32,
        device=runtime.context.device,
    )
    payload[:, 0] = torch.tensor(
        [value.task.ordinal for value in local],
        dtype=torch.float32,
        device=runtime.context.device,
    )
    payload[:, 1:257] = torch.stack([value.correct_feature for value in local])
    payload[:, 257:] = torch.stack([value.negative_feature for value in local])
    cotangents = torch.stack([value.program_cotangent for value in local])
    ready_store, ready_path = _credit_ready_rendezvous(runtime, macro=macro)
    gathered_payload = _all_gather_fixed(payload, runtime.context)
    gathered_cotangents = _all_gather_fixed(cotangents, runtime.context)
    if runtime.context.world_size > 1:
        dist.barrier(device_ids=[runtime.context.local_rank])
    del ready_store
    if runtime.context.is_main and ready_path is not None:
        ready_path.unlink(missing_ok=True)
    if runtime.context.world_size > 1:
        dist.barrier(device_ids=[runtime.context.local_rank])
    ordinals = gathered_payload[:, 0].to(dtype=torch.long)
    order = ordinals.argsort()
    if not torch.equal(
        ordinals.index_select(0, order),
        torch.arange(24, dtype=torch.long, device=runtime.context.device),
    ):
        raise ExpertManifoldError("Reward-Credit full48 task order changed")
    return (
        gathered_payload.index_select(0, order)[:, 1:257],
        gathered_payload.index_select(0, order)[:, 257:],
        gathered_cotangents.index_select(0, order),
    )


def _gather_task_records(
    local: list[dict[str, Any]], context: DistributedContext
) -> list[dict[str, Any]]:
    rows: list[Any] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(rows, local)
    else:
        rows[0] = local
    result = [dict(item) for rank_rows in rows for item in rank_rows]
    result.sort(key=lambda item: int(item["task_ordinal"]))
    if len(result) != 24 or [int(row["task_ordinal"]) for row in result] != list(
        range(24)
    ):
        raise ExpertManifoldError("Reward-Credit cycle did not cover train24")
    return result


def _runtime_maximums(
    context: DistributedContext, started: float
) -> tuple[float, int, int]:
    torch.cuda.synchronize(context.device)
    values = torch.tensor(
        (
            time.monotonic() - started,
            torch.cuda.max_memory_allocated(context.device),
            torch.cuda.max_memory_reserved(context.device),
        ),
        dtype=torch.float64,
        device=context.device,
    )
    if context.world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
    seconds, allocated, reserved = values.detach().cpu().tolist()
    return float(seconds), int(allocated), int(reserved)


def _collect_local_objectives(
    runtime: V6PriorRuntime, schedule_macro: int
) -> list[TaskObjective]:
    values = [
        _task_objective(runtime, schedule_macro=schedule_macro, task=task)
        for task in runtime.local_tasks
    ]
    if any(
        parameter.grad is not None for parameter in runtime.policy.parameters()
    ) or any(
        parameter.grad is not None
        for parameter in runtime.writer.base_writer.parameters()
    ):
        raise ExpertManifoldError("Reward-Credit touched frozen parameter gradients")
    return values


def _apply_update(
    runtime: V6PriorRuntime,
    local: Sequence[TaskObjective],
    *,
    macro: int,
    profile: bool,
) -> tuple[
    AnchoredReconciliationUpdateSummary,
    ProgramDeltaApplicationSummary | None,
    dict[str, float | int] | None,
    float,
]:
    update_started = time.monotonic()
    correct, negative, cotangents = _gather_full48(runtime, local, macro=macro)
    full_features = torch.cat((correct, negative), dim=0)
    before = runtime.writer.program_memory(full_features).clone() if profile else None
    delta, next_precision, update = anchored_reconciliation_program_delta(
        correct,
        negative,
        cotangents,
        runtime.reconciliation,
        step_size=float(runtime.config["update"]["step_size"]),
        relative_damping=float(runtime.config["update"]["relative_damping"]),
        reference_correct_features=None,
    )
    apply_anchored_reconciliation_update_(
        runtime.writer.program_memory,
        runtime.reconciliation,
        delta,
        next_precision,
        assimilated_rows_after=update.assimilated_rows_after,
    )
    application = None
    response = None
    if profile:
        if before is None:
            raise ExpertManifoldError("Reward-Credit profile lost before-memory")
        full_motion = torch.matmul(full_features.float(), delta.flatten(1)).reshape_as(
            before
        )
        application = program_residual_delta_application_evidence(
            runtime.writer.program_memory,
            delta,
            full_features,
            before,
            predicted=full_motion,
        )
        response = profile_lora_response(runtime, local, full_motion[:24])
        response.update(profile_credit_motion(cotangents, full_motion[:24]))
    torch.cuda.synchronize(runtime.context.device)
    elapsed = torch.tensor(
        time.monotonic() - update_started,
        dtype=torch.float64,
        device=runtime.context.device,
    )
    if runtime.context.world_size > 1:
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return update, application, response, float(elapsed)


def _macro_record(
    *,
    macro: int,
    schedule_macro: int,
    records: Sequence[Mapping[str, Any]],
    update: AnchoredReconciliationUpdateSummary,
    application: ProgramDeltaApplicationSummary | None,
    lora_response: Mapping[str, Any] | None,
    update_seconds: float,
    runtime_metrics: tuple[float, int, int],
) -> dict[str, Any]:
    counterfactual_counts = {
        name: sum(row["counterfactual_kind"] == name for row in records)
        for name in ("reversed", "shuffled", "wrong")
    }
    if counterfactual_counts != {"reversed": 8, "shuffled": 8, "wrong": 8}:
        raise ExpertManifoldError("Reward-Credit negative schedule changed")
    seconds, allocated, reserved = runtime_metrics
    mixed_tasks = sum(bool(value["mixed"]) for value in records)
    row = {
        "macro": macro + 1,
        "macro_semantics": "one_complete_full24_reward_cycle",
        "schedule_macro": schedule_macro,
        "tasks": len(records),
        "rollouts": sum(int(value["rollouts"]) for value in records),
        "videos": sum(int(value["videos"]) for value in records),
        "successes": sum(int(value["successes"]) for value in records),
        "failures": sum(int(value["failures"]) for value in records),
        "mixed_tasks": mixed_tasks,
        "homogeneous_tasks": 24 - mixed_tasks,
        "replay_chunks": sum(int(value["replay_chunks"]) for value in records),
        "executed_action_steps": sum(
            int(value["executed_action_steps"]) for value in records
        ),
        "reward_objective_mean": math.fsum(
            float(value["objective"]) for value in records
        )
        / 24,
        "program_cotangent_rms": math.sqrt(
            math.fsum(float(value["program_cotangent_rms"]) ** 2 for value in records)
            / 24
        ),
        "counterfactual_counts": counterfactual_counts,
        "update": asdict(update),
        "application": asdict(application) if application is not None else None,
        "lora_response": dict(lora_response) if lora_response is not None else None,
        "task_records": list(records),
        "rollout_seconds_sum": math.fsum(
            float(value["rollout_seconds"]) for value in records
        ),
        "credit_seconds_sum": math.fsum(
            float(value["credit_seconds"]) for value in records
        ),
        "update_seconds": update_seconds,
        "step_seconds": seconds,
        "max_cuda_allocated_bytes": allocated,
        "max_cuda_reserved_bytes": reserved,
        "old_policy_forwards": 0,
        "negative_policy_forwards": 0,
        "teacher_action_reads": 0,
        "source_action_reads": 0,
        "oom_count": 0,
        "nonfinite_count": 0,
        "watchdog_count": 0,
    }
    if not all(
        math.isfinite(float(row[name]))
        for name in (
            "reward_objective_mean",
            "program_cotangent_rms",
            "step_seconds",
        )
    ):
        raise ExpertManifoldError("Reward-Credit cycle metric became non-finite")
    return row


def _run_one_macro(runtime: V6PriorRuntime, *, macro: int) -> dict[str, Any]:
    profile = runtime.args.mode == "mechanism-profile"
    schedule_macro = runtime.segment.schedule_origin + macro
    versions_before = base_versions(runtime) if profile else ()
    started = time.monotonic()
    local = _collect_local_objectives(runtime, schedule_macro)
    update, application, response, update_seconds = _apply_update(
        runtime, local, macro=macro, profile=profile
    )
    records = _gather_task_records(
        [_task_record(value) for value in local], runtime.context
    )
    runtime_metrics = _runtime_maximums(runtime.context, started)
    if profile and base_versions(runtime) != versions_before:
        raise ExpertManifoldError("historical v6 state changed during profile")
    return _macro_record(
        macro=macro,
        schedule_macro=schedule_macro,
        records=records,
        update=update,
        application=application,
        lora_response=response,
        update_seconds=update_seconds,
        runtime_metrics=runtime_metrics,
    )


def _run_mechanism_profile(runtime: V6PriorRuntime) -> None:
    rows = [_run_one_macro(runtime, macro=runtime.segment.start_macro)]
    passed, gate_evidence = profile_passes(runtime.config, rows)
    result = {
        "schema_version": V6_PRIOR_PROFILE_SCHEMA,
        "passed": passed,
        "schedule_macro": runtime.segment.schedule_origin,
        "retain_weight": False,
        "gates": dict(runtime.config["profile_run"]["gates"]),
        "gate_evidence": gate_evidence,
        "macros": rows,
        "content_hash_policy": "disabled_by_owner",
    }
    if runtime.context.is_main:
        write_json_atomic(runtime.args.output_dir / "mechanism_profile.json", result)
        write_json_atomic(
            runtime.args.output_dir / "completion.json",
            {
                "schema_version": V6_PRIOR_COMPLETION_SCHEMA,
                "mode": "mechanism-profile",
                "completed_diagnostic_macros": 1,
                "passed": passed,
                "retained_checkpoint": False,
                "content_hash_policy": "disabled_by_owner",
            },
        )
        print(json.dumps(result, sort_keys=True), flush=True)


def _run_training(runtime: V6PriorRuntime) -> None:
    started = time.monotonic()
    for macro in range(runtime.segment.start_macro, runtime.segment.stop_macro):
        row = _run_one_macro(runtime, macro=macro)
        row["elapsed_seconds"] = time.monotonic() - started
        if runtime.context.is_main:
            append_jsonl(runtime.metrics_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)
        cursor = macro + 1
        if cursor in runtime.segment.checkpoint_macros:
            save_v6_prior_checkpoint(
                output_dir=runtime.args.output_dir,
                macro=cursor,
                memory=runtime.writer.program_memory,
                reconciliation=runtime.reconciliation,
                context=runtime.context,
                metrics_rows=cursor,
                cursor_contract=cursor_contract(runtime.config, cursor),
                checkpoint_contract=runtime.checkpoint_contract,
                interaction_cursor={
                    "next_macro": cursor,
                    "rollouts": int(runtime.rank_counters["rollouts"]),
                    "environment_actions": int(
                        runtime.rank_counters["environment_actions"]
                    ),
                    "successes": int(runtime.rank_counters["successes"]),
                    "reward_sum": float(runtime.rank_counters["reward_sum"]),
                    "pending_environment_episodes": 0,
                    "pending_policy_action_chunks": 0,
                    "pending_replay_microbatches": 0,
                },
            )
    if runtime.context.is_main:
        write_json_atomic(
            runtime.args.output_dir / "completion.json",
            {
                "schema_version": V6_PRIOR_COMPLETION_SCHEMA,
                "mode": "formal",
                "completed_macro": runtime.segment.stop_macro,
                "macro_semantics": "one_complete_full24_reward_cycle",
                "metrics_rows": runtime.segment.stop_macro,
                "strict400_required_before_continuation": (
                    runtime.segment.stop_macro == 1
                ),
                "content_hash_policy": "disabled_by_owner",
            },
        )


def _start_event(runtime: V6PriorRuntime) -> dict[str, Any]:
    return {
        "event": "start",
        "mode": runtime.args.mode,
        "start_macro": runtime.segment.start_macro,
        "stop_macro": runtime.segment.stop_macro,
        "macro_semantics": "one_complete_full24_reward_cycle",
        "historical_v6_frozen_parameters": runtime.ownership.frozen_parameter_count,
        "residual_memory_values": runtime.writer.program_memory.value.numel(),
        "rollout_policy_batch_size": int(
            runtime.config["optimization"]["rollout_policy_batch_size"]
        ),
        "reward_replay_chunk_batch_size": int(
            runtime.config["optimization"]["reward_replay_chunk_batch_size"]
        ),
        "flow_mc_samples": int(runtime.config["objective"]["flow_mc_samples"]),
    }


def train(args: argparse.Namespace) -> None:
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    runtime: V6PriorRuntime | None = None
    try:
        runtime = _prepare_runtime(args, context)
        if context.is_main:
            print(json.dumps(_start_event(runtime), sort_keys=True), flush=True)
        if args.mode == "mechanism-profile":
            _run_mechanism_profile(runtime)
        else:
            _run_training(runtime)
    finally:
        if runtime is not None:
            runtime.env_pool.close()
            runtime.video_store.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=V6_PRIOR_CANONICAL_CONFIG)
    parser.add_argument("--mode", choices=V6_PRIOR_MODES, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-macro", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in ("config", "source_run", "checkpoint", "tokenizer_path", "data_root"):
        path = getattr(args, name).resolve()
        if not path.exists():
            raise ExpertManifoldError(f"missing Reward-Credit path: {path}")
        setattr(args, name, path)
    args.output_dir = args.output_dir.resolve()
    args.resume = args.resume.resolve() if args.resume else None
    if args.resume is None:
        if args.output_dir.exists() and (
            not args.output_dir.is_dir() or any(args.output_dir.iterdir())
        ):
            raise ExpertManifoldError("fresh Reward-Credit output is not empty")
    elif (
        not args.resume.is_dir()
        or args.resume.parent.name != "checkpoints"
        or args.resume.parent.parent.resolve() != args.output_dir
        or not (args.output_dir / "run_contract.json").is_file()
    ):
        raise ExpertManifoldError("Reward-Credit resume output ownership changed")
    if args.config != V6_PRIOR_CANONICAL_CONFIG.resolve():
        raise ExpertManifoldError("Reward-Credit requires the canonical config")
    load_v6_prior_config(args.config)
    if args.num_workers != 0 or (
        args.stop_after_macro is not None and args.stop_after_macro <= 0
    ):
        raise ExpertManifoldError("invalid Reward-Credit worker or stop boundary")
    return args
