"""OSG-PC task-complete credit, projection, and full48 Program updates."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
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
    REPO_ROOT,
    V6_PRIOR_CANONICAL_CONFIG,
    V6_PRIOR_COMPLETION_SCHEMA,
    V6_PRIOR_MODES,
    V6_PRIOR_PROFILE_SCHEMA,
    load_v6_prior_config,
)
from ember.expert_manifold.v6_prior_profile import (
    base_versions as _base_versions,
    profile_lora_response as _profile_lora_response,
    profile_max_seconds as _profile_max_seconds,
    profile_passes as _profile_passes,
    profile_success_guard_application as _profile_success_guard_application,
    profile_task_local_motion as _profile_task_local_motion,
)
from ember.expert_manifold.v6_prior_runtime import V6PriorRuntime, _prepare_runtime
from ember.expert_manifold.v6_prior_run_contract import cursor_contract
from ember.expert_manifold.v6_prior_step import (
    generate_condition_graph,
    program_cotangent,
    redecode_condition_graph,
)
from ember.expert_manifold.v6_reward_credit import (
    SuccessGuardProjectionSummary,
    SuccessRetentionCreditSummary,
    functional_success_lora_gradients,
    project_blind_program_direction,
    success_projection_is_finite,
    success_retention_is_finite,
)
from ember.lora import copy_task_lora_state_
from ember.pi05_source_checkpoint import DistributedContext, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import initialize_distributed
from ember.reward.protocol import reward_credit_environment_seed
from ember.reward.rollout import (
    RewardTrajectory,
    collect_randomized_reward_trajectories,
    successful_trajectory_batch,
)
from ember.writer.condition_update import (
    CounterfactualNullUpdateSummary,
    ProgramDeltaApplicationSummary,
    apply_program_residual_delta_,
    counterfactual_null_program_delta,
    program_residual_delta_application_evidence,
)
from ember.writer.functional import (
    TASK_LOGICAL_BATCH_POLICY_RNG_SCHEME,
    functional_lora_loss_gradient,
    task_logical_batch_policy_rng_seed,
)


@dataclass(frozen=True)
class TaskObjective:
    task: ExpertTask
    task_visit: int
    teacher_demo: int
    counterfactual_kind: str
    counterfactual_task: ExpertTask | None
    counterfactual_demo: int | None
    functional_loss: torch.Tensor
    correct_feature: torch.Tensor
    negative_feature: torch.Tensor
    program_cotangent: torch.Tensor
    source_program_cotangent: torch.Tensor
    retention_credit: SuccessRetentionCreditSummary
    guard_projection: SuccessGuardProjectionSummary
    retention_program_cotangents: tuple[torch.Tensor, ...] | None
    trajectory_rows: tuple[Mapping[str, Any], ...]
    rollout_seconds: float
    retention_seconds: float
    correct_raw_frames: int
    correct_sampled_frames: int
    negative_raw_frames: int
    negative_sampled_frames: int
    program_before: torch.Tensor | None = None
    correct_lora_before: Mapping[str, torch.Tensor] | None = None
    fixed_policy_query: Mapping[str, torch.Tensor] | None = None


_FIXED_ACTION_PROFILE_TASK_ORDINALS = frozenset((0, 6, 12, 18))
_LOGICAL_POLICY_BATCH_SIZE = 20


def _policy_rng_seed_for_logical_batch(
    config: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    task_id: int,
    task_visit: int,
) -> int:
    """Bind functional randomness to the complete task-local B20 query set."""

    randomness = config["objective"]["positive_policy_randomness"]
    demo_indices = batch.get("demo_index")
    frame_indices = batch.get("frame_index")
    if (
        randomness.get("seed_scheme") != TASK_LOGICAL_BATCH_POLICY_RNG_SCHEME
        or not isinstance(demo_indices, torch.Tensor)
        or not isinstance(frame_indices, torch.Tensor)
        or demo_indices.ndim != 1
        or frame_indices.shape != demo_indices.shape
        or demo_indices.numel() != _LOGICAL_POLICY_BATCH_SIZE
    ):
        raise ExpertManifoldError("OSG-PC action-query randomness changed")
    return task_logical_batch_policy_rng_seed(
        optimization_seed=int(config["optimization"]["seed"]),
        task_id=task_id,
        task_visit=task_visit,
        demo_indices=demo_indices.detach().cpu().tolist(),
        frame_indices=frame_indices.detach().cpu().tolist(),
    )


def _batch_task_id(batch: Mapping[str, Any]) -> int:
    values = batch.get("task_id")
    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise ExpertManifoldError("residual Writer action batch lost task identity")
    unique = values.unique()
    if unique.numel() != 1:
        raise ExpertManifoldError("residual Writer action batch crossed tasks")
    return int(unique.item())


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
        "failure_replay_retained": bool(
            not value.success
            and (
                value.observations
                or value.action_chunks
                or value.valid_action_steps
            )
        ),
    }


def _collect_task_replay(
    runtime: V6PriorRuntime,
    *,
    schedule_macro: int,
    task: ExpertTask,
    lora_state: Mapping[str, torch.Tensor],
) -> tuple[
    tuple[RewardTrajectory, ...],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    float,
]:
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
    environment = runtime.config["environment"]
    copy_task_lora_state_(runtime.policy, lora_state, runtime.lora_contract)
    started = time.monotonic()
    try:
        trajectories = collect_randomized_reward_trajectories(
            envs=tuple(
                runtime.env_pool.get(reward_task, lane=lane) for lane in range(4)
            ),
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
            retain_failure_replay=False,
        )
    finally:
        copy_task_lora_state_(
            runtime.policy, runtime.identity_state, runtime.lora_contract
        )
    seconds = time.monotonic() - started
    replay, episode_ids, successes, panel_rows, panel_chunks = (
        successful_trajectory_batch(trajectories, torch.device("cpu"))
    )
    return (
        trajectories,
        replay,
        episode_ids,
        successes,
        panel_rows,
        panel_chunks,
        seconds,
    )


def _task_objective(
    runtime: V6PriorRuntime,
    *,
    schedule_macro: int,
    microtask: int,
    batch: Mapping[str, Any],
) -> TaskObjective:
    task_id, task_visit = runtime.sampler.task_visit_for_step(schedule_macro, microtask)
    if _batch_task_id(batch) != task_id:
        raise ExpertManifoldError("residual Writer sampler and action batch disagree")
    task = runtime.task_by_global_id[task_id]
    excluded = runtime.sampler.action_demo_indices_for_task_visit(task_id, task_visit)
    teacher_demo = runtime.video_schedule.demos_for_task_visit(
        task_id, task_visit, excluded=excluded
    )[0]
    correct_video = runtime.video_store.load(task_id, teacher_demo)
    kind = counterfactual_kind(task.ordinal, task_visit)
    negative_task = None
    negative_demo = None
    negative_video = None
    if kind == "wrong":
        negative_task = cross_suite_wrong_task(
            runtime.tasks,
            task_ordinal=task.ordinal,
            task_visit=task_visit,
        )
        negative_demo = runtime.video_schedule.demos_for_task_visit(
            negative_task.global_task_id, task_visit
        )[0]
        negative_video = runtime.video_store.load(
            negative_task.global_task_id, negative_demo
        )
    copy_task_lora_state_(
        runtime.policy, runtime.identity_state, runtime.lora_contract
    )
    graph = generate_condition_graph(
        writer=runtime.writer,
        policy=runtime.policy,
        correct_video=correct_video,
        counterfactual_video=negative_video,
        language_tokens=runtime.language_tokens[task_id],
        kind=kind,
        counterfactual_seed=int(runtime.config["data"]["counterfactual_seed"]),
        task_ordinal=task.ordinal,
        task_visit=task_visit,
        teacher_demo=teacher_demo,
        device=runtime.context.device,
    )
    policy_rng_seed = _policy_rng_seed_for_logical_batch(
        runtime.config,
        batch,
        task_id=task_id,
        task_visit=task_visit,
    )
    randomness = runtime.config["objective"]["positive_policy_randomness"]
    policy_batch = runtime.processor.training_batch(dict(batch))
    with torch.autocast(
        device_type=runtime.context.device.type,
        dtype=torch.bfloat16,
        enabled=runtime.context.device.type == "cuda",
    ):
        functional_loss, _, lora_gradients = functional_lora_loss_gradient(
            runtime.policy,
            graph.correct_lora,
            runtime.lora_contract,
            batch=policy_batch,
            policy_rng_seed=policy_rng_seed,
            policy_rng_device=runtime.context.device,
            flow_time_sampling_scheme=str(randomness["flow_time_sampling_scheme"]),
            flow_noise_sampling_scheme=str(randomness["flow_noise_sampling_scheme"]),
            policy_microbatch_size=int(
                runtime.config["optimization"]["functional_policy_microbatch_size"]
            ),
            collect_policy_details=False,
        )
    retain_profile = runtime.args.mode == "mechanism-profile"
    correct_lora_before = (
        {name: value.detach() for name, value in graph.correct_lora.items()}
        if retain_profile
        else None
    )
    fixed_policy_query = None
    if retain_profile and task.ordinal in _FIXED_ACTION_PROFILE_TASK_ORDINALS:
        fixed_policy_query = {
            name: value[:1].detach()
            for name, value in policy_batch.items()
            if name.startswith("observation.")
        }
        if len(fixed_policy_query) != 4:
            raise ExpertManifoldError("fixed-action profile query changed")
    source_cotangent = program_cotangent(graph, lora_gradients)
    (
        trajectories,
        replay,
        episode_ids,
        successes,
        panel_rows,
        panel_chunks,
        rollout_seconds,
    ) = (
        _collect_task_replay(
            runtime,
            schedule_macro=schedule_macro,
            task=task,
            lora_state=graph.correct_lora,
        )
    )
    retention_started = time.monotonic()
    retention_gradients: tuple[dict[str, torch.Tensor], ...] = ()
    with torch.autocast(
        device_type=runtime.context.device.type,
        dtype=torch.bfloat16,
        enabled=runtime.context.device.type == "cuda",
    ):
        if bool(successes.any()):
            retention_graph = redecode_condition_graph(
                graph, runtime.writer, device=runtime.context.device
            )
            retention_gradients, retention_credit = (
                functional_success_lora_gradients(
                    runtime.policy,
                    retention_graph.correct_lora,
                    runtime.lora_contract,
                    replay,
                    episode_ids,
                    successes,
                    panel_row_indices=panel_rows,
                    panel_total_chunks=panel_chunks,
                    mc_samples=int(
                        runtime.config["objective"]["retention_flow_mc_samples"]
                    ),
                    physical_microbatch_size=int(
                        runtime.config["optimization"][
                            "retention_replay_microbatch_size"
                        ]
                    ),
                    flow_seed_root=int(
                        runtime.config["rng"]["retention_flow_seed_root"]
                    ),
                    cycle=schedule_macro,
                    global_task_id=task.global_task_id,
                    device=runtime.context.device,
                )
            )
            retention_cotangents = tuple(
                program_cotangent(
                    retention_graph,
                    gradients,
                    retain_graph=index + 1 < len(retention_gradients),
                )
                for index, gradients in enumerate(retention_gradients)
            )
        else:
            _, retention_credit = functional_success_lora_gradients(
                runtime.policy,
                graph.correct_lora,
                runtime.lora_contract,
                replay,
                episode_ids,
                successes,
                panel_row_indices=panel_rows,
                panel_total_chunks=panel_chunks,
                mc_samples=int(
                    runtime.config["objective"]["retention_flow_mc_samples"]
                ),
                physical_microbatch_size=int(
                    runtime.config["optimization"][
                        "retention_replay_microbatch_size"
                    ]
                ),
                flow_seed_root=int(
                    runtime.config["rng"]["retention_flow_seed_root"]
                ),
                cycle=schedule_macro,
                global_task_id=task.global_task_id,
                device=runtime.context.device,
            )
            retention_cotangents = ()
    program_rms = tuple(
        float(value.square().mean().sqrt().detach().cpu())
        for value in retention_cotangents
    )
    if len(program_rms) != retention_credit.successes or any(
        not math.isfinite(value) or value <= 0 for value in program_rms
    ):
        raise ExpertManifoldError(
            "OSG-PC successful replay did not reach the complete Program"
        )
    retention_credit = replace(
        retention_credit, program_cotangent_rms=program_rms
    )
    safe_direction, guard_projection = project_blind_program_direction(
        -source_cotangent, retention_cotangents
    )
    cotangent = (
        -safe_direction if guard_projection.changed else source_cotangent
    )
    retention_seconds = time.monotonic() - retention_started
    if not success_retention_is_finite(
        retention_credit
    ) or not success_projection_is_finite(guard_projection):
        raise ExpertManifoldError("OSG-PC task evidence became non-finite")
    return TaskObjective(
        task=task,
        task_visit=task_visit,
        teacher_demo=teacher_demo,
        counterfactual_kind=kind,
        counterfactual_task=negative_task,
        counterfactual_demo=negative_demo,
        functional_loss=functional_loss.detach(),
        correct_feature=graph.correct_feature.detach(),
        negative_feature=graph.negative_feature.detach(),
        program_cotangent=cotangent,
        source_program_cotangent=source_cotangent,
        retention_credit=retention_credit,
        guard_projection=guard_projection,
        retention_program_cotangents=(
            retention_cotangents if retain_profile else None
        ),
        trajectory_rows=tuple(_trajectory_record(value) for value in trajectories),
        rollout_seconds=rollout_seconds,
        retention_seconds=retention_seconds,
        correct_raw_frames=graph.correct_raw_frames,
        correct_sampled_frames=graph.correct_sampled_frames,
        negative_raw_frames=graph.negative_raw_frames,
        negative_sampled_frames=graph.negative_sampled_frames,
        program_before=(
            graph.program_input_before[0].detach() if retain_profile else None
        ),
        correct_lora_before=correct_lora_before,
        fixed_policy_query=fixed_policy_query,
    )


def _task_record(value: TaskObjective) -> dict[str, Any]:
    correct_norm = torch.linalg.vector_norm(value.correct_feature)
    negative_norm = torch.linalg.vector_norm(value.negative_feature)
    cosine = torch.dot(value.correct_feature, value.negative_feature) / (
        correct_norm * negative_norm
    ).clamp_min(torch.finfo(torch.float32).tiny)
    scalars = (
        torch.stack(
            (
                value.functional_loss.to(dtype=torch.float32),
                value.program_cotangent.square().mean().sqrt(),
                value.source_program_cotangent.square().mean().sqrt(),
                correct_norm,
                negative_norm,
                cosine,
            )
        )
        .detach()
        .cpu()
        .tolist()
    )
    return {
        "task_ordinal": value.task.ordinal,
        "global_task_id": value.task.global_task_id,
        "suite": value.task.suite,
        "task_id": value.task.task_id,
        "task_visit": value.task_visit,
        "teacher_demo": value.teacher_demo,
        "counterfactual_kind": value.counterfactual_kind,
        "counterfactual_global_task_id": (
            value.counterfactual_task.global_task_id
            if value.counterfactual_task is not None
            else None
        ),
        "counterfactual_demo": value.counterfactual_demo,
        "functional_loss": scalars[0],
        "program_cotangent_rms": scalars[1],
        "source_program_cotangent_rms": scalars[2],
        "correct_feature_norm": scalars[3],
        "negative_feature_norm": scalars[4],
        "correct_negative_feature_cosine": scalars[5],
        "retention_credit": asdict(value.retention_credit),
        "guard_projection": asdict(value.guard_projection),
        "trajectories": list(value.trajectory_rows),
        "rollout_seconds": value.rollout_seconds,
        "retention_seconds": value.retention_seconds,
        "correct_raw_frames": value.correct_raw_frames,
        "correct_sampled_frames": value.correct_sampled_frames,
        "negative_raw_frames": value.negative_raw_frames,
        "negative_sampled_frames": value.negative_sampled_frames,
        "source_action_queries": 20,
        "physical_correct_policy_forwards": 2,
        "retention_policy_forwards": (
            value.retention_credit.functional_policy_forwards
        ),
        "negative_policy_forwards": 0,
        "historical_v6_video_encodes": 1,
        "retention_factorhead_redecodes": int(
            value.retention_credit.successes > 0
        ),
        "policy_innovation_key_count": 2,
        "policy_innovation_unique_video_count": (
            2 if value.counterfactual_kind == "wrong" else 1
        ),
        "policy_innovation_duplicate_frame_forwards": 0,
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


def _gather_full48(
    local: Sequence[TaskObjective],
    context: DistributedContext,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expected_local = 24 // context.world_size
    if len(local) != expected_local:
        raise ExpertManifoldError("residual Writer local task coverage changed")
    payload = torch.empty(
        expected_local,
        1 + 2 * 256,
        dtype=torch.float32,
        device=context.device,
    )
    payload[:, 0] = torch.tensor(
        [value.task.ordinal for value in local],
        dtype=torch.float32,
        device=context.device,
    )
    payload[:, 1:257] = torch.stack([value.correct_feature for value in local])
    payload[:, 257:] = torch.stack([value.negative_feature for value in local])
    cotangents = torch.stack([value.program_cotangent for value in local])
    gathered_payload = _all_gather_fixed(payload, context)
    gathered_cotangents = _all_gather_fixed(cotangents, context)
    ordinals = gathered_payload[:, 0].to(dtype=torch.long)
    order = ordinals.argsort()
    sorted_ordinals = ordinals.index_select(0, order)
    if not torch.equal(
        sorted_ordinals,
        torch.arange(24, dtype=torch.long, device=context.device),
    ):
        raise ExpertManifoldError("residual Writer full48 task order changed")
    return (
        gathered_payload.index_select(0, order)[:, 1:257],
        gathered_payload.index_select(0, order)[:, 257:],
        gathered_cotangents.index_select(0, order),
    )


def _gather_task_records(
    local: list[dict[str, Any]],
    context: DistributedContext,
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
        raise ExpertManifoldError("residual Writer macro did not cover train24")
    return result


def _runtime_maximums(
    context: DistributedContext,
    started: float,
    input_wait_seconds: float,
) -> tuple[float, int, int, float]:
    torch.cuda.synchronize(context.device)
    values = torch.tensor(
        (
            time.monotonic() - started,
            torch.cuda.max_memory_allocated(context.device),
            torch.cuda.max_memory_reserved(context.device),
            input_wait_seconds,
        ),
        dtype=torch.float64,
        device=context.device,
    )
    if context.world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return float(values[0]), int(values[1]), int(values[2]), float(values[3])


def _collect_local_objectives(
    runtime: V6PriorRuntime,
    schedule_macro: int,
) -> tuple[list[TaskObjective], float]:
    local_objectives = []
    input_wait_seconds = 0.0
    for microtask in range(24 // runtime.context.world_size):
        input_started = time.monotonic()
        batch = next(runtime.iterator)
        input_wait_seconds += time.monotonic() - input_started
        local_objectives.append(
            _task_objective(
                runtime,
                schedule_macro=schedule_macro,
                microtask=microtask,
                batch=batch,
            )
        )
    if any(
        parameter.grad is not None for parameter in runtime.policy.parameters()
    ) or any(
        parameter.grad is not None
        for parameter in runtime.writer.base_writer.parameters()
    ):
        raise ExpertManifoldError("residual Writer touched frozen parameter gradients")
    return local_objectives, input_wait_seconds


def _apply_macro_update(
    runtime: V6PriorRuntime,
    local_objectives: Sequence[TaskObjective],
    *,
    profile: bool,
    step_started: float,
) -> tuple[
    CounterfactualNullUpdateSummary,
    ProgramDeltaApplicationSummary | None,
    dict[str, Any] | None,
    dict[str, float] | None,
    dict[str, Any] | None,
    float | None,
    float | None,
    float | None,
]:
    profile_task_seconds = None
    if profile:
        torch.cuda.synchronize(runtime.context.device)
        profile_task_seconds = _profile_max_seconds(
            runtime.context, time.monotonic() - step_started
        )
        torch.cuda.synchronize(runtime.context.device)
        kernel_started = time.monotonic()
    correct, negative, cotangents = _gather_full48(local_objectives, runtime.context)
    delta, update = counterfactual_null_program_delta(
        correct,
        negative,
        cotangents,
        step_size=float(runtime.config["update"]["step_size"]),
        relative_damping=float(runtime.config["update"]["relative_damping"]),
    )
    full_features = torch.cat((correct, negative), dim=0)
    apply_program_residual_delta_(runtime.writer.program_memory, delta)
    if not profile:
        return update, None, None, None, None, None, None, None
    torch.cuda.synchronize(runtime.context.device)
    kernel_seconds = _profile_max_seconds(
        runtime.context, time.monotonic() - kernel_started
    )
    torch.cuda.synchronize(runtime.context.device)
    verification_started = time.monotonic()
    # The mechanism profile is fresh-only, so its pre-write residual is exact
    # zero.  Allocate this verification-only tensor outside production timing.
    before = cotangents.new_zeros(
        full_features.shape[0], cotangents.shape[1], cotangents.shape[2]
    )
    full_motion = torch.matmul(
        full_features.to(dtype=torch.float32), delta.flatten(1)
    ).reshape_as(before)
    application = program_residual_delta_application_evidence(
        runtime.writer.program_memory,
        delta,
        full_features,
        before,
        predicted=full_motion,
    )
    task_local = _profile_task_local_motion(
        cotangents, full_motion, runtime.config["profile_run"]["gates"]
    )
    guard_application = _profile_success_guard_application(
        local_objectives, full_motion[:24], runtime.context
    )
    lora_response = _profile_lora_response(runtime, local_objectives, full_motion[:24])
    torch.cuda.synchronize(runtime.context.device)
    verification_seconds = _profile_max_seconds(
        runtime.context, time.monotonic() - verification_started
    )
    return (
        update,
        application,
        task_local,
        lora_response,
        guard_application,
        profile_task_seconds,
        kernel_seconds,
        verification_seconds,
    )


def _macro_record(
    *,
    macro: int,
    schedule_macro: int,
    records: Sequence[Mapping[str, Any]],
    update: CounterfactualNullUpdateSummary,
    application: ProgramDeltaApplicationSummary | None,
    task_local: Mapping[str, Any] | None,
    lora_response: Mapping[str, float] | None,
    guard_application: Mapping[str, Any] | None,
    profile_task_seconds: float | None,
    kernel_seconds: float | None,
    verification_seconds: float | None,
    runtime_metrics: tuple[float, int, int, float],
) -> dict[str, Any]:
    counterfactual_counts = {
        name: sum(row["counterfactual_kind"] == name for row in records)
        for name in ("reversed", "shuffled", "wrong")
    }
    if counterfactual_counts != {"reversed": 8, "shuffled": 8, "wrong": 8}:
        raise ExpertManifoldError("residual Writer negative schedule changed")
    success_counts = [
        int(value["retention_credit"]["successes"]) for value in records
    ]
    trajectory_rows = [
        trajectory
        for value in records
        for trajectory in value["trajectories"]
    ]
    suite_guarded = {
        suite: sum(
            value["suite"] == suite
            and int(value["retention_credit"]["successes"]) > 0
            for value in records
        )
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    }
    guard_evidence = {
        "rollouts": len(trajectory_rows),
        "success_episodes": sum(success_counts),
        "failure_episodes": 4 * len(records) - sum(success_counts),
        "guarded_tasks": sum(value > 0 for value in success_counts),
        "all_success_tasks": sum(value == 4 for value in success_counts),
        "all_failure_tasks": sum(value == 0 for value in success_counts),
        "guarded_tasks_per_suite": suite_guarded,
        "projection_changed_tasks": sum(
            bool(value["guard_projection"]["changed"]) for value in records
        ),
        "raw_feasible_tasks": sum(
            bool(value["guard_projection"]["raw_feasible"]) for value in records
        ),
        "successful_replay_chunks": sum(
            int(value["retention_credit"]["replay_chunks"]) for value in records
        ),
        "full_k4_flow_panel_chunks": sum(
            int(value["retention_credit"]["flow_panel_chunks"])
            for value in records
        ),
        "successful_executed_action_steps": sum(
            int(value["retention_credit"]["executed_action_steps"])
            for value in records
        ),
        "environment_action_steps": sum(
            int(value["steps"]) for value in trajectory_rows
        ),
        "failure_replay_gradient_episodes": sum(
            not bool(value["success"]) and int(value["replay_chunks"]) > 0
            for value in trajectory_rows
        ),
        "retention_policy_forwards": sum(
            int(value["retention_policy_forwards"]) for value in records
        ),
        "maximum_constraint_value": max(
            float(value["guard_projection"]["maximum_constraint_value"])
            for value in records
        ),
        "minimum_source_descent_ratio": min(
            float(value["guard_projection"]["source_descent_ratio"])
            for value in records
        ),
    }
    seconds, allocated, reserved, input_wait_seconds = runtime_metrics
    row = {
        "macro": macro + 1,
        "schedule_macro": schedule_macro,
        "functional_loss": sum(float(value["functional_loss"]) for value in records)
        / 24,
        "program_cotangent_rms": math.sqrt(
            sum(float(value["program_cotangent_rms"]) ** 2 for value in records) / 24
        ),
        "source_program_cotangent_rms": math.sqrt(
            sum(
                float(value["source_program_cotangent_rms"]) ** 2
                for value in records
            )
            / 24
        ),
        "success_guard": guard_evidence,
        "counterfactual_counts": counterfactual_counts,
        "update": asdict(update),
        "application": asdict(application) if application is not None else None,
        "task_local_motion": task_local,
        "lora_response": lora_response,
        "success_guard_application": guard_application,
        "task_records": list(records),
        "production_kernel_seconds": kernel_seconds,
        "profile_task_seconds": profile_task_seconds,
        "profile_verification_seconds": verification_seconds,
        "step_seconds": seconds,
        "input_wait_seconds": input_wait_seconds,
        "max_cuda_allocated_bytes": allocated,
        "max_cuda_reserved_bytes": reserved,
        "negative_policy_forwards": 0,
        "oom_count": 0,
        "nonfinite_count": 0,
    }
    if not all(
        math.isfinite(float(row[name]))
        for name in (
            "functional_loss",
            "program_cotangent_rms",
            "source_program_cotangent_rms",
            "step_seconds",
        )
    ):
        raise ExpertManifoldError("residual Writer metric became non-finite")
    return row


def _run_one_macro(
    runtime: V6PriorRuntime,
    *,
    macro: int,
) -> dict[str, Any]:
    profile = runtime.args.mode == "mechanism-profile"
    schedule_macro = runtime.segment.schedule_origin + macro
    versions_before = _base_versions(runtime) if profile else ()
    step_started = time.monotonic()
    local, input_wait = _collect_local_objectives(runtime, schedule_macro)
    update_evidence = _apply_macro_update(
        runtime, local, profile=profile, step_started=step_started
    )
    records = _gather_task_records(
        [_task_record(value) for value in local], runtime.context
    )
    runtime_metrics = _runtime_maximums(runtime.context, step_started, input_wait)
    if profile and _base_versions(runtime) != versions_before:
        raise ExpertManifoldError("historical v6 state changed during profile")
    return _macro_record(
        macro=macro,
        schedule_macro=schedule_macro,
        records=records,
        update=update_evidence[0],
        application=update_evidence[1],
        task_local=update_evidence[2],
        lora_response=update_evidence[3],
        guard_application=update_evidence[4],
        profile_task_seconds=update_evidence[5],
        kernel_seconds=update_evidence[6],
        verification_seconds=update_evidence[7],
        runtime_metrics=runtime_metrics,
    )


def _run_mechanism_profile(runtime: V6PriorRuntime) -> None:
    row = _run_one_macro(runtime, macro=0)
    passed, gate_evidence = _profile_passes(runtime.config, row)
    result = {
        "schema_version": V6_PRIOR_PROFILE_SCHEMA,
        "passed": passed,
        "schedule_macro": runtime.segment.schedule_origin,
        "retain_weight": False,
        "gates": dict(runtime.config["profile_run"]["gates"]),
        "gate_evidence": gate_evidence,
        "macro": row,
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
                context=runtime.context,
                metrics_rows=cursor,
                cursor_contract=cursor_contract(runtime.config, cursor),
                checkpoint_contract=runtime.checkpoint_contract,
            )
    if runtime.context.is_main:
        write_json_atomic(
            runtime.args.output_dir / "completion.json",
            {
                "schema_version": V6_PRIOR_COMPLETION_SCHEMA,
                "mode": "formal",
                "completed_macro": runtime.segment.stop_macro,
                "metrics_rows": runtime.segment.stop_macro,
                "content_hash_policy": "disabled_by_owner",
            },
        )


def train(args: argparse.Namespace) -> None:
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    runtime: V6PriorRuntime | None = None
    try:
        runtime = _prepare_runtime(args, context)
        if context.is_main:
            print(
                json.dumps(
                    {
                        "event": "start",
                        "mode": args.mode,
                        "start_macro": runtime.segment.start_macro,
                        "stop_macro": runtime.segment.stop_macro,
                        "historical_v6_frozen_parameters": (
                            runtime.ownership.frozen_parameter_count
                        ),
                        "residual_memory_values": (
                            runtime.writer.program_memory.value.numel()
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.mode == "mechanism-profile":
            _run_mechanism_profile(runtime)
        else:
            _run_training(runtime)
    finally:
        if runtime is not None:
            runtime.env_pool.close()
            runtime.dataset.close()
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
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in ("config", "source_run", "checkpoint", "tokenizer_path", "data_root"):
        path = getattr(args, name).resolve()
        if not path.exists():
            raise ExpertManifoldError(f"missing residual Writer path: {path}")
        setattr(args, name, path)
    args.output_dir = args.output_dir.resolve()
    args.resume = args.resume.resolve() if args.resume else None
    if args.resume is None:
        if args.output_dir.exists() and (
            not args.output_dir.is_dir() or any(args.output_dir.iterdir())
        ):
            raise ExpertManifoldError("fresh residual Writer output is not empty")
    elif (
        not args.resume.is_dir()
        or args.resume.parent.name != "checkpoints"
        or args.resume.parent.parent.resolve() != args.output_dir
        or not (args.output_dir / "run_contract.json").is_file()
    ):
        raise ExpertManifoldError("residual Writer resume output ownership changed")
    if args.config != V6_PRIOR_CANONICAL_CONFIG.resolve():
        raise ExpertManifoldError("residual Writer requires the canonical config")
    load_v6_prior_config(args.config)
    if args.num_workers < 0 or (
        args.stop_after_macro is not None and args.stop_after_macro <= 0
    ):
        raise ExpertManifoldError("invalid residual Writer worker or stop boundary")
    return args
