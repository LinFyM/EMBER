"""Paired-video joint functional credit for the frozen-v6 residual Writer."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import default_collate

from ember.expert_manifold.contract import ExpertManifoldError, ExpertTask
from ember.expert_manifold.v6_prior import counterfactual_kind, cross_suite_wrong_task
from ember.expert_manifold.v6_prior_checkpoint import save_v6_prior_checkpoint
from ember.expert_manifold.v6_prior_contract import (
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
    profile_task_local_motion as _profile_task_local_motion,
)
from ember.expert_manifold.v6_prior_runtime import V6PriorRuntime, _prepare_runtime
from ember.expert_manifold.v6_prior_run_contract import cursor_contract
from ember.expert_manifold.v6_prior_step import (
    GeneratedViewGraph,
    generate_view_graph,
    program_cotangent,
)
from ember.lora import copy_task_lora_state_
from ember.pi05_source_checkpoint import DistributedContext, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import initialize_distributed
from ember.writer.condition_update import (
    PairedVideoJointUpdateSummary,
    ProgramDeltaApplicationSummary,
    apply_program_residual_delta_,
    paired_video_joint_program_delta,
    program_residual_delta_application_evidence,
)
from ember.writer.data import RawTeacherVideo
from ember.writer.functional import (
    TASK_LOGICAL_BATCH_POLICY_RNG_SCHEME,
    functional_lora_loss_gradient,
    task_logical_batch_policy_rng_seed,
)


@dataclass(frozen=True)
class ViewObjective:
    demo: int
    counterfactual_demo: int | None
    functional_loss: torch.Tensor
    correct_feature: torch.Tensor
    negative_feature: torch.Tensor
    program_cotangent: torch.Tensor
    correct_raw_frames: int
    correct_sampled_frames: int
    negative_raw_frames: int
    negative_sampled_frames: int
    profile_graph: GeneratedViewGraph | None = None


@dataclass(frozen=True)
class TaskObjective:
    task: ExpertTask
    task_visit: int
    action_query_demos: tuple[int, ...]
    counterfactual_kind: str
    counterfactual_task: ExpertTask | None
    primary: ViewObjective
    companion: ViewObjective
    fixed_policy_query: Mapping[str, torch.Tensor] | None = None
    phase_a_queue_index: int = -1
    phase_a_rank: int = -1
    phase_a_started_seconds: float = 0.0
    phase_a_finished_seconds: float = 0.0
    phase_a_batch_load_seconds: float = 0.0
    phase_a_claim_seconds: float = 0.0


@dataclass(frozen=True)
class MacroUpdateEvidence:
    update: PairedVideoJointUpdateSummary
    application: ProgramDeltaApplicationSummary | None
    task_local: Mapping[str, Any] | None
    lora_response: Mapping[str, Any] | None
    phase_a_seconds: float
    phase_a_rows: tuple[Mapping[str, Any], ...]
    profile_task_seconds: float | None
    kernel_seconds: float | None
    verification_seconds: float | None


_LOGICAL_POLICY_BATCH_SIZE = 20
_TRAIN_TASK_COUNT = 24
_WORK_QUEUE_MINIMUM_RETAINED_TASK_CAP = 8
_PROFILE_TASK_ORDINALS = frozenset((0, 6, 12, 18))


class _AtomicTaskClaimQueue:
    """Claim deterministic task jobs across same-host torchrun ranks."""

    def __init__(self, path: Path, jobs: Sequence[tuple[int, int]]) -> None:
        self.path = path
        self.jobs = tuple(jobs)

    def claim(self) -> tuple[int, tuple[int, int] | None, float]:
        started = time.monotonic()
        with self.path.open("r+", encoding="ascii") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            index = int(handle.read().strip())
            if index < len(self.jobs):
                handle.seek(0)
                handle.write(str(index + 1))
                handle.truncate()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return index, self.jobs[index] if index < len(self.jobs) else None, (
            time.monotonic() - started
        )


def _retained_task_cap(world_size: int) -> int:
    return max(
        _WORK_QUEUE_MINIMUM_RETAINED_TASK_CAP,
        math.ceil(_TRAIN_TASK_COUNT / world_size),
    )


def _policy_rng_seed_for_logical_batch(
    config: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    task_id: int,
    task_visit: int,
) -> int:
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
        raise ExpertManifoldError("MGCI-JC action-query randomness changed")
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
        raise ExpertManifoldError("MGCI-JC action batch lost task identity")
    unique = values.unique()
    if unique.numel() != 1:
        raise ExpertManifoldError("MGCI-JC action batch crossed tasks")
    return int(unique.item())


def _detach_profile_graph(graph: GeneratedViewGraph) -> GeneratedViewGraph:
    return replace(
        graph,
        correct_lora={name: value.detach() for name, value in graph.correct_lora.items()},
        program_leaf=graph.program_leaf.detach(),
    )


def _view_objective(
    runtime: V6PriorRuntime,
    *,
    task: ExpertTask,
    task_visit: int,
    demo: int,
    correct_video: RawTeacherVideo,
    kind: str,
    negative_demo: int | None,
    negative_video: RawTeacherVideo | None,
    policy_batch: Mapping[str, Any],
    policy_rng_seed: int,
) -> ViewObjective:
    graph = generate_view_graph(
        writer=runtime.writer,
        policy=runtime.policy,
        correct_video=correct_video,
        counterfactual_video=negative_video,
        language_tokens=runtime.language_tokens[task.global_task_id],
        kind=kind,
        counterfactual_seed=int(runtime.config["data"]["counterfactual_seed"]),
        task_ordinal=task.ordinal,
        task_visit=task_visit,
        teacher_demo=demo,
        device=runtime.context.device,
    )
    randomness = runtime.config["objective"]["positive_policy_randomness"]
    with torch.autocast(
        device_type=runtime.context.device.type,
        dtype=torch.bfloat16,
        enabled=runtime.context.device.type == "cuda",
    ):
        loss, _, lora_gradients = functional_lora_loss_gradient(
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
    cotangent = program_cotangent(graph, lora_gradients)
    retain_graph = (
        runtime.args.mode == "mechanism-profile"
        and task.ordinal in _PROFILE_TASK_ORDINALS
    )
    result = ViewObjective(
        demo=demo,
        counterfactual_demo=negative_demo,
        functional_loss=loss.detach(),
        correct_feature=graph.correct_feature.detach(),
        negative_feature=graph.negative_feature.detach(),
        program_cotangent=cotangent,
        correct_raw_frames=graph.correct_raw_frames,
        correct_sampled_frames=graph.correct_sampled_frames,
        negative_raw_frames=graph.negative_raw_frames,
        negative_sampled_frames=graph.negative_sampled_frames,
        profile_graph=_detach_profile_graph(graph) if retain_graph else None,
    )
    del graph, lora_gradients
    return result


def _task_objective(
    runtime: V6PriorRuntime,
    *,
    task_id: int,
    task_visit: int,
    batch: Mapping[str, Any],
) -> TaskObjective:
    if _batch_task_id(batch) != task_id:
        raise ExpertManifoldError("MGCI-JC sampler and action batch disagree")
    task = runtime.task_by_global_id[task_id]
    excluded = runtime.sampler.action_demo_indices_for_task_visit(task_id, task_visit)
    primary_demo = runtime.video_schedule.demos_for_task_visit(
        task_id, task_visit, excluded=excluded
    )[0]
    companion_demo = runtime.video_schedule.companion_demos_for_task_visit(
        task_id, task_visit, excluded=excluded
    )[0]
    if primary_demo == companion_demo or companion_demo in excluded:
        raise ExpertManifoldError("MGCI-JC paired videos lost cross-episode separation")
    kind = counterfactual_kind(task.ordinal, task_visit)
    negative_task = None
    negative_demos: tuple[int | None, int | None] = (None, None)
    negative_videos: tuple[RawTeacherVideo | None, RawTeacherVideo | None] = (
        None,
        None,
    )
    if kind == "wrong":
        negative_task = cross_suite_wrong_task(
            runtime.tasks, task_ordinal=task.ordinal, task_visit=task_visit
        )
        wrong_id = negative_task.global_task_id
        wrong_primary = runtime.video_schedule.demos_for_task_visit(
            wrong_id, task_visit
        )[0]
        wrong_companion = runtime.video_schedule.companion_demos_for_task_visit(
            wrong_id, task_visit
        )[0]
        if wrong_primary == wrong_companion:
            raise ExpertManifoldError("MGCI-JC wrong-video pair collapsed")
        negative_demos = (wrong_primary, wrong_companion)
        negative_videos = (
            runtime.video_store.load(wrong_id, wrong_primary),
            runtime.video_store.load(wrong_id, wrong_companion),
        )
    policy_rng_seed = _policy_rng_seed_for_logical_batch(
        runtime.config,
        batch,
        task_id=task_id,
        task_visit=task_visit,
    )
    policy_batch = runtime.processor.training_batch(dict(batch))
    copy_task_lora_state_(runtime.policy, runtime.identity_state, runtime.lora_contract)
    primary = _view_objective(
        runtime,
        task=task,
        task_visit=task_visit,
        demo=primary_demo,
        correct_video=runtime.video_store.load(task_id, primary_demo),
        kind=kind,
        negative_demo=negative_demos[0],
        negative_video=negative_videos[0],
        policy_batch=policy_batch,
        policy_rng_seed=policy_rng_seed,
    )
    companion = _view_objective(
        runtime,
        task=task,
        task_visit=task_visit,
        demo=companion_demo,
        correct_video=runtime.video_store.load(task_id, companion_demo),
        kind=kind,
        negative_demo=negative_demos[1],
        negative_video=negative_videos[1],
        policy_batch=policy_batch,
        policy_rng_seed=policy_rng_seed,
    )
    fixed_query = None
    if runtime.args.mode == "mechanism-profile" and task.ordinal in _PROFILE_TASK_ORDINALS:
        fixed_query = {
            name: value[:1].detach()
            for name, value in policy_batch.items()
            if name.startswith("observation.")
        }
        if len(fixed_query) != 4:
            raise ExpertManifoldError("MGCI-JC fixed-action profile query changed")
    del policy_batch
    return TaskObjective(
        task=task,
        task_visit=task_visit,
        action_query_demos=tuple(excluded),
        counterfactual_kind=kind,
        counterfactual_task=negative_task,
        primary=primary,
        companion=companion,
        fixed_policy_query=fixed_query,
    )


def _view_record(view: ViewObjective) -> dict[str, Any]:
    correct_norm = torch.linalg.vector_norm(view.correct_feature)
    negative_norm = torch.linalg.vector_norm(view.negative_feature)
    cosine = torch.dot(view.correct_feature, view.negative_feature) / (
        correct_norm * negative_norm
    ).clamp_min(torch.finfo(torch.float32).tiny)
    scalars = (
        torch.stack(
            (
                view.functional_loss.to(dtype=torch.float32),
                view.program_cotangent.square().mean().sqrt(),
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
        "demo": view.demo,
        "counterfactual_demo": view.counterfactual_demo,
        "functional_loss": scalars[0],
        "program_cotangent_rms": scalars[1],
        "correct_feature_norm": scalars[2],
        "negative_feature_norm": scalars[3],
        "correct_negative_feature_cosine": scalars[4],
        "correct_raw_frames": view.correct_raw_frames,
        "correct_sampled_frames": view.correct_sampled_frames,
        "negative_raw_frames": view.negative_raw_frames,
        "negative_sampled_frames": view.negative_sampled_frames,
    }


def _task_record(value: TaskObjective) -> dict[str, Any]:
    primary = _view_record(value.primary)
    companion = _view_record(value.companion)
    return {
        "task_ordinal": value.task.ordinal,
        "global_task_id": value.task.global_task_id,
        "suite": value.task.suite,
        "task_id": value.task.task_id,
        "task_visit": value.task_visit,
        "action_query_demos": list(value.action_query_demos),
        "counterfactual_kind": value.counterfactual_kind,
        "counterfactual_global_task_id": (
            value.counterfactual_task.global_task_id
            if value.counterfactual_task is not None
            else None
        ),
        "primary": primary,
        "companion": companion,
        "functional_loss": 0.5
        * (float(primary["functional_loss"]) + float(companion["functional_loss"])),
        "program_cotangent_rms": math.sqrt(
            0.5
            * (
                float(primary["program_cotangent_rms"]) ** 2
                + float(companion["program_cotangent_rms"]) ** 2
            )
        ),
        "phase_a_queue_index": value.phase_a_queue_index,
        "phase_a_rank": value.phase_a_rank,
        "phase_a_started_seconds": value.phase_a_started_seconds,
        "phase_a_finished_seconds": value.phase_a_finished_seconds,
        "phase_a_batch_load_seconds": value.phase_a_batch_load_seconds,
        "phase_a_claim_seconds": value.phase_a_claim_seconds,
        "distinct_ordered_correct_videos": 2,
        "distinct_wrong_videos": 2 if value.counterfactual_kind == "wrong" else 0,
        "unique_source_action_queries": 20,
        "logical_source_action_queries": 40,
        "physical_correct_policy_forwards": 4,
        "historical_v6_video_encodes": 2,
        "condition_correct_rows": 2,
        "condition_negative_rows": 2,
        "negative_policy_forwards": 0,
        "outcome_rollouts": 0,
        "reward_reads": 0,
        "teacher_action_reads": 0,
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


def _gather_paired_video_rows(
    local: Sequence[TaskObjective], context: DistributedContext
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[Mapping[str, Any], ...],
]:
    maximum_local = _retained_task_cap(context.world_size)
    if not 0 <= len(local) <= maximum_local:
        raise ExpertManifoldError("MGCI-JC local task coverage changed")
    payload = torch.zeros(
        maximum_local, 7 + 4 * 256, dtype=torch.float32, device=context.device
    )
    payload[:, 0].fill_(-1)
    if local:
        payload[: len(local), :7] = torch.tensor(
            [
                (
                    value.task.ordinal,
                    value.phase_a_rank,
                    value.phase_a_queue_index,
                    value.phase_a_started_seconds,
                    value.phase_a_finished_seconds,
                    value.phase_a_batch_load_seconds,
                    value.phase_a_claim_seconds,
                )
                for value in local
            ],
            dtype=torch.float32,
            device=context.device,
        )
        payload[: len(local), 7:263] = torch.stack(
            [value.primary.correct_feature for value in local]
        )
        payload[: len(local), 263:519] = torch.stack(
            [value.primary.negative_feature for value in local]
        )
        payload[: len(local), 519:775] = torch.stack(
            [value.companion.correct_feature for value in local]
        )
        payload[: len(local), 775:] = torch.stack(
            [value.companion.negative_feature for value in local]
        )
    cotangents = torch.zeros(
        maximum_local, 2, 320, 256, dtype=torch.float32, device=context.device
    )
    if local:
        cotangents[: len(local), 0] = torch.stack(
            [value.primary.program_cotangent for value in local]
        )
        cotangents[: len(local), 1] = torch.stack(
            [value.companion.program_cotangent for value in local]
        )
    gathered_payload = _all_gather_fixed(payload, context)
    gathered_cotangents = _all_gather_fixed(cotangents, context)
    present = gathered_payload[:, 0] >= 0
    gathered_payload = gathered_payload[present]
    gathered_cotangents = gathered_cotangents[present]
    if gathered_payload.shape[0] != _TRAIN_TASK_COUNT:
        raise ExpertManifoldError("MGCI-JC padded gather lost train24")
    ordinals = gathered_payload[:, 0].to(dtype=torch.long)
    order = ordinals.argsort()
    if not torch.equal(
        ordinals.index_select(0, order),
        torch.arange(_TRAIN_TASK_COUNT, dtype=torch.long, device=context.device),
    ):
        raise ExpertManifoldError("MGCI-JC full96 task order changed")
    rows = gathered_payload.index_select(0, order)
    gradients = gathered_cotangents.index_select(0, order)
    timing = tuple(
        {
            "task_ordinal": int(row[0]),
            "rank": int(row[1]),
            "queue_index": int(row[2]),
            "started_seconds": float(row[3]),
            "finished_seconds": float(row[4]),
            "batch_load_seconds": float(row[5]),
            "claim_seconds": float(row[6]),
        }
        for row in rows.detach().cpu().tolist()
    )
    correct = torch.cat((rows[:, 7:263], rows[:, 519:775]), dim=0)
    negative = torch.cat((rows[:, 263:519], rows[:, 775:]), dim=0)
    cotangent = torch.cat((gradients[:, 0], gradients[:, 1]), dim=0)
    return correct, negative, cotangent, timing


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
        raise ExpertManifoldError("MGCI-JC macro did not cover train24")
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
    *,
    step_started: float,
) -> tuple[list[TaskObjective], float]:
    jobs = runtime.sampler.task_queue_for_step(schedule_macro)
    if len(jobs) != _TRAIN_TASK_COUNT:
        raise ExpertManifoldError("MGCI-JC lost train24 jobs")
    queue_path = (
        Path("/tmp")
        / f"ember-mgci-jc-{os.getuid()}"
        / runtime.args.output_dir.name
        / f"macro_{schedule_macro:08d}.cursor"
    )
    if runtime.context.is_main:
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        queue_path.write_text("0", encoding="ascii")
    if runtime.context.world_size > 1:
        dist.barrier(device_ids=[runtime.context.local_rank])
    queue = _AtomicTaskClaimQueue(queue_path, jobs)
    local: list[TaskObjective] = []
    input_wait_seconds = 0.0
    cap = _retained_task_cap(runtime.context.world_size)
    while len(local) < cap:
        queue_index, job, claim_seconds = queue.claim()
        if job is None:
            break
        task_id, task_visit = job
        task_started = time.monotonic()
        input_started = time.monotonic()
        indices = runtime.sampler.batch_indices_for_task_visit(
            schedule_macro, task_id, task_visit
        )
        batch = default_collate([runtime.dataset[index] for index in indices])
        if runtime.context.device.type == "cuda":
            batch = {
                name: value.pin_memory() if isinstance(value, torch.Tensor) else value
                for name, value in batch.items()
            }
        batch_load_seconds = time.monotonic() - input_started
        input_wait_seconds += batch_load_seconds
        objective = _task_objective(
            runtime, task_id=task_id, task_visit=task_visit, batch=batch
        )
        if runtime.context.device.type == "cuda":
            torch.cuda.synchronize(runtime.context.device)
        local.append(
            replace(
                objective,
                phase_a_queue_index=queue_index,
                phase_a_rank=runtime.context.rank,
                phase_a_started_seconds=task_started - step_started,
                phase_a_finished_seconds=time.monotonic() - step_started,
                phase_a_batch_load_seconds=batch_load_seconds,
                phase_a_claim_seconds=claim_seconds,
            )
        )
    if any(parameter.grad is not None for parameter in runtime.policy.parameters()) or any(
        parameter.grad is not None for parameter in runtime.writer.base_writer.parameters()
    ):
        raise ExpertManifoldError("MGCI-JC touched frozen parameter gradients")
    return local, input_wait_seconds


def _apply_macro_update(
    runtime: V6PriorRuntime,
    local_objectives: Sequence[TaskObjective],
    *,
    schedule_macro: int,
    profile: bool,
    step_started: float,
) -> MacroUpdateEvidence:
    correct, negative, cotangents, phase_a_rows = _gather_paired_video_rows(
        local_objectives, runtime.context
    )
    phase_a_seconds = _profile_max_seconds(
        runtime.context, time.monotonic() - step_started
    )
    profile_task_seconds = phase_a_seconds if profile else None
    if profile:
        torch.cuda.synchronize(runtime.context.device)
        kernel_started = time.monotonic()
    weights = torch.full(
        (2 * _TRAIN_TASK_COUNT,),
        0.5,
        dtype=torch.float32,
        device=runtime.context.device,
    )
    delta, update, full_motion = paired_video_joint_program_delta(
        correct,
        negative,
        cotangents,
        task_count=_TRAIN_TASK_COUNT,
        view_weights=weights,
        step_size=float(runtime.config["update"]["step_size"]),
        relative_damping=float(runtime.config["update"]["relative_damping"]),
    )
    apply_program_residual_delta_(runtime.writer.program_memory, delta)
    if not profile:
        return MacroUpdateEvidence(
            update=update,
            application=None,
            task_local=None,
            lora_response=None,
            phase_a_seconds=phase_a_seconds,
            phase_a_rows=phase_a_rows,
            profile_task_seconds=None,
            kernel_seconds=None,
            verification_seconds=None,
        )
    torch.cuda.synchronize(runtime.context.device)
    kernel_seconds = _profile_max_seconds(
        runtime.context, time.monotonic() - kernel_started
    )
    torch.cuda.synchronize(runtime.context.device)
    verification_started = time.monotonic()
    full_features = torch.cat((correct, negative), dim=0)
    before = cotangents.new_zeros(
        full_features.shape[0], cotangents.shape[1], cotangents.shape[2]
    )
    application = program_residual_delta_application_evidence(
        runtime.writer.program_memory,
        delta,
        full_features,
        before,
        predicted=full_motion,
    )
    task_local = _profile_task_local_motion(
        cotangents,
        full_motion,
        schedule_macro,
        runtime.config["profile_run"]["gates"],
    )
    lora_response = _profile_lora_response(
        runtime, local_objectives, full_motion[: 2 * _TRAIN_TASK_COUNT]
    )
    torch.cuda.synchronize(runtime.context.device)
    verification_seconds = _profile_max_seconds(
        runtime.context, time.monotonic() - verification_started
    )
    return MacroUpdateEvidence(
        update=update,
        application=application,
        task_local=task_local,
        lora_response=lora_response,
        phase_a_seconds=phase_a_seconds,
        phase_a_rows=phase_a_rows,
        profile_task_seconds=profile_task_seconds,
        kernel_seconds=kernel_seconds,
        verification_seconds=verification_seconds,
    )


def _macro_record(
    *,
    macro: int,
    schedule_macro: int,
    records: Sequence[Mapping[str, Any]],
    evidence: MacroUpdateEvidence,
    runtime_metrics: tuple[float, int, int, float],
) -> dict[str, Any]:
    counterfactual_counts = {
        name: sum(row["counterfactual_kind"] == name for row in records)
        for name in ("reversed", "shuffled", "wrong")
    }
    if counterfactual_counts != {"reversed": 8, "shuffled": 8, "wrong": 8}:
        raise ExpertManifoldError("MGCI-JC negative schedule changed")
    seconds, allocated, reserved, input_wait_seconds = runtime_metrics
    row = {
        "macro": macro + 1,
        "schedule_macro": schedule_macro,
        "functional_loss": sum(float(value["functional_loss"]) for value in records)
        / _TRAIN_TASK_COUNT,
        "program_cotangent_rms": math.sqrt(
            sum(float(value["program_cotangent_rms"]) ** 2 for value in records)
            / _TRAIN_TASK_COUNT
        ),
        "counterfactual_counts": counterfactual_counts,
        "correct_condition_rows": 48,
        "negative_condition_rows": 48,
        "logical_source_action_queries": 960,
        "outcome_rollouts": 0,
        "update": asdict(evidence.update),
        "application": asdict(evidence.application) if evidence.application else None,
        "task_local_motion": evidence.task_local,
        "lora_response": evidence.lora_response,
        "task_records": list(records),
        "production_kernel_seconds": evidence.kernel_seconds,
        "profile_task_seconds": evidence.profile_task_seconds,
        "profile_verification_seconds": evidence.verification_seconds,
        "phase_a_seconds": evidence.phase_a_seconds,
        "phase_a_task_rows": list(evidence.phase_a_rows),
        "queue_claim_seconds": sum(
            float(value["claim_seconds"]) for value in evidence.phase_a_rows
        ),
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
        for name in ("functional_loss", "program_cotangent_rms", "step_seconds")
    ):
        raise ExpertManifoldError("MGCI-JC metric became non-finite")
    return row


def _run_one_macro(runtime: V6PriorRuntime, *, macro: int) -> dict[str, Any]:
    profile = runtime.args.mode == "mechanism-profile"
    schedule_macro = runtime.segment.schedule_origin + macro
    versions_before = _base_versions(runtime) if profile else ()
    step_started = time.monotonic()
    local, input_wait = _collect_local_objectives(
        runtime, schedule_macro, step_started=step_started
    )
    evidence = _apply_macro_update(
        runtime,
        local,
        schedule_macro=schedule_macro,
        profile=profile,
        step_started=step_started,
    )
    records = _gather_task_records(
        [_task_record(value) for value in local], runtime.context
    )
    runtime_metrics = _runtime_maximums(runtime.context, step_started, input_wait)
    if profile and _base_versions(runtime) != versions_before:
        raise ExpertManifoldError("historical v6 state changed during MGCI-JC profile")
    row = _macro_record(
        macro=macro,
        schedule_macro=schedule_macro,
        records=records,
        evidence=evidence,
        runtime_metrics=runtime_metrics,
    )
    task_counts = [
        sum(int(value["rank"]) == rank for value in evidence.phase_a_rows)
        for rank in range(runtime.context.world_size)
    ]
    row["world_size"] = runtime.context.world_size
    row["task_counts_per_rank"] = task_counts
    row["maximum_tasks_per_rank"] = max(task_counts)
    return row


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
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in ("config", "source_run", "checkpoint", "tokenizer_path", "data_root"):
        path = getattr(args, name).resolve()
        if not path.exists():
            raise ExpertManifoldError(f"missing MGCI-JC Writer path: {path}")
        setattr(args, name, path)
    args.output_dir = args.output_dir.resolve()
    args.resume = args.resume.resolve() if args.resume else None
    if args.resume is None:
        if args.output_dir.exists() and (
            not args.output_dir.is_dir() or any(args.output_dir.iterdir())
        ):
            raise ExpertManifoldError("fresh MGCI-JC Writer output is not empty")
    elif (
        not args.resume.is_dir()
        or args.resume.parent.name != "checkpoints"
        or args.resume.parent.parent.resolve() != args.output_dir
        or not (args.output_dir / "run_contract.json").is_file()
    ):
        raise ExpertManifoldError("MGCI-JC Writer resume output ownership changed")
    if args.config != V6_PRIOR_CANONICAL_CONFIG.resolve():
        raise ExpertManifoldError("MGCI-JC Writer requires the canonical config")
    load_v6_prior_config(args.config)
    if args.num_workers < 0 or (
        args.stop_after_macro is not None and args.stop_after_macro <= 0
    ):
        raise ExpertManifoldError("invalid MGCI-JC Writer worker or stop boundary")
    return args
