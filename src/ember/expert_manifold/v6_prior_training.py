"""CVEG blind acquisition, paired protection, and response-preserving write."""

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
from ember.expert_manifold.v6_candidate_guard import (
    CandidateGuardProjectionSummary,
    PairedCandidateClassification,
    PairedTaskEvidence,
    classify_paired_candidate_outcomes,
    collect_paired_task_evidence,
    response_preserving_candidate_guard_correction,
)
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
    profile_success_key_application as _profile_success_key_application,
    profile_task_local_motion as _profile_task_local_motion,
)
from ember.expert_manifold.v6_prior_runtime import V6PriorRuntime, _prepare_runtime
from ember.expert_manifold.v6_prior_run_contract import cursor_contract
from ember.expert_manifold.v6_prior_step import (
    GeneratedConditionGraph,
    generate_condition_graph,
    program_cotangent,
)
from ember.expert_manifold.v6_success_key import (
    PersistedSuccessKeyPlan,
    SuccessKeyBankUpdateSummary,
)
from ember.lora import copy_task_lora_state_
from ember.pi05_source_checkpoint import DistributedContext, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import initialize_distributed
from ember.writer.condition_update import (
    ProgramDeltaApplicationSummary,
    SuccessKeyNullspaceUpdateSummary,
    apply_program_residual_delta_,
    program_residual_delta_application_evidence,
    success_key_constraint_motion,
    success_key_nullspace_program_delta,
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
    companion_demo: int
    action_query_demos: tuple[int, ...]
    counterfactual_kind: str
    counterfactual_task: ExpertTask | None
    counterfactual_demo: int | None
    functional_loss: torch.Tensor
    correct_feature: torch.Tensor
    negative_feature: torch.Tensor
    companion_feature: torch.Tensor
    program_cotangent: torch.Tensor
    graph: GeneratedConditionGraph
    correct_raw_frames: int
    correct_sampled_frames: int
    negative_raw_frames: int
    negative_sampled_frames: int
    companion_raw_frames: int
    companion_sampled_frames: int
    phase_a_queue_index: int = -1
    phase_a_rank: int = -1
    phase_a_started_seconds: float = 0.0
    phase_a_finished_seconds: float = 0.0
    phase_a_batch_load_seconds: float = 0.0
    phase_a_claim_seconds: float = 0.0
    fixed_policy_query: Mapping[str, torch.Tensor] | None = None
    paired: PairedTaskEvidence | None = None


@dataclass(frozen=True)
class MacroUpdateEvidence:
    local_objectives: tuple[TaskObjective, ...]
    blind_update: SuccessKeyNullspaceUpdateSummary
    classification: PairedCandidateClassification
    guard_projection: CandidateGuardProjectionSummary
    bank_update: SuccessKeyBankUpdateSummary
    application: ProgramDeltaApplicationSummary | None
    task_local: Mapping[str, Any] | None
    lora_response: Mapping[str, Any] | None
    success_key_application: Mapping[str, Any] | None
    phase_a_seconds: float
    phase_a_rows: tuple[Mapping[str, Any], ...]
    profile_task_seconds: float | None
    kernel_seconds: float | None
    verification_seconds: float | None


_LOGICAL_POLICY_BATCH_SIZE = 20
_TRAIN_TASK_COUNT = 24
_WORK_QUEUE_MINIMUM_RETAINED_TASK_CAP = 8


def _rms(value: torch.Tensor) -> float:
    return float(value.to(dtype=torch.float32).square().mean().sqrt())


def _motion_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    denominator_rms = _rms(denominator)
    numerator_rms = _rms(numerator)
    return (
        numerator_rms / denominator_rms
        if denominator_rms > 0
        else (0.0 if numerator_rms == 0 else math.inf)
    )


def _feature_rank(value: torch.Tensor) -> int:
    singular_values = torch.linalg.svdvals(value.to(dtype=torch.float64))
    maximum = float(singular_values.max()) if singular_values.numel() else 0.0
    if maximum == 0:
        return 0
    tolerance = max(value.shape) * torch.finfo(torch.float64).eps * maximum
    return int((singular_values > tolerance).sum())


def _nullspace_energy_ratio_median(
    features: torch.Tensor, anchors: torch.Tensor
) -> float:
    features64 = features.to(dtype=torch.float64)
    anchors64 = anchors.to(dtype=torch.float64)
    _, singular_values, vh = torch.linalg.svd(anchors64, full_matrices=False)
    maximum = float(singular_values.max()) if singular_values.numel() else 0.0
    tolerance = max(anchors64.shape) * torch.finfo(torch.float64).eps * maximum
    rank = int((singular_values > tolerance).sum())
    basis = vh[:rank].transpose(0, 1)
    projected = features64 - (features64 @ basis) @ basis.transpose(0, 1)
    ratio = projected.square().sum(dim=1) / features64.square().sum(
        dim=1
    ).clamp_min(torch.finfo(torch.float64).tiny)
    return float(ratio.median())


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
        return (
            index,
            self.jobs[index] if index < len(self.jobs) else None,
            time.monotonic() - started,
        )


class _PhaseAProfileNonPass(ExpertManifoldError):
    def __init__(self, evidence: Mapping[str, Any]) -> None:
        super().__init__("CVEG Work-Queue Phase-A profile gate failed")
        self.evidence = dict(evidence)


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
        raise ExpertManifoldError("PCUG action-query randomness changed")
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
        raise ExpertManifoldError("PCUG action batch lost task identity")
    unique = values.unique()
    if unique.numel() != 1:
        raise ExpertManifoldError("PCUG action batch crossed tasks")
    return int(unique.item())


def _task_objective(
    runtime: V6PriorRuntime,
    *,
    task_id: int,
    task_visit: int,
    batch: Mapping[str, Any],
) -> TaskObjective:
    if _batch_task_id(batch) != task_id:
        raise ExpertManifoldError("CVEG sampler and action batch disagree")
    task = runtime.task_by_global_id[task_id]
    excluded = runtime.sampler.action_demo_indices_for_task_visit(task_id, task_visit)
    teacher_demo = runtime.video_schedule.demos_for_task_visit(
        task_id, task_visit, excluded=excluded
    )[0]
    companion_demo = runtime.video_schedule.companion_demos_for_task_visit(
        task_id, task_visit, excluded=excluded
    )[0]
    correct_video = runtime.video_store.load(task_id, teacher_demo)
    companion_video = runtime.video_store.load(task_id, companion_demo)
    kind = counterfactual_kind(task.ordinal, task_visit)
    negative_task = None
    negative_demo = None
    negative_video = None
    if kind == "wrong":
        negative_task = cross_suite_wrong_task(
            runtime.tasks, task_ordinal=task.ordinal, task_visit=task_visit
        )
        negative_demo = runtime.video_schedule.demos_for_task_visit(
            negative_task.global_task_id, task_visit
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
        companion_video=companion_video,
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
    fixed_query = None
    if runtime.args.mode == "mechanism-profile":
        fixed_query = {
            name: value[:1].detach()
            for name, value in policy_batch.items()
            if name.startswith("observation.")
        }
        if len(fixed_query) != 4:
            raise ExpertManifoldError("PCUG fixed-action profile query changed")
    cotangent = program_cotangent(graph, lora_gradients)
    graph = replace(
        graph,
        correct_lora={name: value.detach() for name, value in graph.correct_lora.items()},
        program_leaf=graph.program_leaf.detach(),
    )
    del lora_gradients, policy_batch
    return TaskObjective(
        task=task,
        task_visit=task_visit,
        teacher_demo=teacher_demo,
        companion_demo=companion_demo,
        action_query_demos=tuple(excluded),
        counterfactual_kind=kind,
        counterfactual_task=negative_task,
        counterfactual_demo=negative_demo,
        functional_loss=functional_loss.detach(),
        correct_feature=graph.correct_feature.detach(),
        negative_feature=graph.negative_feature.detach(),
        companion_feature=graph.companion_feature.detach(),
        program_cotangent=cotangent,
        graph=graph,
        correct_raw_frames=graph.correct_raw_frames,
        correct_sampled_frames=graph.correct_sampled_frames,
        negative_raw_frames=graph.negative_raw_frames,
        negative_sampled_frames=graph.negative_sampled_frames,
        companion_raw_frames=graph.companion_raw_frames,
        companion_sampled_frames=graph.companion_sampled_frames,
        fixed_policy_query=fixed_query,
    )


def _collect_paired_task_evidence(
    runtime: V6PriorRuntime,
    objective: TaskObjective,
    *,
    schedule_macro: int,
    blind_motion: torch.Tensor,
) -> TaskObjective:
    evidence = collect_paired_task_evidence(
        runtime,
        task=objective.task,
        graph=objective.graph,
        schedule_macro=schedule_macro,
        blind_motion=blind_motion,
    )
    return replace(objective, paired=evidence)


def _task_record(value: TaskObjective) -> dict[str, Any]:
    if value.paired is None:
        raise ExpertManifoldError("PCUG task record lacks paired outcomes")
    correct_norm = torch.linalg.vector_norm(value.correct_feature)
    negative_norm = torch.linalg.vector_norm(value.negative_feature)
    companion_norm = torch.linalg.vector_norm(value.companion_feature)
    companion_cosine = torch.dot(
        value.correct_feature, value.companion_feature
    ) / (correct_norm * companion_norm).clamp_min(torch.finfo(torch.float32).tiny)
    cosine = torch.dot(value.correct_feature, value.negative_feature) / (
        correct_norm * negative_norm
    ).clamp_min(torch.finfo(torch.float32).tiny)
    scalars = (
        torch.stack(
            (
                value.functional_loss.to(dtype=torch.float32),
                value.program_cotangent.square().mean().sqrt(),
                correct_norm,
                negative_norm,
                cosine,
                companion_norm,
                companion_cosine,
            )
        )
        .detach()
        .cpu()
        .tolist()
    )
    base = value.paired.base_success
    candidate = value.paired.candidate_success
    losses = sum(left and not right for left, right in zip(base, candidate, strict=True))
    gains = sum(not left and right for left, right in zip(base, candidate, strict=True))
    return {
        "task_ordinal": value.task.ordinal,
        "global_task_id": value.task.global_task_id,
        "suite": value.task.suite,
        "task_id": value.task.task_id,
        "task_visit": value.task_visit,
        "teacher_demo": value.teacher_demo,
        "companion_demo": value.companion_demo,
        "action_query_demos": list(value.action_query_demos),
        "counterfactual_kind": value.counterfactual_kind,
        "counterfactual_global_task_id": (
            value.counterfactual_task.global_task_id
            if value.counterfactual_task is not None
            else None
        ),
        "counterfactual_demo": value.counterfactual_demo,
        "functional_loss": scalars[0],
        "program_cotangent_rms": scalars[1],
        "correct_feature_norm": scalars[2],
        "negative_feature_norm": scalars[3],
        "correct_negative_feature_cosine": scalars[4],
        "companion_feature_norm": scalars[5],
        "correct_companion_feature_cosine": scalars[6],
        "base_success": list(base),
        "candidate_success": list(candidate),
        "paired_losses": losses,
        "paired_gains": gains,
        "harmful": losses > gains,
        "beneficial": gains > losses,
        "indifferent": losses == gains,
        "stable_success": all(base) and all(candidate),
        "exact_pair_count": value.paired.exact_pair_count,
        "trajectories": list(value.paired.trajectory_rows),
        "rollout_seconds": value.paired.rollout_seconds,
        "candidate_program_motion_rms": value.paired.candidate_program_motion_rms,
        "candidate_lora_response_rms": value.paired.candidate_lora_response_rms,
        "candidate_action_response_rms": value.paired.candidate_action_response_rms,
        "correct_raw_frames": value.correct_raw_frames,
        "correct_sampled_frames": value.correct_sampled_frames,
        "negative_raw_frames": value.negative_raw_frames,
        "negative_sampled_frames": value.negative_sampled_frames,
        "companion_raw_frames": value.companion_raw_frames,
        "companion_sampled_frames": value.companion_sampled_frames,
        "phase_a_queue_index": value.phase_a_queue_index,
        "phase_a_rank": value.phase_a_rank,
        "phase_a_started_seconds": value.phase_a_started_seconds,
        "phase_a_finished_seconds": value.phase_a_finished_seconds,
        "phase_a_batch_load_seconds": value.phase_a_batch_load_seconds,
        "phase_a_claim_seconds": value.phase_a_claim_seconds,
        "source_action_queries": 20,
        "physical_correct_policy_forwards": 2,
        "base_rollouts": 2,
        "candidate_rollouts": 2,
        "reward_gradient_count": 0,
        "trajectory_replay_policy_forwards": 0,
        "trajectory_replay_cfm_forwards": 0,
        "negative_policy_forwards": 0,
        "historical_v6_video_encodes": 1,
        "post_candidate_factorhead_redecodes": 1,
        "policy_innovation_key_count": 3,
        "policy_innovation_unique_video_count": (
            3 if value.counterfactual_kind == "wrong" else 2
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


def _sorted_gather_rows(
    payload: torch.Tensor, context: DistributedContext
) -> torch.Tensor:
    gathered = _all_gather_fixed(payload, context)
    present = gathered[:, 0] >= 0
    gathered = gathered[present]
    if gathered.shape[0] != 24:
        raise ExpertManifoldError("PCUG padded gather lost train24")
    ordinals = gathered[:, 0].to(dtype=torch.long)
    order = ordinals.argsort()
    sorted_ordinals = ordinals.index_select(0, order)
    expected = torch.arange(24, dtype=torch.long, device=context.device)
    if not torch.equal(sorted_ordinals, expected):
        raise ExpertManifoldError("PCUG full24 task order changed")
    return gathered.index_select(0, order)


def _gather_full48(
    local: Sequence[TaskObjective], context: DistributedContext
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[Mapping[str, Any], ...],
]:
    maximum_local = _retained_task_cap(context.world_size)
    if not 0 <= len(local) <= maximum_local:
        raise ExpertManifoldError("CVEG local task coverage changed")
    payload = torch.zeros(
        maximum_local, 7 + 3 * 256, dtype=torch.float32, device=context.device
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
            [value.correct_feature for value in local]
        )
        payload[: len(local), 263:519] = torch.stack(
            [value.negative_feature for value in local]
        )
        payload[: len(local), 519:] = torch.stack(
            [value.companion_feature for value in local]
        )
    cotangent_shape = local[0].program_cotangent.shape if local else (320, 256)
    cotangents = torch.zeros(
        maximum_local,
        *cotangent_shape,
        dtype=torch.float32,
        device=context.device,
    )
    if local:
        cotangents[: len(local)] = torch.stack(
            [value.program_cotangent for value in local]
        )
    gathered_payload = _all_gather_fixed(payload, context)
    gathered_cotangents = _all_gather_fixed(cotangents, context)
    present = gathered_payload[:, 0] >= 0
    gathered_payload = gathered_payload[present]
    gathered_cotangents = gathered_cotangents[present]
    if gathered_payload.shape[0] != 24:
        raise ExpertManifoldError("PCUG padded gather lost train24")
    ordinals = gathered_payload[:, 0].to(dtype=torch.long)
    order = ordinals.argsort()
    if not torch.equal(
        ordinals.index_select(0, order),
        torch.arange(24, dtype=torch.long, device=context.device),
    ):
        raise ExpertManifoldError("PCUG full24 task order changed")
    gathered_payload = gathered_payload.index_select(0, order)
    gathered_cotangents = gathered_cotangents.index_select(0, order)
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
        for row in gathered_payload.detach().cpu().tolist()
    )
    return (
        gathered_payload[:, 7:263],
        gathered_payload[:, 263:519],
        gathered_payload[:, 519:],
        gathered_cotangents,
        timing,
    )


def _gather_paired_outcomes(
    local: Sequence[TaskObjective], context: DistributedContext
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum_local = _retained_task_cap(context.world_size)
    payload = torch.full(
        (maximum_local, 5), -1.0, dtype=torch.float32, device=context.device
    )
    for row, value in enumerate(local):
        if value.paired is None:
            raise ExpertManifoldError("PCUG local outcome gather is incomplete")
        payload[row] = torch.tensor(
            (
                value.task.ordinal,
                *value.paired.base_success,
                *value.paired.candidate_success,
            ),
            dtype=torch.float32,
            device=context.device,
        )
    gathered = _sorted_gather_rows(payload, context)
    outcomes = gathered[:, 1:]
    if bool(((outcomes != 0) & (outcomes != 1)).any()):
        raise ExpertManifoldError("PCUG gathered outcomes are not binary")
    return outcomes[:, :2].to(dtype=torch.bool), outcomes[:, 2:].to(dtype=torch.bool)


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
        raise ExpertManifoldError("PCUG macro did not cover train24")
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
        raise ExpertManifoldError("CVEG lost train24 jobs")
    queue_path = (
        Path("/tmp")
        / f"ember-wqpcug-{os.getuid()}"
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
            runtime,
            task_id=task_id,
            task_visit=task_visit,
            batch=batch,
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
        raise ExpertManifoldError("CVEG touched frozen parameter gradients")
    return local, input_wait_seconds


def _final_guard_features(
    plan: PersistedSuccessKeyPlan,
    correct: torch.Tensor,
    classification: PairedCandidateClassification,
) -> torch.Tensor:
    current = classification.stable_success_mask | classification.harmful_mask
    return torch.cat((plan.features, correct[current]), dim=0).contiguous()


def _apply_macro_update(
    runtime: V6PriorRuntime,
    local_objectives: Sequence[TaskObjective],
    *,
    schedule_macro: int,
    profile: bool,
    step_started: float,
) -> MacroUpdateEvidence:
    correct, negative, companion, cotangents, phase_a_rows = _gather_full48(
        local_objectives, runtime.context
    )
    equivariance = companion - correct
    phase_a_seconds = _profile_max_seconds(
        runtime.context, time.monotonic() - step_started
    )
    if profile:
        baseline_contract = runtime.config["profile_run"]["throughput_baseline"]
        baseline_seconds = float(baseline_contract["source_step_seconds"]) * (
            int(baseline_contract["source_world_size"])
            / runtime.context.world_size
        )
        gates = runtime.config["profile_run"]["gates"]
        task_counts = [
            sum(int(row["rank"]) == rank for row in phase_a_rows)
            for rank in range(runtime.context.world_size)
        ]
        claim_seconds = sum(float(row["claim_seconds"]) for row in phase_a_rows)
        queue_rows_valid = (
            sorted(int(row["queue_index"]) for row in phase_a_rows)
            == list(range(_TRAIN_TASK_COUNT))
            and sum(task_counts) == _TRAIN_TASK_COUNT
            and all(
                0 <= float(row["started_seconds"])
                <= float(row["finished_seconds"])
                for row in phase_a_rows
            )
        )
        phase_a_evidence = {
            "passed": (
                queue_rows_valid
                and phase_a_seconds
                <= baseline_seconds * float(gates["phase_a_wall_ratio_max"])
                and claim_seconds <= float(gates["queue_claim_seconds_max"])
                and max(task_counts) <= int(gates["retained_task_cap_max"])
            ),
            "phase_a_seconds": phase_a_seconds,
            "scaled_sknc_step_seconds": baseline_seconds,
            "phase_a_wall_ratio": phase_a_seconds / baseline_seconds,
            "phase_a_wall_ratio_max": float(gates["phase_a_wall_ratio_max"]),
            "queue_claim_seconds": claim_seconds,
            "queue_claim_seconds_max": float(gates["queue_claim_seconds_max"]),
            "task_counts_per_rank": task_counts,
            "queue_rows_valid": queue_rows_valid,
            "retained_task_cap": _retained_task_cap(runtime.context.world_size),
            "task_rows": list(phase_a_rows),
        }
        if runtime.context.is_main:
            print(
                json.dumps({"event": "phase_a_complete", **phase_a_evidence}, sort_keys=True),
                flush=True,
            )
        if not phase_a_evidence["passed"]:
            raise _PhaseAProfileNonPass(phase_a_evidence)
    persisted_plan = runtime.success_key_bank.persisted_plan()
    blind_anchors = torch.cat(
        (persisted_plan.features, equivariance), dim=0
    ).contiguous()
    no_current_guards = torch.zeros(24, dtype=torch.bool, device=runtime.context.device)
    blind_delta, blind_update = success_key_nullspace_program_delta(
        correct,
        negative,
        cotangents,
        blind_anchors,
        no_current_guards,
        step_size=float(runtime.config["update"]["step_size"]),
        relative_damping=float(runtime.config["update"]["relative_damping"]),
    )
    blind_motion = success_key_constraint_motion(correct, blind_delta)
    paired_local = tuple(
        _collect_paired_task_evidence(
            runtime,
            value,
            schedule_macro=schedule_macro,
            blind_motion=blind_motion[value.task.ordinal],
        )
        for value in local_objectives
    )
    base_success, candidate_success = _gather_paired_outcomes(
        paired_local, runtime.context
    )
    classification = classify_paired_candidate_outcomes(
        base_success, candidate_success
    )
    profile_task_seconds = None
    if profile:
        torch.cuda.synchronize(runtime.context.device)
        profile_task_seconds = _profile_max_seconds(
            runtime.context, time.monotonic() - step_started
        )
        torch.cuda.synchronize(runtime.context.device)
        kernel_started = time.monotonic()
    full_features = torch.cat((correct, negative), dim=0)
    preserved_responses = torch.cat((negative, equivariance), dim=0)
    delta, guard_projection = response_preserving_candidate_guard_correction(
        blind_delta,
        persisted_plan.features,
        correct,
        preserved_responses,
        classification.stable_success_mask,
        classification.harmful_mask,
        full_features,
        negative_rows=negative.shape[0],
    )
    if not profile:
        gates = runtime.config["profile_run"]["gates"]
        blind_equivariance_ratio = _motion_ratio(
            success_key_constraint_motion(equivariance, blind_delta),
            success_key_constraint_motion(correct, blind_delta),
        )
        final_equivariance_ratio = _motion_ratio(
            success_key_constraint_motion(equivariance, delta),
            success_key_constraint_motion(correct, delta),
        )
        if not (
            guard_projection.final_guard_violation_count
            == int(gates["final_guard_violation_count"])
            and guard_projection.negative_preservation_violation_count
            == int(gates["negative_preservation_violation_count"])
            and guard_projection.equivariance_preservation_violation_count
            == int(gates["equivariance_preservation_violation_count"])
            and guard_projection.projected_feature_rank
            >= int(gates["projected_feature_rank_min"])
            and guard_projection.projected_to_blind_energy_ratio
            >= float(gates["projected_to_blind_energy_ratio_min"])
            and guard_projection.blind_projected_inner_product > 0
            and guard_projection.blind_projected_cosine > 0
            and blind_update.predicted_negative_to_unprotected_ratio
            <= float(gates["negative_to_unprotected_motion_rms_max"])
            and blind_equivariance_ratio
            <= float(gates["equivariance_to_primary_motion_rms_max"])
            and final_equivariance_ratio
            <= float(gates["equivariance_to_primary_motion_rms_max"])
        ):
            evidence = {
                "blind_equivariance_ratio": blind_equivariance_ratio,
                "blind_projected_cosine": guard_projection.blind_projected_cosine,
                "blind_projected_inner_product": (
                    guard_projection.blind_projected_inner_product
                ),
                "equivariance_preservation_violation_count": (
                    guard_projection.equivariance_preservation_violation_count
                ),
                "final_equivariance_ratio": final_equivariance_ratio,
                "final_guard_violation_count": (
                    guard_projection.final_guard_violation_count
                ),
                "negative_preservation_violation_count": (
                    guard_projection.negative_preservation_violation_count
                ),
                "negative_to_unprotected_motion_ratio": (
                    blind_update.predicted_negative_to_unprotected_ratio
                ),
                "projected_feature_rank": guard_projection.projected_feature_rank,
                "projected_to_blind_energy_ratio": (
                    guard_projection.projected_to_blind_energy_ratio
                ),
            }
            raise ExpertManifoldError(
                "CVEG formal correction left its sealed feasible set: "
                + json.dumps(evidence, sort_keys=True)
            )
    apply_program_residual_delta_(runtime.writer.program_memory, delta)
    bank_update = runtime.success_key_bank.commit_first_stable_successes_(
        correct,
        classification.stable_success_mask,
        persisted_plan,
    )
    if not profile:
        return MacroUpdateEvidence(
            local_objectives=paired_local,
            blind_update=blind_update,
            classification=classification,
            guard_projection=guard_projection,
            bank_update=bank_update,
            application=None,
            task_local=None,
            lora_response=None,
            success_key_application=None,
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
    before = cotangents.new_zeros(
        full_features.shape[0], cotangents.shape[1], cotangents.shape[2]
    )
    full_motion = success_key_constraint_motion(full_features, delta)
    equivariance_motion = success_key_constraint_motion(equivariance, delta)
    blind_correct_motion = success_key_constraint_motion(correct, blind_delta)
    blind_equivariance_motion = success_key_constraint_motion(
        equivariance, blind_delta
    )
    reversed_mask = torch.tensor(
        [
            counterfactual_kind(ordinal, schedule_macro) == "reversed"
            for ordinal in range(_TRAIN_TASK_COUNT)
        ],
        dtype=torch.bool,
        device=correct.device,
    )
    application = program_residual_delta_application_evidence(
        runtime.writer.program_memory,
        delta,
        full_features,
        before,
        predicted=full_motion,
    )
    protected = classification.stable_success_mask | classification.harmful_mask
    task_local = _profile_task_local_motion(
        cotangents, full_motion, protected, runtime.config["profile_run"]["gates"]
    )
    task_local = {
        **task_local,
        "equivariance_rows": int(equivariance.shape[0]),
        "equivariance_rank": _feature_rank(equivariance),
        "correct_feature_retained_energy_ratio_median": (
            blind_update.unprotected_projected_feature_energy_ratio_median
        ),
        "reverse_process_retained_energy_ratio_median": (
            _nullspace_energy_ratio_median(
                correct[reversed_mask] - negative[reversed_mask], equivariance
            )
        ),
        "blind_equivariance_motion_rms": _rms(blind_equivariance_motion),
        "blind_equivariance_to_primary_motion_ratio": _motion_ratio(
            blind_equivariance_motion, blind_correct_motion
        ),
        "final_equivariance_motion_rms": _rms(equivariance_motion),
        "equivariance_to_primary_motion_ratio": _motion_ratio(
            equivariance_motion, full_motion[:24]
        ),
    }
    guard_features = _final_guard_features(persisted_plan, correct, classification)
    success_key_application = _profile_success_key_application(
        guard_features, delta, protected
    )
    lora_response = _profile_lora_response(
        runtime, paired_local, full_motion[:24], protected
    )
    torch.cuda.synchronize(runtime.context.device)
    verification_seconds = _profile_max_seconds(
        runtime.context, time.monotonic() - verification_started
    )
    return MacroUpdateEvidence(
        local_objectives=paired_local,
        blind_update=blind_update,
        classification=classification,
        guard_projection=guard_projection,
        bank_update=bank_update,
        application=application,
        task_local=task_local,
        lora_response=lora_response,
        success_key_application=success_key_application,
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
        raise ExpertManifoldError("PCUG negative schedule changed")
    trajectories = [
        trajectory for record in records for trajectory in record["trajectories"]
    ]
    harmful_by_suite = {
        suite: sum(record["suite"] == suite and record["harmful"] for record in records)
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    }
    candidate_response_by_suite = {
        suite: {
            "program_motion_rms_max": max(
                float(record["candidate_program_motion_rms"])
                for record in records
                if record["suite"] == suite
            ),
            "lora_response_rms_max": max(
                float(record["candidate_lora_response_rms"])
                for record in records
                if record["suite"] == suite
            ),
            "action_response_rms_max": max(
                float(record["candidate_action_response_rms"])
                for record in records
                if record["suite"] == suite
            ),
        }
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    }
    outcome_evidence = {
        **asdict(evidence.classification.summary),
        "rollouts": len(trajectories),
        "exact_pair_records": sum(int(record["exact_pair_count"]) for record in records),
        "harmful_tasks_per_suite": harmful_by_suite,
        "environment_action_steps": sum(int(value["steps"]) for value in trajectories),
        "policy_replans": sum(int(value["replan_count"]) for value in trajectories),
        "reward_gradient_count": 0,
        "trajectory_replay_policy_forwards": 0,
        "trajectory_replay_cfm_forwards": 0,
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
        "paired_outcomes": outcome_evidence,
        "candidate_response_by_suite": candidate_response_by_suite,
        "success_key_bank": asdict(evidence.bank_update),
        "counterfactual_counts": counterfactual_counts,
        "blind_update": asdict(evidence.blind_update),
        "candidate_guard_projection": asdict(evidence.guard_projection),
        "application": asdict(evidence.application) if evidence.application else None,
        "task_local_motion": evidence.task_local,
        "lora_response": evidence.lora_response,
        "success_key_application": evidence.success_key_application,
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
        raise ExpertManifoldError("PCUG metric became non-finite")
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
        [_task_record(value) for value in evidence.local_objectives], runtime.context
    )
    runtime_metrics = _runtime_maximums(runtime.context, step_started, input_wait)
    if profile and _base_versions(runtime) != versions_before:
        raise ExpertManifoldError("historical v6 state changed during PCUG profile")
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
    try:
        row = _run_one_macro(runtime, macro=0)
    except _PhaseAProfileNonPass as error:
        result = {
            "schema_version": V6_PRIOR_PROFILE_SCHEMA,
            "passed": False,
            "failure_stage": "phase_a_full24_gather_before_paired_probe",
            "retain_weight": False,
            "gates": dict(runtime.config["profile_run"]["gates"]),
            "phase_a": error.evidence,
            "paired_probe_started": False,
            "retained_checkpoint": False,
            "content_hash_policy": "disabled_by_owner",
        }
        if runtime.context.is_main:
            write_json_atomic(runtime.args.output_dir / "mechanism_profile.json", result)
            write_json_atomic(
                runtime.args.output_dir / "completion.json",
                {
                    "schema_version": V6_PRIOR_COMPLETION_SCHEMA,
                    "mode": "mechanism-profile",
                    "completed_diagnostic_macros": 0,
                    "passed": False,
                    "failure_stage": result["failure_stage"],
                    "retained_checkpoint": False,
                    "content_hash_policy": "disabled_by_owner",
                },
            )
            print(json.dumps(result, sort_keys=True), flush=True)
        return
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
                success_key_bank=runtime.success_key_bank,
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
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in ("config", "source_run", "checkpoint", "tokenizer_path", "data_root"):
        path = getattr(args, name).resolve()
        if not path.exists():
            raise ExpertManifoldError(f"missing PCUG Writer path: {path}")
        setattr(args, name, path)
    args.output_dir = args.output_dir.resolve()
    args.resume = args.resume.resolve() if args.resume else None
    if args.resume is None:
        if args.output_dir.exists() and (
            not args.output_dir.is_dir() or any(args.output_dir.iterdir())
        ):
            raise ExpertManifoldError("fresh PCUG Writer output is not empty")
    elif (
        not args.resume.is_dir()
        or args.resume.parent.name != "checkpoints"
        or args.resume.parent.parent.resolve() != args.output_dir
        or not (args.output_dir / "run_contract.json").is_file()
    ):
        raise ExpertManifoldError("PCUG Writer resume output ownership changed")
    if args.config != V6_PRIOR_CANONICAL_CONFIG.resolve():
        raise ExpertManifoldError("PCUG Writer requires the canonical config")
    load_v6_prior_config(args.config)
    if args.num_workers < 0 or (
        args.stop_after_macro is not None and args.stop_after_macro <= 0
    ):
        raise ExpertManifoldError("invalid PCUG Writer worker or stop boundary")
    return args
