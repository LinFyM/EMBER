"""Paired classification and final negative-preserving Program correction."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ember.expert_manifold.contract import ExpertManifoldError, ExpertTask
from ember.expert_manifold.v6_prior_step import (
    GeneratedConditionGraph,
    decode_candidate_program,
)
from ember.lora import copy_task_lora_state_
from ember.reward.protocol import reward_credit_environment_seed
from ember.reward.rollout import (
    RewardRolloutOutcome,
    collect_paired_reward_arm_outcomes,
)


_TASK_COUNT = 24
_PAIR_COUNT = 2


@dataclass(frozen=True)
class PairedTaskEvidence:
    """One task's exact K2-by-two-arm probe and response evidence."""

    base_success: tuple[bool, bool]
    candidate_success: tuple[bool, bool]
    trajectory_rows: tuple[Mapping[str, Any], ...]
    exact_pair_count: int
    candidate_program_motion_rms: float
    candidate_lora_response_rms: float
    candidate_action_response_rms: float
    rollout_seconds: float


@dataclass(frozen=True)
class PairedCandidateOutcomeSummary:
    """Discrete evidence from two exact-paired base/candidate states per task."""

    paired_states: int
    base_rollouts: int
    candidate_rollouts: int
    base_successes: int
    candidate_successes: int
    losses: int
    gains: int
    discordant_states: int
    harmful_task_count: int
    harmful_task_ordinals: tuple[int, ...]
    beneficial_task_count: int
    beneficial_task_ordinals: tuple[int, ...]
    indifferent_task_count: int
    indifferent_task_ordinals: tuple[int, ...]
    stable_success_task_count: int
    stable_success_task_ordinals: tuple[int, ...]


@dataclass(frozen=True)
class PairedCandidateClassification:
    """Masks used by the final guard plus their JSON-safe summary."""

    harmful_mask: torch.Tensor
    beneficial_mask: torch.Tensor
    indifferent_mask: torch.Tensor
    stable_success_mask: torch.Tensor
    summary: PairedCandidateOutcomeSummary


@dataclass(frozen=True)
class CandidateGuardProjectionSummary:
    """Geometry and closure of the negative-preserving guarded update."""

    persisted_guard_rows: int
    current_stable_guard_rows: int
    current_harmful_guard_rows: int
    current_guard_rows: int
    total_guard_rows: int
    guard_rank: int
    negative_rows: int
    negative_rank: int
    restricted_guard_rank: int
    original_feature_rank: int
    projected_feature_rank: int
    raw_guard_motion_rms: float
    final_guard_motion_rms: float
    final_guard_motion_max_abs: float
    final_guard_violation_count: int
    blind_negative_motion_rms: float
    final_negative_motion_rms: float
    negative_correction_motion_rms: float
    negative_correction_motion_max_abs: float
    negative_preservation_violation_count: int
    projection_changed: bool
    correction_rms: float
    blind_delta_rms: float
    projected_delta_rms: float
    projected_to_blind_energy_ratio: float
    blind_projected_inner_product: float
    blind_projected_cosine: float


def _trajectory_record(value: RewardRolloutOutcome, *, arm: str) -> dict[str, Any]:
    return {
        "arm": arm,
        "rollout_cursor": value.rollout_cursor,
        "environment_seed": value.env_seed,
        "policy_seed_root": value.policy_seed_root,
        "policy_noise_seeds": list(value.policy_noise_seeds),
        "success": value.success,
        "steps": value.steps,
        "reward_sum": value.reward_sum,
        "replan_count": len(value.policy_noise_seeds),
    }


def _collect_arm(
    runtime: Any,
    *,
    schedule_macro: int,
    task: ExpertTask,
    lora_state: Mapping[str, torch.Tensor],
) -> tuple[RewardRolloutOutcome, RewardRolloutOutcome]:
    reward_task = runtime.reward_task_by_global_id[task.global_task_id]
    cursors = tuple(schedule_macro * _PAIR_COUNT + lane for lane in range(_PAIR_COUNT))
    seeds = tuple(
        reward_credit_environment_seed(
            int(runtime.config["rng"]["environment_seed_root"]),
            reward_task.suite,
            reward_task.task_id,
            int(runtime.config["optimization"]["seed"]),
            cursor,
        )
        for cursor in cursors
    )
    environment = runtime.config["environment"]
    copy_task_lora_state_(runtime.policy, lora_state, runtime.lora_contract)
    try:
        outcomes = collect_paired_reward_arm_outcomes(
            envs=tuple(
                runtime.env_pool.get(reward_task, lane=lane)
                for lane in range(_PAIR_COUNT)
            ),
            policy=runtime.policy,
            preprocess=runtime.processor,
            postprocess=runtime.processor.unnormalize_action,
            suite=reward_task.suite,
            task_id=reward_task.task_id,
            global_task_id=reward_task.global_task_id,
            language=reward_task.language,
            adaptation_seed=int(runtime.config["optimization"]["seed"]),
            rollout_cursors=cursors,
            env_seeds=seeds,
            policy_seed_root=int(runtime.config["rng"]["policy_noise_seed_root"]),
            device=runtime.context.device,
            max_horizon=reward_task.horizon,
            dummy_settling_steps=int(environment["dummy_settling_steps"]),
            dummy_action=environment["dummy_action"],
            action_execution_horizon=int(environment["action_execution_horizon"]),
            num_inference_steps=int(environment["num_inference_steps"]),
        )
    finally:
        copy_task_lora_state_(runtime.policy, runtime.identity_state, runtime.lora_contract)
    if len(outcomes) != _PAIR_COUNT:
        raise ExpertManifoldError("PCUG rollout arm is not exact K2")
    return outcomes


def _lora_response_rms(
    base: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor]
) -> float:
    if set(base) != set(candidate):
        raise ExpertManifoldError("PCUG candidate LoRA topology changed")
    first = next(iter(base.values()))
    square = torch.zeros((), dtype=torch.float32, device=first.device)
    count = 0
    for name in base:
        delta = candidate[name].to(dtype=torch.float32) - base[name].to(dtype=torch.float32)
        square += delta.square().sum()
        count += delta.numel()
    if count <= 0:
        raise ExpertManifoldError("PCUG candidate LoRA is empty")
    return float((square / count).sqrt())


def collect_paired_task_evidence(
    runtime: Any,
    *,
    task: ExpertTask,
    graph: GeneratedConditionGraph,
    schedule_macro: int,
    blind_motion: torch.Tensor,
) -> PairedTaskEvidence:
    """Decode the exact candidate and compare both arms on identical K2 keys."""

    started = time.monotonic()
    candidate_program, candidate_lora = decode_candidate_program(
        graph,
        writer=runtime.writer,
        motion=blind_motion,
        device=runtime.context.device,
    )
    base = _collect_arm(
        runtime,
        schedule_macro=schedule_macro,
        task=task,
        lora_state=graph.correct_lora,
    )
    candidate = _collect_arm(
        runtime,
        schedule_macro=schedule_macro,
        task=task,
        lora_state=candidate_lora,
    )
    action_square = torch.zeros((), dtype=torch.float32, device=runtime.context.device)
    action_count = 0
    for base_row, candidate_row in zip(base, candidate, strict=True):
        shared = min(
            len(base_row.policy_noise_seeds), len(candidate_row.policy_noise_seeds)
        )
        if (
            base_row.rollout_cursor != candidate_row.rollout_cursor
            or base_row.env_seed != candidate_row.env_seed
            or base_row.policy_seed_root != candidate_row.policy_seed_root
            or shared <= 0
            or base_row.policy_noise_seeds[:shared]
            != candidate_row.policy_noise_seeds[:shared]
        ):
            raise ExpertManifoldError("PCUG base/candidate pairing changed")
        action_delta = (
            candidate_row.initial_normalized_action_chunk.to(dtype=torch.float32)
            - base_row.initial_normalized_action_chunk.to(dtype=torch.float32)
        )
        action_square += action_delta.square().sum()
        action_count += action_delta.numel()
    program_delta = candidate_program - graph.program_input_before.to(dtype=torch.float32)
    rows = tuple(
        [*(_trajectory_record(value, arm="base") for value in base)]
        + [*(_trajectory_record(value, arm="candidate") for value in candidate)]
    )
    return PairedTaskEvidence(
        base_success=tuple(bool(value.success) for value in base),
        candidate_success=tuple(bool(value.success) for value in candidate),
        trajectory_rows=rows,
        exact_pair_count=_PAIR_COUNT,
        candidate_program_motion_rms=float(program_delta.square().mean().sqrt()),
        candidate_lora_response_rms=_lora_response_rms(graph.correct_lora, candidate_lora),
        candidate_action_response_rms=float((action_square / action_count).sqrt()),
        rollout_seconds=time.monotonic() - started,
    )


def _ordinals(mask: torch.Tensor) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in torch.nonzero(mask, as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )


def classify_paired_candidate_outcomes(
    base_success: torch.Tensor,
    candidate_success: torch.Tensor,
) -> PairedCandidateClassification:
    """Classify update-specific harm without inventing a reward magnitude."""

    expected = (_TASK_COUNT, _PAIR_COUNT)
    if (
        base_success.shape != expected
        or candidate_success.shape != expected
        or base_success.dtype != torch.bool
        or candidate_success.dtype != torch.bool
        or base_success.device != candidate_success.device
    ):
        raise ExpertManifoldError("invalid PCUG paired-outcome panel")
    losses_by_task = (base_success & ~candidate_success).sum(dim=1)
    gains_by_task = (~base_success & candidate_success).sum(dim=1)
    harmful = losses_by_task > gains_by_task
    beneficial = gains_by_task > losses_by_task
    indifferent = losses_by_task == gains_by_task
    stable = base_success.all(dim=1) & candidate_success.all(dim=1)
    if not bool((harmful | beneficial | indifferent).all()) or bool(
        (harmful & beneficial).any()
        or (harmful & indifferent).any()
        or (beneficial & indifferent).any()
        or (stable & harmful).any()
    ):
        raise ExpertManifoldError("PCUG paired classification is inconsistent")
    losses = int(losses_by_task.sum())
    gains = int(gains_by_task.sum())
    summary = PairedCandidateOutcomeSummary(
        paired_states=_TASK_COUNT * _PAIR_COUNT,
        base_rollouts=_TASK_COUNT * _PAIR_COUNT,
        candidate_rollouts=_TASK_COUNT * _PAIR_COUNT,
        base_successes=int(base_success.sum()),
        candidate_successes=int(candidate_success.sum()),
        losses=losses,
        gains=gains,
        discordant_states=losses + gains,
        harmful_task_count=int(harmful.sum()),
        harmful_task_ordinals=_ordinals(harmful),
        beneficial_task_count=int(beneficial.sum()),
        beneficial_task_ordinals=_ordinals(beneficial),
        indifferent_task_count=int(indifferent.sum()),
        indifferent_task_ordinals=_ordinals(indifferent),
        stable_success_task_count=int(stable.sum()),
        stable_success_task_ordinals=_ordinals(stable),
    )
    return PairedCandidateClassification(
        harmful_mask=harmful,
        beneficial_mask=beneficial,
        indifferent_mask=indifferent,
        stable_success_mask=stable,
        summary=summary,
    )


def _numerical_rank(value: torch.Tensor) -> int:
    if value.numel() == 0:
        return 0
    singular_values = torch.linalg.svdvals(value)
    maximum = float(singular_values.max()) if singular_values.numel() else 0.0
    if maximum == 0:
        return 0
    tolerance = max(value.shape) * torch.finfo(value.dtype).eps * maximum
    return int((singular_values > tolerance).sum())


def _rms(value: torch.Tensor) -> float:
    return float(value.to(dtype=torch.float32).square().mean().sqrt())


@torch.no_grad()
def negative_preserving_candidate_guard_correction(
    blind_delta: torch.Tensor,
    persisted_features: torch.Tensor,
    correct_features: torch.Tensor,
    negative_features: torch.Tensor,
    stable_success_mask: torch.Tensor,
    harmful_mask: torch.Tensor,
    analysis_features: torch.Tensor,
) -> tuple[torch.Tensor, CandidateGuardProjectionSummary]:
    """Apply the minimum guard correction that preserves ``N @ D0``."""

    feature_width = blind_delta.shape[0] if blind_delta.ndim == 3 else 0
    current_shape = (_TASK_COUNT, feature_width)
    if (
        feature_width <= 0
        or min(blind_delta.shape[1:]) <= 0
        or blind_delta.dtype != torch.float32
        or correct_features.shape != current_shape
        or negative_features.shape != current_shape
        or persisted_features.ndim != 2
        or persisted_features.shape[1:] != (feature_width,)
        or analysis_features.ndim != 2
        or analysis_features.shape[1:] != (feature_width,)
        or stable_success_mask.shape != (_TASK_COUNT,)
        or harmful_mask.shape != (_TASK_COUNT,)
        or stable_success_mask.dtype != torch.bool
        or harmful_mask.dtype != torch.bool
        or bool((stable_success_mask & harmful_mask).any())
        or len(
            {
                blind_delta.device,
                persisted_features.device,
                correct_features.device,
                negative_features.device,
                stable_success_mask.device,
                harmful_mask.device,
                analysis_features.device,
            }
        )
        != 1
    ):
        raise ExpertManifoldError("invalid NPCG correction batch")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (
            blind_delta,
            persisted_features,
            correct_features,
            negative_features,
            analysis_features,
        )
    ):
        raise ExpertManifoldError("NPCG correction input is non-finite")

    current_mask = stable_success_mask | harmful_mask
    guards = torch.cat(
        (persisted_features, correct_features[current_mask]), dim=0
    ).contiguous()
    analysis64 = analysis_features.to(dtype=torch.float64)
    negative64 = negative_features.to(dtype=torch.float64)
    original_rank = _numerical_rank(analysis64)
    _, negative_singular_values, negative_vh = torch.linalg.svd(
        negative64, full_matrices=False
    )
    negative_maximum = (
        float(negative_singular_values.max())
        if negative_singular_values.numel()
        else 0.0
    )
    negative_tolerance = (
        max(negative64.shape)
        * torch.finfo(torch.float64).eps
        * negative_maximum
    )
    negative_rank = int((negative_singular_values > negative_tolerance).sum())
    negative_basis64 = negative_vh[:negative_rank].transpose(0, 1).contiguous()
    blind_flat = blind_delta.flatten(1)
    blind_energy = blind_flat.square().sum()
    raw_motion = (
        guards @ blind_flat
        if guards.shape[0]
        else blind_flat.new_empty((0, blind_flat.shape[1]))
    )
    projected = blind_delta
    guard_rank = 0
    restricted_guard_rank = 0
    projected_rank = original_rank
    correction = torch.zeros_like(blind_delta)
    if guards.shape[0]:
        guards64 = guards.to(dtype=torch.float64)
        guard_rank = _numerical_rank(guards64)
        if guard_rank <= 0:
            raise ExpertManifoldError("NPCG guard rows have zero numerical rank")
        restricted64 = guards64 - (
            (guards64 @ negative_basis64) @ negative_basis64.transpose(0, 1)
        )
        restricted_u, restricted_singular_values, restricted_vh = torch.linalg.svd(
            restricted64, full_matrices=False
        )
        restricted_maximum = (
            float(restricted_singular_values.max())
            if restricted_singular_values.numel()
            else 0.0
        )
        restricted_tolerance = (
            max(restricted64.shape)
            * torch.finfo(torch.float64).eps
            * restricted_maximum
        )
        restricted_guard_rank = int(
            (restricted_singular_values > restricted_tolerance).sum()
        )
        if restricted_guard_rank:
            restricted_pinv64 = (
                restricted_vh[:restricted_guard_rank].transpose(0, 1)
                / restricted_singular_values[:restricted_guard_rank]
            ) @ restricted_u[:, :restricted_guard_rank].transpose(0, 1)
            correction_flat = -(
                restricted_pinv64.to(dtype=torch.float32) @ raw_motion
            )
            correction = correction_flat.reshape_as(blind_delta).contiguous()
            projected = (blind_delta + correction).contiguous()
            correction_operator64 = restricted_pinv64 @ guards64
            projected_features64 = analysis64 @ (
                torch.eye(
                    feature_width,
                    dtype=torch.float64,
                    device=blind_delta.device,
                )
                - correction_operator64
            )
        else:
            projected_features64 = analysis64
        projected_rank = _numerical_rank(projected_features64)

    projected_flat = projected.flatten(1)
    correction_flat = correction.flatten(1)
    final_motion = (
        guards @ projected_flat
        if guards.shape[0]
        else projected_flat.new_empty((0, projected_flat.shape[1]))
    )
    if guards.shape[0]:
        row_tolerance = (
            64
            * max(blind_delta.shape)
            * torch.finfo(torch.float32).eps
            * torch.linalg.vector_norm(guards, dim=1)
            * torch.linalg.vector_norm(projected_flat)
        )
        final_norm = torch.linalg.vector_norm(final_motion, dim=1)
        final_violations = int((final_norm > row_tolerance).sum())
    else:
        final_violations = 0
    blind_negative_motion = negative_features @ blind_flat
    final_negative_motion = negative_features @ projected_flat
    negative_correction_motion = negative_features @ correction_flat
    negative_row_tolerance = (
        64
        * max(blind_delta.shape)
        * torch.finfo(torch.float32).eps
        * torch.linalg.vector_norm(negative_features, dim=1)
        * torch.linalg.vector_norm(correction_flat)
    )
    negative_preservation_violations = int(
        (
            torch.linalg.vector_norm(negative_correction_motion, dim=1)
            > negative_row_tolerance
        ).sum()
    )
    projected_energy = projected_flat.square().sum()
    inner = torch.sum(blind_flat * projected_flat)
    cosine = inner / (
        blind_energy.sqrt() * projected_energy.sqrt()
    ).clamp_min(torch.finfo(torch.float32).tiny)
    values = (
        projected,
        raw_motion,
        final_motion,
        correction,
        blind_energy,
        projected_energy,
        inner,
        cosine,
    )
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ExpertManifoldError("NPCG corrected candidate became invalid")
    summary = CandidateGuardProjectionSummary(
        persisted_guard_rows=int(persisted_features.shape[0]),
        current_stable_guard_rows=int(stable_success_mask.sum()),
        current_harmful_guard_rows=int(harmful_mask.sum()),
        current_guard_rows=int(current_mask.sum()),
        total_guard_rows=int(guards.shape[0]),
        guard_rank=guard_rank,
        negative_rows=int(negative_features.shape[0]),
        negative_rank=negative_rank,
        restricted_guard_rank=restricted_guard_rank,
        original_feature_rank=original_rank,
        projected_feature_rank=projected_rank,
        raw_guard_motion_rms=_rms(raw_motion) if raw_motion.numel() else 0.0,
        final_guard_motion_rms=_rms(final_motion) if final_motion.numel() else 0.0,
        final_guard_motion_max_abs=(
            float(final_motion.abs().max()) if final_motion.numel() else 0.0
        ),
        final_guard_violation_count=final_violations,
        blind_negative_motion_rms=_rms(blind_negative_motion),
        final_negative_motion_rms=_rms(final_negative_motion),
        negative_correction_motion_rms=_rms(negative_correction_motion),
        negative_correction_motion_max_abs=float(negative_correction_motion.abs().max()),
        negative_preservation_violation_count=negative_preservation_violations,
        projection_changed=not torch.equal(projected, blind_delta),
        correction_rms=_rms(correction),
        blind_delta_rms=_rms(blind_delta),
        projected_delta_rms=_rms(projected),
        projected_to_blind_energy_ratio=(
            1.0
            if not guards.shape[0]
            else float(
                projected_energy
                / blind_energy.clamp_min(torch.finfo(torch.float32).tiny)
            )
        ),
        blind_projected_inner_product=float(inner),
        blind_projected_cosine=1.0 if not guards.shape[0] else float(cosine),
    )
    if not all(
        math.isfinite(value)
        for value in (
            summary.raw_guard_motion_rms,
            summary.final_guard_motion_rms,
            summary.blind_negative_motion_rms,
            summary.final_negative_motion_rms,
            summary.negative_correction_motion_rms,
            summary.projected_to_blind_energy_ratio,
            summary.blind_projected_inner_product,
            summary.blind_projected_cosine,
        )
    ):
        raise ExpertManifoldError("NPCG correction evidence became invalid")
    return projected, summary
