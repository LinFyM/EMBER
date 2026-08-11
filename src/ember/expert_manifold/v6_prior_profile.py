"""Mechanism evidence for SKNC's outcome-only constrained Program write."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_runtime import V6PriorRuntime
from ember.lora import copy_task_lora_state_, task_lora_state_dict
from ember.pi05_source_checkpoint import DistributedContext


_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def profile_max_seconds(context: DistributedContext, seconds: float) -> float:
    value = torch.tensor(seconds, dtype=torch.float64, device=context.device)
    if context.world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value)


def _rms_from_totals(square_sum: float, count: int) -> float:
    return math.sqrt(square_sum / count) if count else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return numerator / denominator
    return 0.0 if numerator == 0 else math.inf


def _lora_response_row(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
) -> dict[str, float | int]:
    if set(before) != set(after):
        raise ExpertManifoldError("SKNC profile LoRA topology changed")
    a_square = b_square = ba_square = 0.0
    a_count = b_count = ba_count = 0
    for name, before_a in before.items():
        if not name.endswith(".lora_A.default.weight"):
            continue
        b_name = name.replace(
            ".lora_A.default.weight", ".lora_B.default.weight"
        )
        if b_name not in before:
            raise ExpertManifoldError("SKNC profile lost a LoRA factor pair")
        after_a = after[name]
        before_b = before[b_name]
        after_b = after[b_name]
        a_delta = (after_a - before_a).to(dtype=torch.float32)
        b_delta = (after_b - before_b).to(dtype=torch.float32)
        ba_delta = (
            after_b.to(dtype=torch.float32) @ after_a.to(dtype=torch.float32)
            - before_b.to(dtype=torch.float32) @ before_a.to(dtype=torch.float32)
        )
        a_square += float(a_delta.square().sum())
        b_square += float(b_delta.square().sum())
        ba_square += float(ba_delta.square().sum())
        a_count += a_delta.numel()
        b_count += b_delta.numel()
        ba_count += ba_delta.numel()
    if min(a_count, b_count, ba_count) <= 0:
        raise ExpertManifoldError("SKNC profile LoRA response is empty")
    return {
        "lora_a_square_sum": a_square,
        "lora_a_value_count": a_count,
        "lora_b_square_sum": b_square,
        "lora_b_value_count": b_count,
        "effective_ba_square_sum": ba_square,
        "effective_ba_value_count": ba_count,
    }


def _fixed_action_ordinals(
    protected_mask: torch.Tensor, *, protected: bool
) -> tuple[int, ...]:
    result = []
    for start in range(0, 24, 6):
        candidates = [
            ordinal
            for ordinal in range(start, start + 6)
            if bool(protected_mask[ordinal]) is protected
        ]
        if candidates:
            result.append(candidates[0])
    return tuple(result)


def profile_lora_response(
    runtime: V6PriorRuntime,
    local: Sequence[Any],
    correct_motion: torch.Tensor,
    protected_mask: torch.Tensor,
) -> dict[str, Any]:
    """Trace protected/unprotected motion through LoRA, BA, and fixed actions."""

    if (
        correct_motion.shape[0] != 24
        or protected_mask.shape != (24,)
        or protected_mask.dtype != torch.bool
    ):
        raise ExpertManifoldError("SKNC profile protection topology changed")
    protected_action_ordinals = set(
        _fixed_action_ordinals(protected_mask, protected=True)
    )
    unprotected_action_ordinals = set(
        _fixed_action_ordinals(protected_mask, protected=False)
    )
    identity = task_lora_state_dict(runtime.policy, clone=True)
    local_rows: list[dict[str, Any]] = []
    try:
        for objective in local:
            if objective.program_before is None or objective.correct_lora_before is None:
                raise ExpertManifoldError("SKNC profile lost its before state")
            ordinal = int(objective.task.ordinal)
            protected = bool(protected_mask[ordinal])
            motion = correct_motion[ordinal]
            if motion.shape != objective.program_before.shape:
                raise ExpertManifoldError("SKNC profile Program motion changed")
            with torch.autocast(
                device_type=runtime.context.device.type,
                dtype=torch.bfloat16,
                enabled=runtime.context.device.type == "cuda",
            ):
                after_program = (
                    objective.program_before
                    + motion.to(dtype=objective.program_before.dtype)
                ).to(dtype=torch.float32)
                after = runtime.writer.base_writer.decode_slots(after_program[None])
            response = _lora_response_row(objective.correct_lora_before, after)
            fixed_probe = (
                ordinal in protected_action_ordinals
                or ordinal in unprotected_action_ordinals
            )
            action_rms = None
            if fixed_probe:
                if objective.fixed_policy_query is None:
                    raise ExpertManifoldError("SKNC profile lost fixed-action query")
                before_action = _fixed_action(
                    runtime,
                    objective.fixed_policy_query,
                    objective.correct_lora_before,
                    seed=202608110000 + ordinal,
                )
                after_action = _fixed_action(
                    runtime,
                    objective.fixed_policy_query,
                    after,
                    seed=202608110000 + ordinal,
                )
                action_rms = float(
                    (after_action - before_action)
                    .to(dtype=torch.float32)
                    .square()
                    .mean()
                    .sqrt()
                )
            local_rows.append(
                {
                    "task_ordinal": ordinal,
                    "suite": objective.task.suite,
                    "protected": protected,
                    "fixed_action_probe": fixed_probe,
                    "fixed_action_response_rms": action_rms,
                    **response,
                }
            )
    finally:
        copy_task_lora_state_(runtime.policy, identity, runtime.lora_contract)
    gathered: list[Any] = [None] * runtime.context.world_size
    if runtime.context.world_size > 1:
        dist.all_gather_object(gathered, local_rows)
    else:
        gathered[0] = local_rows
    rows = [dict(row) for rank_rows in gathered for row in rank_rows]
    rows.sort(key=lambda row: int(row["task_ordinal"]))
    if len(rows) != 24 or [int(row["task_ordinal"]) for row in rows] != list(
        range(24)
    ):
        raise ExpertManifoldError("SKNC profile LoRA response lost train24")

    def aggregate(prefix: str, protected: bool) -> float:
        square = sum(
            float(row[f"{prefix}_square_sum"])
            for row in rows
            if bool(row["protected"]) is protected
        )
        count = sum(
            int(row[f"{prefix}_value_count"])
            for row in rows
            if bool(row["protected"]) is protected
        )
        return _rms_from_totals(square, count)

    protected_a = aggregate("lora_a", True)
    unprotected_a = aggregate("lora_a", False)
    protected_b = aggregate("lora_b", True)
    unprotected_b = aggregate("lora_b", False)
    protected_ba = aggregate("effective_ba", True)
    unprotected_ba = aggregate("effective_ba", False)
    protected_action = [
        float(row["fixed_action_response_rms"])
        for row in rows
        if bool(row["protected"]) and bool(row["fixed_action_probe"])
    ]
    protected_action_rows = [
        row
        for row in rows
        if bool(row["protected"]) and bool(row["fixed_action_probe"])
    ]
    unprotected_action_rows = [
        row
        for row in rows
        if not bool(row["protected"]) and bool(row["fixed_action_probe"])
    ]
    unprotected_actions = [
        float(row["fixed_action_response_rms"])
        for row in unprotected_action_rows
    ]
    probe_count = len(protected_action) + len(unprotected_actions)
    return {
        "task_count": 24,
        "protected_task_count": sum(bool(row["protected"]) for row in rows),
        "unprotected_task_count": sum(not bool(row["protected"]) for row in rows),
        "protected_lora_a_response_rms": protected_a,
        "unprotected_lora_a_response_rms": unprotected_a,
        "protected_lora_a_to_unprotected_ratio": _ratio(
            protected_a, unprotected_a
        ),
        "protected_lora_b_response_rms": protected_b,
        "unprotected_lora_b_response_rms": unprotected_b,
        "protected_lora_b_to_unprotected_ratio": _ratio(
            protected_b, unprotected_b
        ),
        "protected_effective_ba_response_rms": protected_ba,
        "unprotected_effective_ba_response_rms": unprotected_ba,
        "protected_effective_ba_to_unprotected_ratio": _ratio(
            protected_ba, unprotected_ba
        ),
        "fixed_action_probe_task_count": probe_count,
        "fixed_action_probe_policy_forwards": 2 * probe_count,
        "protected_fixed_action_probe_task_count": len(protected_action),
        "protected_fixed_action_probe_suites": sorted(
            {str(row["suite"]) for row in protected_action_rows}
        ),
        "protected_fixed_action_response_rms": _rms_from_totals(
            sum(value * value for value in protected_action), len(protected_action)
        ),
        "protected_fixed_action_response_max": max(protected_action, default=0.0),
        "unprotected_fixed_action_probe_task_count": len(unprotected_actions),
        "unprotected_fixed_action_probe_suites": sorted(
            {str(row["suite"]) for row in unprotected_action_rows}
        ),
        "unprotected_fixed_action_response_rms": _rms_from_totals(
            sum(value * value for value in unprotected_actions),
            len(unprotected_actions),
        ),
        "unprotected_fixed_action_passing_task_count": sum(
            value > 0 for value in unprotected_actions
        ),
        "rows": rows,
    }


@torch.no_grad()
def profile_task_local_motion(
    cotangents: torch.Tensor,
    full_motion: torch.Tensor,
    protected_mask: torch.Tensor,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        cotangents.shape[0] != 24
        or full_motion.shape[0] != 48
        or protected_mask.shape != (24,)
        or protected_mask.dtype != torch.bool
    ):
        raise ExpertManifoldError("SKNC task-local profile motion changed")
    flat_credit = cotangents.flatten(1).to(dtype=torch.float64)
    flat_correct = full_motion[:24].flatten(1).to(dtype=torch.float64)
    flat_negative = full_motion[24:].flatten(1).to(dtype=torch.float64)
    credit = flat_credit.square().mean(dim=1).sqrt()
    correct = flat_correct.square().mean(dim=1).sqrt()
    negative = flat_negative.square().mean(dim=1).sqrt()
    unprotected = ~protected_mask
    global_unprotected = (
        flat_correct[unprotected].square().mean().sqrt()
        if bool(unprotected.any())
        else flat_correct.new_zeros(())
    )
    protected_rms = (
        flat_correct[protected_mask].square().mean().sqrt()
        if bool(protected_mask.any())
        else flat_correct.new_zeros(())
    )
    negative_rms = flat_negative.square().mean().sqrt()
    tiny = torch.finfo(torch.float64).tiny
    descent = -(
        (flat_credit * flat_correct).sum(dim=1)
        / (
            torch.linalg.vector_norm(flat_credit, dim=1)
            * torch.linalg.vector_norm(flat_correct, dim=1)
        ).clamp_min(tiny)
    )
    negative_ratio = negative / global_unprotected.clamp_min(tiny)
    values = torch.cat((credit, correct, negative, descent, negative_ratio))
    if not bool(torch.isfinite(values).all()):
        raise ExpertManifoldError("SKNC task-local motion became non-finite")
    null_maximum = float(gates["negative_to_unprotected_motion_rms_max"])
    return {
        "task_count": 24,
        "protected_task_count": int(protected_mask.sum()),
        "unprotected_task_count": int(unprotected.sum()),
        "unprotected_correct_motion_rms": float(global_unprotected),
        "protected_correct_motion_rms": float(protected_rms),
        "protected_to_unprotected_motion_ratio": _ratio(
            float(protected_rms), float(global_unprotected)
        ),
        "negative_motion_rms": float(negative_rms),
        "negative_to_unprotected_motion_ratio": _ratio(
            float(negative_rms), float(global_unprotected)
        ),
        "unprotected_descent_passing_tasks": int((descent[unprotected] > 0).sum()),
        "negative_null_passing_tasks": int((negative_ratio <= null_maximum).sum()),
        "rows": [
            {
                "task_ordinal": ordinal,
                "protected": bool(protected_mask[ordinal]),
                "cotangent_rms": float(credit[ordinal]),
                "correct_motion_rms": float(correct[ordinal]),
                "negative_motion_rms": float(negative[ordinal]),
                "descent_cosine": float(descent[ordinal]),
                "negative_to_unprotected_motion_rms": float(
                    negative_ratio[ordinal]
                ),
            }
            for ordinal in range(24)
        ],
    }


@torch.no_grad()
def profile_success_key_application(
    anchor_features: torch.Tensor,
    delta: torch.Tensor,
    protected_mask: torch.Tensor,
) -> dict[str, Any]:
    if (
        anchor_features.ndim != 2
        or delta.ndim != 3
        or anchor_features.shape[1] != delta.shape[0]
        or protected_mask.shape != (24,)
        or protected_mask.dtype != torch.bool
    ):
        raise ExpertManifoldError("SKNC success-key application topology changed")
    motion = anchor_features.to(dtype=torch.float32) @ delta.flatten(1)
    rms = float(motion.square().mean().sqrt()) if motion.numel() else 0.0
    maximum = float(motion.abs().max()) if motion.numel() else 0.0
    if not math.isfinite(rms) or not math.isfinite(maximum):
        raise ExpertManifoldError("SKNC success-key motion became non-finite")
    return {
        "constraint_row_count": int(anchor_features.shape[0]),
        "current_protected_task_count": int(protected_mask.sum()),
        "anchor_program_motion_rms": rms,
        "anchor_program_motion_max_abs": maximum,
    }


@contextmanager
def _policy_attention_state(policy: torch.nn.Module) -> Iterator[None]:
    bridge = policy.model.paligemma_with_expert
    language = bridge.paligemma.model.language_model.config
    expert = bridge.gemma_expert.model.config
    before = (language._attn_implementation, expert._attn_implementation)
    try:
        yield
    finally:
        language._attn_implementation, expert._attn_implementation = before


@torch.inference_mode()
def _fixed_action(
    runtime: V6PriorRuntime,
    query: Mapping[str, torch.Tensor],
    lora_state: Mapping[str, torch.Tensor],
    *,
    seed: int,
) -> torch.Tensor:
    copy_task_lora_state_(runtime.policy, lora_state, runtime.lora_contract)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        1,
        int(runtime.policy.config.chunk_size),
        int(runtime.policy.config.max_action_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(runtime.context.device)
    with (
        _policy_attention_state(runtime.policy),
        torch.autocast(
            device_type=runtime.context.device.type,
            dtype=torch.bfloat16,
            enabled=runtime.context.device.type == "cuda",
        ),
    ):
        action = runtime.policy.predict_action_chunk(
            dict(query), noise=noise, num_steps=10
        )
    if action.ndim != 3 or action.shape[0] != 1:
        raise ExpertManifoldError("SKNC fixed-action profile output changed")
    return action.detach()


def base_versions(runtime: V6PriorRuntime) -> tuple[tuple[str, int], ...]:
    base = runtime.writer.base_writer
    return tuple(
        (name, value._version)
        for name, value in (*base.named_parameters(), *base.named_buffers())
    )


def _negative_null_per_kind(
    row: Mapping[str, Any], gates: Mapping[str, Any]
) -> dict[str, int]:
    ratios = {
        int(value.get("task_ordinal", -1)): float(
            value.get("negative_to_unprotected_motion_rms", math.inf)
        )
        for value in row.get("task_local_motion", {}).get("rows", ())
    }
    maximum = float(gates["negative_to_unprotected_motion_rms_max"])
    return {
        kind: sum(
            record.get("counterfactual_kind") == kind
            and ratios.get(int(record.get("task_ordinal", -1)), math.inf)
            <= maximum
            for record in row.get("task_records", ())
        )
        for kind in ("reversed", "shuffled", "wrong")
    }


def profile_passes(
    config: Mapping[str, Any], row: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    gates = config["profile_run"]["gates"]
    required = (
        "update",
        "application",
        "task_local_motion",
        "lora_response",
        "success_key_application",
        "success_outcomes",
        "success_key_bank",
    )
    if not all(isinstance(row.get(name), Mapping) for name in required):
        raise ExpertManifoldError("SKNC mechanism profile evidence is incomplete")
    update = row["update"]
    application = row["application"]
    task_local = row["task_local_motion"]
    response = row["lora_response"]
    key_application = row["success_key_application"]
    outcomes = row["success_outcomes"]
    bank = row["success_key_bank"]
    records = row.get("task_records", ())
    per_kind = _negative_null_per_kind(row, gates)
    protected = int(outcomes.get("all_success_tasks", -1))
    unprotected = int(update.get("unprotected_correct_conditions", -1))
    required_descent = math.ceil(
        float(gates["unprotected_descent_fraction_min"]) * max(unprotected, 0)
    )
    baseline_contract = config["profile_run"]["throughput_baseline"]
    baseline = float(baseline_contract["step_seconds"]) * (
        int(row["maximum_tasks_per_rank"])
        / int(baseline_contract["source_tasks_per_rank"])
    )
    wall_ratio = float(row["step_seconds"]) / baseline
    outcome_only = (
        int(outcomes.get("retained_observation_tensors", -1)) == 0
        and int(outcomes.get("retained_action_tensors", -1)) == 0
        and int(outcomes.get("trajectory_replay_policy_forwards", -1)) == 0
        and int(outcomes.get("trajectory_replay_cfm_forwards", -1)) == 0
        and int(outcomes.get("reward_gradient_count", -1)) == 0
        and all(
            int(trajectory.get("retained_observation_tensors", -1)) == 0
            and int(trajectory.get("retained_action_tensors", -1)) == 0
            for record in records
            for trajectory in record.get("trajectories", ())
        )
    )
    all_success_by_suite = outcomes.get("all_success_tasks_per_suite", {})
    checks = {
        "full24_information_wall": len(records) == int(gates["task_count"])
        and sum(int(record.get("historical_v6_video_encodes", -1)) for record in records)
        == int(gates["video_count"])
        and sum(int(record.get("source_action_queries", -1)) for record in records)
        == int(gates["source_action_query_count"])
        and sum(int(record.get("negative_policy_forwards", -1)) for record in records)
        == int(gates["negative_policy_forwards"]),
        "k4_outcome_coverage": int(outcomes.get("rollouts", -1))
        == int(gates["rollout_count"])
        and protected >= int(gates["all_success_task_count_min"])
        and isinstance(all_success_by_suite, Mapping)
        and set(all_success_by_suite) == set(_SUITES)
        and all(
            int(all_success_by_suite[suite])
            >= int(gates["all_success_task_per_suite_min"])
            for suite in _SUITES
        ),
        "outcome_only_credit": outcome_only,
        "first_success_bank": int(bank.get("persisted_before_count", -1)) == 0
        and int(bank.get("current_all_success_count", -1)) == protected
        and int(bank.get("newly_stored_count", -1)) == protected
        and int(bank.get("persisted_after_count", -1)) == protected
        and int(bank.get("constraint_row_count", -1)) == protected
        and int(update.get("anchor_constraint_rows", -1)) == protected
        and int(key_application.get("constraint_row_count", -1)) == protected,
        "feature_rank_partition": int(update.get("original_feature_rank", -1))
        == int(gates["original_feature_rank"])
        and int(update.get("anchor_rank", -1)) == protected
        and int(update.get("projected_feature_rank", -1))
        + int(update.get("anchor_rank", -1))
        == int(update.get("original_feature_rank", -2)),
        "projected_condition": float(
            update.get("active_regularized_gram_condition_number", math.inf)
        )
        <= float(gates["active_regularized_gram_condition_number_max"]),
        "projected_energy": float(
            update.get(
                "unprotected_projected_feature_energy_ratio_median", -math.inf
            )
        )
        >= float(gates["unprotected_projected_feature_energy_ratio_median_min"]),
        "protected_program_closure": float(
            task_local.get("protected_to_unprotected_motion_ratio", math.inf)
        )
        <= float(gates["protected_to_unprotected_motion_ratio_max"])
        and float(
            update.get("predicted_protected_to_unprotected_ratio", math.inf)
        )
        <= float(gates["protected_to_unprotected_motion_ratio_max"])
        and _ratio(
            float(key_application.get("anchor_program_motion_rms", math.inf)),
            float(task_local.get("unprotected_correct_motion_rms", 0.0)),
        )
        <= float(gates["protected_to_unprotected_motion_ratio_max"]),
        "unprotected_descent": int(
            task_local.get("unprotected_descent_passing_tasks", -1)
        )
        >= required_descent
        and float(task_local.get("unprotected_correct_motion_rms", 0.0)) > 0,
        "negative_null": float(
            task_local.get("negative_to_unprotected_motion_ratio", math.inf)
        )
        <= float(gates["negative_to_unprotected_motion_rms_max"])
        and int(task_local.get("negative_null_passing_tasks", -1))
        >= int(gates["negative_null_task_count_min"])
        and all(
            count >= int(gates["negative_null_per_kind_min"])
            for count in per_kind.values()
        ),
        "predicted_observed_closure": float(
            application.get("predicted_observed_relative_rms", math.inf)
        )
        <= float(gates["predicted_observed_relative_rms_max"]),
        "protected_lora_closure": all(
            float(response.get(name, math.inf))
            <= float(gates["protected_to_unprotected_lora_response_ratio_max"])
            for name in (
                "protected_lora_a_to_unprotected_ratio",
                "protected_lora_b_to_unprotected_ratio",
                "protected_effective_ba_to_unprotected_ratio",
            )
        ),
        "unprotected_lora_response": all(
            math.isfinite(float(response.get(name, math.nan)))
            and float(response.get(name, 0.0)) > 0
            for name in (
                "unprotected_lora_a_response_rms",
                "unprotected_lora_b_response_rms",
                "unprotected_effective_ba_response_rms",
            )
        ),
        "fixed_action_closure_and_breadth": int(
            response.get("protected_fixed_action_probe_task_count", -1)
        )
        == len(_SUITES)
        and set(response.get("protected_fixed_action_probe_suites", ()))
        == set(_SUITES)
        and float(response.get("protected_fixed_action_response_max", math.inf))
        <= float(gates["protected_fixed_action_response_rms_max"])
        and int(response.get("unprotected_fixed_action_probe_task_count", -1))
        == int(gates["unprotected_fixed_action_probe_task_count"])
        and set(response.get("unprotected_fixed_action_probe_suites", ()))
        == set(_SUITES)
        and int(response.get("unprotected_fixed_action_passing_task_count", -1))
        == int(gates["unprotected_fixed_action_probe_task_count"])
        and int(response.get("fixed_action_probe_policy_forwards", -1))
        == 2 * int(response.get("fixed_action_probe_task_count", -2)),
        "production_wall_overhead": math.isfinite(wall_ratio)
        and wall_ratio <= float(gates["production_wall_ratio_max"]),
        "oom_and_nonfinite": int(row.get("oom_count", -1))
        == int(gates["oom_count"])
        and int(row.get("nonfinite_count", -1))
        == int(gates["nonfinite_count"])
        and all(
            math.isfinite(float(value))
            for value in (
                row.get("functional_loss", math.nan),
                row.get("program_cotangent_rms", math.nan),
                row.get("step_seconds", math.nan),
                update.get("value_delta_rms", math.nan),
            )
        ),
    }
    evidence = {
        "checks": checks,
        "success_outcomes": dict(outcomes),
        "success_key_bank": dict(bank),
        "success_key_application": dict(key_application),
        "rank": {
            "original": int(update["original_feature_rank"]),
            "anchor": int(update["anchor_rank"]),
            "projected": int(update["projected_feature_rank"]),
        },
        "active_regularized_gram_condition_number": float(
            update["active_regularized_gram_condition_number"]
        ),
        "unprotected_projected_feature_energy_ratio_median": float(
            update["unprotected_projected_feature_energy_ratio_median"]
        ),
        "protected_to_unprotected_program_motion_ratio": float(
            task_local["protected_to_unprotected_motion_ratio"]
        ),
        "negative_to_unprotected_program_motion_ratio": float(
            task_local["negative_to_unprotected_motion_ratio"]
        ),
        "negative_null_per_kind": per_kind,
        "lora_response": dict(response),
        "step_seconds": float(row["step_seconds"]),
        "scaled_step_seconds_baseline": baseline,
        "world_size": int(row["world_size"]),
        "task_counts_per_rank": list(row["task_counts_per_rank"]),
        "production_wall_ratio_to_sealed_k4": wall_ratio,
    }
    return all(checks.values()), evidence
