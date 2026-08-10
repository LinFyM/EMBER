"""Discarded full24 Reward-Credit profile gates and mechanism probes."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_runtime import V6PriorRuntime
from ember.lora import copy_task_lora_state_


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
    state: Mapping[str, torch.Tensor],
    *,
    seed: int,
) -> torch.Tensor:
    copy_task_lora_state_(runtime.policy, state, runtime.lora_contract)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        1,
        int(runtime.policy.config.chunk_size),
        int(runtime.policy.config.max_action_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(runtime.context.device)
    moved = {
        name: value.to(runtime.context.device, non_blocking=True)
        for name, value in query.items()
    }
    with (
        _policy_attention_state(runtime.policy),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        action = runtime.policy.predict_action_chunk(moved, noise=noise, num_steps=10)
    if action.ndim != 3 or action.shape[0] != 1:
        raise ExpertManifoldError("fixed-action Reward-Credit probe changed")
    return action.detach()


def profile_lora_response(
    runtime: V6PriorRuntime,
    local: Sequence[Any],
    correct_motion: torch.Tensor,
) -> dict[str, float | int]:
    accumulators = torch.zeros(8, dtype=torch.float64, device=runtime.context.device)
    try:
        for objective in local:
            if (
                objective.program_before is None
                or objective.correct_lora_before is None
            ):
                raise ExpertManifoldError("Reward-Credit profile lost before-state")
            motion = correct_motion[objective.task.ordinal]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                after_program = (
                    objective.program_before
                    + motion.to(dtype=objective.program_before.dtype)
                ).to(dtype=torch.float32)
                after = runtime.writer.base_writer.decode_slots(after_program[None])
            for name, before in objective.correct_lora_before.items():
                difference = after[name] - before
                index = 0 if name.endswith(".lora_A.default.weight") else 1
                accumulators[index].add_(difference.float().square().sum())
                accumulators[index + 2].add_(difference.numel())
            if objective.fixed_policy_query is not None:
                before_action = _fixed_action(
                    runtime,
                    objective.fixed_policy_query,
                    objective.correct_lora_before,
                    seed=202608110000 + objective.task.ordinal,
                )
                after_action = _fixed_action(
                    runtime,
                    objective.fixed_policy_query,
                    after,
                    seed=202608110000 + objective.task.ordinal,
                )
                difference = after_action - before_action
                accumulators[4].add_(difference.float().square().sum())
                accumulators[5].add_(difference.numel())
                accumulators[6].add_(1)
                accumulators[7].add_(
                    difference.float().square().mean().sqrt().gt(0).to(torch.float64)
                )
    finally:
        copy_task_lora_state_(
            runtime.policy, runtime.identity_state, runtime.lora_contract
        )
    if runtime.context.world_size > 1:
        dist.all_reduce(accumulators, op=dist.ReduceOp.SUM)
    values = accumulators.detach().cpu().tolist()
    return {
        "lora_a_response_rms": math.sqrt(values[0] / max(values[2], 1.0)),
        "lora_b_response_rms": math.sqrt(values[1] / max(values[3], 1.0)),
        "fixed_action_response_rms": math.sqrt(values[4] / max(values[5], 1.0)),
        "fixed_action_probe_task_count": int(values[6]),
        "fixed_action_probe_policy_forwards": int(2 * values[6]),
        "fixed_action_passing_task_count": int(values[7]),
    }


@torch.no_grad()
def profile_credit_motion(
    cotangents: torch.Tensor,
    correct_motion: torch.Tensor,
) -> dict[str, float | int]:
    """Measure shared-solve motion injected into exact-zero-credit task rows."""

    if cotangents.shape != correct_motion.shape or cotangents.ndim != 3:
        raise ExpertManifoldError("Reward-Credit profile motion topology changed")
    active = cotangents.flatten(1).square().sum(dim=1) > 0
    mixed_count = int(active.sum())
    homogeneous_count = int((~active).sum())
    if mixed_count == 0:
        mixed_rms = torch.zeros((), device=correct_motion.device)
    else:
        mixed_rms = correct_motion[active].square().mean().sqrt()
    if homogeneous_count == 0:
        homogeneous_rms = torch.zeros((), device=correct_motion.device)
        moving = 0
    else:
        homogeneous_rows = correct_motion[~active].flatten(1).square().mean(1).sqrt()
        homogeneous_rms = homogeneous_rows.square().mean().sqrt()
        moving = int((homogeneous_rows > 0).sum())
    values = torch.stack((mixed_rms, homogeneous_rms)).cpu().tolist()
    return {
        "mixed_correct_motion_rms": float(values[0]),
        "homogeneous_correct_motion_rms": float(values[1]),
        "homogeneous_to_mixed_motion_ratio": float(
            values[1] / values[0] if values[0] > 0 else math.inf
        ),
        "homogeneous_task_count": homogeneous_count,
        "homogeneous_moving_task_count": moving,
    }


def base_versions(runtime: V6PriorRuntime) -> tuple[tuple[str, int], ...]:
    base = runtime.writer.base_writer
    return tuple(
        (name, value._version)
        for name, value in (*base.named_parameters(), *base.named_buffers())
    )


def _profile_sections(
    row: Mapping[str, Any],
) -> tuple[
    Sequence[Mapping[str, Any]],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    records = row.get("task_records", ())
    update = row.get("update", {})
    application = row.get("application", {})
    response = row.get("lora_response", {})
    if (
        not isinstance(records, Sequence)
        or len(records) != 24
        or not all(isinstance(value, Mapping) for value in records)
        or not isinstance(update, Mapping)
        or not isinstance(application, Mapping)
        or not isinstance(response, Mapping)
    ):
        raise ExpertManifoldError("Reward-Credit profile evidence is incomplete")
    return records, update, application, response


def _workload_checks(
    row: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
    *,
    mixed_task_count: int,
    homogeneous_task_count: int,
) -> dict[str, bool]:
    return {
        "full24_tasks": int(row.get("tasks", -1)) == int(gates["tasks"]),
        "k4_rollouts": int(row.get("rollouts", -1)) == int(gates["rollouts"])
        and all(
            int(value.get("rollouts", -1)) == int(gates["rollouts_per_task"])
            for value in records
        ),
        "one_video_per_task": int(row.get("videos", -1)) == int(gates["videos"])
        and all(
            int(value.get("videos", -1)) == int(gates["videos_per_task"])
            for value in records
        ),
        "mixed_task_coverage": mixed_task_count >= int(gates["mixed_tasks_min"]),
        "homogeneous_task_coverage": homogeneous_task_count
        >= int(gates["homogeneous_tasks_min"]),
    }


def _credit_checks(
    mixed: Sequence[Mapping[str, Any]],
    homogeneous: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    return {
        "mixed_cotangents": all(
            math.isfinite(float(value.get("program_cotangent_rms", math.nan)))
            and float(value["program_cotangent_rms"]) > 0
            and int(value.get("functional_policy_forwards", 0)) > 0
            for value in mixed
        ),
        "homogeneous_exact_zero": all(
            float(value.get("program_cotangent_rms", math.nan)) == 0.0
            and int(value.get("functional_policy_forwards", -1)) == 0
            for value in homogeneous
        ),
    }


def _mechanism_checks(
    update: Mapping[str, Any],
    application: Mapping[str, Any],
    response: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "full48_feature_rank": int(update.get("feature_rank", -1))
        >= int(gates["full48_feature_rank_min"]),
        "negative_null": math.isfinite(
            float(update.get("predicted_negative_to_correct_ratio", math.nan))
        )
        and float(update["predicted_negative_to_correct_ratio"])
        <= float(gates["negative_null_motion_ratio_max"]),
        "predicted_observed_closure": math.isfinite(
            float(application.get("predicted_observed_relative_rms", math.nan))
        )
        and float(application["predicted_observed_relative_rms"])
        <= float(gates["predicted_observed_relative_rms_max"]),
        "program_to_lora": math.isfinite(
            float(response.get("lora_a_response_rms", math.nan))
        )
        and math.isfinite(float(response.get("lora_b_response_rms", math.nan)))
        and float(response["lora_a_response_rms"]) > 0
        and float(response["lora_b_response_rms"]) > 0,
        "program_to_action": math.isfinite(
            float(response.get("fixed_action_response_rms", math.nan))
        )
        and float(response["fixed_action_response_rms"]) > 0
        and int(response.get("fixed_action_probe_task_count", -1)) == 4
        and int(response.get("fixed_action_passing_task_count", -1)) == 4,
    }


def _runtime_checks(
    row: Mapping[str, Any], gates: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        "no_retired_forwards": int(row.get("negative_policy_forwards", -1))
        == int(gates["extra_negative_policy_forwards"])
        and int(row.get("old_policy_forwards", -1))
        == int(gates["old_policy_forwards"]),
        "runtime_health": int(row.get("oom_count", -1)) == int(gates["oom_count"])
        and int(row.get("nonfinite_count", -1)) == int(gates["nonfinite_count"])
        and int(row.get("watchdog_count", -1)) == int(gates["watchdog_count"]),
    }


def profile_passes(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[bool, dict[str, Any]]:
    if len(rows) != 1:
        raise ExpertManifoldError("Reward-Credit profile must contain one full cycle")
    row = rows[0]
    gates = config["profile_run"]["gates"]
    records, update, application, response = _profile_sections(row)
    if any(type(value.get("mixed")) is not bool for value in records):
        raise ExpertManifoldError(
            "Reward-Credit profile mixed/homogeneous partition changed"
        )
    mixed = [value for value in records if value.get("mixed") is True]
    homogeneous = [value for value in records if value.get("mixed") is False]
    if len(mixed) + len(homogeneous) != len(records):
        raise ExpertManifoldError(
            "Reward-Credit profile mixed/homogeneous partition changed"
        )
    checks = _workload_checks(
        row,
        records,
        gates,
        mixed_task_count=len(mixed),
        homogeneous_task_count=len(homogeneous),
    )
    checks.update(_credit_checks(mixed, homogeneous))
    checks.update(_mechanism_checks(update, application, response, gates))
    checks.update(_runtime_checks(row, gates))
    return all(checks.values()), {
        "checks": checks,
        "mixed_tasks": len(mixed),
        "homogeneous_tasks": len(homogeneous),
        "successes": int(row.get("successes", -1)),
        "step_seconds": float(row.get("step_seconds", math.nan)),
        "max_cuda_allocated_bytes": int(row.get("max_cuda_allocated_bytes", -1)),
        "max_cuda_reserved_bytes": int(row.get("max_cuda_reserved_bytes", -1)),
    }
