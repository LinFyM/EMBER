"""One-shot mechanism evidence for the frozen-v6 Program residual."""

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


def profile_max_seconds(context: DistributedContext, seconds: float) -> float:
    value = torch.tensor(seconds, dtype=torch.float64, device=context.device)
    if context.world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value)


def profile_lora_response(
    runtime: V6PriorRuntime,
    local: Sequence[Any],
    correct_motion: torch.Tensor,
) -> dict[str, float]:
    accumulators = torch.zeros(8, dtype=torch.float64, device=runtime.context.device)
    identity = task_lora_state_dict(runtime.policy, clone=True)
    try:
        for objective in local:
            if (
                objective.program_before is None
                or objective.correct_lora_before is None
            ):
                raise ExpertManifoldError("mechanism profile lost its before state")
            motion = correct_motion[objective.task.ordinal]
            if motion.shape != objective.program_before.shape:
                raise ExpertManifoldError("mechanism profile Program motion changed")
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
            for name, before in objective.correct_lora_before.items():
                difference = after[name] - before
                index = 0 if name.endswith(".lora_A.default.weight") else 1
                accumulators[index].add_(
                    difference.to(dtype=torch.float32).square().sum()
                )
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
                action_rms = difference.to(dtype=torch.float32).square().mean().sqrt()
                accumulators[4].add_(difference.to(dtype=torch.float32).square().sum())
                accumulators[5].add_(difference.numel())
                accumulators[6].add_(1)
                accumulators[7].add_(int(float(action_rms) > 0))
    finally:
        copy_task_lora_state_(runtime.policy, identity, runtime.lora_contract)
    if runtime.context.world_size > 1:
        dist.all_reduce(accumulators, op=dist.ReduceOp.SUM)
    return {
        "lora_a_response_rms": float(
            (accumulators[0] / accumulators[2].clamp_min(1)).sqrt()
        ),
        "lora_b_response_rms": float(
            (accumulators[1] / accumulators[3].clamp_min(1)).sqrt()
        ),
        "fixed_action_response_rms": float(
            (accumulators[4] / accumulators[5].clamp_min(1)).sqrt()
        ),
        "fixed_action_probe_task_count": int(accumulators[6].item()),
        "fixed_action_probe_policy_forwards": int(2 * accumulators[6].item()),
        "fixed_action_passing_task_count": int(accumulators[7].item()),
    }


@torch.no_grad()
def profile_task_local_motion(
    cotangents: torch.Tensor,
    full_motion: torch.Tensor,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    if cotangents.shape[0] != 24 or full_motion.shape[0] != 48:
        raise ExpertManifoldError("task-local profile motion changed")
    correct = full_motion[:24].flatten(1).square().mean(dim=1).sqrt()
    negative = full_motion[24:].flatten(1).square().mean(dim=1).sqrt()
    credit = cotangents.flatten(1).square().mean(dim=1).sqrt()
    tiny = torch.finfo(torch.float32).tiny
    retained = correct / credit.clamp_min(tiny)
    leakage = negative / correct.clamp_min(tiny)
    values = torch.stack((correct, negative, credit, retained, leakage))
    if not bool(torch.isfinite(values).all()):
        raise ExpertManifoldError("task-local profile motion became non-finite")
    retained_minimum = float(gates["correct_motion_to_cotangent_rms_min"])
    leakage_maximum = float(gates["negative_to_correct_motion_rms_max"])
    return {
        "task_count": 24,
        "correct_retained_threshold": retained_minimum,
        "negative_leakage_threshold": leakage_maximum,
        "correct_retained_passing_tasks": int((retained >= retained_minimum).sum()),
        "negative_null_passing_tasks": int((leakage <= leakage_maximum).sum()),
        "rows": [
            {
                "task_ordinal": ordinal,
                "cotangent_rms": float(credit[ordinal]),
                "correct_motion_rms": float(correct[ordinal]),
                "negative_motion_rms": float(negative[ordinal]),
                "correct_motion_to_cotangent_rms": float(retained[ordinal]),
                "negative_to_correct_motion_rms": float(leakage[ordinal]),
            }
            for ordinal in range(24)
        ],
    }


@torch.no_grad()
def profile_success_guard_application(
    local_objectives: Sequence[Any],
    full_motion: torch.Tensor,
    context: DistributedContext,
) -> dict[str, Any]:
    """Measure continuous and native-Program motion against every success guard."""

    local_rows = []
    for objective in local_objectives:
        guards = objective.retention_program_cotangents
        before = objective.program_before
        if guards is None or before is None:
            raise ExpertManifoldError("OSG-PC profile lost retention cotangents")
        continuous = full_motion[objective.task.ordinal]
        if continuous.shape != before.shape:
            raise ExpertManifoldError("OSG-PC applied guard Program shape changed")
        native = (
            (before + continuous.to(dtype=before.dtype)).to(dtype=torch.float32)
            - before.to(dtype=torch.float32)
        )
        continuous64 = continuous.flatten().to(dtype=torch.float64)
        native64 = native.flatten().to(dtype=torch.float64)
        blind = (-objective.source_program_cotangent).flatten().to(dtype=torch.float64)
        safe = (-objective.program_cotangent).flatten().to(dtype=torch.float64)
        continuous_tolerance = 64 * torch.finfo(torch.float32).eps * max(
            float(torch.linalg.vector_norm(continuous64)),
            torch.finfo(torch.float64).tiny,
        )
        native_tolerance = 64 * torch.finfo(torch.float32).eps * max(
            float(torch.linalg.vector_norm(native64)),
            torch.finfo(torch.float64).tiny,
        )
        desired_values = []
        continuous_values = []
        native_values = []
        for guard in guards:
            normalized = guard.flatten().to(dtype=torch.float64)
            normalized = normalized / torch.linalg.vector_norm(normalized)
            desired_values.append(float(torch.dot(normalized, safe)))
            continuous_values.append(float(torch.dot(normalized, continuous64)))
            native_values.append(float(torch.dot(normalized, native64)))
        blind_energy = torch.dot(blind, blind)
        continuous_source_ratio = (
            float(torch.dot(blind, continuous64) / blind_energy)
            if float(blind_energy) > 0
            else 1.0
        )
        native_source_ratio = (
            float(torch.dot(blind, native64) / blind_energy)
            if float(blind_energy) > 0
            else 1.0
        )
        local_rows.append(
            {
                "task_ordinal": objective.task.ordinal,
                "constraint_count": len(guards),
                "continuous_tolerance": continuous_tolerance,
                "native_program_tolerance": native_tolerance,
                "desired_maximum_constraint_value": (
                    max(desired_values) if desired_values else 0.0
                ),
                "continuous_maximum_constraint_value": (
                    max(continuous_values) if continuous_values else 0.0
                ),
                "native_program_maximum_constraint_value": (
                    max(native_values) if native_values else 0.0
                ),
                "continuous_violating_constraints": sum(
                    value > continuous_tolerance for value in continuous_values
                ),
                "native_program_violating_constraints": sum(
                    value > native_tolerance for value in native_values
                ),
                "continuous_source_descent_ratio": continuous_source_ratio,
                "native_program_source_descent_ratio": native_source_ratio,
            }
        )
    gathered: list[Any] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(gathered, local_rows)
    else:
        gathered[0] = local_rows
    rows = [dict(value) for rank_rows in gathered for value in rank_rows]
    rows.sort(key=lambda value: int(value["task_ordinal"]))
    if len(rows) != 24 or [int(value["task_ordinal"]) for value in rows] != list(
        range(24)
    ):
        raise ExpertManifoldError("OSG-PC applied guard profile lost train24")
    return {
        "task_count": len(rows),
        "constraint_count": sum(int(value["constraint_count"]) for value in rows),
        "continuous_violating_constraint_count": sum(
            int(value["continuous_violating_constraints"]) for value in rows
        ),
        "native_program_violating_constraint_count": sum(
            int(value["native_program_violating_constraints"]) for value in rows
        ),
        "maximum_continuous_constraint_value": max(
            float(value["continuous_maximum_constraint_value"]) for value in rows
        ),
        "maximum_native_program_constraint_value": max(
            float(value["native_program_maximum_constraint_value"])
            for value in rows
        ),
        "minimum_continuous_source_descent_ratio": min(
            float(value["continuous_source_descent_ratio"]) for value in rows
        ),
        "minimum_native_program_source_descent_ratio": min(
            float(value["native_program_source_descent_ratio"]) for value in rows
        ),
        "rows": rows,
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
        raise ExpertManifoldError("fixed-action profile output changed")
    return action.detach()


def base_versions(runtime: V6PriorRuntime) -> tuple[tuple[str, int], ...]:
    base = runtime.writer.base_writer
    return tuple(
        (name, value._version)
        for name, value in (*base.named_parameters(), *base.named_buffers())
    )


def profile_passes(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    gates = config["profile_run"]["gates"]
    update = row["update"]
    application = row["application"]
    response = row["lora_response"]
    task_local = row["task_local_motion"]
    guard_application = row["success_guard_application"]
    if not all(
        isinstance(value, Mapping)
        for value in (update, application, response, task_local, guard_application)
    ):
        raise ExpertManifoldError("mechanism profile evidence is incomplete")
    values = _profile_values(config, row, update, application, response)
    checks = _profile_checks(gates, task_local, response, row, values)
    return all(checks.values()), {
        "checks": checks,
        "success_guard": dict(row["success_guard"]),
        "success_guard_application": dict(guard_application),
        "feature_rank": int(update["feature_rank"]),
        "regularized_gram_condition_number": float(
            update["regularized_gram_condition_number"]
        ),
        "correct_motion_to_cotangent_rms": values["retained"],
        "negative_to_correct_motion_rms": values["negative_ratio"],
        "negative_null_per_kind": _negative_null_per_kind(row, gates),
        "production_kernel_wall_fraction": values["kernel_fraction"],
        "production_wall_ratio_to_sealed_v6": values["production_ratio"],
    }


def _profile_values(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    update: Mapping[str, Any],
    application: Mapping[str, Any],
    response: Mapping[str, Any],
) -> dict[str, float]:
    result = {
        "cotangent": float(update["correct_cotangent_rms"]),
        "predicted_correct": float(update["predicted_correct_motion_rms"]),
        "negative_ratio": float(update["predicted_negative_to_correct_ratio"]),
        "closure": float(application["predicted_observed_relative_rms"]),
        "lora_a": float(response["lora_a_response_rms"]),
        "lora_b": float(response["lora_b_response_rms"]),
        "fixed_action": float(response["fixed_action_response_rms"]),
    }
    result["retained"] = (
        result["predicted_correct"] / result["cotangent"]
        if result["cotangent"] > 0
        else 0.0
    )
    result["production"] = float(row["profile_task_seconds"]) + float(
        row["production_kernel_seconds"]
    )
    result["kernel_fraction"] = float(row["production_kernel_seconds"]) / max(
        result["production"], torch.finfo(torch.float64).tiny
    )
    result["production_ratio"] = result["production"] / float(
        config["profile_run"]["throughput_baseline"]["step_seconds"]
    )
    result["finite"] = float(all(math.isfinite(value) for value in result.values()))
    return result


def _negative_null_per_kind(
    row: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, int]:
    task_rows = row.get("task_local_motion", {}).get("rows", ())
    leakage_by_ordinal = {
        int(value.get("task_ordinal", -1)): float(
            value.get("negative_to_correct_motion_rms", math.inf)
        )
        for value in task_rows
    }
    return {
        kind: sum(
            record.get("counterfactual_kind") == kind
            and leakage_by_ordinal.get(int(record.get("task_ordinal", -1)), math.inf)
            <= float(gates["negative_to_correct_motion_rms_max"])
            for record in row.get("task_records", ())
        )
        for kind in ("reversed", "shuffled", "wrong")
    }


def _profile_checks(
    gates: Mapping[str, Any],
    task_local: Mapping[str, Any],
    response: Mapping[str, Any],
    row: Mapping[str, Any],
    values: Mapping[str, float],
) -> dict[str, bool]:
    task_rows = task_local.get("rows", ())
    task_records = row.get("task_records", ())
    per_kind_null = _negative_null_per_kind(row, gates)
    guard = row.get("success_guard", {})
    guard_application = row.get("success_guard_application", {})
    guarded_by_suite = guard.get("guarded_tasks_per_suite", {})
    success_program_cotangents = [
        float(value)
        for record in task_records
        for value in record.get("retention_credit", {}).get(
            "program_cotangent_rms", ()
        )
    ]
    exact_fallback = all(
        not bool(record.get("guard_projection", {}).get("changed"))
        for record in task_records
        if int(record.get("retention_credit", {}).get("successes", -1)) == 0
        or bool(record.get("guard_projection", {}).get("raw_feasible"))
    )
    return {
        "feature_rank": int(row["update"]["feature_rank"])
        >= int(gates["feature_rank_min"]),
        "feature_condition": float(row["update"]["regularized_gram_condition_number"])
        <= float(gates["regularized_gram_condition_number_max"]),
        "correct_motion_retained": values["retained"]
        >= float(gates["correct_motion_to_cotangent_rms_min"]),
        "counterfactual_null": math.isfinite(values["negative_ratio"])
        and values["negative_ratio"]
        <= float(gates["negative_to_correct_motion_rms_max"]),
        "predicted_observed_closure": math.isfinite(values["closure"])
        and values["closure"] <= float(gates["predicted_observed_relative_rms_max"]),
        "production_wall_overhead": values["production_ratio"]
        <= float(gates["production_wall_ratio_max"]),
        "lora_a_response": math.isfinite(values["lora_a"])
        and values["lora_a"] > float(gates["lora_a_response_rms_min"]),
        "lora_b_response": math.isfinite(values["lora_b"])
        and values["lora_b"] > float(gates["lora_b_response_rms_min"]),
        "fixed_action_response": math.isfinite(values["fixed_action"])
        and values["fixed_action"] > float(gates["fixed_action_response_rms_min"])
        and int(response["fixed_action_probe_task_count"])
        == int(gates["fixed_action_probe_task_count"])
        and int(response["fixed_action_probe_policy_forwards"])
        == 2 * int(gates["fixed_action_probe_task_count"]),
        "fixed_action_breadth": int(response["fixed_action_passing_task_count"])
        >= int(gates["fixed_action_passing_task_count_min"]),
        "task_local_motion_evidence": int(task_local.get("task_count", -1)) == 24
        and len(task_rows) == 24
        and [int(value.get("task_ordinal", -1)) for value in task_rows]
        == list(range(24))
        and int(task_local.get("correct_retained_passing_tasks", -1))
        >= int(gates["correct_retained_task_count_min"])
        and int(task_local.get("negative_null_passing_tasks", -1))
        >= int(gates["negative_null_task_count_min"])
        and len(task_records) == 24
        and all(
            count >= int(gates["negative_null_per_kind_min"])
            for count in per_kind_null.values()
        ),
        "functional_policy_program_credit": math.isfinite(values["cotangent"])
        and values["cotangent"] > 0,
        "full24_success_guard": (
            len(task_records) == int(gates["task_count"])
            and sum(
                int(value.get("historical_v6_video_encodes", -1))
                for value in task_records
            )
            == int(gates["video_count"])
            and sum(
                int(value.get("source_action_queries", -1))
                for value in task_records
            )
            == int(gates["source_action_query_count"])
            and int(guard.get("rollouts", -1)) == int(gates["rollout_count"])
            and int(guard.get("guarded_tasks", -1))
            >= int(gates["guarded_task_count_min"])
            and int(guard.get("all_success_tasks", -1))
            >= int(gates["all_success_task_count_min"])
            and isinstance(guarded_by_suite, Mapping)
            and set(guarded_by_suite)
            == {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
            and all(
                int(value) >= int(gates["guarded_task_per_suite_min"])
                for value in guarded_by_suite.values()
            )
            and len(success_program_cotangents)
            == int(guard.get("success_episodes", -1))
            and all(
                int(record.get("retention_credit", {}).get("flow_panel_chunks", 0))
                >= int(record.get("retention_credit", {}).get("replay_chunks", 0))
                and len(
                    record.get("retention_credit", {}).get(
                        "flow_panel_row_indices", ()
                    )
                )
                == int(record.get("retention_credit", {}).get("replay_chunks", -1))
                for record in task_records
            )
            and all(
                math.isfinite(value) and value > 0
                for value in success_program_cotangents
            )
            == bool(gates["success_program_cotangent_nonzero"])
            and int(guard.get("failure_replay_gradient_episodes", -1))
            == int(gates["failure_replay_gradient_episodes"])
            and int(guard.get("projection_changed_tasks", -1))
            >= int(gates["projection_changed_task_count_min"])
            and exact_fallback == bool(gates["exact_blind_fallback_required"])
            and math.isfinite(float(guard.get("maximum_constraint_value", math.inf)))
            and float(guard.get("minimum_source_descent_ratio", -math.inf)) >= 0
        ),
        "full48_applied_success_guard": (
            isinstance(guard_application, Mapping)
            and int(guard_application.get("task_count", -1))
            == int(gates["task_count"])
            and int(guard_application.get("constraint_count", -1))
            == int(guard.get("success_episodes", -2))
            and int(
                guard_application.get("continuous_violating_constraint_count", -1)
            )
            >= 0
            and int(
                guard_application.get(
                    "native_program_violating_constraint_count", -1
                )
            )
            >= 0
            and math.isfinite(
                float(
                    guard_application.get(
                        "maximum_continuous_constraint_value", math.inf
                    )
                )
            )
            and math.isfinite(
                float(
                    guard_application.get(
                        "maximum_native_program_constraint_value", math.inf
                    )
                )
            )
            and math.isfinite(
                float(
                    guard_application.get(
                        "minimum_continuous_source_descent_ratio", -math.inf
                    )
                )
            )
            and math.isfinite(
                float(
                    guard_application.get(
                        "minimum_native_program_source_descent_ratio", -math.inf
                    )
                )
            )
            and bool(gates["applied_guard_evidence_required"])
        ),
        "negative_policy_forwards": int(row["negative_policy_forwards"]) == 0,
        "oom_and_nonfinite": int(row["oom_count"]) == int(gates["oom_count"])
        and int(row["nonfinite_count"]) == int(gates["nonfinite_count"])
        and bool(values["finite"]),
    }
