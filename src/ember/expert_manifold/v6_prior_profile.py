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
            if objective.program_before is None or objective.correct_lora_before is None:
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
                action_rms = (
                    difference.to(dtype=torch.float32).square().mean().sqrt()
                )
                accumulators[4].add_(
                    difference.to(dtype=torch.float32).square().sum()
                )
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
    if not all(
        isinstance(value, Mapping)
        for value in (update, application, response, task_local)
    ):
        raise ExpertManifoldError("mechanism profile evidence is incomplete")
    values = _profile_values(config, row, update, application, response)
    checks = _profile_checks(gates, task_local, response, row, values)
    return all(checks.values()), {
        "checks": checks,
        "correct_motion_to_cotangent_rms": values["retained"],
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


def _profile_checks(
    gates: Mapping[str, Any],
    task_local: Mapping[str, Any],
    response: Mapping[str, Any],
    row: Mapping[str, Any],
    values: Mapping[str, float],
) -> dict[str, bool]:
    task_rows = task_local.get("rows", ())
    return {
        "feature_rank": int(row["update"]["feature_rank"]
        ) >= int(gates["feature_rank_min"]),
        "correct_motion_retained": values["retained"]
        >= float(gates["correct_motion_to_cotangent_rms_min"]),
        "counterfactual_null": math.isfinite(values["negative_ratio"])
        and values["negative_ratio"]
        <= float(gates["negative_to_correct_motion_rms_max"]),
        "predicted_observed_closure": math.isfinite(values["closure"])
        and values["closure"] <= float(gates["predicted_observed_relative_rms_max"]),
        "production_wall_overhead": values["production_ratio"]
        <= float(gates["production_wall_ratio_max"]),
        "lora_a_response": math.isfinite(values["lora_a"]) and values["lora_a"] > 0,
        "lora_b_response": math.isfinite(values["lora_b"]) and values["lora_b"] > 0,
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
        >= int(gates["negative_null_task_count_min"]),
        "functional_policy_program_credit": math.isfinite(values["cotangent"])
        and values["cotangent"] > 0,
        "negative_policy_forwards": int(row["negative_policy_forwards"]) == 0,
        "oom_and_nonfinite": int(row["oom_count"]) == int(gates["oom_count"])
        and int(row["nonfinite_count"]) == int(gates["nonfinite_count"])
        and bool(values["finite"]),
    }
