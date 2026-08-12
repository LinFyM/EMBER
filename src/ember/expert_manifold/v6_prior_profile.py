"""Mechanism evidence for paired-video joint functional credit."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior import counterfactual_kind
from ember.expert_manifold.v6_prior_runtime import V6PriorRuntime
from ember.lora import copy_task_lora_state_, task_lora_state_dict
from ember.pi05_source_checkpoint import DistributedContext


_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
_TASK_COUNT = 24


def profile_max_seconds(context: DistributedContext, seconds: float) -> float:
    value = torch.tensor(seconds, dtype=torch.float64, device=context.device)
    if context.world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value)


def _rms(value: torch.Tensor) -> float:
    return float(value.to(dtype=torch.float32).square().mean().sqrt())


def _ratio(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return numerator / denominator
    return 0.0 if numerator == 0 else math.inf


def _lora_response_row(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> dict[str, float | int]:
    if set(before) != set(after):
        raise ExpertManifoldError("CGIK-JC profile LoRA topology changed")
    square_terms: dict[str, list[torch.Tensor]] = {
        "lora_a": [],
        "lora_b": [],
        "effective_ba": [],
    }
    counts = {name: 0 for name in square_terms}
    for name, before_a in before.items():
        if not name.endswith(".lora_A.default.weight"):
            continue
        b_name = name.replace(".lora_A.default.weight", ".lora_B.default.weight")
        if b_name not in before:
            raise ExpertManifoldError("CGIK-JC profile lost a LoRA factor pair")
        after_a = after[name]
        before_b = before[b_name]
        after_b = after[b_name]
        values = {
            "lora_a": (after_a - before_a).to(dtype=torch.float32),
            "lora_b": (after_b - before_b).to(dtype=torch.float32),
            "effective_ba": (
                after_b.to(dtype=torch.float32) @ after_a.to(dtype=torch.float32)
                - before_b.to(dtype=torch.float32) @ before_a.to(dtype=torch.float32)
            ),
        }
        for prefix, value in values.items():
            square_terms[prefix].append(value.square().sum())
            counts[prefix] += value.numel()
    if any(value <= 0 for value in counts.values()):
        raise ExpertManifoldError("CGIK-JC profile LoRA response is empty")
    totals = torch.stack(
        [torch.stack(square_terms[name]).sum() for name in square_terms]
    ).detach().cpu().tolist()
    return {
        f"{prefix}_response_rms": math.sqrt(float(total) / counts[prefix])
        for prefix, total in zip(square_terms, totals, strict=True)
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
        raise ExpertManifoldError("CGIK-JC fixed-action profile output changed")
    return action.detach()


def base_versions(runtime: V6PriorRuntime) -> tuple[tuple[str, int], ...]:
    base = runtime.writer.base_writer
    return tuple(
        (name, value._version)
        for name, value in (*base.named_parameters(), *base.named_buffers())
    )


def profile_lora_response(
    runtime: V6PriorRuntime,
    local: Sequence[Any],
    correct_motion: torch.Tensor,
) -> dict[str, Any]:
    """Trace both views in four suites through LoRA, effective BA, and action."""

    if correct_motion.shape[0] != 2 * _TASK_COUNT:
        raise ExpertManifoldError("CGIK-JC profile response topology changed")
    identity = task_lora_state_dict(runtime.policy, clone=True)
    local_rows: list[dict[str, Any]] = []
    try:
        for objective in local:
            ordinal = int(objective.task.ordinal)
            if ordinal not in {0, 6, 12, 18}:
                continue
            if objective.fixed_policy_query is None:
                raise ExpertManifoldError("CGIK-JC profile lost fixed-action query")
            for view_index, (view_name, view) in enumerate(
                (("primary", objective.primary), ("companion", objective.companion))
            ):
                graph = view.profile_graph
                if graph is None or not graph.correct_lora:
                    raise ExpertManifoldError("CGIK-JC profile lost before graph")
                motion = correct_motion[ordinal + view_index * _TASK_COUNT]
                with torch.autocast(
                    device_type=runtime.context.device.type,
                    dtype=torch.bfloat16,
                    enabled=runtime.context.device.type == "cuda",
                ):
                    after_program = graph.base_program_slots + (
                        graph.residual_before + motion.unsqueeze(0)
                    ).to(dtype=graph.base_program_slots.dtype)
                    after = runtime.writer.base_writer.decode_slots(
                        after_program.to(dtype=torch.float32)
                    )
                response = _lora_response_row(graph.correct_lora, after)
                before_action = _fixed_action(
                    runtime,
                    objective.fixed_policy_query,
                    graph.correct_lora,
                    seed=202608120000 + ordinal,
                )
                after_action = _fixed_action(
                    runtime,
                    objective.fixed_policy_query,
                    after,
                    seed=202608120000 + ordinal,
                )
                local_rows.append(
                    {
                        "task_ordinal": ordinal,
                        "suite": objective.task.suite,
                        "view": view_name,
                        "program_motion_rms": _rms(motion),
                        "fixed_action_response_rms": _rms(after_action - before_action),
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
    rows.sort(key=lambda row: (int(row["task_ordinal"]), str(row["view"])))
    expected = {(suite, view) for suite in _SUITES for view in ("primary", "companion")}
    observed = {(str(row["suite"]), str(row["view"])) for row in rows}
    if len(rows) != 8 or observed != expected:
        raise ExpertManifoldError("CGIK-JC profile response lost four-suite paired views")
    return {
        "probe_rows": len(rows),
        "policy_forwards": 2 * len(rows),
        "suite_view_pairs": sorted([list(value) for value in observed]),
        "all_program_motion_nonzero": all(
            float(row["program_motion_rms"]) > 0 for row in rows
        ),
        "all_lora_a_response_nonzero": all(
            float(row["lora_a_response_rms"]) > 0 for row in rows
        ),
        "all_lora_b_response_nonzero": all(
            float(row["lora_b_response_rms"]) > 0 for row in rows
        ),
        "all_effective_ba_response_nonzero": all(
            float(row["effective_ba_response_rms"]) > 0 for row in rows
        ),
        "all_fixed_action_response_nonzero": all(
            float(row["fixed_action_response_rms"]) > 0 for row in rows
        ),
        "rows": rows,
    }


@torch.no_grad()
def profile_task_local_motion(
    cotangents: torch.Tensor,
    full_motion: torch.Tensor,
    schedule_macro: int,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    if cotangents.shape[0] != 48 or full_motion.shape[0] != 96:
        raise ExpertManifoldError("CGIK-JC task-local profile topology changed")
    credit = cotangents.flatten(1).to(dtype=torch.float32)
    correct = full_motion[:48].flatten(1).to(dtype=torch.float32)
    negative = full_motion[48:].flatten(1).to(dtype=torch.float32)
    derivatives = (credit * correct).sum(dim=1)
    primary = derivatives[:24]
    companion = derivatives[24:]
    joint = 0.5 * (primary + companion)
    correct_rms = correct.square().mean().sqrt()
    negative_rms = negative.square().mean().sqrt()
    per_row_negative_ratio = negative.square().mean(dim=1).sqrt() / (
        correct.square().mean(dim=1).sqrt().clamp_min(torch.finfo(torch.float32).tiny)
    )
    kinds = [counterfactual_kind(ordinal, schedule_macro) for ordinal in range(24)]
    maximum = float(gates["negative_to_correct_motion_rms_max"])
    suite_derivatives = {
        suite: float(joint[index * 6 : (index + 1) * 6].sum())
        for index, suite in enumerate(_SUITES)
    }
    rows = [
        {
            "task_ordinal": ordinal,
            "suite": _SUITES[ordinal // 6],
            "counterfactual_kind": kinds[ordinal],
            "primary_directional_derivative": float(primary[ordinal]),
            "companion_directional_derivative": float(companion[ordinal]),
            "joint_directional_derivative": float(joint[ordinal]),
            "both_views_descend": bool(primary[ordinal] < 0 and companion[ordinal] < 0),
            "primary_negative_ratio": float(per_row_negative_ratio[ordinal]),
            "companion_negative_ratio": float(per_row_negative_ratio[24 + ordinal]),
        }
        for ordinal in range(24)
    ]
    return {
        "task_count": 24,
        "view_count": 48,
        "total_directional_derivative": float(joint.sum()),
        "primary_directional_derivative": float(primary.sum()),
        "companion_directional_derivative": float(companion.sum()),
        "suite_joint_directional_derivatives": suite_derivatives,
        "both_view_descent_task_count": sum(row["both_views_descend"] for row in rows),
        "correct_motion_rms": float(correct_rms),
        "negative_motion_rms": float(negative_rms),
        "negative_to_correct_motion_ratio": _ratio(
            float(negative_rms), float(correct_rms)
        ),
        "negative_null_passing_views": int((per_row_negative_ratio <= maximum).sum()),
        "negative_null_per_kind": {
            kind: sum(
                rows[ordinal][f"{view}_negative_ratio"] <= maximum
                for ordinal in range(24)
                if kinds[ordinal] == kind
                for view in ("primary", "companion")
            )
            for kind in ("reversed", "shuffled", "wrong")
        },
        "rows": rows,
    }


def profile_passes(
    config: Mapping[str, Any], row: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    gates = config["profile_run"]["gates"]
    for name in ("update", "application", "task_local_motion", "lora_response"):
        if not isinstance(row.get(name), Mapping):
            raise ExpertManifoldError("CGIK-JC mechanism profile evidence is incomplete")
    update = row["update"]
    application = row["application"]
    task_local = row["task_local_motion"]
    response = row["lora_response"]
    records = row.get("task_records", ())
    phase_rows = row.get("phase_a_task_rows", ())
    task_counts = [int(value) for value in row.get("task_counts_per_rank", ())]
    demos_ok = all(
        record.get("primary", {}).get("demo")
        != record.get("companion", {}).get("demo")
        and record.get("primary", {}).get("demo")
        not in set(record.get("action_query_demos", ()))
        and record.get("companion", {}).get("demo")
        not in set(record.get("action_query_demos", ()))
        and (
            record.get("counterfactual_kind") != "wrong"
            or record.get("primary", {}).get("counterfactual_demo")
            != record.get("companion", {}).get("counterfactual_demo")
        )
        for record in records
    )
    queue_ok = (
        len(phase_rows) == 24
        and sorted(int(value.get("queue_index", -1)) for value in phase_rows)
        == list(range(24))
        and len(task_counts) == int(row["world_size"])
        and sum(task_counts) == 24
        and max(task_counts) <= int(gates["retained_task_cap_max"])
        and float(row.get("queue_claim_seconds", math.inf))
        <= float(gates["queue_claim_seconds_max"])
    )
    checks = {
        "paired_information_wall": len(records) == 24
        and demos_ok
        and int(row.get("correct_condition_rows", -1)) == 48
        and int(row.get("negative_condition_rows", -1)) == 48
        and int(row.get("logical_source_action_queries", -1)) == 960
        and int(row.get("outcome_rollouts", -1)) == 0
        and sum(int(value.get("teacher_action_reads", -1)) for value in records) == 0
        and sum(int(value.get("reward_reads", -1)) for value in records) == 0,
        "work_queue": queue_ok,
        "feature_rank": int(update.get("positive_feature_rank", -1))
        >= int(gates["positive_feature_rank_min"])
        and int(update.get("original_feature_rank", -1))
        >= int(gates["full_feature_rank_min"]),
        "regularized_condition": float(
            update.get("regularized_gram_condition_number", math.inf)
        )
        <= float(gates["regularized_condition_max"]),
        "joint_descent": all(
            float(task_local.get(name, math.inf)) < 0
            for name in (
                "total_directional_derivative",
                "primary_directional_derivative",
                "companion_directional_derivative",
            )
        ),
        "suite_descent": set(
            task_local.get("suite_joint_directional_derivatives", {})
        )
        == set(_SUITES)
        and all(
            float(value) < 0
            for value in task_local.get(
                "suite_joint_directional_derivatives", {}
            ).values()
        ),
        "both_view_descent": int(
            task_local.get("both_view_descent_task_count", -1)
        )
        >= int(gates["both_view_descent_task_count_min"]),
        "negative_null": float(
            task_local.get("negative_to_correct_motion_ratio", math.inf)
        )
        <= float(gates["negative_to_correct_motion_rms_max"])
        and all(
            int(task_local.get("negative_null_per_kind", {}).get(kind, -1))
            >= int(gates["negative_null_per_kind_min"])
            for kind in ("reversed", "shuffled", "wrong")
        ),
        "program_write": float(update.get("value_delta_rms", 0.0)) > 0
        and float(update.get("primary_motion_rms", 0.0)) > 0
        and float(update.get("companion_motion_rms", 0.0)) > 0
        and float(application.get("predicted_observed_relative_rms", math.inf))
        <= float(gates["predicted_observed_relative_rms_max"]),
        "policy_effective_response": int(response.get("probe_rows", -1)) == 8
        and int(response.get("policy_forwards", -1)) == 16
        and all(
            bool(response.get(name))
            for name in (
                "all_program_motion_nonzero",
                "all_lora_a_response_nonzero",
                "all_lora_b_response_nonzero",
                "all_effective_ba_response_nonzero",
                "all_fixed_action_response_nonzero",
            )
        ),
        "wall": float(row.get("step_seconds", math.inf))
        <= float(gates["step_seconds_max"]),
        "oom_and_nonfinite": int(row.get("oom_count", -1)) == 0
        and int(row.get("nonfinite_count", -1)) == 0
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
    return all(checks.values()), {
        "checks": checks,
        "update": dict(update),
        "task_local_motion": dict(task_local),
        "lora_response": dict(response),
        "application": dict(application),
        "step_seconds": float(row["step_seconds"]),
        "phase_a_seconds": float(row["phase_a_seconds"]),
        "world_size": int(row["world_size"]),
        "task_counts_per_rank": task_counts,
    }
