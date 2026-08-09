"""Sealed no-update expert-flow teacher viability audit.

The component has one lifecycle: it either supplies the matched flow primitive
to CEFD after both preregistered gates pass, or is removed when the audit fails.
It is not a second training path and never owns deployment behavior.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_run_contract import (
    V6_PRIOR_TEACHER_AUDIT_COMPLETION_SCHEMA,
    V6_PRIOR_TEACHER_AUDIT_SCHEMA,
)
from ember.expert_manifold.v6_prior_step import parameter_gradient_components
from ember.pi05_source_checkpoint import write_json_atomic


_GRADIENT_COMPONENTS = ("positive", "completion", "ranking", "distillation")
_LOSS_NAMES = (
    "expert_target_loss",
    "macro0_target_loss",
    "tangent10_target_loss",
    "distillation_loss",
)


@dataclass(frozen=True)
class TeacherAuditBindings:
    """Frozen access to the canonical training primitives used by the audit."""

    task_objective: Any
    gather_task_records: Any
    runtime_maximums: Any
    component_layout: Any
    component_norms: Any


def teacher_audit_task_record(value: Any) -> dict[str, Any]:
    audit = value.flow_audit
    if audit is None:
        raise ExpertManifoldError("flow-teacher task record is missing")
    losses = (
        torch.stack(
            (
                audit.expert_target_loss,
                audit.student_target_loss,
                audit.comparison_target_loss,
                audit.distillation_loss,
            )
        )
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .tolist()
    )
    if not all(math.isfinite(item) for item in losses):
        raise ExpertManifoldError("flow-teacher task loss is non-finite")
    expert, macro0, tangent10, distillation = losses
    return {
        "task_ordinal": value.task.ordinal,
        "global_task_id": value.task.global_task_id,
        "suite": value.task.suite,
        "task_id": value.task.task_id,
        "task_visit": value.task_visit,
        "teacher_demo": value.teacher_demo,
        "counterfactual_kind": value.counterfactual_kind,
        "expert_target_loss": expert,
        "macro0_target_loss": macro0,
        "tangent10_target_loss": tangent10,
        "distillation_loss": distillation,
        "expert_better_than_macro0": expert < macro0,
        "expert_better_than_tangent10": expert < tangent10,
        "expert_better_than_both": expert < macro0 and expert < tangent10,
        "correct_raw_frames": value.pair.correct_raw_frames,
        "correct_sampled_frames": value.pair.correct_sampled_frames,
    }


def _gradient_block_vector(
    value: torch.Tensor,
    layout: Sequence[tuple[str, int, int]],
    group: str,
) -> torch.Tensor:
    if group == "global":
        return value
    parts = [
        value[start:stop]
        for name, start, stop in layout
        if name.split(".", 1)[0] == group
    ]
    if not parts:
        raise ExpertManifoldError("flow-teacher gradient block is empty")
    return torch.cat(parts)


def gradient_span_relationships(
    accumulators: Mapping[str, torch.Tensor],
    layout: Sequence[tuple[str, int, int]],
    *,
    pinv_rtol: float,
) -> dict[str, Any]:
    """Project the full24 CEFD mean onto the registered existing-gradient span."""

    if set(accumulators) != set(_GRADIENT_COMPONENTS) or not 0 < pinv_rtol < 1:
        raise ExpertManifoldError("flow-teacher gradient component set changed")
    result = {}
    for group in ("compiler", "factor_heads", "global"):
        vectors = {
            name: _gradient_block_vector(accumulators[name], layout, group)
            for name in _GRADIENT_COMPONENTS
        }
        norms = {
            name: torch.linalg.vector_norm(value) for name, value in vectors.items()
        }
        packed_norms = torch.stack(tuple(norms[name] for name in _GRADIENT_COMPONENTS))
        if not bool(torch.isfinite(packed_norms).all()):
            raise ExpertManifoldError("flow-teacher gradient block is non-finite")
        normalized = {
            name: (
                vectors[name] / norms[name]
                if float(norms[name].detach().cpu()) > 0
                else torch.zeros_like(vectors[name])
            )
            for name in _GRADIENT_COMPONENTS
        }
        cosine = torch.stack(
            tuple(
                torch.dot(normalized[left], normalized[right])
                for left in _GRADIENT_COMPONENTS
                for right in _GRADIENT_COMPONENTS
            )
        ).reshape(len(_GRADIENT_COMPONENTS), len(_GRADIENT_COMPONENTS))
        raw_gram = cosine * packed_norms[:, None] * packed_norms[None, :]
        existing = cosine[:3, :3].detach().to(device="cpu", dtype=torch.float64)
        cross = cosine[:3, 3].detach().to(device="cpu", dtype=torch.float64)
        eigenvalues = torch.linalg.eigvalsh(existing)
        threshold = float(eigenvalues.abs().max()) * pinv_rtol
        if float(eigenvalues.min()) < -threshold:
            raise ExpertManifoldError(
                "flow-teacher gradient Gram is not positive semidefinite"
            )
        effective_rank = int((eigenvalues.abs() > threshold).sum().item())
        coefficients = torch.linalg.pinv(
            existing,
            hermitian=True,
            rtol=pinv_rtol,
        ) @ cross
        residual = normalized["distillation"].clone()
        for coefficient, name in zip(
            coefficients.tolist(), _GRADIENT_COMPONENTS[:3], strict=True
        ):
            residual.add_(normalized[name], alpha=-float(coefficient))
        residual_ratio = float(torch.linalg.vector_norm(residual).detach().cpu())
        if not (
            bool(torch.isfinite(coefficients).all())
            and math.isfinite(residual_ratio)
            and 0 <= residual_ratio <= 1.0001
        ):
            raise ExpertManifoldError("flow-teacher gradient projection is invalid")
        result[group] = {
            "component_norms": {
                name: float(norms[name].detach().cpu())
                for name in _GRADIENT_COMPONENTS
            },
            "distillation_nonzero": float(norms["distillation"].detach().cpu()) > 0,
            "cosine_matrix_order": list(_GRADIENT_COMPONENTS),
            "raw_gram_matrix": raw_gram.detach()
            .to(device="cpu", dtype=torch.float32)
            .tolist(),
            "cosine_matrix": cosine.detach()
            .to(device="cpu", dtype=torch.float32)
            .tolist(),
            "existing_span_order": list(_GRADIENT_COMPONENTS[:3]),
            "existing_span_pinv_rtol": pinv_rtol,
            "existing_span_gram_eigenvalues": [
                float(value) for value in eigenvalues.tolist()
            ],
            "existing_span_effective_rank": effective_rank,
            "normalized_projection_coefficients": dict(
                zip(
                    _GRADIENT_COMPONENTS[:3],
                    (float(value) for value in coefficients.tolist()),
                    strict=True,
                )
            ),
            "existing_span_residual_ratio": residual_ratio,
        }
    return result


def _collect_local_components(
    runtime: Any,
    bindings: TeacherAuditBindings,
) -> tuple[dict[str, torch.Tensor], list[Any], float, float]:
    local_tasks = 24 // runtime.context.world_size
    total_parameters = sum(value.numel() for value in runtime.trainable_parameters)
    accumulators = {
        name: torch.zeros(
            total_parameters, dtype=torch.float32, device=runtime.context.device
        )
        for name in _GRADIENT_COMPONENTS
    }
    local_records = []
    input_wait_seconds = 0.0
    started = time.monotonic()
    for microtask in range(local_tasks):
        input_started = time.monotonic()
        batch = next(runtime.iterator)
        input_wait_seconds += time.monotonic() - input_started
        objective = bindings.task_objective(
            runtime,
            macro=runtime.segment.schedule_start_macro,
            microtask=microtask,
            batch=batch,
        )
        if objective.flow_audit is None:
            raise ExpertManifoldError("flow-teacher objective is missing")
        components = parameter_gradient_components(
            pair=objective.pair,
            functional=objective.flow_audit.positive_gradients,
            distillation=objective.flow_audit.distillation_gradients,
            auxiliary=objective.auxiliary,
            parameters=runtime.trainable_parameters,
            completion_only=True,
        )
        rows = {
            "positive": components.positive,
            "completion": components.projection,
            "ranking": components.ranking,
            "distillation": components.distillation,
        }
        _accumulate_task_gradients(accumulators, rows, total_parameters, local_tasks)
        local_records.append(teacher_audit_task_record(objective))
    return accumulators, local_records, started, input_wait_seconds


def _accumulate_task_gradients(
    accumulators: Mapping[str, torch.Tensor],
    rows: Mapping[str, tuple[torch.Tensor, ...] | None],
    total_parameters: int,
    local_tasks: int,
) -> None:
    for name, values in rows.items():
        if values is None:
            raise ExpertManifoldError("flow-teacher gradient component is missing")
        offset = 0
        for value in values:
            stop = offset + value.numel()
            accumulators[name][offset:stop].add_(
                value.reshape(-1).float(), alpha=1.0 / local_tasks
            )
            offset = stop
        if offset != total_parameters:
            raise ExpertManifoldError("flow-teacher gradient layout changed")


def _reduce_global_components(runtime: Any, accumulators: Mapping[str, torch.Tensor]) -> None:
    if any(parameter.grad is not None for parameter in runtime.policy.parameters()):
        raise ExpertManifoldError("flow-teacher audit touched source policy gradients")
    if any(parameter.grad is not None for parameter in runtime.trainable_parameters):
        raise ExpertManifoldError("flow-teacher audit accumulated update gradients")
    packed = torch.stack(tuple(accumulators[name] for name in _GRADIENT_COMPONENTS))
    if runtime.context.world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed.div_(runtime.context.world_size)
    for name, reduced in zip(_GRADIENT_COMPONENTS, packed.unbind(0), strict=True):
        accumulators[name].copy_(reduced)


def _teacher_quality_summary(
    task_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, Any], int, int]:
    aggregate = {
        name: sum(float(row[name]) for row in task_records) / len(task_records)
        for name in _LOSS_NAMES
    }
    per_suite = {}
    for suite in dict.fromkeys(str(row["suite"]) for row in task_records):
        rows = [row for row in task_records if row["suite"] == suite]
        means = {
            name: sum(float(row[name]) for row in rows) / len(rows)
            for name in _LOSS_NAMES
        }
        per_suite[suite] = {
            **means,
            "expert_better_than_both": (
                means["expert_target_loss"] < means["macro0_target_loss"]
                and means["expert_target_loss"] < means["tangent10_target_loss"]
            ),
        }
    task_passes = sum(bool(row["expert_better_than_both"]) for row in task_records)
    suite_passes = sum(
        bool(row["expert_better_than_both"]) for row in per_suite.values()
    )
    return aggregate, per_suite, task_passes, suite_passes


def _assemble_result(
    runtime: Any,
    bindings: TeacherAuditBindings,
    task_records: Sequence[Mapping[str, Any]],
    accumulators: Mapping[str, torch.Tensor],
    *,
    started: float,
    input_wait_seconds: float,
) -> dict[str, Any]:
    layout = bindings.component_layout(runtime)
    norms = {
        name: bindings.component_norms(value, layout)
        for name, value in accumulators.items()
    }
    audit = runtime.config["teacher_audit"]
    relationships = gradient_span_relationships(
        accumulators,
        layout,
        pinv_rtol=float(audit["gradient_span_pinv_rtol"]),
    )
    seconds, allocated, reserved, input_wait = bindings.runtime_maximums(
        runtime.context, started, input_wait_seconds
    )
    aggregate, per_suite, task_passes, suite_passes = _teacher_quality_summary(
        task_records
    )
    teacher_pass = task_passes >= int(
        audit["teacher_quality_min_tasks"]
    ) and suite_passes >= int(audit["teacher_quality_min_suites"])
    residual_minimum = float(audit["gradient_residual_ratio_min"])
    gradient_pass = all(
        relationships[group]["distillation_nonzero"]
        and relationships[group]["existing_span_residual_ratio"] >= residual_minimum
        for group in ("compiler", "factor_heads")
    )
    query_summary = runtime.run_contract["data"]["consumed_schedule"]["query"]
    total_queries = int(query_summary["global_examples"])
    unique_queries = int(query_summary["unique_query_rows"])
    forwards_per_task = int(audit["physical_policy_forwards_per_task"])
    negative_counts = {
        kind: sum(row["counterfactual_kind"] == kind for row in task_records)
        for kind in ("reversed", "shuffled", "wrong")
    }
    task_ordinals = {int(row["task_ordinal"]) for row in task_records}
    suite_task_counts = {
        suite: sum(str(row["suite"]) == suite for row in task_records)
        for suite in per_suite
    }
    if not (
        len(task_records) == 24
        and task_ordinals == set(range(24))
        and sorted(suite_task_counts.values()) == [6, 6, 6, 6]
        and total_queries == unique_queries == 480
        and forwards_per_task == 6
        and negative_counts == {"reversed": 8, "shuffled": 8, "wrong": 8}
        and int(audit["parameter_updates"]) == 0
        and int(audit["rollouts"]) == 0
    ):
        raise ExpertManifoldError("flow-teacher audit panel changed")
    authorize_cefd = teacher_pass and gradient_pass
    return {
        "schema_version": V6_PRIOR_TEACHER_AUDIT_SCHEMA,
        "schedule_macro": runtime.segment.schedule_start_macro,
        "task_count": len(task_records),
        "action_queries_per_task": 20,
        "total_action_queries": total_queries,
        "unique_action_queries": unique_queries,
        "real_action_dimensions": int(audit["real_action_dimensions"]),
        "physical_policy_forwards_per_task": forwards_per_task,
        "total_physical_policy_forwards": forwards_per_task * len(task_records),
        "matched_randomness": dict(
            runtime.config["objective"]["positive_policy_randomness"]
        ),
        "counterfactual_counts": negative_counts,
        "suite_task_counts": suite_task_counts,
        "aggregate_losses": aggregate,
        "per_suite": per_suite,
        "task_records": list(task_records),
        "unweighted_gradient_norms": norms,
        "gradient_relationships": relationships,
        "teacher_quality_gate": {
            "minimum_tasks": int(audit["teacher_quality_min_tasks"]),
            "minimum_suites": int(audit["teacher_quality_min_suites"]),
            "passing_tasks": task_passes,
            "passing_suites": suite_passes,
            "passed": teacher_pass,
        },
        "gradient_nonredundancy_gate": {
            "minimum_residual_ratio": residual_minimum,
            "compiler_residual_ratio": relationships["compiler"][
                "existing_span_residual_ratio"
            ],
            "factor_heads_residual_ratio": relationships["factor_heads"][
                "existing_span_residual_ratio"
            ],
            "passed": gradient_pass,
        },
        "decision": {
            "authorize_cefd": authorize_cefd,
            "reason": (
                "teacher_quality_and_gradient_nonredundancy_passed"
                if authorize_cefd
                else "one_or_more_preregistered_audit_gates_failed"
            ),
        },
        "parameter_updates": 0,
        "rollouts": 0,
        "step_seconds": seconds,
        "input_wait_seconds": input_wait,
        "max_cuda_allocated_bytes": allocated,
        "max_cuda_reserved_bytes": reserved,
        "oom_count": 0,
        "nonfinite_count": 0,
        "content_hash_policy": "disabled_by_owner",
    }


def _completion(runtime: Any) -> dict[str, Any]:
    return {
        "schema_version": V6_PRIOR_TEACHER_AUDIT_COMPLETION_SCHEMA,
        "mode": "teacher-audit",
        "completed_diagnostic_macros": 1,
        "schedule_start_macro": runtime.segment.schedule_start_macro,
        "schedule_stop_macro": runtime.segment.schedule_stop_macro,
        "parameter_updates": 0,
        "rollouts": 0,
        "oom_count": 0,
        "nonfinite_count": 0,
        "content_hash_policy": "disabled_by_owner",
    }


def run_teacher_audit(runtime: Any, bindings: TeacherAuditBindings) -> None:
    """Run one full24 matched diagnostic and write only evidence artifacts."""

    if runtime.comparison_decoder is None:
        raise ExpertManifoldError("flow-teacher comparison decoder is missing")
    accumulators, local_records, started, input_wait = _collect_local_components(
        runtime, bindings
    )
    _reduce_global_components(runtime, accumulators)
    task_records = bindings.gather_task_records(local_records, runtime.context)
    result = _assemble_result(
        runtime,
        bindings,
        task_records,
        accumulators,
        started=started,
        input_wait_seconds=input_wait,
    )
    if runtime.context.is_main:
        write_json_atomic(runtime.args.output_dir / "teacher_audit.json", result)
        write_json_atomic(runtime.args.output_dir / "completion.json", _completion(runtime))
        print(json.dumps(result, sort_keys=True), flush=True)
