"""Functional qualification for candidate-level Program--bank interaction."""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import torch
import torch.distributed as dist

from ember.ecp.checkpoint import save_ecp_checkpoint
from ember.ecp.joint_program_primal.routing_control import (
    PROGRAM_BANK_INTERACTION_RUN_SCHEMA,
    PROGRAM_BANK_INTERACTION_STAGE,
    fixed_routing_program,
    prepare_routing_control_runtime,
)
from ember.ecp.joint_program_primal.train_step import (
    _clip_gradients,
    _gather_records,
    _rank_performance,
    _sum_gradients,
    _task_assignments,
    backward_functional_derivative,
    functional_loss_derivative,
    functional_panel_batch,
    joint_task_group,
    prepare_program_bank_condition,
)
from ember.ecp.native_materialization import (
    compose_rank12_plus_rank4,
    residual_lora_state,
)
from ember.ecp.shared_compiler_data import (
    prepare_joint_program_primal_condition,
    program_bank_contexts,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import initialize_distributed


PROGRAM_BANK_INTERACTION_COMPLETION_SCHEMA = (
    "ember_ecp_program_bank_candidate_interaction_completion_v3"
)


def _wrong_bank_credit(
    *,
    carrier_loss: float,
    wrong_loss: float,
    free_benefit: float,
    epsilon: float,
    weight: float,
) -> dict[str, float | bool]:
    """Return an anchored raw-unit training credit plus reporting metric."""

    denominator = free_benefit + epsilon
    benefit = carrier_loss - wrong_loss
    active = benefit > 0.0
    normalized_benefit = max(0.0, benefit / denominator)
    values = (
        carrier_loss,
        wrong_loss,
        free_benefit,
        epsilon,
        weight,
        denominator,
        benefit,
        normalized_benefit,
    )
    if (
        not all(math.isfinite(value) for value in values)
        or min(free_benefit, epsilon, weight, denominator) <= 0.0
    ):
        raise RuntimeError("interaction wrong-bank credit changed")
    return {
        "benefit": benefit,
        "normalized_benefit": normalized_benefit,
        "active": active,
        "backward_weight": -weight / 6.0 if active else 0.0,
        "legacy_normalized_amplification": 1.0 / denominator,
    }


def _gradient_snapshot(runtime: Any) -> tuple[torch.Tensor, ...]:
    return tuple(
        (
            torch.zeros_like(parameter)
            if parameter.grad is None
            else parameter.grad.detach().clone()
        )
        for parameter in runtime.trainable_parameters
    )


def _branch_gradient_balance(
    runtime: Any,
    *,
    before: tuple[torch.Tensor, ...],
    after_correct: tuple[torch.Tensor, ...],
    after_wrong: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    correct = tuple(right - left for left, right in zip(before, after_correct))
    wrong = tuple(right - left for left, right in zip(after_correct, after_wrong))

    def norm(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return (
            torch.stack(tuple(value.float().square().sum() for value in values))
            .sum()
            .sqrt()
        )

    correct_norm = norm(correct)
    wrong_norm = norm(wrong)
    dot = torch.stack(
        tuple(
            (left.float() * right.float()).sum()
            for left, right in zip(correct, wrong)
        )
    ).sum()
    denominator = correct_norm * wrong_norm
    cosine = dot / denominator if float(denominator) > 0.0 else torch.zeros_like(dot)
    family = {}
    parameter_index = {
        id(parameter): index
        for index, parameter in enumerate(runtime.trainable_parameters)
    }
    for name, head in runtime.compiler.interaction_scorer.correction.items():
        parameter = head[-1].weight
        index = parameter_index[id(parameter)]
        family[name] = {
            "correct": float(correct[index].float().norm()),
            "wrong": float(wrong[index].float().norm()),
        }
    return {
        "correct_norm": float(correct_norm),
        "wrong_norm": float(wrong_norm),
        "wrong_to_correct_norm": float(wrong_norm / correct_norm)
        if float(correct_norm) > 0.0
        else None,
        "cosine": float(cosine),
        "family_final_weight": family,
    }


def _wrong_bank_pair(runtime: Any, task_id: int) -> tuple[int, int]:
    split = runtime.config["task_split"]
    meta = tuple(map(int, split["gradient_meta"]))
    target = tuple(map(int, split["gradient_target"]))
    candidates = meta if task_id in meta else target
    if len(candidates) != 5 or task_id not in candidates:
        raise RuntimeError("interaction wrong-bank role panel changed")
    offset = 1 + ((runtime.optimizer_steps // 2) % 4)
    wrong_task = candidates[(candidates.index(task_id) + offset) % 5]
    return wrong_task, runtime.optimizer_steps % 2


def generated_interaction_rank16(
    runtime: Any,
    *,
    program_task_id: int,
    bank_condition: Any,
    interaction_off: bool = False,
) -> tuple[dict[str, torch.Tensor], Any, dict[str, Any]]:
    """Use one deployment-shaped forward for correct, wrong, and off arms."""

    prepared, metrics = prepare_program_bank_condition(
        runtime,
        language_authority_id=program_task_id,
        bank_condition=bank_condition,
    )
    if prepared.evidence is None or len(prepared.videos) != 1:
        raise RuntimeError("interaction bank lost frozen K1 evidence")
    query_times = torch.linspace(
        0.0,
        1.0,
        runtime.query_points,
        dtype=torch.float32,
        device=runtime.context.device,
    )[None]
    with torch.no_grad():
        _, bank_output = prepare_joint_program_primal_condition(
            program_model=runtime.program,
            condition=prepared,
            query_times=query_times,
        )
    contexts = program_bank_contexts(bank_output, prepared.evidence)
    output = runtime.compiler.forward_compact(
        fixed_routing_program(runtime, program_task_id),
        prepared.videos,
        s_ref=runtime.ranks.s_ref,
        bank_contexts=contexts,
        interaction_off=interaction_off,
    )
    residual = residual_lora_state(
        output.residual, runtime.rank4_contract, canonicalize=False
    )
    complete = compose_rank12_plus_rank4(
        carrier_state=runtime.ranks.carrier_rank12,
        residual_state=residual,
        rank16_contract=runtime.ranks.contract,
    )
    if (
        output.video_weights.shape != (1,)
        or float(output.video_weights[0]) != 1.0
        or len(complete) != runtime.ranks.contract.state_tensor_count
    ):
        raise RuntimeError("interaction forward escaped K1 unique-rank16 contract")
    return complete, output, metrics


def _run_task(runtime: Any, *, task_id: int, visit_index: int) -> dict[str, Any]:
    tick = time.monotonic()
    record_gradient_balance = runtime.optimizer_steps == 0
    before = _gradient_snapshot(runtime) if record_gradient_balance else None
    batch, panel = functional_panel_batch(
        runtime, task_id=task_id, panel_name="a", visit_index=visit_index
    )
    views = []
    for condition in runtime.task_conditions[task_id].fit_views:
        complete, output, metrics = generated_interaction_rank16(
            runtime,
            program_task_id=task_id,
            bank_condition=condition,
        )
        loss, gradients = functional_loss_derivative(
            runtime,
            state=complete,
            batch=batch,
            policy_rng_seed=panel.policy_rng_seed,
        )
        backward_functional_derivative(complete, gradients, weight=1.0 / 12.0)
        views.append(
            {
                "video_demo": condition.video_demo,
                "sampled_frames": condition.sampled_frames,
                "functional_loss": loss,
                "interaction_off": False,
                "solve_metrics": output.solve_metrics.detach().float().cpu().tolist(),
                "conditioning_metrics": output.conditioning_metrics.detach()
                .float()
                .cpu()
                .tolist(),
                "condition_metrics": metrics,
            }
        )
        del complete, output

    after_correct = (
        _gradient_snapshot(runtime) if record_gradient_balance else None
    )

    wrong_task, wrong_view = _wrong_bank_pair(runtime, task_id)
    wrong_condition = runtime.task_conditions[wrong_task].fit_views[wrong_view]
    wrong, wrong_output, wrong_metrics = generated_interaction_rank16(
        runtime,
        program_task_id=task_id,
        bank_condition=wrong_condition,
    )
    wrong_loss, wrong_gradients = functional_loss_derivative(
        runtime,
        state=wrong,
        batch=batch,
        policy_rng_seed=panel.policy_rng_seed,
    )
    cell = runtime.config["optimization"]["joint"][
        "wrong_bank_neutralization"
    ]
    free_benefit = float(runtime.counterfactual_margin_scales[task_id])
    epsilon = float(cell["epsilon"])
    weight = float(cell["weight"])
    credit = _wrong_bank_credit(
        carrier_loss=float(panel.flow_loss),
        wrong_loss=wrong_loss,
        free_benefit=free_benefit,
        epsilon=epsilon,
        weight=weight,
    )
    if credit["active"]:
        backward_functional_derivative(
            wrong,
            wrong_gradients,
            weight=float(credit["backward_weight"]),
        )
    branch_gradient_balance = None
    if record_gradient_balance:
        assert before is not None and after_correct is not None
        branch_gradient_balance = _branch_gradient_balance(
            runtime,
            before=before,
            after_correct=after_correct,
            after_wrong=_gradient_snapshot(runtime),
        )
    wrong_record = {
        "program_task": task_id,
        "bank_task": wrong_task,
        "bank_video_demo": wrong_condition.video_demo,
        "bank_video_view": wrong_view,
        "conditioning_language_task": task_id,
        "functional_policy_rng_seed": panel.policy_rng_seed,
        "carrier_functional_loss": float(panel.flow_loss),
        "wrong_functional_loss": wrong_loss,
        "wrong_benefit_over_carrier": credit["benefit"],
        "free_primal_benefit_denominator": free_benefit,
        "epsilon": epsilon,
        "normalized_benefit_hinge": credit["normalized_benefit"],
        "active": credit["active"],
        "backward_units": "raw_functional_flow_loss",
        "backward_weight": credit["backward_weight"],
        "legacy_normalized_amplification": credit[
            "legacy_normalized_amplification"
        ],
        "branch_gradient_balance": branch_gradient_balance,
        "interaction_off": False,
        "solve_metrics": wrong_output.solve_metrics.detach().float().cpu().tolist(),
        "conditioning_metrics": wrong_output.conditioning_metrics.detach()
        .float()
        .cpu()
        .tolist(),
        "condition_metrics": wrong_metrics,
    }
    del wrong, wrong_output
    return {
        "authority_id": task_id,
        "role": runtime.task_conditions[task_id].fit_views[0].role,
        "panel": "a",
        "panel_visit": visit_index,
        "functional_policy_rng_seed": panel.policy_rng_seed,
        "action_demos": list(panel.action_demos),
        "action_frames": list(panel.action_frames),
        "mean_correct_functional_loss": sum(
            row["functional_loss"] for row in views
        )
        / len(views),
        "wrong_bank": wrong_record,
        "views": views,
        "task_seconds": time.monotonic() - tick,
    }


def _interaction_gradient_probes(runtime: Any) -> dict[str, Any]:
    named = tuple(runtime.writer_state.named_parameters())
    if not named or any(
        not name.startswith("interaction_scorer.") for name, _ in named
    ):
        raise RuntimeError("interaction checkpoint ownership changed")
    gradients = tuple(parameter.grad for _, parameter in named)
    if any(gradient is None for gradient in gradients) or any(
        not bool(torch.isfinite(gradient).all())
        for gradient in gradients
        if gradient is not None
    ):
        raise RuntimeError("interaction gradient is absent or non-finite")
    family_final = {}
    for family, head in runtime.compiler.interaction_scorer.correction.items():
        gradient = head[-1].weight.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("interaction final-layer gradient is absent")
        family_final[family] = float(gradient.float().norm())
    if min(family_final.values()) <= 0.0:
        raise RuntimeError("interaction functional credit missed a target family")
    norms = torch.stack(
        tuple(gradient.float().norm() for gradient in gradients if gradient is not None)
    )
    return {
        "total": float(norms.square().sum().sqrt()),
        "nonzero_parameter_tensors": int(torch.count_nonzero(norms)),
        "parameter_tensors": len(norms),
        "family_final_weight": family_final,
    }


def run_interaction_optimizer_step(runtime: Any) -> dict[str, Any]:
    group = joint_task_group(runtime, runtime.optimizer_steps)
    assignments = _task_assignments(runtime, group)
    visit_index = runtime.optimizer_steps % int(runtime.config["data"]["panel_visits"])
    if runtime.context.world_size > 1:
        dist.barrier()
    torch.cuda.synchronize(runtime.context.device)
    tick = time.monotonic()
    teacher_reads = runtime.native_teachers.tensor_reads
    runtime.optimizer.zero_grad(set_to_none=True)
    local_records = [
        _run_task(runtime, task_id=task, visit_index=visit_index)
        for task in assignments[runtime.context.rank]
    ]
    if runtime.native_teachers.tensor_reads != teacher_reads:
        raise RuntimeError("interaction training read native-factor teachers")
    if any(parameter.grad is not None for parameter in runtime.frozen_parameters):
        raise RuntimeError("interaction training changed a frozen authority")
    _sum_gradients(runtime)
    probes = _interaction_gradient_probes(runtime)
    gradient_norm = _clip_gradients(
        runtime.trainable_parameters,
        maximum=float(
            runtime.config["optimization"]["joint"]["optimizer"][
                "gradient_clip_norm"
            ]
        ),
    )
    runtime.optimizer.step()
    runtime.scheduler.step()
    runtime.optimizer_steps += 1
    torch.cuda.synchronize(runtime.context.device)
    local_seconds = time.monotonic() - tick
    records = sorted(
        _gather_records(local_records, runtime.context.world_size),
        key=lambda row: int(row["authority_id"]),
    )
    role_counts = {
        role: sum(row["role"] == role for row in records)
        for role in ("meta_fit", "target_fit")
    }
    if (
        len(records) != 6
        or role_counts != {"meta_fit": 3, "target_fit": 3}
        or {int(row["authority_id"]) for row in records} != set(group)
    ):
        raise RuntimeError("interaction optimizer lost task-role weight")
    performance = _rank_performance(
        runtime,
        {
            "rank": runtime.context.rank,
            "seconds": local_seconds,
            "tasks": list(assignments[runtime.context.rank]),
            "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
                runtime.context.device
            ),
            "max_cuda_reserved_bytes": torch.cuda.max_memory_reserved(
                runtime.context.device
            ),
        },
    )
    return {
        "optimizer_step": runtime.optimizer_steps,
        "effective_optimizer_step": max(0, runtime.optimizer_steps - 10),
        "panel_visit": visit_index,
        "task_group": list(group),
        "role_counts": role_counts,
        "mean_correct_functional_loss": sum(
            float(row["mean_correct_functional_loss"]) for row in records
        )
        / len(records),
        "mean_wrong_normalized_benefit_hinge": sum(
            float(row["wrong_bank"]["normalized_benefit_hinge"])
            for row in records
        )
        / len(records),
        "active_wrong_bank_tasks": sum(
            bool(row["wrong_bank"]["active"]) for row in records
        ),
        "gradient_probe_norms": probes,
        "gradient_norm_before_clip": gradient_norm,
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "rank_assignments": [list(row) for row in assignments],
        "rank_performance": performance,
        "global_step_seconds": max(float(row["seconds"]) for row in performance),
        "conditions": records,
        "native_teacher_tensor_reads": runtime.native_teachers.tensor_reads,
        "trainable_partition": "interaction_only",
        "fixed_routing_token_training_only": True,
        "wrong_bank_exact_language_fixed": True,
        "deployment_candidate": False,
    }


def _retained_elapsed_seconds(runtime: Any) -> float:
    if runtime.optimizer_steps == 0:
        return 0.0
    rows = []
    with (runtime.args.output_dir / "metrics.jsonl").open(
        "r", encoding="utf-8"
    ) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    elapsed = [float(row.get("elapsed_seconds", math.nan)) for row in rows]
    if (
        len(rows) != runtime.optimizer_steps
        or [int(row.get("optimizer_step", -1)) for row in rows]
        != list(range(1, runtime.optimizer_steps + 1))
        or not all(math.isfinite(value) and value > 0.0 for value in elapsed)
        or any(right < left for left, right in zip(elapsed, elapsed[1:]))
    ):
        raise ValueError("interaction retained timing evidence changed")
    return elapsed[-1]


def _completion_payload(
    runtime: Any,
    *,
    elapsed_seconds: float,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": PROGRAM_BANK_INTERACTION_COMPLETION_SCHEMA,
        "status": status,
        "stage": PROGRAM_BANK_INTERACTION_STAGE,
        "run_contract_schema": PROGRAM_BANK_INTERACTION_RUN_SCHEMA,
        "completed_optimizer_steps": runtime.optimizer_steps,
        "completed_effective_steps": max(0, runtime.optimizer_steps - 10),
        "metrics_rows": runtime.metrics_rows,
        "elapsed_seconds": elapsed_seconds,
        "checkpoint_optimizer_steps": list(runtime.checkpoint_steps),
        "resume_from": (
            str(runtime.args.resume) if runtime.args.resume is not None else None
        ),
        "deployment_candidate": False,
    }


def _validate_final_evidence(runtime: Any, elapsed_seconds: float) -> None:
    rows = []
    with (runtime.args.output_dir / "metrics.jsonl").open(
        "r", encoding="utf-8"
    ) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if (
        runtime.optimizer_steps != 110
        or runtime.metrics_rows != 110
        or len(rows) != 110
        or [int(row.get("optimizer_step", -1)) for row in rows]
        != list(range(1, 111))
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds <= 0.0
        or abs(float(rows[-1].get("elapsed_seconds", math.nan)) - elapsed_seconds)
        > 1e-6
    ):
        raise ValueError("interaction final metrics authority changed")
    for step in runtime.checkpoint_steps:
        root = runtime.args.output_dir / "checkpoints" / f"macro_{step:08d}"
        manifest = read_json(root / "checkpoint_manifest.json")
        if (
            manifest.get("stage") != PROGRAM_BANK_INTERACTION_STAGE
            or manifest.get("run_contract_schema")
            != PROGRAM_BANK_INTERACTION_RUN_SCHEMA
            or int(manifest.get("next_macro", -1)) != step
        ):
            raise ValueError("interaction final checkpoint authority changed")


def train_program_bank_interaction(args: argparse.Namespace) -> None:
    if args.phase != "joint":
        raise ValueError("interaction qualification supports only joint phase")
    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    runtime = None
    try:
        runtime = prepare_routing_control_runtime(args, context)
        elapsed_offset = _retained_elapsed_seconds(runtime)
        started = time.monotonic()
        last_elapsed = elapsed_offset
        while runtime.optimizer_steps < runtime.stop_after_step:
            row = run_interaction_optimizer_step(runtime)
            row["elapsed_seconds"] = elapsed_offset + time.monotonic() - started
            last_elapsed = float(row["elapsed_seconds"])
            if context.is_main:
                append_jsonl(args.output_dir / "metrics.jsonl", row)
                runtime.metrics_rows += 1
                if runtime.optimizer_steps % args.log_every == 0:
                    print(
                        json.dumps(
                            {
                                name: row[name]
                                for name in (
                                    "optimizer_step",
                                    "effective_optimizer_step",
                                    "global_step_seconds",
                                    "mean_correct_functional_loss",
                                    "mean_wrong_normalized_benefit_hinge",
                                    "active_wrong_bank_tasks",
                                    "gradient_norm_before_clip",
                                    "gradient_probe_norms",
                                    "next_lr",
                                    "task_group",
                                    "role_counts",
                                    "rank_assignments",
                                    "native_teacher_tensor_reads",
                                    "elapsed_seconds",
                                )
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            if runtime.optimizer_steps in runtime.checkpoint_steps:
                save_ecp_checkpoint(
                    output_dir=args.output_dir,
                    macro=runtime.optimizer_steps,
                    stage=PROGRAM_BANK_INTERACTION_STAGE,
                    context=context,
                    model=runtime.writer_state,
                    optimizer=runtime.optimizer,
                    scheduler=runtime.scheduler,
                    run_contract_schema=PROGRAM_BANK_INTERACTION_RUN_SCHEMA,
                    metrics_rows=runtime.metrics_rows,
                )
        if context.is_main:
            completion = _completion_payload(
                runtime,
                elapsed_seconds=last_elapsed,
                status="segment_complete",
            )
            write_json_atomic(args.output_dir / "segment_completion.json", completion)
            if runtime.optimizer_steps == max(runtime.checkpoint_steps):
                _validate_final_evidence(runtime, last_elapsed)
                write_json_atomic(
                    args.output_dir / "completion.json",
                    {**completion, "status": "complete"},
                )
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
