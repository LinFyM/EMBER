"""Direct functional E1 training for PNBTT free-query transport capacity."""

from __future__ import annotations

import math
import time
from typing import Any, Mapping

import torch
import torch.distributed as dist

from ember.ecp.checkpoint import save_ecp_checkpoint
from ember.ecp.joint_program_primal.pnbtt_runtime import (
    PNBTT_E1_STAGE,
    PNBTT_TASKLOCAL_RUN_SCHEMA,
    PNBTTTaskLocalRuntime,
    prepare_pnbtt_tasklocal_runtime,
)
from ember.ecp.joint_program_primal.pnbtt_policy_distance import (
    paired_policy_velocity_distance_gradient,
)
from ember.ecp.joint_program_primal.train_step import (
    functional_loss_derivative,
    functional_panel_batch,
)
from ember.ecp.bank_conditioning.tangent_transport import (
    TangentTransportVideo,
    pnbtt_event_weights,
)
from ember.ecp.joint_program_primal.pnbtt_tasklocal import (
    PreparedPNBTTArm,
    carrier_rank16,
    generated_rank16,
    local_tasks,
    prepare_e1_arms,
)
from ember.ecp.stage0_train_step import _gather_records
from ember.pi05_source_checkpoint import write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import initialize_distributed
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX
from ember.writer.functional import (
    ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    LATIN_BETA_TIME_SAMPLING_SCHEME,
)


def _functional_policy_microbatch_size(
    runtime: PNBTTTaskLocalRuntime, task: int
) -> int:
    by_task = runtime.config["optimization"].get(
        "functional_policy_microbatch_size_by_task", {}
    )
    return int(
        by_task.get(
            str(task),
            runtime.config["optimization"]["functional_policy_microbatch_size"],
        )
    )


def _generated_functional_state(
    runtime: PNBTTTaskLocalRuntime,
    *,
    task: int,
    arm: PreparedPNBTTArm,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    with torch.no_grad():
        # Keep the asymmetric raw factor gauge during optimization: exact-zero
        # balanced SVD would zero both factors and block the first functional
        # gradient.  Formal materialization/evaluation canonicalizes the same
        # B@A update through the required small-core balanced SVD.
        state, output = generated_rank16(
            runtime, task=task, arm=arm, canonicalize=False
        )
        diagnostics = {
            "conditioning": output.conditioning_metrics.detach().float().cpu().tolist(),
            "residual_a_rms": float(
                torch.stack(
                    tuple(value.detach().float().square().mean() for value in output.residual.a)
                ).mean().sqrt()
            ),
            "residual_b_rms": float(
                torch.stack(
                    tuple(value.detach().float().square().mean() for value in output.residual.b)
                ).mean().sqrt()
            ),
        }
    return state, diagnostics


def _functional_leaf_derivative(
    runtime: PNBTTTaskLocalRuntime,
    *,
    task: int,
    state: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    seed: int,
) -> tuple[float, dict[str, torch.Tensor]]:
    runtime.config["optimization"]["functional_policy_microbatch_size"] = (
        _functional_policy_microbatch_size(runtime, task)
    )
    loss, gradients = functional_loss_derivative(
        runtime,
        state=state,
        batch=batch,
        policy_rng_seed=seed,
    )
    return loss, gradients


def _functional_derivative(
    runtime: PNBTTTaskLocalRuntime,
    *,
    task: int,
    arm: PreparedPNBTTArm,
    batch: Mapping[str, Any],
    seed: int,
) -> tuple[float, dict[str, torch.Tensor], dict[str, Any]]:
    state, diagnostics = _generated_functional_state(
        runtime, task=task, arm=arm
    )
    loss, gradients = _functional_leaf_derivative(
        runtime, task=task, state=state, batch=batch, seed=seed
    )
    return loss, gradients, diagnostics


def _chain_rule_backward(
    runtime: PNBTTTaskLocalRuntime,
    *,
    task: int,
    arm: PreparedPNBTTArm,
    gradients: Mapping[str, torch.Tensor],
    weight: float,
) -> None:
    if weight == 0.0:
        return
    if runtime.free_query is None or not math.isfinite(weight):
        raise RuntimeError("PNBTT E1 targetwise chain-rule weight changed")
    tangent_videos = tuple(
        TangentTransportVideo(native=video, context=context)
        for video, context in zip(arm.videos, arm.bank_contexts, strict=True)
    )
    event_weights = pnbtt_event_weights(arm.program)
    residual_rank = runtime.compiler.residual_rank
    residual_offset = 12 if residual_rank == 4 else 0
    if residual_rank not in (4, 16) or residual_offset + residual_rank != 16:
        raise RuntimeError("PNBTT E1 rank allocation changed")
    with runtime.compiler.bank_operator.ieee_matmul(runtime.context.device):
        for target, contract_target in enumerate(runtime.ranks.contract.targets):
            a, b, _ = runtime.compiler.tangent_transport.forward_target(
                target=target,
                target_queries=runtime.free_query.target(task, target),
                videos=tangent_videos,
                event_weights=event_weights,
                s_ref=runtime.ranks.s_ref[target],
            )
            a_gradient = gradients[contract_target.name + LORA_A_SUFFIX][
                residual_offset : residual_offset + residual_rank
            ]
            b_gradient = gradients[contract_target.name + LORA_B_SUFFIX][
                :, residual_offset : residual_offset + residual_rank
            ]
            if a_gradient.shape != a.shape or b_gradient.shape != b.transpose(0, 1).shape:
                raise RuntimeError("PNBTT E1 targetwise leaf gradient changed")
            surrogate = ((a - a.detach()) * a_gradient.to(a)).sum()
            surrogate = surrogate + (
                (b - b.detach()) * b_gradient.transpose(0, 1).to(b)
            ).sum()
            (surrogate * float(weight)).backward()


def _task_record(
    runtime: PNBTTTaskLocalRuntime,
    *,
    task: int,
    arms: Mapping[str, PreparedPNBTTArm],
    visit_index: int,
) -> dict[str, Any]:
    tick = time.monotonic()
    batch, panel = functional_panel_batch(
        runtime,
        task_id=task,
        panel_name="a",
        visit_index=visit_index,
    )
    evaluated: dict[str, tuple[float, dict[str, torch.Tensor], dict[str, Any]]] = {}
    for name in ("correct_fit0", "correct_fit1"):
        evaluated[name] = _functional_derivative(
            runtime,
            task=task,
            arm=arms[name],
            batch=batch,
            seed=panel.policy_rng_seed,
        )
    wrong_state, wrong_diagnostics = _generated_functional_state(
        runtime, task=task, arm=arms["wrong_fit0"]
    )
    wrong_loss, wrong_gradients = _functional_leaf_derivative(
        runtime,
        task=task,
        state=wrong_state,
        batch=batch,
        seed=panel.policy_rng_seed,
    )
    evaluated["wrong_fit0"] = (
        wrong_loss,
        wrong_gradients,
        wrong_diagnostics,
    )
    unrelated_task = int(
        runtime.config["task_local"]["preservation_only_task_by_task"][str(task)]
    )
    unrelated_batch, unrelated_panel = functional_panel_batch(
        runtime,
        task_id=unrelated_task,
        panel_name="a",
        visit_index=visit_index,
    )
    velocity_cache = getattr(runtime, "pnbtt_carrier_velocity_cache", None)
    if velocity_cache is None:
        velocity_cache = {}
        setattr(runtime, "pnbtt_carrier_velocity_cache", velocity_cache)
    cache_key = (
        unrelated_task,
        visit_index % len(runtime.panels[unrelated_task].panel_a),
        int(unrelated_panel.policy_rng_seed),
    )
    unrelated_distance, unrelated_gradients, carrier_velocity = (
        paired_policy_velocity_distance_gradient(
            runtime.policy,
            wrong_state,
            carrier_rank16(runtime),
            runtime.ranks.contract,
            batch=unrelated_batch,
            policy_rng_seed=unrelated_panel.policy_rng_seed,
            policy_rng_device=runtime.context.device,
            flow_time_sampling_scheme=LATIN_BETA_TIME_SAMPLING_SCHEME,
            flow_noise_sampling_scheme=ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
            policy_microbatch_size=_functional_policy_microbatch_size(
                runtime, task
            ),
            cached_carrier_velocity=velocity_cache.get(cache_key),
        )
    )
    velocity_cache[cache_key] = carrier_velocity

    correct0 = evaluated["correct_fit0"][0]
    correct1 = evaluated["correct_fit1"][0]
    wrong = evaluated["wrong_fit0"][0]
    loss_config = runtime.config["optimization"]["joint"]["functional_objective"]
    margin_scale = float(runtime.margin_scales[task])
    necessity_margin = float(loss_config["normalized_necessity_margin"]) * margin_scale
    preservation_tolerance = (
        float(loss_config["normalized_preservation_tolerance"]) * margin_scale
    )
    necessity_hinge = max(0.0, necessity_margin + correct0 - wrong)
    target_preservation_hinge = max(
        0.0, wrong - (float(panel.flow_loss) + preservation_tolerance)
    )
    necessity_active = necessity_hinge > 0.0
    target_preservation_active = target_preservation_hinge > 0.0
    unrelated_preservation_active = unrelated_distance > 0.0

    task_weight = 1.0 / len(runtime.config["task_local"]["task_ids"])
    correct_weight = float(loss_config["correct_weight"]) * 0.5 * task_weight
    necessity_weight = float(loss_config["necessity_weight"]) * task_weight
    preservation_weight = float(loss_config["preservation_weight"]) * task_weight
    _chain_rule_backward(
        runtime,
        task=task,
        arm=arms["correct_fit0"],
        gradients=evaluated["correct_fit0"][1],
        weight=correct_weight + (necessity_weight if necessity_active else 0.0),
    )
    _chain_rule_backward(
        runtime,
        task=task,
        arm=arms["correct_fit1"],
        gradients=evaluated["correct_fit1"][1],
        weight=correct_weight,
    )
    wrong_weight = (
        (-necessity_weight if necessity_active else 0.0)
        + (0.5 * preservation_weight if target_preservation_active else 0.0)
    )
    _chain_rule_backward(
        runtime,
        task=task,
        arm=arms["wrong_fit0"],
        gradients=evaluated["wrong_fit0"][1],
        weight=wrong_weight,
    )
    _chain_rule_backward(
        runtime,
        task=task,
        arm=arms["wrong_fit0"],
        gradients=unrelated_gradients,
        weight=0.5 * preservation_weight,
    )

    recoveries = {
        name: (float(panel.flow_loss) - values[0]) / margin_scale
        for name, values in evaluated.items()
    }
    return {
        "authority_id": task,
        "role": runtime.panels[task].role,
        "panel": "a",
        "panel_visit": visit_index,
        "functional_policy_rng_seed": panel.policy_rng_seed,
        "action_demos": list(panel.action_demos),
        "action_frames": list(panel.action_frames),
        "carrier_loss": float(panel.flow_loss),
        "margin_scale": margin_scale,
        "losses": {name: values[0] for name, values in evaluated.items()},
        "functional_recovery": recoveries,
        "mean_correct_functional_loss": 0.5 * (correct0 + correct1),
        "wrong_functional_loss": wrong,
        "necessity": {
            "margin": necessity_margin,
            "wrong_minus_correct0": wrong - correct0,
            "hinge": necessity_hinge,
            "active": necessity_active,
        },
        "preservation": {
            "active": target_preservation_active or unrelated_preservation_active,
            "loss": 0.5 * (target_preservation_hinge + unrelated_distance),
            "target": {
                "carrier_loss": float(panel.flow_loss),
                "generated_loss": wrong,
                "upper_loss": float(panel.flow_loss) + preservation_tolerance,
                "hinge": target_preservation_hinge,
                "active": target_preservation_active,
            },
            "unrelated": {
                "authority_id": unrelated_task,
                "objective": "paired_generated_to_carrier_policy_velocity_mse",
                "policy_velocity_distance": unrelated_distance,
                "active": unrelated_preservation_active,
                "policy_rng_seed": unrelated_panel.policy_rng_seed,
                "action_demos": list(unrelated_panel.action_demos),
                "action_frames": list(unrelated_panel.action_frames),
            },
        },
        "diagnostics": {
            name: values[2] for name, values in evaluated.items()
        },
        "video_demos": {name: arms[name].video_demo for name in evaluated},
        "task_seconds": time.monotonic() - tick,
    }


def _allreduce_gradients(runtime: PNBTTTaskLocalRuntime) -> None:
    local = tuple(parameter.grad is not None for parameter in runtime.trainable_parameters)
    if not all(local):
        missing = sum(not value for value in local)
        raise RuntimeError(f"PNBTT E1 lost {missing} trainable gradients")
    if runtime.context.world_size <= 1:
        return
    rows: list[Any] = [None] * runtime.context.world_size
    dist.all_gather_object(rows, local)
    if any(row != local for row in rows):
        raise RuntimeError("PNBTT E1 ranks disagreed on gradient ownership")
    for parameter in runtime.trainable_parameters:
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def _gradient_metrics(runtime: PNBTTTaskLocalRuntime) -> dict[str, Any]:
    free = runtime.free_query
    if free is None or free.raw_query.grad is None:
        raise RuntimeError("PNBTT E1 free-query gradient is absent")
    query_norms = free.raw_query.grad.detach().float().flatten(1).norm(dim=1)
    key_gradients = tuple(
        parameter.grad.detach().float().norm()
        for parameter in runtime.compiler.tangent_transport.key_encoder.parameters()
        if parameter.grad is not None
    )
    if not key_gradients:
        raise RuntimeError("PNBTT E1 key projection gradient is absent")
    key_norm = torch.stack(key_gradients).square().sum().sqrt()
    if not bool(torch.isfinite(query_norms).all()) or not bool(torch.isfinite(key_norm)):
        raise RuntimeError("PNBTT E1 gradient became non-finite")
    return {
        "free_query_by_task": {
            str(task): float(value)
            for task, value in zip(free.task_ids, query_norms, strict=True)
        },
        "shared_key_projection": float(key_norm),
    }


def run_e1_optimizer_step(
    runtime: PNBTTTaskLocalRuntime,
    arms: Mapping[int, Mapping[str, PreparedPNBTTArm]],
) -> dict[str, Any]:
    if runtime.config["stage"] != PNBTT_E1_STAGE:
        raise RuntimeError("PNBTT E1 trainer received another stage")
    if runtime.context.world_size > 1:
        dist.barrier()
    if runtime.context.device.type == "cuda":
        torch.cuda.synchronize(runtime.context.device)
    tick = time.monotonic()
    runtime.optimizer.zero_grad(set_to_none=True)
    visit = runtime.optimizer_steps % int(runtime.config["data"]["panel_visits"])
    local_records = [
        _task_record(runtime, task=task, arms=arms[task], visit_index=visit)
        for task in local_tasks(runtime)
    ]
    if any(parameter.grad is not None for parameter in runtime.frozen_parameters):
        raise RuntimeError("PNBTT E1 frozen authority accumulated gradients")
    _allreduce_gradients(runtime)
    probes = _gradient_metrics(runtime)
    clip = float(
        runtime.config["optimization"]["joint"]["optimizer"]["gradient_clip_norm"]
    )
    gradient_norm = torch.nn.utils.clip_grad_norm_(runtime.trainable_parameters, clip)
    if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("PNBTT E1 gradient norm is non-finite")
    runtime.optimizer.step()
    runtime.scheduler.step()
    runtime.optimizer_steps += 1
    if runtime.context.device.type == "cuda":
        torch.cuda.synchronize(runtime.context.device)
    seconds = time.monotonic() - tick
    records = sorted(
        _gather_records(local_records, runtime.context.world_size),
        key=lambda row: int(row["authority_id"]),
    )
    expected = tuple(map(int, runtime.config["task_local"]["task_ids"]))
    if tuple(int(row["authority_id"]) for row in records) != expected:
        raise RuntimeError("PNBTT E1 global task equality changed")
    performance = {
        "rank": runtime.context.rank,
        "tasks": list(local_tasks(runtime)),
        "seconds": seconds,
        "max_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(runtime.context.device))
            if runtime.context.device.type == "cuda"
            else 0
        ),
        "max_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(runtime.context.device))
            if runtime.context.device.type == "cuda"
            else 0
        ),
    }
    performance_rows: list[Any] = [None] * runtime.context.world_size
    if runtime.context.world_size > 1:
        dist.all_gather_object(performance_rows, performance)
    else:
        performance_rows[0] = performance
    return {
        "optimizer_step": runtime.optimizer_steps,
        "effective_optimizer_step": max(
            0,
            runtime.optimizer_steps
            - int(runtime.config["optimization"]["joint"]["warmup_optimizer_steps"]),
        ),
        "panel_visit": visit,
        "mean_correct_functional_loss": sum(
            float(row["mean_correct_functional_loss"]) for row in records
        )
        / len(records),
        "mean_wrong_functional_loss": sum(
            float(row["wrong_functional_loss"]) for row in records
        )
        / len(records),
        "active_necessity_fraction": sum(
            bool(row["necessity"]["active"]) for row in records
        )
        / len(records),
        "active_preservation_fraction": sum(
            bool(row["preservation"]["active"]) for row in records
        )
        / len(records),
        "gradient_probe_norms": probes,
        "gradient_norm_before_clip": float(gradient_norm),
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "rank_performance": performance_rows,
        "global_step_seconds": max(float(row["seconds"]) for row in performance_rows),
        "conditions": records,
    }


def train_pnbtt_tasklocal(args: Any) -> None:
    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    runtime: PNBTTTaskLocalRuntime | None = None
    try:
        runtime = prepare_pnbtt_tasklocal_runtime(args, context)
        arms = prepare_e1_arms(runtime)
        started = time.monotonic()
        while runtime.optimizer_steps < runtime.stop_after_step:
            row = run_e1_optimizer_step(runtime, arms)
            row["elapsed_seconds"] = time.monotonic() - started
            if context.is_main:
                append_jsonl(args.output_dir / "metrics.jsonl", row)
                runtime.metrics_rows += 1
                if runtime.optimizer_steps % args.log_every == 0:
                    print(
                        {
                            key: row[key]
                            for key in (
                                "optimizer_step",
                                "effective_optimizer_step",
                                "global_step_seconds",
                                "mean_correct_functional_loss",
                                "mean_wrong_functional_loss",
                                "active_necessity_fraction",
                                "active_preservation_fraction",
                                "gradient_probe_norms",
                                "gradient_norm_before_clip",
                                "next_lr",
                            )
                        },
                        flush=True,
                    )
            if runtime.optimizer_steps in runtime.checkpoint_steps:
                save_ecp_checkpoint(
                    output_dir=args.output_dir,
                    macro=runtime.optimizer_steps,
                    stage=PNBTT_E1_STAGE,
                    context=context,
                    model=runtime.writer_state,
                    optimizer=runtime.optimizer,
                    scheduler=runtime.scheduler,
                    run_contract_schema=PNBTT_TASKLOCAL_RUN_SCHEMA,
                    metrics_rows=runtime.metrics_rows,
                )
        if context.is_main:
            completion = {
                "stage": PNBTT_E1_STAGE,
                "completed_optimizer_steps": runtime.optimizer_steps,
                "completed_effective_steps": max(
                    0,
                    runtime.optimizer_steps
                    - int(
                        runtime.config["optimization"]["joint"][
                            "warmup_optimizer_steps"
                        ]
                    ),
                ),
            }
            write_json_atomic(args.output_dir / "segment_completion.json", completion)
            if runtime.optimizer_steps == max(runtime.checkpoint_steps):
                write_json_atomic(args.output_dir / "completion.json", completion)
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
