"""Gradient profiling and task-complete optimization for the v6-prior Writer."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.expert_manifold.contract import ExpertManifoldError, ExpertTask
from ember.expert_manifold.effective_objective import (
    EffectiveAuxiliaryGradients,
    effective_auxiliary_output_gradients,
)
from ember.expert_manifold.v6_prior import (
    counterfactual_kind,
    cross_suite_wrong_task,
)
from ember.expert_manifold.v6_prior_checkpoint import save_v6_prior_checkpoint
from ember.expert_manifold.v6_prior_contract import (
    REPO_ROOT,
    V6_PRIOR_CANONICAL_CONFIG,
    load_v6_prior_config,
    suggest_auxiliary_weight,
)
from ember.expert_manifold.v6_prior_policy_batch import (
    policy_rng_seed_for_logical_batch,
)
from ember.expert_manifold.v6_prior_runtime import (
    V6PriorRuntime,
    _cursor_contract,
    _prepare_runtime,
)
from ember.expert_manifold.v6_prior_step import (
    GeneratedCounterfactualPair,
    generate_counterfactual_pair,
    merged_output_gradients,
    parameter_gradient_components,
)
from ember.pi05_source_checkpoint import DistributedContext, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import initialize_distributed
from ember.writer.functional import functional_lora_loss_gradient


V6_PRIOR_COMPLETION_SCHEMA = "ember_pi05_v6_prior_writer_completion_v1"


@dataclass(frozen=True)
class TaskObjective:
    task: ExpertTask
    task_visit: int
    teacher_demo: int
    counterfactual_kind: str
    counterfactual_task: ExpertTask | None
    counterfactual_demo: int | None
    pair: GeneratedCounterfactualPair
    functional_loss: torch.Tensor
    functional_gradients: Mapping[str, torch.Tensor]
    auxiliary: EffectiveAuxiliaryGradients


def _batch_task_id(batch: Mapping[str, Any]) -> int:
    values = batch.get("task_id")
    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise ExpertManifoldError("v6-prior action batch lost task identity")
    unique = values.unique()
    if unique.numel() != 1:
        raise ExpertManifoldError("v6-prior action batch crossed tasks")
    return int(unique.item())


def _target_state(
    runtime: V6PriorRuntime,
    task_ordinal: int,
) -> dict[str, torch.Tensor]:
    return {name: value[task_ordinal] for name, value in runtime.expert_targets.items()}


def _task_objective(
    runtime: V6PriorRuntime,
    *,
    macro: int,
    microtask: int,
    batch: Mapping[str, Any],
) -> TaskObjective:
    task_id, task_visit = runtime.sampler.task_visit_for_step(macro, microtask)
    if _batch_task_id(batch) != task_id:
        raise ExpertManifoldError("v6-prior sampler and action batch disagree")
    task = runtime.task_by_global_id[task_id]
    excluded = runtime.sampler.action_demo_indices_for_task_visit(task_id, task_visit)
    teacher_demo = runtime.video_schedule.demos_for_task_visit(
        task_id,
        task_visit,
        excluded=excluded,
    )[0]
    correct_video = runtime.video_store.load(task_id, teacher_demo)
    kind = counterfactual_kind(task.ordinal, task_visit)
    negative_task = None
    negative_demo = None
    negative_video = None
    if kind == "wrong":
        negative_task = cross_suite_wrong_task(
            runtime.tasks,
            task_ordinal=task.ordinal,
            task_visit=task_visit,
        )
        negative_demo = runtime.video_schedule.demos_for_task_visit(
            negative_task.global_task_id,
            task_visit,
        )[0]
        negative_video = runtime.video_store.load(
            negative_task.global_task_id,
            negative_demo,
        )
    pair = generate_counterfactual_pair(
        writer=runtime.writer,
        policy=runtime.policy,
        correct_video=correct_video,
        counterfactual_video=negative_video,
        language_tokens=runtime.language_tokens[task_id],
        kind=kind,
        counterfactual_seed=int(runtime.config["data"]["counterfactual_seed"]),
        task_ordinal=task.ordinal,
        task_visit=task_visit,
        teacher_demo=teacher_demo,
        device=runtime.context.device,
    )
    policy_rng_seed = policy_rng_seed_for_logical_batch(
        runtime.config,
        batch,
        task_id=task_id,
        task_visit=task_visit,
    )
    policy_batch = runtime.processor.training_batch(dict(batch))
    randomness = runtime.config["objective"]["positive_policy_randomness"]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        functional_loss, _, functional_gradients = functional_lora_loss_gradient(
            runtime.policy,
            pair.correct,
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
    objective = runtime.config["objective"]
    expert_config = objective["expert"]
    rank_config = objective["ranking"]
    auxiliary = effective_auxiliary_output_gradients(
        pair.correct,
        pair.counterfactual,
        _target_state(runtime, task.ordinal),
        runtime.lora_contract,
        norm_weight=float(expert_config["norm_weight"]),
        smooth_l1_beta=float(expert_config["smooth_l1_beta"]),
        required_margin=float(rank_config["required_margin"]),
        temperature=float(rank_config["temperature"]),
    )
    return TaskObjective(
        task=task,
        task_visit=task_visit,
        teacher_demo=teacher_demo,
        counterfactual_kind=kind,
        counterfactual_task=negative_task,
        counterfactual_demo=negative_demo,
        pair=pair,
        functional_loss=functional_loss,
        functional_gradients=functional_gradients,
        auxiliary=auxiliary,
    )


def _task_record(value: TaskObjective) -> dict[str, Any]:
    metric_names = (
        "functional_loss",
        "expert_loss",
        "expert_direction",
        "expert_log_norm",
        "ranking_loss",
        "ranking_margin",
        "correct_expert_cosine",
        "counterfactual_expert_cosine",
        "correct_effective_norm",
        "counterfactual_effective_norm",
        "expert_effective_norm",
    )
    metric_tensors = (
        value.functional_loss,
        value.auxiliary.expert.total,
        value.auxiliary.expert.direction,
        value.auxiliary.expert.log_norm,
        value.auxiliary.ranking.loss,
        value.auxiliary.ranking.margin.mean(),
        value.auxiliary.ranking.correct.cosine.mean(),
        value.auxiliary.ranking.counterfactual.cosine.mean(),
        value.auxiliary.ranking.correct.generated_norm.mean(),
        value.auxiliary.ranking.counterfactual.generated_norm.mean(),
        value.auxiliary.ranking.correct.target_norm.mean(),
    )
    metric_values = (
        torch.stack(
            tuple(item.detach().to(dtype=torch.float32) for item in metric_tensors)
        )
        .to(device="cpu")
        .tolist()
    )
    if not all(math.isfinite(item) for item in metric_values):
        raise ExpertManifoldError("v6-prior task objective metric is non-finite")
    if metric_values[-1] <= 1e-12:
        raise ExpertManifoldError("expert target has zero policy-effective energy")
    return {
        "task_ordinal": value.task.ordinal,
        "global_task_id": value.task.global_task_id,
        "suite": value.task.suite,
        "task_id": value.task.task_id,
        "task_visit": value.task_visit,
        "teacher_demo": value.teacher_demo,
        "counterfactual_kind": value.counterfactual_kind,
        "counterfactual_global_task_id": (
            value.counterfactual_task.global_task_id
            if value.counterfactual_task is not None
            else None
        ),
        "counterfactual_demo": value.counterfactual_demo,
        **dict(zip(metric_names, metric_values, strict=True)),
        "correct_raw_frames": value.pair.correct_raw_frames,
        "correct_sampled_frames": value.pair.correct_sampled_frames,
        "counterfactual_raw_frames": value.pair.counterfactual_raw_frames,
        "counterfactual_sampled_frames": value.pair.counterfactual_sampled_frames,
    }


def _gather_task_records(
    local: list[dict[str, Any]],
    context: DistributedContext,
) -> list[dict[str, Any]]:
    rows: list[Any] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(rows, local)
    else:
        rows[0] = local
    result = [dict(item) for rank_rows in rows for item in rank_rows]
    result.sort(key=lambda item: int(item["task_ordinal"]))
    if len(result) != 24 or len({row["task_ordinal"] for row in result}) != 24:
        raise ExpertManifoldError("v6-prior macro did not cover train24")
    return result


def _mean_trainable_gradients(runtime: V6PriorRuntime) -> torch.Tensor:
    parameters = runtime.trainable_parameters
    if any(parameter.grad is None for parameter in parameters):
        raise ExpertManifoldError("v6-prior trainable gradient ownership changed")
    flat = torch.cat([parameter.grad.reshape(-1) for parameter in parameters])
    if runtime.context.world_size > 1:
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(runtime.context.world_size)
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            parameter.grad.copy_(flat[offset : offset + count].view_as(parameter))
            offset += count
    if offset != flat.numel():
        raise ExpertManifoldError("v6-prior flat gradient layout changed")
    return flat


def _runtime_maximums(
    context: DistributedContext,
    started: float,
    input_wait_seconds: float,
) -> tuple[float, int, int, float]:
    # One macro-boundary synchronization makes throughput evidence include all
    # queued optimizer work. Per-task and per-tensor synchronization stays absent.
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


def _component_layout(
    runtime: V6PriorRuntime,
) -> tuple[tuple[str, int, int], ...]:
    rows = []
    offset = 0
    for name, parameter in zip(
        runtime.trainable_names,
        runtime.trainable_parameters,
        strict=True,
    ):
        stop = offset + parameter.numel()
        rows.append((name, offset, stop))
        offset = stop
    return tuple(rows)


def _component_norms(
    value: torch.Tensor,
    layout: Sequence[tuple[str, int, int]],
) -> dict[str, float]:
    parts: dict[str, list[torch.Tensor]] = {"compiler": [], "factor_heads": []}
    for name, start, stop in layout:
        root = name.split(".", 1)[0]
        parts[root].append(value[start:stop].square().sum())
    packed = (
        torch.stack(
            (
                torch.stack(parts["compiler"]).sum().sqrt(),
                torch.stack(parts["factor_heads"]).sum().sqrt(),
                torch.linalg.vector_norm(value),
            )
        )
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .tolist()
    )
    if not all(math.isfinite(item) for item in packed):
        raise ExpertManifoldError("v6-prior gradient-profile norm is non-finite")
    return dict(zip(("compiler", "factor_heads", "global"), packed, strict=True))


def _run_gradient_profile(runtime: V6PriorRuntime) -> None:
    local_tasks = 24 // runtime.context.world_size
    total_parameters = sum(value.numel() for value in runtime.trainable_parameters)
    accumulators = {
        name: torch.zeros(
            total_parameters,
            dtype=torch.float32,
            device=runtime.context.device,
        )
        for name in ("positive", "expert", "ranking")
    }
    local_records = []
    input_wait_seconds = 0.0
    started = time.monotonic()
    for microtask in range(local_tasks):
        input_started = time.monotonic()
        batch = next(runtime.iterator)
        input_wait_seconds += time.monotonic() - input_started
        objective = _task_objective(
            runtime,
            macro=runtime.segment.schedule_start_macro,
            microtask=microtask,
            batch=batch,
        )
        components = parameter_gradient_components(
            pair=objective.pair,
            functional=objective.functional_gradients,
            auxiliary=objective.auxiliary,
            parameters=runtime.trainable_parameters,
        )
        for name, values in (
            ("positive", components.positive),
            ("expert", components.expert),
            ("ranking", components.ranking),
        ):
            offset = 0
            for value in values:
                stop = offset + value.numel()
                accumulators[name][offset:stop].add_(
                    value.reshape(-1).float(),
                    alpha=1.0 / local_tasks,
                )
                offset = stop
            if offset != total_parameters:
                raise ExpertManifoldError("gradient-profile layout changed")
        local_records.append(_task_record(objective))
    if any(parameter.grad is not None for parameter in runtime.policy.parameters()):
        raise ExpertManifoldError("gradient profile touched source policy gradients")
    for value in accumulators.values():
        if runtime.context.world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value.div_(runtime.context.world_size)
    layout = _component_layout(runtime)
    norms = {
        name: _component_norms(value, layout) for name, value in accumulators.items()
    }
    fraction = float(
        runtime.config["objective"]["auxiliary_weights"][
            "maximum_fraction_of_positive_gradient_per_auxiliary"
        ]
    )
    recommended = {
        name: suggest_auxiliary_weight(
            norms["positive"], norms[name], maximum_fraction=fraction
        )
        for name in ("expert", "ranking")
    }
    seconds, allocated, reserved, input_wait_seconds = _runtime_maximums(
        runtime.context,
        started,
        input_wait_seconds,
    )
    task_records = _gather_task_records(local_records, runtime.context)
    action_queries_per_task = int(runtime.config["data"]["action_queries_per_task"])
    query_summary = runtime.run_contract["data"]["consumed_schedule"]["query"]
    total_action_queries = int(query_summary["global_examples"])
    unique_action_queries = int(query_summary["unique_query_rows"])
    counterfactual_counts = {
        name: sum(record["counterfactual_kind"] == name for record in task_records)
        for name in ("reversed", "shuffled", "wrong")
    }
    if (
        len(task_records) != 24
        or action_queries_per_task != 20
        or total_action_queries != 480
        or unique_action_queries != 480
        or total_action_queries != len(task_records) * action_queries_per_task
        or counterfactual_counts != {"reversed": 8, "shuffled": 8, "wrong": 8}
    ):
        raise ExpertManifoldError("v6-prior gradient-profile panel changed")
    if runtime.context.is_main:
        write_json_atomic(
            runtime.args.output_dir / "gradient_profile.json",
            {
                "schema_version": "ember_pi05_v6_prior_gradient_profile_seal_v1",
                "schedule_macro": runtime.segment.schedule_start_macro,
                "task_count": len(task_records),
                "action_queries_per_task": action_queries_per_task,
                "total_action_queries": total_action_queries,
                "unique_action_queries": unique_action_queries,
                "counterfactual_counts": counterfactual_counts,
                "unweighted_gradient_norms": norms,
                "maximum_auxiliary_fraction": fraction,
                "recommended_weights": recommended,
                "seal_rule": runtime.config["gradient_profile"]["seal_rule"],
                "task_records": task_records,
                "step_seconds": seconds,
                "input_wait_seconds": input_wait_seconds,
                "max_cuda_allocated_bytes": allocated,
                "max_cuda_reserved_bytes": reserved,
                "oom_count": 0,
                "nonfinite_count": 0,
                "content_hash_policy": "disabled_by_owner",
            },
        )
        write_json_atomic(
            runtime.args.output_dir / "completion.json",
            {
                "schema_version": V6_PRIOR_COMPLETION_SCHEMA,
                "mode": "gradient-profile",
                "completed_diagnostic_macros": 1,
                "schedule_start_macro": runtime.segment.schedule_start_macro,
                "schedule_stop_macro": runtime.segment.schedule_stop_macro,
                "gradient_profile_complete": True,
                "oom_count": 0,
                "nonfinite_count": 0,
                "content_hash_policy": "disabled_by_owner",
            },
        )


def _run_training(runtime: V6PriorRuntime) -> None:
    weights = runtime.config["objective"]["auxiliary_weights"]
    expert_weight = float(weights["expert"])
    ranking_weight = float(weights["ranking"])
    local_tasks = 24 // runtime.context.world_size
    started = time.monotonic()
    for macro in range(runtime.segment.start_macro, runtime.segment.stop_macro):
        step_started = time.monotonic()
        runtime.optimizer.zero_grad(set_to_none=True)
        local_records = []
        input_wait_seconds = 0.0
        for microtask in range(local_tasks):
            input_started = time.monotonic()
            batch = next(runtime.iterator)
            input_wait_seconds += time.monotonic() - input_started
            objective = _task_objective(
                runtime,
                macro=macro,
                microtask=microtask,
                batch=batch,
            )
            outputs, gradients = merged_output_gradients(
                pair=objective.pair,
                functional=objective.functional_gradients,
                auxiliary=objective.auxiliary,
                expert_weight=expert_weight,
                ranking_weight=ranking_weight,
                task_scale=1.0 / local_tasks,
            )
            torch.autograd.backward(outputs, gradients)
            local_records.append(_task_record(objective))
        if any(parameter.grad is not None for parameter in runtime.policy.parameters()):
            raise ExpertManifoldError("v6-prior source policy accumulated gradients")
        _mean_trainable_gradients(runtime)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            runtime.trainable_parameters,
            float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"]),
        )
        gradient_norm_value = float(gradient_norm)
        if not math.isfinite(gradient_norm_value):
            raise ExpertManifoldError("v6-prior gradient norm is non-finite")
        applied_lr = float(runtime.optimizer.param_groups[0]["lr"])
        runtime.optimizer.step()
        runtime.scheduler.step()
        cursor = macro + 1
        task_records = _gather_task_records(local_records, runtime.context)
        metric_names = (
            "functional_loss",
            "expert_loss",
            "expert_direction",
            "expert_log_norm",
            "ranking_loss",
            "ranking_margin",
            "correct_expert_cosine",
            "counterfactual_expert_cosine",
            "correct_effective_norm",
            "counterfactual_effective_norm",
            "expert_effective_norm",
        )
        metrics = {
            name: sum(float(row[name]) for row in task_records) / 24
            for name in metric_names
        }
        seconds, allocated, reserved, input_wait_seconds = _runtime_maximums(
            runtime.context,
            step_started,
            input_wait_seconds,
        )
        row = {
            "macro": cursor,
            **metrics,
            "expert_weight": expert_weight,
            "ranking_weight": ranking_weight,
            "gradient_norm_before_clip": gradient_norm_value,
            "applied_lr": applied_lr,
            "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
            "counterfactual_counts": {
                name: sum(
                    record["counterfactual_kind"] == name for record in task_records
                )
                for name in ("reversed", "shuffled", "wrong")
            },
            "task_records": task_records,
            "step_seconds": seconds,
            "input_wait_seconds": input_wait_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "max_cuda_allocated_bytes": allocated,
            "max_cuda_reserved_bytes": reserved,
        }
        if not all(
            math.isfinite(float(row[name]))
            for name in (*metric_names, "gradient_norm_before_clip")
        ):
            raise ExpertManifoldError("v6-prior training metric is non-finite")
        if runtime.context.is_main:
            append_jsonl(runtime.metrics_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)
        if cursor in runtime.segment.checkpoint_macros:
            save_v6_prior_checkpoint(
                output_dir=runtime.args.output_dir,
                macro=cursor,
                writer=runtime.writer,
                optimizer=runtime.optimizer,
                scheduler=runtime.scheduler,
                context=runtime.context,
                metrics_rows=cursor,
                cursor_contract=_cursor_contract(runtime.config, cursor),
                checkpoint_contract=runtime.checkpoint_contract,
            )
    if runtime.context.is_main:
        write_json_atomic(
            runtime.args.output_dir / "completion.json",
            {
                "schema_version": V6_PRIOR_COMPLETION_SCHEMA,
                "mode": runtime.args.mode,
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
                        "trainable_parameters": runtime.ownership.trainable_parameter_count,
                        "frozen_parameters": runtime.ownership.frozen_parameter_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.mode == "gradient-profile":
            _run_gradient_profile(runtime)
        else:
            _run_training(runtime)
    finally:
        if runtime is not None:
            runtime.dataset.close()
            runtime.video_store.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_v6_prior_policy_effective_writer_v1.json",
    )
    parser.add_argument(
        "--mode",
        choices=("gradient-profile", "profile", "formal"),
        required=True,
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--expert-bank-root", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-macro", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
        "source_run",
        "checkpoint",
        "tokenizer_path",
        "data_root",
        "expert_bank_root",
        "warm_start",
    ):
        path = getattr(args, name).resolve()
        if not path.exists():
            raise ExpertManifoldError(f"missing v6-prior path: {path}")
        setattr(args, name, path)
    args.output_dir = args.output_dir.resolve()
    args.resume = args.resume.resolve() if args.resume else None
    if args.config != V6_PRIOR_CANONICAL_CONFIG:
        raise ExpertManifoldError(
            "v6-prior runtime requires the tracked canonical config"
        )
    configured_warm_start = (
        REPO_ROOT
        / str(load_v6_prior_config(args.config)["initialization"]["checkpoint"])
    ).resolve()
    if args.warm_start != configured_warm_start:
        raise ExpertManifoldError("v6-prior warm-start path changed")
    if args.num_workers < 0 or (
        args.stop_after_macro is not None and args.stop_after_macro <= 0
    ):
        raise ExpertManifoldError("v6-prior worker count or stop boundary is invalid")
    return args
