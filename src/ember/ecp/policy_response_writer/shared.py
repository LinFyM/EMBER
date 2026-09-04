"""Shared positive-only training for the Policy-Response Writer.

Each optimizer update is role-balanced across correct videos only.  Frozen
policy LoRA-leaf VJPs provide exact functional credit, while a prefix-only
future-response objective trains the same Frame/Event variables.  No held,
wrong, shuffled, reversed, no-video, or language-only condition receives a
gradient in this module.
"""

from __future__ import annotations

import gc
import math
import os
import random
import shutil
import socket
import statistics
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ember.ecp.policy_response_writer.capture import FrozenPolicyResponseVideo
from ember.ecp.policy_response_writer.shared_contract import (
    build_shared_run_contract,
    seal_or_validate_shared_run_contract,
)
from ember.ecp.policy_response_writer.shared_schedule import (
    VideoSplit,
    _evaluation_ids,
    _split_ids,
    _video_splits,
    balanced_task_owners,
    configured_task_group,
    evaluation_task_costs,
    owner_balanced_task_group,
    role_balanced_task_owners,
    scheduled_task_costs,
    shared_task_group,
    task_group_counts,
    training_video_demos,
)
from ember.ecp.policy_response_writer.shared_execution import (
    shared_mmap_execution_plan,
    selective_replication_plan,
)
from ember.ecp.policy_response_writer.shared_video_cache import (
    SharedPolicyResponseVideoCache,
)
from ember.ecp.policy_response_writer.training import (
    REPO_ROOT,
    PolicyResponseRuntime,
    capture_video,
)
from ember.ecp.shared_compiler_assets import authority_path
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import barrier, read_json, write_json_atomic
from ember.pi05_source_setup import initialize_deferred_process_group


SHARED_STAGE = "policy_response_writer_shared_positive_only"
SHARED_RUN_SCHEMA = "ember_policy_response_writer_shared_run_v1"


@dataclass
class SharedEvidenceCache:
    videos: dict[tuple[int, int], FrozenPolicyResponseVideo]
    capture_records: list[dict[str, Any]]
    functional_normalizers: dict[int, float]
    process_normalizers: dict[int, float]


def causal_pair(
    frame_count: int, event_slots: int, *, optimizer_step: int, task: int, demo: int
) -> tuple[int, int]:
    """Uniformly sample a reproducible legal prefix and positive future offset."""

    span = int(frame_count) - int(event_slots)
    if span <= 0:
        raise ValueError("shared Writer video is too short for causal prediction")
    seed = (
        (int(optimizer_step) + 1) * 1_000_003
        + (int(task) + 1) * 10_000_019
        + (int(demo) + 1) * 100_000_007
    )
    generator = random.Random(seed)
    cutoff = generator.randrange(int(event_slots), int(frame_count))
    future_offset = generator.randrange(1, int(frame_count) - cutoff + 1)
    return cutoff, future_offset


def functional_objective(
    *,
    generated_loss: float,
    carrier_loss: float,
    normalizer: float,
    task_weight: float,
    preservation_weight: float,
    preservation_epsilon: float,
) -> dict[str, float | bool]:
    """Return the scalar VJP mass for functional plus one-sided preservation."""

    values = (
        generated_loss,
        carrier_loss,
        normalizer,
        task_weight,
        preservation_weight,
        preservation_epsilon,
    )
    if (
        not all(math.isfinite(float(value)) for value in values)
        or normalizer <= 0
        or task_weight <= 0
        or min(preservation_weight, preservation_epsilon) < 0
    ):
        raise ValueError("shared Writer functional objective changed")
    excess = generated_loss - carrier_loss - preservation_epsilon
    active = excess > 0
    gradient_mass = (
        task_weight * (1.0 + (preservation_weight if active else 0.0)) / normalizer
    )
    return {
        "functional_normalized": generated_loss / normalizer,
        "preservation_normalized": max(0.0, excess) / normalizer,
        "preservation_active": active,
        "gradient_mass": gradient_mass,
    }


def _initialize_collectives(runtime: PolicyResponseRuntime) -> None:
    initialize_deferred_process_group(
        runtime.context,
        rendezvous_root=(
            runtime.args.asset_root / ".codex/tmp/policy_response_writer_rendezvous"
        ),
    )
    if runtime.context.world_size > 1:
        for parameter in runtime.writer.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in runtime.writer.buffers():
            dist.broadcast(buffer.data, src=0)
        barrier(runtime.context)


def _world_topology(runtime: PolicyResponseRuntime) -> list[dict[str, Any]]:
    local = {
        "rank": runtime.context.rank,
        "local_rank": runtime.context.local_rank,
        "device": str(runtime.context.device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": torch.cuda.get_device_name(runtime.context.device),
        "hostname": socket.gethostname(),
        "numa_node": runtime.context.numa_node,
        "cpu_affinity": list(runtime.context.cpu_affinity or ()),
    }
    rows: list[Any] = [None] * runtime.context.world_size
    if runtime.context.world_size > 1:
        dist.all_gather_object(rows, local)
    else:
        rows[0] = local
    return list(rows)


def _optimizer(
    runtime: PolicyResponseRuntime,
) -> tuple[
    tuple[torch.nn.Parameter, ...],
    torch.optim.AdamW,
    torch.optim.lr_scheduler.LambdaLR,
]:
    runtime.writer.requires_grad_(True).train()
    named_parameters = tuple(runtime.writer.named_parameters())
    parameters = tuple(value for _, value in named_parameters)
    prediction_parameters = tuple(
        value
        for name, value in named_parameters
        if name.startswith("process.prediction_")
    )
    prediction_ids = {id(value) for value in prediction_parameters}
    remaining_parameters = tuple(
        value for value in parameters if id(value) not in prediction_ids
    )
    cell = runtime.config["optimization"]["shared"]
    learning_rate = float(cell["learning_rate"])
    prediction_multiplier = float(cell["process_prediction_lr_multiplier"])
    optimizer = torch.optim.AdamW(
        (
            {"params": remaining_parameters},
            {
                "params": prediction_parameters,
                "lr": learning_rate * prediction_multiplier,
            },
        ),
        lr=learning_rate,
        betas=tuple(cell["betas"]),
        weight_decay=float(cell["weight_decay"]),
    )
    warmup = int(cell["warmup_updates"])
    effective = int(cell["effective_updates"])
    floor = float(cell["decay_learning_rate"]) / float(cell["learning_rate"])

    def scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(effective, 1)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    if (
        not parameters
        or not prediction_parameters
        or not remaining_parameters
        or len(prediction_ids) != len(prediction_parameters)
        or not math.isfinite(prediction_multiplier)
        or prediction_multiplier <= 1.0
        or runtime.writer.composer.task_query is not None
        or any(value.requires_grad for value in runtime.policy.parameters())
        or any(value.requires_grad for value in runtime.stage0.parameters())
    ):
        raise RuntimeError("shared Writer parameter ownership changed")
    return parameters, optimizer, scheduler


@torch.no_grad()
def _target_only_process_normalizer(
    runtime: PolicyResponseRuntime,
    video: FrozenPolicyResponseVideo,
    *,
    task: int,
    demo: int,
    pair_count: int,
) -> float:
    """Estimate task scale from several fixed targets, independent of Writer state."""

    if pair_count <= 0:
        raise ValueError("process normalizer pair count changed")
    process = runtime.writer.process
    teacher = process.fixed_teacher_response(video).detach()
    losses = []
    for pair_index in range(pair_count):
        cutoff, future_offset = causal_pair(
            video.frame_count,
            process.event_slots,
            optimizer_step=pair_index,
            task=task,
            demo=demo,
        )
        target = process.standardized_teacher_delta(
            teacher,
            cutoff=cutoff,
            future_offset=future_offset,
        )
        losses.append(
            float(
                F.smooth_l1_loss(
                    torch.zeros_like(target),
                    target,
                    beta=1.0,
                    reduction="mean",
                )
            )
        )
    del teacher
    result = statistics.fmean(losses)
    if not math.isfinite(result) or result <= 1e-8:
        raise RuntimeError("shared Writer target-only process normalizer changed")
    return result


def _capture_missing(
    runtime: PolicyResponseRuntime,
    cache: SharedEvidenceCache,
    *,
    task: int,
    demos: Sequence[int],
    shared_video_cache: SharedPolicyResponseVideoCache | None = None,
    record_capture: bool = True,
) -> None:
    for demo in map(int, demos):
        key = (int(task), demo)
        if key in cache.videos:
            continue
        if shared_video_cache is None:
            evidence, record = capture_video(runtime, task_id=task, video_demo=demo)
            frozen = evidence.to("cpu")
            record = {**record, "tensor_bytes": frozen.tensor_bytes}
            cache.videos[key] = frozen
            if record_capture:
                cache.capture_records.append(record)
            del evidence
        else:

            def builder() -> tuple[FrozenPolicyResponseVideo, Mapping[str, Any]]:
                evidence, record = capture_video(
                    runtime, task_id=task, video_demo=demo
                )
                frozen = evidence.to("cpu")
                return frozen, {**record, "tensor_bytes": frozen.tensor_bytes}

            result = shared_video_cache.get_or_build(
                task=task,
                demo=demo,
                builder=builder,
            )
            cache.videos[key] = result.video
            if record_capture:
                cache.capture_records.append(
                    {
                        **result.capture,
                        "shared_mmap_cache_hit": result.hit,
                        "shared_mmap_file_bytes": result.file_bytes,
                        "shared_mmap_build_seconds": result.build_seconds,
                        "shared_mmap_load_seconds": result.load_seconds,
                    }
                )
        torch.cuda.empty_cache()


def _functional_normalizer(runtime: PolicyResponseRuntime, task: int) -> float:
    values = tuple(float(value.flow_loss) for value in runtime.panels[task].panel_a)
    result = math.sqrt(statistics.fmean(value * value for value in values))
    if not math.isfinite(result) or result <= 1e-8:
        raise RuntimeError("shared Writer functional normalizer changed")
    return result


def _prepare_training_cache(
    runtime: PolicyResponseRuntime,
    *,
    owned_tasks: Sequence[int],
    video_splits: Mapping[int, VideoSplit],
    shared_video_cache: SharedPolicyResponseVideoCache | None = None,
) -> SharedEvidenceCache:
    cache = SharedEvidenceCache({}, [], {}, {})
    pair_count = int(
        runtime.config["optimization"]["shared"][
            "process_normalizer_pairs_per_fit_video"
        ]
    )
    runtime.writer.eval()
    for task in map(int, owned_tasks):
        fit, _ = video_splits[task]
        _capture_missing(
            runtime,
            cache,
            task=task,
            demos=fit,
            shared_video_cache=shared_video_cache,
        )
        cache.functional_normalizers[task] = _functional_normalizer(runtime, task)
        losses = []
        for demo in fit:
            video = cache.videos[(task, demo)].to(runtime.context.device)
            losses.append(
                _target_only_process_normalizer(
                    runtime,
                    video,
                    task=task,
                    demo=demo,
                    pair_count=pair_count,
                )
            )
            del video
        normalizer = statistics.fmean(losses)
        if not math.isfinite(normalizer) or normalizer <= 1e-8:
            raise RuntimeError("shared Writer process normalizer changed")
        cache.process_normalizers[task] = normalizer
        torch.cuda.empty_cache()
    runtime.writer.train()
    return cache


def _materialized_state(
    runtime: PolicyResponseRuntime,
    videos: Sequence[FrozenPolicyResponseVideo],
    *,
    canonicalize: bool,
) -> dict[str, torch.Tensor]:
    output = runtime.writer(
        tuple(videos),
        s_ref=runtime.ranks.s_ref,
        representation=runtime.args.representation,
    )
    return runtime.writer.materialize(
        output,
        carrier_state=runtime.ranks.carrier_rank12,
        rank4_contract=runtime.rank4_contract,
        rank16_contract=runtime.ranks.contract,
        canonicalize=canonicalize,
    )


def _gather_flat(
    runtime: PolicyResponseRuntime, values: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[Any] = [None] * runtime.context.world_size
    local = [dict(value) for value in values]
    if runtime.context.world_size > 1:
        dist.all_gather_object(rows, local)
    else:
        rows[0] = local
    return [dict(value) for rank_rows in rows for value in rank_rows]


def _gather_mapping(
    runtime: PolicyResponseRuntime, values: Mapping[int, float]
) -> dict[int, float]:
    rows: list[Any] = [None] * runtime.context.world_size
    local = {int(key): float(value) for key, value in values.items()}
    if runtime.context.world_size > 1:
        dist.all_gather_object(rows, local)
    else:
        rows[0] = local
    result: dict[int, float] = {}
    for row in rows:
        overlap = set(result).intersection(map(int, row))
        if overlap:
            raise RuntimeError("shared Writer normalizer ownership overlapped")
        result.update({int(key): float(value) for key, value in row.items()})
    return result


def _seal_normalizers(
    runtime: PolicyResponseRuntime,
    cache: SharedEvidenceCache,
    normalizers: Mapping[str, Mapping[int, float]],
    expected_tasks: set[int],
) -> dict[str, dict[int, float]]:
    canonical = {
        axis: {int(task): float(value) for task, value in rows.items()}
        for axis, rows in normalizers.items()
    }
    if (
        set(canonical) != {"functional", "process"}
        or any(set(rows) != expected_tasks for rows in canonical.values())
        or any(
            not math.isfinite(value) or value <= 0
            for rows in canonical.values()
            for value in rows.values()
        )
    ):
        raise RuntimeError("shared Writer normalizers lost a gradient task")
    if runtime.args.mode != "formal":
        return canonical
    path = runtime.args.output_dir / "normalizers.json"
    if runtime.context.is_main and runtime.args.resume is None:
        write_json_atomic(path, canonical)
    barrier(runtime.context)
    stored = {
        axis: {int(task): float(value) for task, value in rows.items()}
        for axis, rows in read_json(path).items()
    }
    if set(stored) != {"functional", "process"} or any(
        set(rows) != expected_tasks for rows in stored.values()
    ):
        raise ValueError("shared Writer frozen normalizer authority changed")
    owned = set(cache.functional_normalizers)
    cache.functional_normalizers = {task: stored["functional"][task] for task in owned}
    cache.process_normalizers = {task: stored["process"][task] for task in owned}
    return stored


def _shared_video_cache(
    runtime: PolicyResponseRuntime,
) -> SharedPolicyResponseVideoCache | None:
    root = runtime.args.shared_evidence_cache_root
    if root is None:
        return None
    authority = {
        "run_root": str(runtime.args.output_dir),
        "git": git_state(REPO_ROOT),
        "config": {
            "path": str(runtime.args.config),
            "bytes": runtime.args.config.stat().st_size,
        },
        "source_checkpoint": str(
            authority_path(
                runtime.base,
                "source_checkpoint",
                asset_root=runtime.args.asset_root,
            )
        ),
        "native_observer_checkpoint": str(
            authority_path(
                runtime.base,
                "native_observer_checkpoint",
                asset_root=runtime.args.asset_root,
            )
        ),
        "representation": runtime.args.representation,
        "frame_stride": int(runtime.config["data"]["frame_stride"]),
        "owner_shapes": [
            [owner.family.value, owner.in_features, owner.out_features]
            for owner in runtime.owners
        ],
    }
    return SharedPolicyResponseVideoCache(root, authority=authority)


def _remove_shared_video_cache(
    runtime: PolicyResponseRuntime,
    cache: SharedEvidenceCache,
    shared_video_cache: SharedPolicyResponseVideoCache,
) -> None:
    """Release every rank's mmap tensors before rank zero removes the cache."""

    # The cache root can live on NFS. Unlinking a safetensors file while another
    # rank still maps it creates an ephemeral .nfs handle, so rmtree can fail
    # with ENOTEMPTY and strand the remaining ranks at the following barrier.
    cache.videos.clear()
    gc.collect()
    barrier(runtime.context)
    removal_error = None
    if runtime.context.is_main:
        try:
            shutil.rmtree(shared_video_cache.root)
        except OSError as error:
            removal_error = f"{type(error).__name__}: {error}"
    errors = [removal_error]
    if runtime.context.world_size > 1:
        dist.broadcast_object_list(
            errors,
            src=0,
            device=runtime.context.device,
        )
    if errors[0] is not None:
        raise RuntimeError(
            f"shared policy-response cache cleanup failed: {errors[0]}"
        )
    barrier(runtime.context)


def _validate_shared_run(
    runtime: PolicyResponseRuntime, cell: Mapping[str, Any]
) -> None:
    if not 1 <= runtime.context.world_size <= 6:
        raise ValueError("shared Writer supports one through six local GPUs")
    if runtime.args.mode == "formal" and runtime.args.task is not None:
        raise ValueError("formal shared Writer must use its registered task split")
    cardinalities = tuple(
        map(
            int,
            runtime.config["data"].get(
                "training_K", (runtime.config["data"]["initial_K"],)
            ),
        )
    )
    meta, target, _ = _split_ids(runtime)
    task_group_counts(cell, meta=meta, target=target)
    replication_budget = float(runtime.args.cache_replication_budget_gib)
    if (
        int(runtime.config["data"]["initial_K"]) != 1
        or tuple(sorted(set(cardinalities))) != cardinalities
        or not cardinalities
        or not set(cardinalities) <= set(runtime.config["data"]["supported_K"])
        or bool(runtime.config["information_wall"]["wrong_training_loss"])
        or not math.isfinite(replication_budget)
        or replication_budget < 0
        or (
            runtime.args.shared_evidence_cache_root is not None
            and replication_budget != 0.0
        )
    ):
        raise ValueError("shared Writer positive-only training contract changed")


def _base_cache_bytes_by_task(
    records: Sequence[Mapping[str, Any]], expected_tasks: set[int]
) -> dict[int, int]:
    result = {task: 0 for task in expected_tasks}
    for row in records:
        task = int(row["task_id"])
        if task not in result:
            raise RuntimeError("shared Writer base cache escaped gradient tasks")
        result[task] += int(row["tensor_bytes"])
    if any(value <= 0 for value in result.values()):
        raise RuntimeError("shared Writer base cache lost a gradient task")
    return result


def _json_execution_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "ember_policy_response_writer_execution_plan_v3",
        "strategy": str(plan["strategy"]),
        "replica_search": str(plan["replica_search"]),
        "execution_ownership": [
            list(map(int, row)) for row in plan["execution_ownership"]
        ],
        "replicas": [list(map(int, row)) for row in plan["replicas"]],
        "extra_cache_bytes": int(plan["extra_cache_bytes"]),
        "shared_cache_bytes": int(plan.get("shared_cache_bytes", 0)),
        "budget_bytes": int(plan["budget_bytes"]),
        "base_total_cost": int(plan["base_total_cost"]),
        "base_tail_cost": int(plan["base_tail_cost"]),
        "predicted_total_cost": int(plan["predicted_total_cost"]),
        "predicted_tail_cost": int(plan["predicted_tail_cost"]),
        "ideal_total_cost": int(plan["ideal_total_cost"]),
        "ideal_tail_cost": int(plan["ideal_tail_cost"]),
        "unique_step_signatures": int(plan["unique_step_signatures"]),
        "planned_steps": int(plan["planned_steps"]),
        "selection_uses_outcomes": False,
        "changes_task_group_or_weight": False,
    }


def _seal_execution_plan(
    runtime: PolicyResponseRuntime, plan: Mapping[str, Any]
) -> dict[str, Any]:
    canonical = _json_execution_plan(plan)
    if runtime.args.mode != "formal":
        return canonical
    path = runtime.args.output_dir / "execution_plan.json"
    if runtime.context.is_main and runtime.args.resume is None:
        write_json_atomic(path, canonical)
    barrier(runtime.context)
    stored = read_json(path)
    if stored != canonical:
        raise ValueError("shared Writer frozen execution plan changed")
    return stored


def _build_execution_plan(
    runtime: PolicyResponseRuntime,
    *,
    task_owners: Sequence[Sequence[int]],
    video_splits: Mapping[int, VideoSplit],
    cache_bytes: Mapping[int, int],
    stop: int,
) -> dict[str, Any]:
    if runtime.args.mode == "profile" and runtime.args.task is not None:
        groups = ((int(runtime.args.task),),) * stop
    else:
        groups = tuple(
            configured_task_group(runtime, step, task_owners=task_owners)
            for step in range(stop)
        )
    costs = tuple(
        scheduled_task_costs(
            runtime,
            video_splits,
            group,
            optimizer_step=step,
        )
        for step, group in enumerate(groups)
    )
    if runtime.args.shared_evidence_cache_root is not None:
        return _seal_execution_plan(
            runtime,
            shared_mmap_execution_plan(
                costs,
                cache_bytes=cache_bytes,
                world_size=runtime.context.world_size,
            ),
        )
    budget = int(float(runtime.args.cache_replication_budget_gib) * (2**30))
    return _seal_execution_plan(
        runtime,
        selective_replication_plan(
            costs,
            base_task_owners=task_owners,
            cache_bytes=cache_bytes,
            extra_budget_bytes=budget,
        ),
    )


def _install_execution_caches(
    runtime: PolicyResponseRuntime,
    cache: SharedEvidenceCache,
    *,
    execution_owners: Sequence[Sequence[int]],
    video_splits: Mapping[int, VideoSplit],
    normalizers: Mapping[str, Mapping[int, float]],
    shared_video_cache: SharedPolicyResponseVideoCache | None = None,
) -> None:
    for task in map(int, execution_owners[runtime.context.rank]):
        fit, _ = video_splits[task]
        _capture_missing(
            runtime,
            cache,
            task=task,
            demos=fit,
            shared_video_cache=shared_video_cache,
            record_capture=False,
        )
        cache.functional_normalizers[task] = float(normalizers["functional"][task])
        cache.process_normalizers[task] = float(normalizers["process"][task])
    barrier(runtime.context)


def _build_result(
    runtime: PolicyResponseRuntime,
    *,
    stop: int,
    start_step: int,
    total: int,
    task_owners: Sequence[Sequence[int]],
    evaluation_task_owners: Sequence[Sequence[int]],
    execution_plan: Mapping[str, Any],
    normalizers: Mapping[str, Mapping[int, float]],
    training: Mapping[str, Any],
    evaluations: Mapping[str, Any],
    evaluation_seconds: float,
    capture_records: Sequence[Mapping[str, Any]],
    parameters: Sequence[torch.nn.Parameter],
    started: float,
) -> dict[str, Any]:
    latest = sorted(evaluations)[-1] if evaluations else None
    return {
        "schema_version": SHARED_RUN_SCHEMA,
        "status": "complete",
        "phase": "shared",
        "mode": runtime.args.mode,
        "representation": runtime.args.representation,
        "initialization_request": runtime.args.initialization,
        "initialization": runtime.initialization,
        "optimizer_steps": stop,
        "resume_start_step": start_step,
        "configured_total_steps": total,
        "task_ownership": [list(value) for value in task_owners],
        "evaluation_task_ownership": [
            list(value) for value in evaluation_task_owners
        ],
        "execution_plan": dict(execution_plan),
        "normalizers": normalizers,
        "curve": training["curve"],
        "metrics_rows": training["metrics_rows"],
        "train_seconds": training["train_seconds"],
        "evaluations": evaluations,
        "evaluation": evaluations[latest] if latest is not None else None,
        "evaluation_seconds": evaluation_seconds,
        "capture": list(capture_records),
        "frozen_evidence_cache": "ephemeral_cpu_frozen_policy_response",
        "frozen_evidence_tensor_bytes": sum(
            int(value["tensor_bytes"]) for value in capture_records
        ),
        "trainable_parameter_count": sum(value.numel() for value in parameters),
        "source_policy_trainable_parameter_count": 0,
        "native_observer_trainable_parameter_count": 0,
        "task_local_parameter_count": 0,
        "action_meta_installed": False,
        "wrong_video_backward_calls": 0,
        "true_task_held_backward_calls": 0,
        "same_task_held_backward_calls": 0,
        "panel_b_backward_calls": 0,
        "shuffled_or_reversed_reads": 0,
        "single_complete_rank16": True,
        "max_cuda_allocated_bytes": training["max_cuda_allocated_bytes"],
        "max_cuda_reserved_bytes": training["max_cuda_reserved_bytes"],
        "total_seconds": time.monotonic() - started,
    }


def run_shared(runtime: PolicyResponseRuntime) -> dict[str, Any]:
    meta, target, held = _split_ids(runtime)
    gradient_tasks = set((*meta, *target))
    cell = runtime.config["optimization"]["shared"]
    _validate_shared_run(runtime, cell)
    _initialize_collectives(runtime)
    task_ids = (
        (int(runtime.args.task),)
        if runtime.args.mode == "profile" and runtime.args.task is not None
        else (*meta, *target, *held)
    )
    video_splits, costs = _video_splits(
        runtime, task_ids, gradient_tasks=gradient_tasks
    )
    if (runtime.args.mode == "profile" and runtime.args.task is not None) or cell.get(
        "tasks_per_update_by_role"
    ) is not None:
        task_owners = balanced_task_owners(costs, runtime.context.world_size)
    else:
        # Preserve the current qualification run's owner-coupled 3+3 sequence.
        # New explicit samplers are owner-independent and use global cost balance.
        task_owners = role_balanced_task_owners(
            costs,
            meta=meta,
            target=target,
            held=held,
            world_size=runtime.context.world_size,
        )
    parameters, optimizer, scheduler = _optimizer(runtime)
    total = int(cell["warmup_updates"]) + int(cell["effective_updates"])
    stop = int(
        runtime.args.stop_after_step
        or (int(cell["profile_updates"]) if runtime.args.mode == "profile" else total)
    )
    if not 0 < stop <= total or (runtime.args.mode == "formal" and stop != total):
        raise ValueError("shared Writer stop step changed")
    contract = build_shared_run_contract(
        runtime,
        schema=SHARED_RUN_SCHEMA,
        stage=SHARED_STAGE,
        stop=stop,
        parameters=parameters,
        topology=_world_topology(runtime),
        task_owners=task_owners,
        video_splits=video_splits,
    )
    seal_or_validate_shared_run_contract(runtime, contract)
    shared_video_cache = _shared_video_cache(runtime)
    started = time.monotonic()
    owned_gradient = tuple(
        task for task in task_owners[runtime.context.rank] if task in gradient_tasks
    )
    cache = _prepare_training_cache(
        runtime,
        owned_tasks=owned_gradient,
        video_splits=video_splits,
        shared_video_cache=shared_video_cache,
    )
    observed_normalizers = {
        "functional": _gather_mapping(runtime, cache.functional_normalizers),
        "process": _gather_mapping(runtime, cache.process_normalizers),
    }
    expected_gradient = set(task_ids).intersection(gradient_tasks)
    normalizers = _seal_normalizers(
        runtime, cache, observed_normalizers, expected_gradient
    )
    base_capture_records = _gather_flat(runtime, cache.capture_records)
    execution_plan = _build_execution_plan(
        runtime,
        task_owners=task_owners,
        video_splits=video_splits,
        cache_bytes=_base_cache_bytes_by_task(base_capture_records, expected_gradient),
        stop=stop,
    )
    execution_owners = tuple(
        tuple(map(int, row)) for row in execution_plan["execution_ownership"]
    )
    _install_execution_caches(
        runtime,
        cache,
        execution_owners=execution_owners,
        video_splits=video_splits,
        normalizers=normalizers,
        shared_video_cache=shared_video_cache,
    )
    from ember.ecp.policy_response_writer.shared_training import (
        resume_cursor,
        train,
    )

    start_step, metrics_rows = resume_cursor(
        runtime,
        optimizer=optimizer,
        scheduler=scheduler,
        stop=stop,
    )
    checkpoint_steps = {
        int(cell["warmup_updates"]) + int(value)
        for value in cell["checkpoint_effective_updates"]
    }
    training = train(
        runtime,
        cache,
        parameters=parameters,
        optimizer=optimizer,
        scheduler=scheduler,
        task_owners=task_owners,
        execution_owners=execution_owners,
        video_splits=video_splits,
        start_step=start_step,
        stop=stop,
        metrics_rows=metrics_rows,
        checkpoint_steps=checkpoint_steps,
    )
    from ember.ecp.policy_response_writer.shared_evaluation import (
        evaluate_checkpoints,
    )

    evaluation_task_owners = task_owners
    evaluation_splits = video_splits
    if runtime.args.mode == "formal":
        evaluation_splits = {
            task: video_splits[task] for task in _evaluation_ids(runtime)
        }
        evaluation_task_owners = balanced_task_owners(
            evaluation_task_costs(runtime, evaluation_splits),
            runtime.context.world_size,
        )
    if runtime.args.mode == "profile" and runtime.args.task is None:
        evaluations, evaluation_seconds = {}, 0.0
    else:
        evaluations, evaluation_seconds = evaluate_checkpoints(
            runtime,
            cache,
            task_owners=evaluation_task_owners,
            video_splits=evaluation_splits,
        )
    capture_records = sorted(
        _gather_flat(runtime, cache.capture_records),
        key=lambda value: (value["task_id"], value["video_demo"]),
    )
    result = _build_result(
        runtime,
        stop=stop,
        start_step=start_step,
        total=total,
        task_owners=task_owners,
        evaluation_task_owners=evaluation_task_owners,
        execution_plan=execution_plan,
        normalizers=normalizers,
        training=training,
        evaluations=evaluations,
        evaluation_seconds=evaluation_seconds,
        capture_records=capture_records,
        parameters=parameters,
        started=started,
    )
    if shared_video_cache is not None:
        _remove_shared_video_cache(runtime, cache, shared_video_cache)
    runtime.writer.train()
    return result
