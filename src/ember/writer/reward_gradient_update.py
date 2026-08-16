"""Aggregate direct-factor reward gradients and measure task coexistence."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.lora import copy_task_lora_state_
from ember.writer.as_step import ParameterSlice, assign_flat_gradient
from ember.writer.errors import WriterModelError
from ember.writer.model import WriterConditioningState
from ember.writer.reward_preference import (
    functional_matched_stratified_occupancy_endpoint_margin,
)

if TYPE_CHECKING:
    from ember.reward.protocol import RewardTask
    from ember.reward.rollout import RewardTrajectory
    from ember.writer.reward_training import RewardRuntime


@dataclass(frozen=True)
class AppliedStep:
    active_tasks: int
    gradient_norm: float
    gradient_rms: float
    parameter_delta_rms: Mapping[str, float]
    gradient_coexistence: Mapping[str, Any]
    commitment_geometry: Mapping[str, Any]
    commitment_preference_rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class RewardPreferenceView:
    conditioning_state: WriterConditioningState
    condition_video_offsets: torch.Tensor
    before_preference_margin: float


@dataclass(frozen=True)
class RewardProbe:
    global_task_id: int
    suite: str
    conditioning_state: WriterConditioningState
    condition_video_offsets: torch.Tensor
    before_lora: Mapping[str, torch.Tensor]
    query: Mapping[str, torch.Tensor]
    before_action: torch.Tensor
    policy_noise_seed: int
    preference_batch: Mapping[str, torch.Tensor]
    preference_trajectory_ids: torch.Tensor
    before_preference_margin: float
    preference_views: tuple[RewardPreferenceView, ...]


@torch.inference_mode()
def fixed_action_for_lora(
    runtime: RewardRuntime,
    query: Mapping[str, torch.Tensor],
    lora: Mapping[str, torch.Tensor],
    *,
    policy_noise_seed: int,
) -> torch.Tensor:
    """Evaluate one retained query through the fixed batch-one action path."""

    try:
        copy_task_lora_state_(runtime.policy, lora, runtime.lora_contract)
        generator = torch.Generator(device="cpu").manual_seed(policy_noise_seed)
        noise = torch.randn(
            (
                1,
                int(runtime.policy.config.chunk_size),
                int(runtime.policy.config.max_action_dim),
            ),
            generator=generator,
        ).to(runtime.context.device)
        device_query = {
            name: value.to(runtime.context.device, non_blocking=True)
            for name, value in query.items()
        }
        with torch.autocast(
            device_type=runtime.context.device.type,
            dtype=torch.bfloat16,
            enabled=runtime.context.device.type == "cuda",
        ):
            action = runtime.policy.predict_action_chunk(
                device_query,
                noise=noise,
                num_steps=int(runtime.config["environment"]["num_inference_steps"]),
            )
        return action.detach().cpu()
    finally:
        copy_task_lora_state_(
            runtime.policy, runtime.identity_state, runtime.lora_contract
        )


def make_reward_probe(
    runtime: RewardRuntime,
    task: RewardTask,
    conditioning_state: WriterConditioningState,
    condition_video_offsets: torch.Tensor,
    candidate_lora: Mapping[str, torch.Tensor],
    candidate: Sequence[RewardTrajectory],
    labels: Sequence[str],
    preference_batch: Mapping[str, torch.Tensor],
    preference_trajectory_ids: torch.Tensor,
    before_preference_margin: float,
    preference_views: tuple[RewardPreferenceView, ...],
) -> RewardProbe:
    index = next(
        index
        for index, label in enumerate(labels)
        if label in {"candidate", "reference"}
    )
    trajectory = candidate[index]
    query = {name: value.clone() for name, value in trajectory.observations[0].items()}
    return RewardProbe(
        global_task_id=task.global_task_id,
        suite=task.suite,
        conditioning_state=conditioning_state,
        condition_video_offsets=condition_video_offsets,
        before_lora={
            name: value.detach().clone() for name, value in candidate_lora.items()
        },
        query=query,
        before_action=fixed_action_for_lora(
            runtime,
            query,
            candidate_lora,
            policy_noise_seed=trajectory.policy_noise_seeds[0],
        ),
        policy_noise_seed=trajectory.policy_noise_seeds[0],
        preference_batch=preference_batch,
        preference_trajectory_ids=preference_trajectory_ids,
        before_preference_margin=before_preference_margin,
        preference_views=preference_views,
    )


def lora_response(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    sums: dict[str, torch.Tensor | None] = dict.fromkeys(
        ("lora_a", "lora_b", "effective_ba")
    )
    counts = {name: 0 for name in sums}
    by_kind: dict[str, torch.Tensor | None] = dict.fromkeys(("q", "v", "action"))
    kind_counts = {name: 0 for name in by_kind}
    for name, before_a in before.items():
        if not name.endswith(".lora_A.default.weight"):
            continue
        module = name.removesuffix(".lora_A.default.weight")
        kind = (
            "q"
            if module.endswith("q_proj")
            else "v" if module.endswith("v_proj") else "action"
        )
        b_name = name.replace(".lora_A.default.weight", ".lora_B.default.weight")
        after_a, before_b, after_b = after[name], before[b_name], after[b_name]
        values = {
            "lora_a": after_a.float() - before_a.float(),
            "lora_b": after_b.float() - before_b.float(),
            "effective_ba": after_b.float() @ after_a.float()
            - before_b.float() @ before_a.float(),
        }
        for key, value in values.items():
            squared = value.square().sum()
            sums[key] = squared if sums[key] is None else sums[key] + squared
            counts[key] += value.numel()
            if key == "effective_ba":
                by_kind[kind] = (
                    squared if by_kind[kind] is None else by_kind[kind] + squared
                )
                kind_counts[kind] += value.numel()
    response = {
        f"{key}_response_rms": math.sqrt(float(value) / counts[key])
        for key, value in sums.items()
        if value is not None and counts[key]
    }
    response["effective_ba_response_rms_by_kind"] = {
        kind: math.sqrt(float(value) / kind_counts[kind])
        for kind, value in by_kind.items()
        if value is not None and kind_counts[kind]
    }
    return response


@torch.inference_mode()
def evaluate_preference_views(
    runtime: RewardRuntime, probe: RewardProbe
) -> tuple[Mapping[str, Any], ...]:
    """Re-evaluate the exact four-view matched panel at current Writer parameters."""

    if len(probe.preference_views) != 4:
        raise WriterModelError("direct commitment lost four video views")
    rows = []
    for index, view in enumerate(probe.preference_views):
        with torch.autocast(
            device_type=runtime.context.device.type,
            dtype=torch.bfloat16,
            enabled=runtime.context.device.type == "cuda",
        ):
            encoded = runtime.writer.compile_conditioning_state(
                view.conditioning_state,
                view.condition_video_offsets,
                use_query_delta=True,
            )
            lora = runtime.writer.decode_output(encoded)
        preference = functional_matched_stratified_occupancy_endpoint_margin(
            runtime.policy,
            lora,
            runtime.lora_contract,
            probe.preference_batch,
            probe.preference_trajectory_ids,
            endpoint_action_batch_size=int(
                runtime.config["optimization"]["endpoint_action_batch_size"]
            ),
            num_inference_steps=int(
                runtime.config["environment"]["num_inference_steps"]
            ),
            device=runtime.context.device,
        )
        delta = preference["preference_margin"] - view.before_preference_margin
        rows.append(
            {
                "task_id": probe.global_task_id,
                "suite": probe.suite,
                "view_index": index,
                "before_preference_margin": view.before_preference_margin,
                "after_preference_margin": preference["preference_margin"],
                "preference_margin_delta": delta,
                "after_preference_objective": preference["preference_objective"],
                "preference_descent": delta < 0,
            }
        )
    return tuple(rows)


@torch.inference_mode()
def probe_after_update(
    runtime: RewardRuntime,
    probe: RewardProbe,
    preference_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    started = time.monotonic()
    with torch.autocast(
        device_type=runtime.context.device.type,
        dtype=torch.bfloat16,
        enabled=runtime.context.device.type == "cuda",
    ):
        encoded = runtime.writer.compile_conditioning_state(
            probe.conditioning_state,
            probe.condition_video_offsets,
            use_query_delta=True,
        )
        after = runtime.writer.decode_output(encoded)
    response = lora_response(probe.before_lora, after)
    action = fixed_action_for_lora(
        runtime,
        probe.query,
        after,
        policy_noise_seed=probe.policy_noise_seed,
    )
    response["fixed_action_response_rms"] = float(
        (action.float() - probe.before_action.float()).square().mean().sqrt()
    )
    if len(preference_rows) != 4 or {
        int(row["task_id"]) for row in preference_rows
    } != {probe.global_task_id}:
        raise WriterModelError("direct commitment lost task-view evidence")
    first = preference_rows[0]
    response.update(
        before_preference_margin=first["before_preference_margin"],
        after_preference_margin=first["after_preference_margin"],
        preference_margin_delta=first["preference_margin_delta"],
        after_preference_objective=first["after_preference_objective"],
        view_preference_probes=list(preference_rows),
        all_view_preference_descent=all(
            bool(row["preference_descent"]) for row in preference_rows
        ),
    )
    response["probe_seconds"] = time.monotonic() - started
    return {"task_id": probe.global_task_id, "suite": probe.suite, **response}


@torch.inference_mode()
def probes_after_update(
    runtime: RewardRuntime,
    probes: Sequence[RewardProbe],
    preference_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        probe_after_update(
            runtime,
            probe,
            tuple(
                row
                for row in preference_rows
                if int(row["task_id"]) == probe.global_task_id
            ),
        )
        for probe in probes
    ]


def _trainable_named(
    runtime: RewardRuntime,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    return tuple(
        (name, value)
        for name, value in runtime.writer.named_parameters()
        if value.requires_grad
    )


def _distributed_task_gradient_panel(
    runtime: RewardRuntime,
    task_gradients: Mapping[int, torch.Tensor],
) -> tuple[list[int], torch.Tensor]:
    task_ids = tuple(task.global_task_id for task in runtime.tasks)
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    matrix = torch.zeros(
        len(task_ids),
        runtime.gradient_layout[-1].stop,
        dtype=torch.float32,
        device=runtime.context.device,
    )
    active = torch.zeros(
        len(task_ids), dtype=torch.long, device=runtime.context.device
    )
    for task_id, gradient in task_gradients.items():
        index = task_index[int(task_id)]
        matrix[index].copy_(gradient)
        active[index] = 1
    if runtime.context.world_size > 1:
        dist.all_reduce(matrix, op=dist.ReduceOp.SUM)
        dist.all_reduce(active, op=dist.ReduceOp.SUM)
    selected = active == 1
    rows = matrix[selected]
    selected_ids = [
        task_id
        for task_id, keep in zip(task_ids, selected.cpu().tolist(), strict=True)
        if keep
    ]
    if rows.shape[0] == 0 or bool((active > 1).any()):
        raise WriterModelError("direct-factor task gradients lost unique ownership")
    return selected_ids, rows


def median_capped_task_mean(
    rows: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Cap only above-median task norms, without amplifying small tangents."""

    if (
        rows.ndim != 2
        or rows.dtype != torch.float32
        or rows.shape[0] == 0
        or not bool(torch.isfinite(rows).all())
    ):
        raise WriterModelError("median-capped task tangent panel changed")
    norms = torch.linalg.vector_norm(rows, dim=1)
    if not bool((norms > 0).all()):
        raise WriterModelError("median-capped task tangent lost a nonzero row")
    ordered = norms.sort().values
    middle = ordered.shape[0] // 2
    median = (
        ordered[middle]
        if ordered.shape[0] % 2
        else (ordered[middle - 1] + ordered[middle]) * 0.5
    )
    scales = torch.minimum(torch.ones_like(norms), median / norms)
    capped = rows * scales[:, None]
    capped_norms = torch.linalg.vector_norm(capped, dim=1)
    shared = capped.mean(dim=0)
    if not bool(torch.isfinite(shared).all()) or not bool(torch.count_nonzero(shared)):
        raise WriterModelError("median-capped shared task tangent is invalid")
    return shared, {
        "kind": "median_upper_norm_cap_without_small_task_amplification",
        "raw_task_gradient_norms": norms.cpu().tolist(),
        "median_task_gradient_norm": float(median),
        "task_gradient_scales": scales.cpu().tolist(),
        "capped_task_gradient_norms": capped_norms.cpu().tolist(),
        "capped_task_count": int((scales < 1).sum()),
        "small_task_amplification": False,
    }


def _coexistence(
    runtime: RewardRuntime,
    selected_ids: Sequence[int],
    rows: torch.Tensor,
    shared_mean: torch.Tensor,
    final_delta: torch.Tensor,
    task_tangent_balance: Mapping[str, Any],
) -> dict[str, Any]:

    row_norms = torch.linalg.vector_norm(rows, dim=1)
    mean_norm = torch.linalg.vector_norm(shared_mean)
    dots = rows @ shared_mean
    cosines = dots / (row_norms * mean_norm).clamp_min(1e-30)
    unit = rows / row_norms[:, None].clamp_min(1e-30)
    pairwise = unit @ unit.T
    if rows.shape[0] > 1:
        offdiag = pairwise[
            ~torch.eye(rows.shape[0], dtype=torch.bool, device=rows.device)
        ]
        pairwise_values = (
            torch.stack((offdiag.mean(), offdiag.min(), offdiag.max())).cpu().tolist()
        )
        pairwise_summary = dict(
            zip(("mean", "minimum", "maximum"), pairwise_values, strict=True)
        )
    else:
        pairwise_summary = {"mean": 1.0, "minimum": 1.0, "maximum": 1.0}

    parameter_energy = {}
    for (name, _), item in zip(
        _trainable_named(runtime), runtime.gradient_layout, strict=True
    ):
        values = rows[:, item.start : item.stop]
        mean_values = shared_mean[item.start : item.stop]
        task_rms, shared_rms = (
            torch.stack(
                (
                    values.square().mean(dim=1).sqrt().mean(),
                    mean_values.square().mean().sqrt(),
                )
            )
            .cpu()
            .tolist()
        )
        parameter_energy[name] = {
            "task_gradient_rms_mean": task_rms,
            "shared_mean_gradient_rms": shared_rms,
        }

    delta_norm = torch.linalg.vector_norm(final_delta)
    descent_dots = -(rows @ final_delta)
    descent_cosines = descent_dots / (row_norms * delta_norm).clamp_min(1e-30)
    task_values = (
        torch.stack((row_norms, dots, cosines, descent_dots, descent_cosines), dim=1)
        .cpu()
        .tolist()
    )
    coverage, cosine_mean, cosine_minimum = (
        torch.stack(((dots > 0).float().mean(), cosines.mean(), cosines.min()))
        .cpu()
        .tolist()
    )
    final_coverage, final_cosine_mean, final_cosine_minimum = (
        torch.stack(
            (
                (descent_dots > 0).float().mean(),
                descent_cosines.mean(),
                descent_cosines.min(),
            )
        )
        .cpu()
        .tolist()
    )
    return {
        "active_task_ids": list(selected_ids),
        "task_tangent_balance": dict(task_tangent_balance),
        "shared_mean_descent_coverage": coverage,
        "task_to_shared_cosine_mean": cosine_mean,
        "task_to_shared_cosine_minimum": cosine_minimum,
        "final_delta_descent_coverage": final_coverage,
        "task_to_final_descent_cosine_mean": final_cosine_mean,
        "task_to_final_descent_cosine_minimum": final_cosine_minimum,
        "pairwise_task_gradient_cosine": pairwise_summary,
        "per_task": [
            {
                "task_id": task_id,
                "gradient_norm": values[0],
                "dot_shared_mean": values[1],
                "cosine_shared_mean": values[2],
                "dot_final_descent": values[3],
                "cosine_final_descent": values[4],
            }
            for task_id, values in zip(selected_ids, task_values, strict=True)
        ],
        "per_parameter": parameter_energy,
    }


def preconditioned_candidate_commitment(
    optimizer_gradient: torch.Tensor,
    adam_candidate_delta: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the exact finite, nonzero AdamW candidate and its geometry."""

    if (
        optimizer_gradient.ndim != 1
        or adam_candidate_delta.shape != optimizer_gradient.shape
        or optimizer_gradient.dtype != torch.float32
        or adam_candidate_delta.dtype != torch.float32
        or not bool(torch.isfinite(optimizer_gradient).all())
        or not bool(torch.isfinite(adam_candidate_delta).all())
    ):
        raise WriterModelError("preconditioned commitment inputs changed")
    gradient_norm = torch.linalg.vector_norm(optimizer_gradient)
    candidate_norm = torch.linalg.vector_norm(adam_candidate_delta)
    if float(gradient_norm) <= 0 or float(candidate_norm) <= 0:
        raise WriterModelError("preconditioned commitment lost a nonzero direction")
    negative_gradient = -optimizer_gradient
    values = (
        torch.stack(
            (
                gradient_norm,
                candidate_norm,
                torch.dot(adam_candidate_delta, negative_gradient)
                / (candidate_norm * gradient_norm),
            )
        )
        .cpu()
        .tolist()
    )
    count = optimizer_gradient.numel()
    return adam_candidate_delta, {
        "optimizer_gradient_l2": values[0],
        "optimizer_gradient_rms": values[0] / math.sqrt(count),
        "adam_candidate_delta_l2": values[1],
        "adam_candidate_delta_rms": values[1] / math.sqrt(count),
        "adam_candidate_to_negative_optimizer_gradient_cosine": values[2],
        "full_candidate_to_adam_candidate_cosine": 1.0,
        "radius_relative_error": 0.0,
    }


def _write_parameter_delta(
    named: Sequence[tuple[str, torch.nn.Parameter]],
    layout: Sequence[ParameterSlice],
    before: Mapping[str, torch.Tensor],
    delta: torch.Tensor,
) -> None:
    with torch.no_grad():
        for (name, value), item in zip(named, layout, strict=True):
            value.copy_(
                before[name]
                + delta[item.start : item.stop].view_as(value).to(dtype=value.dtype)
            )


def _gather_global_preference_rows(
    runtime: RewardRuntime,
    local_rows: Sequence[Mapping[str, Any]],
    *,
    expected_task_count: int,
) -> tuple[Mapping[str, Any], ...]:
    shards: list[Any] = [None] * runtime.context.world_size
    if runtime.context.world_size > 1:
        dist.all_gather_object(shards, [dict(row) for row in local_rows])
    else:
        shards[0] = [dict(row) for row in local_rows]
    rows = tuple(dict(row) for shard in shards for row in shard)
    keys = [(int(row["task_id"]), int(row["view_index"])) for row in rows]
    task_ids = {task_id for task_id, _ in keys}
    valid_views = all(
        sorted(view for task_id, view in keys if task_id == target) == list(range(4))
        for target in task_ids
    )
    if (
        expected_task_count <= 0
        or len(task_ids) != expected_task_count
        or len(rows) != 4 * expected_task_count
        or len(set(keys)) != len(keys)
        or not valid_views
    ):
        raise WriterModelError("global commitment lost active task-view coverage")
    return tuple(
        sorted(rows, key=lambda row: (int(row["task_id"]), int(row["view_index"])))
    )


def _direct_adam_commitment(
    *,
    runtime: RewardRuntime,
    named: Sequence[tuple[str, torch.nn.Parameter]],
    layout: Sequence[ParameterSlice],
    before: Mapping[str, torch.Tensor],
    full_delta: torch.Tensor,
    base_geometry: Mapping[str, float],
    evaluator: Callable[[float], Sequence[Mapping[str, Any]]],
    expected_task_count: int,
) -> tuple[torch.Tensor, tuple[Mapping[str, Any], ...], dict[str, Any]]:
    started = time.monotonic()
    final_delta = full_delta
    _write_parameter_delta(named, layout, before, final_delta)
    accepted_rows = _gather_global_preference_rows(
        runtime, evaluator(1.0), expected_task_count=expected_task_count
    )
    margin_deltas = [
        float(row["preference_margin_delta"]) for row in accepted_rows
    ]
    if not all(math.isfinite(value) for value in margin_deltas):
        raise WriterModelError("direct Adam diagnostic margin is nonfinite")
    all_view_descent = all(value < 0 for value in margin_deltas)
    trials = [
        {
            "backtrack_index": 0,
            "radius_scale": 1.0,
            "global_task_view_preference_margin_deltas": [
                {
                    "task_id": int(row["task_id"]),
                    "view_index": int(row["view_index"]),
                    "delta": float(row["preference_margin_delta"]),
                }
                for row in accepted_rows
            ],
            "all_active_task_view_preference_descent": all_view_descent,
        }
    ]
    final_norm = torch.linalg.vector_norm(final_delta)
    candidate_norm = float(base_geometry["adam_candidate_delta_l2"])
    expected_norm = candidate_norm
    final_norm_value = float(final_norm)
    final_to_candidate_cosine = float(
        base_geometry["full_candidate_to_adam_candidate_cosine"]
    )
    final_to_gradient_cosine = float(
        base_geometry["adam_candidate_to_negative_optimizer_gradient_cosine"]
    )
    radius_error = abs(final_norm_value - expected_norm) / expected_norm
    count = full_delta.numel()
    geometry = {
        **base_geometry,
        "search_kind": "direct_actual_adam_candidate_with_all_view_diagnostic",
        "global_active_task_ids": sorted(
            {int(row["task_id"]) for row in accepted_rows}
        ),
        "global_active_task_count": expected_task_count,
        "global_task_view_count": len(accepted_rows),
        "max_backtracks": 0,
        "search_accepted": True,
        "accepted_backtrack_index": 0,
        "accepted_radius_scale": 1.0,
        "accepted_expected_delta_l2": expected_norm,
        "final_delta_l2": final_norm_value,
        "final_delta_rms": final_norm_value / math.sqrt(count),
        "final_to_adam_candidate_cosine": final_to_candidate_cosine,
        "final_to_negative_optimizer_gradient_cosine": final_to_gradient_cosine,
        "radius_relative_error": radius_error,
        "search_trials": trials,
        "search_trial_count": len(trials),
        "all_active_task_view_preference_descent_diagnostic": all_view_descent,
        "descending_task_view_count": sum(value < 0 for value in margin_deltas),
        "diagnostic_reference": "stored_gradient_path_preupdate_margin",
        "repeated_step0_baseline_forward": False,
        "search_seconds": time.monotonic() - started,
    }
    return final_delta, accepted_rows, geometry


def apply_reward_step(
    runtime: RewardRuntime,
    local_active_tasks: int,
    task_gradients: Mapping[int, torch.Tensor],
    commitment_evaluator: Callable[[float], Sequence[Mapping[str, Any]]],
) -> AppliedStep:
    if local_active_tasks != len(task_gradients):
        raise WriterModelError("median-capped local active-task panel changed")
    selected_ids, task_rows = _distributed_task_gradient_panel(
        runtime, task_gradients
    )
    active_tasks = len(selected_ids)
    shared_gradient, task_tangent_balance = median_capped_task_mean(task_rows)
    assign_flat_gradient(shared_gradient, runtime.gradient_layout)
    clip = float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"])
    grad_norm = torch.nn.utils.clip_grad_norm_(runtime.trainable_parameters, clip)
    named = _trainable_named(runtime)
    before = {name: value.detach().clone() for name, value in named}
    committed_gradient = torch.empty_like(shared_gradient)
    for item in runtime.gradient_layout:
        if item.parameter.grad is None:
            raise WriterModelError(
                "direct Adam commitment lost a parameter gradient"
            )
        committed_gradient[item.start : item.stop].copy_(
            item.parameter.grad.detach().reshape(-1).float()
        )
    runtime.optimizer.step()
    candidate_delta = torch.empty_like(shared_gradient)
    for (name, value), item in zip(named, runtime.gradient_layout, strict=True):
        candidate_delta[item.start : item.stop].copy_(
            (value.detach() - before[name]).reshape(-1).float()
        )
    full_delta, base_commitment = preconditioned_candidate_commitment(
        committed_gradient, candidate_delta
    )
    final_delta, preference_rows, commitment = _direct_adam_commitment(
        runtime=runtime,
        named=named,
        layout=runtime.gradient_layout,
        before=before,
        full_delta=full_delta,
        base_geometry=base_commitment,
        evaluator=commitment_evaluator,
        expected_task_count=active_tasks,
    )
    delta = {
        name: float((value.detach() - before[name]).float().square().mean().sqrt())
        for name, value in named
    }
    coexistence = _coexistence(
        runtime,
        selected_ids,
        task_rows,
        committed_gradient,
        final_delta,
        task_tangent_balance,
    )
    return AppliedStep(
        active_tasks=active_tasks,
        gradient_norm=float(grad_norm),
        gradient_rms=float(shared_gradient.square().mean().sqrt()),
        parameter_delta_rms=delta,
        gradient_coexistence=coexistence,
        commitment_geometry=commitment,
        commitment_preference_rows=preference_rows,
    )


def apply_direct_reward_step(
    runtime: RewardRuntime,
    local_active_tasks: int,
    task_gradients: Mapping[int, torch.Tensor],
    probes: Sequence[RewardProbe],
) -> AppliedStep:
    if len(probes) != local_active_tasks:
        raise WriterModelError("direct commitment lost local active-task probes")
    return apply_reward_step(
        runtime,
        local_active_tasks,
        task_gradients,
        lambda _scale: tuple(
            row for probe in probes for row in evaluate_preference_views(runtime, probe)
        ),
    )
