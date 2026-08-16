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
        raise WriterModelError("backtracking commitment lost four video views")
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
    probe: RewardProbe | None,
    preference_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if probe is None:
        return None
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
    if len(preference_rows) != 4:
        raise WriterModelError("backtracking commitment lost accepted view evidence")
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


def _trainable_named(
    runtime: RewardRuntime,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    return tuple(
        (name, value)
        for name, value in runtime.writer.named_parameters()
        if value.requires_grad
    )


def _coexistence(
    runtime: RewardRuntime,
    task_gradients: Mapping[int, torch.Tensor],
    shared_mean: torch.Tensor,
    final_delta: torch.Tensor,
) -> dict[str, Any]:
    task_ids = tuple(task.global_task_id for task in runtime.tasks)
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    matrix = torch.zeros(
        len(task_ids),
        shared_mean.numel(),
        dtype=torch.float32,
        device=shared_mean.device,
    )
    active = torch.zeros(len(task_ids), dtype=torch.long, device=shared_mean.device)
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
        "active_task_ids": selected_ids,
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
    """Commit along the actual AdamW candidate ray before native backtracking."""

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


def _baseline_preference_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "view_index": int(row["view_index"]),
            "before_preference_margin": float(row["after_preference_margin"]),
            "after_preference_margin": float(row["after_preference_margin"]),
            "preference_margin_delta": 0.0,
            "after_preference_objective": float(row["after_preference_objective"]),
            "preference_descent": False,
            "gradient_path_before_preference_margin": float(
                row["before_preference_margin"]
            ),
            "gradient_path_to_inference_baseline_margin_delta": float(
                row["after_preference_margin"] - row["before_preference_margin"]
            ),
        }
        for row in rows
    )


def _rebase_preference_rows(
    rows: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    if len(rows) != 4 or len(baseline) != 4:
        raise WriterModelError("monotone backtracking lost four view margins")
    rebased = []
    for row, base in zip(rows, baseline, strict=True):
        if int(row["view_index"]) != int(base["view_index"]):
            raise WriterModelError("monotone backtracking view order changed")
        before_margin = float(base["after_preference_margin"])
        after_margin = float(row["after_preference_margin"])
        delta = after_margin - before_margin
        rebased.append(
            {
                "view_index": int(row["view_index"]),
                "before_preference_margin": before_margin,
                "after_preference_margin": after_margin,
                "preference_margin_delta": delta,
                "after_preference_objective": float(row["after_preference_objective"]),
                "preference_descent": delta < 0,
                "gradient_path_before_preference_margin": float(
                    base["before_preference_margin"]
                ),
                "gradient_path_to_inference_baseline_margin_delta": float(
                    base["after_preference_margin"] - base["before_preference_margin"]
                ),
            }
        )
    return tuple(rebased)


def _monotone_backtracking_commitment(
    *,
    named: Sequence[tuple[str, torch.nn.Parameter]],
    layout: Sequence[ParameterSlice],
    before: Mapping[str, torch.Tensor],
    full_delta: torch.Tensor,
    base_geometry: Mapping[str, float],
    evaluator: Callable[[float], Sequence[Mapping[str, Any]]],
    max_backtracks: int,
) -> tuple[torch.Tensor, tuple[Mapping[str, Any], ...], dict[str, Any]]:
    if max_backtracks < 0:
        raise WriterModelError("monotone backtracking count changed")
    started = time.monotonic()
    accepted_scale = 0.0
    accepted_index: int | None = None
    accepted_rows: tuple[Mapping[str, Any], ...] = ()
    trials = []
    zero_delta = torch.zeros_like(full_delta)
    _write_parameter_delta(named, layout, before, zero_delta)
    baseline_rows = tuple(evaluator(0.0))
    if len(baseline_rows) != 4:
        raise WriterModelError("monotone backtracking lost four baseline margins")
    baseline_offsets = [
        float(row["after_preference_margin"] - row["before_preference_margin"])
        for row in baseline_rows
    ]
    if not all(math.isfinite(value) for value in baseline_offsets):
        raise WriterModelError("monotone backtracking baseline margin is nonfinite")
    for index in range(max_backtracks + 1):
        scale = 2.0**-index
        trial_delta = full_delta * scale
        _write_parameter_delta(named, layout, before, trial_delta)
        rows = _rebase_preference_rows(tuple(evaluator(scale)), baseline_rows)
        margin_deltas = [float(row["preference_margin_delta"]) for row in rows]
        if not all(math.isfinite(value) for value in margin_deltas):
            raise WriterModelError("monotone backtracking margin is nonfinite")
        accepted = all(value < 0 for value in margin_deltas)
        trials.append(
            {
                "backtrack_index": index,
                "radius_scale": scale,
                "view_preference_margin_deltas": margin_deltas,
                "all_view_preference_descent": accepted,
            }
        )
        if accepted:
            accepted_scale = scale
            accepted_index = index
            accepted_rows = rows
            break
    if accepted_index is None:
        final_delta = torch.zeros_like(full_delta)
        _write_parameter_delta(named, layout, before, final_delta)
        accepted_rows = _baseline_preference_rows(baseline_rows)
    else:
        final_delta = full_delta * accepted_scale
    final_norm = torch.linalg.vector_norm(final_delta)
    candidate_norm = float(base_geometry["adam_candidate_delta_l2"])
    expected_norm = candidate_norm * accepted_scale
    final_norm_value = float(final_norm)
    if accepted_index is None:
        final_to_candidate_cosine = 0.0
        final_to_gradient_cosine = 0.0
        radius_error = 0.0
    else:
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
        "search_kind": "first_all_view_monotone_power_of_two_backtracking",
        "max_backtracks": max_backtracks,
        "search_accepted": accepted_index is not None,
        "accepted_backtrack_index": accepted_index,
        "accepted_radius_scale": accepted_scale,
        "accepted_expected_delta_l2": expected_norm,
        "final_delta_l2": final_norm_value,
        "final_delta_rms": final_norm_value / math.sqrt(count),
        "final_to_adam_candidate_cosine": final_to_candidate_cosine,
        "final_to_negative_optimizer_gradient_cosine": final_to_gradient_cosine,
        "radius_relative_error": radius_error,
        "search_trials": trials,
        "search_trial_count": len(trials),
        "inference_baseline_preference_margins": [
            float(row["after_preference_margin"]) for row in baseline_rows
        ],
        "gradient_path_to_inference_baseline_margin_deltas": baseline_offsets,
        "search_seconds": time.monotonic() - started,
    }
    return final_delta, accepted_rows, geometry


def apply_reward_step(
    runtime: RewardRuntime,
    gradient_sum: torch.Tensor,
    local_active_tasks: int,
    task_gradients: Mapping[int, torch.Tensor],
    commitment_evaluator: Callable[[float], Sequence[Mapping[str, Any]]],
) -> AppliedStep:
    active = torch.tensor(
        local_active_tasks, dtype=torch.long, device=runtime.context.device
    )
    if runtime.context.world_size > 1:
        dist.all_reduce(gradient_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(active, op=dist.ReduceOp.SUM)
    active_tasks = int(active)
    if active_tasks <= 0:
        raise WriterModelError("direct-factor cycle has no discordant success")
    gradient_sum.div_(active_tasks)
    if not bool(torch.isfinite(gradient_sum).all()) or not bool(
        torch.count_nonzero(gradient_sum)
    ):
        raise WriterModelError("direct-factor gradient is invalid")
    assign_flat_gradient(gradient_sum, runtime.gradient_layout)
    clip = float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"])
    grad_norm = torch.nn.utils.clip_grad_norm_(runtime.trainable_parameters, clip)
    named = _trainable_named(runtime)
    before = {name: value.detach().clone() for name, value in named}
    committed_gradient = torch.empty_like(gradient_sum)
    for item in runtime.gradient_layout:
        if item.parameter.grad is None:
            raise WriterModelError(
                "monotone backtracking commitment lost a parameter gradient"
            )
        committed_gradient[item.start : item.stop].copy_(
            item.parameter.grad.detach().reshape(-1).float()
        )
    runtime.optimizer.step()
    candidate_delta = torch.empty_like(gradient_sum)
    for (name, value), item in zip(named, runtime.gradient_layout, strict=True):
        candidate_delta[item.start : item.stop].copy_(
            (value.detach() - before[name]).reshape(-1).float()
        )
    full_delta, base_commitment = preconditioned_candidate_commitment(
        committed_gradient, candidate_delta
    )
    final_delta, preference_rows, commitment = _monotone_backtracking_commitment(
        named=named,
        layout=runtime.gradient_layout,
        before=before,
        full_delta=full_delta,
        base_geometry=base_commitment,
        evaluator=commitment_evaluator,
        max_backtracks=int(runtime.config["commitment"]["max_backtracks"]),
    )
    delta = {
        name: float((value.detach() - before[name]).float().square().mean().sqrt())
        for name, value in named
    }
    coexistence = _coexistence(runtime, task_gradients, committed_gradient, final_delta)
    return AppliedStep(
        active_tasks=active_tasks,
        gradient_norm=float(grad_norm),
        gradient_rms=float(gradient_sum.square().mean().sqrt()),
        parameter_delta_rms=delta,
        gradient_coexistence=coexistence,
        commitment_geometry=commitment,
        commitment_preference_rows=preference_rows,
    )


def apply_monotone_reward_step(
    runtime: RewardRuntime,
    gradient_sum: torch.Tensor,
    local_active_tasks: int,
    task_gradients: Mapping[int, torch.Tensor],
    probe: RewardProbe | None,
) -> AppliedStep:
    if probe is None:
        raise WriterModelError("monotone backtracking has no active four-view probe")
    return apply_reward_step(
        runtime,
        gradient_sum,
        local_active_tasks,
        task_gradients,
        lambda _scale: evaluate_preference_views(runtime, probe),
    )
