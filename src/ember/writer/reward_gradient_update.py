"""Aggregate direct-factor reward gradients and measure task coexistence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import torch
import torch.distributed as dist

from ember.lora import copy_task_lora_state_
from ember.writer.as_step import assign_flat_gradient
from ember.writer.errors import WriterModelError
from ember.writer.model import WriterConditioningState
from ember.writer.reward_preference import functional_paired_common_state_margin

if TYPE_CHECKING:
    from ember.writer.reward_training import RewardRuntime


@dataclass(frozen=True)
class AppliedStep:
    active_tasks: int
    gradient_norm: float
    gradient_rms: float
    parameter_delta_rms: Mapping[str, float]
    gradient_coexistence: Mapping[str, Any]


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
    cycle: int
    preference_batch: Mapping[str, torch.Tensor]
    before_preference_margin: float


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
        kind = "q" if module.endswith("q_proj") else "v" if module.endswith("v_proj") else "action"
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
                by_kind[kind] = squared if by_kind[kind] is None else by_kind[kind] + squared
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
def probe_after_update(
    runtime: RewardRuntime, probe: RewardProbe | None
) -> dict[str, Any] | None:
    if probe is None:
        return None
    with torch.autocast(
        device_type=runtime.context.device.type,
        dtype=torch.bfloat16,
        enabled=runtime.context.device.type == "cuda",
    ):
        encoded = runtime.writer.compile_conditioning_state(
            probe.conditioning_state, probe.condition_video_offsets, use_query_delta=True
        )
        after = runtime.writer.decode_output(encoded)
    response = lora_response(probe.before_lora, after)
    try:
        copy_task_lora_state_(runtime.policy, after, runtime.lora_contract)
        generator = torch.Generator(device="cpu").manual_seed(probe.policy_noise_seed)
        noise = torch.randn(
            (1, int(runtime.policy.config.chunk_size), int(runtime.policy.config.max_action_dim)),
            generator=generator,
        ).to(runtime.context.device)
        query = {
            name: value.to(runtime.context.device, non_blocking=True)
            for name, value in probe.query.items()
        }
        with torch.autocast(
            device_type=runtime.context.device.type,
            dtype=torch.bfloat16,
            enabled=runtime.context.device.type == "cuda",
        ):
            action = runtime.policy.predict_action_chunk(
                query,
                noise=noise,
                num_steps=int(runtime.config["environment"]["num_inference_steps"]),
            )
    finally:
        copy_task_lora_state_(runtime.policy, runtime.identity_state, runtime.lora_contract)
    response["fixed_action_response_rms"] = float(
        (action.cpu().float() - probe.before_action.float()).square().mean().sqrt()
    )
    if runtime.args.mode == "smoke":
        preference = functional_paired_common_state_margin(
            runtime.policy,
            after,
            runtime.lora_contract,
            probe.preference_batch,
            mc_samples=int(runtime.config["objective"]["flow_mc_samples"]),
            flow_seed_root=int(runtime.config["rng"]["flow_credit_seed_root"]),
            cycle=probe.cycle,
            global_task_id=probe.global_task_id,
            device=runtime.context.device,
        )
        response.update(
            before_preference_margin=probe.before_preference_margin,
            after_preference_margin=preference["preference_margin"],
            preference_margin_delta=(
                preference["preference_margin"] - probe.before_preference_margin
            ),
            after_preference_objective=preference["preference_objective"],
        )
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
) -> dict[str, Any]:
    task_ids = tuple(task.global_task_id for task in runtime.tasks)
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    matrix = torch.zeros(
        len(task_ids), shared_mean.numel(), dtype=torch.float32, device=shared_mean.device
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
        pairwise_values = torch.stack(
            (offdiag.mean(), offdiag.min(), offdiag.max())
        ).cpu().tolist()
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
        task_rms, shared_rms = torch.stack(
            (
                values.square().mean(dim=1).sqrt().mean(),
                mean_values.square().mean().sqrt(),
            )
        ).cpu().tolist()
        parameter_energy[name] = {
            "task_gradient_rms_mean": task_rms,
            "shared_mean_gradient_rms": shared_rms,
        }

    task_values = torch.stack((row_norms, dots, cosines), dim=1).cpu().tolist()
    coverage, cosine_mean, cosine_minimum = torch.stack(
        ((dots > 0).float().mean(), cosines.mean(), cosines.min())
    ).cpu().tolist()
    return {
        "active_task_ids": selected_ids,
        "shared_mean_descent_coverage": coverage,
        "task_to_shared_cosine_mean": cosine_mean,
        "task_to_shared_cosine_minimum": cosine_minimum,
        "pairwise_task_gradient_cosine": pairwise_summary,
        "per_task": [
            {
                "task_id": task_id,
                "gradient_norm": values[0],
                "dot_shared_mean": values[1],
                "cosine_shared_mean": values[2],
            }
            for task_id, values in zip(selected_ids, task_values, strict=True)
        ],
        "per_parameter": parameter_energy,
    }


def apply_reward_step(
    runtime: RewardRuntime,
    gradient_sum: torch.Tensor,
    local_active_tasks: int,
    task_gradients: Mapping[int, torch.Tensor],
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
    coexistence = _coexistence(runtime, task_gradients, gradient_sum)
    assign_flat_gradient(gradient_sum, runtime.gradient_layout)
    clip = float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"])
    grad_norm = torch.nn.utils.clip_grad_norm_(runtime.trainable_parameters, clip)
    named = _trainable_named(runtime)
    before = {name: value.detach().clone() for name, value in named}
    runtime.optimizer.step()
    delta = {
        name: float((value.detach() - before[name]).float().square().mean().sqrt())
        for name, value in named
    }
    return AppliedStep(
        active_tasks=active_tasks,
        gradient_norm=float(grad_norm),
        gradient_rms=float(gradient_sum.square().mean().sqrt()),
        parameter_delta_rms=delta,
        gradient_coexistence=coexistence,
    )
