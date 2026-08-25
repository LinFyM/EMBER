"""One role-balanced optimizer update for the G3 shared compiler."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from ember.ecp.native_materialization import (
    compose_rank12_plus_rank4,
    residual_lora_state,
)
from ember.ecp.natural_program_data import NaturalProgramSample, NaturalProgramTask
from ember.ecp.policy_effects import (
    ExecutionPolicyPrefix,
    PolicyEffectResponse,
    capture_policy_effect_response,
    prepare_policy_effect_prefix_cache,
)
from ember.ecp.shared_compiler_data import (
    pack_shared_compiler_videos,
    prepare_shared_compiler_condition,
)
from ember.ecp.shared_compiler_effects import (
    SharedCompilerEffectBank,
    carrier_preservation_loss,
    effective_update_loss,
    member_effect_losses,
    response_consistency_loss,
)
from ember.ecp.stage0_train_step import _gather_records, _reduce_gradients
from ember.pi05_lora import derive_pi05_lora_rank

if TYPE_CHECKING:
    from ember.ecp.shared_compiler_training import SharedCompilerRuntime


@dataclass(frozen=True)
class SharedCompilerTaskLoss:
    global_member_effect: torch.Tensor
    family_functional: torch.Tensor
    cross_episode_flow: torch.Tensor
    effective_update: torch.Tensor
    carrier_preservation: torch.Tensor
    same_task_consistency: torch.Tensor
    action_response: torch.Tensor
    total: torch.Tensor


def _different_sample(
    runtime: SharedCompilerRuntime,
    *,
    task_id: int,
    macro: int,
    primary: NaturalProgramSample,
) -> tuple[NaturalProgramSample, int]:
    visit = macro + int(runtime.config["data"]["same_task_other_offset"])
    primary_demos = set(primary.video_demos)
    for _ in range(50):
        candidate = runtime.schedule.sample(task_id, visit)
        if primary_demos.isdisjoint(candidate.video_demos):
            return candidate, visit
        visit += 1
    raise RuntimeError("G3 could not form a disjoint same-task video condition")


def _effect_responses(
    runtime: SharedCompilerRuntime,
    *,
    bank: SharedCompilerEffectBank,
    states: Sequence[Mapping[str, torch.Tensor]],
) -> tuple[PolicyEffectResponse, ...]:
    microbatch = int(runtime.config["optimization"]["effect_microbatch_states"])
    rows: list[list[PolicyEffectResponse]] = [[] for _ in states]
    for start in range(0, bank.state_count, microbatch):
        stop = min(start + microbatch, bank.state_count)
        prefix = ExecutionPolicyPrefix(
            embeddings=bank.prefix.embeddings[start:stop],
            padding=bank.prefix.padding[start:stop],
        )
        cache = prepare_policy_effect_prefix_cache(runtime.policy, prefix)
        for index, state in enumerate(states):
            rows[index].append(
                capture_policy_effect_response(
                    policy=runtime.policy,
                    observer=runtime.program.encoder.observer,
                    lora=runtime.lora,
                    state=state,
                    prefix=prefix,
                    suffix_noise=bank.suffix_noise[start:stop],
                    denoising_steps=int(
                        runtime.config["optimization"]["denoising_steps"]
                    ),
                    prepared_prefix_cache=cache,
                )
            )
        del cache
    responses = tuple(
        PolicyEffectResponse(
            owner=torch.cat([value.owner for value in values]),
            flow=torch.cat([value.flow for value in values]),
            action=torch.cat([value.action for value in values]),
        )
        for values in rows
    )
    if any(value.owner.shape != bank.carrier.owner.shape for value in responses):
        raise RuntimeError("G3 microbatched policy response changed")
    return responses


def _candidate(
    runtime: SharedCompilerRuntime,
    *,
    task_id: int,
    sample: NaturalProgramSample,
) -> tuple[Any, Mapping[str, torch.Tensor]]:
    packed = pack_shared_compiler_videos(
        task=runtime.task_by_id[task_id],
        sample=sample,
        video_store=runtime.video_store,
        query_points=runtime.query_points,
        device=runtime.context.device,
    )
    tokens, mask = runtime.language_tokens[task_id]
    condition = prepare_shared_compiler_condition(
        policy=runtime.policy,
        program_model=runtime.program,
        owners=runtime.owners,
        packed=packed,
        language_tokens=tokens,
        language_mask=mask,
        chunk_size=int(runtime.config["model"]["frame_chunk_size"]),
    )
    output = runtime.compiler(
        condition.program, condition.videos, s_ref=runtime.ranks.s_ref
    )
    residual = residual_lora_state(
        output.residual,
        derive_pi05_lora_rank(runtime.ranks.contract, rank=4),
        canonicalize=False,
    )
    complete = compose_rank12_plus_rank4(
        carrier_state=runtime.ranks.carrier_rank12,
        residual_state=residual,
        rank16_contract=runtime.ranks.contract,
    )
    return (output, residual, complete, condition.metrics)


def _task_record(
    *,
    task: NaturalProgramTask,
    sample: NaturalProgramSample,
    other_visit: int | None,
    output: Any,
    metrics: Mapping[str, Any],
    loss: SharedCompilerTaskLoss,
    responsibilities: torch.Tensor,
) -> dict[str, Any]:
    beta = output.video_weights.detach().float()
    return {
        "authority_id": task.authority_id,
        "domain": task.domain,
        "domain_task_id": task.domain_task_id,
        "role": task.role,
        "K": sample.k,
        "video_demos": list(sample.video_demos),
        "reserved_action_demos": list(sample.action_demos),
        "same_task_other_visit": other_visit,
        **{
            name: float(getattr(loss, name).detach())
            for name in SharedCompilerTaskLoss.__dataclass_fields__
        },
        "responsibilities": responsibilities.detach().float().cpu().tolist(),
        "video_weights": beta.cpu().tolist(),
        "maximum_video_weight": float(beta.max()),
        "maximum_uniform_deviation": float((beta - 1.0 / sample.k).abs().max()),
        **metrics,
    }


def _run_task(
    runtime: SharedCompilerRuntime,
    *,
    task_id: int,
    macro: int,
    global_task_count: int,
) -> dict[str, Any]:
    task = runtime.task_by_id[task_id]
    sample = runtime.schedule.sample(task_id, macro)
    output, residual, complete, metrics = _candidate(
        runtime, task_id=task_id, sample=sample
    )
    bank = runtime.effect_banks.get(task_id)
    consistency = (macro + 1) % int(
        runtime.config["optimization"]["same_task_consistency_interval"]
    ) == 0
    other_visit = None
    states = [complete]
    if consistency:
        other, other_visit = _different_sample(
            runtime, task_id=task_id, macro=macro, primary=sample
        )
        _, _, other_complete, _ = _candidate(
            runtime, task_id=task_id, sample=other
        )
        states.append(other_complete)
    responses = _effect_responses(runtime, bank=bank, states=states)
    member = member_effect_losses(responses[0], bank)
    update, _ = effective_update_loss(
        candidate_state=residual,
        bank=bank,
        contract=runtime.rank4_contract,
        s_ref=runtime.ranks.s_ref,
        responsibilities=member.responsibilities,
    )
    carrier = carrier_preservation_loss(responses[0], bank)
    same = (
        response_consistency_loss(responses[0], responses[1], bank)
        if consistency
        else responses[0].owner.new_zeros(())
    )
    weights = runtime.config["optimization"]["loss_weights"]
    total = (
        float(weights["global_member_effect"]) * member.global_effect
        + float(weights["family_functional"]) * member.family_functional
        + float(weights["cross_episode_flow"]) * member.cross_episode_flow
        + float(weights["effective_update"]) * update
        + float(weights["carrier_preservation"]) * carrier
        + float(weights["same_task_consistency"]) * same
    )
    loss = SharedCompilerTaskLoss(
        global_member_effect=member.global_effect,
        family_functional=member.family_functional,
        cross_episode_flow=member.cross_episode_flow,
        effective_update=update,
        carrier_preservation=carrier,
        same_task_consistency=same,
        action_response=member.action_response,
        total=total,
    )
    if not bool(torch.isfinite(total)):
        raise RuntimeError(f"non-finite G3 loss at macro {macro}, task {task_id}")
    (total / float(global_task_count)).backward()
    return _task_record(
        task=task,
        sample=sample,
        other_visit=other_visit,
        output=output,
        metrics=metrics,
        loss=loss,
        responsibilities=member.responsibilities,
    )


def run_shared_compiler_optimizer_step(
    runtime: SharedCompilerRuntime,
    *,
    macro: int,
    assignments: tuple[tuple[int, ...], ...],
) -> tuple[list[dict[str, Any]], dict[str, Any], tuple[int, ...]]:
    tick = time.monotonic()
    task_ids = assignments[runtime.context.rank]
    global_task_count = sum(map(len, assignments))
    if global_task_count != 2 or len(task_ids) > 2:
        raise RuntimeError("G3 optimizer step lost its target/meta pair")
    runtime.optimizer.zero_grad(set_to_none=True)
    records = [
        _run_task(
            runtime,
            task_id=task_id,
            macro=macro,
            global_task_count=global_task_count,
        )
        for task_id in task_ids
    ]
    if any(parameter.grad is not None for parameter in runtime.frozen_parameters):
        raise RuntimeError("frozen G3 authority accumulated gradients")
    _reduce_gradients(runtime.trainable_parameters, runtime.context.world_size)
    probes = {
        "input_query": runtime.compiler.input_query.weight.grad,
        "output_query": runtime.compiler.output_query.weight.grad,
        "scale_head": runtime.compiler.scale_head.weight.grad,
    }
    probe_norms = {}
    for name, gradient in probes.items():
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError(f"G3 {name} lost its finite gradient")
        probe_norms[name] = float(gradient.float().norm())
    if min(probe_norms.values()) <= 0.0:
        raise RuntimeError("G3 shared query or scale path has zero gradient")
    clip = float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"])
    gradient_norm = torch.nn.utils.clip_grad_norm_(runtime.trainable_parameters, clip)
    if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError(f"non-finite G3 gradient at macro {macro}")
    runtime.optimizer.step()
    runtime.scheduler.step()
    runtime.optimizer_steps += 1
    global_records = sorted(
        _gather_records(records, runtime.context.world_size),
        key=lambda row: int(row["authority_id"]),
    )
    role_counts = {
        role: sum(row["role"] == role for row in global_records)
        for role in ("meta_fit", "target_fit")
    }
    if len(global_records) != 2 or role_counts != {"meta_fit": 1, "target_fit": 1}:
        raise RuntimeError("G3 optimizer step lost role balance")
    return global_records, {
        "optimizer_step": runtime.optimizer_steps,
        "global_task_count": 2,
        "role_counts": role_counts,
        "gradient_probe_norms": probe_norms,
        "gradient_norm_before_clip": float(gradient_norm),
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "step_seconds": time.monotonic() - tick,
    }, task_ids
