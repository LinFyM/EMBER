"""One role-balanced J2 functional optimizer update."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import default_collate

from ember.ecp.bank_conditioning.mapping import (
    MappingCondition,
    SharedCompilerMappingSchedule,
)
from ember.ecp.natural_program_data import NaturalProgramSample
from ember.ecp.native_materialization import (
    compose_rank12_plus_rank4,
    residual_lora_state,
)
from ember.ecp.shared_compiler_data import (
    pack_shared_compiler_videos,
    prepare_joint_program_primal_condition,
    prepare_shared_compiler_condition,
)
from ember.ecp.stage0_train_step import _gather_records
from ember.writer.functional import (
    ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    LATIN_BETA_TIME_SAMPLING_SCHEME,
    functional_lora_loss_gradient,
    writer_chain_rule_surrogate,
)

if TYPE_CHECKING:
    from ember.ecp.joint_program_primal.runtime import (
        FunctionalPanelVisit,
        JointProgramPrimalRuntime,
    )


def joint_task_group(
    runtime: JointProgramPrimalRuntime, optimizer_step: int
) -> tuple[int, ...]:
    """Cycle three of five tasks per role; every task appears thrice per five steps."""

    split = runtime.config["task_split"]
    meta = tuple(map(int, split["gradient_meta"]))
    target = tuple(map(int, split["gradient_target"]))
    if len(meta) != len(target) or len(meta) != 5 or optimizer_step < 0:
        raise RuntimeError("J2 gradient-task schedule changed")
    offset = optimizer_step % 5
    return tuple(
        (*
            (meta[(offset + index) % 5] for index in range(3)),
         *
            (target[(offset + index) % 5] for index in range(3)))
    )


def _condition_cost(condition: MappingCondition) -> int:
    return int(condition.sampled_frames)


def _task_assignments(
    runtime: JointProgramPrimalRuntime, group: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    conditions = {
        task: MappingCondition(
            authority_id=task,
            role=runtime.task_conditions[task].fit_views[0].role,
            video_demo=runtime.task_conditions[task].fit_views[0].video_demo,
            sampled_frames=sum(
                _condition_cost(value)
                for value in runtime.task_conditions[task].fit_views
            ),
        )
        for task in group
    }
    return SharedCompilerMappingSchedule.assignments(
        group, conditions, runtime.context.world_size
    )


def functional_panel_batch(
    runtime: JointProgramPrimalRuntime,
    *,
    task_id: int,
    panel_name: str,
    visit_index: int,
) -> tuple[dict[str, Any], FunctionalPanelVisit]:
    """Load one exact frozen action panel once and retain its processed batch."""

    if panel_name not in {"a", "b"}:
        raise ValueError("J2 functional panel name changed")
    panel = runtime.panels[task_id]
    visits = panel.panel_a if panel_name == "a" else panel.panel_b
    visit = visits[visit_index % len(visits)]
    key = (task_id, panel_name, visit_index % len(visits))
    cached = runtime.panel_batch_cache.get(key)
    if cached is not None:
        return cached, visit

    rows_by_demo = runtime.query_dataset.task_episode_rows[task_id]
    frame_index = runtime.query_dataset.frame_index
    selected = []
    for demo, frame in zip(
        visit.action_demos, visit.action_frames, strict=True
    ):
        rows = rows_by_demo.get(demo)
        if rows is None or not 0 <= frame < len(rows):
            raise ValueError("J2 functional panel escaped its sealed episode")
        row = int(rows[frame])
        if frame_index[row] != (task_id, demo, frame):
            raise ValueError("J2 functional panel row pairing changed")
        selected.append(row)
    batch = runtime.query_processor.training_batch(
        default_collate([runtime.query_dataset[index] for index in selected])
    )
    runtime.panel_batch_cache[key] = batch
    return batch, visit


def _pack_condition(
    runtime: JointProgramPrimalRuntime, condition: MappingCondition
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    sample = NaturalProgramSample(
        video_demos=(condition.video_demo,),
        action_demos=(),
        k=1,
        robustness_view="j2_joint_functional_k1",
    )
    packed = pack_shared_compiler_videos(
        task=runtime.task_by_id[condition.authority_id],
        sample=sample,
        video_store=runtime.video_store,
        query_points=runtime.query_points,
        device=runtime.context.device,
    )
    tokens, mask = runtime.language_tokens[condition.authority_id]
    return packed, tokens, mask


def prepare_joint_condition(
    runtime: JointProgramPrimalRuntime, condition: MappingCondition
) -> tuple[Any, dict[str, Any]]:
    """Load cached frozen evidence/X/Y/operator without caching Program output."""

    def builder():
        packed, tokens, mask = _pack_condition(runtime, condition)
        return prepare_shared_compiler_condition(
            policy=runtime.policy,
            program_model=runtime.program,
            owners=runtime.owners,
            packed=packed,
            language_tokens=tokens,
            language_mask=mask,
            chunk_size=int(runtime.base_config["model"]["frame_chunk_size"]),
        )

    result = runtime.condition_cache.get_or_build(
        authority_id=condition.authority_id,
        video_demo=condition.video_demo,
        device=runtime.context.device,
        builder=builder,
        retain=True,
    )
    if result.condition.program is not None or result.condition.evidence is None:
        raise RuntimeError("J2 cache retained Program output or lost frozen evidence")
    return result.condition, {
        **result.condition.metrics,
        "frozen_condition_cache": "hit" if result.hit else "built",
        "frozen_condition_cache_file_bytes": result.file_bytes,
        "frozen_condition_cache_build_seconds": result.build_seconds,
        "frozen_condition_cache_load_seconds": result.load_seconds,
    }


def generated_rank16(
    runtime: JointProgramPrimalRuntime, condition: MappingCondition
) -> tuple[dict[str, torch.Tensor], Any, Any, Mapping[str, Any]]:
    prepared, metrics = prepare_joint_condition(runtime, condition)
    query_times = torch.linspace(
        0.0,
        1.0,
        runtime.query_points,
        dtype=torch.float32,
        device=runtime.context.device,
    )[None]
    program, program_output = prepare_joint_program_primal_condition(
        program_model=runtime.program,
        condition=prepared,
        query_times=query_times,
    )
    output = runtime.compiler.forward_compact(
        program, prepared.videos, s_ref=runtime.ranks.s_ref
    )
    residual = residual_lora_state(
        output.residual, runtime.rank4_contract, canonicalize=False
    )
    complete = compose_rank12_plus_rank4(
        carrier_state=runtime.ranks.carrier_rank12,
        residual_state=residual,
        rank16_contract=runtime.ranks.contract,
    )
    if output.video_weights.shape != (1,) or float(output.video_weights[0]) != 1.0:
        raise RuntimeError("J2 generated adapter escaped K1 identity")
    return complete, output, program_output, metrics


def functional_loss_and_backward(
    runtime: JointProgramPrimalRuntime,
    *,
    state: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    policy_rng_seed: int,
    loss_divisor: float,
) -> float:
    value, details, gradients = functional_lora_loss_gradient(
        runtime.policy,
        state,
        runtime.ranks.contract,
        batch=batch,
        policy_rng_seed=policy_rng_seed,
        policy_rng_device=runtime.context.device,
        flow_time_sampling_scheme=LATIN_BETA_TIME_SAMPLING_SCHEME,
        flow_noise_sampling_scheme=ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
        policy_microbatch_size=int(
            runtime.config["optimization"]["functional_policy_microbatch_size"]
        ),
        collect_policy_details=False,
    )
    if details or not bool(torch.isfinite(value)) or loss_divisor <= 0:
        raise RuntimeError("J2 functional policy loss changed")
    (writer_chain_rule_surrogate(state, gradients) / loss_divisor).backward()
    return float(value.float())


def _run_task(
    runtime: JointProgramPrimalRuntime,
    *,
    task_id: int,
    visit_index: int,
    loss_divisor: float,
) -> dict[str, Any]:
    tick = time.monotonic()
    panel_batch, panel = functional_panel_batch(
        runtime,
        task_id=task_id,
        panel_name="a",
        visit_index=visit_index,
    )
    views = []
    for condition in runtime.task_conditions[task_id].fit_views:
        complete, output, program_output, metrics = generated_rank16(
            runtime, condition
        )
        loss = functional_loss_and_backward(
            runtime,
            state=complete,
            batch=panel_batch,
            policy_rng_seed=panel.policy_rng_seed,
            loss_divisor=loss_divisor,
        )
        views.append(
            {
                "video_demo": condition.video_demo,
                "sampled_frames": condition.sampled_frames,
                "functional_loss": loss,
                "canonical_active_events": int(
                    (program_output.program.rho[0].detach() > 0.2).sum()
                ),
                "solve_metrics": output.solve_metrics.detach().float().cpu().tolist(),
                "conditioning_metrics": output.conditioning_metrics.detach()
                .float()
                .cpu()
                .tolist(),
                "condition_metrics": metrics,
            }
        )
        del complete, output, program_output
    role = runtime.task_conditions[task_id].fit_views[0].role
    return {
        "authority_id": task_id,
        "role": role,
        "panel": "a",
        "panel_visit": visit_index,
        "functional_policy_rng_seed": panel.policy_rng_seed,
        "action_demos": list(panel.action_demos),
        "action_frames": list(panel.action_frames),
        "mean_functional_loss": sum(row["functional_loss"] for row in views)
        / len(views),
        "views": views,
        "task_seconds": time.monotonic() - tick,
    }


def _gradient_presence(
    runtime: JointProgramPrimalRuntime,
) -> tuple[bool, ...]:
    local = tuple(parameter.grad is not None for parameter in runtime.trainable_parameters)
    if runtime.context.world_size <= 1:
        return local
    rows: list[Any] = [None] * runtime.context.world_size
    dist.all_gather_object(rows, local)
    if any(row != local for row in rows):
        raise RuntimeError("J2 ranks disagreed on parameter gradient ownership")
    return local


def _sum_gradients(runtime: JointProgramPrimalRuntime) -> None:
    presence = _gradient_presence(runtime)
    if runtime.gradient_presence is None:
        runtime.gradient_presence = presence
    elif runtime.gradient_presence != presence:
        raise RuntimeError("J2 gradient presence changed across optimizer steps")
    if runtime.context.world_size <= 1:
        return
    for parameter, present in zip(
        runtime.trainable_parameters, presence, strict=True
    ):
        if present:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def _gradient_probes(runtime: JointProgramPrimalRuntime) -> dict[str, float]:
    scorer = runtime.compiler.primal_scorer
    probes = {
        "program_language": runtime.program.language_reader.queries.grad,
        "program_scene": runtime.program.scene_reader.queries.grad,
        "program_process": runtime.program.process_fusion[0].weight.grad,
        "primal_input": scorer.input_primal_heads[0].weight.grad,
        "primal_output": scorer.output_primal_heads[0].weight.grad,
        "primal_program_context": scorer.program_context["q"][1].weight.grad,
        "primal_event_score": scorer.event_score["q"].weight.grad,
    }
    result = {}
    for name, gradient in probes.items():
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError(f"J2 {name} gradient is absent or non-finite")
        result[name] = float(gradient.float().norm())
    if min(result.values()) <= 0:
        raise RuntimeError("J2 Program--primal functional gradient is zero")
    return result


def _clip_gradients(
    parameters: Sequence[torch.nn.Parameter], *, maximum: float
) -> float:
    norm = torch.nn.utils.clip_grad_norm_(parameters, float(maximum))
    if not bool(torch.isfinite(norm)):
        raise RuntimeError("J2 gradient norm is non-finite")
    return float(norm)


def _rank_performance(
    runtime: JointProgramPrimalRuntime, local: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    rows: list[Any] = [None] * runtime.context.world_size
    if runtime.context.world_size > 1:
        dist.all_gather_object(rows, dict(local))
    else:
        rows[0] = dict(local)
    return rows


def run_joint_program_primal_optimizer_step(
    runtime: JointProgramPrimalRuntime,
) -> dict[str, Any]:
    """Run exactly six tasks x two fit videos with task/role-equal weight."""

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
        _run_task(
            runtime,
            task_id=task,
            visit_index=visit_index,
            loss_divisor=12.0,
        )
        for task in assignments[runtime.context.rank]
    ]
    if runtime.native_teachers.tensor_reads != teacher_reads:
        raise RuntimeError("J2 joint loss read training-only native teachers")
    if any(parameter.grad is not None for parameter in runtime.frozen_parameters):
        raise RuntimeError("J2 frozen authority accumulated gradients")
    _sum_gradients(runtime)
    probes = _gradient_probes(runtime)
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
        raise RuntimeError("J2 optimizer update lost fixed task-role weight")
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
        "mean_functional_loss": sum(
            float(row["mean_functional_loss"]) for row in records
        )
        / len(records),
        "gradient_probe_norms": probes,
        "gradient_norm_before_clip": gradient_norm,
        "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
        "rank_assignments": [list(row) for row in assignments],
        "rank_performance": performance,
        "global_step_seconds": max(float(row["seconds"]) for row in performance),
        "conditions": records,
        "native_teacher_tensor_reads": runtime.native_teachers.tensor_reads,
    }
