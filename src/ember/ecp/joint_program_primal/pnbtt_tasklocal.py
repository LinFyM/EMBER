"""Shared E1 task-local arm preparation and unique rank16 generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ember.ecp.joint_program_primal.pnbtt_runtime import PNBTTTaskLocalRuntime
from ember.ecp.joint_program_primal.train_step import prepare_program_bank_condition
from ember.ecp.native_factors import G1_RESIDUAL_RANK
from ember.ecp.native_materialization import (
    compose_rank12_plus_rank4,
    residual_lora_state,
)
from ember.ecp.shared_compiler_data import (
    prepare_joint_program_primal_condition,
    program_bank_contexts,
)
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX


@dataclass(frozen=True)
class PreparedPNBTTArm:
    """One frozen exact-language Program and one compact real native bank."""

    name: str
    program_task: int
    bank_task: int
    video_demo: int
    sampled_frames: int
    program: Any
    videos: tuple[Any, ...]
    bank_contexts: tuple[Any, ...]
    condition_metrics: Mapping[str, Any]
    receives_gradient: bool


def local_tasks(runtime: PNBTTTaskLocalRuntime) -> tuple[int, ...]:
    tasks = tuple(map(int, runtime.config["task_local"]["task_ids"]))
    if runtime.context.world_size == 1:
        return tasks
    if runtime.context.world_size == len(tasks):
        return (tasks[runtime.context.rank],)
    raise RuntimeError("PNBTT E1 task/rank topology changed")


def _prepare_arm(
    runtime: PNBTTTaskLocalRuntime,
    *,
    name: str,
    program_task: int,
    bank_task: int,
    condition: Any,
    receives_gradient: bool,
) -> PreparedPNBTTArm:
    prepared, metrics = prepare_program_bank_condition(
        runtime,
        language_authority_id=program_task,
        bank_condition=condition,
    )
    if prepared.evidence is None or len(prepared.videos) != 1:
        raise RuntimeError("PNBTT E1 arm lost frozen K1 evidence")
    query_times = torch.linspace(
        0.0,
        1.0,
        runtime.query_points,
        dtype=torch.float32,
        device=runtime.context.device,
    )[None]
    with torch.no_grad():
        program, output = prepare_joint_program_primal_condition(
            program_model=runtime.program,
            condition=prepared,
            query_times=query_times,
        )
        contexts = program_bank_contexts(output, prepared.evidence)
    return PreparedPNBTTArm(
        name=name,
        program_task=int(program_task),
        bank_task=int(bank_task),
        video_demo=int(condition.video_demo),
        sampled_frames=int(condition.sampled_frames),
        program=program,
        videos=prepared.videos,
        bank_contexts=contexts,
        condition_metrics=metrics,
        receives_gradient=bool(receives_gradient),
    )


def prepare_e1_arms(
    runtime: PNBTTTaskLocalRuntime,
) -> dict[int, dict[str, PreparedPNBTTArm]]:
    """Prepare all local preregistered arms without retaining generated LoRAs."""

    cached = getattr(runtime, "pnbtt_arms", None)
    if cached is not None:
        return cached
    result: dict[int, dict[str, PreparedPNBTTArm]] = {}
    for task in local_tasks(runtime):
        wrong_task = int(
            runtime.config["task_local"]["wrong_task_by_task"][str(task)]
        )
        correct = runtime.task_conditions[task]
        wrong = runtime.task_conditions[wrong_task]
        specifications = (
            ("correct_fit0", task, correct.fit_views[0], True),
            ("correct_fit1", task, correct.fit_views[1], True),
            ("correct_held", task, correct.held_video, False),
            ("wrong_fit0", wrong_task, wrong.fit_views[0], True),
            ("wrong_fit1", wrong_task, wrong.fit_views[1], False),
        )
        result[task] = {
            name: _prepare_arm(
                runtime,
                name=name,
                program_task=task,
                bank_task=bank_task,
                condition=condition,
                receives_gradient=gradient,
            )
            for name, bank_task, condition, gradient in specifications
        }
    setattr(runtime, "pnbtt_arms", result)
    return result


def carrier_rank16(runtime: PNBTTTaskLocalRuntime) -> dict[str, torch.Tensor]:
    """Return the frozen carrier in the same unique rank16 topology as PNBTT."""

    cached = getattr(runtime, "pnbtt_carrier_rank16", None)
    if cached is not None:
        return cached
    residual: dict[str, torch.Tensor] = {}
    for target in runtime.ranks.contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        carrier_a = runtime.ranks.carrier_rank12[a_name]
        carrier_b = runtime.ranks.carrier_rank12[b_name]
        residual[a_name] = carrier_a.new_zeros(
            (G1_RESIDUAL_RANK, target.in_features)
        )
        residual[b_name] = carrier_b.new_zeros(
            (target.out_features, G1_RESIDUAL_RANK)
        )
    complete = compose_rank12_plus_rank4(
        carrier_state=runtime.ranks.carrier_rank12,
        residual_state=residual,
        rank16_contract=runtime.ranks.contract,
    )
    setattr(runtime, "pnbtt_carrier_rank16", complete)
    return complete


def generated_rank16(
    runtime: PNBTTTaskLocalRuntime,
    *,
    task: int,
    arm: PreparedPNBTTArm,
    canonicalize: bool,
) -> tuple[dict[str, torch.Tensor], Any]:
    if runtime.free_query is None or arm.program_task != task:
        raise RuntimeError("PNBTT E1 lost its task-shared free query")
    output = runtime.compiler.forward_compact(
        arm.program,
        arm.videos,
        s_ref=runtime.ranks.s_ref,
        bank_contexts=arm.bank_contexts,
        query_override=runtime.free_query(task),
    )
    residual = residual_lora_state(
        output.residual, runtime.rank4_contract, canonicalize=bool(canonicalize)
    )
    complete = compose_rank12_plus_rank4(
        carrier_state=runtime.ranks.carrier_rank12,
        residual_state=residual,
        rank16_contract=runtime.ranks.contract,
    )
    if (
        len(runtime.owners) != 38
        or output.video_weights.shape != (1,)
        or float(output.video_weights[0]) != 1.0
        or len(complete) != runtime.ranks.contract.state_tensor_count
    ):
        raise RuntimeError("PNBTT E1 escaped unique 38-target K1 rank16")
    return complete, output
