"""One role-balanced J2 functional optimizer update."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import default_collate

from ember.ecp.bank_conditioning.mapping import (
    MappingCondition,
    SharedCompilerMappingSchedule,
)
from ember.ecp.native_factors import native_output_group_count, rms_normalize
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
from ember.ecp.joint_program_primal.raw_stage0 import (
    RAW_STAGE0_PROGRAM_INPUT,
    prepare_raw_stage0_primal_condition,
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


def counterfactual_task_pairs(
    runtime: JointProgramPrimalRuntime, group: tuple[int, ...]
) -> dict[int, int]:
    """Pair each task to the next active task within the same fixed role."""

    split = runtime.config["task_split"]
    meta = set(map(int, split["gradient_meta"]))
    target = set(map(int, split["gradient_target"]))
    roles = (
        tuple(task for task in group if task in meta),
        tuple(task for task in group if task in target),
    )
    if any(len(tasks) != 3 for tasks in roles):
        raise RuntimeError("J3 counterfactual role pairing changed")
    pairs = {
        task: tasks[(index + 1) % len(tasks)]
        for tasks in roles
        for index, task in enumerate(tasks)
    }
    if set(pairs) != set(group) or any(task == wrong for task, wrong in pairs.items()):
        raise RuntimeError("J3 counterfactual task pairing changed")
    return pairs


def counterfactual_arm(optimizer_step: int) -> str:
    if optimizer_step < 0:
        raise ValueError("J3 optimizer step must be non-negative")
    return "wrong_program" if optimizer_step % 2 == 0 else "wrong_bank"


def counterfactual_hinge(
    *,
    correct_loss: float,
    negative_loss: float,
    margin_scale: float,
    normalized_margin: float,
) -> tuple[bool, float, float, float]:
    """Return active, gap, raw margin and the bounded-support hinge value."""

    values = (correct_loss, negative_loss, margin_scale, normalized_margin)
    if not all(math.isfinite(value) for value in values) or min(
        margin_scale, normalized_margin
    ) <= 0:
        raise ValueError("J3 counterfactual margin input changed")
    gap = negative_loss - correct_loss
    margin = normalized_margin * margin_scale
    hinge = max(0.0, margin - gap)
    return hinge > 0, gap, margin, hinge


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


def compile_joint_program(
    runtime: JointProgramPrimalRuntime,
    *,
    condition: Any,
    query_times: torch.Tensor,
) -> tuple[Any, Any]:
    """Compile the config-selected input while preserving one scorer path."""

    if runtime.config["model"].get("program_input") == RAW_STAGE0_PROGRAM_INPUT:
        return prepare_raw_stage0_primal_condition(
            program_model=runtime.program,
            condition=condition,
            query_times=query_times,
        )
    return prepare_joint_program_primal_condition(
        program_model=runtime.program,
        condition=condition,
        query_times=query_times,
    )


def generated_rank16_pair(
    runtime: JointProgramPrimalRuntime,
    *,
    program_condition: MappingCondition,
    bank_condition: MappingCondition,
    inverse_covariance_power_override: float | None = None,
) -> tuple[dict[str, torch.Tensor], Any, Any, Mapping[str, Any]]:
    """Generate one adapter from an explicit Program/bank pairing."""

    program_prepared, program_metrics = prepare_joint_condition(
        runtime, program_condition
    )
    if bank_condition == program_condition:
        bank_prepared, bank_metrics = program_prepared, program_metrics
    else:
        bank_prepared, bank_metrics = prepare_joint_condition(runtime, bank_condition)
    query_times = torch.linspace(
        0.0,
        1.0,
        runtime.query_points,
        dtype=torch.float32,
        device=runtime.context.device,
    )[None]
    program, program_output = compile_joint_program(
        runtime,
        condition=program_prepared,
        query_times=query_times,
    )
    output = runtime.compiler.forward_compact(
        program,
        bank_prepared.videos,
        s_ref=runtime.ranks.s_ref,
        inverse_covariance_power_override=inverse_covariance_power_override,
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
        raise RuntimeError("J3 generated adapter escaped K1 identity")
    return complete, output, program_output, {
        "program": program_metrics,
        "bank": bank_metrics,
    }


def generated_rank16(
    runtime: JointProgramPrimalRuntime,
    condition: MappingCondition,
    *,
    inverse_covariance_power_override: float | None = None,
) -> tuple[dict[str, torch.Tensor], Any, Any, Mapping[str, Any]]:
    return generated_rank16_pair(
        runtime,
        program_condition=condition,
        bank_condition=condition,
        inverse_covariance_power_override=inverse_covariance_power_override,
    )


def functional_loss_derivative(
    runtime: JointProgramPrimalRuntime,
    *,
    state: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    policy_rng_seed: int,
) -> tuple[float, dict[str, torch.Tensor]]:
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
    if details or not bool(torch.isfinite(value)):
        raise RuntimeError("J3 functional policy loss changed")
    return float(value.float()), gradients


def backward_functional_derivative(
    state: Mapping[str, torch.Tensor],
    gradients: Mapping[str, torch.Tensor],
    *,
    weight: float,
) -> None:
    if not math.isfinite(weight) or weight == 0:
        raise ValueError("J3 functional surrogate weight changed")
    (writer_chain_rule_surrogate(state, gradients) * float(weight)).backward()


def functional_loss_and_backward(
    runtime: JointProgramPrimalRuntime,
    *,
    state: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    policy_rng_seed: int,
    loss_divisor: float,
) -> float:
    """Retained task-local positive-control wrapper around the shared derivative."""

    if loss_divisor <= 0:
        raise ValueError("functional loss divisor must be positive")
    value, gradients = functional_loss_derivative(
        runtime,
        state=state,
        batch=batch,
        policy_rng_seed=policy_rng_seed,
    )
    backward_functional_derivative(state, gradients, weight=1.0 / loss_divisor)
    return value


def _run_correct_task(
    runtime: JointProgramPrimalRuntime,
    *,
    task_id: int,
    visit_index: int,
) -> dict[str, Any]:
    """Use only the generated-policy functional objective for one task."""

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
        loss, gradients = functional_loss_derivative(
            runtime,
            state=complete,
            batch=panel_batch,
            policy_rng_seed=panel.policy_rng_seed,
        )
        backward_functional_derivative(complete, gradients, weight=1.0 / 12.0)
        views.append(
            {
                "video_demo": condition.video_demo,
                "sampled_frames": condition.sampled_frames,
                "functional_loss": loss,
                "canonical_active_events": int(
                    (program_output.program.rho[0].detach() > 0.2).sum()
                ),
                "solve_metrics": output.solve_metrics.detach()
                .float()
                .cpu()
                .tolist(),
                "conditioning_metrics": output.conditioning_metrics.detach()
                .float()
                .cpu()
                .tolist(),
                "condition_metrics": metrics,
            }
        )
        del complete, output, program_output
    return {
        "authority_id": task_id,
        "role": runtime.task_conditions[task_id].fit_views[0].role,
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


def _run_bank_compatibility_task(
    runtime: JointProgramPrimalRuntime,
    *,
    task_id: int,
    wrong_task_id: int,
    visit_index: int,
) -> dict[str, Any]:
    """Preserve the R10 full direction while learning current-bank routing."""

    tick = time.monotonic()
    probe_only = (
        runtime.config["optimization"]["loss"]
        == "cross_video_bank_compatibility_probe_only"
    )
    panel_batch = panel = None
    if not probe_only:
        panel_batch, panel = functional_panel_batch(
            runtime,
            task_id=task_id,
            panel_name="a",
            visit_index=visit_index,
        )
    fit_views = runtime.task_conditions[task_id].fit_views
    wrong_views = runtime.task_conditions[wrong_task_id].fit_views
    cell = runtime.config["optimization"]["joint"]["bank_compatibility"]
    threshold = float(cell["threshold"])
    temperature = float(cell["temperature"])
    weight = float(cell["weight"])
    if temperature <= 0.0 or weight <= 0.0:
        raise RuntimeError("R12 compatibility loss scale changed")

    query_times = torch.linspace(
        0.0,
        1.0,
        runtime.query_points,
        dtype=torch.float32,
        device=runtime.context.device,
    )[None]
    views = []
    for view_index, condition in enumerate(fit_views):
        functional_loss = None
        functional_metrics = None
        if not probe_only:
            if panel_batch is None or panel is None:
                raise RuntimeError("R12 functional panel is absent")
            complete, output, program_output, functional_metrics = generated_rank16(
                runtime,
                condition,
                inverse_covariance_power_override=1.0,
            )
            functional_loss, gradients = functional_loss_derivative(
                runtime,
                state=complete,
                batch=panel_batch,
                policy_rng_seed=panel.policy_rng_seed,
            )
            backward_functional_derivative(
                complete, gradients, weight=1.0 / 12.0
            )
            if (
                output.compatibility_supports is not None
                or output.selected_inverse_covariance_powers is None
                or not bool(
                    torch.all(output.selected_inverse_covariance_powers == 1.0)
                )
            ):
                raise RuntimeError("R12 correct functional teacher route changed")

        program_prepared, program_metrics = prepare_joint_condition(
            runtime, condition
        )
        swapped_index = 1 - view_index
        positive_condition = fit_views[swapped_index]
        negative_condition = wrong_views[swapped_index]
        positive_prepared, positive_metrics = prepare_joint_condition(
            runtime, positive_condition
        )
        negative_prepared, negative_metrics = prepare_joint_condition(
            runtime, negative_condition
        )
        program, compatibility_program_output = compile_joint_program(
            runtime,
            condition=program_prepared,
            query_times=query_times,
        )
        support_rows = runtime.compiler.bank_compatibility_supports(
            program,
            (
                positive_prepared.videos[0],
                negative_prepared.videos[0],
            ),
        )
        if len(support_rows) != 2:
            raise RuntimeError("R12 compatibility pair topology changed")
        (positive_route, positive_training), (
            negative_route,
            negative_training,
        ) = support_rows
        positive_logit = (positive_training - threshold) / temperature
        negative_logit = (negative_training - threshold) / temperature
        compatibility_loss = 0.5 * (
            torch.nn.functional.softplus(-positive_logit)
            + torch.nn.functional.softplus(negative_logit)
        )
        (compatibility_loss * (weight / 12.0)).backward()
        views.append(
            {
                "video_demo": condition.video_demo,
                "sampled_frames": condition.sampled_frames,
                "functional_loss": functional_loss,
                "functional_operator": (
                    "frozen_r12_not_evaluated"
                    if probe_only
                    else "full_inverse_teacher_forced"
                ),
                "compatibility": {
                    "positive_video_demo": positive_condition.video_demo,
                    "negative_task": wrong_task_id,
                    "negative_video_demo": negative_condition.video_demo,
                    "route_threshold": threshold,
                    "temperature": temperature,
                    "weight": weight,
                    "positive_route_support": float(positive_route.detach()),
                    "positive_training_support": float(
                        positive_training.detach()
                    ),
                    "negative_route_support": float(negative_route.detach()),
                    "negative_training_support": float(
                        negative_training.detach()
                    ),
                    "positive_full_route": bool(
                        float(positive_route.detach()) >= threshold
                    ),
                    "negative_full_route": bool(
                        float(negative_route.detach()) >= threshold
                    ),
                    "training_support_margin": float(
                        (positive_training - negative_training).detach()
                    ),
                    "loss": float(compatibility_loss.detach()),
                },
                "canonical_active_events": int(
                    (
                        compatibility_program_output.program.rho[0].detach()
                        > 0.2
                    ).sum()
                ),
                "condition_metrics": {
                    "functional": functional_metrics,
                    "program": program_metrics,
                    "positive_bank": positive_metrics,
                    "negative_bank": negative_metrics,
                },
            }
        )
        del (
            program,
            compatibility_program_output,
            program_prepared,
            positive_prepared,
            negative_prepared,
        )
        if not probe_only:
            del complete, output, program_output
    result = {
        "authority_id": task_id,
        "role": runtime.task_conditions[task_id].fit_views[0].role,
        "panel": (
            "cross_video_bank_compatibility_probe_only"
            if probe_only
            else "a_plus_cross_video_bank_compatibility"
        ),
        "panel_visit": visit_index,
        "mean_bank_compatibility_loss": sum(
            row["compatibility"]["loss"] for row in views
        )
        / len(views),
        "wrong_task": wrong_task_id,
        "views": views,
        "task_seconds": time.monotonic() - tick,
    }
    if not probe_only:
        if panel is None:
            raise RuntimeError("R12 functional panel record is absent")
        result.update(
            {
                "functional_policy_rng_seed": panel.policy_rng_seed,
                "action_demos": list(panel.action_demos),
                "action_frames": list(panel.action_frames),
                "mean_functional_loss": sum(
                    float(row["functional_loss"]) for row in views
                )
                / len(views),
            }
        )
    return result


def _outer_update_cosine(
    predicted_input: torch.Tensor,
    predicted_output: torch.Tensor,
    target_input: torch.Tensor,
    target_output: torch.Tensor,
    rank_scale: torch.Tensor,
) -> torch.Tensor:
    """Gauge-invariant cosine between two rank-four primal updates."""

    predicted_a = rms_normalize(predicted_input.float())
    target_a = rms_normalize(target_input.float())
    predicted_b = rms_normalize(predicted_output.float()).permute(1, 0, 2).flatten(1)
    target_b = rms_normalize(target_output.float()).permute(1, 0, 2).flatten(1)
    scale = rank_scale.float()[:, None]
    predicted_b = predicted_b * scale
    target_b = target_b * scale
    cross_a = predicted_a @ target_a.transpose(0, 1)
    cross_b = predicted_b @ target_b.transpose(0, 1)
    predicted_norm = (
        (predicted_a @ predicted_a.transpose(0, 1))
        * (predicted_b @ predicted_b.transpose(0, 1))
    ).sum().clamp_min(1e-12).sqrt()
    target_norm = (
        (target_a @ target_a.transpose(0, 1))
        * (target_b @ target_b.transpose(0, 1))
    ).sum().clamp_min(1e-12).sqrt()
    return ((cross_a * cross_b).sum() / (predicted_norm * target_norm)).clamp(
        -1.0, 1.0
    )


def _functional_code_outer_loss(
    runtime: JointProgramPrimalRuntime,
    *,
    task_id: int,
    program: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = runtime.functional_code_targets.get(task_id)
    if target is None:
        raise RuntimeError("R7 functional-code target is absent")
    scorer = runtime.compiler.primal_scorer
    state = scorer.program_state(program)
    predicted_inputs = scorer.input_primals(state)
    predicted_outputs = scorer.output_primals(state)
    family_rows: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("q", "v", "action_in", "action_out")
    }
    for index, owner in enumerate(runtime.owners):
        expected_groups = native_output_group_count(owner)
        if predicted_outputs[index].shape[0] != expected_groups:
            raise RuntimeError("R7 output primal grouping changed")
        family_rows[owner.family.value].append(
            _outer_update_cosine(
                predicted_inputs[index],
                predicted_outputs[index],
                target.inputs[index],
                target.outputs[index],
                runtime.compiler.scale_prior_ratio[index],
            )
        )
    family = {
        name: torch.stack(values).mean() for name, values in family_rows.items()
    }
    loss = torch.stack(tuple(1.0 - family[name] for name in family_rows)).mean()
    return loss, {name: float(value.detach()) for name, value in family.items()}


def _run_functional_code_task(
    runtime: JointProgramPrimalRuntime,
    *,
    task_id: int,
    visit_index: int,
) -> dict[str, Any]:
    """Acquire a content chart from one task-level code shared by two videos."""

    tick = time.monotonic()
    views = []
    for condition in runtime.task_conditions[task_id].fit_views:
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
        loss, family = _functional_code_outer_loss(
            runtime, task_id=task_id, program=program
        )
        (loss / 12.0).backward()
        views.append(
            {
                "video_demo": condition.video_demo,
                "sampled_frames": condition.sampled_frames,
                "functional_code_outer_loss": float(loss.detach()),
                "family_outer_cosine": family,
                "canonical_active_events": int(
                    (program_output.program.rho[0].detach() > 0.2).sum()
                ),
                "condition_metrics": metrics,
            }
        )
        del prepared, program, program_output
    return {
        "authority_id": task_id,
        "role": runtime.task_conditions[task_id].fit_views[0].role,
        "panel": "fit_only_functional_code",
        "panel_visit": visit_index,
        "mean_acquisition_loss": sum(
            row["functional_code_outer_loss"] for row in views
        )
        / len(views),
        "views": views,
        "task_seconds": time.monotonic() - tick,
    }


def _run_task(
    runtime: JointProgramPrimalRuntime,
    *,
    task_id: int,
    wrong_task_id: int,
    visit_index: int,
    counterfactual_view_index: int,
    arm: str,
) -> dict[str, Any]:
    tick = time.monotonic()
    panel_batch, panel = functional_panel_batch(
        runtime,
        task_id=task_id,
        panel_name="a",
        visit_index=visit_index,
    )
    fit_views = runtime.task_conditions[task_id].fit_views
    wrong_views = runtime.task_conditions[wrong_task_id].fit_views
    if counterfactual_view_index not in {0, 1} or arm not in {
        "wrong_program",
        "wrong_bank",
    }:
        raise RuntimeError("J3 counterfactual view or arm changed")
    views = []
    selected_gradients: dict[str, torch.Tensor] | None = None
    selected_loss: float | None = None
    for view_index, condition in enumerate(fit_views):
        complete, output, program_output, metrics = generated_rank16(
            runtime, condition
        )
        loss, gradients = functional_loss_derivative(
            runtime,
            state=complete,
            batch=panel_batch,
            policy_rng_seed=panel.policy_rng_seed,
        )
        backward_functional_derivative(complete, gradients, weight=1.0 / 12.0)
        if view_index == counterfactual_view_index:
            selected_loss = loss
            selected_gradients = gradients
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
    if selected_loss is None or selected_gradients is None:
        raise RuntimeError("J3 lost its paired correct functional derivative")

    correct_condition = fit_views[counterfactual_view_index]
    wrong_condition = wrong_views[counterfactual_view_index]
    program_condition = (
        wrong_condition if arm == "wrong_program" else correct_condition
    )
    bank_condition = wrong_condition if arm == "wrong_bank" else correct_condition
    negative, negative_output, negative_program, negative_metrics = (
        generated_rank16_pair(
            runtime,
            program_condition=program_condition,
            bank_condition=bank_condition,
        )
    )
    negative_loss, negative_gradients = functional_loss_derivative(
        runtime,
        state=negative,
        batch=panel_batch,
        policy_rng_seed=panel.policy_rng_seed,
    )
    counterfactual = runtime.config["optimization"]["joint"]["counterfactual"]
    margin_scale = float(runtime.counterfactual_margin_scales[task_id])
    normalized_margin = float(counterfactual["normalized_margin"])
    active, gap, margin, hinge_loss = counterfactual_hinge(
        correct_loss=selected_loss,
        negative_loss=negative_loss,
        margin_scale=margin_scale,
        normalized_margin=normalized_margin,
    )
    if active:
        pair_weight = float(counterfactual["weight"]) / 6.0
        backward_functional_derivative(
            negative, negative_gradients, weight=-pair_weight
        )
        del negative, negative_output, negative_program
        correct, correct_output, correct_program, _ = generated_rank16(
            runtime, correct_condition
        )
        backward_functional_derivative(
            correct, selected_gradients, weight=pair_weight
        )
        del correct, correct_output, correct_program
    else:
        del negative, negative_output, negative_program
    counterfactual_record = {
        "arm": arm,
        "wrong_task": wrong_task_id,
        "view_index": counterfactual_view_index,
        "correct_video_demo": correct_condition.video_demo,
        "wrong_video_demo": wrong_condition.video_demo,
        "program_task": wrong_task_id if arm == "wrong_program" else task_id,
        "bank_task": wrong_task_id if arm == "wrong_bank" else task_id,
        "correct_functional_loss": selected_loss,
        "negative_functional_loss": negative_loss,
        "negative_minus_correct": gap,
        "margin_scale": margin_scale,
        "normalized_margin": normalized_margin,
        "normalized_gap": gap / margin_scale,
        "hinge_loss": hinge_loss,
        "active": active,
        "condition_metrics": negative_metrics,
    }
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
        "counterfactual": counterfactual_record,
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
    probes: dict[str, torch.Tensor | None] = {
        "program_language": runtime.program.language_reader.queries.grad,
        "program_scene": runtime.program.scene_reader.queries.grad,
    }
    if runtime.config["model"].get("program_input") != RAW_STAGE0_PROGRAM_INPUT:
        probes["program_process"] = runtime.program.process_fusion[0].weight.grad
    partition = runtime.config["model"].get(
        "primal_scorer_trainable_partition", "all"
    )
    if partition == "compatibility_probes_only":
        if any(gradient is not None for gradient in probes.values()):
            raise RuntimeError("R13 frozen Program accumulated gradients")
        heads = scorer.compatibility_input_heads
        if heads is None:
            raise RuntimeError("R13 compatibility probes are absent")
        gradients = tuple(head.weight.grad for head in heads)
        if any(gradient is None for gradient in gradients):
            raise RuntimeError("R13 compatibility-probe gradient is absent")
        norms = torch.stack(
            tuple(gradient.float().norm() for gradient in gradients)
        )
        if not bool(torch.isfinite(norms).all()) or not bool(torch.any(norms > 0)):
            raise RuntimeError("R13 compatibility-probe gradient is invalid")
        frozen = (
            *(head.weight.grad for head in scorer.input_primal_heads),
            *(
                head.weight.grad
                for owner_heads in scorer.output_primal_heads
                for head in owner_heads
            ),
            scorer.program_context["q"][1].weight.grad,
            scorer.input_trunk["q"][1].weight.grad,
        )
        if any(gradient is not None for gradient in frozen):
            raise RuntimeError("R13 functional scorer accumulated gradients")
        return {
            "compatibility_input": float(norms.square().sum().sqrt()),
            "compatibility_active_heads": float((norms > 0).sum()),
        }
    result = {}
    for name, gradient in probes.items():
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError(f"J2 {name} gradient is absent or non-finite")
        result[name] = float(gradient.float().norm())
    input_gradients = tuple(
        head.weight.grad for head in scorer.input_primal_heads
    )
    output_gradients = tuple(
        head.weight.grad
        for owner_heads in scorer.output_primal_heads
        for head in owner_heads
    )
    feature_probes = {
        "primal_program_context": scorer.program_context["q"][1].weight.grad,
        "primal_rank_context": scorer.rank_context["q"][1].weight.grad,
        "primal_event_score": scorer.event_score["q"].weight.grad,
        "primal_input_trunk": scorer.input_trunk["q"][1].weight.grad,
        "primal_output_trunk": scorer.output_trunk["q"][1].weight.grad,
        "owner_embedding": scorer.owner_embedding.grad,
        "rank_embedding": scorer.rank_embedding.grad,
    }
    if partition in {"all", "native_heads_only"}:
        if any(
            gradient is None for gradient in (*input_gradients, *output_gradients)
        ):
            raise RuntimeError("J2 native-head gradient is absent")
        input_norms = torch.stack(
            tuple(gradient.float().norm() for gradient in input_gradients)
        )
        output_norms = torch.stack(
            tuple(gradient.float().norm() for gradient in output_gradients)
        )
        if (
            not bool(torch.isfinite(input_norms).all())
            or not bool(torch.all(input_norms > 0))
            or not bool(torch.isfinite(output_norms).all())
            or not bool(torch.all(output_norms > 0))
        ):
            raise RuntimeError("J2 native-head gradient is non-finite or zero")
        result["primal_input"] = float(input_norms.square().sum().sqrt())
        result["primal_output"] = float(output_norms.square().sum().sqrt())
    elif partition == "feature_chart_only":
        if any(
            gradient is not None for gradient in (*input_gradients, *output_gradients)
        ):
            raise RuntimeError("R7 frozen native heads accumulated gradients")
    else:
        raise RuntimeError("J2 primal-scorer partition changed")
    if partition in {"all", "feature_chart_only"}:
        for name, gradient in feature_probes.items():
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise RuntimeError(f"J2 {name} gradient is absent or non-finite")
            result[name] = float(gradient.float().norm())
    elif partition == "native_heads_only":
        if any(gradient is not None for gradient in feature_probes.values()):
            raise RuntimeError("J2 frozen primal feature chart has gradients")
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
    use_counterfactual = "counterfactual" in runtime.config["optimization"]["joint"]
    use_functional_code = (
        runtime.config["optimization"]["loss"]
        == "fit_only_functional_code_outer_direction_only"
    )
    loss_name = runtime.config["optimization"]["loss"]
    use_decoupled_compatibility = (
        loss_name == "cross_video_bank_compatibility_probe_only"
    )
    use_bank_compatibility = loss_name in {
        "correct_flow_plus_cross_video_bank_compatibility",
        "cross_video_bank_compatibility_probe_only",
    }
    pairs = (
        counterfactual_task_pairs(runtime, group)
        if use_counterfactual or use_bank_compatibility
        else {}
    )
    arm = counterfactual_arm(runtime.optimizer_steps) if use_counterfactual else None
    counterfactual_view_index = runtime.optimizer_steps % 2
    assignments = _task_assignments(runtime, group)
    visit_index = runtime.optimizer_steps % int(runtime.config["data"]["panel_visits"])
    if runtime.context.world_size > 1:
        dist.barrier()
    torch.cuda.synchronize(runtime.context.device)
    tick = time.monotonic()
    teacher_reads = runtime.native_teachers.tensor_reads
    runtime.optimizer.zero_grad(set_to_none=True)
    if use_counterfactual:
        local_records = [
            _run_task(
                runtime,
                task_id=task,
                wrong_task_id=pairs[task],
                visit_index=visit_index,
                counterfactual_view_index=counterfactual_view_index,
                arm=str(arm),
            )
            for task in assignments[runtime.context.rank]
        ]
    elif use_bank_compatibility:
        local_records = [
            _run_bank_compatibility_task(
                runtime,
                task_id=task,
                wrong_task_id=pairs[task],
                visit_index=visit_index,
            )
            for task in assignments[runtime.context.rank]
        ]
    elif use_functional_code:
        local_records = [
            _run_functional_code_task(
                runtime,
                task_id=task,
                visit_index=visit_index,
            )
            for task in assignments[runtime.context.rank]
        ]
    else:
        local_records = [
            _run_correct_task(
                runtime,
                task_id=task,
                visit_index=visit_index,
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
    primary_metric = (
        "mean_bank_compatibility_loss"
        if use_decoupled_compatibility
        else (
            "mean_acquisition_loss"
            if use_functional_code
            else "mean_functional_loss"
        )
    )
    row = {
        "optimizer_step": runtime.optimizer_steps,
        "effective_optimizer_step": max(0, runtime.optimizer_steps - 10),
        "panel_visit": visit_index,
        "task_group": list(group),
        "role_counts": role_counts,
        primary_metric: sum(float(row[primary_metric]) for row in records)
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
    if use_counterfactual:
        row.update(
            {
                "counterfactual_arm": arm,
                "counterfactual_view_index": counterfactual_view_index,
                "counterfactual_pairs": {
                    str(task): wrong for task, wrong in sorted(pairs.items())
                },
                "mean_counterfactual_normalized_gap": sum(
                    float(value["counterfactual"]["normalized_gap"])
                    for value in records
                )
                / len(records),
                "mean_counterfactual_hinge_loss": sum(
                    float(value["counterfactual"]["hinge_loss"])
                    for value in records
                )
                / len(records),
                "active_counterfactual_fraction": sum(
                    bool(value["counterfactual"]["active"])
                    for value in records
                )
                / len(records),
            }
        )
    if use_bank_compatibility:
        compatibility_rows = [
            view["compatibility"]
            for record in records
            for view in record["views"]
        ]
        row.update(
            {
                "mean_bank_compatibility_loss": sum(
                    float(value["loss"]) for value in compatibility_rows
                )
                / len(compatibility_rows),
                "mean_positive_route_support": sum(
                    float(value["positive_route_support"])
                    for value in compatibility_rows
                )
                / len(compatibility_rows),
                "minimum_positive_route_support": min(
                    float(value["positive_route_support"])
                    for value in compatibility_rows
                ),
                "positive_full_route_fraction": sum(
                    bool(value["positive_full_route"])
                    for value in compatibility_rows
                )
                / len(compatibility_rows),
                "mean_negative_route_support": sum(
                    float(value["negative_route_support"])
                    for value in compatibility_rows
                )
                / len(compatibility_rows),
                "maximum_negative_route_support": max(
                    float(value["negative_route_support"])
                    for value in compatibility_rows
                ),
                "negative_full_route_fraction": sum(
                    bool(value["negative_full_route"])
                    for value in compatibility_rows
                )
                / len(compatibility_rows),
                "mean_training_support_margin": sum(
                    float(value["training_support_margin"])
                    for value in compatibility_rows
                )
                / len(compatibility_rows),
                "bank_compatibility_pairs": {
                    str(task): wrong for task, wrong in sorted(pairs.items())
                },
            }
        )
    return row
