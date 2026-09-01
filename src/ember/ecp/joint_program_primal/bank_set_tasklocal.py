"""Task-local S0/S1 qualification for event-conditioned bank-set interaction."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.ecp.checkpoint import save_ecp_checkpoint
from ember.ecp.contracts import ACTION_HORIZON, TargetFamily
from ember.ecp.g1_objective import low_rank_distance_squared
from ember.ecp.joint_program_primal.gate import _functional_value
from ember.ecp.joint_program_primal.bank_set_tasklocal_evaluation import (
    FAMILIES,
    RESULT_SCHEMA,
    TASKS,
    CorrectionCollector as _CorrectionCollector,
    EffectiveTarget,
    effective_rank4_diagnostics as _effective_rank4_diagnostics,
)
from ember.ecp.joint_program_primal.bank_set_tasklocal_contract import (
    TASKLOCAL_FREE_B0_QUERY,
)
from ember.ecp.joint_program_primal.routing_control import (
    BANK_CONDITIONED_PRIMAL_STAGE,
    BANK_SET_S0_STAGE,
    BANK_SET_S1_STAGE,
    fixed_routing_program,
    prepare_routing_control_runtime,
    routing_run_schema,
    routing_stage,
)
from ember.ecp.joint_program_primal.train_step import (
    backward_functional_derivative,
    functional_loss_derivative,
    functional_panel_batch,
    prepare_program_bank_condition,
)
from ember.ecp.native_factors import (
    G1_PROBE_COUNT,
    G1_RESIDUAL_RANK,
    native_output_group_count,
)
from ember.ecp.native_materialization import (
    compose_rank12_plus_rank4,
    residual_lora_state,
)
from ember.ecp.shared_compiler_data import (
    prepare_joint_program_primal_condition,
    program_bank_contexts,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import initialize_distributed


ARM_SCHEDULE = ("wrong_fit0", "correct_fit0", "wrong_fit0", "correct_fit1")


@dataclass(frozen=True)
class PreparedTaskLocalBank:
    program_task: int
    bank_task: int
    video_demo: int
    condition_metrics: Mapping[str, Any]
    video: Any
    context: Any
    program_state: Any
    input_primals: tuple[torch.Tensor, ...]
    output_primals: tuple[torch.Tensor, ...]
    plan: Any | None
    frozen_descriptors: Any | None


@dataclass(frozen=True)
class TaskLocalArm:
    name: str
    bank: PreparedTaskLocalBank
    interaction_state: Any
    summaries: Any
    receives_gradient: bool


class _TargetFreeDelta(torch.nn.Module):
    def __init__(self, *, frames: int, groups: int, device: torch.device) -> None:
        super().__init__()
        self.input_logits = torch.nn.Parameter(
            torch.zeros(
                G1_RESIDUAL_RANK,
                frames,
                G1_PROBE_COUNT,
                ACTION_HORIZON,
                device=device,
            )
        )
        self.output_logits = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.zeros(
                        G1_RESIDUAL_RANK,
                        frames,
                        G1_PROBE_COUNT,
                        ACTION_HORIZON,
                        4,
                        device=device,
                    )
                )
                for _ in range(groups)
            ]
        )

    @staticmethod
    def _bias(value: torch.Tensor) -> torch.Tensor:
        delta = 0.1 * torch.tanh(value)
        return torch.stack((delta, -delta), dim=1)

    def input_bias(self) -> torch.Tensor:
        return self._bias(self.input_logits)

    def output_bias(self, group: int) -> torch.Tensor:
        return self._bias(self.output_logits[group])


class _FreeDeltaBank(torch.nn.Module):
    def __init__(self, runtime: Any, bank: PreparedTaskLocalBank) -> None:
        super().__init__()
        frames = int(bank.video.frame_measure.shape[0])
        self.targets = torch.nn.ModuleList(
            [
                _TargetFreeDelta(
                    frames=frames,
                    groups=native_output_group_count(owner),
                    device=runtime.context.device,
                )
                for owner in runtime.owners
            ]
        )


def _prepare_bank(
    runtime: Any,
    *,
    program_task: int,
    bank_task: int,
    condition: Any,
) -> PreparedTaskLocalBank:
    prepared, metrics = prepare_program_bank_condition(
        runtime,
        language_authority_id=program_task,
        bank_condition=condition,
    )
    if prepared.evidence is None or len(prepared.videos) != 1:
        raise RuntimeError("bank-set task-local condition lost frozen K1 evidence")
    query_times = torch.linspace(
        0.0,
        1.0,
        runtime.query_points,
        dtype=torch.float32,
        device=runtime.context.device,
    )[None]
    with torch.no_grad():
        _, bank_output = prepare_joint_program_primal_condition(
            program_model=runtime.program,
            condition=prepared,
            query_times=query_times,
        )
        context = program_bank_contexts(bank_output, prepared.evidence)[0]
        program = fixed_routing_program(runtime, program_task)
        with runtime.compiler.bank_operator.ieee_matmul(runtime.context.device):
            state = runtime.compiler.primal_scorer.program_state(program)
            input_primals = tuple(
                value.detach()
                for value in runtime.compiler.primal_scorer.input_primals(state)
            )
            output_primals = tuple(
                value.detach()
                for value in runtime.compiler.primal_scorer.output_primals(state)
            )
            interaction_state = runtime.compiler._interaction_states(
                state, (context,)
            )[0]
            if routing_stage(runtime.config) == BANK_CONDITIONED_PRIMAL_STAGE:
                plan = None
                frozen_descriptors = None
            else:
                plan = runtime.compiler.bank_operator._plan(
                    prepared.videos[0], input_primals, output_primals
                )
                frozen_descriptors = runtime.compiler.bank_operator.describe_compact(
                    prepared.videos[0],
                    plan=plan,
                    interaction_state=interaction_state,
                )
    return PreparedTaskLocalBank(
        program_task=program_task,
        bank_task=bank_task,
        video_demo=int(condition.video_demo),
        condition_metrics=metrics,
        video=prepared.videos[0],
        context=context,
        program_state=state,
        input_primals=input_primals,
        output_primals=output_primals,
        plan=plan,
        frozen_descriptors=frozen_descriptors,
    )


def _prepare_arms(runtime: Any, task: int) -> dict[str, TaskLocalArm]:
    wrong_task = int(runtime.config["task_local"]["wrong_task_by_task"][str(task)])
    correct = runtime.task_conditions[task]
    wrong = runtime.task_conditions[wrong_task]
    specifications = (
        ("correct_fit0", task, correct.fit_views[0], True),
        ("correct_fit1", task, correct.fit_views[1], True),
        ("correct_held", task, correct.held_video, False),
        ("wrong_fit0", wrong_task, wrong.fit_views[0], True),
        ("wrong_fit1", wrong_task, wrong.fit_views[1], False),
    )
    result = {}
    for name, bank_task, condition, gradient in specifications:
        bank = _prepare_bank(
            runtime,
            program_task=task,
            bank_task=bank_task,
            condition=condition,
        )
        interaction_state = runtime.compiler._interaction_states(
            bank.program_state, (bank.context,)
        )[0]
        summaries = None
        if routing_stage(runtime.config) == BANK_SET_S0_STAGE:
            # These frozen summaries later generate trainable candidate heads.
            # no_grad keeps them ordinary tensors; inference tensors cannot be
            # saved by autograd for the head-weight backward.
            with torch.no_grad():
                summaries = runtime.compiler.bank_operator.summarize_compact(
                    bank.video,
                    bank_set_interaction=runtime.compiler.bank_set_interaction,
                    interaction_state=interaction_state,
                )
        result[name] = TaskLocalArm(
            name=name,
            bank=bank,
            interaction_state=interaction_state,
            summaries=summaries,
            receives_gradient=gradient,
        )
    return result


def _output(runtime: Any, bank: PreparedTaskLocalBank, pooled: Any) -> Any:
    return runtime.compiler._output(
        bank.program_state, (pooled,), s_ref=runtime.ranks.s_ref
    )


def _complete(runtime: Any, output: Any) -> dict[str, torch.Tensor]:
    residual = residual_lora_state(
        output.residual, runtime.rank4_contract, canonicalize=False
    )
    state = compose_rank12_plus_rank4(
        carrier_state=runtime.ranks.carrier_rank12,
        residual_state=residual,
        rank16_contract=runtime.ranks.contract,
    )
    if (
        len(runtime.owners) != 38
        or output.video_weights.shape != (1,)
        or float(output.video_weights[0]) != 1.0
        or len(state) != runtime.ranks.contract.state_tensor_count
    ):
        raise RuntimeError("bank-set task-local output escaped unique rank16/K1")
    return state


def _base_output(runtime: Any, bank: PreparedTaskLocalBank) -> Any:
    pooled = runtime.compiler.bank_operator.apply_compact(
        bank.video, bank.input_primals, bank.output_primals
    )
    return _output(runtime, bank, pooled)


def _interaction_output(
    runtime: Any,
    arm: TaskLocalArm,
    *,
    summary_token: str | None = None,
    correction_observer: Callable[[str, Any, torch.Tensor], None] | None = None,
    primal_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> Any:
    stage = routing_stage(runtime.config)
    if stage == BANK_CONDITIONED_PRIMAL_STAGE:
        if summary_token is not None or correction_observer is not None:
            raise ValueError("bank-conditioned primal has no correction intervention")
        summaries, input_primals, output_primals = runtime.compiler._condition_bank(
            arm.bank.video,
            input_primals=arm.bank.input_primals,
            output_primals=arm.bank.output_primals,
            interaction_state=arm.interaction_state,
        )
        if primal_observer is not None:
            base_square = torch.zeros((), device=runtime.context.device)
            delta_square = torch.zeros_like(base_square)
            anchor_square = torch.zeros_like(base_square)
            anchor_count = 0
            gates = []
            interaction = runtime.compiler.bank_set_interaction
            for target, owner in enumerate(runtime.owners):
                base = arm.bank.input_primals[target].float()
                delta = input_primals[target].float() - base
                base_square = base_square + base.square().sum()
                delta_square = delta_square + delta.square().sum()
                anchor = summaries.inputs[target].native_anchor.float()
                anchor_square = anchor_square + anchor.square().sum()
                anchor_count += anchor.numel()
                gates.append(
                    interaction.input_primal_gate[owner.family.value](
                        summaries.inputs[target].condition
                    ).squeeze(-1)
                )
                for group, summary in enumerate(summaries.outputs[target]):
                    base = arm.bank.output_primals[target][group].float()
                    delta = output_primals[target][group].float() - base
                    base_square = base_square + base.square().sum()
                    delta_square = delta_square + delta.square().sum()
                    anchor = summary.all_types.native_anchor.float()
                    anchor_square = anchor_square + anchor.square().sum()
                    anchor_count += anchor.numel()
                    condition = torch.cat(
                        (
                            summary.all_types.condition,
                            *(scope.condition for scope in summary.by_type),
                        ),
                        dim=-1,
                    )
                    gates.append(
                        interaction.output_primal_gate[owner.family.value](
                            condition
                        ).squeeze(-1)
                    )
            gate = torch.cat(tuple(value.reshape(-1) for value in gates))
            primal_observer(
                {
                    "response_source": "real_b0_summary_and_native_anchor",
                    "delta_to_base_rms_ratio": float(
                        (delta_square / base_square.clamp_min(1e-30)).sqrt()
                    ),
                    "native_anchor_rms": float(
                        (anchor_square / anchor_count).sqrt()
                    ),
                    "gate_rms": float(gate.float().square().mean().sqrt()),
                    "gate_maximum_absolute": float(gate.float().abs().max()),
                    "candidate_logit_bias_calls": 0,
                }
            )
        pooled = runtime.compiler.bank_operator.apply_compact(
            arm.bank.video,
            input_primals,
            output_primals,
        )
        return _output(runtime, arm.bank, pooled)
    summaries = arm.summaries
    if stage == BANK_SET_S0_STAGE:
        token_kind = summary_token or (
            "correct" if arm.name.startswith("correct") else "wrong"
        )
        if token_kind not in {"correct", "wrong"}:
            raise ValueError("bank-set summary-token intervention changed")
        tree = (
            runtime.writer_state.free_correct
            if token_kind == "correct"
            else runtime.writer_state.free_wrong
        )
        summaries = summaries.with_condition(tree.conditions())
    elif stage != BANK_SET_S1_STAGE:
        raise RuntimeError("bank-set task-local summary source changed")
    pooled = runtime.compiler.bank_operator.apply_compact(
        arm.bank.video,
        arm.bank.input_primals,
        arm.bank.output_primals,
        bank_set_interaction=runtime.compiler.bank_set_interaction,
        interaction_state=arm.interaction_state,
        summaries=summaries if stage == BANK_SET_S0_STAGE else None,
        correction_observer=correction_observer,
        frozen_descriptors=arm.bank.frozen_descriptors,
        interaction_group_batch_size=int(
            runtime.config["model"]["interaction_group_batch_size"]
        ),
        replay_plan=arm.bank.plan,
    )
    return _output(runtime, arm.bank, pooled)


def _free_pool(runtime: Any, bank: PreparedTaskLocalBank, code: _FreeDeltaBank) -> Any:
    return runtime.compiler.bank_operator.apply_compact_free_bias(
        bank.video,
        bank.input_primals,
        bank.output_primals,
        direct_input_logit_biases=tuple(
            target.input_bias() for target in code.targets
        ),
        direct_output_logit_biases=tuple(
            tuple(target.output_bias(group) for group in range(len(target.output_logits)))
            for target in code.targets
        ),
    )


def _wrong_teacher(runtime: Any, task: int, arm: TaskLocalArm, base_output: Any) -> Any:
    settings = runtime.config["optimization"]["joint"]["wrong_free_delta_teacher"]
    if int(settings["updates"]) != 1:
        raise RuntimeError("S0 wrong free-delta teacher update count changed")
    code = _FreeDeltaBank(runtime, arm.bank)
    optimizer = torch.optim.Adam(
        code.parameters(),
        lr=float(settings["learning_rate"]),
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    batch, panel = functional_panel_batch(
        runtime,
        task_id=task,
        panel_name="a",
        visit_index=int(settings["panel_a_visit"]),
    )
    with torch.no_grad():
        zero_output = _output(runtime, arm.bank, _free_pool(runtime, arm.bank, code))
    zero_error = max(
        float((left.float() - right.float()).abs().max())
        for left, right in zip(
            (*zero_output.residual.a, *zero_output.residual.b),
            (*base_output.residual.a, *base_output.residual.b),
            strict=True,
        )
    )
    if zero_error > 2e-6:
        raise RuntimeError(f"free-delta zero did not reproduce R5: {zero_error}")
    optimizer.zero_grad(set_to_none=True)
    output = _output(runtime, arm.bank, _free_pool(runtime, arm.bank, code))
    state = _complete(runtime, output)
    loss, gradients = functional_loss_derivative(
        runtime,
        state=state,
        batch=batch,
        policy_rng_seed=panel.policy_rng_seed,
    )
    denominator = float(runtime.counterfactual_margin_scales[task])
    recovery = (float(panel.flow_loss) - loss) / denominator
    if not math.isfinite(recovery) or recovery <= 0.25:
        raise RuntimeError(
            "wrong free-delta teacher has no active suppression margin: "
            f"recovery={recovery}, carrier={float(panel.flow_loss)}, loss={loss}, "
            f"denominator={denominator}"
        )
    backward_functional_derivative(state, gradients, weight=-1.0 / denominator)
    norm = torch.nn.utils.clip_grad_norm_(code.parameters(), 1.0)
    if not bool(torch.isfinite(norm)):
        raise RuntimeError("wrong free-delta teacher gradient is non-finite")
    optimizer.step()
    bias_count = 0
    bias_square_sum = torch.zeros((), device=runtime.context.device)
    bias_maximum = torch.zeros((), device=runtime.context.device)
    with torch.no_grad():
        for target_code in code.targets:
            for logits in (target_code.input_logits, *target_code.output_logits):
                bias = 0.1 * torch.tanh(logits.detach().float())
                bias_count += bias.numel()
                bias_square_sum = bias_square_sum + bias.square().sum()
                bias_maximum = torch.maximum(bias_maximum, bias.abs().max())
    with torch.no_grad():
        teacher = _output(runtime, arm.bank, _free_pool(runtime, arm.bank, code))
        teacher_state = _complete(runtime, teacher)
        loss_after = _functional_value(
            runtime,
            state=teacher_state,
            batch=batch,
            seed=panel.policy_rng_seed,
        )
    recovery_after = (float(panel.flow_loss) - loss_after) / denominator
    if (
        not math.isfinite(recovery_after)
        or recovery_after >= recovery
        or recovery_after > 0.25
    ):
        raise RuntimeError(
            "wrong free-delta teacher did not realize functional suppression: "
            f"before={recovery}, after={recovery_after}, "
            f"carrier={float(panel.flow_loss)}, loss_after={loss_after}"
        )
    return teacher, {
        "panel_a_loss_before_update": loss,
        "panel_a_recovery_before_update": recovery,
        "panel_a_loss_after_update": loss_after,
        "panel_a_recovery_after_update": recovery_after,
        "panel_a_functional_suppression": recovery - recovery_after,
        "gradient_norm_before_clip": float(norm),
        "parameter_count": sum(value.numel() for value in code.parameters()),
        "candidate_correction": {
            "count": bias_count,
            "rms": float((bias_square_sum / bias_count).sqrt()),
            "maximum_absolute": float(bias_maximum),
        },
        "zero_maximum_absolute_error": zero_error,
    }


def _target(output: Any) -> EffectiveTarget:
    target = EffectiveTarget(
        a=tuple(value.detach().clone() for value in output.residual.a),
        b=tuple(value.detach().clone() for value in output.residual.b),
    )
    if any(torch.is_inference(value) for value in (*target.a, *target.b)):
        raise RuntimeError("effective-rank4 target became an inference tensor")
    return target


def _family_distances(
    runtime: Any, output: Any, target: EffectiveTarget
) -> dict[TargetFamily, torch.Tensor]:
    zero = output.residual.scales.new_zeros(())
    result = {family: zero.clone() for family in FAMILIES}
    for owner, actual_a, actual_b, target_a, target_b in zip(
        runtime.owners,
        output.residual.a,
        output.residual.b,
        target.a,
        target.b,
        strict=True,
    ):
        result[owner.family] = result[owner.family] + low_rank_distance_squared(
            actual_a,
            actual_b.transpose(0, 1),
            target_a,
            target_b.transpose(0, 1),
        )
    return result


def _targets(runtime: Any, task: int, arms: Mapping[str, TaskLocalArm]):
    # Training saves these references for low-rank distance backward.  They
    # must be ordinary frozen tensors, not inference tensors.
    with torch.no_grad():
        correct = {
            name: _target(_base_output(runtime, arms[name].bank))
            for name in ("correct_fit0", "correct_fit1", "correct_held")
        }
        wrong_base = _base_output(runtime, arms["wrong_fit0"].bank)
    wrong_teacher, teacher_metrics = _wrong_teacher(
        runtime, task, arms["wrong_fit0"], wrong_base
    )
    target = {**correct, "wrong_fit0": _target(wrong_teacher)}
    denominator = _family_distances(runtime, wrong_base, target["wrong_fit0"])
    if any(
        not bool(torch.isfinite(value)) or float(value) <= 1e-12
        for value in denominator.values()
    ):
        raise RuntimeError("wrong effective-rank4 teacher missed a family")
    return (
        target,
        {family: value.detach() for family, value in denominator.items()},
        teacher_metrics,
        wrong_teacher,
    )


def _zero_equivalence(runtime: Any, arms: Mapping[str, TaskLocalArm]) -> float:
    errors = []
    with torch.inference_mode():
        for arm in arms.values():
            base = _base_output(runtime, arm.bank)
            observed = _interaction_output(runtime, arm)
            for left, right in zip(
                (*base.residual.a, *base.residual.b),
                (*observed.residual.a, *observed.residual.b),
                strict=True,
            ):
                errors.append((left.float() - right.float()).abs().max())
    maximum = float(torch.stack(errors).max())
    if maximum > 2e-6:
        raise RuntimeError(f"bank-set step0 did not reproduce R5: {maximum}")
    return maximum


def _summary_mass_metrics(
    runtime: Any, arms: Mapping[str, TaskLocalArm]
) -> dict[str, float] | None:
    if routing_stage(runtime.config) == BANK_CONDITIONED_PRIMAL_STAGE:
        values = tuple(
            torch.einsum(
                "f,fe->e",
                arm.bank.video.frame_measure.detach().float(),
                arm.bank.context.canonical_assignment.detach().float(),
            )
            for arm in arms.values()
        )
        masses = torch.cat(values)
        return {"minimum": float(masses.min()), "maximum": float(masses.max())}
    values = []
    for arm in arms.values():
        if arm.summaries is None:
            return None
        values.extend(summary.event_mass for summary in arm.summaries.inputs)
        for groups in arm.summaries.outputs:
            for summary in groups:
                values.extend(
                    scope.event_mass for scope in (summary.all_types, *summary.by_type)
                )
    masses = torch.cat(tuple(value.detach().float().reshape(-1) for value in values))
    return {"minimum": float(masses.min()), "maximum": float(masses.max())}


def _train(
    runtime: Any,
    arms: Mapping[str, TaskLocalArm],
    targets: Mapping[str, EffectiveTarget],
    denominators: Mapping[TargetFamily, torch.Tensor],
) -> None:
    schedule = tuple(runtime.config["task_local"].get("arm_schedule", ARM_SCHEDULE))
    if not schedule or any(name not in ARM_SCHEDULE for name in schedule):
        raise ValueError("bank-set task-local arm schedule changed")
    while runtime.optimizer_steps < runtime.stop_after_step:
        tick = time.monotonic()
        name = schedule[runtime.optimizer_steps % len(schedule)]
        runtime.optimizer.zero_grad(set_to_none=True)
        output = _interaction_output(runtime, arms[name])
        distance = _family_distances(runtime, output, targets[name])
        normalized = {
            family: distance[family] / denominators[family] for family in FAMILIES
        }
        loss = torch.stack(tuple(normalized.values())).mean()
        loss.backward()
        if (
            runtime.optimizer_steps == 1
            and runtime.config.get("model", {}).get("b0_query_source")
            == TASKLOCAL_FREE_B0_QUERY
        ):
            query = runtime.writer_state.bank_set_interaction.tasklocal_free_b0_query
            if (
                query.grad is None
                or not bool(torch.isfinite(query.grad).all())
                or not bool(query.grad.abs().sum() > 0)
            ):
                raise RuntimeError("task-local free B0 query has no finite gradient")
        norm = torch.nn.utils.clip_grad_norm_(runtime.trainable_parameters, 1.0)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):
            raise RuntimeError("bank-set effective-rank4 training became non-finite")
        runtime.optimizer.step()
        runtime.scheduler.step()
        runtime.optimizer_steps += 1
        next_lrs = tuple(map(float, runtime.scheduler.get_last_lr()))
        row = {
            "optimizer_step": runtime.optimizer_steps,
            "arm": name,
            "normalized_effective_rank4_mse": float(loss.detach()),
            "families": {
                family.value: float(value.detach())
                for family, value in normalized.items()
            },
            "gradient_norm_before_clip": float(norm),
            "next_lr": next_lrs[0],
            "step_seconds": time.monotonic() - tick,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
        if len(next_lrs) > 1:
            row["tasklocal_free_b0_query_next_lr"] = next_lrs[1]
        if runtime.context.is_main:
            append_jsonl(runtime.args.output_dir / "metrics.jsonl", row)
            runtime.metrics_rows += 1
            if runtime.optimizer_steps % runtime.args.log_every == 0:
                print(row, flush=True)
        if runtime.optimizer_steps in runtime.checkpoint_steps:
            save_ecp_checkpoint(
                output_dir=runtime.args.output_dir,
                macro=runtime.optimizer_steps,
                stage=routing_stage(runtime.config),
                context=runtime.context,
                model=runtime.writer_state,
                optimizer=runtime.optimizer,
                scheduler=runtime.scheduler,
                run_contract_schema=routing_run_schema(runtime.config),
                metrics_rows=runtime.metrics_rows,
            )


def _panel_b(
    runtime: Any,
    *,
    task: int,
    state: Mapping[str, torch.Tensor],
    free_loss: float,
    visits: int,
) -> dict[str, Any]:
    rows = []
    with torch.inference_mode():
        for visit in range(visits):
            batch, panel = functional_panel_batch(
                runtime, task_id=task, panel_name="b", visit_index=visit
            )
            loss = _functional_value(
                runtime, state=state, batch=batch, seed=panel.policy_rng_seed
            )
            rows.append(
                {
                    "visit": visit,
                    "carrier_loss": float(panel.flow_loss),
                    "generated_loss": loss,
                    "benefit_over_carrier": float(panel.flow_loss) - loss,
                }
            )
    carrier = statistics.fmean(row["carrier_loss"] for row in rows)
    generated = statistics.fmean(row["generated_loss"] for row in rows)
    denominator = carrier - float(free_loss)
    return {
        "visits": rows,
        "carrier_loss": carrier,
        "generated_loss": generated,
        "free_primal_loss": float(free_loss),
        "free_primal_benefit": denominator,
        "functional_recovery": (carrier - generated) / denominator,
    }


def _correct_fit_free_loss(
    runtime: Any, task: int, arms: Mapping[str, TaskLocalArm]
) -> float:
    root = (
        runtime.args.asset_root / runtime.config["authorities"]["positive_control_root"]
    ).resolve()
    source = read_json(root / f"task_{task:03d}" / "result.json")
    rows = (*source["evaluation"]["fit_videos"], source["evaluation"]["held_video"])
    free = {
        int(row["video_demo"]): float(row["panel_b"]["free_primal_loss"])
        for row in rows
    }
    return statistics.fmean(
        free[arms[name].bank.video_demo] for name in ("correct_fit0", "correct_fit1")
    )


def _low_rank_inner(
    left_a: torch.Tensor,
    left_b: torch.Tensor,
    right_a: torch.Tensor,
    right_b: torch.Tensor,
) -> torch.Tensor:
    return (
        (left_b.float() @ right_b.float().transpose(0, 1))
        * (left_a.float() @ right_a.float().transpose(0, 1))
    ).sum()


def _delta_geometry(
    runtime: Any,
    *,
    generated: Any,
    base: Any,
    target: EffectiveTarget,
) -> dict[str, Any]:
    rows = {
        family: {
            "prediction": generated.residual.scales.new_zeros(()),
            "target": generated.residual.scales.new_zeros(()),
            "cross": generated.residual.scales.new_zeros(()),
        }
        for family in FAMILIES
    }
    for owner, actual_a, actual_b, base_a, base_b, target_a, target_b in zip(
        runtime.owners,
        generated.residual.a,
        generated.residual.b,
        base.residual.a,
        base.residual.b,
        target.a,
        target.b,
        strict=True,
    ):
        aa = _low_rank_inner(actual_a, actual_b, actual_a, actual_b)
        bb = _low_rank_inner(base_a, base_b, base_a, base_b)
        tt = _low_rank_inner(target_a, target_b, target_a, target_b)
        ab = _low_rank_inner(actual_a, actual_b, base_a, base_b)
        at = _low_rank_inner(actual_a, actual_b, target_a, target_b)
        bt = _low_rank_inner(base_a, base_b, target_a, target_b)
        row = rows[owner.family]
        row["prediction"] = row["prediction"] + (aa + bb - 2.0 * ab)
        row["target"] = row["target"] + (tt + bb - 2.0 * bt)
        row["cross"] = row["cross"] + (at - ab - bt + bb)
    result = {}
    for family, row in rows.items():
        prediction = row["prediction"].clamp_min(0.0)
        target_norm = row["target"].clamp_min(1e-24)
        cross = row["cross"]
        cosine = cross / (prediction * target_norm).clamp_min(1e-24).sqrt()
        best_scale = cross / prediction.clamp_min(1e-24)
        result[family.value] = {
            "prediction_to_target_norm_ratio": float(
                (prediction / target_norm).sqrt()
            ),
            "delta_cosine": float(cosine.clamp(-1.0, 1.0)),
            "best_prediction_scale": float(best_scale),
            "best_scaled_normalized_mse": float(
                ((target_norm - cross.square() / prediction.clamp_min(1e-24)) / target_norm)
                .clamp(0.0, 1.0)
            ),
        }
    return result


def _identifiability_diagnostic(
    runtime: Any,
    *,
    task: int,
    arms: Mapping[str, TaskLocalArm],
    targets: Mapping[str, EffectiveTarget],
    denominators: Mapping[TargetFamily, torch.Tensor],
    wrong_teacher: Any,
) -> dict[str, Any]:
    teacher_state = _complete(runtime, wrong_teacher)
    teacher_panel_b = _panel_b(
        runtime,
        task=task,
        state=teacher_state,
        free_loss=_correct_fit_free_loss(runtime, task, arms),
        visits=16,
    )
    interventions = {}
    with torch.no_grad():
        for name, arm in arms.items():
            base = _base_output(runtime, arm.bank)
            base_target = _target(base)
            token_outputs = {}
            token_rows = {}
            for token_kind in ("correct", "wrong"):
                collector = _CorrectionCollector(
                    float(runtime.config["model"]["interaction_correction_bound"])
                )
                output = _interaction_output(
                    runtime,
                    arm,
                    summary_token=token_kind,
                    correction_observer=collector.observe,
                )
                token_outputs[token_kind] = output
                departure = _family_distances(runtime, output, base_target)
                row = {
                    "departure_normalized_by_wrong_teacher": {
                        family.value: float(departure[family] / denominators[family])
                        for family in FAMILIES
                    },
                    "correction": collector.finalize()["all"],
                }
                if name == "wrong_fit0":
                    row["teacher_delta_geometry"] = _delta_geometry(
                        runtime,
                        generated=output,
                        base=base,
                        target=targets["wrong_fit0"],
                    )
                token_rows[token_kind] = row
            token_effect = _family_distances(
                runtime, token_outputs["wrong"], _target(token_outputs["correct"])
            )
            interventions[name] = {
                "tokens": token_rows,
                "token_effect_normalized_by_wrong_teacher": {
                    family.value: float(token_effect[family] / denominators[family])
                    for family in FAMILIES
                },
            }
    return {
        "teacher_panel_b": teacher_panel_b,
        "token_interventions": interventions,
    }


def _evaluate(
    runtime: Any,
    task: int,
    arms: Mapping[str, TaskLocalArm],
    targets: Mapping[str, EffectiveTarget],
    denominators: Mapping[TargetFamily, torch.Tensor],
) -> dict[str, Any]:
    root = (
        runtime.args.asset_root / runtime.config["authorities"]["positive_control_root"]
    ).resolve()
    source = read_json(root / f"task_{task:03d}" / "result.json")
    rows = (
        *source["evaluation"]["fit_videos"],
        source["evaluation"]["held_video"],
    )
    free = {
        int(row["video_demo"]): float(row["panel_b"]["free_primal_loss"])
        for row in rows
    }
    correct_fit_free = statistics.fmean(
        free[arms[name].bank.video_demo] for name in ("correct_fit0", "correct_fit1")
    )
    visits = 16 if runtime.args.mode == "formal" else 1
    evaluated = {}
    primal_stage = routing_stage(runtime.config) == BANK_CONDITIONED_PRIMAL_STAGE
    for name, arm in arms.items():
        collector = (
            None
            if primal_stage
            else _CorrectionCollector(
                float(runtime.config["model"]["interaction_correction_bound"])
            )
        )
        primal_diagnostics: dict[str, Any] = {}
        with torch.inference_mode():
            output = _interaction_output(
                runtime,
                arm,
                correction_observer=(None if collector is None else collector.observe),
                primal_observer=(
                    primal_diagnostics.update if primal_stage else None
                ),
            )
            state = _complete(runtime, output)
            effective = _effective_rank4_diagnostics(
                runtime,
                output,
                targets[name if name in targets else "wrong_fit0"],
                denominators,
            )
        evaluated[name] = {
            **_panel_b(
                runtime,
                task=task,
                state=state,
                free_loss=(
                    free[arm.bank.video_demo]
                    if name.startswith("correct")
                    else correct_fit_free
                ),
                visits=visits,
            ),
            "effective_rank4": effective,
        }
        if collector is not None:
            evaluated[name]["correction"] = collector.finalize()
        if primal_stage:
            evaluated[name]["primal_response"] = primal_diagnostics
    correct = [
        evaluated[name]["functional_recovery"]
        for name in ("correct_fit0", "correct_fit1")
    ]
    wrong = [
        evaluated[name]["functional_recovery"]
        for name in ("wrong_fit0", "wrong_fit1")
    ]
    gate = runtime.config["gate"]
    checks = {
        "correct_fit_each": min(correct) >= float(gate["correct_fit_each_minimum"]),
        "correct_held": evaluated["correct_held"]["functional_recovery"]
        >= float(gate["correct_held_minimum"]),
        "wrong_each": max(wrong) <= float(gate["wrong_each_maximum"]),
        "margin": min(correct) - max(wrong)
        >= float(gate["minimum_correct_minus_maximum_wrong"]),
        "all_pairs": min(
            (*correct, evaluated["correct_held"]["functional_recovery"])
        )
        > max(wrong),
    }
    if not primal_stage:
        checks["correction_not_broadly_saturated"] = max(
            row["correction"]["all"]["near_bound_fraction"]
            for row in evaluated.values()
        ) < float(gate["maximum_near_bound_fraction"])
    return {
        "arms": evaluated,
        "checks": checks,
        "gate": "pass" if all(checks.values()) else "non_pass",
        "panel_b_backward_calls": 0,
    }


def run(args: Any) -> None:
    if args.task not in TASKS or args.phase != "joint":
        raise ValueError("bank-set task-local runner requires task 1 or 93")
    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    runtime = None
    try:
        runtime = prepare_routing_control_runtime(args, context)
        inventory = runtime.run_contract["inventory"]
        if (
            int(inventory.get("action_meta_module_count", -1)) != 0
            or int(inventory.get("action_meta_parameter_count", -1)) != 0
        ):
            raise RuntimeError("bank-set task-local runtime loaded Action Meta")
        started = time.monotonic()
        arms = _prepare_arms(runtime, args.task)
        step0_error = _zero_equivalence(runtime, arms)
        targets, denominators, teacher, _wrong_teacher_output = _targets(
            runtime, args.task, arms
        )
        _train(runtime, arms, targets, denominators)
        evaluation = _evaluate(runtime, args.task, arms, targets, denominators)
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "stage": routing_stage(runtime.config),
            "task": args.task,
            "role": runtime.panels[args.task].role,
            "step0_maximum_absolute_error": step0_error,
            "summary_event_mass": _summary_mass_metrics(runtime, arms),
            "wrong_teacher": teacher,
            "effective_family_denominators": {
                family.value: float(value) for family, value in denominators.items()
            },
            "evaluation": evaluation,
            "completed_optimizer_steps": runtime.optimizer_steps,
            "elapsed_seconds": time.monotonic() - started,
            "information_wall": {
                "correct_held_backward_calls": 0,
                "wrong_fit1_backward_calls": 0,
                "panel_b_backward_calls": 0,
                "validation_or_test_reads": 0,
                "action_meta_installed": False,
                "single_complete_rank16": True,
                "shuffled_or_reversed_use": False,
            },
        }
        if context.is_main:
            write_json_atomic(args.output_dir / "evaluation.json", result)
            write_json_atomic(
                args.output_dir / "completion.json",
                {
                    "stage": routing_stage(runtime.config),
                    "task": args.task,
                    "gate": evaluation["gate"],
                    "completed_optimizer_steps": runtime.optimizer_steps,
                },
            )
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def diagnose(args: Any) -> None:
    """Audit teacher representativeness and free-summary control without training."""

    if (
        args.task not in TASKS
        or args.phase != "joint"
        or args.resume is not None
        or args.diagnose_checkpoint is None
    ):
        raise ValueError("bank-set identifiability diagnostic arguments changed")
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    runtime = None
    try:
        runtime = prepare_routing_control_runtime(args, context)
        if routing_stage(runtime.config) != BANK_SET_S0_STAGE:
            raise ValueError("bank-set free-summary diagnostic requires S0")
        checkpoint = args.diagnose_checkpoint
        tensor_path = checkpoint / "ecp.safetensors" if checkpoint.is_dir() else checkpoint
        if not tensor_path.is_file():
            raise FileNotFoundError(tensor_path)
        runtime.writer_state.load_state_dict(
            load_file(str(tensor_path), device=str(context.device)), strict=True
        )
        started = time.monotonic()
        arms = _prepare_arms(runtime, args.task)
        targets, denominators, teacher, wrong_teacher = _targets(
            runtime, args.task, arms
        )
        diagnostic = _identifiability_diagnostic(
            runtime,
            task=args.task,
            arms=arms,
            targets=targets,
            denominators=denominators,
            wrong_teacher=wrong_teacher,
        )
        result = {
            "schema_version": "ember_ecp_event_bank_set_s0_identifiability_v1",
            "status": "complete",
            "stage": routing_stage(runtime.config),
            "task": args.task,
            "writer_checkpoint": {
                "path": str(tensor_path.resolve()),
                "bytes": tensor_path.stat().st_size,
            },
            "wrong_teacher": teacher,
            "diagnostic": diagnostic,
            "elapsed_seconds": time.monotonic() - started,
            "information_wall": {
                "panel_b_backward_calls": 0,
                "validation_or_test_reads": 0,
                "action_meta_installed": False,
                "single_complete_rank16": True,
                "shuffled_or_reversed_use": False,
            },
        }
        if context.is_main:
            write_json_atomic(args.output_dir / "identifiability.json", result)
            write_json_atomic(
                args.output_dir / "completion.json",
                {
                    "stage": routing_stage(runtime.config),
                    "task": args.task,
                    "status": "complete",
                },
            )
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
