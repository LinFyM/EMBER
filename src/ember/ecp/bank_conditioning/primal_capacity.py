"""Task-local primal capacity objects for the G3 P1 qualification."""

from __future__ import annotations

import statistics
from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch

from ember.ecp.bank_conditioning.consensus import truncated_mean_update
from ember.ecp.bank_conditioning.mapping import paired_mapping_loss
from ember.ecp.bank_conditioning.primal_dual_runtime import (
    CompactPrimalDualVideo,
    MaterializedPrimalDualVideo,
    PrimalDualVideoOperator,
)
from ember.ecp.contracts import TargetOwner
from ember.ecp.native_factors import (
    NativeFactorResidual,
    native_output_group_count,
    rms_normalize,
)
from ember.ecp.shared_compiler import SharedCompilerOutput
from ember.ecp.shared_compiler_native_teacher import NativeTeacherFactors


BANK_INTERACTION_CONTROL_SCHEMA = "ember_ecp_bank_interaction_positive_control_v1"


def is_bank_interaction_control_config(config: Mapping[str, Any]) -> bool:
    return config.get("schema_version") == BANK_INTERACTION_CONTROL_SCHEMA


def bank_interaction_control_config_valid(config: Mapping[str, Any]) -> bool:
    model = config.get("model", {})
    optimization = config.get("optimization", {})
    wall = config.get("information_wall", {})
    return all(
        (
            is_bank_interaction_control_config(config),
            config.get("status") == "active_bank_interaction_positive_control",
            model.get("program_initialization") == "c1493a1_macro20_model_tensors",
            model.get("primal_scorer_initialization") == "fresh",
            model.get("inverse_covariance_power") == 0.5,
            optimization.get("loss")
            == "task_local_fit_symmetric_transport_functional_only",
            optimization.get("task_local_positive_control", {}).get("initialization")
            == "fit_symmetric_transport",
            "counterfactual" not in optimization.get("joint", {}),
            wall.get("diagnostic_only") is True,
            wall.get("deployment_candidate") is False,
            wall.get("task_local_primals_training_only") is True,
            wall.get("held_video_backward_calls") == 0,
            wall.get("wrong_bank_backward_calls") == 0,
            wall.get("single_complete_rank16") is True,
        )
    )


class TaskLocalPrimalCode(torch.nn.Module):
    """One free primal code and one fixed fit-only scale shared across videos."""

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        teachers: Sequence[NativeTeacherFactors],
        *,
        s_ref: torch.Tensor,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        rows = tuple(teachers)
        if not self.owners or not rows or s_ref.shape != (len(self.owners),):
            raise ValueError("P1 task-local primal initialization changed")
        input_rows = []
        output_rows = []
        scales = []
        for target, owner in enumerate(self.owners):
            pairs = tuple(
                (
                    teacher.a[target],
                    teacher.b[target].transpose(0, 1)
                    * teacher.scales[target][None],
                )
                for teacher in rows
            )
            a, b = truncated_mean_update(pairs, rank=4)
            a_direction, b_direction, scale = _factor_directions(a, b)
            groups = native_output_group_count(owner)
            output_rows.append(
                b_direction.reshape(4, groups, owner.out_features // groups)
                .permute(1, 0, 2)
                .contiguous()
            )
            input_rows.append(a_direction)
            scales.append(scale)
        scale = torch.stack(scales).to(s_ref)
        ratio = (scale / s_ref[:, None].clamp_min(1e-12)).clamp(1e-4, 0.95)
        self.input_code = torch.nn.ParameterList(
            torch.nn.Parameter(value.to(s_ref)) for value in input_rows
        )
        self.output_code = torch.nn.ParameterList(
            torch.nn.Parameter(value.to(s_ref)) for value in output_rows
        )
        self.register_buffer("fixed_scales", s_ref[:, None] * ratio)

    @classmethod
    def from_serialized(
        cls,
        owners: Sequence[TargetOwner],
        state: dict[str, torch.Tensor],
    ) -> TaskLocalPrimalCode:
        """Load a sealed task code without rereading training-only teachers."""

        owner_rows = tuple(owners)
        expected = {
            "fixed_scales",
            *(f"input_code.{index}" for index in range(len(owner_rows))),
            *(f"output_code.{index}" for index in range(len(owner_rows))),
        }
        if set(state) != expected:
            raise ValueError("serialized task-local primal inventory changed")
        fixed_scales = state["fixed_scales"]
        if fixed_scales.shape != (len(owner_rows), 4):
            raise ValueError("serialized task-local scales changed")
        code = cls.__new__(cls)
        torch.nn.Module.__init__(code)
        code.owners = owner_rows
        input_rows = tuple(
            state[f"input_code.{index}"] for index in range(len(owner_rows))
        )
        output_rows = tuple(
            state[f"output_code.{index}"] for index in range(len(owner_rows))
        )
        if any(
            value.shape != (4, owner.in_features)
            for value, owner in zip(input_rows, owner_rows, strict=True)
        ) or any(
            value.shape
            != (
                native_output_group_count(owner),
                4,
                owner.out_features // native_output_group_count(owner),
            )
            for value, owner in zip(output_rows, owner_rows, strict=True)
        ):
            raise ValueError("serialized task-local primal shape changed")
        code.input_code = torch.nn.ParameterList(
            torch.nn.Parameter(value.clone()) for value in input_rows
        )
        code.output_code = torch.nn.ParameterList(
            torch.nn.Parameter(value.clone()) for value in output_rows
        )
        code.register_buffer("fixed_scales", fixed_scales.clone())
        return code

    def input_primals(self) -> tuple[torch.Tensor, ...]:
        return tuple(rms_normalize(value) for value in self.input_code)

    def output_primals(self) -> tuple[torch.Tensor, ...]:
        return tuple(rms_normalize(value) for value in self.output_code)

    def scales(self, s_ref: torch.Tensor) -> torch.Tensor:
        if s_ref.shape != (len(self.owners),):
            raise ValueError("P1 fixed scale authority changed")
        return self.fixed_scales.to(s_ref)


def inverse_fractional_primal_transport(
    operator: object,
    primal: torch.Tensor,
    *,
    inverse_power: float,
) -> torch.Tensor:
    if inverse_power not in (0.25, 0.5):
        raise ValueError("fit spectral transport inverse power changed")
    basis = operator.basis.to(primal).float()
    eigenvalues = operator.eigenvalues.to(primal).float()
    relative = eigenvalues / eigenvalues[-1].clamp_min(1e-30)
    coordinates = primal.float() @ basis
    denominator = (
        relative.sqrt() if inverse_power == 0.5 else relative.pow(0.25)
    )
    transported = (coordinates / denominator[None]) @ basis.T
    if not bool(torch.isfinite(transported).all()):
        raise ValueError("fit spectral transport became non-finite")
    return transported.to(primal)


def initialize_fit_spectral_transport(
    code: TaskLocalPrimalCode,
    banks: Sequence[CompactPrimalDualVideo],
    *,
    operator_inverse_power: float = 0.5,
) -> dict[str, float]:
    """Put one fit-only task code in the shared spectral coordinate.

    Each fit video independently transports the same teacher-initialized code;
    the arithmetic mean is only a task-local capacity initialization.  Held or
    wrong-task banks never enter this operation.
    """

    rows = tuple(banks)
    if len(rows) != 2 or operator_inverse_power not in (0.5, 0.75):
        raise ValueError("fit spectral transport contract changed")
    transport_power = 1.0 - operator_inverse_power
    base_inputs = code.input_primals()
    base_outputs = code.output_primals()
    transported_inputs = []
    transported_outputs = []
    for bank in rows:
        if (
            len(bank.input_operators) != len(base_inputs)
            or len(bank.output_operators) != len(base_outputs)
        ):
            raise ValueError("fit symmetric transport target count changed")
        transported_inputs.append(
            tuple(
                inverse_fractional_primal_transport(
                    operator, primal, inverse_power=transport_power
                )
                for operator, primal in zip(
                    bank.input_operators, base_inputs, strict=True
                )
            )
        )
        transported_outputs.append(
            tuple(
                torch.stack(
                    tuple(
                        inverse_fractional_primal_transport(
                            operator,
                            primal[group],
                            inverse_power=transport_power,
                        )
                        for group, operator in enumerate(operators)
                    )
                )
                for operators, primal in zip(
                    bank.output_operators, base_outputs, strict=True
                )
            )
        )

    averaged_inputs = tuple(
        0.5 * (left + right)
        for left, right in zip(
            transported_inputs[0], transported_inputs[1], strict=True
        )
    )
    averaged_outputs = tuple(
        0.5 * (left + right)
        for left, right in zip(
            transported_outputs[0], transported_outputs[1], strict=True
        )
    )
    cosines = []
    for left, right in zip(
        (*transported_inputs[0], *transported_outputs[0]),
        (*transported_inputs[1], *transported_outputs[1]),
        strict=True,
    ):
        width = left.shape[-1]
        cosines.extend(
            torch.nn.functional.cosine_similarity(
                left.float().reshape(-1, width),
                right.float().reshape(-1, width),
                dim=-1,
            ).tolist()
        )
    with torch.no_grad():
        for parameter, value in zip(
            code.input_code, averaged_inputs, strict=True
        ):
            parameter.copy_(value.to(parameter))
        for parameter, value in zip(
            code.output_code, averaged_outputs, strict=True
        ):
            parameter.copy_(value.to(parameter))
    return {
        "minimum": min(cosines),
        "median": statistics.median(cosines),
        "mean": statistics.fmean(cosines),
    }


def subset_teacher(
    teacher: NativeTeacherFactors, target_indices: Sequence[int]
) -> NativeTeacherFactors:
    indices = tuple(map(int, target_indices))
    return replace(
        teacher,
        a=tuple(teacher.a[index] for index in indices),
        b=tuple(teacher.b[index] for index in indices),
        scales=teacher.scales[list(indices)],
        provenance={**teacher.provenance, "P1_target_indices": list(indices)},
    )


def task_local_output(
    *,
    operator: PrimalDualVideoOperator,
    prepared: MaterializedPrimalDualVideo | CompactPrimalDualVideo,
    code: TaskLocalPrimalCode,
    s_ref: torch.Tensor,
    inverse_covariance_power_override: float | None = None,
) -> SharedCompilerOutput:
    if isinstance(prepared, CompactPrimalDualVideo):
        result = operator.apply_compact(
            prepared,
            code.input_primals(),
            code.output_primals(),
            inverse_covariance_power_override=inverse_covariance_power_override,
        )
    else:
        result = operator.apply_materialized(
            prepared,
            code.input_primals(),
            code.output_primals(),
            inverse_covariance_power_override=inverse_covariance_power_override,
        )
    scales = code.scales(s_ref)
    inputs = tuple(rms_normalize(value) for value in result.input_values)
    outputs = tuple(rms_normalize(value) for value in result.output_values)
    scaled = tuple(
        value * scales[target, :, None]
        for target, value in enumerate(outputs)
    )
    return SharedCompilerOutput(
        residual=NativeFactorResidual(a=inputs, b=scaled, scales=scales),
        input_directions=inputs,
        output_directions=outputs,
        video_weights=scales.new_ones(1),
        frame_measures=(result.frame_measure,),
        output_group_gains=(result.group_gains,),
        solve_metrics=result.solve_metrics[None],
        conditioning_metrics=result.conditioning_metrics[None],
        compatibility_supports=(
            None
            if result.compatibility_support is None
            else result.compatibility_support[None]
        ),
        selected_inverse_covariance_powers=(
            None
            if result.selected_inverse_covariance_power is None
            else scales.new_tensor((result.selected_inverse_covariance_power,))
        ),
    )


def teacher_output(teacher: NativeTeacherFactors) -> SharedCompilerOutput:
    scaled = tuple(
        value * teacher.scales[target, :, None]
        for target, value in enumerate(teacher.b)
    )
    device = teacher.scales.device
    return SharedCompilerOutput(
        residual=NativeFactorResidual(
            a=teacher.a,
            b=scaled,
            scales=teacher.scales,
        ),
        input_directions=teacher.a,
        output_directions=teacher.b,
        video_weights=teacher.scales.new_ones(1),
        frame_measures=(),
        output_group_gains=(),
        solve_metrics=torch.empty(0, device=device),
        conditioning_metrics=torch.empty(0, device=device),
    )


def recovery_record(
    output: SharedCompilerOutput,
    teachers: Sequence[NativeTeacherFactors],
    owners: Sequence[TargetOwner],
    *,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    loss = paired_mapping_loss(
        output=output,
        teachers=teachers,
        owners=owners,
        temperature=temperature,
    )
    family = loss.best_family_recovery
    return loss.total, {
        "mean_recovery": float(family.detach().mean()),
        "family_recovery": {
            name: float(value)
            for name, value in zip(
                ("q", "v", "action_in", "action_out"),
                family.detach().cpu(),
                strict=True,
            )
        },
        "best_member": int(loss.best_member),
        "input_subspace_loss": float(loss.input_subspace.detach()),
        "output_subspace_loss": float(loss.output_subspace.detach()),
        "update_direction_loss": float(loss.update_direction.detach()),
    }


def optimistic_recovery_record(
    video_teachers: Sequence[NativeTeacherFactors],
    consensus_teachers: Sequence[NativeTeacherFactors],
    owners: Sequence[TargetOwner],
    *,
    temperature: float,
) -> dict[str, object]:
    rows = []
    for teacher in video_teachers:
        _, record = recovery_record(
            teacher_output(teacher),
            consensus_teachers,
            owners,
            temperature=temperature,
        )
        rows.append(record)
    return max(rows, key=lambda value: float(value["mean_recovery"]))


def _factor_directions(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a_scale = a.square().mean(-1).sqrt().clamp_min(1e-12)
    b_rows = b.transpose(0, 1)
    b_scale = b_rows.square().mean(-1).sqrt().clamp_min(1e-12)
    return a / a_scale[:, None], b_rows / b_scale[:, None], a_scale * b_scale
