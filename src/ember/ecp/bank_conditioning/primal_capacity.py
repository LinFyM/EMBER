"""Task-local primal capacity objects for the G3 P1 qualification."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

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

    def input_primals(self) -> tuple[torch.Tensor, ...]:
        return tuple(rms_normalize(value) for value in self.input_code)

    def output_primals(self) -> tuple[torch.Tensor, ...]:
        return tuple(rms_normalize(value) for value in self.output_code)

    def scales(self, s_ref: torch.Tensor) -> torch.Tensor:
        if s_ref.shape != (len(self.owners),):
            raise ValueError("P1 fixed scale authority changed")
        return self.fixed_scales.to(s_ref)


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
) -> SharedCompilerOutput:
    if isinstance(prepared, CompactPrimalDualVideo):
        result = operator.apply_compact(
            prepared, code.input_primals(), code.output_primals()
        )
    else:
        result = operator.apply_materialized(
            prepared, code.input_primals(), code.output_primals()
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
