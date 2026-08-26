from __future__ import annotations

import torch

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.native_factors import NativeFactorResidual
from ember.ecp.shared_compiler import SharedCompilerOutput
from ember.ecp.bank_conditioning.mapping import (
    MappingCondition,
    SharedCompilerMappingSchedule,
    SharedCompilerMappingSplit,
    cross_video_consistency_loss,
    paired_mapping_loss,
)
from ember.ecp.bank_conditioning.mapping_eval_runtime import (
    balanced_mapping_assignments,
)
from ember.ecp.bank_conditioning.mapping_gate import summarize_mapping_rows
from ember.ecp.shared_compiler_native_teacher import NativeTeacherFactors


def _owners() -> tuple[TargetOwner, ...]:
    return tuple(
        TargetOwner(index, family.value, family, None, 6, 5)
        for index, family in enumerate(TargetFamily)
    )


def _output(seed: int = 7) -> SharedCompilerOutput:
    generator = torch.Generator().manual_seed(seed)
    a = tuple(
        torch.randn(4, 6, generator=generator).requires_grad_() for _ in _owners()
    )
    directions = tuple(
        torch.randn(4, 5, generator=generator).requires_grad_() for _ in _owners()
    )
    scales = torch.ones(len(_owners()), 4, requires_grad=True)
    return SharedCompilerOutput(
        residual=NativeFactorResidual(
            a=a,
            b=tuple(value * scales[index, :, None] for index, value in enumerate(directions)),
            scales=scales,
        ),
        input_directions=a,
        output_directions=directions,
        video_weights=torch.ones(1),
        frame_measures=(torch.ones(len(_owners()), 4, 2) * 0.5,),
        output_group_gains=(torch.ones(4, 4),),
        solve_metrics=torch.ones(1, 4),
        global_statistics_enabled=True,
    )


def _teacher(output: SharedCompilerOutput, *, member: str) -> NativeTeacherFactors:
    return NativeTeacherFactors(
        authority_id=1,
        video_demo=2,
        member_name=member,
        a=tuple(value.detach().clone() for value in output.input_directions),
        b=tuple(value.detach().clone() for value in output.output_directions),
        scales=output.residual.scales.detach().clone(),
        provenance={},
    )


def test_mapping_credit_is_set_valued_family_balanced_and_scale_stopped() -> None:
    output = _output()
    exact = _teacher(output, member="exact")
    other_output = _output(seed=11)
    other = _teacher(other_output, member="other")
    loss = paired_mapping_loss(
        output=output,
        teachers=(exact, other),
        owners=_owners(),
        temperature=0.1,
    )
    assert loss.best_member == 0
    torch.testing.assert_close(
        loss.best_family_recovery, torch.ones(4), rtol=1e-5, atol=1e-5
    )
    loss.total.backward()
    assert all(value.grad is not None for value in output.input_directions)
    assert all(value.grad is not None for value in output.output_directions)
    assert output.residual.scales.grad is None

    companion = _output()
    consistency = cross_video_consistency_loss(
        primary_output=output,
        companion_output=companion,
        primary_teachers=(exact,),
        companion_teachers=(_teacher(companion, member="exact"),),
        owners=_owners(),
        responsibilities=torch.ones(1),
        margin=0.05,
    )
    assert float(consistency.total.detach()) == 0.0


def test_mapping_schedule_keeps_fixed_role_weight_and_world_invariance() -> None:
    fit = []
    members = {}
    for task in range(40):
        role = "meta_fit" if task < 25 else "target_fit"
        members[task] = ("member",)
        fit.extend(
            MappingCondition(task, role, video, sampled_frames=10 + task + video)
            for video in (0, 1)
        )
    split = SharedCompilerMappingSplit(
        fit=tuple(fit), video_held=(), task_held=(), member_names=members
    )
    schedule = SharedCompilerMappingSchedule(split, seed=31)
    groups = schedule.task_groups(0)
    assert len(groups) == 5
    assert all(
        sum(task < 25 for task in group) == 3
        and sum(task >= 25 for task in group) == 3
        for group in groups
    )
    group = groups[0]
    conditions = {
        task: schedule.condition(task, macro=0, update=0) for task in group
    }
    for world_size in range(1, 7):
        assignments = schedule.assignments(group, conditions, world_size)
        assert sorted(task for row in assignments for task in row) == sorted(group)
        assert len(assignments) == world_size


def test_mapping_evaluation_balances_cost_and_weights_tasks_not_videos() -> None:
    conditions = tuple(
        (
            "fit",
            MappingCondition(task, "meta_fit", video, sampled_frames=cost),
        )
        for task, video, cost in ((1, 0, 90), (1, 1, 10), (2, 0, 50), (3, 0, 50))
    )
    assignments = balanced_mapping_assignments(conditions, worker_count=2)
    loads = [sum(row.sampled_frames for _, row in worker) for worker in assignments]
    assert sorted(loads) == [100, 100]

    rows = []
    for split in ("fit", "video_holdout", "task_holdout"):
        rows.extend(
            (
                {
                    "split": split,
                    "authority_id": 1,
                    "mean_best_recovery": recovery,
                    "best_family_recovery": {
                        name: recovery for name in ("q", "v", "action_in", "action_out")
                    },
                }
                for recovery in (0.0, 1.0)
            )
        )
        rows.append(
            {
                "split": split,
                "authority_id": 2,
                "mean_best_recovery": 1.0,
                "best_family_recovery": {
                    name: 1.0 for name in ("q", "v", "action_in", "action_out")
                },
            }
        )
    summary = summarize_mapping_rows(rows)
    assert summary["fit"]["condition_recovery"]["mean"] == 2.0 / 3.0
    assert summary["fit"]["task_recovery"]["mean"] == 0.75
