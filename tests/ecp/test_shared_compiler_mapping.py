from __future__ import annotations

from pathlib import Path

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
from ember.ecp.bank_conditioning.mapping_gate import (
    _gate_report,
    summarize_mapping_rows,
)
from ember.ecp.bank_conditioning.program_causality import (
    load_program_causality_contract,
    program_causality_checks,
    program_causality_pairs,
    summarize_program_causality_rows,
)
from ember.ecp.bank_conditioning.consensus import truncated_mean_update
from ember.ecp.bank_conditioning.f0 import _low_rank_update_similarity
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
        conditioning_metrics=torch.ones(1, 6),
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


def test_f0_chunk_reference_compares_effective_update_not_rank_gauge() -> None:
    generator = torch.Generator().manual_seed(17)
    a = torch.randn(4, 9, generator=generator, dtype=torch.float64)
    b = torch.randn(4, 7, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(
        torch.randn(4, 4, generator=generator, dtype=torch.float64)
    )
    rotated_a = q @ a
    rotated_b = q @ b

    assert float((a - rotated_a).abs().max()) > 1e-2
    cosine, relative_error = _low_rank_update_similarity(
        a, b, rotated_a, rotated_b
    )
    assert cosine >= 1.0 - 1e-6
    assert relative_error <= 1e-6


def test_functional_consensus_averages_updates_not_factor_gauges() -> None:
    generator = torch.Generator().manual_seed(19)
    a = torch.randn(4, 9, generator=generator)
    b = torch.randn(7, 4, generator=generator)
    q, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator))
    consensus_a, consensus_b = truncated_mean_update(
        ((a, b), (q @ a, b @ q.transpose(0, 1))), rank=4
    )
    torch.testing.assert_close(
        consensus_b @ consensus_a, b @ a, rtol=2e-5, atol=2e-5
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
    components = (
        loss.input_subspace,
        loss.output_subspace,
        loss.update_direction,
    )
    assert all(
        torch.isfinite(component) and component >= 0 for component in components
    )
    torch.testing.assert_close(loss.total, loss.update_direction)
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
    augmented = balanced_mapping_assignments(
        conditions, worker_count=2, extra_costs={(1, 1): 80}
    )
    augmented_loads = [
        sum(
            row.sampled_frames
            + (80 if (row.authority_id, row.video_demo) == (1, 1) else 0)
            for _, row in worker
        )
        for worker in augmented
    ]
    assert sorted(augmented_loads) == [140, 140]

    rows = []
    for split in ("fit", "video_holdout", "task_holdout"):
        rows.extend(
            (
                {
                    "split": split,
                    "authority_id": 1,
                    "role": "meta_fit" if split != "task_holdout" else "meta_held",
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
                "role": "target_fit" if split != "task_holdout" else "target_held",
                "mean_best_recovery": 1.0,
                "best_family_recovery": {
                    name: 1.0 for name in ("q", "v", "action_in", "action_out")
                },
            }
        )
    summary = summarize_mapping_rows(rows)
    assert summary["fit"]["condition_recovery"]["mean"] == 2.0 / 3.0
    assert summary["fit"]["task_recovery"]["mean"] == 0.75
    assert summary["fit"]["role_task_recovery"]["meta_fit"]["mean"] == 0.5
    assert summary["fit"]["role_task_recovery"]["target_fit"]["mean"] == 1.0


def test_program_causality_panel_is_fit_only_same_role_and_task_robust() -> None:
    contract = load_program_causality_contract(
        Path(__file__).resolve().parents[2]
        / "configs/pi05_ecp_shared_compiler_g3_f3_program_causality_v1.json"
    )
    fit = []
    members = {}
    for task in range(40):
        role = "meta_fit" if task < 25 else "target_fit"
        members[task] = ("member",)
        fit.extend(
            MappingCondition(task, role, video, sampled_frames=10 + video)
            for video in (7, 3)
        )
    split = SharedCompilerMappingSplit(
        fit=tuple(fit), video_held=(), task_held=(), member_names=members
    )
    pairs = program_causality_pairs(split)
    assert len(pairs) == 40
    assert all(pair.primary.video_demo == 3 for pair in pairs)
    assert all(
        pair.primary.authority_id != pair.wrong.authority_id
        and pair.primary.role == pair.wrong.role
        for pair in pairs
    )

    rows = []
    for pair in pairs:
        correct = 0.8 if pair.primary.role == "meta_fit" else 0.7
        wrong = correct - 0.2
        rows.append(
            {
                "authority_id": pair.primary.authority_id,
                "role": pair.primary.role,
                "video_demo": pair.primary.video_demo,
                "wrong_authority_id": pair.wrong.authority_id,
                "wrong_video_demo": pair.wrong.video_demo,
                "correct": {
                    "mean_best_recovery": correct,
                    "best_family_recovery": {
                        family: correct
                        for family in ("q", "v", "action_in", "action_out")
                    },
                },
                "wrong": {
                    "mean_best_recovery": wrong,
                    "best_family_recovery": {
                        family: wrong
                        for family in ("q", "v", "action_in", "action_out")
                    },
                },
            }
        )
    summary = summarize_program_causality_rows(rows)
    assert program_causality_checks(summary, contract) == {
        "meta_fit": True,
        "target_fit": True,
    }
    assert summary["meta_fit"]["positive_task_fraction"] == 1.0

    mapping_summary = {
        "fit": {"task_recovery": {"median": 0.8}},
        "video_holdout": {
            "task_recovery": {"median": 0.8, "p10": 0.6},
            "tasks": [],
        },
    }
    mapping_config = {
        "mapping_gate": {
            "f3_held_video_median_minimum": 0.75,
            "f3_held_video_p10_minimum": 0.5,
            "f3_held_to_fit_minimum": 0.8,
            "adjacent_checkpoint_stability": {
                "maximum_median_absolute_task_delta": 0.1,
                "maximum_held_median_drop": 0.05,
            },
        },
        "formal_run": {"checkpoint_macros": [1, 2]},
    }
    gate = _gate_report(
        phase="f3",
        macro=1,
        config=mapping_config,
        summary=mapping_summary,
        program_causality=summary,
        program_causality_contract=contract,
        previous=None,
    )
    assert gate["primary_pass"]
    summary["target_fit"]["correct_minus_wrong_program"]["median"] = 0.09
    failed = _gate_report(
        phase="f3",
        macro=1,
        config=mapping_config,
        summary=mapping_summary,
        program_causality=summary,
        program_causality_contract=contract,
        previous=None,
    )
    assert not failed["primary_checks"]["correct_vs_wrong_program"]
