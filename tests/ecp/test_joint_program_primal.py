from types import SimpleNamespace

import torch

from ember.ecp.joint_program_primal.evaluation import (
    _language_program,
    _normalized,
    _task_conditions,
)
from ember.ecp.joint_program_primal.evaluation_gate import _interaction
from ember.ecp.joint_program_primal.train_step import joint_task_group
from ember.ecp.bank_conditioning.mapping import MappingCondition
from ember.ecp.natural_program import NaturalProgram


def test_joint_task_schedule_is_role_balanced_and_task_equal() -> None:
    runtime = SimpleNamespace(
        config={
            "task_split": {
                "gradient_meta": [1, 8, 9, 32, 52],
                "gradient_target": [72, 73, 75, 93, 94],
            }
        }
    )
    groups = tuple(joint_task_group(runtime, step) for step in range(5))
    assert all(len(group) == len(set(group)) == 6 for group in groups)
    assert all(sum(task < 72 for task in group) == 3 for group in groups)
    counts = {
        task: sum(task in group for group in groups)
        for task in (1, 8, 9, 32, 52, 72, 73, 75, 93, 94)
    }
    assert set(counts.values()) == {3}


def test_language_control_removes_every_video_program_field() -> None:
    program = NaturalProgram(
        p_lang=torch.randn(38, 4),
        p_scene=torch.randn(38, 4),
        p_process=torch.randn(8, 38, 4),
        rho=torch.rand(8),
        tau=torch.rand(8, 2),
        sigma=torch.randn(8, 38, 4),
    )
    control = _language_program(program)
    assert torch.equal(control.p_lang, program.p_lang)
    assert torch.count_nonzero(control.p_scene) == 0
    assert torch.count_nonzero(control.p_process) == 0
    assert torch.count_nonzero(control.sigma) == 0
    assert torch.allclose(control.rho, torch.full((8,), 0.125))
    assert torch.allclose(control.tau[0], torch.tensor([0.0, 0.125]))
    assert torch.allclose(control.tau[-1], torch.tensor([0.875, 1.0]))


def test_functional_recovery_and_interaction_use_one_free_primal_measure() -> None:
    def arm(generated: float) -> dict[str, float]:
        return _normalized(
            {
                "carrier_loss": 1.0,
                "generated_loss": generated,
                "benefit_over_carrier": 1.0 - generated,
                "visits": [],
            },
            free_loss=0.5,
        )

    controls = {
        "primary_correct": arm(0.6),
        "wrong_program_correct_bank": arm(0.8),
        "correct_program_wrong_bank": arm(0.85),
        "wrong_program_wrong_bank": arm(0.95),
    }
    assert controls["primary_correct"]["functional_recovery"] == 0.8
    assert abs(_interaction(controls) - 0.2) < 1e-12


def test_true_task_holdout_uses_lowest_three_sealed_videos() -> None:
    sealed = (4, 25, 27, 31, 32, 38, 39, 40, 46)
    rows = tuple(MappingCondition(2, "meta_fit", demo, 10 + demo) for demo in sealed)
    runtime = SimpleNamespace(
        mapping_split=SimpleNamespace(
            fit_by_task={}, video_held_by_task={}, task_held=rows
        ),
        # Functional-panel action episodes are a separate authority from the
        # sealed native-teacher videos used to compile the Program and bank.
        panels={
            2: SimpleNamespace(
                role="meta_fit", program_video_demos=tuple(range(9))
            )
        },
    )
    selected = _task_conditions(runtime, 2)
    assert tuple(row.video_demo for row in selected) == (4, 25, 27)
