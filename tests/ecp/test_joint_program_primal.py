from pathlib import Path
from types import SimpleNamespace

import torch

from ember.ecp.joint_program_primal.evaluation import (
    _language_program,
    _normalized,
    _task_conditions,
    balanced_task_assignments,
)
from ember.ecp.joint_program_primal.evaluation_gate import _interaction
from ember.ecp.joint_program_primal.routing_control import (
    ROUTING_TASK_IDS,
    fixed_routing_program,
    fixed_routing_token,
    load_routing_control_config,
)
from ember.ecp.joint_program_primal.routing_control_evaluation import (
    _training_world_size,
    routing_task_assignments,
)
from ember.ecp.joint_program_primal.train_step import (
    counterfactual_arm,
    counterfactual_hinge,
    counterfactual_task_pairs,
    joint_task_group,
)
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


def test_counterfactual_schedule_is_same_role_cyclic_and_alternating() -> None:
    runtime = SimpleNamespace(
        config={
            "task_split": {
                "gradient_meta": [1, 8, 9, 32, 52],
                "gradient_target": [72, 73, 75, 93, 94],
            }
        }
    )
    group = joint_task_group(runtime, 0)
    pairs = counterfactual_task_pairs(runtime, group)
    assert pairs == {1: 8, 8: 9, 9: 1, 72: 73, 73: 75, 75: 72}
    assert counterfactual_arm(0) == "wrong_program"
    assert counterfactual_arm(1) == "wrong_bank"
    active, gap, margin, hinge = counterfactual_hinge(
        correct_loss=0.20,
        negative_loss=0.205,
        margin_scale=0.10,
        normalized_margin=0.10,
    )
    assert active is True
    assert torch.allclose(
        torch.tensor((gap, margin, hinge)), torch.tensor((0.005, 0.01, 0.005))
    )
    assert counterfactual_hinge(
        correct_loss=0.20,
        negative_loss=0.22,
        margin_scale=0.10,
        normalized_margin=0.10,
    )[0] is False


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


def test_six_worker_gate_assignment_keeps_exactly_two_tasks_per_worker() -> None:
    task_ids = (1, 2, 8, 9, 32, 52, 72, 73, 74, 75, 93, 94)
    fit_ids = set(task_ids) - {2, 74}
    fit = {
        task: tuple(
            MappingCondition(task, "meta_fit" if task < 72 else "target_fit", demo, 10)
            for demo in (0, 1)
        )
        for task in fit_ids
    }
    held = {
        task: (
            MappingCondition(
                task, "meta_fit" if task < 72 else "target_fit", 2, 10
            ),
        )
        for task in fit_ids
    }
    task_held = tuple(
        MappingCondition(task, "meta_fit" if task < 72 else "target_fit", demo, 10)
        for task in (2, 74)
        for demo in (0, 1, 2)
    )
    runtime = SimpleNamespace(
        mapping_split=SimpleNamespace(
            fit_by_task=fit,
            video_held_by_task=held,
            task_held=task_held,
        ),
        panels={
            task: SimpleNamespace(role="meta_fit" if task < 72 else "target_fit")
            for task in task_ids
        },
    )
    assignments = balanced_task_assignments(runtime, worker_count=6)
    assert all(len(tasks) == 2 for tasks in assignments)
    assert {task for tasks in assignments for task in tasks} == set(task_ids)


def test_routing_control_tokens_are_fixed_mean_zero_and_orthogonal() -> None:
    tokens = torch.stack([fixed_routing_token(task) for task in ROUTING_TASK_IDS])
    assert torch.equal(tokens.mean(1), torch.zeros(len(ROUTING_TASK_IDS)))
    assert torch.equal(tokens @ tokens.T, 128 * torch.eye(len(ROUTING_TASK_IDS)))

    runtime = SimpleNamespace(
        owners=tuple(range(38)),
        compiler=SimpleNamespace(event_slots=8, program_width=128),
        context=SimpleNamespace(device=torch.device("cpu")),
    )
    program = fixed_routing_program(runtime, ROUTING_TASK_IDS[0])
    assert program.p_lang.shape == (38, 128)
    assert program.p_scene.shape == (38, 128)
    assert program.p_process.shape == (8, 38, 128)
    assert program.sigma.shape == (8, 38, 128)
    assert torch.allclose(program.rho, torch.full((8,), 0.125))
    assert torch.allclose(program.tau[0], torch.tensor([0.0, 0.125]))
    assert torch.allclose(program.tau[-1], torch.tensor([0.875, 1.0]))


def test_routing_control_gate_balances_only_ten_gradient_tasks() -> None:
    costs = {
        "1": 197,
        "8": 195,
        "9": 97,
        "32": 199,
        "52": 195,
        "72": 195,
        "73": 99,
        "75": 198,
        "93": 108,
        "94": 105,
    }
    assignments = routing_task_assignments(
        worker_count=6, task_cost_seconds=costs
    )
    assert max(map(len, assignments)) == 2
    assert {task for row in assignments for task in row} == set(ROUTING_TASK_IDS)
    loads = [sum(costs[str(task)] for task in row) for row in assignments]
    assert max(loads) == 303


def test_routing_control_gate_accepts_the_recorded_flexible_training_world() -> None:
    runtime = SimpleNamespace(
        config={"profile": {"allowed_world_sizes": [1, 2, 3, 4, 5, 6]}}
    )
    run_contract = {
        "world_topology": [{"rank": rank} for rank in range(3)]
    }
    manifest = {"world_size": 3}
    assert _training_world_size(runtime, run_contract, manifest) == 3


def test_routing_control_critic_is_fit_only_and_never_a_deployment_input() -> None:
    root = Path(__file__).resolve().parents[2]
    critic_control = load_routing_control_config(
        root / "configs/pi05_ecp_routing_token_grouped_decoder_r3_v1.json"
    )
    assert (
        critic_control["model"]["output_primal_decoder"]
        == "owner_group_specific_linear_heads"
    )
    critic = critic_control["optimization"]["privileged_critic"]
    assert critic["kind"] == "fit_only_set_valued_paired_update_direction"
    assert critic["weight"] == 0.2
    assert critic["deployment_input"] is False
    assert critic["held_or_validation_reads"] is False
