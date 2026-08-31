from types import SimpleNamespace

import pytest
import torch

from ember.ecp.joint_program_primal import bank_set_shared_training
from ember.ecp.joint_program_primal.bank_set_shared_gradient_combiner import (
    balanced_condition_assignments,
    paired_conditions,
    run_paired_unit_gradient_step,
)
from ember.ecp.joint_program_primal.bank_set_shared_training import (
    ALL_INTERACTION_TASKS,
    GRADIENT_META_TASKS,
    GRADIENT_TARGET_TASKS,
    GRADIENT_TASKS,
    HELD_INTERACTION_TASKS,
    WRONG_TASK_RING,
    _apply_task_profile,
    _functional_arm_objective,
    _functional_task_loss,
    _shared_wrong_teacher,
    _validate_shared_config,
    _validate_task_cursors,
    balanced_task_assignments,
    task_cursor_counts,
    task_panel_a_visit,
)


def test_s2_schedule_is_role_task_and_world_size_invariant() -> None:
    even = paired_conditions(0, GRADIENT_TASKS)
    odd = paired_conditions(1, GRADIENT_TASKS)
    assert len(even) == len(set(even)) == 16
    assert len(odd) == len(set(odd)) == 16
    assert {task for task, _ in even} == set(GRADIENT_TASKS)
    assert not set(HELD_INTERACTION_TASKS).intersection(task for task, _ in even)
    assert {arm for _, arm in even} == {"correct_fit0", "wrong_fit0"}
    assert {arm for _, arm in odd} == {"correct_fit1", "wrong_fit0"}
    assert sum(task in GRADIENT_META_TASKS for task, _ in even) == 8
    assert sum(task in GRADIENT_TARGET_TASKS for task, _ in even) == 8

    for world_size in range(1, 7):
        costs = {condition: condition[0] % 37 + 1 for condition in even}
        assignments = balanced_condition_assignments(even, costs, world_size)
        assert {value for row in assignments for value in row} == set(even)
        assert sum(len(row) for row in assignments) == 16
        assert sum(1.0 / 16.0 for row in assignments for _ in row) == pytest.approx(1.0)


def test_s2_each_task_cursor_advances_once_per_paired_step() -> None:
    assert task_cursor_counts(0) == {task: 0 for task in GRADIENT_TASKS}
    assert task_cursor_counts(4) == {task: 4 for task in GRADIENT_TASKS}
    assert task_cursor_counts(8) == {task: 8 for task in GRADIENT_TASKS}


def test_s2_unit_gradient_combiner_preserves_scheduled_condition_mass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD((parameter,), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    runtime = SimpleNamespace(
        optimizer_steps=0,
        optimizer=optimizer,
        scheduler=scheduler,
        trainable_parameters=(parameter,),
        context=SimpleNamespace(
            world_size=1, rank=0, device=torch.device("cpu")
        ),
        config={
            "optimization": {
                "direct_functional": {
                    "condition_gradient_norm_epsilon": 1e-12,
                },
                "joint": {"optimizer": {"gradient_clip_norm": 10.0}},
            }
        },
    )

    monkeypatch.setattr(
        bank_set_shared_training,
        "_validate_task_cursors",
        lambda value: {
            task: int(value.optimizer_steps) for task in GRADIENT_TASKS
        },
    )
    monkeypatch.setattr(
        bank_set_shared_training, "_advance_task_cursors", lambda *_: None
    )
    monkeypatch.setattr(
        bank_set_shared_training,
        "_arm_spec",
        lambda *_args: SimpleNamespace(
            condition=SimpleNamespace(sampled_frames=1)
        ),
    )

    def condition_loss(_runtime, task, arm, **_kwargs):
        active = not (task == GRADIENT_TASKS[0] and arm == "wrong_fit0")
        parameter.grad = torch.tensor(
            2.0 if arm.startswith("correct") else (-3.0 if active else 0.0)
        )
        return {
            "task": task,
            "arm": arm,
            "gradient_active": active,
            "training_objective": 1.0,
            "generated_flow_loss": 1.0,
        }

    monkeypatch.setattr(
        bank_set_shared_training, "_functional_task_loss", condition_loss
    )
    row = run_paired_unit_gradient_step(runtime)
    assert parameter.item() == pytest.approx(-1.0 / 16.0)
    assert row["unit_gradient_active_condition_count"] == 15
    assert row["world_size_invariant_scheduled_condition_weight"] == 1.0 / 16.0
    assert sum(task["effective_unit_gradient_mass"] for task in row["tasks"]) == (
        pytest.approx(15.0 / 16.0)
    )


def test_s2_panel_a_visit_and_raw_unit_objectives_follow_task_cycle() -> None:
    assert [task_panel_a_visit(cursor, 16) for cursor in range(8)] == [
        0, 1, 2, 3, 4, 5, 6, 7
    ]
    assert task_panel_a_visit(15, 16) == 15
    assert task_panel_a_visit(16, 16) == 0

    correct = _functional_arm_objective(
        "correct_fit0", generated_loss=0.7, carrier_loss=1.0,
        correct_backward_mass=1.0, wrong_backward_mass=0.5,
    )
    active_wrong = _functional_arm_objective(
        "wrong_fit0", generated_loss=0.4, carrier_loss=1.0,
        correct_backward_mass=1.0, wrong_backward_mass=0.5,
    )
    inactive_wrong = _functional_arm_objective(
        "wrong_fit0", generated_loss=1.2, carrier_loss=1.0,
        correct_backward_mass=1.0, wrong_backward_mass=0.5,
    )
    assert (correct.kind, correct.value, correct.backward_mass) == (
        "raw_flow_loss", 0.7, 1.0
    )
    assert (active_wrong.value, active_wrong.backward_mass) == pytest.approx(
        (0.6, -0.5)
    )
    assert inactive_wrong.value == 0.0
    assert inactive_wrong.gradient_active is False


def test_s2_cursor_buffer_is_checkpoint_reconstructible() -> None:
    runtime = SimpleNamespace(
        optimizer_steps=8,
        writer_state=SimpleNamespace(
            task_arm_cursors=torch.tensor(
                [task_cursor_counts(8)[task] for task in GRADIENT_TASKS],
                dtype=torch.int64,
            )
        ),
    )
    assert _validate_task_cursors(runtime) == task_cursor_counts(8)
    runtime.writer_state.task_arm_cursors[0] += 1
    with pytest.raises(ValueError, match="disagree with macro"):
        _validate_task_cursors(runtime)


def _config() -> dict:
    profiles = {
        str(task): {
            "replay_frame_chunk_size": 16,
            "interaction_group_batch_size": 8,
            "functional_policy_microbatch_size": 2,
        }
        for task in ALL_INTERACTION_TASKS
    }
    profiles["1"].update(
        replay_frame_chunk_size=4, interaction_group_batch_size=16
    )
    profiles["93"].update(
        replay_frame_chunk_size=32, interaction_group_batch_size=4
    )
    profiles["94"].update(
        replay_frame_chunk_size=32, interaction_group_batch_size=4
    )
    return {
        "task_split": {
            "gradient_meta": list(GRADIENT_META_TASKS),
            "gradient_target": list(GRADIENT_TARGET_TASKS),
        },
        "shared_training": {
            "gradient_task_ids": list(GRADIENT_TASKS),
            "held_interaction_task_ids": list(HELD_INTERACTION_TASKS),
            "wrong_task_by_task": {
                str(task): wrong for task, wrong in WRONG_TASK_RING.items()
            },
            "optimizer_step_arms": [
                "alternating_correct_fit0_fit1", "wrong_fit0"
            ],
            "correct_view_schedule": "fit0_even_fit1_odd_optimizer_step",
            "task_profiles": profiles,
        },
        "model": {},
        "optimization": {},
    }


def test_s2_profiles_switch_b1_per_task_and_held_never_enters_train_ring() -> None:
    config = _config()
    _validate_shared_config(config)
    operator = SimpleNamespace(covariance_frame_chunk=-1)
    runtime = SimpleNamespace(
        config=config,
        compiler=SimpleNamespace(bank_operator=operator),
    )
    assert _apply_task_profile(runtime, 1)["replay_frame_chunk_size"] == 4
    assert operator.covariance_frame_chunk == 4
    assert _apply_task_profile(runtime, 8)["interaction_group_batch_size"] == 8
    assert operator.covariance_frame_chunk == 16
    assert _apply_task_profile(runtime, 94)["interaction_group_batch_size"] == 4
    assert operator.covariance_frame_chunk == 32

    config["shared_training"]["wrong_task_by_task"]["8"] = 1
    with pytest.raises(ValueError, match="task contract changed"):
        _validate_shared_config(config)


def test_s2_adapts_sealed_wrong_teacher_settings_without_mutating_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = {"updates": 1, "learning_rate": 0.02, "panel_a_visit": 0}
    runtime = SimpleNamespace(
        config={
            "optimization": {"joint": {}},
            "evaluation": {"target_cache_wrong_free_delta_teacher": settings},
        }
    )
    observed = {}

    def teacher(runtime, task, arm, base_output):
        observed.update(runtime.config["optimization"]["joint"])
        return "teacher", {"task": task}

    monkeypatch.setattr(bank_set_shared_training, "_wrong_teacher", teacher)
    assert _shared_wrong_teacher(runtime, 8, object(), object()) == (
        "teacher",
        {"task": 8},
    )
    assert observed == {"wrong_free_delta_teacher": settings}
    assert runtime.config["optimization"]["joint"] == {}


def test_s2_functional_task_uses_two_passes_and_structural_zero_hinge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    calls: list[bool] = []
    loss = {"value": 0.7}
    condition = SimpleNamespace(video_demo=80)
    panel = SimpleNamespace(
        policy_rng_seed=123,
        flow_loss=1.0,
        action_demos=tuple(range(16)),
        action_frames=tuple(range(16)),
    )
    runtime = SimpleNamespace(
        config={
            "data": {"panel_visits": 16},
            "optimization": {"direct_functional": {
                "correct_backward_mass": 1.0, "wrong_backward_mass": 0.5,
            }},
        },
        context=SimpleNamespace(device=torch.device("cpu")),
    )

    monkeypatch.setattr(
        bank_set_shared_training,
        "_arm_spec",
        lambda _runtime, task, name: SimpleNamespace(
            task=task, role="meta_fit", name=name, bank_task=9,
            condition=condition, receives_gradient=True,
        ),
    )
    monkeypatch.setattr(
        bank_set_shared_training,
        "_prepare_arm",
        lambda *_args: SimpleNamespace(
            bank=SimpleNamespace(condition_metrics={"ok": 1})
        ),
    )

    def interaction_output(*_args):
        calls.append(torch.is_grad_enabled())
        return parameter * 1.0

    def panel_batch(_runtime, *, task_id, panel_name, visit_index):
        assert (task_id, panel_name, visit_index) == (8, "a", 9)
        return {"batch": True}, panel

    monkeypatch.setattr(
        bank_set_shared_training, "_shared_interaction_output", interaction_output
    )
    monkeypatch.setattr(
        bank_set_shared_training, "_complete", lambda _runtime, output: {"leaf": output}
    )
    monkeypatch.setattr(bank_set_shared_training, "functional_panel_batch", panel_batch)
    monkeypatch.setattr(
        bank_set_shared_training,
        "functional_loss_derivative",
        lambda *_args, **_kwargs: (loss["value"], {"leaf": torch.tensor(3.0)}),
    )
    monkeypatch.setattr(
        bank_set_shared_training, "_apply_task_profile", lambda *_args: {"ok": 1}
    )
    monkeypatch.setattr(bank_set_shared_training, "_clear_panel_cache", lambda *_: None)

    row = _functional_task_loss(
        runtime, 8, "correct_fit0", task_cursor=9, task_weight=1.0 / 6.0
    )
    assert calls == [False, True]
    assert parameter.grad == pytest.approx(torch.tensor(0.5))
    assert row["panel_visit"] == 9
    assert row["backward_mass"] == 1.0

    calls.clear()
    parameter.grad = None
    loss["value"] = 1.2
    row = _functional_task_loss(
        runtime, 8, "wrong_fit0", task_cursor=9, task_weight=1.0 / 6.0
    )
    assert calls == [False, True]
    assert parameter.grad is not None and float(parameter.grad) == 0.0
    assert row["gradient_active"] is False
    assert row["backward_mass"] == -0.5
