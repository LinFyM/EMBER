from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.stage1_calibration import (
    action_guided_factor_perturbation,
    action_guided_outcome_leaf_gradients,
)
from ember.ecp.stage1_calibration_contract import (
    STRUCTURED_CALIBRATION_FILE,
    STRUCTURED_CALIBRATION_SCHEMA,
    build_calibration_assignments,
    validate_structured_calibration,
)
from ember.pi05_source_checkpoint import write_json_atomic


def _tiny_surface() -> tuple[tuple[TargetOwner, ...], dict[str, torch.Tensor]]:
    owner = TargetOwner(
        index=0,
        target_name="tiny",
        family=TargetFamily.Q,
        layer=0,
        in_features=5,
        out_features=6,
    )
    state = {
        "tiny.lora_A.default.weight": torch.randn(2, 5),
        "tiny.lora_B.default.weight": torch.randn(6, 2),
    }
    return (owner,), state


def test_structured_calibration_assigns_each_fit_mapping_once() -> None:
    tasks = tuple(
        SimpleNamespace(
            ordinal=ordinal,
            fold_role="fit" if ordinal < 90 else "held_transform_only",
            suite="libero_90" if ordinal < 71 else "libero_spatial",
            episode_lengths=(80 + ordinal % 7, 90 + ordinal % 5),
        )
        for ordinal in range(95)
    )
    assignments = build_calibration_assignments(
        tasks, world_size=6, frame_stride=5, task_count=90
    )
    assert sorted(ordinal for values in assignments for ordinal in values) == list(
        range(90)
    )
    assert all(values for values in assignments)


def test_structured_calibration_uses_owner_local_factor_directions() -> None:
    owners, state = _tiny_surface()
    gradients = {name: torch.ones_like(value) for name, value in state.items()}
    perturbation = action_guided_factor_perturbation(
        state, gradients, owners, sigma=0.05, seed=7
    )
    assert perturbation.epsilon.shape == (1, 1)
    assert perturbation.active_owners == 1
    assert all(
        not torch.equal(perturbation.plus_state[name], perturbation.minus_state[name])
        for name in state
    )
    leaf = action_guided_outcome_leaf_gradients(
        perturbation, owners, torch.tensor([[2.0]]), weight=1.0
    )
    directional_derivative = sum(
        (leaf[name].float() * perturbation.directions[name].float()).sum()
        for name in leaf
    )
    torch.testing.assert_close(directional_derivative, torch.tensor(-2.0))


def test_materialization_requires_completed_fit90_calibration(
    tmp_path: Path,
) -> None:
    write_json_atomic(
        tmp_path / STRUCTURED_CALIBRATION_FILE,
        {
            "schema_version": STRUCTURED_CALIBRATION_SCHEMA,
            "status": "complete_fit90_structured_calibration",
            "mode": "formal",
            "applied_after_task_visits": 540,
            "task_count": 90,
            "task_weight": "equal",
            "global_16d_estimator": False,
            "optimizer_updates": 1,
            "assignments": [list(range(90))],
            "information_wall": {
                "held5_reward_reads": 0,
                "validation_reward_reads": 0,
                "test_reward_reads": 0,
            },
            "tasks": [{"task_ordinal": ordinal} for ordinal in range(90)],
        },
    )
    assert (
        validate_structured_calibration(tmp_path, checkpoint_task_visits=540)[
            "task_count"
        ]
        == 90
    )
    with pytest.raises(ValueError, match="structured calibration"):
        validate_structured_calibration(tmp_path, checkpoint_task_visits=539)
