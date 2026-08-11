from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import ember.pi05_eval.reward_credit_gate as gate_module
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.reward_credit_gate import (
    V6_PRIOR_CHECKPOINT_SCHEMA,
    load_reward_credit_control_trigger_evidence,
    reward_credit_decision_evidence,
    reward_credit_six_arm_evidence,
)


def _rows(successes: set[tuple[str, int, int]]) -> dict:
    return {
        (suite, task, state): {
            "suite": suite,
            "task_id": task,
            "init_state_id": state,
            "split_role": "validation",
            "language": f"task-{suite}-{task}",
            "success": (suite, task, state) in successes,
        }
        for suite, task in (("suite-a", 1), ("suite-b", 2))
        for state in range(2)
    }


def _result(
    *,
    family: str,
    macro: int,
    commit: str,
    rows: dict,
    checkpoint: Path | None = None,
) -> dict:
    asset = {
        "method_macro": macro,
        "kind": (
            "historical_v6_macro400_load_only"
            if macro == 0
            else "v6_condition_program_residual_checkpoint"
        ),
    }
    if checkpoint is not None:
        asset.update(
            {
                "checkpoint": str(checkpoint),
                "training_mode": "formal",
                "manifest": {"schema": V6_PRIOR_CHECKPOINT_SCHEMA},
            }
        )
    return {
        "adapter": {
            "family": family,
            "video_condition": "correct",
            "writer_asset": asset,
        },
        "paired_control": {"git": {"commit": commit}},
        "indexed": rows,
    }


@pytest.fixture
def isolated_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gate_module, "_formal_panel_index", lambda result: result["indexed"]
    )
    monkeypatch.setattr(
        gate_module,
        "_writer_family",
        lambda adapter: (adapter["family"], {}),
    )
    monkeypatch.setattr(
        gate_module,
        "_method_macro",
        lambda result, **_kwargs: result["adapter"]["writer_asset"]["method_macro"],
    )

    def paired(reference: dict, candidate: dict, **kwargs: object) -> None:
        assert set(reference) == set(candidate)
        assert kwargs == {"require_same_actual_video": True}

    monkeypatch.setattr(gate_module, "_assert_row_pairing", paired)


def _arguments(
    macro0_root: Path,
    macro1_root: Path,
    checkpoint: Path,
) -> dict:
    return {
        "macro0_root": macro0_root,
        "macro1_root": macro1_root,
        "resume_checkpoint": checkpoint,
        "expected_macro0_commit": "baseline-commit",
        "expected_macro0_correct": 2,
        "expected_macro0_breadth": 2,
        "expected_current_commit": "current-commit",
        "decision_gates": {
            "macro1_support_correct_min": 3,
            "macro1_support_lost_to_macro0_max": 0,
            "macro1_support_breadth_min": 2,
            "macro1_support_gained_must_exceed_lost": True,
            "first_full_six_arm_correct_min": 4,
            "goal_full_six_arm_correct_min": 4,
            "goal_correct_strictly_exceeds_negative_controls": True,
            "goal_same_task_other_correct_ratio_min": 0.9,
        },
    }


def test_gate_recomputes_paired_reward_credit_support(
    tmp_path: Path,
    isolated_analysis: None,
) -> None:
    macro0_root = tmp_path / "macro0"
    macro1_root = tmp_path / "macro1"
    checkpoint = tmp_path / "checkpoints/macro_00000001"
    baseline_success = {("suite-a", 1, 0), ("suite-b", 2, 0)}
    candidate_success = baseline_success | {("suite-a", 1, 1)}
    roots = {
        str(macro0_root.resolve()): _result(
            family="v6_condition_residual_v2",
            macro=0,
            commit="baseline-commit",
            rows=_rows(baseline_success),
        ),
        str(macro1_root.resolve()): _result(
            family="v6_reward_credit_program_v1",
            macro=1,
            commit="current-commit",
            rows=_rows(candidate_success),
            checkpoint=checkpoint,
        ),
    }
    evidence = reward_credit_decision_evidence(
        roots, **_arguments(macro0_root, macro1_root, checkpoint)
    )
    assert evidence["passed"] is True
    assert evidence["macro1"]["correct"] == 3
    assert evidence["transition"]["gained"] == 1
    assert evidence["transition"]["lost"] == 0
    assert evidence["six_arm_required"] is False
    assert all(evidence["checks"].values())


def test_gate_rejects_checkpoint_identity_and_reports_scientific_nonpass(
    tmp_path: Path,
    isolated_analysis: None,
) -> None:
    macro0_root = tmp_path / "macro0"
    macro1_root = tmp_path / "macro1"
    checkpoint = tmp_path / "checkpoints/macro_00000001"
    successes = {("suite-a", 1, 0), ("suite-b", 2, 0)}
    baseline = _result(
        family="v6_condition_residual_v2",
        macro=0,
        commit="baseline-commit",
        rows=_rows(successes),
    )
    candidate = _result(
        family="v6_reward_credit_program_v1",
        macro=1,
        commit="wrong-commit",
        rows=_rows(successes),
        checkpoint=checkpoint,
    )
    roots = {
        str(macro0_root.resolve()): baseline,
        str(macro1_root.resolve()): candidate,
    }
    arguments = _arguments(macro0_root, macro1_root, checkpoint)
    with pytest.raises(Pi05EvaluationError, match="checkpoint identity"):
        reward_credit_decision_evidence(roots, **arguments)

    corrected = deepcopy(candidate)
    corrected["paired_control"]["git"]["commit"] = "current-commit"
    roots[str(macro1_root.resolve())] = corrected
    evidence = reward_credit_decision_evidence(roots, **arguments)
    assert evidence["passed"] is False
    assert evidence["checks"] == {
        "correct": False,
        "lost_to_macro0": True,
        "breadth": True,
        "gained_exceeds_lost": False,
    }


def test_six_arm_goal_consumes_registered_roots_and_causality_thresholds(
    tmp_path: Path,
) -> None:
    conditions = (
        "correct",
        "same_task_other",
        "cross_suite_wrong",
        "shuffled",
        "reversed",
        "no_video",
    )
    roots = {condition: tmp_path / condition for condition in conditions}
    analysis = {
        "method_family": "v6_reward_credit_program_v1",
        "winner": {"method_macro": 1},
        "roots": [
            {"condition": condition, "root": str(roots[condition])}
            for condition in conditions
        ],
        "arms": {
            condition: {
                "overall": {
                    "successes": {
                        "correct": 151,
                        "same_task_other": 140,
                        "cross_suite_wrong": 120,
                        "shuffled": 110,
                        "reversed": 100,
                        "no_video": 90,
                    }[condition]
                }
            }
            for condition in conditions
        },
    }
    decision = {
        "macro1_registered_root": str(roots["correct"]),
        "macro1_control_registered_roots": {
            condition: str(roots[condition]) for condition in conditions[1:]
        },
    }
    gates = {
        "goal_full_six_arm_correct_min": 151,
        "goal_correct_strictly_exceeds_negative_controls": True,
        "goal_same_task_other_correct_ratio_min": 0.9,
    }
    evidence = reward_credit_six_arm_evidence(
        analysis,
        macro=1,
        decision_evaluation=decision,
        decision_gates=gates,
    )
    assert evidence["goal_passed"] is True
    assert evidence["same_task_other_to_correct_ratio"] == pytest.approx(140 / 151)
    analysis["arms"]["reversed"]["overall"]["successes"] = 151
    failed = reward_credit_six_arm_evidence(
        analysis,
        macro=1,
        decision_evaluation=decision,
        decision_gates=gates,
    )
    assert failed["goal_passed"] is False
    assert failed["negative_control_checks"]["reversed"] is False


def _control_trigger_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    macro1_correct: int,
    macro2_correct: int,
) -> tuple[Path, Path, dict, dict]:
    training_root = tmp_path / "training"
    checkpoint1 = training_root / "checkpoints/macro_00000001"
    checkpoint2 = training_root / "checkpoints/macro_00000002"
    roots = {1: tmp_path / "correct1", 2: tmp_path / "correct2"}
    all_successes = list(_rows(set()))
    results = {
        str(roots[1].resolve()): _result(
            family="v6_reward_credit_program_v1",
            macro=1,
            commit="current-commit",
            rows=_rows(set(all_successes[:macro1_correct])),
            checkpoint=checkpoint1,
        ),
        str(roots[2].resolve()): _result(
            family="v6_reward_credit_program_v1",
            macro=2,
            commit="current-commit",
            rows=_rows(set(all_successes[:macro2_correct])),
            checkpoint=checkpoint2,
        ),
    }

    def validated(requested: tuple[Path, ...]) -> dict:
        return {str(path.resolve()): results[str(path.resolve())] for path in requested}

    monkeypatch.setattr(gate_module, "_validated_roots", validated)
    decision = {
        "macro1_registered_root": str(roots[1]),
        "macro2_registered_root": str(roots[2]),
    }
    gates = {
        "first_full_six_arm_correct_min": 3,
        "goal_full_six_arm_correct_min": 4,
    }
    return training_root, checkpoint2, decision, gates


def test_control_trigger_uses_correct_threshold_not_support_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_analysis: None,
) -> None:
    training_root, _, decision, gates = _control_trigger_fixture(
        tmp_path,
        monkeypatch,
        macro1_correct=3,
        macro2_correct=3,
    )
    checkpoint1 = training_root / "checkpoints/macro_00000001"
    evidence = load_reward_credit_control_trigger_evidence(
        training_root=training_root,
        current_checkpoint=checkpoint1,
        macro=1,
        expected_commit="current-commit",
        decision_evaluation=decision,
        decision_gates=gates,
    )
    assert evidence["correct"] == 3
    assert evidence["support_gate_independent"] is True


def test_control_trigger_rejects_macro1_below_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_analysis: None,
) -> None:
    training_root, _, decision, gates = _control_trigger_fixture(
        tmp_path,
        monkeypatch,
        macro1_correct=2,
        macro2_correct=3,
    )
    with pytest.raises(Pi05EvaluationError, match="not authorized"):
        load_reward_credit_control_trigger_evidence(
            training_root=training_root,
            current_checkpoint=training_root / "checkpoints/macro_00000001",
            macro=1,
            expected_commit="current-commit",
            decision_evaluation=decision,
            decision_gates=gates,
        )


@pytest.mark.parametrize(
    ("macro1_correct", "macro2_correct", "allowed", "reason"),
    (
        (2, 3, True, "first_checkpoint_at_or_above_control_threshold"),
        (3, 3, False, None),
        (3, 4, True, "goal_candidate_requires_same_checkpoint_controls"),
    ),
)
def test_macro2_controls_run_only_for_a_new_first_trigger_or_goal_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_analysis: None,
    macro1_correct: int,
    macro2_correct: int,
    allowed: bool,
    reason: str | None,
) -> None:
    training_root, checkpoint2, decision, gates = _control_trigger_fixture(
        tmp_path,
        monkeypatch,
        macro1_correct=macro1_correct,
        macro2_correct=macro2_correct,
    )
    arguments = {
        "training_root": training_root,
        "current_checkpoint": checkpoint2,
        "macro": 2,
        "expected_commit": "current-commit",
        "decision_evaluation": decision,
        "decision_gates": gates,
    }
    if not allowed:
        with pytest.raises(Pi05EvaluationError, match="not authorized"):
            load_reward_credit_control_trigger_evidence(**arguments)
        return
    evidence = load_reward_credit_control_trigger_evidence(**arguments)
    assert evidence["reason"] == reason
    assert evidence["previous_correct"] == macro1_correct
