from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import ember.pi05_eval.anchored_reconciliation_gate as gate_module
from ember.expert_manifold.v6_prior_checkpoint import V6_PRIOR_CHECKPOINT_SCHEMA
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.anchored_reconciliation_gate import (
    anchored_reconciliation_decision_evidence,
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
        gate_module,
        "_formal_panel_index",
        lambda result: result["indexed"],
    )
    monkeypatch.setattr(
        gate_module,
        "_writer_family",
        lambda adapter: (adapter["family"], {}),
    )
    monkeypatch.setattr(
        gate_module,
        "_method_macro",
        lambda result, **_kwargs: result["adapter"]["writer_asset"][
            "method_macro"
        ],
    )

    def paired(reference: dict, candidate: dict, **kwargs: object) -> None:
        assert set(reference) == set(candidate)
        assert kwargs == {"require_same_actual_video": True}

    monkeypatch.setattr(gate_module, "_assert_row_pairing", paired)


def _arguments(
    macro0_root: Path,
    macro10_root: Path,
    checkpoint: Path,
) -> dict:
    return {
        "macro0_root": macro0_root,
        "macro10_root": macro10_root,
        "resume_checkpoint": checkpoint,
        "expected_macro0_commit": "baseline-commit",
        "expected_macro0_correct": 2,
        "expected_macro0_breadth": 2,
        "expected_current_commit": "current-commit",
        "decision_gates": {
            "macro10_support_correct_min": 3,
            "macro10_support_lost_to_macro0_max": 0,
            "macro10_support_breadth_min": 2,
        },
    }


def test_gate_recomputes_paired_support(
    tmp_path: Path,
    isolated_analysis: None,
) -> None:
    macro0_root = tmp_path / "macro0"
    macro10_root = tmp_path / "macro10"
    checkpoint = tmp_path / "checkpoints/macro_00000010"
    baseline_success = {("suite-a", 1, 0), ("suite-b", 2, 0)}
    candidate_success = baseline_success | {("suite-a", 1, 1)}
    roots = {
        str(macro0_root.resolve()): _result(
            family="v6_condition_residual_v2",
            macro=0,
            commit="baseline-commit",
            rows=_rows(baseline_success),
        ),
        str(macro10_root.resolve()): _result(
            family="v6_anchored_reconciliation_v3",
            macro=10,
            commit="current-commit",
            rows=_rows(candidate_success),
            checkpoint=checkpoint,
        ),
    }
    evidence = anchored_reconciliation_decision_evidence(
        roots,
        **_arguments(macro0_root, macro10_root, checkpoint),
    )
    assert evidence["passed"] is True
    assert evidence["macro10"]["correct"] == 3
    assert evidence["transition"]["gained"] == 1
    assert evidence["transition"]["lost"] == 0


def test_gate_rejects_identity_and_reports_scientific_nonpass(
    tmp_path: Path,
    isolated_analysis: None,
) -> None:
    macro0_root = tmp_path / "macro0"
    macro10_root = tmp_path / "macro10"
    checkpoint = tmp_path / "checkpoints/macro_00000010"
    successes = {("suite-a", 1, 0), ("suite-b", 2, 0)}
    baseline = _result(
        family="v6_condition_residual_v2",
        macro=0,
        commit="baseline-commit",
        rows=_rows(successes),
    )
    candidate = _result(
        family="v6_anchored_reconciliation_v3",
        macro=10,
        commit="wrong-commit",
        rows=_rows(successes),
        checkpoint=checkpoint,
    )
    roots = {
        str(macro0_root.resolve()): baseline,
        str(macro10_root.resolve()): candidate,
    }
    arguments = _arguments(macro0_root, macro10_root, checkpoint)
    with pytest.raises(Pi05EvaluationError, match="checkpoint identity"):
        anchored_reconciliation_decision_evidence(roots, **arguments)

    corrected = deepcopy(candidate)
    corrected["paired_control"]["git"]["commit"] = "current-commit"
    roots[str(macro10_root.resolve())] = corrected
    evidence = anchored_reconciliation_decision_evidence(roots, **arguments)
    assert evidence["passed"] is False
    assert evidence["checks"] == {
        "correct": False,
        "lost_to_macro0": True,
        "breadth": True,
    }
