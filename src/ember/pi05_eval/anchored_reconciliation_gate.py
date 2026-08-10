"""Strict paired support gate for anchored-reconciliation continuation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.expert_manifold.v6_prior_checkpoint import V6_PRIOR_CHECKPOINT_SCHEMA
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.analysis import (
    _assert_row_pairing,
    _formal_panel_index,
    _method_macro,
    _validated_roots,
    _writer_family,
)
from ember.pi05_eval.paired_metrics import (
    EpisodeKey,
    paired_transition_summary,
    summarize_panel,
)


DECISION_EVIDENCE_SCHEMA = (
    "ember_pi05_v6_anchored_reconciliation_decision_evidence_v1"
)


def _fail(message: str) -> None:
    raise Pi05EvaluationError(message)


def _baseline_record(
    result: Mapping[str, Any],
    *,
    root: Path,
    expected_commit: str,
    expected_correct: int,
    expected_breadth: int,
) -> tuple[dict[EpisodeKey, Mapping[str, Any]], dict[str, Any]]:
    rows = _formal_panel_index(result)
    summary = summarize_panel(list(rows.values()))
    correct = int(summary["overall"]["successes"])
    breadth = int(summary["nonzero_task_breadth"])
    valid = (
        _writer_family(result["adapter"])[0] == "v6_condition_residual_v2"
        and result["adapter"].get("video_condition") == "correct"
        and _method_macro(
            result,
            allowed_macros=(0,),
            context="anchored reconciliation macro0 reference",
        )
        == 0
        and result.get("paired_control", {}).get("git", {}).get("commit")
        == expected_commit
        and correct == expected_correct
        and breadth == expected_breadth
    )
    if not valid:
        _fail("anchored reconciliation macro0 reference changed")
    return rows, {
        "root": str(root.resolve()),
        "git_commit": expected_commit,
        "correct": correct,
        "breadth": breadth,
    }


def _candidate_rows(
    result: Mapping[str, Any],
    *,
    resume_checkpoint: Path,
    expected_commit: str,
) -> dict[EpisodeKey, Mapping[str, Any]]:
    rows = _formal_panel_index(result)
    asset = result["adapter"].get("writer_asset", {})
    manifest = asset.get("manifest", {})
    try:
        checkpoint_matches = (
            Path(str(asset.get("checkpoint", ""))).resolve()
            == resume_checkpoint.resolve()
        )
    except (OSError, RuntimeError):
        checkpoint_matches = False
    valid = (
        _writer_family(result["adapter"])[0]
        == "v6_anchored_reconciliation_v3"
        and result["adapter"].get("video_condition") == "correct"
        and _method_macro(
            result,
            allowed_macros=(10,),
            context="anchored reconciliation macro10 support gate",
        )
        == 10
        and result.get("paired_control", {}).get("git", {}).get("commit")
        == expected_commit
        and checkpoint_matches
        and asset.get("training_mode") == "formal"
        and isinstance(manifest, Mapping)
        and manifest.get("schema") == V6_PRIOR_CHECKPOINT_SCHEMA
    )
    if not valid:
        _fail("anchored reconciliation macro10 checkpoint identity changed")
    return rows


def _support_record(
    baseline_rows: Mapping[EpisodeKey, Mapping[str, Any]],
    candidate_rows: Mapping[EpisodeKey, Mapping[str, Any]],
    *,
    root: Path,
    resume_checkpoint: Path,
    expected_commit: str,
    decision_gates: Mapping[str, Any],
) -> dict[str, Any]:
    _assert_row_pairing(
        baseline_rows,
        candidate_rows,
        require_same_actual_video=True,
    )
    candidate_values = list(candidate_rows.values())
    summary = summarize_panel(candidate_values)
    transition = paired_transition_summary(
        list(baseline_rows.values()), candidate_values
    )["overall"]
    correct = int(summary["overall"]["successes"])
    breadth = int(summary["nonzero_task_breadth"])
    checks = {
        "correct": correct
        >= int(decision_gates["macro10_support_correct_min"]),
        "lost_to_macro0": int(transition["lost"])
        <= int(decision_gates["macro10_support_lost_to_macro0_max"]),
        "breadth": breadth
        >= int(decision_gates["macro10_support_breadth_min"]),
    }
    return {
        "macro10": {
            "root": str(root.resolve()),
            "git_commit": expected_commit,
            "checkpoint": str(resume_checkpoint.resolve()),
            "correct": correct,
            "breadth": breadth,
        },
        "transition": {
            name: transition[name]
            for name in (
                "retained_success",
                "gained",
                "lost",
                "retained_failure",
                "net",
                "churn",
            )
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def anchored_reconciliation_decision_evidence(
    results_by_root: Mapping[str, Mapping[str, Any]],
    *,
    macro0_root: Path,
    macro10_root: Path | None,
    resume_checkpoint: Path | None,
    expected_macro0_commit: str,
    expected_macro0_correct: int,
    expected_macro0_breadth: int,
    expected_current_commit: str | None,
    decision_gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the pre-registered macro10 support decision from strict rows."""

    macro0_key = str(macro0_root.resolve())
    expected_keys = {macro0_key}
    if macro10_root is not None:
        expected_keys.add(str(macro10_root.resolve()))
    if set(results_by_root) != expected_keys:
        _fail("anchored reconciliation decision roots changed")
    baseline_rows, baseline = _baseline_record(
        results_by_root[macro0_key],
        root=macro0_root,
        expected_commit=expected_macro0_commit,
        expected_correct=expected_macro0_correct,
        expected_breadth=expected_macro0_breadth,
    )
    evidence: dict[str, Any] = {
        "schema_version": DECISION_EVIDENCE_SCHEMA,
        "macro0": baseline,
        "macro10": None,
        "transition": None,
        "checks": None,
        "passed": None,
    }
    if macro10_root is None:
        return evidence
    if resume_checkpoint is None or expected_current_commit is None:
        _fail("anchored reconciliation macro10 decision lacks resume identity")
    candidate_rows = _candidate_rows(
        results_by_root[str(macro10_root.resolve())],
        resume_checkpoint=resume_checkpoint,
        expected_commit=expected_current_commit,
    )
    evidence.update(
        _support_record(
            baseline_rows,
            candidate_rows,
            root=macro10_root,
            resume_checkpoint=resume_checkpoint,
            expected_commit=expected_current_commit,
            decision_gates=decision_gates,
        )
    )
    return evidence


def load_anchored_reconciliation_decision_evidence(
    *,
    macro0_root: Path,
    macro10_root: Path | None,
    resume_checkpoint: Path | None,
    expected_macro0_commit: str,
    expected_macro0_correct: int,
    expected_macro0_breadth: int,
    expected_current_commit: str | None,
    decision_gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Reaggregate immutable roots before applying the RLS continuation gate."""

    roots = (macro0_root,) if macro10_root is None else (macro0_root, macro10_root)
    return anchored_reconciliation_decision_evidence(
        _validated_roots(roots),
        macro0_root=macro0_root,
        macro10_root=macro10_root,
        resume_checkpoint=resume_checkpoint,
        expected_macro0_commit=expected_macro0_commit,
        expected_macro0_correct=expected_macro0_correct,
        expected_macro0_breadth=expected_macro0_breadth,
        expected_current_commit=expected_current_commit,
        decision_gates=decision_gates,
    )
