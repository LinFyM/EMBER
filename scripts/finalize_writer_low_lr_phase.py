#!/usr/bin/env python3
"""Finalize the registered low-LR Writer phase without using control outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from ember.pi05_eval_results import AGGREGATE_SCHEMA
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record, inspect_writer_checkpoint, method_metadata
from ember.writer.video_controls import DIAGNOSTIC_DECLARATION, METHOD_FREEZE_DECLARATION


PHASE_SCHEMA = "ember_writer_low_lr_phase_history_v1"
SELECTION_SCHEMA = "ember_writer_low_lr_selection_v1"
REFERENCE_STEP = 1000
REFERENCE_SUCCESSES = 117
VALIDATION_TASKS = (3, 6, 11, 16, 23, 26, 31, 39)
VIDEO_SEED = 20260911


def choose_phase_checkpoint(
    phase: Mapping[str, Any], other_successes: Mapping[int, int] | None = None, *, owner_stop: bool = False,
) -> dict[str, Any]:
    """Apply the frozen qualification and tie-break rules to a stopped phase."""
    if phase.get("schema_version") != PHASE_SCHEMA or phase.get("selection_reference") != {
        "step": REFERENCE_STEP, "successes": REFERENCE_SUCCESSES,
    }:
        raise ValueError("unexpected low-LR phase history or selection reference")
    history = phase.get("history")
    decision = phase.get("decision")
    if (not isinstance(history, list) or not history or not isinstance(decision, dict)
            or (not decision.get("stop") and not owner_stop)):
        raise ValueError("selection requires a completed stopped low-LR phase")
    if any(row.get("episodes") != 400 or row.get("complete") is not True for row in history):
        raise ValueError("selection accepts only complete Validation400 phase nodes")
    observed_best = max(int(row["successes"]) for row in history)
    tied = sorted(int(row["step"]) for row in history if int(row["successes"]) == observed_best)
    if (decision.get("best_successes") != observed_best
            or sorted(map(int, decision.get("tied_best_steps", ()))) != tied):
        raise ValueError("phase decision disagrees with its complete history")
    common = {
        "phase_best_successes": observed_best,
        "phase_tied_best_steps": tied,
        "phase_stop_reason": decision.get("reason") if decision.get("stop") else "owner_requested_stop",
    }
    if observed_best <= REFERENCE_SUCCESSES:
        return {
            **common,
            "status": "selected",
            "outcome": "retain_original_n1000",
            "selected_study": "original",
            "selected_macro": REFERENCE_STEP,
            "correct_successes": REFERENCE_SUCCESSES,
            "other_tiebreak_required": False,
            "other_successes": {},
            "controls_required": [],
        }
    if len(tied) == 1:
        selected = tied[0]
        return {
            **common,
            "status": "selected",
            "outcome": "select_low_lr_phase",
            "selected_study": "phase",
            "selected_macro": selected,
            "correct_successes": observed_best,
            "other_tiebreak_required": False,
            "other_successes": {},
            "controls_required": ["same_task_other", "cross_suite_wrong"],
        }
    scores = {int(step): int(score) for step, score in (other_successes or {}).items()}
    missing = [step for step in tied if step not in scores]
    if missing:
        return {
            **common,
            "status": "awaiting_tied_other400",
            "outcome": None,
            "required_other_steps": missing,
            "other_tiebreak_required": True,
            "other_successes": {str(step): scores[step] for step in tied if step in scores},
        }
    best_other = max(scores[step] for step in tied)
    selected = min(step for step in tied if scores[step] == best_other)
    return {
        **common,
        "status": "selected",
        "outcome": "select_low_lr_phase",
        "selected_study": "phase",
        "selected_macro": selected,
        "correct_successes": observed_best,
        "other_tiebreak_required": True,
        "other_successes": {str(step): scores[step] for step in tied},
        "selected_other_successes": best_other,
        "controls_required": ["same_task_other", "cross_suite_wrong"],
    }


def _complete_other_successes(study: Path, step: int) -> int | None:
    evaluation = study / "evaluation" / f"writer_{step:08d}_other"
    manifest_path = study / "materialized" / f"writer_{step:08d}_other" / "manifest.json"
    required = [evaluation / "results.json", evaluation / "run_contract.json",
                evaluation / "launcher_completion.json", manifest_path]
    if not any(path.exists() for path in required):
        return None
    if not all(path.is_file() for path in required):
        raise ValueError(f"incomplete same-task-other400 artifacts for step {step}")
    results, contract, completion, manifest = map(read_json, required)
    if (results.get("schema_version") != AGGREGATE_SCHEMA
            or contract.get("role") != "validation" or contract.get("mode") != "formal"
            or len({results.get("contract_reference"), contract.get("contract_reference"),
                    completion.get("contract_reference")}) != 1
            or not completion.get("return_codes")
            or any(code != 0 for code in completion["return_codes"].values())):
        raise ValueError(f"invalid same-task-other400 completion for step {step}")
    rows = results.get("rows", ())
    keys = {(row.get("horizon_writer_lora", {}).get("global_task_id"), row.get("init_state_id")) for row in rows}
    expected = {(task, state) for task in VALIDATION_TASKS for state in range(50)}
    successes = sum(bool(row.get("success")) for row in rows)
    if (len(rows) != 400 or keys != expected
            or results.get("overall", {}).get("episodes") != 400
            or results.get("overall", {}).get("successes") != successes
            or completion.get("queue", {}).get("completed_rows") != 400
            or completion.get("queue", {}).get("successes") != successes):
        raise ValueError(f"same-task-other panel is not a complete Validation400 at step {step}")
    selection = manifest.get("selection", {})
    if (manifest.get("arm") != "same_task_other"
            or manifest.get("writer_checkpoint", {}).get("macro") != step
            or selection.get("evaluation_role") != "validation"
            or selection.get("task_ids") != list(VALIDATION_TASKS)
            or selection.get("init_state_ids") != list(range(50))
            or selection.get("seed") != VIDEO_SEED):
        raise ValueError(f"same-task-other manifest changed at step {step}")
    return successes


def _write_selected_phase_artifacts(study: Path, selected_macro: int) -> tuple[Path, Path, dict[str, Any]]:
    checkpoint = study / "training" / "writer" / "checkpoints" / f"macro_{selected_macro:08d}"
    run, checkpoint_record = inspect_writer_checkpoint(checkpoint)
    correct_manifest = study / "materialized" / f"writer_{selected_macro:08d}_correct" / "manifest.json"
    if not correct_manifest.is_file():
        raise ValueError("selected low-LR checkpoint is missing its complete correct400 manifest")
    correct = read_json(correct_manifest)
    if (correct.get("arm") != "correct"
            or correct.get("writer_checkpoint") != checkpoint_record
            or correct.get("selection", {}).get("evaluation_role") != "validation"
            or correct.get("selection", {}).get("task_ids") != list(VALIDATION_TASKS)
            or correct.get("selection", {}).get("init_state_ids") != list(range(50))
            or correct.get("selection", {}).get("seed") != VIDEO_SEED):
        raise ValueError("selected correct400 manifest does not match the frozen checkpoint")
    freeze = {
        **METHOD_FREEZE_DECLARATION,
        "terminal_macro": selected_macro,
        "writer_checkpoint": checkpoint_record,
        "method": method_metadata(run),
    }
    freeze_path = study / "method_freeze.json"
    write_json_atomic(freeze_path, freeze)
    diagnostic = {
        **DIAGNOSTIC_DECLARATION,
        "checkpoint_macro": selected_macro,
        "paired_correct_manifest": file_record(correct_manifest),
    }
    diagnostic_path = study / "launch" / f"writer_selected_{selected_macro:08d}_diagnostic_contract.json"
    write_json_atomic(diagnostic_path, diagnostic)
    return freeze_path, diagnostic_path, checkpoint_record


def _inspect_owner_stop(path: Path, phase: Mapping[str, Any]) -> dict[str, Any]:
    value = read_json(path)
    history = phase["history"]
    expected_nodes = [{"step": row["step"], "successes": row["successes"], "episodes": row["episodes"]}
                      for row in history]
    if (value.get("schema_version") != "ember_writer_low_lr_owner_stop_v1"
            or value.get("completed_validation_nodes") != expected_nodes
            or value.get("formal_evidence_cutoff_step") != history[-1]["step"]
            or value.get("phase_best_successes") != max(row["successes"] for row in history)
            or value.get("reference_n1000_successes") != REFERENCE_SUCCESSES
            or value.get("training_completed_through_step", 0) < history[-1]["step"]
            or value.get("downstream_launched") is not False):
        raise ValueError("owner stop record does not match the completed low-LR evidence")
    return value


def finalize(study: Path, original_study: Path, owner_stop_path: Path | None = None) -> dict[str, Any]:
    history_path = study / "writer_phase_validation_history.json"
    phase = read_json(history_path)
    owner_stop = _inspect_owner_stop(owner_stop_path, phase) if owner_stop_path else None
    preliminary = choose_phase_checkpoint(phase, owner_stop=owner_stop is not None)
    if preliminary["status"] == "awaiting_tied_other400":
        scores = {}
        for step in preliminary["phase_tied_best_steps"]:
            score = _complete_other_successes(study, step)
            if score is not None:
                scores[step] = score
        decision = choose_phase_checkpoint(phase, scores, owner_stop=owner_stop is not None)
        if decision["status"] != "selected":
            return decision
    else:
        decision = preliminary

    if decision["selected_study"] == "original":
        selected_checkpoint = original_study / "training" / "writer" / "checkpoints" / "macro_00001000"
        if not selected_checkpoint.is_dir():
            raise ValueError("registered original N1000 checkpoint is missing")
        original_run, checkpoint_record = inspect_writer_checkpoint(selected_checkpoint)
        original_freeze_path = original_study / "method_freeze.json"
        expected_freeze = {
            **METHOD_FREEZE_DECLARATION,
            "terminal_macro": REFERENCE_STEP,
            "writer_checkpoint": checkpoint_record,
            "method": method_metadata(original_run),
        }
        if not original_freeze_path.is_file() or read_json(original_freeze_path) != expected_freeze:
            raise ValueError("registered original N1000 method freeze changed")
        decision["selected_checkpoint"] = str(selected_checkpoint)
        decision["method_freeze"] = str(original_freeze_path)
    else:
        freeze, diagnostic, checkpoint = _write_selected_phase_artifacts(study, decision["selected_macro"])
        decision.update(selected_checkpoint=checkpoint["path"], method_freeze=str(freeze),
                        diagnostic_contract=str(diagnostic))
    result = {
        "schema_version": SELECTION_SCHEMA,
        "qualification_arm": "correct",
        **decision,
        "phase_history": str(history_path),
        "selection_rule": "phase correct400 must strictly exceed original N1000=117; tied phase correct uses other400 then earliest",
        "wrong_used_for_selection": False,
    }
    if owner_stop is not None:
        result.update(
            termination="owner_requested_stop",
            owner_stop=str(owner_stop_path.resolve()),
            phase_last_complete_validation_step=owner_stop["formal_evidence_cutoff_step"],
            phase_training_end_step=owner_stop["training_completed_through_step"],
            abandoned_evaluation_step=owner_stop.get("abandoned_evaluation_step"),
        )
    write_json_atomic(study / "writer_low_lr_selection.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--original-study", type=Path, required=True)
    parser.add_argument("--owner-stop", type=Path)
    args = parser.parse_args()
    result = finalize(args.study.resolve(), args.original_study.resolve(),
                      args.owner_stop.resolve() if args.owner_stop else None)
    print(json.dumps(result, indent=2))
    if result["status"] != "selected":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
