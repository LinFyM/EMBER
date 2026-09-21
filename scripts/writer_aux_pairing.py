#!/usr/bin/env python3
"""CPU preflight and complete-panel decisions for the cross-episode Writer run.

This command never launches training or evaluations. Call it from the existing
local stage controller after resource checks / completed job exits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ember.writer.auxiliary_pairing import (
    DECLARATION, INTERVAL, REFERENCE_CONFIG, audit_actual_exposures, audit_events,
    auxiliary_validation_decision, require_matched_config, selection_plan,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/libero_24_8_8_coverage_v1/writer_aux_cross_episode.json"
# Existing canonical aggregate version; no new evaluator or aggregation code.
AGGREGATE_SCHEMA = "ember_pi05_target_eval_results_v2"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def preflight(args) -> dict[str, Any]:
    # Import the real parser here, so the local environment also checks all
    # existing source, shape, objective, topology and continuation contracts.
    from ember.task_protocol import load_task_authorities
    from ember.writer.training import _config

    candidate = _config(args.config)
    reference = read_json(ROOT / REFERENCE_CONFIG)
    require_matched_config(candidate, reference)
    _, manifest = load_task_authorities(args.asset_root, candidate["data"]["protocol"])
    tasks = set(candidate["data"]["task_ids"])
    rows = {row["global_task_id"]: row for row in manifest["tasks"]}
    if any(rows[t]["split_role"] != "train" for t in tasks):
        raise ValueError("preflight task selection crossed the Train role boundary")
    lengths = {t: rows[t]["demonstrations"]["episode_lengths"] for t in tasks}
    audit, expected = audit_events(reference, candidate, lengths, updates=args.updates)
    if args.reference_exposures is not None:
        count = audit_actual_exposures(args.reference_exposures, expected)
        audit.update(actual_reference_exposures_checked=True, actual_reference_rows=count)
    else:
        audit["warning"] = "Replay equality checked; archived actual exposures have not been verified."
    source = args.asset_root / candidate["source"]["checkpoint"]
    if not source.is_dir():
        raise FileNotFoundError(f"canonical Source checkpoint is unavailable: {source}")
    return {"schema_version": "ember_aux_pairing_preflight_v1", "experiment": DECLARATION,
            "config": str(args.config), "event_alignment": audit,
            "only_scientific_difference": "data.teaching_episode", "model_or_loss_code_changed": False,
            "source_checkpoint_present": True, "gpu_execution_tested": False,
            "resources_checked": False, "training_started": False,
            "next": "Verify local GPU/NUMA/quota and frozen runtime, then start the one fresh run."}


def load_panel(study: Path, step: int, config: dict[str, Any], *, arm: str = "correct") -> dict[str, Any]:
    if arm not in {"correct", "same_task_other"}:
        raise ValueError("only qualification arms may enter checkpoint selection")
    suffix = "" if arm == "correct" else "_other"
    mat_suffix = "_correct" if arm == "correct" else "_other"
    directory = study / "evaluation" / f"writer_{step:08d}{suffix}"
    result = read_json(directory / "results.json")
    contract = read_json(directory / "run_contract.json")
    done = read_json(directory / "launcher_completion.json")
    manifest = read_json(study / "materialized" / f"writer_{step:08d}{mat_suffix}" / "manifest.json")
    reference = result.get("contract_reference")
    if (result.get("schema_version") != AGGREGATE_SCHEMA or not reference
            or reference != contract.get("contract_reference") or reference != done.get("contract_reference")
            or contract.get("role") != "validation" or contract.get("mode") != "formal"):
        raise ValueError(f"invalid formal Validation panel contract at {step} / {arm}")
    codes = done.get("return_codes")
    if (not isinstance(codes, dict) or not codes
            or any(type(code) is not int or code != 0 for code in codes.values())):
        raise ValueError("evaluation workers did not complete successfully")
    panel = config["evidence"]["qualification"]
    expected = {(task, state) for task in panel["tasks"] for state in range(50)}
    rows = result.get("rows", [])
    if len(rows) != 400 or len(expected) != 400:
        raise ValueError("selection requires complete 8-task x 50-state panels")
    keys = [(row.get("horizon_writer_lora", {}).get("global_task_id"), row.get("init_state_id")) for row in rows]
    if len(set(keys)) != 400 or set(keys) != expected:
        raise ValueError("validation panel has missing, unexpected or duplicate task/state rows")
    if any(type(row.get("success")) is not bool for row in rows):
        raise ValueError("success labels must be real JSON booleans")
    scores = sum(row["success"] for row in rows)
    if (result.get("overall", {}).get("episodes") != 400
            or result.get("overall", {}).get("successes") != scores
            or done.get("queue", {}).get("completed_rows") != 400
            or done.get("queue", {}).get("successes") != scores):
        raise ValueError("row counts or success aggregates disagree with completion")
    selection = manifest.get("selection", {})
    if (manifest.get("arm") != arm or manifest.get("writer_checkpoint", {}).get("macro") != step
            or selection.get("evaluation_role") != "validation"
            or selection.get("task_ids") != panel["tasks"]
            or selection.get("init_state_ids") != list(range(50))
            or selection.get("seed") != panel["seed"]):
        raise ValueError("materialized checkpoint or video selection manifest differs")
    schedules = {task: set() for task in panel["tasks"]}
    for row in rows:
        meta = row["horizon_writer_lora"]
        if (meta.get("K") != 1 or meta.get("selection_seed") != panel["seed"]
                or meta.get("selection_mode") != panel["selection_mode"]
                or type(meta.get("video_ordinal")) is not int or not 0 <= meta["video_ordinal"] < 50
                or row.get("split_role") != "validation"):
            raise ValueError("row-level video schedule or held role changed")
        schedules[meta["global_task_id"]].add(meta["video_ordinal"])
    if any(len(values) != 50 for values in schedules.values()):
        raise ValueError("each task must cover all 50 video ordinals")
    return result


def status(args) -> dict[str, Any]:
    from ember.pi05_eval_results import paired_success_comparison

    config = read_json(args.config)
    require_matched_config(config, read_json(ROOT / REFERENCE_CONFIG))
    if args.endpoint <= 0 or args.endpoint % INTERVAL:
        raise ValueError("endpoint must be a positive complete 200-update node")
    training_root = args.study / "training" / "writer"
    actual_run = read_json(training_root / "run_contract.json")
    if actual_run.get("config") != config:
        raise ValueError("status config differs from the actual candidate training contract")
    training_done = read_json(training_root / "completion.json")
    if (training_done.get("status") != "segment_complete"
            or training_done.get("mode") != "formal"
            or training_done.get("optimizer_updates") != args.endpoint):
        raise ValueError("finish the registered training segment before reading Validation")
    history, comparisons, panels = [], [], {}
    for step in range(INTERVAL, args.endpoint + 1, INTERVAL):
        result = load_panel(args.study, step, config)
        if panels:
            comparisons.append({"step": step, **paired_success_comparison(panels[step - INTERVAL], result)})
        panels[step] = result
        history.append({"step": step, "episodes": 400, "complete": True,
                        "successes": result["overall"]["successes"],
                        "results": str(args.study / "evaluation" / f"writer_{step:08d}" / "results.json")})
        if step < args.endpoint and auxiliary_validation_decision(history)["stop"]:
            raise ValueError("candidate continued past its first registered stopping decision")
    decision = auxiliary_validation_decision(history)
    others = {}
    if decision["stop"] and len(decision["tied_best_steps"]) > 1:
        for step in decision["tied_best_steps"]:
            directory = args.study / "evaluation" / f"writer_{step:08d}_other"
            if directory.exists():
                other = load_panel(args.study, step, config, arm="same_task_other")
                paired_success_comparison(panels[step], other)
                others[step] = other["overall"]["successes"]
    plan = selection_plan(history, others)
    return {"schema_version": "ember_aux_pairing_validation_history_v1", "experiment": DECLARATION,
            "history": history, "decision": decision, "selection_plan": plan,
            "adjacent_comparisons": comparisons, "test_allowed": False,
            "old_writer_reference": {"step": 1000, "successes": 117},
            "mtbc_reference": {"step": 300, "successes": 155},
            "next": plan["status"], "training_or_evaluation_launched": False}


def finalize_selection(study: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Publish immutable method/diagnostic declarations for the selected candidate."""
    from ember.writer.materialization import file_record, inspect_writer_checkpoint, method_metadata
    from ember.writer.video_controls import DIAGNOSTIC_DECLARATION, METHOD_FREEZE_DECLARATION

    plan = result["selection_plan"]
    if plan["status"] != "selected":
        raise ValueError("finish training and every tied other400 before finalizing")
    step = plan["selected_macro"]
    checkpoint = study / "training" / "writer" / "checkpoints" / f"macro_{step:08d}"
    run, checkpoint_record = inspect_writer_checkpoint(checkpoint)
    correct_path = study / "materialized" / f"writer_{step:08d}_correct" / "manifest.json"
    correct = read_json(correct_path)
    if correct.get("writer_checkpoint") != checkpoint_record or correct.get("arm") != "correct":
        raise ValueError("selected checkpoint does not match its evaluated correct manifest")
    freeze = {**METHOD_FREEZE_DECLARATION, "terminal_macro": step,
              "writer_checkpoint": checkpoint_record, "method": method_metadata(run)}
    diagnostic = {**DIAGNOSTIC_DECLARATION, "checkpoint_macro": step,
                  "paired_correct_manifest": file_record(correct_path)}
    selection = {"schema_version": "ember_aux_pairing_selection_v1", "experiment": DECLARATION,
                 **plan, "run_contract": str(checkpoint.parent.parent / "run_contract.json"),
                 "checkpoint": checkpoint_record,
                 "controls_define_measurement_budget_only": True,
                 "no_test_ft_rl_authorized": True}
    values = [(study / "writer_aux_selection.json", selection)]
    if plan["controls_required"]:
        values += [(study / "method_freeze.json", freeze),
                   (study / "launch" / f"writer_selected_{step:08d}_diagnostic_contract.json", diagnostic)]
    # Pre-check every file before publishing any declaration. Exact reruns are
    # idempotent; a different selection requires explicit human review.
    for path, value in values:
        if path.exists() and read_json(path) != value:
            raise ValueError(f"refusing to overwrite a frozen selection: {path}")
    for path, value in values:
        if not path.exists():
            write_json(path, value)
    return {"published": [str(path) for path, _ in values],
            "old_selected_model_overwritten": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight", help="CPU/metadata audit; reads no action arrays")
    check.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    check.add_argument("--asset-root", type=Path, required=True)
    check.add_argument("--reference-exposures", type=Path)
    check.add_argument("--updates", type=int, default=1800)
    check.add_argument("--output", type=Path, required=True)
    review = sub.add_parser("status", help="validate complete panels and return the next mechanical action")
    review.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    review.add_argument("--study", type=Path, required=True)
    review.add_argument("--endpoint", type=int, required=True)
    review.add_argument("--finalize", action="store_true", help="publish selected candidate declarations after stopping/ties")
    args = parser.parse_args()
    output = args.output if args.command == "preflight" else args.study / "writer_aux_validation_history.json"
    value = preflight(args) if args.command == "preflight" else status(args)
    if args.command == "status" and args.finalize:
        value["finalization"] = finalize_selection(args.study, value)
    write_json(output, value)
    print(json.dumps({k: v for k, v in value.items() if k not in {"history", "adjacent_comparisons"}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
