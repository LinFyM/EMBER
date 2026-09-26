"""Bounded S0/C0 staged cases for the canonical PI0.5 evaluator and passive capture."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_target_data import SUITE_ORDER
from ember.writer.learned_initial_content_contract import (
    ROOT, SPEC_PATH, STUDY, evaluation_panel, spec,
)


TAG = "ember_learned_initial_content_passive_capture_v1"


def cases(row: Mapping[str, Any]) -> set[tuple[str, int, int]]:
    stage = row["stage"]
    result = set()
    for global_task in row["task_ids"]:
        suite, local = SUITE_ORDER[global_task // 10], global_task % 10
        for state in row["state_ids"]:
            pilot = row["kind"] == "held_correct" and global_task in (0, 36) and state == 0
            if stage == "complete" or (stage == "pilot") == pilot:
                result.add((suite, local, state))
    if len(result) != row["rows"]:
        raise Pi05EvaluationError("S0 staged evaluation rows changed")
    return result


def full_cases(row: Mapping[str, Any]) -> set[tuple[str, int, int]]:
    capture = spec()["evaluation"]["capture"]
    held = row["kind"] != "seen_correct"
    keys = ((capture["held_full_task_ids"], capture["held_full_state_ids"])
            if held else (capture["seen_full_task_ids"], capture["seen_full_state_ids"]))
    allowed = {(SUITE_ORDER[t // 10], t % 10, state) for t in keys[0] for state in keys[1]}
    return cases(row) & allowed


def bank_tasks(installed: Sequence[Any], row: Mapping[str, Any]) -> tuple[Any, ...]:
    expected = {(SUITE_ORDER[t // 10], t % 10) for t in row["task_ids"]}
    selected = tuple(replace(task, init_state_ids=tuple(row["state_ids"]))
                     for task in installed if (task.suite, task.task_id) in expected)
    if {(task.suite, task.task_id) for task in selected} != expected:
        raise Pi05EvaluationError("S0/C0 bank tasks changed")
    return selected


def bank_manifest(row: Mapping[str, Any]) -> Path:
    study = spec()
    if row["model"] == "S0":
        return Path(study["outputs"]["planned_run_root"]).resolve() / "materialization" / row["id"] / "manifest.json"
    reference = study["frozen_references"]["C0"]
    return Path(reference[{"held_correct": "held_correct_bank", "held_other": "held_other_bank",
                           "seen_correct": "seen_correct_bank"}[row["kind"]]]).resolve()


def select_tasks(args: Any, installed: Sequence[Any], repo_root: Path):
    row = evaluation_panel(Path(args.output_dir))
    if row is None:
        raise Pi05EvaluationError("S0/C0 evaluation output is unregistered")
    study = spec()
    expected = bank_manifest(row)
    state_count = 4 if row["kind"] == "seen_correct" else 10
    if (args.role != "development_train" or args.mode != "screen"
            or args.state_count != state_count or getattr(args, "init_state_ids", None) is not None
            or Path(args.config).resolve() != (repo_root / study["assets"]["source_config"]["evaluation_config"]).resolve()
            or getattr(args, "static_task_lora_manifest", None) is None
            or Path(args.static_task_lora_manifest).resolve() != expected
            or any(getattr(args, field, None) for field in (
                "task_subset_selection", "trajectory_capture_selection", "occupancy_capture_selection",
                "frozen_replay_registration", "frozen_prefix_panel", "approach_channel_panel",
                "readout_realization_panel", "native_reader_transfer_cell", "support_slot_model",
                "source_sft_checkpoint", "task_expert_config"))
            or getattr(args, "capture_stage_predicates", False)
            or getattr(args, "exploration_sigma", False)):
        raise Pi05EvaluationError("S0/C0 evaluator args, bank, or source scope changed")
    selected = tuple(replace(task, init_state_ids=tuple(
        state for state in row["state_ids"] if (task.suite, task.task_id, state) in cases(row)))
        for task in bank_tasks(installed, row)
        if any((task.suite, task.task_id, state) in cases(row) for state in row["state_ids"]))
    if {(task.suite, task.task_id, state) for task in selected for state in task.init_state_ids} != cases(row):
        raise Pi05EvaluationError("S0/C0 staged cases changed")
    output = Path(args.output_dir).resolve()
    spec_path = (repo_root / SPEC_PATH).resolve()
    capture = {
        "schema_version": "ember_pi05_registered_trajectory_capture_v1", "mode": "compact",
        "full_conditions": [{"suite": suite, "task_id": task, "init_state_id": state}
                            for suite, task, state in sorted(full_cases(row))],
        "trajectory_root": str(output / "trajectories"),
        "passive_trace": {"schema_version": TAG, "spec_path": str(spec_path),
                          "spec_bytes": spec_path.stat().st_size,
                          "trace_root": str(output / "continuous_traces")},
        "training_gradient_use": False, "checkpoint_selection_use": False,
        "validation_use": False, "test_use": False,
    }
    stage = {
        "schema_version": "ember_pi05_stage_predicate_capture_v1",
        "capture": "all_rows_post_settling_then_every_executed_control_step",
        "predicate_source": "installed_LIBERO_BDDL_goal_conjunction",
        "full_conditions_only": False, "training_gradient_use": False,
        "checkpoint_selection_use": False, "validation_action_reads": 0,
        "validation_reward_reads": 0, "held_data_use": False,
        "claim_boundary": "BDDL predicates are partial progress signals",
    }
    return selected, capture, stage


def attach(contract: dict[str, Any], repo_root: Path) -> None:
    row = evaluation_panel(Path(contract["output_dir"]))
    if row is None:
        return
    study = spec()
    adapter = contract.get("adapter") or {}
    capture = contract.get("diagnostic_occupancy_capture") or {}
    actual = {(task["suite"], int(task["task_id"]), int(state))
              for task in contract["tasks"] for state in task["init_state_ids"]}
    full = {(part["suite"], int(part["task_id"]), int(part["init_state_id"]))
            for part in capture.get("full_conditions", ())}
    spec_path = (repo_root / SPEC_PATH).resolve()
    output = Path(contract["output_dir"]).resolve()
    expected_capture = {
        "schema_version": "ember_pi05_registered_trajectory_capture_v1", "mode": "compact",
        "full_conditions": [{"suite": suite, "task_id": task, "init_state_id": state}
                            for suite, task, state in sorted(full_cases(row))],
        "trajectory_root": str(output / "trajectories"),
        "passive_trace": {"schema_version": TAG, "spec_path": str(spec_path),
                          "spec_bytes": spec_path.stat().st_size,
                          "trace_root": str(output / "continuous_traces")},
        "training_gradient_use": False, "checkpoint_selection_use": False,
        "validation_use": False, "test_use": False,
    }
    if (contract["role"] != "development_train" or contract["mode"] != "screen"
            or contract["arm"] != row["condition"] or actual != cases(row)
            or full != full_cases(row) or capture != expected_capture
            or contract.get("diagnostic_stage_predicates", {}).get("full_conditions_only") is not False
            or contract.get("diagnostic_stage_predicates", {}).get("capture") !=
               "all_rows_post_settling_then_every_executed_control_step"
            or adapter.get("manifest") != {"path": str(bank_manifest(row)),
                                           "bytes": bank_manifest(row).stat().st_size}
            or Path(contract["model"]["checkpoint"]).resolve() !=
               Path(study["assets"]["source_checkpoint"]).resolve()):
        raise Pi05EvaluationError("S0/C0 cases, all-row capture, source, or bank changed")
    contract["learned_initial_content"] = {
        "schema_version": "ember_learned_initial_content_evaluation_v1",
        "study_id": STUDY, "panel": row["id"], "stage": row["stage"],
        "spec_path": str(spec_path), "spec_bytes": spec_path.stat().st_size,
        "bank_manifest": adapter["manifest"],
        "bank_training_commit": adapter["writer_checkpoint"]["training_commit"],
        "bank_materialization_commit": adapter["materialization_git"]["commit"],
        "evaluation_commit": contract["git"]["commit"],
    }


def validate_contract(contract: Mapping[str, Any], repo_root: Path) -> None:
    expected = dict(contract)
    expected.pop("learned_initial_content", None)
    attach(expected, repo_root)
    if expected.get("learned_initial_content") != contract.get("learned_initial_content"):
        raise Pi05EvaluationError("S0/C0 evaluation provenance changed after prepare")
