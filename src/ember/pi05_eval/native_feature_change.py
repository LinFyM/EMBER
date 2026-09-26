"""Exact pilot/remainder cases for the frozen native-feature intervention."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_target_data import SUITE_ORDER
from ember.writer.native_feature_change import CELLS, SPEC_PATH, STUDY, spec


TAG = "ember_native_feature_change_passive_capture_v1"
PILOT_TASKS = (0, 36)


def scope(output: Path) -> tuple[str, str] | None:
    root = Path(spec()["outputs"]["planned_run_root"]).resolve() / "evaluation"
    path = output.resolve()
    if path.parent.parent != root:
        return None
    stage, cell = path.parent.name, path.name
    if stage not in ("pilot", "remaining") or cell not in CELLS:
        raise Pi05EvaluationError("native-feature evaluation output is outside its eight stages")
    return stage, cell


def cases(stage: str) -> set[tuple[str, int, int]]:
    study = spec()
    if stage not in ("pilot", "remaining"):
        raise Pi05EvaluationError("native-feature stage is not registered")
    result = set()
    for task in study["evaluation"]["task_ids"]:
        suite, local = SUITE_ORDER[task // 10], task % 10
        for state in study["evaluation"]["state_ids"]:
            if (task in PILOT_TASKS and state == 0) == (stage == "pilot"):
                result.add((suite, local, state))
    if len(result) != (2 if stage == "pilot" else 78):
        raise Pi05EvaluationError("native-feature stage rows changed")
    return result


def full_cases(stage: str) -> set[tuple[str, int, int]]:
    capture = spec()["evaluation"]["capture"]
    registered = {(SUITE_ORDER[task // 10], task % 10, state)
                  for task in capture["full_task_ids"] for state in capture["full_state_ids"]}
    return cases(stage) & registered


def bank_tasks(installed: Sequence[Any]) -> tuple[Any, ...]:
    study = spec()
    keys = {(SUITE_ORDER[task // 10], task % 10) for task in study["evaluation"]["task_ids"]}
    selected = tuple(replace(task, init_state_ids=tuple(study["evaluation"]["state_ids"]))
                     for task in installed if (task.suite, task.task_id) in keys)
    if len(selected) != 8 or {(task.suite, task.task_id) for task in selected} != keys:
        raise Pi05EvaluationError("native-feature bank tasks differ from frozen held8")
    return selected


def select_tasks(args: Any, installed: Sequence[Any], repo_root: Path):
    registered = scope(Path(args.output_dir))
    if registered is None:
        raise Pi05EvaluationError("native-feature output is unregistered")
    stage, cell = registered
    study = spec()
    root = Path(study["outputs"]["planned_run_root"]).resolve()
    bank = root / "materialization" / cell / "manifest.json"
    if (args.role != "development_train" or args.mode != "screen" or args.state_count != 10
            or getattr(args, "init_state_ids", None) is not None
            or Path(args.config).resolve() !=
               (repo_root / "configs/libero_24_8_8_coverage_v1/evaluation.json").resolve()
            or getattr(args, "static_task_lora_manifest", None) is None
            or Path(args.static_task_lora_manifest).resolve() != bank
            or any(getattr(args, field, None) for field in (
                "task_subset_selection", "trajectory_capture_selection", "occupancy_capture_selection",
                "frozen_replay_registration", "frozen_prefix_panel", "approach_channel_panel",
                "readout_realization_panel", "native_reader_transfer_cell", "support_slot_model",
                "source_sft_checkpoint", "task_expert_config"))
            or getattr(args, "capture_stage_predicates", False)
            or getattr(args, "exploration_sigma", False)):
        raise Pi05EvaluationError("native-feature evaluator args, bank, or source scope changed")
    expected = cases(stage)
    all_bank_tasks = bank_tasks(installed)
    selected = tuple(replace(task, init_state_ids=tuple(
        state for state in study["evaluation"]["state_ids"]
        if (task.suite, task.task_id, state) in expected))
        for task in all_bank_tasks
        if any((task.suite, task.task_id, state) in expected
               for state in study["evaluation"]["state_ids"]))
    actual = {(task.suite, task.task_id, state) for task in selected for state in task.init_state_ids}
    if actual != expected:
        raise Pi05EvaluationError("native-feature pilot/remainder selection changed")
    output = Path(args.output_dir).resolve()
    spec_path = (repo_root / SPEC_PATH).resolve()
    full = [{"suite": suite, "task_id": task, "init_state_id": state}
            for suite, task, state in sorted(full_cases(stage))]
    capture = {
        "schema_version": "ember_pi05_registered_trajectory_capture_v1",
        "mode": "compact", "full_conditions": full,
        "trajectory_root": str(output / "trajectories"),
        "passive_trace": {"schema_version": TAG, "spec_path": str(spec_path),
                          "spec_bytes": spec_path.stat().st_size,
                          "trace_root": str(output / "continuous_traces")},
        "training_gradient_use": False, "checkpoint_selection_use": False,
        "validation_use": False, "test_use": False,
    }
    predicates = {
        "schema_version": "ember_pi05_stage_predicate_capture_v1",
        "capture": "all_rows_post_settling_then_every_executed_control_step",
        "predicate_source": "installed_LIBERO_BDDL_goal_conjunction",
        "full_conditions_only": False, "training_gradient_use": False,
        "checkpoint_selection_use": False, "validation_action_reads": 0,
        "validation_reward_reads": 0, "held_data_use": False,
        "claim_boundary": "BDDL predicates are partial progress signals",
    }
    return selected, capture, predicates


def attach(contract: dict[str, Any], repo_root: Path) -> None:
    registered = scope(Path(contract["output_dir"]))
    if registered is None:
        return
    stage, cell = registered
    study = spec()
    root = Path(study["outputs"]["planned_run_root"]).resolve()
    bank = root / "materialization" / cell / "manifest.json"
    adapter = contract.get("adapter") or {}
    capture = contract.get("diagnostic_occupancy_capture") or {}
    predicates = contract.get("diagnostic_stage_predicates") or {}
    actual = {(row["suite"], int(row["task_id"]), state)
              for row in contract["tasks"] for state in row["init_state_ids"]}
    full = {(row["suite"], int(row["task_id"]), int(row["init_state_id"]))
            for row in capture.get("full_conditions", ())}
    spec_path = (repo_root / SPEC_PATH).resolve()
    expected_capture = {
        "schema_version": "ember_pi05_registered_trajectory_capture_v1",
        "mode": "compact",
        "full_conditions": [{"suite": suite, "task_id": task, "init_state_id": state}
                            for suite, task, state in sorted(full_cases(stage))],
        "trajectory_root": str(Path(contract["output_dir"]) / "trajectories"),
        "passive_trace": {"schema_version": TAG, "spec_path": str(spec_path),
                          "spec_bytes": spec_path.stat().st_size,
                          "trace_root": str(Path(contract["output_dir"]) / "continuous_traces")},
        "training_gradient_use": False, "checkpoint_selection_use": False,
        "validation_use": False, "test_use": False,
    }
    if (contract["role"] != "development_train" or contract["mode"] != "screen"
            or contract["arm"] != "correct" or actual != cases(stage)
            or full != full_cases(stage) or capture != expected_capture
            or predicates.get("full_conditions_only") is not False
            or predicates.get("capture") != "all_rows_post_settling_then_every_executed_control_step"
            or adapter.get("manifest") != {"path": str(bank), "bytes": bank.stat().st_size}
            or Path(contract["model"]["checkpoint"]).resolve() !=
               Path(study["frozen_input"]["source_checkpoint"]).resolve()
            or contract["git"]["commit"] != adapter["materialization_git"]["commit"]):
        raise Pi05EvaluationError("native-feature cases, passive trace, source, or bank changed")
    contract["native_feature_change"] = {
        "schema_version": "ember_native_feature_change_evaluation_v1",
        "study_id": STUDY, "cell": cell, "stage": stage,
        "spec_path": str(spec_path), "spec_bytes": spec_path.stat().st_size,
        "bank_manifest": adapter["manifest"], "evaluation_commit": contract["git"]["commit"],
    }


def validate_contract(contract: Mapping[str, Any], repo_root: Path) -> None:
    expected = dict(contract)
    expected.pop("native_feature_change", None)
    attach(expected, repo_root)
    if expected.get("native_feature_change") != contract.get("native_feature_change"):
        raise Pi05EvaluationError("native-feature evaluation provenance changed after prepare")
