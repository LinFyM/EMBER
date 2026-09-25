"""Exact final two-task panels for the registered support-slot fork."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.pi05_assets import Pi05EvaluationError
from ember.writer.support_slot_credit import ARMS, SPEC_PATH, authority, registered_bank_request, root


TAG = "ember_support_slot_passive_capture_v1"
MODELS = ("P1155", *ARMS)


def _stage(output: Path, model: str) -> str:
    for stage in ("pilot", "remaining"):
        if output.resolve() == root() / "evaluation" / stage / model:
            return stage
    raise Pi05EvaluationError("support-slot output is outside its exact pilot/remaining panel")


def _states(stage: str) -> tuple[int, ...]:
    return (0,) if stage == "pilot" else tuple(range(1, 50))


def bank_path(model: str) -> Path:
    if model not in MODELS:
        raise Pi05EvaluationError("unregistered support-slot model")
    return root() / "materialization" / "final" / model / "manifest.json"


def select_tasks(args: Any, tasks: Sequence[Any], repo_root: Path) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    model = str(args.support_slot_model)
    stage = _stage(Path(args.output_dir), model)
    spec = authority()
    if (args.role != "development_train" or args.mode != "formal" or args.state_count != 50
            or Path(args.config).resolve() != (repo_root / "configs/relational_support_causality_v1/evaluation.json").resolve()
            or Path(args.static_task_lora_manifest).resolve() != bank_path(model)
            or any(getattr(args, field, None) for field in (
                "task_subset_selection", "trajectory_capture_selection", "occupancy_capture_selection",
                "frozen_replay_registration", "frozen_prefix_panel", "approach_channel_panel",
                "readout_realization_panel", "native_reader_transfer_cell", "source_sft_checkpoint",
                "task_expert_config"))
            or getattr(args, "capture_stage_predicates", False)
            or getattr(args, "exploration_sigma", False)):
        raise Pi05EvaluationError("support-slot evaluation arguments changed")
    wanted = {(row["suite"], row["task_id"]) for row in spec["evaluation"]["tasks"]}
    selected = tuple(replace(task, init_state_ids=_states(stage)) for task in tasks
                     if (task.suite, task.task_id) in wanted)
    if len(selected) != 2 or {(task.suite, task.task_id) for task in selected} != wanted:
        raise Pi05EvaluationError("support-slot target task identity changed")
    output = Path(args.output_dir).resolve()
    spec_path = (repo_root / SPEC_PATH).resolve()
    full = [{"suite": task.suite, "task_id": task.task_id, "init_state_id": state}
            for task in selected for state in spec["evaluation"]["full_states"] if state in _states(stage)]
    capture = {
        "schema_version": "ember_pi05_registered_trajectory_capture_v1", "mode": "compact",
        "full_conditions": full, "trajectory_root": str(output / "trajectories"),
        "passive_trace": {"schema_version": TAG, "spec_path": str(spec_path),
                          "spec_bytes": spec_path.stat().st_size,
                          "trace_root": str(output / "continuous_traces")},
        "training_gradient_use": False, "checkpoint_selection_use": False,
        "validation_use": False, "test_use": False,
    }
    stage_predicates = {
        "schema_version": "ember_pi05_stage_predicate_capture_v1",
        "capture": "all_rows_post_settling_then_every_executed_control_step",
        "predicate_source": "installed_LIBERO_BDDL_goal_conjunction", "full_conditions_only": False,
        "training_gradient_use": False, "checkpoint_selection_use": False,
        "validation_action_reads": 0, "validation_reward_reads": 0, "held_data_use": False,
        "claim_boundary": "BDDL predicates are partial progress signals",
    }
    return selected, capture, stage_predicates


def attach(contract: dict[str, Any], *, model: str, repo_root: Path) -> None:
    spec = authority()
    stage = _stage(Path(contract["output_dir"]), model)
    path = bank_path(model)
    bank = (contract.get("adapter") or {}).get("manifest") or {}
    actual = {(row["suite"], int(row["task_id"]), int(state))
              for row in contract["tasks"] for state in row["init_state_ids"]}
    expected = {(row["suite"], row["task_id"], state)
                for row in spec["evaluation"]["tasks"] for state in _states(stage)}
    capture = contract.get("diagnostic_occupancy_capture") or {}
    predicates = contract.get("diagnostic_stage_predicates") or {}
    full = {(row["suite"], int(row["task_id"]), int(row["init_state_id"]))
            for row in capture.get("full_conditions", ())}
    spec_path = (repo_root / SPEC_PATH).resolve()
    if (contract.get("role") != "development_train" or contract.get("mode") != "formal"
            or contract.get("arm") != "correct" or actual != expected
            or full != {case for case in expected if case[2] in (0, 25)}
            or bank != {"path": str(path), "bytes": path.stat().st_size}
            or capture.get("schema_version") != "ember_pi05_registered_trajectory_capture_v1"
            or capture.get("mode") != "compact"
            or capture.get("trajectory_root") != str(Path(contract["output_dir"]) / "trajectories")
            or any(capture.get(key) is not False for key in (
                "training_gradient_use", "checkpoint_selection_use", "validation_use", "test_use"))
            or capture.get("passive_trace") != {"schema_version": TAG,
                "spec_path": str(spec_path), "spec_bytes": spec_path.stat().st_size,
                "trace_root": str(Path(contract["output_dir"]) / "continuous_traces")}
            or predicates.get("schema_version") != "ember_pi05_stage_predicate_capture_v1"
            or predicates.get("capture") != "all_rows_post_settling_then_every_executed_control_step"
            or predicates.get("full_conditions_only") is not False
            or predicates.get("predicate_source") != "installed_LIBERO_BDDL_goal_conjunction"):
        raise Pi05EvaluationError("support-slot panel, bank or all-row passive trace changed")
    contract["support_slot_credit"] = {
        "schema_version": "ember_support_slot_evaluation_v1", "model": model, "stage": stage,
        "study_spec": {"path": str(spec_path), "bytes": spec_path.stat().st_size},
        "bank_manifest": bank, "evaluation_commit": contract["git"]["commit"],
    }


def validate_contract(contract: Mapping[str, Any], repo_root: Path) -> None:
    registration = contract.get("support_slot_credit") or {}
    if not registration:
        raise Pi05EvaluationError("support-slot evaluation registration missing")
    expected = dict(contract)
    expected.pop("support_slot_credit", None)
    attach(expected, model=registration.get("model"), repo_root=repo_root)
    if expected["support_slot_credit"] != registration:
        raise Pi05EvaluationError("support-slot evaluation registration changed after prepare")


def validate_bank(manifest: Mapping[str, Any], path: Path, model: str, current_commit: str,
                  run: Mapping[str, Any], checkpoint: Mapping[str, Any], *, phase: str = "final") -> None:
    expected_path = root() / "materialization" / phase / model / "manifest.json"
    if (model not in MODELS or phase not in ("final", "first_slot", "donor_fm")
            or path.resolve() != expected_path):
        raise Pi05EvaluationError("support-slot bank is outside its registered model and phase")
    token = registered_bank_request({"support_slot_model": model, "support_slot_phase": phase,
        "selection": manifest["selection"], "checkpoint": Path(checkpoint["path"]),
        "output": path.parent, "reuse_manifest": None, "diagnostic_contract": None},
        run, checkpoint, {"commit": current_commit})
    compilation = manifest.get("compilation") or {}
    expected_count = {"final": 100, "first_slot": 10, "donor_fm": 2}[phase]
    if (manifest.get("support_slot_credit") != token
            or manifest.get("materialization_git", {}).get("commit") != current_commit
            or manifest.get("writer_checkpoint") != checkpoint
            or compilation.get("new_conditions") != expected_count
            or compilation.get("reused_conditions") != 0
            or len(manifest.get("conditions", ())) != expected_count):
        raise Pi05EvaluationError("support-slot bank generation or source changed")
