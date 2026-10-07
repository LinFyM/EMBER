"""Registered passive capture and provenance for operator bank evaluations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_source_checkpoint import read_json

from . import scope as seen_scope


PASSIVE_TAG = "ember_operator_read_write_passive_capture_v1"


def registered_capture(args, tasks, output_dir: Path, path: Path, manifest: Mapping,
                       task_subset: Mapping | None) -> tuple[dict, dict]:
    from .bank import KIND

    bank_path = Path(args.static_task_lora_manifest).resolve()
    bank = read_json(bank_path)
    if bank.get("cross_context_pairing") is not None:
        from ember.cross_context_pairing.readout import registered_capture as pairing_capture

        return pairing_capture(args, tasks, output_dir, path, manifest, task_subset, bank)
    if bank.get("reexpression_panel") is not None:
        from .reexpression import registered_capture as reexpression_capture

        return reexpression_capture(args, tasks, output_dir, path, manifest, task_subset, bank)
    if bank.get("learning_limit_panel") is not None:
        from .learning_limit import registered_capture as learning_capture

        return learning_capture(args, tasks, output_dir, path, manifest, task_subset, bank)
    try:
        if bank.get("joint_public_study") is True:
            from .joint_readout import capture_expectations

            expected = capture_expectations(bank, bank_path, tasks, output_dir)
        else:
            expected = seen_scope.capture_expectations(bank, bank_path, tasks, output_dir)
    except ValueError as error:
        raise Pi05EvaluationError(str(error)) from error
    facts = ((path.resolve(), expected["capture"].resolve()),
             (bank.get("kind"), KIND),
             (manifest.get("schema_version"), "ember_pi05_registered_trajectory_capture_v1"),
             (manifest.get("study_id"), expected["study"]),
             (manifest.get("task_subset_selection"), None),
             (manifest.get("full_conditions"), expected["full"]),
             (manifest.get("mode"), "compact"),
             (manifest.get("passive_control_trace"), PASSIVE_TAG),
             (manifest.get("stage_predicates"), True),
             (args.role, expected["role"]), (args.mode, "formal"),
             (len(tasks), expected["task_count"]),
             (output_dir.resolve(), expected["output"].resolve()))
    if (any(actual != wanted for actual, wanted in facts)
            or expected["expected_bank"] is not None and bank_path != expected["expected_bank"]
            or task_subset is not None
            or any(tuple(task.init_state_ids) != expected["states"] for task in tasks)
            or any(manifest.get(key) is not False for key in (
                "training_gradient_use", "checkpoint_selection_use", "validation_use", "test_use"))):
        raise Pi05EvaluationError("operator official full/compact capture scope changed")
    capture = {"schema_version": "ember_pi05_registered_trajectory_capture_v1",
               "selection_path": str(path), "selection_bytes": path.stat().st_size,
               "mode": "compact", "full_conditions": expected["full"],
               "trajectory_root": str((output_dir / "trajectories").resolve()),
               "passive_trace": {"schema_version": PASSIVE_TAG,
                                 "trace_root": str((output_dir / "continuous_traces").resolve())},
               "training_gradient_use": False, "checkpoint_selection_use": False,
               "validation_use": False, "test_use": False}
    stage = {"schema_version": "ember_pi05_stage_predicate_capture_v1",
             "capture": "all_rows_post_settling_then_every_executed_control_step",
             "predicate_source": "installed_LIBERO_BDDL_goal_conjunction",
             "full_conditions_only": False, "training_gradient_use": False,
             "checkpoint_selection_use": False, "validation_action_reads": 0,
             "validation_reward_reads": 0, "held_data_use": False,
             "claim_boundary": "BDDL predicates are partial progress signals"}
    return capture, stage


def attach_capture_provenance(contract: dict, repo_root: Path) -> None:
    from .bank import KIND

    del repo_root
    adapter = contract.get("adapter") or {}
    if adapter.get("cross_context_pairing") is not None:
        from ember.cross_context_pairing.readout import attach_provenance

        attach_provenance(contract)
        return
    scene = contract.get("operator_read_write_scene") or {}
    legacy_test = contract.get("operator_read_write_legacy_test")
    if adapter.get("kind") != KIND:
        raise Pi05EvaluationError("operator capture adapter kind changed")
    if legacy_test is None:
        if scene.get("manifest") != adapter.get("scene_manifest") or not scene:
            raise Pi05EvaluationError("operator scene and adapter are not paired")
    elif (contract.get("role") != "test" or scene or adapter.get("scene_manifest") is not None
          or legacy_test != {"initialization": adapter.get("legacy_test_initialization"),
                             "selection": (adapter.get("selected_test") or {}).get("selection"),
                             "historical_test_exposure": True}):
        raise Pi05EvaluationError("operator selected Test legacy initialization changed")
    contract["passive_capture_provenance"] = {
        "schema_version": PASSIVE_TAG, "bank": adapter["manifest"],
        "scene": None if legacy_test is not None else adapter["scene_manifest"],
        "checkpoint": adapter["checkpoint"],
        "evaluation_commit": contract["git"]["commit"]}
    if legacy_test is not None:
        contract["passive_capture_provenance"]["legacy_test_initialization"] = legacy_test


def validate_capture_contract(contract: Mapping, repo_root: Path) -> None:
    adapter = contract.get("adapter") or {}
    capture = contract.get("diagnostic_occupancy_capture") or {}
    path = Path(capture["selection_path"])
    args = SimpleNamespace(static_task_lora_manifest=Path(adapter["manifest"]["path"]),
                           role=contract["role"], mode=contract["mode"])
    tasks = [SimpleNamespace(**row) for row in contract["tasks"]]
    expected, stage = registered_capture(args, tasks, Path(contract["output_dir"]),
                                         path, read_json(path), contract["diagnostic_task_subset"])
    regenerated = dict(contract)
    regenerated.pop("passive_capture_provenance", None)
    attach_capture_provenance(regenerated, repo_root)
    if (capture != expected or contract.get("diagnostic_stage_predicates") != stage
            or contract.get("passive_capture_provenance") != regenerated["passive_capture_provenance"]):
        raise Pi05EvaluationError("operator passive capture or scene provenance changed")
