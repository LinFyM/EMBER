"""Frozen video controls and final Test admission with explicit provenance."""

from __future__ import annotations

from pathlib import Path

import torch

from ember.expert_manifold.video_schedule import frame_order_seed, task_video_mapping
from ember.pi05_source_checkpoint import read_json
from ember.pi05_target_data import SUITE_ORDER
from ember.video_conditions import frame_control


CONTROL_ARMS = ("cross_suite_wrong", "shuffled", "reversed", "no_video")
DIAGNOSTIC_DECLARATION = {
    "schema_version": "ember_selected_writer_controls_v1",
    "purpose": "sealed_posthoc_video_causality",
    "checkpoint_selection": False,
    "training_feedback": False,
}
SEALED_TEST_DECLARATION = {
    "schema_version": "ember_selected_writer_test_v1",
    "purpose": "frozen_method_test",
    "checkpoint_selection": False,
    "training_feedback": False,
}
METHOD_FREEZE_DECLARATION = {
    "schema_version": "ember_selected_writer_method_freeze_v1",
    "further_training": False,
    "further_architecture_changes": False,
    "test_gradient_use": False,
    "checkpoint_selection": False,
}


def _registered_conditional_held_control(selection) -> bool:
    """Admit only the sealed train-role held400 post-selection video controls."""
    if selection["evaluation_role"] != "development_train" or selection["arm"] not in {
        "same_task_other", "cross_suite_wrong"
    }:
        return False
    spec = read_json(Path(__file__).resolve().parents[3] /
                     "configs/conditional_compilation_diagnostics_v1/experiment_spec.json")
    held = spec["evaluation"]["diagnostic_held"]
    return (selection["task_ids"] == held["task_ids"]
            and selection["init_state_ids"] == held["state_ids"]
            and selection["video_pool"] == held["teacher_demos"]
            and selection["seed"] == spec["evaluation"]["video_schedule_seed"]
            and selection["K"] == 1 and selection["mode"] == "per_init_ordinal"
            and selection["fixed_videos"] == {})


def require_control_selection(selection):
    if selection["evaluation_role"] == "test" and (
            selection["arm"] not in {"correct", "same_task_other", "cross_suite_wrong", "shuffled", "reversed"}
            or selection["K"] != 1
            or selection["mode"] != "per_init_ordinal" or selection["fixed_videos"]
            or selection["init_state_ids"] != list(range(50))
            or selection["video_pool"] != list(range(50))):
        raise ValueError("sealed Test requires fixed test8, K1 correct or registered controls and all 50 canonical state/video ordinals")
    if selection["arm"] in CONTROL_ARMS and not _registered_conditional_held_control(selection) and (
            selection["evaluation_role"] not in {"validation", "test"} or selection["K"] != 1
            or selection["mode"] != "per_init_ordinal" or selection["fixed_videos"]
            or selection["init_state_ids"] != list(range(50))
            or selection["video_pool"] != list(range(50))):
        raise ValueError("selected-checkpoint controls require validation8/test8, K1 and all 50 canonical state/video ordinals")


def video_task_id(selection, task):
    if selection["arm"] == "no_video":
        return None
    if selection["arm"] != "cross_suite_wrong":
        return task
    keys = [(SUITE_ORDER[value // 10], value % 10) for value in selection["task_ids"]]
    mapping = task_video_mapping(keys, {key: selection["evaluation_role"] for key in keys}, "cross_suite_wrong")
    return next(row["video_global_task_id"] for row in mapping if row["language_global_task_id"] == task)


def control_provenance(selection, task, rows):
    donor = video_task_id(selection, task)
    source = rows[donor] if donor is not None else None
    return {"arm": selection["arm"], "selection_seed": selection["seed"],
            "language_global_task_id": task, "language_split_role": selection["evaluation_role"],
            "video_global_task_id": donor,
            "video_suite": source["suite"] if source else None,
            "video_task_id": source["task_id"] if source else None,
            "video_split_role": source["split_role"] if source else None,
            "teacher_source": source["teacher_source"] if source else None,
            "rgb_video_reads": 1 if source else 0,
            "full_observer_and_source_reencode": source is not None,
            "identity_zero_delta": source is None}


def controlled_frames(indices, *, control, demo):
    """One content permutation acts on every declared real camera stream."""
    seed = frame_order_seed(control["selection_seed"], SUITE_ORDER[control["language_global_task_id"] // 10],
                            control["language_global_task_id"] % 10, demo)
    order = frame_control(len(indices), condition=control["arm"], order_seed=seed)
    indices = torch.as_tensor(indices)
    evidence = {"frame_order_seed": seed, "frame_permutation": order.content.tolist(),
                "source_frame_indices": indices[order.content].tolist(),
                "frame_indices": indices[order.positions].tolist(),
                "transform_stage": "declared_real_camera_RGB_before_complete_Writer_forward"}
    return order.content, indices[order.positions], evidence


def inspect_diagnostic_contract(value, *, selection, checkpoint, run, asset_root):
    """Bind controls to one frozen selected checkpoint and its actual correct map."""
    if selection["evaluation_role"] == "test":
        sealed = _inspect_sealed_test(value, selection=selection, checkpoint=checkpoint, run=run)
        if selection["arm"] != "correct":
            record, correct = _inspect_paired_correct(value["paired_correct_manifest"], selection=selection,
                checkpoint=checkpoint, run=run, asset_root=asset_root)
            reference_freeze = _inspect_sealed_test(correct.get("diagnostic_contract"),
                selection=correct["selection"], checkpoint=checkpoint, run=run)
            if reference_freeze["method_freeze"] != sealed["method_freeze"]:
                raise ValueError("Test controls must retain the paired correct bank method freeze")
            sealed["paired_correct_manifest"] = record
        return sealed
    if selection["arm"] not in CONTROL_ARMS and not _registered_conditional_held_control(selection):
        if value is not None:
            raise ValueError("a frozen diagnostic declaration belongs only to video controls")
        return None
    require_control_selection(selection)
    if (not isinstance(value, dict)
            or set(value) != {*DIAGNOSTIC_DECLARATION, "checkpoint_macro", "paired_correct_manifest"}
            or any(value.get(key) != expected for key, expected in DIAGNOSTIC_DECLARATION.items())
            or type(value.get("checkpoint_macro")) is not int
            or value["checkpoint_macro"] <= 0 or checkpoint.get("macro") != value["checkpoint_macro"]):
        raise ValueError("video controls require the explicit frozen selected-checkpoint diagnostic declaration")
    record, _ = _inspect_paired_correct(value["paired_correct_manifest"], selection=selection,
        checkpoint=checkpoint, run=run, asset_root=asset_root)
    return {**DIAGNOSTIC_DECLARATION, "checkpoint_macro": value["checkpoint_macro"], "paired_correct_manifest": record}


def _inspect_paired_correct(reference, *, selection, checkpoint, run, asset_root):
    """Reuse the canonical complete correct400 map for either held split."""
    from ember.writer.materialization import file_record, method_metadata
    from ember.writer.evaluation import _inspect_scope

    path = Path(reference["path"] if isinstance(reference, dict) else reference).resolve()
    record = file_record(path)
    if isinstance(reference, dict) and reference != record:
        raise ValueError("paired correct manifest changed after diagnostic registration")
    correct = read_json(path)
    if (correct.get("arm") != "correct" or correct.get("writer_checkpoint") != checkpoint
            or correct.get("selection") != dict(selection, arm="correct")
            or correct.get("method") != method_metadata(run)
            or correct.get("task_protocol") != run["config"]["data"].get("protocol")
            or Path(correct["asset_root"]).resolve() != asset_root.resolve()):
        raise ValueError("diagnostic controls must retain the frozen selected-checkpoint correct400 checkpoint and map")
    keys = [(row["suite"], row["task_id"]) for row in correct["tasks"]]
    _inspect_scope(correct, run["source"], keys, selection["evaluation_role"], None, True)
    references = {episode["condition_id"] for row in correct["tasks"] for episode in row["episodes"]}
    if len(correct["conditions"]) != 400 or {row["condition_id"] for row in correct["conditions"]} != references:
        raise ValueError("diagnostic reference must retain the complete 400-condition correct bank")
    return record, correct


def _inspect_sealed_test(value, *, selection, checkpoint, run):
    """Require the explicit end of training/design before correct Test400 or its paired frozen controls."""
    from ember.writer.materialization import file_record, method_metadata

    require_control_selection(selection)
    fields = {*SEALED_TEST_DECLARATION, "checkpoint_macro", "method_freeze"}
    if selection["arm"] != "correct":
        fields.add("paired_correct_manifest")
    if (not isinstance(value, dict)
            or set(value) != fields
            or any(value.get(key) != expected for key, expected in SEALED_TEST_DECLARATION.items())
            or type(value.get("checkpoint_macro")) is not int
            or value["checkpoint_macro"] <= 0 or checkpoint.get("macro") != value["checkpoint_macro"]):
        raise ValueError("Test requires the explicit frozen selected-checkpoint Test400 declaration")
    reference = value["method_freeze"]
    path = Path(reference["path"] if isinstance(reference, dict) else reference).resolve()
    record = file_record(path)
    if isinstance(reference, dict) and reference != record:
        raise ValueError("method freeze changed after Test registration")
    expected = {**METHOD_FREEZE_DECLARATION, "terminal_macro": checkpoint["macro"],
                "writer_checkpoint": checkpoint, "method": method_metadata(run)}
    if read_json(path) != expected:
        raise ValueError("Test method freeze must end training/design and bind the identical selected-checkpoint checkpoint and method")
    return {**SEALED_TEST_DECLARATION, "checkpoint_macro": value["checkpoint_macro"], "method_freeze": record}
