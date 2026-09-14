"""Frozen post-hoc video controls, with explicit donor and real-frame provenance."""

from __future__ import annotations

from pathlib import Path

import torch

from ember.expert_manifold.video_schedule import frame_order_seed, task_video_mapping
from ember.pi05_source_checkpoint import read_json
from ember.pi05_target_data import SUITE_ORDER
from ember.video_conditions import frame_control


CONTROL_ARMS = ("cross_suite_wrong", "shuffled", "reversed", "no_video")
DIAGNOSTIC_DECLARATION = {
    "schema_version": "ember_process_pullback_terminal900_controls_v1",
    "purpose": "sealed_posthoc_video_causality",
    "checkpoint_macro": 900,
    "checkpoint_selection": False,
    "training_feedback": False,
}


def require_control_selection(selection):
    if selection["arm"] in CONTROL_ARMS and (
            selection["evaluation_role"] != "validation" or selection["K"] != 1
            or selection["mode"] != "per_init_ordinal" or selection["fixed_videos"]
            or selection["init_state_ids"] != list(range(50))
            or selection["video_pool"] != list(range(50))):
        raise ValueError("terminal900 controls require validation8, K1 and all 50 canonical state/video ordinals")


def video_task_id(selection, task):
    if selection["arm"] == "no_video":
        return None
    if selection["arm"] != "cross_suite_wrong":
        return task
    keys = [(SUITE_ORDER[value // 10], value % 10) for value in selection["task_ids"]]
    mapping = task_video_mapping(keys, {key: "validation" for key in keys}, "cross_suite_wrong")
    return next(row["video_global_task_id"] for row in mapping if row["language_global_task_id"] == task)


def control_provenance(selection, task, rows):
    donor = video_task_id(selection, task)
    source = rows[donor] if donor is not None else None
    return {"arm": selection["arm"], "selection_seed": selection["seed"],
            "language_global_task_id": task, "language_split_role": "validation",
            "video_global_task_id": donor,
            "video_suite": source["suite"] if source else None,
            "video_task_id": source["task_id"] if source else None,
            "video_split_role": source["split_role"] if source else None,
            "teacher_source": source["teacher_source"] if source else None,
            "rgb_video_reads": 1 if source else 0,
            "full_observer_and_source_reencode": source is not None,
            "identity_zero_delta": source is None}


def controlled_frames(indices, *, control, demo):
    """The same content permutation acts on both real camera streams."""
    seed = frame_order_seed(control["selection_seed"], SUITE_ORDER[control["language_global_task_id"] // 10],
                            control["language_global_task_id"] % 10, demo)
    order = frame_control(len(indices), condition=control["arm"], order_seed=seed)
    indices = torch.as_tensor(indices)
    evidence = {"frame_order_seed": seed, "frame_permutation": order.content.tolist(),
                "source_frame_indices": indices[order.content].tolist(),
                "frame_indices": indices[order.positions].tolist(),
                "transform_stage": "both_real_camera_RGB_before_complete_observer_and_bare_source_forward"}
    return order.content, indices[order.positions], evidence


def inspect_diagnostic_contract(value, *, selection, checkpoint, run, asset_root):
    """Bind controls to one frozen terminal checkpoint and its actual correct map."""
    from ember.writer.materialization import file_record, method_metadata
    from ember.writer.evaluation import _inspect_scope

    if selection["arm"] not in CONTROL_ARMS:
        if value is not None:
            raise ValueError("a frozen diagnostic declaration belongs only to video controls")
        return None
    require_control_selection(selection)
    if (not isinstance(value, dict)
            or set(value) != {*DIAGNOSTIC_DECLARATION, "paired_correct_manifest"}
            or any(value.get(key) != expected for key, expected in DIAGNOSTIC_DECLARATION.items())
            or checkpoint.get("macro") != 900):
        raise ValueError("video controls require the explicit frozen terminal900 diagnostic declaration")
    reference = value["paired_correct_manifest"]
    path = Path(reference["path"] if isinstance(reference, dict) else reference).resolve()
    record = file_record(path)
    if isinstance(reference, dict) and reference != record:
        raise ValueError("paired correct manifest changed after diagnostic registration")
    correct = read_json(path)
    if (correct.get("arm") != "correct" or correct.get("writer_checkpoint") != checkpoint
            or correct.get("selection") != dict(selection, arm="correct")
            or correct.get("method") != method_metadata(run)
            or Path(correct["asset_root"]).resolve() != asset_root.resolve()):
        raise ValueError("diagnostic controls must retain the frozen terminal900 correct400 checkpoint and map")
    keys = [(row["suite"], row["task_id"]) for row in correct["tasks"]]
    _inspect_scope(correct, run["source"], keys, "validation", None, True)
    references = {episode["condition_id"] for row in correct["tasks"] for episode in row["episodes"]}
    if len(correct["conditions"]) != 400 or {row["condition_id"] for row in correct["conditions"]} != references:
        raise ValueError("diagnostic reference must retain the complete 400-condition correct bank")
    return {**DIAGNOSTIC_DECLARATION, "paired_correct_manifest": record}
