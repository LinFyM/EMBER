"""Registered video pairing and fixed-case capture for the four-arm diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.preparation import _registered_trajectory_capture
from ember.pi05_eval.preparation import _task_subset_tasks
from ember.pi05_eval_contract import resolve_role_task_keys
from ember.writer import materialization, video_controls
from ember.writer.materialization import planned_episodes, selection_contract
from ember.writer.video_controls import video_task_id


ROOT = Path(__file__).resolve().parents[1]


def test_seen_panel_uses_reserved_videos_in_registered_state_order():
    config = json.loads((ROOT / "configs/conditional_compilation_diagnostics_v1/train_C_video_fm.json").read_text())
    seen = selection_contract(role="development_train", task_ids=(2, 4, 5, 7, 12, 13, 17, 19,
        22, 25, 28, 29, 32, 34, 35, 37), cardinality=1, arm="correct", mode="per_init_ordinal",
        seed=20260911, init_state_ids=(0, 1, 2, 3), video_pool=(46, 47, 48, 49))
    materialization._validate_conditional_selection(seen, config)
    assert materialization.request_init_state_ids(role="development_train", init_state_ids=(0, 1, 2, 3),
                                                   state_count=4) == (0, 1, 2, 3)
    assert [row["teacher_demo_indices"] for row in planned_episodes(seen, 2)] == [[46], [47], [48], [49]]
    with pytest.raises(ValueError, match="registered panel"):
        materialization._validate_conditional_selection(seen | {"seed": 7}, config)


def test_held_video_controls_keep_registered_train_scope(monkeypatch):
    spec = json.loads((ROOT / "configs/conditional_compilation_diagnostics_v1/experiment_spec.json").read_text())
    evaluation = spec["evaluation"]
    held = evaluation["diagnostic_held"]
    common = dict(role="development_train", task_ids=held["task_ids"], cardinality=1,
                  mode="per_init_ordinal", seed=evaluation["video_schedule_seed"],
                  init_state_ids=held["state_ids"], video_pool=held["teacher_demos"])
    correct = selection_contract(arm="correct", **common)
    other = selection_contract(arm="same_task_other", **common)
    wrong = selection_contract(arm="cross_suite_wrong", **common)
    assert correct["task_ids"] == other["task_ids"] == wrong["task_ids"]
    assert {task: video_task_id(wrong, task) for task in held["task_ids"]} == {
        int(task): donor for task, donor in evaluation["selected_video_controls"]["wrong_donor_global_ids"].items()
    }
    with pytest.raises(ValueError, match="selected-checkpoint controls"):
        selection_contract(arm="cross_suite_wrong", **(common | {"seed": 7}))
    declaration = {**video_controls.DIAGNOSTIC_DECLARATION,
                   "checkpoint_macro": 630, "paired_correct_manifest": "/sealed/correct.json"}
    monkeypatch.setattr(video_controls, "_inspect_paired_correct",
                        lambda *_args, **_kwargs: ({"path": "/sealed/correct.json", "bytes": 1}, {}))
    for control in (other, wrong):
        inspected = video_controls.inspect_diagnostic_contract(
            declaration, selection=control, checkpoint={"macro": 630}, run={}, asset_root=ROOT
        )
        assert inspected["checkpoint_macro"] == 630


def test_registered_trajectory_capture_selects_only_fixed_cases(tmp_path):
    subset_path = str(tmp_path / "subset.json")
    manifest_path = tmp_path / "capture.json"
    manifest = {
        "schema_version": "ember_pi05_registered_trajectory_capture_v1",
        "task_subset_selection": subset_path, "mode": "compact",
        "full_conditions": [{"suite": "libero_spatial", "task_id": 0, "init_state_id": 25}],
        "stage_predicates": True, "training_gradient_use": False,
        "checkpoint_selection_use": False, "validation_use": False, "test_use": False,
    }
    manifest_path.write_text(json.dumps(manifest))
    task = SimpleNamespace(suite="libero_spatial", task_id=0, init_state_ids=tuple(range(50)))
    args = SimpleNamespace(role="development_train", trajectory_capture_selection=manifest_path,
                           occupancy_capture_selection=None, capture_stage_predicates=False)
    capture, stage = _registered_trajectory_capture(
        args, (task,), tmp_path / "output", {"selection_path": subset_path}
    )
    assert capture["full_conditions"] == manifest["full_conditions"]
    assert stage["full_conditions_only"] is True
    manifest_path.write_text(json.dumps(manifest | {"full_conditions": [
        {"suite": "libero_spatial", "task_id": 0, "init_state_id": 50}]}))
    with pytest.raises(Pi05EvaluationError, match="capture selection changed"):
        _registered_trajectory_capture(args, (task,), tmp_path / "output",
                                       {"selection_path": subset_path})


@pytest.mark.parametrize("panel,mode,count", [("diagnostic_held", "formal", 50), ("seen", "screen", 4)])
def test_full_registered_eval_subsets_keep_train_role(tmp_path, panel, mode, count):
    spec = json.loads((ROOT / "configs/conditional_compilation_diagnostics_v1/experiment_spec.json").read_text())
    protocol = json.loads((ROOT / spec["protocol"]["parent"]).read_text())
    keys = resolve_role_task_keys(protocol, "development_train")
    installed = tuple(SimpleNamespace(suite=suite, task_id=task_id,
                                     init_state_ids=tuple(range(count))) for suite, task_id in keys)
    selected_ids = spec["evaluation"][panel]["task_ids"]
    ordinals = [index for index, (suite, task_id) in enumerate(keys)
                if 10 * ("libero_spatial", "libero_object", "libero_goal", "libero_10").index(suite)
                + task_id in selected_ids]
    path = tmp_path / "subset.json"
    manifest = {
        "schema_version": "ember_pi05_task_subset_selection_v1",
        "role": "development_train", "mode": mode, "state_count": count,
        "init_state_ids": list(range(count)), "task_ordinals": ordinals,
        "global_task_ids": selected_ids,
        "tasks": [{"global_task_id": gid, "suite": keys[index][0], "task_id": keys[index][1]}
                  for index, gid in zip(ordinals, selected_ids, strict=True)],
        "outcome_dependence": False, "validation_use": False, "test_use": False,
    }
    path.write_text(json.dumps(manifest))
    args = SimpleNamespace(role="development_train", mode=mode, state_count=count,
                           init_state_ids=None, occupancy_capture_selection=None,
                           task_subset_selection=path)
    selected, record = _task_subset_tasks(args, installed, adapter_kind="static_task_lora")
    assert len(selected) == len(selected_ids)
    assert record["global_task_ids"] == selected_ids
