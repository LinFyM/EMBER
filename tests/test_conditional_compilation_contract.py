"""Registered video pairing and fixed-case capture for the four-arm diagnostic."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.preparation import _registered_trajectory_capture
from ember.pi05_eval.preparation import _task_subset_tasks
from ember.pi05_eval_contract import resolve_role_task_keys
from ember.writer import learning_data, materialization, video_controls
from ember.writer.learning_data import WriterTrainingData
from ember.writer.materialization import planned_episodes, selection_contract
from ember.writer.relational_contract import registered_stage1_bank_panel
from ember.writer.training import _config
from ember.writer.video_controls import video_task_id


ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "configs/relational_support_causality_v1/experiment_spec.json").read_text())


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
        args, (task,), tmp_path / "output", {"selection_path": subset_path}, ROOT
    )
    assert capture["full_conditions"] == manifest["full_conditions"]
    assert stage["full_conditions_only"] is True
    manifest_path.write_text(json.dumps(manifest | {"full_conditions": [
        {"suite": "libero_spatial", "task_id": 0, "init_state_id": 50}]}))
    with pytest.raises(Pi05EvaluationError, match="capture selection changed"):
        _registered_trajectory_capture(args, (task,), tmp_path / "output",
                                       {"selection_path": subset_path}, ROOT)


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


# Frozen relation-support task pools reuse the conditional Writer execution owner.
def _arm_config(arm):
    return _config(ROOT / f"configs/relational_support_causality_v1/train_{arm}.json")


def test_six_fresh_configs_only_admit_their_registered_fit28(tmp_path):
    held = set(SPEC["protocol"]["diagnostic_held8"])
    for arm in SPEC["arms"]:
        config = _arm_config(arm["id"])
        assert config["data"]["task_ids"] == arm["fit28"]
        assert len(config["data"]["task_ids"]) == 28
        assert not held.intersection(config["data"]["task_ids"])
        changed = deepcopy(config)
        changed["data"]["task_ids"][0] = SPEC["protocol"]["diagnostic_held8"][0]
        path = tmp_path / "invalid.json"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="recipe"):
            _config(path)


def test_common26_events_and_same_pool_language_video_are_identical(monkeypatch):
    ids = set(task for arm in SPEC["arms"] for task in arm["fit28"])
    tasks = {task: SimpleNamespace(suite="libero_90" if task >= 40 else "target",
                                  authority=object(), episode_lengths=(51,) * 50) for task in ids}
    monkeypatch.setattr(learning_data, "load_learning_tasks", lambda *_a, **_kw: tasks_for(_a[1], tasks))
    monkeypatch.setattr(learning_data, "RawTeacherVideoStore", lambda *_a, **_kw:
                        SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(learning_data, "FunctionalQueryDataset", lambda authorities, **_kw:
                        SimpleNamespace(task_episode_rows={task: {demo: range(50) for demo in range(46)}
                                                              for task in ids}, close=lambda: None))
    plans = {}
    for arm in SPEC["arms"]:
        config = _arm_config(arm["id"])
        data = WriterTrainingData(ROOT, config["data"], camera_view="agentview",
                                  use_videos=arm["parameterization"] == "video_writer")
        plans[arm["id"]] = data.event_plan()
        assert len(plans[arm["id"]]["groups"]) == 1260
        assert len(plans[arm["id"]]["events"]) == 5040
        data.close()
    common = set(SPEC["protocol"]["common26"])
    baseline = plans["C_S00"]
    assert plans["B_S11"] == plans["C_S11"]
    for arm in SPEC["arms"]:
        plan = plans[arm["id"]]
        if arm["pool"] == "S00":
            assert plan == baseline
        for group, reference in zip(plan["groups"], baseline["groups"], strict=True):
            tasks_now = [plan["events"][index]["task"] for index in group]
            tasks_before = [baseline["events"][index]["task"] for index in reference]
            assert [task for task in tasks_now if task in common] == [task for task in tasks_before if task in common]
        now = {event["task"]: [] for event in plan["events"]}
        before = {event["task"]: [] for event in baseline["events"]}
        for event in plan["events"]:
            now[event["task"]].append(event)
        for event in baseline["events"]:
            before[event["task"]].append(event)
        assert all(now[task] == before[task] for task in common)


def tasks_for(task_ids, tasks):
    return {task: tasks[task] for task in task_ids}


def test_support_stage1_uses_original_50_video_schedule_prefix():
    arm = SPEC["arms"][0]
    support = SPEC["evaluation"]["support_knowledge_panels"]
    selection = materialization.selection_contract(
        role="nonheld_meta", task_ids=arm["support_eval_global_ids"], cardinality=1,
        arm="correct", mode="per_init_ordinal", seed=SPEC["evaluation"]["video_schedule_seed"],
        init_state_ids=support["states"], video_pool=support["teacher_demos"])
    materialization._validate_conditional_selection(selection, _arm_config(arm["id"]))
    episodes = materialization.planned_episodes(selection, 58)
    assert len(episodes) == 20
    from ember.expert_manifold.video_schedule import reference_demo_index
    expected=[reference_demo_index(SPEC["evaluation"]["video_schedule_seed"],"libero_90",18,state,
                                   demo_count=50,sampling_mode="without_replacement") for state in range(50)]
    assert [row["teacher_demo_indices"][0] for row in episodes] == expected[:20]
    assert len(set(expected)) == 50
    assert episodes[0]["init_state_id"] == 0


def test_support_evaluator_subset_is_explicit(tmp_path):
    arm = SPEC["arms"][0]
    ids = SPEC["evaluation"]["source_reference"]["support_global_ids"]
    installed = tuple(SimpleNamespace(suite="libero_90", task_id=task - 40,
                                      init_state_ids=tuple(range(20))) for task in ids)
    selected_ids = arm["support_eval_global_ids"]
    ordinals = [ids.index(task) for task in selected_ids]
    manifest = {
        "schema_version": "ember_pi05_task_subset_selection_v1", "role": "nonheld_meta",
        "mode": "screen", "state_count": 20, "init_state_ids": list(range(20)),
        "task_ordinals": ordinals, "global_task_ids": selected_ids,
        "tasks": [{"global_task_id": task, "suite": "libero_90", "task_id": task - 40}
                  for task in selected_ids],
        "study_spec": "configs/relational_support_causality_v1/experiment_spec.json",
        "arm_id": arm["id"], "outcome_dependence": False,
        "validation_use": False, "test_use": False,
    }
    path = tmp_path / "subset.json"
    path.write_text(json.dumps(manifest))
    args = SimpleNamespace(role="nonheld_meta", mode="screen", state_count=20,
                           init_state_ids=None, occupancy_capture_selection=None,
                           task_subset_selection=path, trajectory_capture_selection=tmp_path/"capture.json")
    selected, record = _task_subset_tasks(args, installed, adapter_kind="static_task_lora")
    assert [task.task_id for task in selected] == [task - 40 for task in selected_ids]
    assert record["diagnostic_subset"] == "registered_relational_support_tasks"


# Exact stage1 bank admission without loading models or consuming GPU work.
STAGE_RUN = Path(SPEC["outputs"]["planned_run_root"])

def _selection(panel):
    support = panel['kind'] == 'support_correct'
    return materialization.selection_contract(
        role='nonheld_meta' if support else 'development_train',
        task_ids=panel['task_ids'], cardinality=1,
        arm='same_task_other' if panel['kind'] == 'target_other' else 'correct',
        mode='per_init_ordinal', seed=SPEC['evaluation']['video_schedule_seed'],
        init_state_ids=panel['state_ids'], video_pool=list(range(50)))


def _config_for(panel):
    return _config(ROOT/'configs/relational_support_causality_v1'/f"train_{panel['arm']}.json")


def _panel(arm, kind):
    return next(row for row in SPEC['evaluation']['stage1']['panels']
                if row['arm'] == arm and row['kind'] == kind)


def test_exact_sixteen_stage1_bank_requests_and_outputs():
    banks = [row for row in SPEC['evaluation']['stage1']['panels'] if row['arm'] != 'Source']
    assert len(banks) == 16
    for panel in banks:
        selection = _selection(panel)
        config = _config_for(panel)
        checkpoint = STAGE_RUN/'training'/panel['arm']/'checkpoints/macro_00001260'
        output = STAGE_RUN/'materialization'/panel['id']
        materialization._validate_conditional_selection(selection, config)
        assert registered_stage1_bank_panel(config, selection,
            checkpoint=checkpoint, output=output)['id'] == panel['id']
        assert sum(len(materialization.planned_episodes(selection,gid))
                   for gid in panel['task_ids']) == panel['expected_rows']
        assert selection['video_pool'] == list(range(50))
    assert sum(row['expected_rows'] for row in banks) == 1280


@pytest.mark.parametrize('change', ['deferred_task','support_state','wrong_control','wrong_arm',
                                   'wrong_seed','wrong_pool','wrong_checkpoint','wrong_output'])
def test_stage1_bank_scope_rejects_deferred_and_wrong_identity(change):
    panel = _panel('C_S00','support_correct' if change == 'support_state' else 'target_correct')
    selection = deepcopy(_selection(panel))
    config = _config_for(panel)
    checkpoint = STAGE_RUN/'training'/panel['arm']/'checkpoints/macro_00001260'
    output = STAGE_RUN/'materialization'/panel['id']
    if change == 'deferred_task':
        selection['task_ids'].append(0)
    elif change == 'support_state':
        selection['init_state_ids'].append(20)
    elif change == 'wrong_control':
        selection['arm'] = 'cross_suite_wrong'
    elif change == 'wrong_arm':
        config = _config(ROOT/'configs/relational_support_causality_v1/train_C_S11.json')
    elif change == 'wrong_seed':
        selection['seed'] += 1
    elif change == 'wrong_pool':
        selection['video_pool'] = list(range(20))
    elif change == 'wrong_checkpoint':
        checkpoint = STAGE_RUN/'training'/panel['arm']/'checkpoints/macro_00001050'
    else:
        output = STAGE_RUN/'materialization'/f"{panel['id']}_extra"
    with pytest.raises(ValueError, match='stage1|registered'):
        registered_stage1_bank_panel(config, selection, checkpoint=checkpoint, output=output)


def test_json_request_admits_registered_nonheld20_only_before_gpu(monkeypatch):
    panel = _panel('C_S00','support_correct')
    request = {'checkpoint':str(STAGE_RUN/'training/C_S00/checkpoints/macro_00001260'),
               'output':str(STAGE_RUN/'materialization'/panel['id']),
               'role':'nonheld_meta','task_ids':panel['task_ids'],'k':1,
               'arm':'correct','selection_mode':'per_init_ordinal',
               'video_pool':list(range(50)),'state_count':20,
               'seed':SPEC['evaluation']['video_schedule_seed']}
    monkeypatch.setattr(materialization,'_materialize_batch',
                        lambda **kwargs: kwargs['requests'])
    normalized = materialization.materialize_requests(asset_root=ROOT,requests=[request],device=None)
    assert normalized[0]['selection']['init_state_ids'] == list(range(20))
    assert normalized[0]['selection']['video_pool'] == list(range(50))
    with pytest.raises(ValueError,match='stage1 support'):
        materialization.materialize_requests(asset_root=ROOT,
            requests=[request|{'output':str(STAGE_RUN/'materialization/deferred_support')}],device=None)
    with pytest.raises(ValueError,match='registered nonheld20'):
        materialization.request_init_state_ids(role='nonheld_meta',state_count=20)


def test_goal21_other_requires_same_arm1260_reference_declaration(monkeypatch):
    panel = _panel('C_S00','target_other')
    selection = _selection(panel)
    config = _config_for(panel)
    record = {'path':'/sealed/correct100/manifest.json','bytes':123}
    monkeypatch.setattr(video_controls,'_inspect_stage1_goal_correct',lambda *_args,**_kwargs:record)
    declaration = {**video_controls.DIAGNOSTIC_DECLARATION,
                   'stage1_panel_id':panel['id'],'checkpoint_macro':1260,
                   'paired_correct_manifest':record}
    result = video_controls.inspect_diagnostic_contract(declaration,selection=selection,
        checkpoint={'macro':1260},run={'config':config},asset_root=ROOT)
    assert result['paired_correct_manifest'] == record
    with pytest.raises(ValueError,match='stage1 Goal21'):
        video_controls.inspect_diagnostic_contract(declaration|{'stage1_panel_id':'wrong'},
            selection=selection,checkpoint={'macro':1260},run={'config':config},asset_root=ROOT)
    with pytest.raises(ValueError,match='stage1 Goal21'):
        video_controls.inspect_diagnostic_contract(declaration,
            selection=selection,checkpoint={'macro':1050},run={'config':config},asset_root=ROOT)
