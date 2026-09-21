"""CPU contract tests; pure production methods are loaded without LeRobot.

A normal environment also executes the full trainer-parser integration test.
No source weights, robot data or GPU are needed by the other tests.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ember.writer.auxiliary_pairing import (
    DECLARATION, REFERENCE_CONFIG, audit_actual_exposures,
    auxiliary_validation_decision, declared_dynamic_episode,
    measurement_arms, require_matched_config, selection_plan,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "configs/libero_24_8_8_coverage_v1/writer_aux_cross_episode.json"


def configs():
    return json.loads((ROOT / REFERENCE_CONFIG).read_text()), json.loads(CANDIDATE.read_text())


def production_function(path, name, namespace=None, *, owner=None):
    """Compile the exact pure function from its source, bypassing heavy imports."""
    module = ast.parse((ROOT / path).read_text())
    nodes = module.body
    if owner:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner).body
    node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    env = dict(namespace or {})
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(ROOT / path), "exec"), env)
    return env[name]


def history(scores):
    return [{"step": (i + 1) * 200, "episodes": 400, "complete": True, "successes": score}
            for i, score in enumerate(scores)]


def test_candidate_changes_only_registered_scientific_leaf():
    ref, cfg = configs()
    require_matched_config(cfg, ref)
    assert declared_dynamic_episode(cfg, ROOT) == "cross_episode"
    assert declared_dynamic_episode(ref, ROOT) == "same_video"
    assert cfg["data"]["task_ids"] == ref["data"]["task_ids"]
    assert cfg["model"] == ref["model"]
    assert cfg["source"] == ref["source"]
    assert cfg["optimization"] == ref["optimization"]


@pytest.mark.parametrize("section,key,value", [
    ("optimization", "lr", 1e-5), ("optimization", "teaching_weight", 0),
    ("optimization", "teaching_flow_time", .5), ("optimization", "teaching_prefix_steps", 50),
    ("optimization", "tail_start_update", 900), ("optimization", "betas", [.9,.999]),
    ("data", "task_ids", [1,2,3,4]), ("data", "video_demos", list(range(50))),
    ("data", "teaching_episode", "same_video"), ("data", "action_start_offset", 0),
    ("data", "tasks_per_update", 36), ("data", "queries_per_task", 28),
    ("data", "maximum_updates", 1800), ("model", "camera_view", "dual"),
    ("model", "factor_hidden_width", 256), ("observer", "frame_chunk", 16),
    ("source", "checkpoint", "merged_mtbc"), ("training_control", "validation_interval", 100),
])
def test_unregistered_scientific_changes_rejected(section, key, value):
    ref, cfg = configs()
    cfg[section][key] = value
    with pytest.raises(ValueError):
        require_matched_config(cfg, ref)


@pytest.mark.parametrize("field", ["experiment", "continuation", "phase_continuation"])
def test_bad_declaration_and_parent_rejected(field):
    ref, cfg = configs()
    cfg[field] = {"kind": "unregistered"}
    with pytest.raises(ValueError):
        require_matched_config(cfg, ref)


def test_qualification_and_diagnostic_guards():
    ref, cfg = configs()
    cfg["evidence"]["qualification"]["tasks"][0] = 0
    with pytest.raises(ValueError):
        require_matched_config(cfg, ref)
    _, cfg = configs()
    cfg["evidence"]["test_use"] = True
    with pytest.raises(ValueError):
        require_matched_config(cfg, ref)
    _, cfg = configs()
    cfg["evidence"]["supervised_validation"]["optimizer_updates"] = [0,200]
    with pytest.raises(ValueError):
        require_matched_config(cfg, ref)


def test_actual_dynamic_gate_is_opt_in_and_retains_low_lr_legacy():
    check = production_function("src/ember/writer/training.py", "_validate_dynamic_schedule",
                                {"declared_dynamic_episode": declared_dynamic_episode, "REPO_ROOT": ROOT})
    ref, cfg = configs()
    check(ref)
    check(cfg)
    raw_cross = deepcopy(ref)
    raw_cross["data"]["teaching_episode"] = "cross_episode"
    with pytest.raises(ValueError):
        check(raw_cross)
    low_lr = deepcopy(ref)
    low_lr["phase_continuation"] = {"kind": "coverage_writer_low_lr_repair_v1"}
    check(low_lr)  # Original continuation validator still owns its detailed check.


def test_full_parser_when_project_dependencies_available():
    pytest.importorskip("lerobot", reason="full project runtime is required for trainer integration")
    from ember.writer.training import _config
    assert _config(CANDIDATE)["data"]["teaching_episode"] == "cross_episode"
    assert _config(ROOT / REFERENCE_CONFIG)["data"]["teaching_episode"] == "same_video"
    legacy = ROOT / "configs/pi05_writer_cross_episode_ablation.json"
    if legacy.exists():
        assert _config(legacy)["data"]["teaching_episode"] == "cross_episode"


@pytest.mark.parametrize("length", [6, 10, 35, 36, 52, 118, 520])
def test_real_aux_sampler_preserves_noise_and_valid_future_controls(length):
    draw = production_function("src/ember/writer/learning_data.py", "_teaching_event",
                               {"np": np}, owner="WriterTrainingData")
    same = SimpleNamespace(config={"teaching_seed": 20260919, "teaching_episode": "same_video"},
                           action_pool=tuple(range(46)), seed=7)
    cross = SimpleNamespace(config={**same.config, "teaching_episode": "cross_episode"},
                            action_pool=same.action_pool, seed=7)
    for occurrence in range(50):
        teacher = occurrence % 46
        a = draw(same, 5, occurrence, teacher, (length,) * 50)
        b = draw(cross, 5, occurrence, teacher, (length,) * 50)
        assert a["action_demos"] == [teacher] * 7
        assert len(set(b["action_demos"])) == 1
        assert b["action_demos"][0] != teacher
        assert b["action_demos"][0] in range(46)
        assert b == draw(cross, 5, occurrence, teacher, (length,) * 50)
        assert a["policy_rng_seed"] == b["policy_rng_seed"]
        assert a["action_frames"] == b["action_frames"]  # Equal lengths in this fixture.
        assert b["action_start_indices"] == [p + 1 for p in b["action_frames"]]
        assert all(p % 5 == 0 and p + 5 < length for p in b["action_frames"])
        if len(range(0, length - 5, 5)) >= 7:
            assert len(set(b["action_frames"])) == 7


def test_loss_supervises_different_horizon_without_padded_action_dimensions():
    loss = production_function("src/ember/writer/function_credit.py", "mean_velocity_loss")
    pred = torch.zeros(2, 50, 32)
    target = torch.zeros_like(pred)
    pred[:, 5:, :7] = 1
    pred[:, :, 7:] = 100
    assert loss(pred, target, 7).item() == pytest.approx(.9)
    assert loss(pred, target, 7, prefix_steps=5).item() == 0
    pred[:, :5, :7] = 2
    assert loss(pred, target, 7, prefix_steps=5).item() == 4
    assert loss(pred, target, 7).item() == pytest.approx(1.3)


def test_loss_denominators_keep_main_and_aux_separate():
    # Each task contributes 1/4 (mean_21 main + 1/3 mean_7 aux).
    main = torch.arange(21, dtype=torch.float64)
    aux = torch.arange(7, dtype=torch.float64)
    unsharded = .25 * (main.mean() + aux.mean() / 3)
    split_main = sum(.25 * len(x) / 21 * x.mean() for x in main.split(8))
    split_aux = sum(.25 / 3 * len(x) / 7 * x.mean() for x in aux.split(3))
    assert (split_main + split_aux).item() == pytest.approx(unsharded.item())
    assert ((main.sum() + aux.sum()) / 28).item() != pytest.approx((main.mean() + aux.mean()/3).item())


def test_original_decline_curve_and_new_record_behavior():
    old = [110,92,115,87,117,80,97,111,92]
    assert auxiliary_validation_decision(history(old))["reason"] == "sustained_decline"
    for scores in ([100], [100,90], [100,90,110], [100,90,110,111]):
        assert not auxiliary_validation_decision(history(scores))["stop"]


def test_resource_stop_is_distinct_from_scientific_convergence():
    scores = [120,119,100,118,100,119,118]
    result = auxiliary_validation_decision(history(scores))
    assert result["stop"] and result["reason"] == "resource_review_no_progress"
    assert result["resource_review"]
    assert result["nodes_without_strict_new_best"] == 6
    scores[-1] = 121
    assert not auxiliary_validation_decision(history(scores))["stop"]


@pytest.mark.parametrize("scores", [[100,100.0], [True], [401], [-1]])
def test_bad_history_scores_rejected(scores):
    with pytest.raises(ValueError):
        auxiliary_validation_decision(history(scores))


def test_history_must_be_contiguous_complete400():
    rows = history([100,110])
    rows[-1]["step"] = 600
    with pytest.raises(ValueError):
        auxiliary_validation_decision(rows)
    rows = history([100])
    rows[0]["complete"] = False
    with pytest.raises(ValueError):
        auxiliary_validation_decision(rows)


@pytest.mark.parametrize("score,expected", [
    (0,[]),(117,[]),(118,["same_task_other","cross_suite_wrong"]),
    (155,["same_task_other","cross_suite_wrong"]),
    (156,["same_task_other","cross_suite_wrong","shuffled","reversed"]),
    (400,["same_task_other","cross_suite_wrong","shuffled","reversed"]),
])
def test_measurement_budgets(score, expected):
    assert measurement_arms(score) == expected


def test_selection_waits_for_stop_and_all_tied_other_panels():
    assert selection_plan(history([100,110]))["status"] == "continue_training"
    rows = history([160,160,150,149,148])
    result = selection_plan(rows)
    assert result["status"] == "awaiting_tied_other400"
    assert result["required_other_steps"] == [200,400]
    assert selection_plan(rows, {200:150})["required_other_steps"] == [400]
    result = selection_plan(rows, {200:150,400:151})
    assert result["selected_macro"] == 400
    assert result["replace_original_selected"]
    assert not result["test_allowed"]
    assert selection_plan(rows, {200:151,400:151})["selected_macro"] == 200
    with pytest.raises(ValueError):
        selection_plan(rows, {200:150,400:151,600:400})


def test_actual_reference_exposure_audit(tmp_path):
    event = {"task":5,"occurrence":0,"teacher_demo":2,"query_seed":8,
             "action_demos":[1]*21,"action_frames":[5]*21,"action_start_indices":[6]*21,
             "policy_rng_seed":9,"teaching":{"policy_rng_seed":10,"action_demos":[2]*7,"action_frames":[0]*7}}
    row = {**{k:v for k,v in event.items() if k not in {"teaching","teacher_demo"}},
           "step":1,"video_demos":[2],"teaching_policy_rng_seed":10,
           "teaching_action_demos":[2]*7,"teaching_action_frames":[0]*7}
    path = tmp_path/'exposures.jsonl'
    path.write_text(json.dumps(row)+'\n')
    assert audit_actual_exposures(path, {(1,5):event}) == 1
    path.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
    with pytest.raises(ValueError):
        audit_actual_exposures(path, {(1,5):event})


def panel_reader():
    spec = importlib.util.spec_from_file_location("writer_aux_pairing_cli", ROOT / "scripts/writer_aux_pairing.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def make_panel(study, *, step=200, arm="correct"):
    _, config = configs()
    selected = config["evidence"]["qualification"]
    rows = [{"init_state_id": s, "success": s < 10, "split_role": "validation",
             "horizon_writer_lora": {"global_task_id": t, "K":1,"selection_seed":20260911,
                                     "selection_mode":"per_init_ordinal","video_ordinal":s}}
            for t in selected["tasks"] for s in range(50)]
    contract = {"contract_reference":"fixture", "role":"validation", "mode":"formal"}
    result = {"schema_version":"ember_pi05_target_eval_results_v2","contract_reference":"fixture",
              "overall":{"episodes":400,"successes":80},"rows":rows}
    done = {"contract_reference":"fixture","return_codes":{"0":0},
            "queue":{"completed_rows":400,"successes":80}}
    manifest = {"arm":arm,"writer_checkpoint":{"macro":step},
                "selection":{"evaluation_role":"validation","task_ids":selected["tasks"],
                             "init_state_ids":list(range(50)),"seed":20260911}}
    suffix = "" if arm == "correct" else "_other"
    mat_suffix = "_correct" if arm == "correct" else "_other"
    eval_root = study / 'evaluation' / f'writer_{step:08d}{suffix}'
    mat_root = study / 'materialized' / f'writer_{step:08d}{mat_suffix}'
    for directory in (eval_root, mat_root):
        directory.mkdir(parents=True)
    files = {eval_root/'results.json':result, eval_root/'run_contract.json':contract,
             eval_root/'launcher_completion.json':done, mat_root/'manifest.json':manifest}
    for path, data in files.items():
        path.write_text(json.dumps(data))
    return config, files


def test_complete_panel_reader_accepts_only_full_artifacts(tmp_path):
    module = panel_reader()
    config, _ = make_panel(tmp_path)
    assert module.load_panel(tmp_path, 200, config)["overall"]["successes"] == 80
    (tmp_path/'evaluation/writer_00000200/launcher_completion.json').unlink()
    with pytest.raises(FileNotFoundError):
        module.load_panel(tmp_path,200,config)


@pytest.mark.parametrize("case", ["worker_failure","bool_exit","wrong_count","duplicate","string_success",
                                  "wrong_checkpoint","wrong_arm","wrong_role","missing_reference","reused_video"])
def test_panel_corruption_cannot_select_checkpoint(tmp_path, case):
    module = panel_reader()
    config, files = make_panel(tmp_path)
    result_path = tmp_path/'evaluation/writer_00000200/results.json'
    done_path = tmp_path/'evaluation/writer_00000200/launcher_completion.json'
    mat_path = tmp_path/'materialized/writer_00000200_correct/manifest.json'
    contract_path = tmp_path/'evaluation/writer_00000200/run_contract.json'
    if case == 'worker_failure': files[done_path]['return_codes']['0'] = 1
    elif case == 'bool_exit': files[done_path]['return_codes']['0'] = False
    elif case == 'wrong_count': files[done_path]['queue']['completed_rows'] = 399
    elif case == 'duplicate': files[result_path]['rows'][-1] = files[result_path]['rows'][0]
    elif case == 'string_success': files[result_path]['rows'][0]['success'] = 'false'
    elif case == 'wrong_checkpoint': files[mat_path]['writer_checkpoint']['macro'] = 400
    elif case == 'wrong_arm': files[mat_path]['arm'] = 'cross_suite_wrong'
    elif case == 'wrong_role': files[contract_path]['role'] = 'test'
    elif case == 'missing_reference': files[result_path]['contract_reference'] = None
    elif case == 'reused_video': files[result_path]['rows'][0]['horizon_writer_lora']['video_ordinal'] = 1
    for path, data in files.items(): path.write_text(json.dumps(data))
    with pytest.raises(ValueError): module.load_panel(tmp_path,200,config)


def metadata_sampler_type():
    """Bind unchanged canonical pure sampler methods, with the real seed mixer."""
    mix = production_function('src/ember/writer/functional.py', '_splitmix64',
                              {'_UINT64_MASK': (1 << 64) - 1})
    seed = production_function('src/ember/writer/functional.py', 'task_logical_batch_policy_rng_seed',
                               {'_splitmix64':mix, 'WriterModelError':ValueError})
    query = production_function('src/ember/writer/learning_data.py', '_episode_queries', {'np':np})
    env = {'np':np, 'task_logical_batch_policy_rng_seed':seed, '_episode_queries':query,
           'EVENT_SCHEMA':'video_teaching_joint_query_events_v1', 'MAXIMUM_UPDATES':2100}
    names = ('_validate_config','_build_groups','_build_events','_teaching_event')
    return type('CanonicalMetadataSampler', (), {
        name:production_function('src/ember/writer/learning_data.py',name,env,owner='WriterTrainingData')
        for name in names})


def test_metadata_preflight_replays_main_and_aux_events_without_hdf5():
    from ember.writer.auxiliary_pairing import audit_events
    ref, cfg = configs()
    lengths = {task:[6 + ((task + demo) % 50) * 5 for demo in range(50)] for task in cfg['data']['task_ids']}
    audit, expected = audit_events(ref, cfg, lengths, updates=37, sampler_type=metadata_sampler_type())
    assert audit['conditions'] == 148
    assert audit['teacher_main_events_equal'] and audit['auxiliary_noise_equal']
    assert audit['auxiliary_teacher_excluded'] and audit['future_five_actions_valid']
    assert not audit['actual_reference_exposures_checked']
    assert len(expected) == 148
    assert all(event['teacher_demo'] not in event['action_demos'] for event in expected.values())
    bad = dict(lengths)
    bad.pop(next(iter(bad)))
    with pytest.raises(ValueError):
        audit_events(ref,cfg,bad,updates=37,sampler_type=metadata_sampler_type())
