"""Actual frozen sources and bounded CPU consumers for the seen-task panel."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from ember.operator_writer import bank, scope
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.scene import inspect_registered_scenes, scene_path
from ember.pi05_eval_contract import (inspect_source_checkpoint, load_evaluation_authorities,
                                      inspect_installed_target_tasks, resolve_role_task_keys)
from ember.pi05_source_checkpoint import read_json, write_json_atomic


ASSET = Path("/data1/user/ymdai/projects/EMBER")


def test_actual_sources_and_exact_four_state_video_panel(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBER_LIBERO_ASSETS_ROOT", str(
        ASSET / "data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"))
    registered = scope.registration()
    spec = read_json(bank.CONTINUATION1800_FROZEN_SPEC_PATH)
    authorities = load_evaluation_authorities(
        ASSET / spec["source"]["evaluation_config"], ASSET)
    keys = scope.task_keys(authorities.protocol, authorities.meta_protocol)
    assert keys == resolve_role_task_keys(authorities.protocol, scope.ROLE,
                                          meta_protocol=authorities.meta_protocol)
    installed, _ = inspect_installed_target_tasks(
        authorities, role=scope.ROLE, state_count=4,
        libero_config_dir=tmp_path / "libero_config")
    assert len(installed) == 36
    assert [(row.suite, row.task_id) for row in installed] == list(keys)
    assert all(row.split_role == "train" and row.init_state_ids == (0, 1, 2, 3)
               for row in installed)
    tasks, conditions = scope.task_rows(ASSET, spec)
    assert len(tasks) == 36 and len(conditions) == 144
    assert [(row["suite"], row["task_id"]) for row in tasks] == list(keys)
    for task in tasks:
        assert task["split_role"] == "train"
        episodes = task["episodes"]
        assert [row["init_state_id"] for row in episodes] == list(scope.STATES)
        assert len({row["teacher_demo_indices"][0] for row in episodes}) == 4
        assert [row["video_ordinal"] for row in episodes] == list(scope.STATES)
    source_checkpoint = ASSET / spec["source"]["checkpoint"]
    source = inspect_source_checkpoint(authorities, source_checkpoint.parent.parent,
                                       source_checkpoint, evaluation_mode="formal")
    mt_spec = read_json(bank.SEALED_SPEC_PATH)
    mt, mt_run, _ = bank._mt_source(mt_spec, source)
    assert str(mt) == registered["checkpoint_mt"]
    assert mt_run["git"]["commit"].startswith(registered["training_git_mt"])
    t = bank.inspect_training_source(spec, Path(registered["checkpoint_t"]), "T",
                                     sealed_evaluation=True)
    assert t["git"]["commit"] == registered["training_git_t"]
    with pytest.raises(ValueError):
        bank.inspect_training_source(spec, Path(registered["checkpoint_t"]), "U",
                                     sealed_evaluation=True)
    with pytest.raises(ValueError, match="fixed T1800"):
        bank.materialize("T", Path(registered["checkpoint_t"]).with_name("macro_00001710"),
                         ASSET, None, seen_task=True)


def test_scope_rejects_crossed_role_and_scene_registry(tmp_path):
    registered = scope.registration()
    full = read_json(scope.CAPTURE_PATH)["full_conditions"]
    assert full == [{"suite": suite, "task_id": task, "init_state_id": 32}
                    for suite, task in scope.task_keys_from_ids(registered["global_task_ids"])]
    tasks = [SimpleNamespace(suite=row["suite"], task_id=row["task_id"])
             for row in full]
    assert scope.capture_registration(tasks)[0] == full
    with pytest.raises(ValueError, match="full capture"):
        scope.capture_registration(tasks[:-1])
    rows = []
    for task in tasks:
        for state in scope.STATES:
            path = scene_path(tmp_path, vars(task), state)
            path.write_bytes(b"registered")
            rows.append({"suite": task.suite, "task_id": task.task_id,
                         "state": state, "path": str(path), "bytes": path.stat().st_size})
    write_json_atomic(tmp_path / "manifest.json", {
        "schema_version": "ember_operator_seen_task_scenes_v1", "seed": 7,
        "dummy_steps": 10, "scenes": rows})
    assert len(inspect_registered_scenes(tmp_path, [vars(task) for task in tasks],
                                         states=scope.STATES,
                                         schema="ember_operator_seen_task_scenes_v1")["scenes"]) == 144
    with pytest.raises(ValueError, match="registry"):
        inspect_registered_scenes(tmp_path, [vars(task) for task in tasks],
                                  states=(0, 1, 2, 3),
                                  schema="ember_operator_seen_task_scenes_v1")


def test_registered_144_capture_routes_actual_cases_and_rejects_other_scope(tmp_path, monkeypatch):
    registered = scope.registration() | {"run_root": str(tmp_path)}
    monkeypatch.setattr(scope, "registration", lambda: registered)
    full = read_json(scope.CAPTURE_PATH)["full_conditions"]
    tasks = [SimpleNamespace(suite=row["suite"], task_id=row["task_id"],
                             init_state_ids=scope.STATES) for row in full]
    bank_path = tmp_path / "T/banks/1800/manifest.json"
    bank_path.parent.mkdir(parents=True)
    write_json_atomic(bank_path, {"kind": bank.KIND, "mode": "T",
                                  "evaluation_scope": {"path": str(scope.PATH)}})
    args = SimpleNamespace(static_task_lora_manifest=bank_path, role=scope.ROLE,
                           mode="formal")
    output = tmp_path / "T/evaluation/correct144"
    capture, stage = bank.registered_capture(args, tasks, output, scope.CAPTURE_PATH,
                                               read_json(scope.CAPTURE_PATH), None)
    assert len(capture["full_conditions"]) == 36
    assert capture["full_conditions"] == full and stage["full_conditions_only"] is False
    with pytest.raises(Pi05EvaluationError, match="capture scope"):
        bank.registered_capture(args, tasks, tmp_path / "T/evaluation/correct400",
                                scope.CAPTURE_PATH, read_json(scope.CAPTURE_PATH), None)
    with pytest.raises(Pi05EvaluationError, match="full capture"):
        bank.registered_capture(args, tasks[:35], output, scope.CAPTURE_PATH,
                                read_json(scope.CAPTURE_PATH), None)
