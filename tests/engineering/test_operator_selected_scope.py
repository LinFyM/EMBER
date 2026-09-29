"""Real selected-control source consumption with an isolated selection fixture."""

from pathlib import Path
from types import SimpleNamespace

from ember.operator_writer import bank, run, selected_scope
from ember.pi05_eval_contract import inspect_source_checkpoint, load_evaluation_authorities
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record


ASSET = Path("/data1/user/ymdai/projects/EMBER")


def test_selected_other_reuses_real_complete_1890_bank_and_paired_capture(tmp_path, monkeypatch):
    root = tmp_path / "selected"
    root.mkdir()
    candidate_bank, official = selected_scope._correct_paths(1890)
    selection = root / "selection.json"
    write_json_atomic(selection, {
        "schema_version": selected_scope.SCHEMA,
        "selected_by": "science_main",
        "selection_rule": "highest_complete_correct400_tie_earlier",
        "macro": 1890,
        "correct_bank": file_record(candidate_bank),
        "correct_results": file_record(official / "results.json"),
    })
    monkeypatch.setattr(selected_scope, "ROOT", root)
    monkeypatch.setattr(selected_scope, "SELECTION", selection)
    clean_git = {"commit": "fixture-evaluation-commit", "branch": "",
                 "dirty_paths": [], "pushed_ref": "origin/main"}
    monkeypatch.setattr(run, "frozen_git", lambda *, continuation=False: clean_git)
    actual = selected_scope.register_other()
    old = read_json(candidate_bank)
    new = read_json(actual)
    assert new["conditions"] == old["conditions"]
    assert new["shared"] == old["shared"]
    assert new["tasks"] != old["tasks"]
    spec = read_json(Path(old["spec"]["path"]))
    authorities = load_evaluation_authorities(
        ASSET / spec["source"]["evaluation_config"], ASSET)
    checkpoint = ASSET / spec["source"]["checkpoint"]
    source = inspect_source_checkpoint(authorities, checkpoint.parent.parent,
                                       checkpoint, evaluation_mode="formal")
    keys = tuple((task["suite"], task["task_id"]) for task in old["tasks"])
    inspected = bank.inspect_bank(manifest_path=actual, source=source, task_keys=keys,
                                  evaluation_role="validation", require_formal=True,
                                  task_init_state_ids={key: tuple(range(50)) for key in keys})
    assert inspected["arm"] == "same_task_other"
    task = new["tasks"][0]
    assert task["episodes"][0]["teacher_demo_indices"] != old["tasks"][0]["episodes"][0]["teacher_demo_indices"]
    tasks = [SimpleNamespace(suite=suite, task_id=task_id,
                             init_state_ids=tuple(range(50))) for suite, task_id in keys]
    capture = selected_scope.capture_expectations(new, actual, tasks,
                                                  selected_scope.output_path("same_task_other"))
    assert capture["study"] == selected_scope.STUDY
    assert capture["expected_bank"] == actual.resolve()
    beta_path = selected_scope.materialize_public_beta(ASSET)
    beta = bank.inspect_bank(manifest_path=beta_path, source=source, task_keys=keys,
                             evaluation_role="validation", require_formal=True,
                             task_init_state_ids={key: tuple(range(50)) for key in keys})
    assert beta["arm"] == "public_beta" and len(beta["conditions"]) == 400
    assert beta["information_wall"]["teacher_video_values_read"] == 0
