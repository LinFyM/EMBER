"""Real selected-control source consumption with an isolated selection fixture."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ember.operator_writer import bank, run, selected_scope
from ember.pi05_eval_contract import inspect_source_checkpoint, load_evaluation_authorities
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record


ASSET = Path("/data1/user/ymdai/projects/EMBER")


@pytest.fixture
def selected_1890(tmp_path, monkeypatch):
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
    return candidate_bank


def test_selected_other_reuses_real_complete_1890_bank_and_paired_capture(selected_1890):
    candidate_bank = selected_1890
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
    from ember.pi05_eval.preparation import _registered_trajectory_capture

    args = SimpleNamespace(static_task_lora_manifest=actual, role="validation", mode="formal",
                           trajectory_capture_selection=selected_scope.CAPTURE)
    registered, stage = _registered_trajectory_capture(
        args, tasks=tasks, output_dir=selected_scope.output_path("same_task_other"),
        task_subset=None, repo_root=ASSET)
    assert registered["full_conditions"] == capture["full"]
    assert len(registered["full_conditions"]) == 8 and stage["full_conditions_only"] is False
    beta_path = selected_scope.materialize_public_beta(ASSET)
    beta = bank.inspect_bank(manifest_path=beta_path, source=source, task_keys=keys,
                             evaluation_role="validation", require_formal=True,
                             task_init_state_ids={key: tuple(range(50)) for key in keys})
    assert beta["arm"] == "public_beta" and len(beta["conditions"]) == 400
    assert beta["information_wall"]["teacher_video_values_read"] == 0


@pytest.mark.parametrize("arm", ["cross_suite_wrong", "shuffled"])
def test_selected_video_reaches_T_compile_with_target_language_and_real_frame_order(
        selected_1890, monkeypatch, arm):
    """Exercise the real materializer through its compile boundary without a model forward."""
    from ember.operator_writer import data
    from ember.writer.video_controls import controlled_frames

    source = read_json(selected_1890)
    tasks, _ = selected_scope.task_rows(source, arm)
    task, episode = tasks[0], tasks[0]["episodes"][0]
    donor, demo = episode["video_global_task_id"], episode["teacher_demo_indices"][0]
    indices = np.array([0, 5, 10, 15, 20, 25, 30, 34])
    frames = np.broadcast_to(np.arange(16, dtype=np.uint8).reshape(8, 2, 1, 1, 1),
                             (8, 2, 3, 2, 2)).copy()
    expected = torch.from_numpy(frames)
    if arm == "shuffled":
        order, _, _ = controlled_frames(indices, control={
            "selection_seed": 7, "language_global_task_id": task["global_task_id"],
            "arm": arm}, demo=demo)
        expected = expected.index_select(0, order)
        assert not torch.equal(expected, torch.from_numpy(frames))
        assert donor == task["global_task_id"]
    else:
        assert donor // 10 != task["global_task_id"] // 10
    observed = {"closed": False, "compile_calls": 0}

    class FakeData:
        def __init__(self, _asset, _spec, *, query_labels, task_ids, role):
            assert query_labels is False and role == "validation"
            self.tasks = {row["global_task_id"]: SimpleNamespace(
                authority=SimpleNamespace(language=row["language"])) for row in source["tasks"]}
            assert set(task_ids) == set(self.tasks)
            self.videos = self

        def load(self, video_task, video_demo):
            assert (video_task, video_demo) == (donor, demo)
            return SimpleNamespace(frames=frames, frame_indices=indices, raw_frame_count=35)

        def close(self):
            observed["closed"] = True

    def tokenizer(languages):
        assert languages == [task["language"]]
        return torch.tensor([[7]]), torch.tensor([[True]]), None

    def compile_boundary(condition, *, frame_chunk):
        observed["compile_calls"] += 1
        pixels, displayed_indices, tokens, mask = condition
        assert frame_chunk == 8 and torch.equal(pixels, expected)
        assert torch.equal(displayed_indices, torch.from_numpy(indices))
        assert tokens.tolist() == [[7]] and mask.tolist() == [[True]]
        raise RuntimeError("verified compile boundary; no model forward")

    def build_runtime(_asset, _spec, device, mode):
        assert mode == "T" and device == torch.device("cpu")
        return SimpleNamespace(source=source["source"], tokenizer=tokenizer,
            writer=SimpleNamespace(load_state_dict=lambda *_a, **_k: None, eval=lambda: None),
            compile=compile_boundary)

    monkeypatch.setattr(run, "build_runtime", build_runtime)
    monkeypatch.setattr(data, "FormalData", FakeData)
    monkeypatch.setattr(selected_scope, "load_file", lambda *_a, **_k: {})
    with pytest.raises(RuntimeError, match="verified compile boundary"):
        selected_scope.materialize_video(arm, ASSET, torch.device("cpu"))
    assert observed == {"closed": True, "compile_calls": 1}
