"""Real old450 public factors, canonical144 capture and immutable A28 sources."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors import safe_open

from ember.eval_adapters import inspect_static_task_lora_adapter
from ember.operator_writer import bank, joint_readout as readout, run
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.preparation import _explicit_diagnostic_states, _registered_trajectory_capture
from ember.pi05_source_checkpoint import read_json, write_json_atomic


ASSET = Path("/data1/user/ymdai/projects/EMBER")


def _task_request(manifest):
    return tuple(SimpleNamespace(suite=row["suite"], task_id=row["task_id"],
                                 init_state_ids=(32, 33, 34, 35)) for row in manifest["tasks"])


def test_real_old450_public_bank_and_actual_registered_consumer(tmp_path, monkeypatch):
    monkeypatch.setattr(readout, "ROOT", tmp_path)
    monkeypatch.setattr(run, "frozen_git", lambda: {"commit": "fixture-read-only-export",
                        "branch": "", "dirty_paths": [], "pushed_ref": "origin/main"})
    path = readout.materialize("T450_public", readout.OLD_CHECKPOINT, ASSET)
    manifest = read_json(path)
    tasks = _task_request(manifest)
    assert manifest["training_git"] == "81846ed35933222b14ac693a0b760268ecff7f17"
    assert manifest["information_wall"]["teacher_video_values_read"] == 0
    assert len(manifest["tasks"]) == 36 and len(manifest["conditions"]) == 144
    assert manifest["scene_root"] == str(readout.PUBLIC_SCENES)
    with safe_open(manifest["shared"]["path"], framework="pt", device="cpu") as handle:
        assert len(handle.keys()) == 76
    assert len(list(path.parent.glob("*.safetensors"))) == 1
    adapter = inspect_static_task_lora_adapter(
        manifest_path=path, source=manifest["source"], tasks=tasks,
        evaluation_role=readout.scope.ROLE, require_formal=True)
    assert adapter["scene_manifest"]["path"] == str(readout.PUBLIC_SCENES / "manifest.json")
    common = object()
    consumer = SimpleNamespace(bank=adapter, common=common)
    for condition in manifest["conditions"][:2]:
        assert bank.FrozenOperatorAdapter._state(consumer, condition["condition_id"]) is common
    capture_path = readout.REPO / "configs/operator_read_write_v1/joint_public_capture.json"
    args = SimpleNamespace(role=readout.scope.ROLE, mode="formal", state_count=4,
                           init_state_ids=(32, 33, 34, 35), static_task_lora_manifest=path,
                           trajectory_capture_selection=capture_path)
    assert _explicit_diagnostic_states(args) == (32, 33, 34, 35)
    output = tmp_path / "T450_public/evaluation/public144"
    capture, stage = _registered_trajectory_capture(args, tasks, output, None, readout.REPO)
    assert len(capture["full_conditions"]) == 36
    assert stage["full_conditions_only"] is False and capture["passive_trace"]
    altered = deepcopy(manifest)
    altered["tasks"][0]["episodes"][0]["teacher_demo_indices"] = [49]
    with pytest.raises(ValueError, match="scope changed"):
        readout.inspect(altered, path, manifest["source"],
                        tuple((t.suite, t.task_id) for t in tasks), readout.scope.ROLE, True,
                        {(t.suite, t.task_id): t.init_state_ids for t in tasks})
    with pytest.raises(Pi05EvaluationError):
        inspect_static_task_lora_adapter(manifest_path=path, source=manifest["source"], tasks=tasks,
                                        evaluation_role="validation", require_formal=True)
    with pytest.raises(ValueError, match="450 endpoint"):
        readout.source_record("T450_public", readout.OLD_CHECKPOINT.with_name("macro_00000900"))


def test_capture_rejects_wrong_scene_panel_and_frozen_A28_reuses_real_FM():
    capture = read_json(readout.REPO / "configs/operator_read_write_v1/joint_public_capture.json")
    tasks = [SimpleNamespace(suite=r["suite"], task_id=r["task_id"], init_state_ids=readout.scope.STATES)
             for r in capture["full_conditions"]]
    expected = readout.capture_expectations({"mode": "joint_public", "joint_public_study": True},
                    readout.bank_path("joint_public"), tasks)
    assert expected["states"] == (32, 33, 34, 35) and expected["task_count"] == 36
    tasks[0].init_state_ids = (0, 1, 2, 3)
    with pytest.raises(ValueError, match="scope changed"):
        readout.capture_expectations({"mode": "joint_public", "joint_public_study": True},
                                      readout.bank_path("joint_public"), tasks)
    spec = importlib.util.spec_from_file_location("joint_fixed_FM_consumer",
                                                  readout.REPO / "scripts/operator_joint_readouts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    panels = read_json(readout.FIXED_PANEL / "fixed_panels.json")
    for task in readout.TRAIN_TASKS:
        flow, path = module.fixed_flow(task, panels[str(task)])
        assert path.name == f"task{task:03d}_A_query_flow_target.pt"
        assert flow["FM_target"].shape == (28, 50, 32) and flow["time"].shape == (28,)
        assert len(flow["queries"]) == 28


def test_full400_actual_capture_preparation_with_scope_fixture(tmp_path, monkeypatch):
    """Exercise scope wiring only; a future joint450 bank is not present yet."""
    monkeypatch.setattr(readout, "ROOT", tmp_path)
    prior = read_json(readout.OLD_CHECKPOINT.parents[4] / "banks/450/manifest.json")
    tasks = [SimpleNamespace(suite=r["suite"], task_id=r["task_id"], init_state_ids=tuple(range(50)))
             for r in prior["tasks"]]
    path = readout.bank_path("joint")
    path.parent.mkdir(parents=True)
    write_json_atomic(path, {"kind": bank.KIND, "mode": "joint", "joint_public_study": True})
    args = SimpleNamespace(role="validation", mode="formal", state_count=50, init_state_ids=None,
                           static_task_lora_manifest=path,
                           trajectory_capture_selection=readout.REPO /
                               "configs/operator_read_write_v1/joint_official_capture.json")
    assert _explicit_diagnostic_states(args) is None
    output = tmp_path / "joint/evaluation/correct400"
    capture, stage = _registered_trajectory_capture(args, tasks, output, None, readout.REPO)
    assert len(capture["full_conditions"]) == 8
    assert all(r["init_state_id"] == 0 for r in capture["full_conditions"])
    assert stage["full_conditions_only"] is False and capture["passive_trace"]
    with pytest.raises(Pi05EvaluationError, match="scope changed"):
        _registered_trajectory_capture(args, tasks, output.with_name("correct80"), None, readout.REPO)
