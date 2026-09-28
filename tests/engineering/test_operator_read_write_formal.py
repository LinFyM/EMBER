"""CPU contract checks for the formal operator comparison, without model execution."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.lora import expected_lora_state_shapes
from ember.operator_writer.bank import (_mt_source, assemble_state, registered_capture,
                                        task_rows)
from ember.operator_writer.run import (CHECKPOINTS, TASKS, FormalData, audit,
                                       resume_contract_compatible, specification, validate_attempt)
from ember.pi05_eval.scene import inspect_registered_scenes, validate_scene_row
from ember.pi05_eval.preparation import _registered_trajectory_capture
from ember.pi05_eval_contract import inspect_source_checkpoint, load_evaluation_authorities
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic


ROOT = Path(__file__).resolve().parents[2]
ASSET = Path("/data1/user/ymdai/projects/EMBER")
SCENES = Path("/data1/user/ymdai/ember_runs/demonstration_transfer_learning_20260927/scenes")


def test_full_270_event_stream_and_exact_resume_cursor():
    spec = specification()
    data = FormalData(ASSET, spec, query_labels=False)
    try:
        summary = audit(spec, ASSET)
        assert (summary["updates_per_mode"], summary["conditions_per_mode"],
                summary["queries_per_mode"]) == (270, 1080, 30240)
        assert set(map(int, summary["teacher_order"])) == set(TASKS)
        for task, teachers in summary["teacher_order"].items():
            assert len(teachers) == len(set(teachers)) == 30
        for step in (0, 1, 8, 9, 269):
            jobs = [data.event(step, task) for task in data.tasks_for_step(step)]
            assert len(jobs) == 4 and all(row["visit"] == step // 9 for row in jobs)
            assert all(row["query_offset"] == 1 and len(row["queries"]) == 28 for row in jobs)
            assert all(row == data.event(step, row["task"]) for row in jobs)
        for cursor in CHECKPOINTS:
            state = data.sampler_state() | {"next_step": cursor}
            data.restore(state)
            assert data.next_step == cursor
        with pytest.raises(ValueError):
            data.restore(data.sampler_state() | {"schema_version": "ember_operator_read_write_events_v1"})
    finally:
        data.close()


def test_resume_only_allows_physical_oom_repacking():
    parent = {"git": "one", "mode": "T", "source": {"checkpoint": "source1000"},
              "topology": {"ranks": ["uuid0", "uuid1"]}, "microbatch": 28, "frame_chunk": 8}
    current = deepcopy(parent)
    assert resume_contract_compatible(parent, current)
    current.update(microbatch=14, frame_chunk=4)
    assert resume_contract_compatible(parent, current)
    for changed in ({"mode": "U"}, {"git": "engineering"},
                    {"source": {"checkpoint": "wrong"}},
                    {"topology": {"ranks": ["uuid1", "uuid0"]}},
                    {"microbatch": 13}, {"frame_chunk": 2}):
        candidate = current | changed
        assert not resume_contract_compatible(parent, candidate)


def test_resume_requires_latest_complete_same_arm_ecp(tmp_path):
    spec = {"run_root": str(tmp_path)}
    contract = {"git": "formal", "mode": "T", "source": "source1000",
                "topology": ["uuid0", "uuid1"], "microbatch": 28, "frame_chunk": 8}
    parent = tmp_path / "T/train/attempts/fresh"
    checkpoint = parent / "checkpoints/macro_00000090"
    checkpoint.mkdir(parents=True)
    write_json_atomic(parent / "run_contract.json", contract)
    files = {name: {"bytes": 1} for name in (
        "ecp.safetensors", "trainer_state.pt", "rank_00_state.pt", "rank_01_state.pt")}
    for name in files:
        (checkpoint / name).write_bytes(b"x")
    write_json_atomic(checkpoint / "checkpoint_manifest.json", {
        "stage": "operator_read_write_learning", "run_contract_schema": "ember_operator_read_write_formal_run_v1",
        "next_macro": 90, "files": files})
    args = SimpleNamespace(mode="T", resume=checkpoint)
    output = tmp_path / "T/train/attempts/recover1"
    validate_attempt(spec, args, contract, output)
    with pytest.raises(ValueError):
        validate_attempt(spec, SimpleNamespace(mode="U", resume=checkpoint), contract,
                         tmp_path / "U/train/attempts/recover1")
    with pytest.raises(ValueError):
        validate_attempt(spec, args, contract | {"git": "engineering"}, output)
    later = parent / "checkpoints/macro_00000180"
    later.mkdir()
    for name in files:
        (later / name).write_bytes(b"x")
    write_json_atomic(later / "checkpoint_manifest.json", {
        "stage": "operator_read_write_learning", "run_contract_schema": "ember_operator_read_write_formal_run_v1",
        "next_macro": 180, "files": files})
    with pytest.raises(ValueError):
        validate_attempt(spec, args, contract, output)


def test_fixed_mt_source_400_scene_and_single_rank128_state():
    spec = specification()
    authority = load_evaluation_authorities(ASSET / spec["source"]["evaluation_config"], ASSET)
    source_checkpoint = ASSET / spec["source"]["checkpoint"]
    source = inspect_source_checkpoint(authority, source_checkpoint.parent.parent,
                                       source_checkpoint, evaluation_mode="formal")
    checkpoint, run, manifest = _mt_source(spec, source)
    assert checkpoint == Path(spec["evaluation"]["mt_checkpoint"])
    assert manifest["consumed"]["next_optimizer_step"] == 300
    assert len(run["tasks"]) == 36
    with pytest.raises(ValueError):
        _mt_source(spec, source | {"model_path": "/wrong/source/policy"})
    tasks, conditions = task_rows(spec, ASSET)
    assert len(tasks) == 8 and len(conditions) == 400
    held = FormalData(ASSET, spec, query_labels=False,
                      task_ids=tuple(spec["evaluation"]["task_ids"]), role="validation")
    try:
        assert held.queries is None and held.rows is None and len(held.tasks) == 8
        with pytest.raises(ValueError):
            held.batch({"task": spec["evaluation"]["task_ids"][0], "queries": []})
    finally:
        held.close()
    assert all(len({e["teacher_demo_indices"][0] for e in task["episodes"]}) == 50 for task in tasks)
    prior = read_json(Path("/data1/user/ymdai/ember_runs/demonstration_transfer_learning_20260927"
                           "/stage1/P/banks/288/manifest.json"))
    assert [[episode["teacher_demo_indices"][0] for episode in task["episodes"]] for task in tasks] == [
        [episode["teacher_demo"] for episode in task["episodes"]] for task in prior["tasks"]]
    inspect_registered_scenes(SCENES, tasks)

    # The materialized T/U state has one A and one B for every target, with no second adapter.
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(ASSET / spec["source"]["lora_contract"]), rank=128)
    shapes = expected_lora_state_shapes(lora)
    shared = {name: torch.empty(shape) for name, shape in shapes.items() if ".lora_A." in name}
    conditional = {name: torch.empty(shape) for name, shape in shapes.items() if ".lora_B." in name}
    assert len(assemble_state(shared, conditional, lora)) == 76
    with pytest.raises(ValueError):
        assemble_state(shared, {name: value for name, value in conditional.items() if name != next(iter(conditional))}, lora)


def test_official_capture_and_scene_route_are_registered(tmp_path):
    spec = specification()
    tasks, _ = task_rows(spec, ASSET)
    rows = [SimpleNamespace(**task, init_state_ids=tuple(range(50))) for task in tasks]
    bank_path = tmp_path / "T" / "banks" / "270" / "manifest.json"
    bank_path.parent.mkdir(parents=True)
    write_json_atomic(bank_path, {"kind": "operator_read_write_lora_bank"})
    capture_path = ROOT / "configs/operator_read_write_v1/official_capture.json"
    selector = read_json(capture_path)
    output = bank_path.parent.parent.parent / "evaluation/correct400"
    capture, stage = registered_capture(
        SimpleNamespace(static_task_lora_manifest=bank_path, role="validation", mode="formal"),
        rows, output, capture_path, selector, None)
    assert len(capture["full_conditions"]) == 8 and stage["full_conditions_only"] is False
    prepared, prepared_stage = _registered_trajectory_capture(
        SimpleNamespace(static_task_lora_manifest=bank_path, role="validation", mode="formal",
                        trajectory_capture_selection=capture_path), rows, output, None, ROOT)
    assert (prepared, prepared_stage) == (capture, stage)
    scene_file = SCENES / "libero_spatial_task_03_state_000.npz"
    contract = {"operator_read_write_scene": {"root": str(SCENES)}}
    row = {"init_state_id": 0, "scene_reference": {
        "path": str(scene_file), "bytes": scene_file.stat().st_size,
        "restoration": "full_model_body_pose_post_dummy_sim_controller_and_dual_rgb_verified"}}
    validate_scene_row(row, tasks[0], contract)
    with pytest.raises(ValueError):
        validate_scene_row({**row, "scene_reference": None}, tasks[0], contract)
