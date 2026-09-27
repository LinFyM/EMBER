"""Contract checks for fixed phased P/I events and source-isolated compiles."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ember.demonstration_learning.data import SUPPORTED, TransferData, audit_full_cycle, audit_two_cycles
from ember.demonstration_learning.bank import (_held_condition_data, _source_checkpoint, _task_rows,
                                                materialize, registered_capture)
from ember.demonstration_learning.model import concatenate_factors, split_identity_template
from ember.demonstration_learning.run import Runtime, _registered_resume, train
from ember.lora import (LoRATarget, copy_task_lora_state_, identity_lora_state,
                        task_lora_state_dict)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.preparation import _registered_trajectory_capture


REPO = Path(__file__).resolve().parents[2]
ASSETS = Path("/data1/user/ymdai/projects/EMBER")


def test_complete_factor_concatenation_keeps_two_BA_terms_and_gradient() -> None:
    full = {}
    for index in range(38):
        stem = f"target_{index}"
        full[stem + ".lora_A.default.weight"] = torch.randn(144, 5)
        full[stem + ".lora_B.default.weight"] = torch.zeros(3, 144)
    common, video = split_identity_template(full)
    for name in common:
        common[name].requires_grad_()
        video[name].requires_grad_()
    compiled = concatenate_factors(common, video)
    for index in range(38):
        stem = f"target_{index}"
        a, b = stem + ".lora_A.default.weight", stem + ".lora_B.default.weight"
        assert compiled[b].shape == (3, 144)
        assert compiled[a].shape == (144, 5)
        assert torch.allclose(compiled[b] @ compiled[a], common[b] @ common[a] + video[b] @ video[a])
    a, b = "target_0.lora_A.default.weight", "target_0.lora_B.default.weight"
    with torch.no_grad():
        common[b].fill_(0.25)
        video[b].fill_(0.5)
    (concatenate_factors(common, video)[b] @ concatenate_factors(common, video)[a]).square().sum().backward()
    assert all(value.grad is not None and torch.count_nonzero(value.grad)
               for value in (common[a], common[b], video[a], video[b]))


def test_full_events_and_actual_new_query_offset() -> None:
    spec = json.loads((REPO / "configs/demonstration_transfer_v1/learning_spec.json").read_text())
    data = TransferData(ASSETS, spec)
    try:
        audit = audit_full_cycle(data.events)
        assert (audit["queries_total"], audit["new_queries"], audit["old_queries"]) == (32256, 8960, 23296)
        assert set(audit["supported"]) == set(SUPPORTED)
        assert audit["new_events_per_round"] == [10] * 32
        two = audit_two_cycles(data.events)
        assert (two["queries_total"], two["new_queries"], two["old_queries"]) == (64512, 17920, 46592)
        assert {row["task"] for row in data.events.event(0, "P") if row["kind"] == "new"} == {29}
        assert {row["task"] for row in data.events.event(1, "P") if row["kind"] == "new"} == {4, 35}
        assert {row["task"] for row in data.events.event(5, "P") if row["kind"] == "new"} == {97}
        new = next(row for row in data.events.event(0, "P") if row["kind"] == "new")
        paired = next(row for row in data.events.event(0, "I")
                      if row["task"] == new["task"])
        assert new["queries"] == paired["queries"] and new["flow_seed"] == paired["flow_seed"]
        raw = data.batch(new)
        assert raw["action"].shape == (28, 50, 7)
        assert raw["observation.state"].shape == (28, 8)
        assert raw["observation.images.camera1"].shape == (28, 3, 256, 256)
        first = new["queries"][0]
        with np.load(first["path"], allow_pickle=False) as trace:
            assert torch.equal(raw["action"][0, 0], torch.from_numpy(trace["actions"][first["frame"]]))
            assert int(trace["rgb_steps"][first["rgb_ordinal"]]) == first["frame"]
    finally:
        data.close()


def test_direct_mt_hierarchy_and_resume_cursor() -> None:
    spec = json.loads((REPO / "configs/demonstration_transfer_v1/learning_spec.json").read_text())
    data = TransferData(ASSETS, spec, arm="M")
    try:
        assert data.videos is None
        for update in range(3):
            jobs = data.events.event(update)
            assert len(jobs) == 36 and sum(len(row["queries"]) for row in jobs) == 576
            assert sum(query["kind"] == "new" for row in jobs for query in row["queries"]) == 160
            assert sum(query["kind"] == "old" for row in jobs for query in row["queries"]) == 416
            for row in jobs:
                if row["task"] in SUPPORTED:
                    assert {source: sum(q.get("source") == source for q in row["queries"])
                            for source in range(4)} == {source: 2 for source in range(4)}
        segments = data.physical_segments(jobs[0])
        assert [offset for offset, _ in segments] == [0, 8]
        assert [batch["action"].shape for _, batch in segments] == [(8, 50, 7)] * 2
        assert data.events.sampler_state()["next_step"] == 0
    finally:
        data.close()


def test_crossing_resume_identity_rejects_changed_cursor() -> None:
    spec = json.loads((REPO / "configs/demonstration_transfer_v1/learning_spec.json").read_text())
    data = TransferData(ASSETS, spec)
    try:
        state = data.events.sampler_state()
        state["next_step"] = 288
        data.events.restore(state)
        assert data.events.next_step == 288
        data.events.restore({**state, "next_step": 360})
        assert data.events.next_step == 360
        with pytest.raises(ValueError):
            data.events.restore({**state, "event_seeds": {**state["event_seeds"], "flow": 0}})
        with pytest.raises(ValueError):
            data.events.restore({**state, "next_step": 2})
        with pytest.raises(ValueError):
            data.events.restore({**state, "schema_version": "ember_demonstration_transfer_learning_events_v1"})
        with pytest.raises(ValueError):
            data.events.restore({**state, "supported_task_phases": {"0": 1}})
    finally:
        data.close()


def test_formal_resume_requires_latest_same_arm_and_exact_logical_contract(tmp_path: Path) -> None:
    from ember.demonstration_learning import run

    spec = json.loads((REPO / "configs/demonstration_transfer_v1/learning_spec.json").read_text())
    spec["runtime"]["run_root"] = str(tmp_path)
    parent = run._stage1_checkpoint(spec, "P")
    old = json.loads((parent.parent.parent / "run_contract.json").read_text())
    contract = {**old, "schema_version": run.RUN_SCHEMA, "stage": run.STAGE,
                "git": {"commit": "new-frozen", "branch": "", "dirty_paths": [],
                        "pushed_ref": "origin/main"},
                "spec": str(run.SPEC_PATH), "qualification": "formal_stage2_macro576",
                "stage1_parent": {"git": spec["execution"]["stage1_parent"]["git"],
                                  "checkpoint": str(parent)}}
    assert _registered_resume(parent, contract, spec) == 288
    other_parent = run._stage1_checkpoint(spec, "I")
    assert _registered_resume(other_parent, {**contract, "arm": "I",
                                             "stage1_parent": {**contract["stage1_parent"],
                                                               "checkpoint": str(other_parent)}}, spec) == 288
    physical = {**contract, "microbatch": 14, "frame_chunk": 4}
    with pytest.raises(ValueError, match="physical chunks"):
        _registered_resume(parent, physical, spec)
    for changed in ({**contract, "arm": "I"}, {**contract, "source": {"revision": 2}},
                    {**contract, "topology": {**contract["topology"], "visible_devices": "3,6"}},
                    {**contract, "microbatch": 7, "frame_chunk": 8}):
        with pytest.raises(ValueError):
            _registered_resume(parent, changed, spec)
    with pytest.raises(ValueError):
        _registered_resume(tmp_path / "old_engineering/P/fresh/checkpoints/macro_00000006", contract, spec)

    actual_read = run.read_json
    def wrong_parent_git(path):
        result = actual_read(path)
        if Path(path) == parent.parent.parent / "run_contract.json":
            return {**result, "git": {**result["git"], "commit": "wrong-parent"}}
        return result
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run, "read_json", wrong_parent_git)
        with pytest.raises(ValueError, match="parent identity"):
            _registered_resume(parent, contract, spec)
    changed_spec = json.loads(json.dumps(spec))
    changed_spec["optimization"]["lr"] = 1e-4
    with pytest.raises(ValueError, match="numerical learning dictionaries"):
        _registered_resume(parent, contract, changed_spec)

    attempt = tmp_path / "P/train/attempts/after288"
    checkpoint = attempt / "checkpoints/macro_00000360"
    checkpoint.mkdir(parents=True)
    (attempt / "run_contract.json").write_text(json.dumps(contract))
    files = {}
    for name in ("ecp.safetensors", "trainer_state.pt", "rank_00_state.pt", "rank_01_state.pt"):
        (checkpoint / name).write_bytes(b"x")
        files[name] = {"bytes": 1}
    (checkpoint / "checkpoint_manifest.json").write_text(json.dumps({
        "schema_version": "ember_ecp_checkpoint_v1", "stage": run.STAGE,
        "run_contract_schema": run.RUN_SCHEMA, "world_size": 2,
        "next_macro": 360, "files": files}))
    assert _registered_resume(checkpoint, contract, spec) == 360
    with pytest.raises(ValueError, match="superseded"):
        _registered_resume(parent, contract, spec)
    with pytest.raises(ValueError, match="frozen run contract"):
        _registered_resume(checkpoint, {**contract, "git": {"commit": "wrong"}}, spec)
    with pytest.raises(ValueError):
        _registered_resume(checkpoint, {**contract, "source": {"revision": 2}}, spec)
    newer = attempt / "checkpoints/macro_00000432"
    newer.mkdir()
    for name in files:
        (newer / name).write_bytes(b"x")
    (newer / "checkpoint_manifest.json").write_text(json.dumps({
        "schema_version": "ember_ecp_checkpoint_v1", "stage": run.STAGE,
        "run_contract_schema": run.RUN_SCHEMA, "world_size": 2,
        "next_macro": 432, "files": files}))
    with pytest.raises(ValueError, match="latest"):
        _registered_resume(checkpoint, contract, spec)


def test_validation_bank_metadata_and_capture_scope_without_held_query(monkeypatch, tmp_path: Path) -> None:
    spec = json.loads((REPO / "configs/demonstration_transfer_v1/learning_spec.json").read_text())
    tasks, conditions, authority = _task_rows(spec, ASSETS)
    assert len(tasks) == 8 and len(conditions) == 400
    assert [row["global_task_id"] for row in tasks] == [3, 6, 11, 16, 23, 26, 31, 39]
    assert all(row["split_role"] == "validation" and len(row["episodes"]) == 50
               and len({ep["teacher_demo"] for ep in row["episodes"]}) == 50 for row in tasks)
    seen = []
    class VideoStub:
        def __init__(self, rows, **kwargs):
            seen.append((tuple(row.task_id for row in rows), kwargs))
    monkeypatch.setattr("ember.demonstration_learning.bank.RawTeacherVideoStore", VideoStub)
    data = _held_condition_data(spec, authority)
    assert data.tasks is authority and isinstance(data.videos, VideoStub)
    assert seen[0][1] == {"frame_stride": 5, "camera_view": "agentview"}

    bank_path = tmp_path / "P/banks/576/manifest.json"
    bank_path.parent.mkdir(parents=True)
    bank_path.write_text(json.dumps({"kind": "demonstration_comparison_lora_bank", "arm": "P"}))
    train = tmp_path / "P/train"
    train.mkdir(parents=True)
    (train / "final_checkpoint.json").write_text(json.dumps({
        "arm": "P", "checkpoint": str(tmp_path / "P/train/attempts/fresh/checkpoints/macro_00000288"),
        "run_contract": str(tmp_path / "P/train/attempts/fresh/run_contract.json")}))
    narrowed = {**spec, "runtime": {**spec["runtime"], "run_root": str(tmp_path)}}
    with pytest.raises(ValueError, match="final checkpoint pointer"):
        _source_checkpoint("P", narrowed)
    (train / "final_checkpoint.json").write_text(json.dumps({
        "arm": "P", "checkpoint": str(tmp_path / "P/train/attempts/from288/checkpoints/macro_00000576"),
        "run_contract": str(tmp_path / "P/train/attempts/from288/run_contract.json")}))
    assert _source_checkpoint("P", narrowed).name == "macro_00000576"
    capture_path = tmp_path / "P_capture.json"
    capture_path.write_text("{}")
    rows = [SimpleNamespace(suite=row["suite"], task_id=row["task_id"],
                            init_state_ids=tuple(range(50))) for row in tasks]
    full = [{"suite": row.suite, "task_id": row.task_id, "init_state_id": 0} for row in rows]
    manifest = {"schema_version": "ember_pi05_registered_trajectory_capture_v1",
                "study_id": "demonstration_transfer_learning_20260927", "task_subset_selection": None,
                "full_conditions": full, "mode": "compact", "passive_control_trace":
                "ember_demonstration_comparison_passive_capture_v1", "stage_predicates": True,
                "training_gradient_use": False, "checkpoint_selection_use": False,
                "validation_use": False, "test_use": False}
    args = SimpleNamespace(static_task_lora_manifest=bank_path, role="validation", mode="formal",
                           trajectory_capture_selection=capture_path)
    output = tmp_path / "P/evaluation/correct400"
    capture_path.write_text(json.dumps(manifest))
    capture, stage = registered_capture(args, rows, output, capture_path, manifest, None)
    assert len(capture["full_conditions"]) == 8 and stage["validation_action_reads"] == 0
    routed, routed_stage = _registered_trajectory_capture(args, rows, output, None, REPO)
    assert (routed, routed_stage) == (capture, stage)
    with pytest.raises(Pi05EvaluationError):
        registered_capture(SimpleNamespace(static_task_lora_manifest=bank_path,
                                           role="development_train", mode="screen"),
                           rows, output, capture_path, manifest, None)


def test_final_cursor_resume_publishes_checkpoint_without_an_update(monkeypatch, tmp_path: Path) -> None:
    from ember.demonstration_learning import run

    spec = json.loads((REPO / "configs/demonstration_transfer_v1/learning_spec.json").read_text())
    spec["runtime"]["run_root"] = str(tmp_path)
    output = tmp_path / "P/train/attempts/finish"
    output.mkdir(parents=True)
    session = SimpleNamespace(output=output, context=SimpleNamespace(is_main=True),
                              data=SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(run, "_prepare_train", lambda *_: session)
    monkeypatch.setattr(run, "_restore", lambda *_: (576, 576))
    monkeypatch.setattr(run, "_step", lambda *_: pytest.fail("completed ECP must not train again"))
    saved = []

    def save_fixture(current, cursor, rows):
        saved.append((cursor, rows))
        (current.output / "checkpoints" / f"macro_{cursor:08d}").mkdir(parents=True)

    monkeypatch.setattr(run, "_save_checkpoint", save_fixture)
    args = SimpleNamespace(arm="P", stop_after=576, frame_chunk=8, microbatch=28,
                           resume=tmp_path / "P/train/attempts/fresh/checkpoints/macro_00000288",
                           attempt="finish")
    run.train(spec, args)
    pointer = json.loads((tmp_path / "P/train/final_checkpoint.json").read_text())
    completion = json.loads((output / "completion.json").read_text())
    assert Path(pointer["checkpoint"]).is_dir() and saved == [(576, 576)]
    assert completion["checkpoint"] == pointer["checkpoint"]
    assert completion["actual_segment_updates"] == completion["actual_segment_queries"] == 0


def test_formal_entrypoints_refuse_retired_m(monkeypatch) -> None:
    spec = json.loads((REPO / "configs/demonstration_transfer_v1/learning_spec.json").read_text())
    with pytest.raises(ValueError, match="formal P/I"):
        train(spec, SimpleNamespace(arm="M", stop_after=576, frame_chunk=8, microbatch=28,
                                    resume=None, attempt="fresh"))
    monkeypatch.setattr("ember.demonstration_learning.bank._frozen_git", lambda: {})
    with pytest.raises(ValueError, match="formal bank admits P/I"):
        materialize("M", ASSETS, torch.device("cpu"))


def test_sequential_compile_restores_physical_source_identity() -> None:
    class Target(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A = torch.nn.Module()
            self.lora_A.add_module("default", torch.nn.Linear(2, 2, bias=False))
            self.lora_B = torch.nn.Module()
            self.lora_B.add_module("default", torch.nn.Linear(2, 3, bias=False))

    class Policy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.target = Target()

    contract = SimpleNamespace(targets=(LoRATarget("target", 2, 3),), rank=2,
                               alpha=2, dropout=0.0, identity_seed=7)
    policy = Policy()
    identity = identity_lora_state(contract)
    copy_task_lora_state_(policy, identity, contract)
    observed = []

    def read_source(model, _condition):
        actual = task_lora_state_dict(model)
        observed.append(all(torch.equal(actual[k], v) for k, v in identity.items()))
        return {"read": torch.tensor(1.0)}

    runtime = Runtime(policy, read_source, None, None, contract, {}, torch.device("cpu"), identity)
    runtime.compile(())
    previous_output = {name: value.clone() for name, value in identity.items()}
    previous_output["target.lora_B.default.weight"].fill_(0.5)
    copy_task_lora_state_(policy, previous_output, contract)
    assert runtime.source_delta_norm() > 0
    runtime.compile(())
    assert observed == [True, True]
    assert runtime.source_delta_norm() == 0
    assert runtime.source_identity_restores == 2
