"""Contract checks for fixed phased P/I events and source-isolated compiles."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ember.demonstration_learning.data import SUPPORTED, TransferData, audit_full_cycle, audit_two_cycles
from ember.demonstration_learning.bank import _held_condition_data, _task_rows, materialize, registered_capture
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
        state = data.events.sampler_state()
        state["next_step"] = 1
        state["old_cursor"] = {str(task): 8 if task in SUPPORTED else 16
                               for task in data.events.lengths}
        state["new_cursor"] = {str(task): 2 for task in SUPPORTED}
        data.events.restore(state)
        assert data.events.next_step == 1
        with pytest.raises(ValueError):
            data.events.restore({**state, "old_cursor": {**state["old_cursor"], "0": 7}})
    finally:
        data.close()


def test_crossing_resume_identity_rejects_changed_cursor() -> None:
    spec = json.loads((REPO / "configs/demonstration_transfer_v1/learning_spec.json").read_text())
    data = TransferData(ASSETS, spec)
    try:
        state = data.events.sampler_state()
        state["next_step"] = 72
        data.events.restore(state)
        assert data.events.next_step == 72
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
    spec = {"runtime": {"run_root": str(tmp_path)},
            "execution": {"checkpoint_macros": [72, 144, 216, 288]}}
    contract = {"arm": "P", "git": {"commit": "frozen"}, "source": {"revision": 1},
                "topology": {"world_size": 2, "ranks": ["gpu-a", "gpu-b"]},
                "microbatch": 28, "frame_chunk": 8}
    attempt = tmp_path / "P/train/attempts/fresh"
    checkpoint = attempt / "checkpoints/macro_00000072"
    checkpoint.mkdir(parents=True)
    (attempt / "run_contract.json").write_text(json.dumps(contract))
    (checkpoint / "checkpoint_manifest.json").write_text(json.dumps({
        "stage": "demonstration_transfer_learning_stage1",
        "run_contract_schema": "ember_demonstration_transfer_formal_stage1_run_v1", "next_macro": 72}))
    assert _registered_resume(checkpoint, contract, spec) == 72
    physical = {**contract, "microbatch": 14, "frame_chunk": 4}
    assert _registered_resume(checkpoint, physical, spec) == 72
    for changed in ({**contract, "arm": "I"}, {**contract, "source": {"revision": 2}},
                    {**contract, "topology": {"world_size": 1}},
                    {**contract, "microbatch": 7, "frame_chunk": 8}):
        with pytest.raises(ValueError):
            _registered_resume(checkpoint, changed, spec)
    with pytest.raises(ValueError):
        _registered_resume(tmp_path / "old_engineering/P/fresh/checkpoints/macro_00000006", contract, spec)
    newer = attempt / "checkpoints/macro_00000144"
    newer.mkdir()
    (newer / "checkpoint_manifest.json").write_text("{}")
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

    bank_path = tmp_path / "P/banks/288/manifest.json"
    bank_path.parent.mkdir(parents=True)
    bank_path.write_text(json.dumps({"kind": "demonstration_comparison_lora_bank", "arm": "P"}))
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


def test_formal_entrypoints_refuse_retired_m(monkeypatch) -> None:
    spec = json.loads((REPO / "configs/demonstration_transfer_v1/learning_spec.json").read_text())
    with pytest.raises(ValueError, match="formal P/I"):
        train(spec, SimpleNamespace(arm="M", stop_after=288, frame_chunk=8, microbatch=28,
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
