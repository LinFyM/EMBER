"""Actual control1890 continuation and new fixed event window."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ember.operator_writer import bank
from ember.operator_writer.data import FormalData, TASKS
from ember.operator_writer.run import (
    CONTINUATION2340_SPEC_PATH, PILOT_SPEC_PATH, specification,
    validate_attempt, validate_train_request,
)

ASSET = Path("/data1/user/ymdai/projects/EMBER")
PARENT = Path("/data1/user/ymdai/ember_runs/operator_public_function_pilot_20260929"
              "/control/train/attempts/continuation/checkpoints/macro_00001890")


def test_actual_parent_events_and_explicit_control_lineage(tmp_path):
    old_spec = specification(PILOT_SPEC_PATH)
    spec = specification(CONTINUATION2340_SPEC_PATH)
    old = FormalData(ASSET, old_spec, query_labels=False)
    new = FormalData(ASSET, spec, query_labels=False)
    try:
        for step in (0, 899, 1799, 1800, 1889):
            assert old.tasks_for_step(step) == new.tasks_for_step(step)
            for task in old.tasks_for_step(step):
                assert old.event(step, task) == new.event(step, task)
        for task in TASKS:
            for visit in (210, 249, 250, 259):
                step = next(step for step in range(visit * 9, visit * 9 + 9)
                            if task in new.tasks_for_step(step))
                event = new.event(step, task)
                round_index = 4 if visit < 250 else 5
                order = np.random.default_rng(np.random.SeedSequence(
                    [20260928, 1, task, round_index])).permutation(50)
                assert event["teacher_demo"] == int(order[visit % 50])
                assert len(event["queries"]) == 28
                assert all(row["demo"] != event["teacher_demo"] for row in event["queries"])
        trainer = torch.load(PARENT / "trainer_state.pt", map_location="meta", mmap=True,
                             weights_only=True)
        assert trainer["next_macro"] == trainer["metrics_rows"] == 1890
        assert trainer["scheduler"]["last_epoch"] == 1890
        assert trainer["training_state"] == {
            "updates": 1890, "mode": "T", "pilot_arm": "control", "loss_variant": "full"}
        assert new.restore(trainer["sampler_state"], migrate_pilot_1890=True) == {
            "from_schema": "ember_operator_read_write_events_v6",
            "to_schema": "ember_operator_read_write_events_v7", "cursor": 1890,
            "appended_teacher_round": [20260928, 1, "task", 5]}
        old_run = bank.inspect_training_source(old_spec, PARENT, "T", sealed_evaluation=True)
        assert old_run["git"]["commit"] == spec["continuation"]["parent_training_git"]
        assert old_run["pilot_arm"] == "control" and old_run["loss_variant"] == "full"
        prefix = [json.loads(line) for line in
                  (PARENT.parent.parent / "metrics.jsonl").read_text().splitlines()]
        assert len(prefix) == 1890
        assert [row["update"] for row in prefix] == list(range(1, 1891))
        assert all(row.get("pilot_arm") == "control" and row.get("loss_variant") == "full"
                   for row in prefix[1800:])
        args = SimpleNamespace(mode="T", pilot_arm=None, resume=PARENT,
                               attempt="continuation", microbatch=28, frame_chunk=8,
                               stop_after_macro=None)
        validate_train_request(spec, args)
        contract = {**old_run,
                    "git": {"commit": "new-pushed-freeze", "branch": "",
                            "dirty_paths": [], "pushed_ref": "origin/codex/demonstration-transfer"},
                    "spec": str(CONTINUATION2340_SPEC_PATH), "events": spec["events"],
                    "sampler": {k: v for k, v in new.sampler_state().items() if k != "next_step"},
                    "continuation": spec["continuation"], "parent_checkpoint": str(PARENT),
                    "loss_variant": "full"}
        contract.pop("pilot_arm")
        contract.pop("pilot")
        output = tmp_path / "T/train/attempts/continuation"
        validate_attempt(spec | {"run_root": str(tmp_path)}, args, contract, output)
        with pytest.raises(ValueError, match="unregistered continuation parent or arm"):
            validate_attempt(spec | {"run_root": str(tmp_path)}, args,
                             contract | {"loss_variant": "full_plus_public_beta"}, output)
        with pytest.raises(ValueError, match="control2340"):
            validate_train_request(old_spec, args)
        with pytest.raises(ValueError, match="complete intermediate"):
            validate_train_request(spec, SimpleNamespace(**{**vars(args),
                                                            "stop_after_macro": 1900}))
    finally:
        old.close()
        new.close()


@pytest.mark.parametrize("request_kind", ["fixed", "file"])
def test_controlled_exit_occurs_after_published_complete_boundary(tmp_path, monkeypatch, request_kind):
    from ember.operator_writer import run

    spec = specification(CONTINUATION2340_SPEC_PATH)
    closed, visited, published = [], [], []
    data = SimpleNamespace(updates=2340, close=lambda: closed.append(True))
    session = SimpleNamespace(
        data=data, output=tmp_path, mode="T",
        context=SimpleNamespace(is_main=True, world_size=3),
    )
    args = SimpleNamespace(
        mode="T", pilot_arm=None, resume=PARENT, attempt="continuation",
        microbatch=28, frame_chunk=8,
        stop_after_macro=1980 if request_kind == "fixed" else None,
    )

    def update_stub(current, updates, rows):
        visited.append(updates + 1)
        if request_kind == "file" and updates + 1 == 1970:
            (tmp_path / "stop_at_next_ecp.request").write_text("stop")
        if updates + 1 == 1980:
            checkpoint = tmp_path / "checkpoints/macro_00001980"
            checkpoint.mkdir(parents=True)
            published.append(checkpoint)
        return updates + 1, rows + 1

    real_write = run.write_json_atomic

    def verify_publication_before_stop(path, payload):
        assert len(published) == 1 and Path(payload["checkpoint"]) == published[0]
        assert payload["updates"] == payload["metrics_rows"] == 1980
        real_write(path, payload)

    monkeypatch.setattr(run, "prepare_train", lambda *_: session)
    monkeypatch.setattr(run, "restore", lambda *_: (1890, 1890))
    monkeypatch.setattr(run, "update", update_stub)
    monkeypatch.setattr(run, "gather", lambda value, world: [value, False, False])
    monkeypatch.setattr(run, "write_json_atomic", verify_publication_before_stop)
    run.train(spec, args)
    assert visited == list(range(1891, 1981))
    assert closed == [True]
    assert json.loads((tmp_path / "stopped_at_ecp.json").read_text())["next_resume_from_this_ecp"]
    assert not (tmp_path / "completion.json").exists()
