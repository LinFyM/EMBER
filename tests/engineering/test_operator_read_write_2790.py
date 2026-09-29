"""T2340 parent migration and the only open formal T2790 training entry."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ember.operator_writer import bank, run
from ember.operator_writer.data import FormalData, TASKS
from ember.pi05_source_checkpoint import write_json_atomic

PARENT = (run.CONTINUATION2340_ROOT / "T/train/attempts/continuation/checkpoints"
          / "macro_00002340")


def events(spec):
    """Use the real deterministic event methods without reopening 36 HDF5 authorities."""
    value = FormalData.__new__(FormalData)
    value.seed = spec["events"]["seed"]
    value.event_schema = spec["events"]["schema_version"]
    value.updates = spec["execution"]["updates_per_mode"]
    value.checkpoints = tuple(spec["execution"]["checkpoints"])
    value.next_step = 0
    value.tasks = {task: SimpleNamespace(episode_lengths=(1000,) * 50) for task in TASKS}
    return value


def args(resume=PARENT):
    return SimpleNamespace(mode="T", pilot_arm=None, resume=resume,
                           attempt="continuation", microbatch=28, frame_chunk=8,
                           stop_after_macro=None)


def test_actual_parent_clock_and_v8_event_prefix():
    prior = run.specification(run.CONTINUATION2340_SPEC_PATH)
    current = run.specification(run.CONTINUATION2790_SPEC_PATH)
    source = bank.inspect_training_source(prior, PARENT, "T", sealed_evaluation=True)
    assert source["mode"] == "T" and source["loss_variant"] == "full"
    assert source["git"]["commit"] == current["continuation"]["parent_training_git"]
    trainer = torch.load(PARENT / "trainer_state.pt", map_location="meta", mmap=True,
                         weights_only=True)
    assert trainer["next_macro"] == trainer["metrics_rows"] == 2340
    assert trainer["scheduler"]["last_epoch"] == 2340
    assert trainer["training_state"] == {"updates": 2340, "mode": "T", "loss_variant": "full"}
    prefix = (PARENT.parent.parent / "metrics.jsonl").read_bytes().splitlines()
    assert len(prefix) == 2340
    assert [json.loads(row)["update"] for row in prefix] == list(range(1, 2341))
    old, new = events(prior), events(current)
    assert new.restore(trainer["sampler_state"], migrate_continuation_2340=True) == {
        "from_schema": "ember_operator_read_write_events_v7",
        "to_schema": "ember_operator_read_write_events_v8", "cursor": 2340,
        "appended_teacher_round": [20260928, 1, "task", 6]}
    for step in (0, 899, 1799, 2249, 2339):
        assert old.tasks_for_step(step) == new.tasks_for_step(step)
        for task in old.tasks_for_step(step):
            assert old.event(step, task) == new.event(step, task)
    for task in TASKS:
        for visit in (260, 299, 300, 309):
            step = next(s for s in range(visit * 9, visit * 9 + 9)
                        if task in new.tasks_for_step(s))
            event = new.event(step, task)
            order = np.random.default_rng(np.random.SeedSequence(
                [20260928, 1, task, 5 if visit < 300 else 6])).permutation(50)
            assert event["teacher_demo"] == int(order[visit % 50])
            assert len(event["queries"]) == 28
            assert all(q["demo"] != event["teacher_demo"] for q in event["queries"])


def test_only_new_train_entry_and_parent_replay_rejection(tmp_path, monkeypatch):
    spec = run.specification(run.CONTINUATION2790_SPEC_PATH)
    run.validate_train_request(spec, args())
    with pytest.raises(ValueError, match="T2790"):
        run.validate_train_request(run.specification(run.CONTINUATION2340_SPEC_PATH), args())
    with pytest.raises(ValueError, match="intermediate"):
        run.validate_train_request(spec, SimpleNamespace(**{
            **vars(args()), "stop_after_macro": 2350}))
    old = bank.inspect_training_source(run.specification(run.CONTINUATION2340_SPEC_PATH),
                                       PARENT, "T", sealed_evaluation=True)
    contract = {**old, "git": {"commit": "new-pushed-freeze", "branch": "",
                               "dirty_paths": [],
                               "pushed_ref": "origin/codex/demonstration-transfer"},
                "spec": str(run.CONTINUATION2790_SPEC_PATH),
                "events": spec["events"], "continuation": spec["continuation"],
                "sampler": {k: v for k, v in events(spec).sampler_state().items()
                            if k != "next_step"},
                "parent_checkpoint": str(PARENT), "loss_variant": "full"}
    trial = spec | {"run_root": str(tmp_path)}
    output = tmp_path / "T/train/attempts/continuation"
    run.validate_attempt(trial, args(), contract, output)
    with pytest.raises(ValueError, match="unregistered continuation parent or arm"):
        run.validate_attempt(trial, args(), contract | {"loss_variant": "wrong"}, output)
    with pytest.raises(ValueError, match="registered complete"):
        run.validate_attempt(trial, args(PARENT.parent / "macro_00002250"), contract, output)
    later = tmp_path / "T/train/attempts/prior/checkpoints/macro_00002430"
    later.mkdir(parents=True)
    real_complete = run.complete_checkpoint
    monkeypatch.setattr(run, "complete_checkpoint", lambda p: p == later or real_complete(p))
    with pytest.raises(ValueError, match="latest complete"):
        run.validate_attempt(trial, args(), contract, output)
    write_json_atomic(later.parent.parent / "run_contract.json", contract)
    run.validate_attempt(trial, args(later), contract, output)
    write_json_atomic(later.parent.parent / "run_contract.json",
                      contract | {"git": {**contract["git"], "commit": "wrong"}})
    with pytest.raises(ValueError, match="frozen Git changed"):
        run.validate_attempt(trial, args(later), contract, output)


def test_actual_2430_bank_accepts_full_training_state_and_rejects_wrong_arm():
    spec = run.specification(run.CONTINUATION2790_SPEC_PATH)
    checkpoint = (run.CONTINUATION2790_ROOT
                  / "T/train/attempts/continuation/checkpoints/macro_00002430")
    source = bank.inspect_training_source(spec, checkpoint, "T", sealed_evaluation=True)
    assert source["loss_variant"] == "full"
    assert source["git"] == bank.CONTINUATION2790_TRAINING_GIT
    with pytest.raises(ValueError, match="same-arm"):
        bank.inspect_training_source(spec, checkpoint, "U", sealed_evaluation=True)
