"""Actual parent and bounded CPU consumers for the sole T1350→1800 continuation."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ember.operator_writer import bank
from ember.operator_writer.data import FormalData, TASKS
from ember.operator_writer.run import (CONTINUATION1350_SPEC_PATH, CONTINUATION1800_SPEC_PATH,
                                       complete_checkpoint, specification, validate_attempt)
from ember.pi05_source_checkpoint import read_json, write_json_atomic


ASSET = Path("/data1/user/ymdai/projects/EMBER")
PARENT = Path("/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928"
              "/continuation1350/T/train/attempts/continuation/checkpoints/macro_00001350")
GIT = {"commit": "new-pushed-frozen", "branch": "", "dirty_paths": [],
       "pushed_ref": "origin/codex/demonstration-transfer"}


def _contract(spec, sampler):
    original = read_json(PARENT.parent.parent / "run_contract.json")
    return {**original, "git": GIT, "spec": str(CONTINUATION1800_SPEC_PATH),
            "events": spec["events"], "sampler": sampler,
            "continuation": spec["continuation"], "parent_checkpoint": str(PARENT),
            "stop_after_macro": None}


def _checkpoint(path, macro, sampler):
    path.mkdir(parents=True)
    files = {name: {"bytes": 1} for name in
             ("ecp.safetensors", "rank_00_state.pt", "rank_01_state.pt")}
    for name in files:
        (path / name).write_bytes(b"x")
    torch.save({"schema_version": "ember_ecp_checkpoint_v1",
                "stage": "operator_read_write_learning", "next_macro": macro,
                "metrics_rows": macro, "optimizer": {"param_groups": [{"lr": 1e-5}]},
                "scheduler": {"last_epoch": macro}, "scaler": None,
                "training_state": {"updates": macro, "mode": "T"},
                "sampler_state": sampler | {"next_step": macro}},
               path / "trainer_state.pt")
    files["trainer_state.pt"] = {"bytes": (path / "trainer_state.pt").stat().st_size}
    write_json_atomic(path / "checkpoint_manifest.json", {
        "stage": "operator_read_write_learning",
        "run_contract_schema": "ember_operator_read_write_formal_run_v1",
        "next_macro": macro, "world_size": 2, "files": files})
    assert complete_checkpoint(path)


def test_real_1350_source_migration_and_fourth_round():
    old_spec = specification(CONTINUATION1350_SPEC_PATH)
    spec = specification(CONTINUATION1800_SPEC_PATH)
    assert spec["execution"]["modes"] == ["T"]
    assert spec["execution"]["checkpoints"] == [1440, 1530, 1620, 1710, 1800]
    assert spec["optimization"]["floor_lr"] == 1e-5
    old = FormalData(ASSET, old_spec, query_labels=False)
    new = FormalData(ASSET, spec, query_labels=False)
    try:
        for step in (0, 269, 449, 899, 1349):
            assert new.tasks_for_step(step) == old.tasks_for_step(step)
            for task in old.tasks_for_step(step):
                assert new.event(step, task) == old.event(step, task)
        for task in TASKS:
            observed = []
            for visit in range(150, 200):
                step = next(step for step in range(visit * 9, visit * 9 + 9)
                            if task in new.tasks_for_step(step))
                event = new.event(step, task)
                assert event["visit"] == visit and len(event["queries"]) == 28
                assert all(row["demo"] != event["teacher_demo"] for row in event["queries"])
                observed.append(event["teacher_demo"])
            expected = np.random.default_rng(
                np.random.SeedSequence([20260928, 1, task, 3])).permutation(50).tolist()
            assert observed == expected and len(set(observed)) == 50
        trainer = torch.load(PARENT / "trainer_state.pt", map_location="meta", mmap=True,
                             weights_only=True)
        assert trainer["next_macro"] == trainer["metrics_rows"] == 1350
        assert trainer["scheduler"]["last_epoch"] == 1350
        assert trainer["optimizer"]["param_groups"]
        migration = new.restore(trainer["sampler_state"], migrate_continuation_1350=True)
        assert migration == {
            "from_schema": "ember_operator_read_write_events_v4",
            "to_schema": "ember_operator_read_write_events_v5", "cursor": 1350,
            "appended_teacher_round": [20260928, 1, "task", 3]}
        with pytest.raises(ValueError, match="migration source"):
            new.restore(trainer["sampler_state"] | {"next_step": 1349},
                        migrate_continuation_1350=True)
        prefix = [json.loads(line) for line in
                  (PARENT.parent.parent / "metrics.jsonl").read_text().splitlines()]
        assert len(prefix) == 1350
        assert [row["update"] for row in prefix] == list(range(1, 1351))
        parent = bank.inspect_training_source(old_spec, PARENT, "T", sealed_evaluation=True)
        assert parent["git"]["commit"] == spec["continuation"]["parent_training_git"]
    finally:
        old.close()
        new.close()


def test_latest_ecp_and_selected_banks_reject_wrong_lineage(tmp_path, monkeypatch):
    spec = {**specification(CONTINUATION1800_SPEC_PATH), "run_root": str(tmp_path)}
    data = FormalData(ASSET, spec, query_labels=False)
    try:
        sampler = {key: value for key, value in data.sampler_state().items()
                   if key != "next_step"}
        contract = _contract(spec, sampler)
        args = SimpleNamespace(mode="T", resume=PARENT)
        first = tmp_path / "T/train/attempts/continuation"
        validate_attempt(spec, args, contract, first)
        with pytest.raises(ValueError, match="arm"):
            validate_attempt(spec, SimpleNamespace(mode="U", resume=PARENT), contract,
                             tmp_path / "U/train/attempts/wrong")
        with pytest.raises(ValueError):
            validate_attempt(spec, args, contract | {"source": {"checkpoint": "/wrong"}}, first)
        write_json_atomic(first / "run_contract.json", contract)
        write_json_atomic(first / "resume_provenance.json", {"checkpoint": str(PARENT)})
        prefix = (PARENT.parent.parent / "metrics.jsonl").read_text().splitlines()
        for macro in (1710, 1800):
            checkpoint = first / "checkpoints" / f"macro_{macro:08d}"
            _checkpoint(checkpoint, macro, sampler)
            (first / "metrics.jsonl").write_text("\n".join(
                prefix + [json.dumps({"update": step}) for step in range(1351, macro + 1)])
                + "\n")
            monkeypatch.setattr(bank, "CONTINUATION1800_TRAINING_GIT", GIT)
            monkeypatch.setattr(bank, "CONTINUATION1800_FROZEN_SPEC_PATH",
                                CONTINUATION1800_SPEC_PATH)
            assert bank.inspect_training_source(spec, checkpoint, "T",
                                                sealed_evaluation=True) == contract
        with pytest.raises(ValueError, match="latest complete"):
            validate_attempt(spec, args, contract, tmp_path / "T/train/attempts/replay1350")
        with pytest.raises(ValueError, match="latest complete"):
            validate_attempt(spec, SimpleNamespace(mode="T", resume=first /
                             "checkpoints/macro_00001710"), contract,
                             tmp_path / "T/train/attempts/replay1710")
        bad = contract | {"git": {"commit": "wrong"}}
        write_json_atomic(first / "run_contract.json", bad)
        with pytest.raises(ValueError, match="source/ECP"):
            bank.inspect_training_source(spec, first / "checkpoints/macro_00001800", "T",
                                         sealed_evaluation=True)
        assert bank.EVALUATION_SPEC_PATHS[1710] == bank.EVALUATION_SPEC_PATHS[1800]
        assert bank.EVALUATION_SPEC_PATHS[1800] == CONTINUATION1800_SPEC_PATH
        with pytest.raises(ValueError, match="registered T ECP"):
            bank.materialize("T", PARENT, ASSET, torch.device("cpu"))
    finally:
        data.close()


def test_sealed_1350_bank_remains_readable():
    path = PARENT.parents[4] / "banks/1350/manifest.json"
    manifest = read_json(path)
    accepted = bank.inspect_bank(
        manifest_path=path, source=manifest["source"],
        task_keys=tuple((row["suite"], row["task_id"]) for row in manifest["tasks"]),
        evaluation_role="validation", require_formal=True)
    assert accepted["training_git"] == "14bac4cdd6c27eee06f5574317da8257834e3884"
    assert len(accepted["conditions"]) == 400
