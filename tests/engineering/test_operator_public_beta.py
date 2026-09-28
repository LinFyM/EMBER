"""Real T1800 public-factor export and canonical evaluator CPU consumers."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from ember.eval_adapters import inspect_static_task_lora_adapter
from ember.operator_writer import bank
from ember.operator_writer.public_beta import inspect as inspect_public_beta
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.preparation import _registered_trajectory_capture
from ember.pi05_source_checkpoint import read_json


ASSET = Path("/data1/user/ymdai/projects/EMBER")
OLD = Path("/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928")
ECP = OLD / "continuation1800/T/train/attempts/continuation/checkpoints/macro_00001800"


def _request(path):
    manifest = read_json(path)
    tasks = tuple(SimpleNamespace(suite=row["suite"], task_id=row["task_id"],
                                  init_state_ids=tuple(range(50))) for row in manifest["tasks"])
    return manifest, tasks, dict(manifest_path=path, source=manifest["source"],
                                 tasks=tasks, evaluation_role="validation", require_formal=True)


def test_real_public_beta_and_registered_shared_consumer(tmp_path, monkeypatch):
    monkeypatch.setattr(bank, "PUBLIC_BETA_ROOT", tmp_path)
    path = bank.materialize_public_beta(ECP, ASSET)
    manifest, tasks, request = _request(path)
    assert manifest["training_git"] == bank.CONTINUATION1800_TRAINING_GIT["commit"]
    assert manifest["mode"] == "public_beta" and len(manifest["conditions"]) == 400
    assert manifest["information_wall"]["teacher_video_values_read"] == 0
    assert {row["teacher_demo"] for row in manifest["conditions"]} == set(range(50))
    assert len(list(path.parent.glob("*.safetensors"))) == 1

    adapter = inspect_static_task_lora_adapter(**request)
    assert adapter["arm"] == adapter["mode"] == "public_beta"
    state = load_file(adapter["shared"]["path"], device="cpu")
    with safe_open(str(ECP / "ecp.safetensors"), framework="pt", device="cpu") as reader:
        for row in manifest["public_intervention"]["factor_map"]:
            assert torch.equal(state[row["lora_name"]], reader.get_tensor(row["ecp_name"]).float())
    consumer = SimpleNamespace(bank=adapter, common=state)
    assert bank.FrozenOperatorAdapter._state(consumer, manifest["conditions"][0]["condition_id"]) is state
    assert bank.FrozenOperatorAdapter._state(consumer, manifest["conditions"][1]["condition_id"]) is state
    episode = bank.episode_evidence(adapter, adapter["tasks"][0], adapter["tasks"][0]["episodes"][0])
    assert episode["intervention"] == "public_B0_A"
    assert episode["teacher_video_values_read"] == 0

    output = tmp_path / "public_beta/evaluation/1800/correct400"
    args = SimpleNamespace(static_task_lora_manifest=path, role="validation", mode="formal",
                           trajectory_capture_selection=bank.PUBLIC_BETA_CAPTURE_PATH)
    capture, stage = _registered_trajectory_capture(args, tasks, output, None, ASSET)
    assert len(capture["full_conditions"]) == 8 and stage["full_conditions_only"] is False
    with pytest.raises(Pi05EvaluationError):
        inspect_static_task_lora_adapter(**(request | {
            "source": manifest["source"] | {"checkpoint": "/wrong/source"}}))
    with pytest.raises(ValueError, match="fixed complete T1800"):
        bank.materialize_public_beta(ECP.parent.parent.parent.parent / "U/train/checkpoints/macro_00001800", ASSET)
    with pytest.raises(ValueError, match="same-arm"):
        bank.inspect_training_source(bank.specification(bank.CONTINUATION1800_SPEC_PATH),
                                     ECP, "U", sealed_evaluation=True)
    # A valid ECP from this same training window is still outside this diagnostic.
    with pytest.raises(ValueError, match="fixed complete T1800"):
        inspect_public_beta(manifest | {"checkpoint": str(ECP.with_name("macro_00001710"))},
                            path, manifest["source"],
                            tuple((task.suite, task.task_id) for task in tasks),
                            "validation", True, None)


def test_old_full_t1800_and_mt_consumers_remain_sealed():
    for path in (OLD / "continuation1800/T/banks/1800/manifest.json",
                 OLD / "stage1/MT/banks/300/manifest.json"):
        manifest, _, request = _request(path)
        inspected = inspect_static_task_lora_adapter(**request)
        assert inspected["mode"] == manifest["mode"] and inspected["arm"] == "correct"
        assert len(inspected["conditions"]) == 400
