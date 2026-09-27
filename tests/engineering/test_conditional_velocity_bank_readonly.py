"""Small CPU checks for the retained historical bank reader; no model run."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.pi05_assets import Pi05EvaluationError
from ember.writer.conditional_velocity_bank import (
    BANK_KIND, PASSIVE_TAG, SPEC_SCHEMA, STUDY_ROOT, _bank_spec,
    _expected_checkpoint, attach_capture_provenance, compile_velocity_state,
    registered_capture, validate_capture_contract,
)
from ember.writer.materialization import file_record


def test_sealed_factors_rebuild_one_rank135_state():
    common = {}
    for index in range(37):
        name = f"example.{index}."
        common[name + "lora_A.default.weight"] = torch.randn(128, 2)
        common[name + "lora_B.default.weight"] = torch.randn(3, 128)
    name = "model.action_out_proj."
    common[name + "lora_A.default.weight"] = torch.randn(128, 1024)
    common[name + "lora_B.default.weight"] = torch.randn(32, 128)
    R, U = torch.randn(7, 256), torch.randn(256, 1024)
    state = compile_velocity_state(common, R, U)
    assert len(state) == 76
    for prefix in [*(f"example.{index}." for index in range(37)), name]:
        a, b = prefix + "lora_A.default.weight", prefix + "lora_B.default.weight"
        expected = common[b] @ common[a]
        if prefix == name:
            expected = expected.clone()
            expected[:7] += R @ U
        assert torch.allclose(state[b] @ state[a], expected, atol=1e-4, rtol=1e-5)
    with pytest.raises(ValueError, match="coefficient"):
        compile_velocity_state(common, R[:6], U)


def test_frozen_spec_and_capture_are_read_only_authorities(tmp_path):
    frozen = tmp_path / "frozen"
    spec_path = frozen / "configs/conditional_velocity_operator_v1/learning_spec.json"
    spec_path.parent.mkdir(parents=True)
    spec = {"schema_version": SPEC_SCHEMA, "run_root": str(STUDY_ROOT),
            "first_stop_update": 270, "continuation_450": {
                "qualification_update": 450, "subdir": "continuation_450"}}
    spec_path.write_text(json.dumps(spec))
    run_path = tmp_path / "V/run_contract.json"
    run_path.parent.mkdir()
    run_path.write_text(json.dumps({"git": {"commit": "sealed"}}))
    bank_path = tmp_path / "bank.json"
    bank = {"kind": BANK_KIND, "mode": "V", "spec": file_record(spec_path),
            "training_run": file_record(run_path)}
    bank_path.write_text(json.dumps(bank))
    assert _bank_spec(bank) == (spec, frozen)
    assert _expected_checkpoint(spec, "V", 270)[0] == STUDY_ROOT / "V/checkpoints/macro_00000270"
    assert _expected_checkpoint(spec, "V", 450)[0] == STUDY_ROOT / "continuation_450/V/checkpoints/macro_00000450"
    with pytest.raises(ValueError, match="outside"):
        _expected_checkpoint(spec, "V", 630)

    selection_path = frozen / "configs/conditional_velocity_operator_v1/official_capture.json"
    full = [{"suite": suite, "task_id": task, "init_state_id": 0}
            for suite, task in (("libero_spatial", 3), ("libero_spatial", 6),
                                ("libero_object", 1), ("libero_object", 6),
                                ("libero_goal", 3), ("libero_goal", 6),
                                ("libero_10", 1), ("libero_10", 9))]
    selection = {"schema_version": "ember_pi05_registered_trajectory_capture_v1",
                 "mode": "compact", "full_conditions": full,
                 "passive_control_trace": PASSIVE_TAG, "stage_predicates": True,
                 "training_gradient_use": False, "checkpoint_selection_use": False,
                 "validation_use": False, "test_use": False}
    selection_path.write_text(json.dumps(selection))
    tasks = [SimpleNamespace(**row, init_state_ids=tuple(range(50))) for row in full]
    output_dir = run_path.parent / "evaluation/correct400"
    args = SimpleNamespace(role="validation", static_task_lora_manifest=bank_path)
    capture, stage = registered_capture(args, tasks, output_dir, selection_path,
                                        selection, None, tmp_path, selection["schema_version"])
    assert capture["passive_trace"]["spec_path"] == str(spec_path)
    assert capture["passive_trace"]["spec_bytes"] == spec_path.stat().st_size
    contract = {"role": "validation", "output_dir": str(output_dir),
                "tasks": [vars(task) for task in tasks], "git": {"commit": "sealed"},
                "adapter": {"kind": BANK_KIND, "manifest": file_record(bank_path),
                            "checkpoint": {}},
                "diagnostic_occupancy_capture": capture,
                "diagnostic_stage_predicates": stage}
    attach_capture_provenance(contract, tmp_path)
    validate_capture_contract(contract, tmp_path)
    contract["git"]["commit"] = "other"
    with pytest.raises(Pi05EvaluationError, match="identity"):
        attach_capture_provenance(contract, tmp_path)
    spec_path.write_text("{}")
    with pytest.raises(ValueError, match="frozen spec"):
        _bank_spec(bank)
