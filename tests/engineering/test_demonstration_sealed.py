"""CPU contracts for retained formal bank and scene readers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ember.demonstration_learning.bank import (
    ASSET_ROOT, STUDY, _inspect_bank_scope, _task_rows, concatenate_factors,
    inspect_formal_provenance, registered_capture,
)
from ember.pi05_eval.scene import _body_id, _pose, _restore_scene, validate_scene_row
from ember.pi05_source_checkpoint import read_json


def test_four_sealed_banks_keep_frozen_spec_checkpoint_git_and_capture() -> None:
    for stage, macro in (("stage1", 288), ("stage2", 576)):
        for arm in ("P", "I"):
            path = STUDY / stage / arm / "banks" / str(macro) / "manifest.json"
            bank = read_json(path)
            spec, checkpoint, run = inspect_formal_provenance(bank, path, bank["source"])
            assert checkpoint.name == f"macro_{macro:08d}"
            assert spec["bank"]["source_macro"] == macro
            assert run["git"]["commit"] == bank["training_git"]
            expected_tasks, expected_conditions = _task_rows(spec, ASSET_ROOT)
            _inspect_bank_scope(bank, bank["source"],
                                tuple((row["suite"], row["task_id"]) for row in expected_tasks),
                                {(row["suite"], row["task_id"]): tuple(range(50))
                                 for row in expected_tasks}, "validation", checkpoint, run, expected_tasks)
            stripped = [{k: v for k, v in row.items()
                         if k not in ("factors", "raw_frames", "sampled_frames")}
                        for row in bank["conditions"]]
            assert stripped == expected_conditions
            selector = STUDY / stage / "selectors" / f"{arm}_capture.json"
            tasks = [SimpleNamespace(suite=row["suite"], task_id=row["task_id"],
                                     init_state_ids=tuple(range(50))) for row in bank["tasks"]]
            args = SimpleNamespace(static_task_lora_manifest=path, role="validation", mode="formal")
            capture, predicates = registered_capture(
                args, tasks, STUDY / stage / arm / "evaluation" / "correct400",
                selector, read_json(selector), None,
            )
            assert len(capture["full_conditions"]) == 8
            assert predicates["validation_action_reads"] == 0
            with pytest.raises(ValueError):
                inspect_formal_provenance({**bank, "training_git": "wrong"}, path, bank["source"])
            with pytest.raises(ValueError):
                inspect_formal_provenance(bank, path, {"checkpoint": "wrong"})
            with pytest.raises(ValueError):
                inspect_formal_provenance(bank, path.parent.parent / "other" / "manifest.json",
                                          bank["source"])


def test_sealed_rank144_is_one_complete_adapter_without_cross_terms() -> None:
    common, conditional = {}, {}
    for index in range(38):
        stem = f"target_{index}"
        a, b = stem + ".lora_A.default.weight", stem + ".lora_B.default.weight"
        common[a], conditional[a] = torch.randn(128, 5), torch.randn(16, 5)
        common[b], conditional[b] = torch.randn(3, 128), torch.randn(3, 16)
    complete = concatenate_factors(common, conditional)
    assert len(complete) == 76
    for index in range(38):
        stem = f"target_{index}"
        a, b = stem + ".lora_A.default.weight", stem + ".lora_B.default.weight"
        assert complete[a].shape == (144, 5) and complete[b].shape == (3, 144)
        torch.testing.assert_close(complete[b] @ complete[a],
                                   common[b] @ common[a] + conditional[b] @ conditional[a],
                                   atol=1e-4, rtol=1e-5)
    changed = dict(conditional)
    changed["target_0.lora_A.default.weight"] = torch.empty(15, 5)
    with pytest.raises(ValueError, match="shapes differ"):
        concatenate_factors(common, changed)


def test_scene_restore_and_registered_row_contract() -> None:
    class Model:
        nbody = 2
        body_pos = np.array([[0., 0., 0.], [8., 9., 10.]])
        body_quat = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])

        def body_id2name(self, index):
            return ("world", "fixture")[index]

    class Env:
        env = SimpleNamespace(sim=SimpleNamespace(model=Model()))

        def regenerate_obs_from_state(self, state):
            return {"state": state.copy()}

    snapshot = {"model_body_names": np.array(["world", "fixture"]),
                "model_body_pos": np.array([[0., 0., 0.], [.2, .3, .4]]),
                "model_body_quat": np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
                "sim_state": np.array([1., 2., 3.])}
    env = Env()
    observation = _restore_scene(env, snapshot)
    np.testing.assert_array_equal(env.env.sim.model.body_pos, snapshot["model_body_pos"])
    np.testing.assert_array_equal(env.env.sim.model.body_quat, snapshot["model_body_quat"])
    np.testing.assert_array_equal(observation["state"], snapshot["sim_state"])

    scene = read_json(STUDY / "scenes" / "manifest.json")["scenes"][0]
    row = {"init_state_id": scene["state"], "scene_reference": {
        "path": scene["path"], "bytes": scene["bytes"],
        "restoration": "full_model_body_pose_post_dummy_sim_controller_and_dual_rgb_verified"}}
    contract = {"demonstration_comparison_scene": {"root": str(STUDY / "scenes")}}
    task = {"suite": scene["suite"], "task_id": scene["task_id"]}
    validate_scene_row(row, task, contract)
    with pytest.raises(ValueError, match="scene row"):
        validate_scene_row({**row, "scene_reference": {**row["scene_reference"], "bytes": 0}},
                           task, contract)


def test_scene_body_lookup_keeps_exact_arena_reference() -> None:
    class Model:
        def body_name2id(self, name):
            if name == "fixture":
                return 1
            raise ValueError(name)

    data = SimpleNamespace(body_xpos=np.array([[1., 2., 3.], [4., 5., 6.]]),
                           body_xmat=np.tile(np.eye(3).reshape(1, 9), (2, 1)))
    owner = SimpleNamespace(obj_body_id={"mug": 0}, sim=SimpleNamespace(model=Model(), data=data))
    assert _body_id(owner, "mug") == 0
    assert _body_id(owner, "fixture") == 1
    position, rotation = _pose(owner.sim, _body_id(owner, "fixture"))
    np.testing.assert_array_equal(position, [4., 5., 6.])
    np.testing.assert_array_equal(rotation, np.eye(3))
    with pytest.raises(ValueError, match="scene body absent"):
        _body_id(owner, "missing")
