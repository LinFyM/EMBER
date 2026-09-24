"""The frozen intervention replays actual actions and retains the global RNG clock."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.frozen_prefix import _panel_scope
from ember.pi05_eval.approach_channel import _panel_scope as _approach_panel_scope
from ember.pi05_eval.approach_channel_replay import compose_prefix_commands
from ember.pi05_eval.contact_trace import audit_contact_geometries, sample_robot_object_contacts
from ember.pi05_eval.prefix_replay import _state_error, replay_prefix
from ember.pi05_eval.trajectory_capture import initialize_capture, record_replan, save_capture
from ember.pi05_eval_contract import policy_noise_seed


def test_registered_panel_split_covers_self_and_cross_without_overlap():
    spec = {"study_id": "frozen_prefix_causality_20260924", "anchors": ["B", "C_correct"],
            "followers": ["B", "C_correct", "C_other", "C_wrong"],
            "cut_control_steps": [25, 50], "state_ids": list(range(50))}
    common = {"schema_version": "ember_frozen_prefix_panel_v1", "study_id": spec["study_id"],
              "anchor": "B", "follower": "B", "cut_control_steps": 25,
              "training_gradient_use": False, "checkpoint_selection_use": False,
              "validation_use": False, "test_use": False}
    pilot = {**common, "phase": "pilot", "state_ids": [0, 25]}
    remaining = {**common, "phase": "remaining",
                 "state_ids": [state for state in range(50) if state not in (0, 25)]}
    assert _panel_scope(pilot, spec)[-1] == (0, 25)
    assert set(_panel_scope(pilot, spec)[-1]).isdisjoint(_panel_scope(remaining, spec)[-1])
    with pytest.raises(Pi05EvaluationError, match="coverage changed"):
        _panel_scope({**pilot, "state_ids": [0]}, spec)


def test_saved_state_error_uses_registered_component_tolerances():
    saved = torch.zeros(1, 8)
    actual = saved.clone()
    actual[0, 3] = 0.0009
    assert _state_error(actual, saved, step=5)["axis_angle_abs_max"] == pytest.approx(0.0009)
    actual[0, 6] = 0.00011
    with pytest.raises(Pi05EvaluationError, match="diverged"):
        _state_error(actual, saved, step=5)


class _FakeEnv:
    def __init__(self):
        self.actions = []
        self.step_number = 0

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.step_number += 1
        return {"step": self.step_number}, 0.0, False, {}

    def get_sim_state(self):
        return np.asarray([self.step_number], dtype=np.float64)


def test_replay_uses_executed_actions_and_starts_tail_at_global_fifth_replan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import ember.pi05_eval.prefix_replay as prefix
    import ember.pi05_eval.episode as episode

    suite, task_id, state_id = "libero_goal", 1, 0
    path = tmp_path / "original.pt"
    executed = [torch.full((5, 7), float(index + 1)) for index in range(10)]
    seeds = [policy_noise_seed(7, suite, task_id, state_id, index) for index in range(10)]
    torch.save({
        "schema_version": "ember_pi05_diagnostic_trajectory_v2", "suite": suite,
        "task_id": task_id, "init_state_id": state_id, "success": False, "steps": 50,
        "replan_steps": tuple(range(0, 50, 5)), "policy_noise_seeds": tuple(seeds),
        "states": tuple(torch.full((1, 8), float(step)) for step in range(0, 50, 5)),
        "action_chunks": tuple(torch.zeros((1, 50, 32)) for _ in range(10)),
        "executed_action_prefixes": tuple(executed),
    }, path)
    reference = {"trajectory": {"path": str(path), "bytes": path.stat().st_size},
                 "success": False, "steps": 50, "policy_noise_seeds": seeds}
    contract = {"policy": {"replan_steps": 5}, "rng": {"inference_seed": 7}, "frozen_prefix_intervention": {
        "cut_control_steps": 25, "references": {f"{suite}:{task_id}:{state_id}": reference}}}
    slot = {"init_state_id": state_id, "obs": {"step": 0}, "steps": 0,
            "replan_index": 0, "policy_noise_seeds": [], "stage_predicate_last": (False,)}
    initialize_capture(slot, "compact")
    monkeypatch.setattr(prefix, "_roles", lambda env, task: (("object",), {"target": "object"}))
    monkeypatch.setattr(prefix, "_sample", lambda env, obs, slot, names: {
        "object_positions": np.zeros((1, 3), dtype=np.float32),
        "eef_pos": np.zeros(3, dtype=np.float32), "eef_quat": np.zeros(4, dtype=np.float32),
        "gripper_qpos": np.zeros(2, dtype=np.float32),
        "predicates": np.zeros(1, dtype=np.bool_),
    })
    monkeypatch.setattr(prefix, "_controller_state", lambda env: {})
    monkeypatch.setattr(prefix, "libero_policy_input", lambda obs, language: {
        "observation.state": torch.full((1, 8), float(obs["step"]))})
    monkeypatch.setattr(episode, "update_stage_predicates", lambda env, slot: None)
    env = _FakeEnv()
    replay_prefix(env=env, slot=slot, task={"suite": suite, "task_id": task_id,
                  "language": "put the bowl on the stove"}, contract=contract,
                  preprocess=lambda value: value)
    assert len(env.actions) == 25
    assert all(np.array_equal(action, np.full(7, float(index // 5 + 1)))
               for index, action in enumerate(env.actions))
    assert slot["replan_index"] == 5
    assert slot["policy_noise_seeds"] == seeds[:5]
    assert policy_noise_seed(7, suite, task_id, state_id, slot["replan_index"]) == seeds[5]
    assert slot["prefix_terminal"] is False


APPROACH_SPEC = json.loads((Path(__file__).resolve().parents[1] /
                   "configs/approach_channel_causality_v1/experiment_spec.json").read_text())


def _panel(prefix: str, follower: str, phase: str, states: list[int], task_ids: list[int]) -> dict:
    return {
        "schema_version": "ember_approach_channel_panel_v1",
        "study_id": APPROACH_SPEC["study_id"], "prefix": prefix, "follower": follower,
        "phase": phase, "state_ids": states, "task_ids": task_ids,
        "training_gradient_use": False, "checkpoint_selection_use": False,
        "validation_use": False, "test_use": False,
    }


def test_panel_scope_preserves_pilot_and_mixed_coverage():
    pilot = _panel("B_all", "C_correct", "pilot", [0, 25], [14, 21])
    mixed = _panel("B_xy_C_rest", "B", "mixed", list(range(50)), [14, 21])
    assert _approach_panel_scope(pilot, APPROACH_SPEC)[3] == (0, 25)
    assert _approach_panel_scope(mixed, APPROACH_SPEC)[3] == tuple(range(50))
    assert _approach_panel_scope(_panel("B_xy_C_rest", "C_correct", "smoke", [0], [21]), APPROACH_SPEC)[4] == (("libero_goal", 1),)
    with pytest.raises(Pi05EvaluationError):
        _approach_panel_scope({**mixed, "state_ids": list(range(49))}, APPROACH_SPEC)
    with pytest.raises(Pi05EvaluationError):
        _approach_panel_scope({**pilot, "validation_use": True}, APPROACH_SPEC)


def test_composition_keeps_exact_registered_environment_channels():
    offsets = np.arange(25, dtype=np.float32)[:, None]
    b = np.broadcast_to(offsets + np.arange(7, dtype=np.float32)[None, :], (25, 7)).copy()
    c = -b - 1
    donors = {"B": b, "C_correct": c}
    for name, spec in APPROACH_SPEC["prefixes"].items():
        sources = tuple(spec["channel_donors"])
        actual = compose_prefix_commands(donors, sources)
        for channel, donor in enumerate(sources):
            assert np.array_equal(actual[:, channel], donors[donor][:, channel]), name
    assert np.array_equal(compose_prefix_commands(donors, ("B",) * 7), b)
    with pytest.raises(Pi05EvaluationError):
        compose_prefix_commands({"B": b, "C_correct": np.full((25, 7), np.nan)}, ("B",) * 7)


def test_contact_identity_uses_geom_and_body_ancestry():
    class Model:
        ngeom = 4
        body_parentid = [0, 0, 1, 0, 3, 0]
        geom_bodyid = [2, 4, 5, 1]

        def body_name2id(self, name):
            return {"robot0_base": 1}[name]

        def body_id2name(self, index):
            return {1: "robot0_base", 2: "robot0_gripper", 3: "ketchup_root",
                    4: "ketchup_shell", 5: "basket_root"}[index]

        def geom_id2name(self, index):
            return ("robot_finger", "ketchup_geom", "basket_geom", "robot_arm")[index]

    data = SimpleNamespace(ncon=3, contact=[
        SimpleNamespace(geom1=0, geom2=1, dist=-0.002),
        SimpleNamespace(geom1=3, geom2=2, dist=0.005),
        SimpleNamespace(geom1=1, geom2=2, dist=-0.001),
    ])
    owner = SimpleNamespace(
        sim=SimpleNamespace(model=Model(), data=data),
        robots=[SimpleNamespace(robot_model=SimpleNamespace(root_body="robot0_base"))],
        obj_body_id={"ketchup_1": 3, "basket_1": 5},
    )
    indexed, audit = audit_contact_geometries(
        SimpleNamespace(env=owner), ("ketchup_1", "basket_1"))
    pairs = sample_robot_object_contacts(SimpleNamespace(env=owner), indexed)
    assert audit["robot_geom_count"] == 2
    assert audit["object_geom_counts"] == {"ketchup_1": 1, "basket_1": 1}
    assert [(p["object_name"], p["robot_body_name"], p["minimum_distance_m"])
            for p in pairs] == [("ketchup_1", "robot0_gripper", -0.002),
                               ("basket_1", "robot0_base", 0.005)]


def test_capture_marks_external_prefix_without_fake_prediction(tmp_path):
    slot = {"init_state_id": 0, "steps": 10, "policy_noise_seeds": [1, 2]}
    initialize_capture(slot, "compact")
    raw = {"observation.state": torch.zeros((1, 8))}
    record_replan(slot, raw, {}, None, np.ones((5, 7), dtype=np.float32),
                  command_kind="external_saved_action")
    record_replan(slot, raw, {}, torch.zeros((1, 50, 32)), np.zeros((5, 7), dtype=np.float32))
    task = {"suite": "libero_goal", "task_id": 1}
    capture = {"mode": "compact", "trajectory_root": str(tmp_path),
               "external_prefix_commands": True}
    result = save_capture(capture, task, slot, success=False)
    saved = torch.load(result["path"], map_location="cpu", weights_only=True)
    assert saved["schema_version"] == "ember_pi05_approach_channel_trajectory_v1"
    assert saved["command_kinds"] == ("external_saved_action", "model_prediction")
    assert saved["action_chunks"][0] is None
    assert torch.equal(saved["executed_action_prefixes"][0], torch.ones((5, 7)))
