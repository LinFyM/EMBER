"""The frozen intervention replays actual actions and retains the global RNG clock."""

from pathlib import Path

import numpy as np
import pytest
import torch

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.frozen_prefix import _panel_scope
from ember.pi05_eval.prefix_replay import _state_error, replay_prefix
from ember.pi05_eval.trajectory_capture import initialize_capture
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
