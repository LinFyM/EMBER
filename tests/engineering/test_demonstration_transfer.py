"""CPU contracts for the fixed train-only demonstration transfer."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ember.demonstration_transfer import (_authorities, _extract_one, _interpolate,
                                          _inverse_osc, _restore_scene, _target)


def test_fixed_sources_restore_without_source_steps():
    repo = Path(__file__).resolve().parents[2]
    rows, _, paths = _authorities(repo)
    expected = {(34, 0): (59, 116, 194), (34, 1): (64, 129, 217),
                (38, 0): (130, 184, 369), (38, 1): (134, 191, 363)}
    for task, demo in expected:
        metadata, arrays = _extract_one(repo, rows[task], paths, demo)
        assert tuple(metadata["source_boundaries"].values()) == expected[task, demo]
        assert metadata["source_steps_executed"] == 0
        assert arrays["goal_pos"].shape == (metadata["steps"], 3)
        assert np.isfinite(arrays["goal_rot"]).all()
        assert sum(s["stop"] - s["start"] for s in metadata["segments"]) == metadata["steps"]


def test_reference_transform_and_inverse_osc_use_left_rotation():
    source_rot = Rotation.from_euler("xyz", [.2, -.1, .3]).as_matrix()
    new_rot = Rotation.from_euler("xyz", [-.4, .15, .1]).as_matrix()
    target_rot = Rotation.from_euler("xyz", [.1, .2, -.1]).as_matrix()
    source_pos = np.array([.1, .2, .3])
    new_pos = np.array([-.1, .4, .6])
    target_pos = np.array([.12, .18, .35])
    p, r = _target(source_pos, source_rot, new_pos, new_rot, target_pos, target_rot)
    np.testing.assert_allclose(new_rot.T @ (p - new_pos), source_rot.T @ (target_pos - source_pos))
    np.testing.assert_allclose(new_rot.T @ r, source_rot.T @ target_rot)
    start_rot = Rotation.from_rotvec([-.1, .1, -.05]).as_matrix() @ r
    class Controller:
        ee_pos = p + np.array([.01, -.02, .03])
        ee_ori_mat = start_rot
        output_max = np.array([.05, .05, .05, .5, .5, .5])

        def update(self, force):
            assert force

    action = _inverse_osc(Controller(), p, r, -1)
    np.testing.assert_allclose(action[:3] * .05 + Controller.ee_pos, p)
    np.testing.assert_allclose(Rotation.from_rotvec(action[3:6] * .5).as_matrix() @ start_rot, r)
    assert action[6] == -1
    connection = _interpolate(Controller.ee_pos, start_rot, p, r)
    assert len(connection) == 10
    np.testing.assert_allclose(connection[-1][0], p)
    np.testing.assert_allclose(connection[-1][1], r)


def test_scene_restore_copies_all_body_poses_and_settled_sim_state():
    class Model:
        nbody = 2
        body_pos = np.array([[0., 0., 0.], [8., 9., 10.]])
        body_quat = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])

        def body_id2name(self, index):
            return ("world", "fixture")[index]

    class Sim:
        model = Model()

    class Owner:
        sim = Sim()

    class Env:
        env = Owner()

        def regenerate_obs_from_state(self, state):
            self.restored_state = state.copy()
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
