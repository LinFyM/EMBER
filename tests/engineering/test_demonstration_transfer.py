"""CPU contracts for the fixed train-only demonstration transfer."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ember.demonstration_transfer import _interpolate, _inverse_osc, _restore_scene, _target
from ember.demonstration_transfer_source import (SourceStructureUnsupported,
                                                 movement_goals, segment_boundaries)


def test_uniform_movement_roles_and_boundaries():
    class Model:
        def body_name2id(self, name):
            if name == "main_table":
                return 3
            raise ValueError(name)

    class Sim:
        model = Model()

    class Owner:
        sim = Sim()
        obj_body_id = {"bowl": 0, "mug": 1, "plate": 2}
        objects_dict = {"bowl": object(), "mug": object()}
        parsed_problem = {"regions": {"main_table_zone": {"target": "main_table"}}}

    goals = [["on", "bowl", "plate"], ["in", "mug", "main_table_zone"],
             ["turnon", "fixture"]]
    movements = movement_goals(Owner(), goals, np.array([False, False, True]))
    assert [item["reference_root"] for item in movements] == ["plate", "main_table"]
    names = ["bowl", "mug", "plate", "main_table"]
    positions = np.zeros((8, len(names), 3))
    positions[2:, 0, 2] = .04
    positions[5:, 1, 2] = .04
    predicates = np.zeros((8, len(goals)), dtype=bool)
    predicates[3:, 0] = True
    actions = np.ones((8, 7))
    actions[3:, 6] = -1
    ordered, segments = segment_boundaries(movements, names, positions, predicates, actions)
    assert [(item["object"], item["lift"], item["release"]) for item in ordered] == [
        ("bowl", 2, 3), ("mug", 5, None)]
    assert [(item["start"], item["stop"], item["reference_body"]) for item in segments] == [
        (0, 2, "bowl"), (2, 4, "plate"), (4, 5, "mug"), (5, 8, "main_table")]
    predicates[3:, 0] = False
    with pytest.raises(SourceStructureUnsupported, match="release"):
        segment_boundaries(movements, names, positions, predicates, actions)


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
