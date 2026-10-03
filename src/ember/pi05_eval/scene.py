"""Restore and validate sealed full-scene starts for official evaluation."""

from __future__ import annotations

from typing import Any
from pathlib import Path
import time

import numpy as np

from ember.pi05_source_checkpoint import read_json


def _body_id(owner: Any, name: str) -> int:
    if name in owner.obj_body_id:
        return int(owner.obj_body_id[name])
    try:
        result = int(owner.sim.model.body_name2id(name))
        if result < 0:
            raise ValueError(name)
        return result
    except (ValueError, KeyError) as error:
        raise ValueError(f"registered scene body absent: {name}") from error


def _pose(sim: Any, body: int) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray(sim.data.body_xpos[body], dtype=np.float64).copy(),
            np.asarray(sim.data.body_xmat[body], dtype=np.float64).reshape(3, 3).copy())


def _query_sample(env: Any, observation: dict[str, Any], names: list[str], goals: list[list[str]]) -> dict[str, np.ndarray]:
    owner = env.env
    return {
        "body_pos": np.stack([_pose(owner.sim, _body_id(owner, name))[0] for name in names]),
        "body_rot": np.stack([_pose(owner.sim, _body_id(owner, name))[1] for name in names]),
        "eef_pos": np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy(),
        "eef_quat": np.asarray(observation["robot0_eef_quat"], dtype=np.float64).copy(),
        "gripper_qpos": np.asarray(observation["robot0_gripper_qpos"], dtype=np.float64).copy(),
        "predicates": np.asarray([owner._eval_predicate(goal) for goal in goals], dtype=np.bool_),
    }


def _capture_image(observation: dict[str, Any]) -> np.ndarray:
    image = np.stack([np.asarray(observation[name])[::-1, ::-1].copy()
                      for name in ("agentview_image", "robot0_eye_in_hand_image")])
    if image.shape != (2, 256, 256, 3) or image.dtype != np.uint8:
        raise ValueError("canonical dual-camera RGB changed")
    return image


def _scene_snapshot(env: Any, observation: dict[str, Any], names: list[str],
                    goals: list[list[str]], *, image: bool) -> dict[str, np.ndarray]:
    """All mutable scene positions plus the settled robot/controller state."""
    owner = env.env
    model = owner.sim.model
    controller = owner.robots[0].controller
    sample = _query_sample(env, observation, names, goals)
    snapshot = {
        "model_body_names": np.asarray([model.body_id2name(i) for i in range(model.nbody)]),
        "model_body_pos": np.asarray(model.body_pos, dtype=np.float64).copy(),
        "model_body_quat": np.asarray(model.body_quat, dtype=np.float64).copy(),
        "sim_state": np.asarray(env.get_sim_state(), dtype=np.float64).copy(),
        "controller_goal_pos": np.asarray(controller.goal_pos, dtype=np.float64).copy(),
        "controller_goal_ori": np.asarray(controller.goal_ori, dtype=np.float64).copy(),
        **{f"initial_{key}": value for key, value in sample.items()},
    }
    if image:
        snapshot["initial_rgb_canonical180"] = _capture_image(observation)
    return snapshot


def _restore_scene(env: Any, snapshot: dict[str, np.ndarray]) -> dict[str, Any]:
    """Rebuild the first reference's post-dummy physical state after a fresh reset."""
    model = env.env.sim.model
    actual_names = np.asarray([model.body_id2name(i) for i in range(model.nbody)])
    if not np.array_equal(actual_names, snapshot["model_body_names"]):
        raise ValueError("full-scene body registry changed between references")
    if (model.body_pos.shape != snapshot["model_body_pos"].shape
            or model.body_quat.shape != snapshot["model_body_quat"].shape):
        raise ValueError("full-scene model layout changed")
    model.body_pos[:] = snapshot["model_body_pos"]
    model.body_quat[:] = snapshot["model_body_quat"]
    return env.regenerate_obs_from_state(snapshot["sim_state"])


def _assert_scene_pair(env: Any, observation: dict[str, Any], names: list[str],
                       goals: list[list[str]], snapshot: dict[str, np.ndarray], *, image: bool) -> None:
    actual = _scene_snapshot(env, observation, names, goals, image=image)
    for key, expected in snapshot.items():
        observed = actual[key]
        if key == "model_body_names" or key == "initial_rgb_canonical180" or key == "initial_predicates":
            equal = np.array_equal(observed, expected)
        else:
            equal = observed.shape == expected.shape and np.allclose(observed, expected, rtol=0, atol=1e-8)
        if not equal:
            raise ValueError(f"restored full-scene start differs: {key}")


def scene_path(root: Path, task: dict, state: int) -> Path:
    return root / f"{task['suite']}_task_{int(task['task_id']):02d}_state_{state:03d}.npz"


def inspect_registered_scenes(root: Path, tasks: list[dict], *,
                              states: tuple[int, ...] = tuple(range(50)),
                              schema: str = "ember_demonstration_formal_scenes_v1") -> dict:
    """Check an exact sealed scene registry without loading simulator arrays."""
    registered = read_json(root / "manifest.json")
    rows = registered.get("scenes", ())
    expected = {(task["suite"], int(task["task_id"]), state)
                for task in tasks for state in states}
    actual = {(row["suite"], int(row["task_id"]), int(row["state"])) for row in rows}
    if (registered.get("schema_version") != schema
            or registered.get("seed") != 7 or registered.get("dummy_steps") != 10
            or len(rows) != len(expected) or actual != expected):
        raise ValueError("sealed common scene task/state registry changed")
    for row in rows:
        path = scene_path(root, row, int(row["state"]))
        if row["path"] != str(path) or path.stat().st_size != row["bytes"]:
            raise ValueError("sealed common scene file/path changed")
    return registered


def restore_registered_scene(env: Any, observation: dict, task: dict, state: int,
                             root: Path, *, diagnostic_output: Path | None = None) -> tuple[dict, dict]:
    path = scene_path(root, task, state)
    with np.load(path, allow_pickle=False) as sealed:
        snapshot = {name: sealed[name] for name in sealed.files}
    owner = env.env
    names = sorted(owner.obj_body_id)
    goals = [list(row) for row in owner.parsed_problem["goal_state"]]
    observation = _restore_scene(env, snapshot)
    try:
        _assert_scene_pair(env, observation, names, goals, snapshot, image=True)
    except ValueError as error:
        if str(error) != "restored full-scene start differs: initial_rgb_canonical180":
            raise
        # Re-render a stale observation without advancing or resetting physics.
        physical = {key: value for key, value in snapshot.items()
                    if key != "initial_rgb_canonical180"}
        _assert_scene_pair(env, observation, names, goals, physical, image=False)
        before = _capture_image(observation) if diagnostic_output is not None else None
        observation = owner._get_observations(force_update=True)
        try:
            _assert_scene_pair(env, observation, names, goals, snapshot, image=True)
        except ValueError:
            if diagnostic_output is not None:
                diagnostic_output.mkdir(parents=True, exist_ok=True)
                artifact = diagnostic_output / (
                    f"scene_restore_{task['suite']}_{task['task_id']}_{state}_{time.time_ns()}.npz")
                np.savez_compressed(artifact, expected_rgb=snapshot["initial_rgb_canonical180"],
                                    before_refresh_rgb=before, after_refresh_rgb=_capture_image(observation))
                print(f"registered_scene_render_failure state={state} artifact={artifact}", flush=True)
            raise
        print(f"registered_scene_render_refresh suite={task['suite']} task={task['task_id']} state={state}",
              flush=True)
    return observation, {"path": str(path), "bytes": path.stat().st_size,
                         "restoration": "full_model_body_pose_post_dummy_sim_controller_and_dual_rgb_verified"}


def validate_scene_row(row: dict, task: dict, contract: dict) -> None:
    registered = contract.get("demonstration_comparison_scene") or contract.get("operator_read_write_scene")
    if registered is None:
        if row.get("scene_reference") is not None:
            raise ValueError("unregistered evaluation row contains a transfer scene")
        return
    path = scene_path(Path(registered["root"]), task, int(row["init_state_id"]))
    expected = {"path": str(path), "bytes": path.stat().st_size,
                "restoration": "full_model_body_pose_post_dummy_sim_controller_and_dual_rgb_verified"}
    if row.get("scene_reference") != expected:
        raise ValueError("paired canonical scene row is missing or changed")
    if contract.get('object_position_transport') is not None:
        from ember.pi05_eval.object_position_transport import validate_row

        validate_row(row, task, contract)
