"""Shared full-scene initialization for transfer collection and official evaluation."""

from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np

from ember.demonstration_transfer_source import body_id, pose


def _query_sample(env: Any, observation: dict[str, Any], names: list[str], goals: list[list[str]]) -> dict[str, np.ndarray]:
    owner = env.env
    return {
        "body_pos": np.stack([pose(owner.sim, body_id(owner, name))[0] for name in names]),
        "body_rot": np.stack([pose(owner.sim, body_id(owner, name))[1] for name in names]),
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


def _initialize_query(env: Any, init_states: Any, state: int, recipe: dict[str, Any]) -> dict[str, Any]:
    """Use the unchanged official fixed-state and ten-dummy initialization."""
    env.seed(int(recipe["rng"]["inference_seed"]))
    env.reset()
    observation = env.set_init_state(init_states[state])
    for _ in range(10):
        observation, _, _, _ = env.step(np.asarray(recipe["environment"]["dummy_action"], dtype=np.float64))
    return observation


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


def freeze_registered_scenes(asset_root: Path, output: Path, *, physical_gpu_id: int) -> dict:
    """Freeze only the eight prespecified train scenes before any policy behavior."""
    from dataclasses import asdict
    from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
    from ember.pi05_eval_contract import load_evaluation_authorities, inspect_installed_target_tasks
    from ember.pi05_assets import configure_libero_runtime_assets
    from ember.pi05_source_checkpoint import write_json_atomic

    authorities = load_evaluation_authorities(
        asset_root / "configs/libero_24_8_8_coverage_v1/evaluation.json", asset_root)
    installed, paths = inspect_installed_target_tasks(
        authorities, role="development_train", state_count=4,
        libero_config_dir=output / "libero_config")
    configure_libero_runtime_assets(Path(paths["assets"]))
    contract = dict(authorities.config)
    contract["libero_paths"] = paths
    contract["parallel"] = {**contract["parallel"], "envs_per_replica": 1}
    selected = {("libero_spatial", 2), ("libero_10", 8)}
    tasks = [asdict(task) for task in installed if (task.suite, task.task_id) in selected]
    if len(tasks) != 2 or output.exists():
        raise ValueError("eight-scene freeze scope changed or output already exists")
    output.mkdir(parents=True)
    pool = PersistentTaskEnvironmentPool(contract, physical_gpu_id=physical_gpu_id)
    records = []
    try:
        for task in tasks:
            envs, initial = pool.switch(task)
            env = envs[0]
            for state in range(4):
                obs = _initialize_query(env, initial, state, contract)
                owner = env.env
                names = sorted(owner.obj_body_id)
                goals = [list(row) for row in owner.parsed_problem["goal_state"]]
                # Regeneration is already part of the accepted collector semantics.
                snapshot = _scene_snapshot(env, obs, names, goals, image=False)
                obs = _restore_scene(env, snapshot)
                _assert_scene_pair(env, obs, names, goals, snapshot, image=False)
                full = _scene_snapshot(env, obs, names, goals, image=True)
                path = scene_path(output, task, state)
                np.savez_compressed(path, **full)
                _assert_scene_pair(env, obs, names, goals, full, image=True)
                records.append({"suite": task["suite"], "task_id": task["task_id"],
                                "state": state, "path": str(path), "bytes": path.stat().st_size})
    finally:
        pool.close()
    if len(records) != 8:
        raise ValueError("canonical scene freeze incomplete")
    write_json_atomic(output / "manifest.json", {"schema_version": "ember_demonstration_comparison_scenes_v1",
                                                  "seed": 7, "dummy_steps": 10, "scenes": records})
    return {"scenes": len(records), "manifest": str(output / "manifest.json")}


def restore_registered_scene(env: Any, observation: dict, task: dict, state: int,
                             root: Path) -> tuple[dict, dict]:
    path = scene_path(root, task, state)
    with np.load(path, allow_pickle=False) as sealed:
        snapshot = {name: sealed[name] for name in sealed.files}
    owner = env.env
    names = sorted(owner.obj_body_id)
    goals = [list(row) for row in owner.parsed_problem["goal_state"]]
    observation = _restore_scene(env, snapshot)
    _assert_scene_pair(env, observation, names, goals, snapshot, image=True)
    return observation, {"path": str(path), "bytes": path.stat().st_size,
                         "restoration": "full_model_body_pose_post_dummy_sim_controller_and_dual_rgb_verified"}


def validate_scene_row(row: dict, task: dict, contract: dict) -> None:
    registered = contract.get("demonstration_comparison_scene")
    if registered is None:
        if row.get("scene_reference") is not None:
            raise ValueError("unregistered evaluation row contains a transfer scene")
        return
    path = scene_path(Path(registered["root"]), task, int(row["init_state_id"]))
    expected = {"path": str(path), "bytes": path.stat().st_size,
                "restoration": "full_model_body_pose_post_dummy_sim_controller_and_dual_rgb_verified"}
    if row.get("scene_reference") != expected:
        raise ValueError("paired canonical scene row is missing or changed")
