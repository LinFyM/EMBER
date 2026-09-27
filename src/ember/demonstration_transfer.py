"""Bounded, train-only OSC demonstration transfer for the 2026-09-27 study.

This module has one temporary data-construction entry. Source state/action/XML
remain privileged construction inputs; they are never a Writer condition.
Retire this entry when the fixed construction is closed or integrated into a
single canonical data owner after a separate scientific decision.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ember.pi05_assets import configure_libero_runtime_assets, prepare_libero_config, write_json_atomic
from ember.pi05_eval.episode import stage_predicate_snapshot


SCHEMA = "ember_demonstration_transfer_engineering_v1"
ROOT = Path("/data1/user/ymdai/ember_runs/demonstration_transfer_engineering_20260927")
CANONICAL_ASSET_REPO = Path("/data1/user/ymdai/projects/EMBER")
DATA = CANONICAL_ASSET_REPO / "data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a"
AUTHORITY = Path("configs/pi05_target_data_v1/manifest.json")
EVAL_CONFIG = Path("configs/pi05_target_evaluation_v1.json")
TASKS = {
    34: (("porcelain_mug_1", "plate_1", "plate_1"),
         ("white_yellow_mug_1", "plate_2", "plate_2")),
    38: (("moka_pot_2", "flat_stove_1", "flat_stove_1_cook_region"),
         ("moka_pot_1", "flat_stove_1", "flat_stove_1_cook_region")),
}
STATES = (40, 41, 42, 43)
DEMOS = (0, 1)
ASSETS = CANONICAL_ASSET_REPO / "data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
OLD_ROBOSUITE = "/home/yifengz/workspace/robosuite-master/robosuite/models/assets"
OLD_LIBERO = "/home/yifengz/workspace/libero-dev/chiliocosm/assets"


def _authorities(repo: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any], dict[str, str]]:
    manifest = json.loads((repo / AUTHORITY).read_text())
    rows = {int(row["global_task_id"]): row for row in manifest["tasks"] if row["global_task_id"] in TASKS}
    if set(rows) != set(TASKS) or any(row["split_role"] != "train" or row["suite"] != "libero_10"
                                      for row in rows.values()):
        raise ValueError("fixed sources are not coverage train34/38")
    recipe = json.loads((repo / EVAL_CONFIG).read_text())
    environment = recipe["environment"]
    if (environment["dummy_settling_steps"] != 10 or environment["render_resolution"] != 256
            or environment["horizons"]["libero_10"] != 520
            or environment["dummy_action"] != [0, 0, 0, 0, 0, 0, -1]
            or recipe["rng"]["inference_seed"] != 7):
        raise ValueError("official environment recipe changed")
    paths = prepare_libero_config(repo / ".codex/tmp/demonstration_transfer_libero_config")
    configure_libero_runtime_assets(ASSETS)
    return rows, recipe, paths


def _source_xml(xml: str, *, repo: Path) -> str:
    import robosuite

    current_robosuite = str(Path(robosuite.__file__).resolve().parent / "models/assets")
    value = xml.replace(OLD_ROBOSUITE, current_robosuite).replace(OLD_LIBERO, str(ASSETS))
    if value == xml or OLD_ROBOSUITE in value or OLD_LIBERO in value:
        raise ValueError("source XML asset roots changed")
    return value


def _pose(sim: Any, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray(sim.data.body_xpos[body_id], dtype=np.float64).copy(),
            np.asarray(sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3).copy())


def _roles(owner: Any, task: int) -> tuple[list[str], list[list[str]]]:
    names = sorted(owner.obj_body_id)
    goals = [[str(x).lower() for x in row] for row in owner.parsed_problem["goal_state"]]
    expected = [["on", obj, goal] for obj, _, goal in TASKS[task]]
    if task == 34 and (set(names) != {"porcelain_mug_1", "red_coffee_mug_1",
                                        "white_yellow_mug_1", "plate_1", "plate_2"}
                       or goals != expected):
        raise ValueError("task34 installed body/BDDL roles changed")
    if task == 38 and (set(names) != {"moka_pot_1", "moka_pot_2", "flat_stove_1"}
                       or goals != [expected[1], expected[0], ["turnon", "flat_stove_1"]]):
        raise ValueError("task38 installed body/BDDL roles changed")
    return names, goals


def _extract_one(repo: Path, row: dict[str, Any], paths: dict[str, str], demo: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Restore saved kinematics and OSC goals; never call source env.step."""
    from libero.libero.envs.env_wrapper import ControlEnv

    task = int(row["global_task_id"])
    source = DATA / row["hdf5"]["relative_path"]
    bddl = Path(paths["bddl_files"]) / row["problem_folder"] / row["bddl"]["filename"]
    if source.stat().st_size != row["hdf5"]["bytes"] or not bddl.is_file():
        raise ValueError(f"source file or BDDL missing: {task}")
    with h5py.File(source) as handle:
        group = handle[f"data/demo_{demo}"]
        actions = np.asarray(group["actions"], dtype=np.float64)
        states = np.asarray(group["states"], dtype=np.float64)
        xml = _source_xml(str(group.attrs["model_file"]), repo=repo)
    if actions.ndim != 2 or actions.shape[1] != 7 or states.shape[0] != len(actions):
        raise ValueError("source action/state clock changed")
    env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_offscreen_renderer=False)
    try:
        env.reset()
        env.reset_from_xml_string(xml)
        owner = env.env
        names, goals = _roles(owner, task)
        controller = owner.robots[0].controller
        if (controller.eef_name != "gripper0_grip_site" or not controller.use_delta
                or controller.impedance_mode != "fixed"
                or not np.allclose(controller.output_max, [.05, .05, .05, .5, .5, .5])):
            raise ValueError("installed OSC_POSE interface changed")
        positions = np.empty((len(actions), len(names), 3), dtype=np.float64)
        rotations = np.empty((len(actions), len(names), 3, 3), dtype=np.float64)
        eef_goals_pos = np.empty((len(actions), 3), dtype=np.float64)
        eef_goals_rot = np.empty((len(actions), 3, 3), dtype=np.float64)
        predicates = np.empty((len(actions), len(goals)), dtype=np.bool_)
        for step, (state, action) in enumerate(zip(states, actions, strict=True)):
            env.set_state(state)
            env.sim.forward()
            controller.update(force=True)
            for j, name in enumerate(names):
                positions[step, j], rotations[step, j] = _pose(owner.sim, owner.obj_body_id[name])
            predicates[step] = [bool(owner._eval_predicate(goal)) for goal in goals]
            controller.set_goal(action[:6])
            eef_goals_pos[step] = controller.goal_pos
            eef_goals_rot[step] = controller.goal_ori
        first, second = TASKS[task]
        first_id, second_id = names.index(first[0]), names.index(second[0])
        l1 = next((i for i in range(1, len(actions))
                   if positions[i, first_id, 2] - positions[0, first_id, 2] >= .03), None)
        l2 = next((i for i in range(1, len(actions))
                   if positions[i, second_id, 2] - positions[0, second_id, 2] >= .03), None)
        first_goal = goals.index(["on", first[0], first[2]])
        r1 = (next((i for i in range(l1, len(actions))
                    if predicates[i, first_goal] and actions[i, 6] < 0), None)
              if l1 is not None else None)
        second_goal = goals.index(["on", second[0], second[2]])
        if (predicates[0, first_goal] or predicates[0, second_goal]
                or l1 is None or l2 is None or r1 is None
                or not (0 < l1 <= r1 < l2 < len(actions))):
            raise ValueError(f"fixed source boundaries unavailable: task{task}/demo{demo}, {l1=}, {r1=}, {l2=}")
        segments = ((0, l1, first[0]), (l1, r1 + 1, first[1]),
                    (r1 + 1, l2, second[0]), (l2, len(actions), second[1]))
        if any(start >= stop for start, stop, _ in segments):
            raise ValueError("empty fixed source segment")
        metadata = {
            "schema_version": SCHEMA, "task": task, "demo": demo, "hdf5": str(source),
            "bddl": str(bddl), "steps": len(actions), "body_names": names,
            "goals": goals, "source_boundaries": {"l1": l1, "r1": r1, "l2": l2},
            "segments": [{"start": start, "stop": stop, "reference_body": ref}
                         for start, stop, ref in segments],
            "source_initial_object_positions": {name: positions[0, names.index(name)].tolist()
                                                for name in (first[0], second[0])},
            "source_xml": "demo model_file with only installed asset-root remap",
            "source_state_clock": "states[k] pre-action[k]; obs[k] post-action[k]",
            "source_steps_executed": 0,
        }
        arrays = {"actions": actions, "goal_pos": eef_goals_pos, "goal_rot": eef_goals_rot,
                  "body_pos": positions, "body_rot": rotations, "predicates": predicates}
        return metadata, arrays
    finally:
        env.close()


def _target(source_pos: np.ndarray, source_rot: np.ndarray, new_pos: np.ndarray,
            new_rot: np.ndarray, goal_pos: np.ndarray, goal_rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    relative_pos = source_rot.T @ (goal_pos - source_pos)
    relative_rot = source_rot.T @ goal_rot
    return new_pos + new_rot @ relative_pos, new_rot @ relative_rot


def _inverse_osc(controller: Any, goal_pos: np.ndarray, goal_rot: np.ndarray,
                 gripper: float) -> np.ndarray:
    controller.update(force=True)
    delta = np.concatenate((goal_pos - controller.ee_pos,
                            Rotation.from_matrix(goal_rot @ controller.ee_ori_mat.T).as_rotvec()))
    output = np.asarray(controller.output_max, dtype=np.float64)
    if delta.shape != (6,) or not np.isfinite(delta).all() or not np.all(output > 0):
        raise ValueError("invalid inverse OSC target")
    return np.r_[np.clip(delta / output, -1, 1), float(gripper)]


def _interpolate(start_pos: np.ndarray, start_rot: np.ndarray, end_pos: np.ndarray,
                 end_rot: np.ndarray, steps: int = 10) -> list[tuple[np.ndarray, np.ndarray]]:
    rotations = Rotation.from_matrix(np.stack((start_rot, end_rot)))
    interpolation = Slerp([0.0, 1.0], rotations)
    return [(start_pos + alpha * (end_pos - start_pos), interpolation([alpha]).as_matrix()[0])
            for alpha in np.arange(1, steps + 1, dtype=np.float64) / steps]


def _source_phase(repo: Path, output: Path) -> None:
    rows, recipe, paths = _authorities(repo)
    sources = output / "sources"
    sources.mkdir(parents=True, exist_ok=False)
    summary = []
    for task in sorted(TASKS):
        for demo in DEMOS:
            metadata, arrays = _extract_one(repo, rows[task], paths, demo)
            stem = f"task_{task}_demo_{demo}"
            np.savez_compressed(sources / f"{stem}.npz", **arrays)
            write_json_atomic(sources / f"{stem}.json", metadata)
            summary.append({"task": task, "demo": demo, "boundaries": metadata["source_boundaries"],
                            "steps": metadata["steps"]})
    write_json_atomic(output / "source_completion.json", {
        "schema_version": SCHEMA, "sources": summary, "source_steps_executed": 0,
        "canonical_recipe": str(repo / EVAL_CONFIG), "seed": recipe["rng"]["inference_seed"]})


def _query_sample(env: Any, observation: dict[str, Any], names: list[str], goals: list[list[str]]) -> dict[str, np.ndarray]:
    owner = env.env
    return {
        "body_pos": np.stack([_pose(owner.sim, owner.obj_body_id[name])[0] for name in names]),
        "body_rot": np.stack([_pose(owner.sim, owner.obj_body_id[name])[1] for name in names]),
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


def _run_one(env: Any, init_states: Any, row: dict[str, Any], demo: int, state: int,
             metadata: dict[str, Any], source: dict[str, np.ndarray], recipe: dict[str, Any],
             output: Path) -> dict[str, Any]:
    started = time.monotonic()
    env.seed(int(recipe["rng"]["inference_seed"]))
    env.reset()
    observation = env.set_init_state(init_states[state])
    for _ in range(10):
        observation, _, _, _ = env.step(np.asarray(recipe["environment"]["dummy_action"], dtype=np.float64))
    owner = env.env
    names, goals = _roles(owner, int(row["global_task_id"]))
    if names != metadata["body_names"] or goals != metadata["goals"]:
        raise ValueError("query body/BDDL roles differ from source")
    initial = _query_sample(env, observation, names, goals)
    if all(initial["predicates"]):
        raise ValueError("query initial target already complete")
    differences = {name: float(np.linalg.norm(initial["body_pos"][names.index(name)]
                                              - metadata["source_initial_object_positions"][name]))
                   for name, _, _ in TASKS[int(row["global_task_id"])]}
    if not any(value > 1e-4 for value in differences.values()):
        raise ValueError("query physical initial objects do not differ from source")
    samples = {key: [value] for key, value in initial.items()}
    actions: list[np.ndarray] = []
    targets_pos: list[np.ndarray] = []
    targets_rot: list[np.ndarray] = []
    segment_ids: list[int] = []
    kinds: list[int] = []
    rgb_steps, rgb = [0], [_capture_image(observation)]
    segment_transforms = []
    success = False
    stop_reason = "waypoints_exhausted"
    controller = owner.robots[0].controller

    def execute(target_pos: np.ndarray, target_rot: np.ndarray, gripper: float,
                segment: int, kind: int) -> None:
        nonlocal observation, success, stop_reason
        action = _inverse_osc(controller, target_pos, target_rot, gripper)
        observation, _, _, _ = env.step(action)
        actions.append(action)
        targets_pos.append(target_pos.copy())
        targets_rot.append(target_rot.copy())
        segment_ids.append(segment)
        kinds.append(kind)  # 0 connector, 1 source waypoint
        sample = _query_sample(env, observation, names, goals)
        for key, value in sample.items():
            samples[key].append(value)
        step = len(actions)
        if step % 5 == 0:
            rgb_steps.append(step)
            rgb.append(_capture_image(observation))
        success = bool(env.check_success())
        if success:
            stop_reason = "official_success"
        elif step >= 520:
            stop_reason = "horizon"

    for segment, declaration in enumerate(metadata["segments"]):
        if success or len(actions) >= 520:
            break
        start, stop = int(declaration["start"]), int(declaration["stop"])
        ref = str(declaration["reference_body"])
        ref_id = names.index(ref)
        source_pos = source["body_pos"][start, ref_id]
        source_rot = source["body_rot"][start, ref_id]
        new_pos, new_rot = _pose(owner.sim, owner.obj_body_id[ref])
        segment_transforms.append({"segment": segment, "query_step": len(actions),
                                   "reference_body": ref, "source_pos": source_pos.tolist(),
                                   "source_rot": source_rot.tolist(), "query_pos": new_pos.tolist(),
                                   "query_rot": new_rot.tolist()})
        first_pos, first_rot = _target(source_pos, source_rot, new_pos, new_rot,
                                      source["goal_pos"][start], source["goal_rot"][start])
        controller.update(force=True)
        initial_pos, initial_rot = controller.ee_pos.copy(), controller.ee_ori_mat.copy()
        gripper = float(source["actions"][0, 6] if segment == 0 else source["actions"][start - 1, 6])
        for target_pos, target_rot in _interpolate(initial_pos, initial_rot, first_pos, first_rot):
            execute(target_pos, target_rot, gripper, segment, 0)
            if success or len(actions) >= 520:
                break
        if success or len(actions) >= 520:
            break
        for source_step in range(start, stop):
            target_pos, target_rot = _target(source_pos, source_rot, new_pos, new_rot,
                                            source["goal_pos"][source_step],
                                            source["goal_rot"][source_step])
            execute(target_pos, target_rot, float(source["actions"][source_step, 6]), segment, 1)
            if success or len(actions) >= 520:
                break
    if rgb_steps[-1] != len(actions):
        rgb_steps.append(len(actions))
        rgb.append(_capture_image(observation))
    steps = len(actions)
    arrays = {"actions": np.stack(actions).astype(np.float32),
              "target_pos": np.stack(targets_pos).astype(np.float32),
              "target_rot": np.stack(targets_rot).astype(np.float32),
              "segment_id": np.asarray(segment_ids, dtype=np.int8),
              "step_kind": np.asarray(kinds, dtype=np.int8),
              "rgb_steps": np.asarray(rgb_steps, dtype=np.int16),
              "rgb_canonical180": np.stack(rgb),
              **{key: np.stack(value) for key, value in samples.items()}}
    expected = {"actions": (steps, 7), "body_pos": (steps + 1, len(names), 3),
                "eef_pos": (steps + 1, 3), "predicates": (steps + 1, len(goals))}
    if any(arrays[key].shape != shape for key, shape in expected.items()):
        raise ValueError("T action / T+1 observation capture incomplete")
    path = output / "episodes" / f"task_{row['global_task_id']}_state_{state}_demo_{demo}.npz"
    np.savez_compressed(path, **arrays)
    return {"schema_version": SCHEMA, "task": int(row["global_task_id"]), "state": state,
            "demo": demo, "success": success, "steps": steps, "stop_reason": stop_reason,
            "wall_seconds": time.monotonic() - started, "trace": str(path),
            "trace_bytes": path.stat().st_size, "source": metadata["hdf5"],
            "source_boundaries": metadata["source_boundaries"],
            "query_initial_object_distance_from_source_m": differences,
            "segment_transforms": segment_transforms,
            "body_names": names, "goals": goals,
            "image_orientation": "raw LIBERO OpenGL rotated 180 degrees by both axis reversals",
            "image_time": "pre-action state at rgb_steps; offset0; final T included",
            "source_privileged_fields_excluded_from_future_writer_input": True}


def _episode_phase(repo: Path, output: Path, gpu_index: int) -> None:
    rows, recipe, paths = _authorities(repo)
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    if not (output / "source_completion.json").is_file():
        raise ValueError("CPU source phase missing")
    (output / "episodes").mkdir(exist_ok=False)
    results = output / "rows.jsonl"
    for task in sorted(TASKS):
        row = rows[task]
        suite = benchmark.get_benchmark_dict()[row["suite"]]()
        if suite.get_task(row["task_id"]).language != row["language"]:
            raise ValueError("benchmark language changed")
        init_states = suite.get_task_init_states(row["task_id"])
        bddl = Path(paths["bddl_files"]) / row["problem_folder"] / row["bddl"]["filename"]
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                                 camera_widths=256, render_gpu_device_id=gpu_index)
        try:
            for state in STATES:
                for demo in DEMOS:
                    stem = f"task_{task}_demo_{demo}"
                    metadata = json.loads((output / "sources" / f"{stem}.json").read_text())
                    with np.load(output / "sources" / f"{stem}.npz") as source_file:
                        source = {name: source_file[name] for name in source_file.files}
                    result = _run_one(env, init_states, row, demo, state, metadata, source, recipe, output)
                    with results.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(result, sort_keys=True) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
        finally:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("source", "episodes"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--gpu-index", type=int)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    if output != ROOT or (args.phase == "episodes" and args.gpu_index is None):
        raise ValueError("fixed study root or rendering GPU missing")
    if args.phase == "source":
        _source_phase(repo, output)
    else:
        _episode_phase(repo, output, args.gpu_index)


if __name__ == "__main__":
    main()
