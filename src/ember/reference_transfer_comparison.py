"""One sealed, train-only comparison of saved B actions and two OSC references.

This diagnostic entry is retired after the fixed twelve rows are adjudicated.
It reads the 81dc1e45 control operators from their frozen source file and
does not expose a new data-construction or training path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from ember.pi05_assets import configure_libero_runtime_assets, prepare_libero_config, write_json_atomic
from ember.pi05_eval.scene import (
    _assert_scene_pair, _capture_image, _query_sample, _restore_scene, _scene_snapshot,
)
from ember.pi05_eval.trajectory_capture import _passive_body_registry


SCHEMA = "ember_reference_transfer_same_scene_v1"
SOURCE_ROOT = Path("/data1/user/ymdai/ember_runs/demonstration_transfer_engineering_20260927/sources")
OLD_FROZEN = Path("/data1/user/ymdai/projects/EMBER-demonstration-transfer-paired-formal")
OLD_COMMIT = "81dc1e45dc5c32b82475a6d95356a9cae92d3993"
ASSETS = Path("/data1/user/ymdai/projects/EMBER/data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6")
ROOT = Path("/data1/user/ymdai/ember_runs/reference_transfer_comparison_20260928")
TASKS = (34, 38)
DEMOS = (0, 1)
ARMS = ("R", "S", "X")
HORIZON = 520


def _operators() -> Any:
    """Import only the pinned old file, never the retired package/CLI path."""
    from subprocess import check_output

    actual = check_output(["git", "-C", str(OLD_FROZEN), "rev-parse", "HEAD"], text=True).strip()
    if actual != OLD_COMMIT:
        raise ValueError(f"old pure-operator source changed: {actual}")
    file = OLD_FROZEN / "src/ember/demonstration_transfer.py"
    spec = importlib.util.spec_from_file_location("ember_reference_old_operators", file)
    if spec is None or spec.loader is None:
        raise ValueError("pinned pure-operator file unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(task: int, demo: int, old: Any) -> tuple[dict, dict[str, np.ndarray], np.ndarray, str]:
    stem = f"task_{task}_demo_{demo}"
    meta = json.loads((SOURCE_ROOT / f"{stem}.json").read_text())
    with np.load(SOURCE_ROOT / f"{stem}.npz", allow_pickle=False) as file:
        arrays = {key: file[key] for key in file.files}
    if (meta["task"], meta["demo"]) != (task, demo) or meta["body_names"] != sorted(meta["body_names"]):
        raise ValueError("sealed source identity/body order changed")
    if len(meta["segments"]) != 4 or meta["segments"][0]["start"] != 0:
        raise ValueError("sealed source four-segment boundary changed")
    cursor = 0
    for row in meta["segments"]:
        if row["start"] != cursor or row["stop"] <= cursor or row["reference_body"] not in meta["body_names"]:
            raise ValueError("sealed source segment clock changed")
        cursor = row["stop"]
    if cursor != meta["steps"] or arrays["actions"].shape != (cursor, 7):
        raise ValueError("source action clock changed")
    for name in ("goal_pos", "goal_rot", "body_pos", "body_rot", "predicates"):
        if len(arrays[name]) != cursor:
            raise ValueError(f"source {name} clock changed")
    with h5py.File(meta["hdf5"], "r") as file:
        group = file[f"data/demo_{demo}"]
        states = np.asarray(group["states"], dtype=np.float64)
        actions = np.asarray(group["actions"], dtype=np.float64)
        xml = old._source_xml(str(group.attrs["model_file"]), repo=OLD_FROZEN)
    if states.shape[0] != cursor or not np.array_equal(actions, arrays["actions"]):
        raise ValueError("saved pre-action states or source actions changed")
    return meta, arrays, states, xml


def _new_env(meta: dict, xml: str, *, gpu: int | None) -> Any:
    from libero.libero.envs.env_wrapper import ControlEnv

    env = ControlEnv(bddl_file_name=meta["bddl"], use_camera_obs=gpu is not None,
                     has_offscreen_renderer=gpu is not None,
                     camera_heights=256, camera_widths=256,
                     render_gpu_device_id=-1 if gpu is None else gpu)
    env.reset()
    env.reset_from_xml_string(xml)
    return env


def _initial(env: Any, state: np.ndarray, meta: dict) -> dict:
    observation = env.regenerate_obs_from_state(state)
    owner = env.env
    names = sorted(owner.obj_body_id)
    goals = [[str(value).lower() for value in row] for row in owner.parsed_problem["goal_state"]]
    if (names, goals) != (meta["body_names"], meta["goals"]):
        raise ValueError("source/current body registry or goal operands changed")
    controller = owner.robots[0].controller
    if (controller.eef_name != "gripper0_grip_site" or not controller.use_delta
            or controller.impedance_mode != "fixed"
            or not np.allclose(controller.output_max, [.05, .05, .05, .5, .5, .5])):
        raise ValueError("source OSC interface changed")
    return observation


def _freeze_start(env: Any, observation: dict, meta: dict, *, image: bool) -> dict[str, np.ndarray]:
    snapshot = _scene_snapshot(env, observation, meta["body_names"], meta["goals"], image=image)
    gripper = env.env.robots[0].gripper
    action = np.asarray(gripper.current_action, dtype=np.float64).copy()
    if action.shape != (gripper.dof,) or not np.isfinite(action).all():
        raise ValueError("gripper internal action unavailable")
    snapshot["gripper_current_action"] = action
    return snapshot


def _restore_start(env: Any, state: np.ndarray, meta: dict, snapshot: dict,
                   *, image: bool) -> dict:
    _initial(env, state, meta)
    scene = {key: value for key, value in snapshot.items() if key != "gripper_current_action"}
    observation = _restore_scene(env, scene)
    robot = env.env.robots[0]
    robot.controller.update(force=True)
    robot.controller.goal_pos = np.asarray(scene["controller_goal_pos"], dtype=np.float64).copy()
    robot.controller.goal_ori = np.asarray(scene["controller_goal_ori"], dtype=np.float64).copy()
    robot.gripper.current_action = np.asarray(snapshot["gripper_current_action"], dtype=np.float64).copy()
    _assert_scene_pair(env, observation, meta["body_names"], meta["goals"], scene, image=image)
    if not np.array_equal(robot.gripper.current_action, snapshot["gripper_current_action"]):
        raise ValueError("restored gripper current_action differs")
    return observation


def _saved_reference(env: Any, states: np.ndarray, meta: dict) -> dict[str, np.ndarray]:
    """Read pre-action source kinematics at fixed k without stepping the source."""
    values: dict[str, list[np.ndarray]] = {key: [] for key in (
        "body_pos", "body_rot", "eef_pos", "eef_quat", "gripper_qpos", "predicates")}
    for state in states:
        observation = env.regenerate_obs_from_state(state)
        sample = _query_sample(env, observation, meta["body_names"], meta["goals"])
        for key in values:
            values[key].append(sample[key])
    return {key: np.stack(rows) for key, rows in values.items()}


def _capture(env: Any, observation: dict, meta: dict, samples: dict[str, list[np.ndarray]]) -> None:
    sample = _query_sample(env, observation, meta["body_names"], meta["goals"])
    for key, value in sample.items():
        samples[key].append(value)


def _replay_reference(env: Any, meta: dict, source: dict[str, np.ndarray], old: Any,
                      execute: Callable[..., None], finished: Callable[[], bool],
                      steps: Callable[[], int], transforms: list[dict]) -> None:
    """The pinned 81dc four-segment connector and waypoint traversal."""
    controller = env.env.robots[0].controller
    names = meta["body_names"]
    for segment, declaration in enumerate(meta["segments"]):
        if finished():
            break
        start, end = int(declaration["start"]), int(declaration["stop"])
        ref = str(declaration["reference_body"])
        ref_id = names.index(ref)
        source_pos, source_rot = source["body_pos"][start, ref_id], source["body_rot"][start, ref_id]
        body_id = int(env.env.obj_body_id[ref])
        query_pos = np.asarray(env.env.sim.data.body_xpos[body_id], dtype=np.float64).copy()
        query_rot = np.asarray(env.env.sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3).copy()
        transforms.append({"segment": segment, "reference_body": ref, "query_step": steps(),
                           "source_start": start, "source_stop": end,
                           "source_pos": source_pos.tolist(), "source_rot": source_rot.tolist(),
                           "query_pos": query_pos.tolist(), "query_rot": query_rot.tolist()})
        first_pos, first_rot = old._target(source_pos, source_rot, query_pos, query_rot,
                                          source["goal_pos"][start], source["goal_rot"][start])
        controller.update(force=True)
        initial_pos, initial_rot = controller.ee_pos.copy(), controller.ee_ori_mat.copy()
        grip = float(source["actions"][0, 6] if segment == 0 else source["actions"][start - 1, 6])
        for goal_pos, goal_rot in old._interpolate(initial_pos, initial_rot, first_pos, first_rot):
            execute(old._inverse_osc(controller, goal_pos, goal_rot, grip),
                    target_pos=goal_pos, target_rot=goal_rot,
                    source_step=-1, segment=segment, is_connector=True)
            if finished():
                break
        if finished():
            break
        for step in range(start, end):
            goal_pos, goal_rot = old._target(source_pos, source_rot, query_pos, query_rot,
                                            source["goal_pos"][step], source["goal_rot"][step])
            action = old._inverse_osc(controller, goal_pos, goal_rot, float(source["actions"][step, 6]))
            execute(action, target_pos=goal_pos, target_rot=goal_rot,
                    source_step=step, segment=segment, is_connector=False)
            if finished():
                break


def _episode(env: Any, observation: dict, task: int, demo: int, arm: str,
             reference: tuple[dict, dict[str, np.ndarray]], old: Any,
             output: Path, snapshot_path: Path) -> dict:
    meta, source = reference
    names = meta["body_names"]
    slot = {"stage_predicate_states": meta["goals"]}
    bodies, operands = _passive_body_registry(env, slot)
    if [row["name"] for row in bodies] != names:
        raise ValueError("passive capture body registry differs from sealed source")
    samples: dict[str, list[np.ndarray]] = {key: [] for key in (
        "body_pos", "body_rot", "eef_pos", "eef_quat", "gripper_qpos", "predicates")}
    _capture(env, observation, meta, samples)
    actions: list[np.ndarray] = []
    targets_pos: list[np.ndarray] = []
    targets_rot: list[np.ndarray] = []
    source_steps: list[int] = []
    segment_ids: list[int] = []
    connector: list[bool] = []
    rgb_steps, rgb = [0], [_capture_image(observation)]
    transforms: list[dict] = []
    started = time.monotonic()
    success, stop = False, "source_actions_exhausted" if arm == "R" else "waypoints_exhausted"
    def execute(action: np.ndarray, *, target_pos: np.ndarray | None = None,
                target_rot: np.ndarray | None = None, source_step: int, segment: int,
                is_connector: bool) -> None:
        nonlocal observation, success, stop
        applied = np.asarray(action, dtype=np.float64)
        if applied.shape != (7,) or not np.isfinite(applied).all():
            raise ValueError("actual OSC command invalid")
        observation, _, _, _ = env.step(applied)
        actions.append(applied.copy())
        source_steps.append(source_step)
        segment_ids.append(segment)
        connector.append(is_connector)
        if arm != "R":
            assert target_pos is not None and target_rot is not None
            targets_pos.append(target_pos.copy())
            targets_rot.append(target_rot.copy())
        _capture(env, observation, meta, samples)
        step = len(actions)
        if step % 5 == 0:
            rgb_steps.append(step)
            rgb.append(_capture_image(observation))
        success = bool(env.check_success())
        if success:
            stop = "official_success"
        elif step >= HORIZON:
            stop = "horizon"

    error: str | None = None
    try:
        if arm == "R":
            for step, action in enumerate(source["actions"]):
                execute(action, source_step=step, segment=-1, is_connector=False)
                if success or len(actions) >= HORIZON:
                    break
        else:
            _replay_reference(env, meta, source, old, execute,
                              lambda: success or len(actions) >= HORIZON,
                              lambda: len(actions), transforms)
    except Exception:
        stop = "engineering_error"
        error = traceback.format_exc()
    if rgb_steps[-1] != len(actions):
        rgb_steps.append(len(actions))
        rgb.append(_capture_image(observation))
    steps = len(actions)
    arrays = {"actions": np.asarray(actions, dtype=np.float32).reshape(steps, 7),
              "source_step": np.asarray(source_steps, dtype=np.int16),
              "segment_id": np.asarray(segment_ids, dtype=np.int8),
              "connector": np.asarray(connector, dtype=np.bool_),
              "rgb_steps": np.asarray(rgb_steps, dtype=np.int16),
              "rgb_canonical180": np.stack(rgb),
              **{key: np.stack(value) for key, value in samples.items()}}
    if arm != "R":
        arrays["target_pos"] = np.asarray(targets_pos, dtype=np.float64).reshape(steps, 3)
        arrays["target_rot"] = np.asarray(targets_rot, dtype=np.float64).reshape(steps, 3, 3)
    if any(len(arrays[key]) != steps + 1 for key in samples):
        raise ValueError("actual T/T+1 trace capture incomplete")
    trace = output / "episodes" / f"task_{task}_B{demo}_{arm}.npz"
    with trace.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return {"schema_version": SCHEMA, "task": task, "B_demo": demo, "arm": arm,
            "reference_demo": demo if arm != "X" else 1 - demo,
            "success": success, "steps": steps, "stop_reason": stop,
            "error": error, "wall_seconds": time.monotonic() - started,
            "trace": str(trace), "trace_bytes": trace.stat().st_size,
            "initial_scene": str(snapshot_path), "body_registry": bodies,
            "goal_operands": operands, "goal_predicates": meta["goals"],
            "segment_transforms": transforms,
            "source_clock": "states[k] pre-action[k]; obs[k] post-action[k]",
            "trace_clock": "offset0 pre-action and T+1 after each executed action"}


def _check(repo: Path) -> dict:
    old = _operators()
    prepare_libero_config(repo / ".codex/tmp/reference_transfer_libero_config")
    configure_libero_runtime_assets(ASSETS)
    report = []
    for task in TASKS:
        for demo in DEMOS:
            meta, source, states, xml = _source(task, demo, old)
            env = _new_env(meta, xml, gpu=None)
            try:
                observation = _initial(env, states[0], meta)
                bodies, _ = _passive_body_registry(env, {"stage_predicate_states": meta["goals"]})
                if [row["name"] for row in bodies] != meta["body_names"]:
                    raise ValueError("passive body registry differs from source")
                snapshot = _freeze_start(env, observation, meta, image=False)
                for _ in ARMS:
                    env.reset_from_xml_string(xml)
                    _restore_start(env, states[0], meta, snapshot, image=False)
                reference = _saved_reference(env, states[:2], meta)
                if reference["body_pos"].shape != (2, len(meta["body_names"]), 3):
                    raise ValueError("saved B kinematic read changed")
                report.append({"task": task, "B_demo": demo, "states": len(states),
                               "segments": meta["segments"], "actions": len(source["actions"]),
                               "scene_restore_checks": 3, "saved_kinematic_reads": 2,
                               "source_steps": 0})
            finally:
                env.close()
    return {"schema_version": SCHEMA, "check": report, "environment_steps": 0}


def _run(output: Path, gpu: int, repo: Path) -> None:
    if output != ROOT or output.exists():
        raise ValueError("fixed output root must be absent before this one-shot run")
    old = _operators()
    output.mkdir(parents=True, exist_ok=False)
    prepare_libero_config(output / "runtime_config")
    configure_libero_runtime_assets(ASSETS)
    for child in ("episodes", "scenes", "saved_B"):
        (output / child).mkdir()
    rows_path = output / "rows.jsonl"
    results = []
    try:
        for task in TASKS:
            for demo in DEMOS:
                meta, source, states, xml = _source(task, demo, old)
                other_meta, other_source, _, _ = _source(task, 1 - demo, old)
                if (other_meta["body_names"], other_meta["goals"]) != (meta["body_names"], meta["goals"]):
                    raise ValueError("same-task A/B roles differ")
                source_env = _new_env(meta, xml, gpu=None)
                try:
                    _initial(source_env, states[0], meta)
                    saved = _saved_reference(source_env, states, meta)
                finally:
                    source_env.close()
                saved_path = output / "saved_B" / f"task_{task}_B{demo}.npz"
                with saved_path.open("xb") as handle:
                    np.savez_compressed(handle, **saved)
                env = _new_env(meta, xml, gpu=gpu)
                try:
                    observation = _initial(env, states[0], meta)
                    snapshot = _freeze_start(env, observation, meta, image=True)
                    scene_path = output / "scenes" / f"task_{task}_B{demo}.npz"
                    with scene_path.open("xb") as handle:
                        np.savez_compressed(handle, **snapshot)
                    for arm in ARMS:
                        env.reset_from_xml_string(xml)
                        current = _restore_start(env, states[0], meta, snapshot, image=True)
                        reference = (meta, source) if arm != "X" else (other_meta, other_source)
                        row = _episode(env, current, task, demo, arm, reference, old, output, scene_path)
                        row["saved_B_reference"] = str(saved_path)
                        row["common_scene_asserted_before_action"] = True
                        results.append(row)
                        with rows_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                        if row["error"] is not None:
                            raise RuntimeError(f"engineering error in task{task}/B{demo}/{arm}")
                finally:
                    env.close()
    except Exception:
        write_json_atomic(output / "failure.json", {"schema_version": SCHEMA,
                          "completed_rows": len(results), "error": traceback.format_exc()})
        raise
    write_json_atomic(output / "completion.json", {"schema_version": SCHEMA,
                      "rows": len(results), "steps": sum(row["steps"] for row in results),
                      "episodes": [str(row["trace"]) for row in results],
                      "repo": str(repo), "old_pure_operator_commit": OLD_COMMIT})


def _distribution(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {"median": float(np.median(flat)), "p95": float(np.percentile(flat, 95)),
            "max": float(np.max(flat))}


def _rotation_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    relative = np.einsum("...ij,...kj->...ik", a, b)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.arccos(cosine)


def _quaternion_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.abs(np.sum(a * b, axis=-1))
    norm = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return 2 * np.arccos(np.clip(dot / norm, 0, 1))


def _summarize_row(row: dict, trace: dict[str, np.ndarray]) -> dict:
    actions, eef, predicates = trace["actions"], trace["eef_pos"], trace["predicates"]
    goal_times = []
    for column in range(predicates.shape[1]):
        hit = np.flatnonzero(predicates[:, column])
        goal_times.append({"first": int(hit[0]) if len(hit) else None,
                           "last": int(hit[-1]) if len(hit) else None,
                           "final": bool(predicates[-1, column])})
    result = {"arm": row["arm"], "steps": row["steps"], "success": row["success"],
              "stop_reason": row["stop_reason"],
              "eef_path_m": float(np.linalg.norm(np.diff(eef, axis=0), axis=-1).sum()),
              "action_total_variation_l2": float(np.linalg.norm(np.diff(actions, axis=0), axis=-1).sum()),
              "gripper_command_sign_transitions": int(np.count_nonzero(
                  np.diff(np.sign(actions[:, 6])))) if len(actions) > 1 else 0,
              "goal_times": goal_times, "trace": row["trace"]}
    if row["arm"] != "R":
        segments = []
        for transform in row["segment_transforms"]:
            segment = int(transform["segment"])
            hits = np.flatnonzero(trace["segment_id"] == segment)
            if not len(hits):
                continue
            first, last = int(hits[0] + 1), int(hits[-1] + 1)
            segments.append({"segment": segment, "reference_body": transform["reference_body"],
                             "source_start": transform["source_start"],
                             "source_stop": transform["source_stop"],
                             "first_step": first, "last_step": last,
                             "first_source_step": int(trace["source_step"][hits[0]]),
                             "last_source_step": int(trace["source_step"][hits[-1]]),
                             "first_target_pos": trace["target_pos"][hits[0]].tolist(),
                             "last_target_pos": trace["target_pos"][hits[-1]].tolist(),
                             "first_eef_pos": eef[first].tolist(), "last_eef_pos": eef[last].tolist(),
                             "first_predicates": predicates[first].tolist(),
                             "last_predicates": predicates[last].tolist(),
                             "connector_actions": int(np.count_nonzero(trace["connector"][hits]))})
        result["segments"] = segments
    return result


def _pair(a: dict, b: dict, left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict:
    n = min(a["steps"], b["steps"])
    return {"common_actions": n, "left_tail": a["steps"] - n, "right_tail": b["steps"] - n,
            "action_l2": _distribution(np.linalg.norm(left["actions"][:n] - right["actions"][:n], axis=-1)),
            "eef_position_m": _distribution(np.linalg.norm(
                left["eef_pos"][:n + 1] - right["eef_pos"][:n + 1], axis=-1)),
            "eef_rotation_rad": _distribution(_quaternion_distance(
                left["eef_quat"][:n + 1], right["eef_quat"][:n + 1])),
            "body_origin_position_m": _distribution(np.linalg.norm(
                left["body_pos"][:n + 1] - right["body_pos"][:n + 1], axis=-1))}


def _analyze(output: Path) -> None:
    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    if len(rows) != 12 or any(row["error"] is not None for row in rows):
        raise ValueError("twelve complete fixed episodes required for mechanical comparison")
    groups = []
    for task in TASKS:
        for demo in DEMOS:
            selected = {row["arm"]: row for row in rows
                        if (row["task"], row["B_demo"]) == (task, demo)}
            if set(selected) != set(ARMS):
                raise ValueError("fixed R/S/X group incomplete")
            arrays = {}
            for arm, row in selected.items():
                with np.load(row["trace"], allow_pickle=False) as file:
                    arrays[arm] = {name: file[name] for name in file.files}
            with np.load(selected["R"]["saved_B_reference"], allow_pickle=False) as file:
                saved = {name: file[name] for name in file.files}
            r = arrays["R"]
            n = min(len(r["body_pos"]), len(saved["body_pos"]))
            original = {"matched_pre_action_states": n,
                        "saved_terminal_post_state_unavailable": len(r["body_pos"]) > len(saved["body_pos"]),
                        "body_origin_position_m": _distribution(np.linalg.norm(
                            r["body_pos"][:n] - saved["body_pos"][:n], axis=-1)),
                        "body_origin_rotation_rad": _distribution(_rotation_distance(
                            r["body_rot"][:n], saved["body_rot"][:n])),
                        "eef_position_m": _distribution(np.linalg.norm(
                            r["eef_pos"][:n] - saved["eef_pos"][:n], axis=-1)),
                        "eef_rotation_rad": _distribution(_quaternion_distance(
                            r["eef_quat"][:n], saved["eef_quat"][:n])),
                        "gripper_qpos_l2": _distribution(np.linalg.norm(
                            r["gripper_qpos"][:n] - saved["gripper_qpos"][:n], axis=-1)),
                        "predicate_mismatch_states": int(np.count_nonzero(np.any(
                            r["predicates"][:n] != saved["predicates"][:n], axis=-1))),
                        "actual_success": bool(selected["R"]["success"])}
            groups.append({"task": task, "B_demo": demo, "source_clock": "states[k] before actions[k]",
                           "unobserved_source_internal_state": ["historical_controller_goal",
                                                                 "historical_gripper_current_action"],
                           "R_vs_saved_B": original,
                           "arms": {arm: _summarize_row(selected[arm], arrays[arm]) for arm in ARMS},
                           "pairs_common_raw_control_time": {
                               f"{a}-{b}": _pair(selected[a], selected[b], arrays[a], arrays[b])
                               for a, b in (("R", "S"), ("S", "X"), ("R", "X"))}})
    write_json_atomic(output / "comparison.json", {"schema_version": SCHEMA,
                      "comparison": "descriptive same-current-scene raw-clock only; no DTW or causal percentage",
                      "groups": groups})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("check", "run", "analyze"))
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "check":
        if args.gpu is not None:
            parser.error("CPU no-action check cannot select a GPU")
        print(json.dumps(_check(args.repo), sort_keys=True))
    elif args.phase == "run":
        if args.gpu is None:
            parser.error("one-shot render run requires a live admitted GPU")
        _run(args.output, args.gpu, args.repo)
    else:
        if args.gpu is not None:
            parser.error("mechanical analysis runs on CPU")
        _analyze(args.output)


if __name__ == "__main__":
    main()
