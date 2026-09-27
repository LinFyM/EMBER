"""One bounded train-only OSC transfer data owner for the 2026-09-27 study.

Source state/action/XML are privileged construction inputs, never Writer input.
Retire this entry when the fixed construction is closed or integrated after a
separate scientific decision. Old fixed engineering runs live in frozen Git.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ember.demonstration_transfer_source import (ASSETS, CANONICAL_ASSET_REPO, DATA,
                                                SCHEMA, SourceStructureUnsupported,
                                                body_id, extract_one, pose, roles)
from ember.pi05_assets import configure_libero_runtime_assets, prepare_libero_config, write_json_atomic


ROOT = Path("/data1/user/ymdai/ember_runs/demonstration_transfer_training_support_20260927")
OLD_SOURCES = Path("/data1/user/ymdai/ember_runs/demonstration_transfer_engineering_20260927/sources")
AUTHORITY = Path("configs/libero_24_8_8_coverage_v1/manifest.json")
EVAL_CONFIG = Path("configs/pi05_target_evaluation_v1.json")
SPEC = Path("configs/demonstration_transfer_v1/training_support_spec.json")
REUSED = {(34, 0), (34, 1), (38, 0), (38, 1)}


def _checked_spec(repo: Path) -> dict[str, Any]:
    spec = json.loads((repo / SPEC).read_text())
    ids = tuple(spec["task_ids"])
    group_ids = set().union(*map(set, spec["task_groups"].values()))
    if (len(ids) != 27 or len(set(ids)) != 27 or set(ids) != group_ids
            or spec["source_demo_ids"] != [0, 1, 2, 3]
            or spec["query_init_state_ids"] != [44, 45, 46, 47]
            or spec["source_cases_total"] != 108 or spec["query_cases_max"] != 432):
        raise ValueError("training support spec changed")
    return spec


def _checked_recipe(repo: Path) -> dict[str, Any]:
    recipe = json.loads((repo / EVAL_CONFIG).read_text())
    environment = recipe["environment"]
    if (environment["dummy_settling_steps"] != 10 or environment["render_resolution"] != 256
            or environment["dummy_action"] != [0, 0, 0, 0, 0, 0, -1]
            or recipe["rng"]["inference_seed"] != 7):
        raise ValueError("official environment recipe changed")
    horizons = (("libero_spatial", 220), ("libero_object", 280),
                ("libero_goal", 300), ("libero_10", 520))
    if any(environment["horizons"][suite] != horizon for suite, horizon in horizons):
        raise ValueError("official suite horizons changed")
    return recipe


def _authorities(repo: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    spec = _checked_spec(repo)
    manifest = json.loads((repo / AUTHORITY).read_text())
    ids = tuple(spec["task_ids"])
    rows = {int(row["global_task_id"]): dict(row) for row in manifest["tasks"]
            if row["global_task_id"] in ids}
    if set(rows) != set(ids) or any(row["split_role"] != "train" for row in rows.values()):
        raise ValueError("selected sources are not coverage-v1 train")
    recipe = _checked_recipe(repo)
    paths = prepare_libero_config(repo / ".codex/tmp/demonstration_transfer_libero_config")
    configure_libero_runtime_assets(ASSETS)
    from libero.libero import benchmark

    suites = {name: cls() for name, cls in benchmark.get_benchmark_dict().items()
              if name in {row["suite"] for row in rows.values()}}
    for row in rows.values():
        suite = suites[row["suite"]]
        task = suite.get_task(int(row["task_id"]))
        if (task.name != row["task_name"] or task.language != row["language"]
                or suite.get_task_demonstration(int(row["task_id"])) != row["hdf5"]["relative_path"]):
            raise ValueError(f"official task/source identity changed: {row['global_task_id']}")
        row["bddl_path"] = str(Path(paths["bddl_files"]) / task.problem_folder / task.bddl_file)
    return rows, recipe, spec


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


def _source_record(row: dict[str, Any], demo: int, sources: Path) -> dict[str, Any]:
    task = int(row["global_task_id"])
    stem = f"task_{task}_demo_{demo}"
    if (task, demo) in REUSED:
        metadata = json.loads((OLD_SOURCES / f"{stem}.json").read_text())
        if (metadata["task"] != task or metadata["demo"] != demo
                or metadata["hdf5"] != str(DATA / row["hdf5"]["relative_path"])
                or not (OLD_SOURCES / f"{stem}.npz").is_file()):
            raise ValueError("original source identity changed")
        return {"task": task, "demo": demo, "status": "compatible_reused",
                "metadata": str(OLD_SOURCES / f"{stem}.json"),
                "arrays": str(OLD_SOURCES / f"{stem}.npz"),
                "steps": metadata["steps"], "source_env_steps": 0}
    try:
        metadata, arrays = extract_one(row, Path(row["bddl_path"]), demo)
    except SourceStructureUnsupported as exc:
        return {"task": task, "demo": demo, "status": "structure_unsupported",
                "reason": str(exc), "source_env_steps": 0}
    np.savez_compressed(sources / f"{stem}.npz", **arrays)
    write_json_atomic(sources / f"{stem}.json", metadata)
    return {"task": task, "demo": demo, "status": "compatible",
            "metadata": str(sources / f"{stem}.json"),
            "arrays": str(sources / f"{stem}.npz"),
            "steps": metadata["steps"], "source_env_steps": 0}


def _source_plan(repo: Path, output: Path, seen: dict[tuple[int, int], dict[str, Any]],
                 spec: dict[str, Any], recipe: dict[str, Any]) -> None:
    if len(seen) != 108:
        raise ValueError("incomplete fixed source registry")
    compatible_tasks = [task for task in spec["task_ids"] if all(
        seen[task, demo]["status"] in {"compatible", "compatible_reused"}
        for demo in spec["source_demo_ids"])]
    groups = {name: [task for task in ids if task in compatible_tasks]
              for name, ids in spec["task_groups"].items()}
    gate = len(compatible_tasks) >= 20 and all(len(ids) >= 2 for ids in groups.values())
    with (output / "case_plan.jsonl").open("w", encoding="utf-8") as handle:
        for task in spec["task_ids"]:
            bad = [seen[task, demo] for demo in spec["source_demo_ids"]
                   if seen[task, demo]["status"] == "structure_unsupported"]
            for state in spec["query_init_state_ids"]:
                for demo in spec["source_demo_ids"]:
                    entry = {"task": task, "state": state, "demo": demo,
                             "status": ("ready" if gate and not bad else
                                        "skipped_source_structure" if bad else "skipped_source_gate"),
                             "reasons": [f"demo{x['demo']}: {x['reason']}" for x in bad]}
                    handle.write(json.dumps(entry, sort_keys=True) + "\n")
    write_json_atomic(output / "source_completion.json", {
        "schema_version": SCHEMA, "sources": list(seen.values()), "source_cases": len(seen),
        "new_sources": sum(v["status"] != "compatible_reused" for v in seen.values()),
        "source_env_steps": 0, "compatible_tasks": compatible_tasks, "compatible_groups": groups,
        "source_gate_pass": gate, "canonical_recipe": str(repo / EVAL_CONFIG),
        "seed": recipe["rng"]["inference_seed"], "spec": str(repo / SPEC)})


def _source_phase(repo: Path, output: Path) -> None:
    rows, recipe, spec = _authorities(repo)
    sources = output / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    results = output / "source_results.jsonl"
    prior = [json.loads(line) for line in results.read_text().splitlines()] if results.exists() else []
    seen = {(r["task"], r["demo"]): r for r in prior}
    if len(seen) != len(prior):
        raise ValueError("duplicate source result; do not silently rerun")
    for task in spec["task_ids"]:
        for demo in spec["source_demo_ids"]:
            if (task, demo) in seen:
                continue
            result = _source_record(rows[task], demo, sources)
            _append_jsonl(results, result)
            seen[task, demo] = result
    _source_plan(repo, output, seen, spec, recipe)


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


def _run_one(env: Any, row: dict[str, Any], demo: int, state: int,
             metadata: dict[str, Any], source: dict[str, np.ndarray],
             output: Path, horizon: int, *, initialized_observation: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    observation = initialized_observation
    owner = env.env
    references = tuple(item["reference_body"] for item in metadata["segments"])
    names, goals = roles(owner, references)
    if names != metadata["body_names"] or goals != metadata["goals"]:
        raise ValueError("query body/BDDL roles differ from source")
    initial = _query_sample(env, observation, names, goals)
    if all(initial["predicates"]):
        raise ValueError("query initial target already complete")
    differences = {name: float(np.linalg.norm(initial["body_pos"][names.index(name)] - value))
                   for name, value in metadata["source_initial_object_positions"].items()}
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
        elif step >= horizon:
            stop_reason = "horizon"

    for segment, declaration in enumerate(metadata["segments"]):
        if success or len(actions) >= horizon:
            break
        start, stop = int(declaration["start"]), int(declaration["stop"])
        ref = str(declaration["reference_body"])
        ref_id = names.index(ref)
        source_pos = source["body_pos"][start, ref_id]
        source_rot = source["body_rot"][start, ref_id]
        new_pos, new_rot = pose(owner.sim, body_id(owner, ref))
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
            if success or len(actions) >= horizon:
                break
        if success or len(actions) >= horizon:
            break
        for source_step in range(start, stop):
            target_pos, target_rot = _target(source_pos, source_rot, new_pos, new_rot,
                                            source["goal_pos"][source_step],
                                            source["goal_rot"][source_step])
            execute(target_pos, target_rot, float(source["actions"][source_step, 6]), segment, 1)
            if success or len(actions) >= horizon:
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


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _freeze_task_scenes(env: Any, init_states: Any, task: int, states: list[int],
                        recipe: dict[str, Any], names: list[str], goals: list[list[str]],
                        output: Path) -> None:
    """Complete four physical starts before running any reference behavior."""
    for state in states:
        path = output / "scenes" / f"task_{task}_state_{state}.npz"
        if path.exists():
            continue
        observation = _initialize_query(env, init_states, state, recipe)
        actual_names, actual_goals = roles(env.env, tuple(names))
        if actual_names != names or actual_goals != goals:
            raise ValueError(f"task{task} source/query roles changed")
        settled = _scene_snapshot(env, observation, names, goals, image=False)
        observation = _restore_scene(env, settled)
        np.savez_compressed(path, **_scene_snapshot(env, observation, names, goals, image=True))


def _collect_case(env: Any, init_states: Any, row: dict[str, Any], demo: int, state: int,
                  metadata: dict[str, Any], source_item: dict[str, Any], recipe: dict[str, Any],
                  spec: dict[str, Any], output: Path, scene_path: Path,
                  snapshot: dict[str, np.ndarray]) -> dict[str, Any]:
    task = int(row["global_task_id"])
    observation = _initialize_query(env, init_states, state, recipe)
    observation = _restore_scene(env, snapshot)
    _assert_scene_pair(env, observation, metadata["body_names"], metadata["goals"], snapshot, image=True)
    with np.load(source_item["arrays"]) as source_file:
        source = {key: source_file[key] for key in source_file.files}
    identity = {"task": task, "state": state, "demo": demo}
    _append_jsonl(output / "attempts.jsonl", {**identity, "status": "started"})
    try:
        result = _run_one(env, row, demo, state, metadata, source, output,
                          int(spec["construction"]["horizons"][row["suite"]]),
                          initialized_observation=observation)
    except Exception as exc:
        _append_jsonl(output / "attempts.jsonl", {**identity, "status": "engineering_error",
                                                    "error": f"{type(exc).__name__}: {exc}"})
        raise
    result["initial_scene_snapshot"] = str(scene_path)
    result["initial_scene_restoration"] = "all model body poses and post-dummy sim state"
    _append_jsonl(output / "rows.jsonl", result)
    _append_jsonl(output / "attempts.jsonl", {**identity, "status": "completed"})
    return result


def _collect_task(env: Any, init_states: Any, row: dict[str, Any], source_rows: dict,
                  recipe: dict[str, Any], spec: dict[str, Any], output: Path,
                  completed: set[tuple[int, int, int]], process_started: float,
                  existing_bytes: int) -> int:
    task = int(row["global_task_id"])
    source_meta = {demo: json.loads(Path(source_rows[task, demo]["metadata"]).read_text())
                   for demo in spec["source_demo_ids"]}
    if len({tuple(meta["body_names"]) for meta in source_meta.values()}) != 1 or len(
            {json.dumps(meta["goals"]) for meta in source_meta.values()}) != 1:
        raise ValueError(f"four source roles differ for task{task}")
    names, goals = source_meta[0]["body_names"], source_meta[0]["goals"]
    _freeze_task_scenes(env, init_states, task, spec["query_init_state_ids"],
                        recipe, names, goals, output)
    budget = float(spec["runtime"]["full_gpu_hours_limit"]) * 3600
    output_limit = int(spec["runtime"]["new_output_peak_gib"] * 1024**3)
    for state in spec["query_init_state_ids"]:
        scene_path = output / "scenes" / f"task_{task}_state_{state}.npz"
        with np.load(scene_path) as scene_file:
            snapshot = {key: scene_file[key] for key in scene_file.files}
        for demo in spec["source_demo_ids"]:
            if (task, state, demo) in completed:
                continue
            if time.monotonic() - process_started >= budget - 60:
                raise RuntimeError("full GPU time guard reached before next attempt")
            if existing_bytes + 50 * 1024**2 > output_limit:
                raise RuntimeError("data1 output guard reached before next attempt")
            result = _collect_case(env, init_states, row, demo, state, source_meta[demo],
                                   source_rows[task, demo], recipe, spec, output, scene_path, snapshot)
            completed.add((task, state, demo))
            existing_bytes += result["trace_bytes"]
    return existing_bytes


def _collect_phase(repo: Path, output: Path, gpu_index: int, process_started: float) -> None:
    rows, recipe, spec = _authorities(repo)
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    source_status = json.loads((output / "source_completion.json").read_text())
    if not source_status["source_gate_pass"] or source_status["source_cases"] != 108:
        raise ValueError("complete compatible CPU source phase required")
    source_rows = {(item["task"], item["demo"]): item for item in source_status["sources"]}
    (output / "episodes").mkdir(exist_ok=True)
    (output / "scenes").mkdir(exist_ok=True)
    results = output / "rows.jsonl"
    prior = [json.loads(line) for line in results.read_text().splitlines()] if results.exists() else []
    completed = {(item["task"], item["state"], item["demo"]) for item in prior}
    if len(completed) != len(prior):
        raise ValueError("duplicate behavior row; never retry a physical attempt")
    attempts = output / "attempts.jsonl"
    if attempts.exists():
        attempted = [json.loads(line) for line in attempts.read_text().splitlines()]
        if any(item["status"] == "started" and not any(
                later["status"] in {"completed", "engineering_error"}
                and all(later[key] == item[key] for key in ("task", "state", "demo"))
                for later in attempted[index + 1:]) for index, item in enumerate(attempted)):
            raise ValueError("unfinished physical attempt; do not silently retry")
    existing_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    suites: dict[str, Any] = {}
    for task in spec["task_ids"]:
        if task not in source_status["compatible_tasks"]:
            continue
        if all((task, state, demo) in completed for state in spec["query_init_state_ids"]
               for demo in spec["source_demo_ids"]):
            continue
        row = rows[task]
        if row["suite"] not in suites:
            suites[row["suite"]] = benchmark.get_benchmark_dict()[row["suite"]]()
        suite = suites[row["suite"]]
        init_states = suite.get_task_init_states(row["task_id"])
        env = OffScreenRenderEnv(bddl_file_name=row["bddl_path"], camera_heights=256,
                                 camera_widths=256, render_gpu_device_id=gpu_index)
        try:
            existing_bytes = _collect_task(env, init_states, row, source_rows, recipe, spec,
                                           output, completed, process_started, existing_bytes)
        finally:
            env.close()


def main() -> None:
    process_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("source", "collect"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--gpu-index", type=int)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    if output != ROOT or (args.phase == "collect" and args.gpu_index is None):
        raise ValueError("fixed study root or rendering GPU missing")
    if args.phase == "source":
        _source_phase(repo, output)
    else:
        _collect_phase(repo, output, args.gpu_index, process_started)


if __name__ == "__main__":
    main()
