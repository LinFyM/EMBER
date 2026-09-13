#!/usr/bin/env python3
"""CPU causal motion-effect audit; operation_semantics_feasibility.md section 6."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
from bddl.parsing import scan_tokens
from scipy.spatial.transform import Rotation

from ember.pi05_assets import configure_libero_runtime_assets, prepare_libero_config


EPSILON = .1
DIRECTION = np.array([1, -1, 1, -1, 1, -1], dtype=float) / np.sqrt(6)
VARIANTS = ["reference", *[f"axis{axis}_{sign}" for axis in range(6) for sign in ("plus", "minus")],
            "combined_plus", "combined_minus", "gripper_open", "gripper_close"]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def mapped_xml(text, assets, robosuite):
    xml = ET.fromstring(text)
    for element in xml.findall("./asset/*"):
        name = element.get("file")
        if name is None:
            continue
        if "/robosuite/" in name:
            path = robosuite / name.split("/robosuite/")[-1]
        elif "/assets/" in name:
            path = assets / name.split("/assets/")[-1]
        else:
            raise ValueError(f"Unmapped asset {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        element.set("file", str(path))
    return ET.tostring(xml, encoding="unicode")


def points_for(model, objects):
    points, weights = [], []
    for obj in objects:
        group = []
        for kind, count in (("body", model.nbody), ("site", model.nsite)):
            for index in range(count):
                name = getattr(model, f"{kind}_id2name")(index) or ""
                if name == obj or name.startswith(obj + "_"):
                    group.append((kind, index, name, obj))
        if not group:
            raise ValueError(f"Missing object points: {obj}")
        points.extend(group)
        weights.extend([1 / np.sqrt(len(objects) * len(group))] * len(group))
    return points, np.asarray(weights)


def geometry(env, points):
    env.sim.forward()
    data = env.sim.data
    positions, rotations = [], []
    for kind, index, _, _ in points:
        positions.append(getattr(data, "body_xpos" if kind == "body" else "site_xpos")[index].copy())
        rotations.append(getattr(data, "body_xmat" if kind == "body" else "site_xmat")[index].reshape(3, 3).copy())
    return np.asarray(positions), np.asarray(rotations)


def rotation_difference(reference, value):
    relative = reference.swapaxes(-1, -2) @ value
    return Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(*relative.shape[:-2], 3)


def action_variants(actions):
    variants = [actions.copy()]
    for axis in range(6):
        for sign in (1, -1):
            value = actions.copy()
            value[:, axis] += sign * EPSILON
            variants.append(np.clip(value, -1, 1))
    for sign in (1, -1):
        value = actions.copy()
        value[:, :6] += sign * EPSILON * DIRECTION
        variants.append(np.clip(value, -1, 1))
    for sign in (-1, 1):
        value = actions.copy()
        value[:, -1] = sign
        variants.append(value)
    return np.asarray(variants)


def gripper_history(env, actions):
    gripper = env.robots[0].gripper
    gripper.current_action[:] = 0
    substeps = round(env.env.control_timestep / env.env.model_timestep)
    history = []
    for action in actions:
        for _ in range(substeps):
            gripper.format_action(action[-1:])
        history.append(gripper.current_action.copy())
    return np.asarray(history)


def execute(env, state, grip, actions, points):
    env.sim.reset()
    env.set_state(state)
    env.sim.data.qacc_warmstart[:] = 0
    env.sim.forward()
    env.env.done, env.env.timestep, env.env.cur_time = False, 0, float(state[0])
    robot = env.robots[0]
    robot.gripper.current_action[:] = grip
    robot.controller.update(force=True)
    robot.controller.reset_goal()
    for action in actions:
        env.step(action)  # Rewards and done are computed by LIBERO, but never consumed.
    return geometry(env, points)


def episode(env, demo, objects, assets, robosuite):
    env.reset_from_xml_string(mapped_xml(demo.attrs["model_file"], assets, robosuite))
    states, actions = demo["states"][:], demo["actions"][:]
    if states.shape != (len(actions), 1 + env.sim.model.nq + env.sim.model.nv):
        raise ValueError("Stored state schema changed")
    if actions.shape[1:] != (7,) or not np.isfinite(actions).all():
        raise ValueError("Expected finite seven-dimensional controls")
    points, weights = points_for(env.sim.model, objects)
    frames = np.array([int((len(states) - 7) * fraction / 5) * 5 for fraction in (.25, .5, .75)])
    if len(set(frames)) != 3 or min(frames) < 5:
        raise ValueError("Three registered positions require sufficient recorded history")
    history = gripper_history(env, actions)
    all_positions, all_rotations, all_effects, predictions, scores, recorded_errors = [], [], [], [], [], []
    for frame in frames:
        variants = action_variants(actions[frame + 1:frame + 6])
        outputs = [execute(env, states[frame + 1], history[frame], a, points) for a in variants]
        positions, rotations = np.asarray([x[0] for x in outputs]), np.asarray([x[1] for x in outputs])
        effects = np.concatenate(((positions - positions[0]) / .05,
                                  rotation_difference(rotations[0], rotations) / .5), axis=-1)
        effects *= weights[None, :, None]
        jacobian = (effects[1:13:2] - effects[2:13:2]) / (2 * EPSILON)
        prediction = np.einsum("a,apd->pd", EPSILON * DIRECTION, jacobian)
        prediction = np.stack((prediction, -prediction))
        actual = effects[13:15]
        scores.append([float(np.square(actual).sum() / 12),
                       float(np.square(actual - prediction).sum() / 12)])
        env.set_state(states[frame + 6])
        target_p, target_r = geometry(env, points)
        recorded_errors.append([float(np.sqrt(np.mean(np.square(positions[0] - target_p).sum(-1)))),
                                float(np.sqrt(np.mean(np.square(rotation_difference(target_r, rotations[0])).sum(-1))))])
        all_positions.append(positions)
        all_rotations.append(rotations)
        all_effects.append(effects)
        predictions.append(prediction)
    arrays = {"frames": frames, "positions": np.asarray(all_positions), "rotations": np.asarray(all_rotations),
              "effects": np.asarray(all_effects), "combined_prediction": np.asarray(predictions),
              "scores": np.asarray(scores), "reference_vs_recorded_rms": np.asarray(recorded_errors),
              "point_weights": weights}
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Nonfinite physical effects")
    return arrays, points


def summarize(rows):
    tasks = sorted({row["task_id"] for row in rows})
    task_values = np.array([np.mean([row["scores"] for row in rows if row["task_id"] == task], axis=0) for task in tasks])
    difference = task_values[:, 0] - task_values[:, 1]
    rng = np.random.default_rng(20260917)
    ci = np.quantile(difference[rng.integers(0, len(tasks), (20000, len(tasks)))].mean(-1), [.025, .975])
    suites = {}
    for suite in sorted({row["suite"] for row in rows}):
        suites[suite] = np.mean([row["scores"] for row in rows if row["suite"] == suite], axis=0).tolist()
    global_values = task_values.mean(0)
    ratio = float(global_values[1] / global_values[0]) if global_values[0] else None
    positive_suites = sum(value[0] > value[1] for value in suites.values())
    return {"score_order": ["zero_change_mse", "jacobian_prediction_mse"],
            "scores": global_values.tolist(), "prediction_to_zero_mse_ratio": ratio,
            "zero_minus_prediction_ci95": ci.tolist(), "positive_suites": positive_suites,
            "suite_scores": suites, "task_scores": dict(zip(tasks, task_values.tolist(), strict=True)),
            "local_effect_premise_passed": bool(ratio is not None and ratio < .25 and ci[0] > 0 and positive_suites >= 2),
            "not_a_writer_or_video_qualification": True}


def run(args):
    start = time.time()
    root, out = args.asset_root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    reuse = json.loads((root / "configs/pi05_writer_data_v1.json").read_text())["authorities"]
    assets = (root / reuse["libero_assets"]).resolve()
    os.environ["EMBER_LIBERO_ASSETS_ROOT"] = str(assets)
    paths = prepare_libero_config(out / "libero_config")
    configure_libero_runtime_assets(assets)
    from libero.libero.envs.env_wrapper import ControlEnv
    robosuite = Path(importlib.util.find_spec("robosuite").origin).parent
    manifest = json.loads((root / "configs/pi05_target_data_v1/manifest.json").read_text())
    protocol = json.loads((root / "configs/libero_24_8_8_v1/protocol.json").read_text())
    tasks = [row for row in manifest["tasks"] if row["split_role"] == "train"]
    if len(tasks) != 24:
        raise ValueError("Expected fixed train24")
    save(out / "registration.json", {"contract": "docs/operation_semantics_feasibility.md section 6",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "demos": [16, 17, 18, 19], "variants": VARIANTS, "epsilon": EPSILON, "direction": DIRECTION.tolist(),
        "learned_parameters": 0, "camera_rendering": False, "reward_or_terminal_scoring": False})
    rows, schemas = [], []
    for task in tasks:
        tid, suite = task["global_task_id"], task["suite"]
        if task["task_id"] not in protocol["split"]["suites"][suite]["train"]:
            raise ValueError("Task crossed the fixed split")
        bddl = Path(paths["bddl_files"]) / task["problem_folder"] / task["bddl"]["filename"]
        objects = next(g[1:] for g in scan_tokens(filename=str(bddl)) if isinstance(g, list) and g[0] == ":obj_of_interest")
        env = ControlEnv(str(bddl), use_camera_obs=False, has_offscreen_renderer=False,
                         has_renderer=False, ignore_done=True)
        try:
            path = root / "data/datasets" / manifest["dataset"]["revision"] / task["hdf5"]["relative_path"]
            with h5py.File(path, "r") as data:
                for demo in range(16, 20):
                    arrays, points = episode(env, data[f"data/demo_{demo}"], objects, assets, robosuite)
                    np.savez_compressed(out / f"task{tid}_demo{demo}.npz", **arrays)
                    rows.append({"task_id": tid, "suite": suite, "demo": demo,
                                 "scores": arrays["scores"].mean(0).tolist(),
                                 "recorded_rms": arrays["reference_vs_recorded_rms"].tolist()})
                    schemas.append({"task_id": tid, "demo": demo, "objects": objects, "points": points})
        finally:
            env.close()
        print(json.dumps({"task_id": tid, "episodes": 4}), flush=True)
    save(out / "episode_scores.json", rows)
    save(out / "schema.json", schemas)
    save(out / "summary.json", summarize(rows))
    save(out / "completion.json", {"status": "complete", "exit_code": 0, "elapsed_seconds": time.time() - start,
                                    "episodes": len(rows), "conditions": 3 * len(rows), "action_steps": 3 * len(rows) * 17 * 5})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
