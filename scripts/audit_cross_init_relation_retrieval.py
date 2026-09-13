#!/usr/bin/env python3
"""Disposable CPU analysis; contract: docs/cross_init_relation_retrieval_audit.md."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import mujoco
import numpy as np
from bddl.parsing import scan_tokens
from scipy.spatial.transform import Rotation


ARMS = ("absolute", "relative", "video_mean")
COMPONENTS = ("all7", "translation3", "rotation3", "gripper1")
TEACHERS, QUERIES = tuple(range(16, 20)), tuple(range(42, 46))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def geometry_episode(demo, objects, asset_root: Path, robosuite_root: Path):
    xml = ET.fromstring(demo.attrs["model_file"])
    for element in xml.findall("./asset/*"):
        name = element.get("file")
        if name is None:
            continue
        if "/robosuite/" in name:
            path = robosuite_root / name.split("/robosuite/")[-1]
        elif "/assets/" in name:
            path = asset_root / name.split("/assets/")[-1]
        else:
            raise ValueError(f"Unmapped asset {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        element.set("file", str(path))
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    data = mujoco.MjData(model)
    points, signature, weights = [], [], []
    for obj in objects:
        group = []
        for kind, count in (("body", model.nbody), ("site", model.nsite)):
            for index in range(count):
                name = getattr(model, kind)(index).name
                if name == obj or name.startswith(obj + "_"):
                    group.append((kind, index, name))
        if not group:
            raise ValueError(f"No body/site for {obj}")
        group.sort(key=lambda item: (item[0], item[2]))
        points.extend((kind, index) for kind, index, _ in group)
        signature.extend((obj, kind, name) for kind, _, name in group)
        weights.extend([1 / (len(objects) * len(group))] * len(group))
    weights = np.sqrt(np.asarray(weights, dtype=np.float64))
    grip_sites = [i for i in range(model.nsite) if model.site(i).name.endswith("grip_site")]
    if len(grip_sites) != 1:
        raise ValueError("Expected the single Panda grip site for coordinate alignment")
    states = demo["states"][:]
    count = demo["actions"].shape[0]
    if states.shape != (count, 1 + model.nq + model.nv) or model.na != 0:
        raise ValueError("Stored simulation state layout differs from the contract")
    frames = np.arange(0, count - 5, 5, dtype=np.int64)
    if len(frames) == 0:
        raise ValueError("No full five-action future chunk")
    ee = np.asarray(demo["obs/ee_states"][:], dtype=np.float64)[frames]
    gripper = np.asarray(demo["obs/gripper_states"][:], dtype=np.float64)[frames]
    if ee.shape != (len(frames), 6) or gripper.shape != (len(frames), 2):
        raise ValueError("Expected official six-dimensional EEF and two-finger state")
    positions, rotations, coordinate_errors = [], [], []
    for frame, own in zip(frames, ee):
        state = states[frame + 1]
        data.time = state[0]
        data.qpos[:] = state[1:1 + model.nq]
        data.qvel[:] = state[1 + model.nq:]
        mujoco.mj_forward(model, data)
        coordinate_errors.append(float(np.linalg.norm(data.site_xpos[grip_sites[0]] - own[:3])))
        positions.append([getattr(data, "xpos" if kind == "body" else "site_xpos")[i].copy()
                          for kind, i in points])
        rotations.append([getattr(data, "xmat" if kind == "body" else "site_xmat")[i].copy()
                          for kind, i in points])
    if max(coordinate_errors) > 1e-4:
        raise ValueError(f"Post-action coordinate alignment failed: {max(coordinate_errors)} m")
    p, r = np.asarray(positions), np.asarray(rotations)
    common = np.concatenate([
        Rotation.from_rotvec(ee[:, 3:]).as_matrix().reshape(len(frames), -1) / np.sqrt(2),
        (r * weights[None, :, None] / np.sqrt(2)).reshape(len(frames), -1),
        gripper / (0.04 * np.sqrt(2)),
    ], axis=1)
    absolute = np.concatenate([
        common, ee[:, :3] / 0.10,
        (p * weights[None, :, None] / 0.10).reshape(len(frames), -1),
    ], axis=1)
    relative = np.concatenate([
        common, ((p - ee[:, None, :3]) * weights[None, :, None] / 0.10).reshape(len(frames), -1),
    ], axis=1)
    if not np.isfinite(absolute).all() or not np.isfinite(relative).all():
        raise ValueError("Nonfinite geometry")
    actions = np.asarray(demo["actions"][:], dtype=np.float64)
    chunks = np.stack([actions[i + 1:i + 6] for i in frames])
    if chunks.shape != (len(frames), 5, 7) or not np.isfinite(chunks).all():
        raise ValueError("Invalid future action chunks")
    return {"frames": frames, "absolute": absolute, "relative": relative, "actions": chunks,
            "signature": signature, "max_coordinate_error_m": max(coordinate_errors)}


def nearest(query, teacher):
    # Direct squared distance avoids cancellation for near-identical coordinates.
    distance = np.square(query[:, None, :] - teacher[None, :, :]).sum(axis=-1)
    indices = distance.argmin(axis=1)
    return indices, distance[np.arange(len(indices)), indices]


def run(args):
    start = time.time()
    root, out = args.asset_root.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError("A completed or partial analysis root must not be overwritten")
    out.mkdir(parents=True)
    manifest = json.loads((root / "configs/pi05_target_data_v1/manifest.json").read_text())
    protocol = json.loads((root / "configs/libero_24_8_8_v1/protocol.json").read_text())
    norm = json.loads((root / "configs/pi05_source_corpus_v1/source_normalization.json").read_text())["stats"]["action"]
    low, high = np.asarray(norm["q01"]), np.asarray(norm["q99"])
    reuse = json.loads((root / "configs/pi05_writer_data_v1.json").read_text())["authorities"]
    asset_root = (root / reuse["libero_assets"]).resolve()
    libero_root = Path(importlib.util.find_spec("libero").origin).parent / "libero"
    robosuite_root = Path(importlib.util.find_spec("robosuite").origin).parent
    tasks = [row for row in manifest["tasks"] if row["split_role"] == "train"]
    if len(tasks) != 24:
        raise ValueError("Expected the fixed train24")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write_json(out / "registration.json", {
        "contract": "docs/cross_init_relation_retrieval_audit.md", "code_commit": commit,
        "asset_root": str(root), "task_ids": [r["global_task_id"] for r in tasks],
        "teacher_demos": TEACHERS, "query_demos": QUERIES, "stride": 5, "action_start_offset": 1,
        "chunk_size": 5, "bootstrap_seed": 20260916, "bootstrap_replicates": 20000,
        "position_scale_m": 0.10, "gripper_scale_m": 0.04,
        "learned_parameters": 0, "uses_privileged_teacher_actions_and_geometry": True,
    })
    all_ids, all_distances, raw = [], [], {name: [] for name in ("target", *ARMS)}
    task_scores, episode_rows, schemas = [], [], []
    for task in tasks:
        tid, suite = task["global_task_id"], task["suite"]
        if task["task_id"] not in protocol["split"]["suites"][suite]["train"]:
            raise ValueError("Task crosses fixed development split")
        bddl = libero_root / "bddl_files" / task["problem_folder"] / task["bddl"]["filename"]
        ast = scan_tokens(filename=str(bddl))
        objects = next(group[1:] for group in ast if isinstance(group, list) and group[0] == ":obj_of_interest")
        if not objects or len(set(objects)) != len(objects):
            raise ValueError("Expected unique official objects of interest")
        path = root / "data/datasets" / manifest["dataset"]["revision"] / task["hdf5"]["relative_path"]
        episodes = {}
        with h5py.File(path, "r") as data:
            for demo_id in (*TEACHERS, *QUERIES):
                episodes[demo_id] = geometry_episode(data[f"data/demo_{demo_id}"], objects, asset_root, robosuite_root)
                episodes[demo_id]["actions"] = 2 * (episodes[demo_id]["actions"] - low) / (high - low + 1e-6) - 1
        reference = episodes[TEACHERS[0]]["signature"]
        if any(ep["signature"] != reference for ep in episodes.values()):
            raise ValueError("Body/site correspondence differs across episodes")
        schemas.append({"task_id": tid, "suite": suite, "objects": objects, "points": reference,
                        "frames_by_demo": {str(i): ep["frames"].tolist() for i, ep in episodes.items()},
                        "max_coordinate_error_m": max(ep["max_coordinate_error_m"] for ep in episodes.values())})
        cell_scores = []
        for teacher_id in TEACHERS:
            teacher = episodes[teacher_id]
            for query_id in QUERIES:
                query = episodes[query_id]
                abs_idx, abs_distance = nearest(query["absolute"], teacher["absolute"])
                rel_idx, rel_distance = nearest(query["relative"], teacher["relative"])
                predicted = {"absolute": teacher["actions"][abs_idx], "relative": teacher["actions"][rel_idx],
                             "video_mean": np.broadcast_to(teacher["actions"].mean(axis=0), query["actions"].shape)}
                scores = []
                for arm in ARMS:
                    error = np.square(predicted[arm] - query["actions"])
                    scores.append([error.mean(), error[..., :3].mean(), error[..., 3:6].mean(), error[..., 6].mean()])
                    raw[arm].append(predicted[arm].astype(np.float32))
                cell_scores.append(scores)
                episode_rows.append({"task_id": tid, "teacher_demo": teacher_id, "query_demo": query_id,
                                     "query_count": len(query["frames"]), "scores": np.asarray(scores).tolist()})
                n = len(query["frames"])
                all_ids.append(np.column_stack([np.full(n, tid), np.full(n, teacher_id), np.full(n, query_id),
                                                query["frames"], teacher["frames"][abs_idx], teacher["frames"][rel_idx]]))
                all_distances.append(np.column_stack([abs_distance, rel_distance]))
                raw["target"].append(query["actions"].astype(np.float32))
        score = np.mean(cell_scores, axis=0)
        task_scores.append(score)
        print(json.dumps({"task_id": tid, "suite": suite, "mse": dict(zip(ARMS, score[:, 0].tolist()))}), flush=True)
    values = np.asarray(task_scores)
    suites = sorted({task["suite"] for task in tasks})
    suite_values = {suite: values[[i for i, task in enumerate(tasks) if task["suite"] == suite]].mean(axis=0)
                    for suite in suites}
    rng = np.random.default_rng(20260916)
    draws = rng.integers(0, len(tasks), size=(20000, len(tasks)))
    comparisons = {}
    for name, index in (("absolute_minus_relative", 0), ("video_mean_minus_relative", 2)):
        difference = values[:, index, 0] - values[:, 1, 0]
        comparisons[name] = {"mean": float(difference.mean()), "ci95": np.quantile(difference[draws].mean(axis=1), [.025, .975]).tolist(),
                             "positive_tasks": int((difference > 0).sum()),
                             "suite_differences": {suite: float(score[index, 0] - score[1, 0]) for suite, score in suite_values.items()}}
    good_suites = [suite for suite, score in suite_values.items() if score[0, 0] > score[1, 0] and score[2, 0] > score[1, 0]]
    passed = all(result["ci95"][0] > 0 for result in comparisons.values()) and len(good_suites) >= 2
    np.savez_compressed(out / "raw_rows.npz", ids=np.concatenate(all_ids).astype(np.int32),
                        distances=np.concatenate(all_distances).astype(np.float32), **{key: np.concatenate(chunks) for key, chunks in raw.items()})
    summary = {"arms": ARMS, "components": COMPONENTS, "global_scores": values.mean(axis=0).tolist(),
               "suite_scores": {suite: score.tolist() for suite, score in suite_values.items()},
               "task_scores": [{"task_id": task["global_task_id"], "suite": task["suite"], "scores": score.tolist()}
                               for task, score in zip(tasks, values)],
               "comparisons": comparisons, "positive_suites_for_both": good_suites,
               "registered_geometry_transfer_premise_passed": passed,
               "episode_pairs": len(episode_rows), "row_count": sum(len(ids) for ids in all_ids),
               "not_writer_or_closed_loop_qualification": True}
    write_json(out / "summary.json", summary)
    write_json(out / "episode_scores.json", episode_rows)
    write_json(out / "geometry_schema.json", schemas)
    write_json(out / "completion.json", {"status": "complete", "exit_code": 0, "elapsed_seconds": time.time() - start,
                                         "task_count": len(tasks), "episode_pairs": len(episode_rows), "row_count": summary["row_count"]})
    print(json.dumps({"comparisons": comparisons, "premise_passed": passed, "row_count": summary["row_count"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        run(arguments)
    except FileExistsError:
        raise
    except Exception:
        if arguments.output.exists() and not (arguments.output / "completion.json").exists():
            write_json(arguments.output / "failure.json", {"traceback": traceback.format_exc()})
        raise
