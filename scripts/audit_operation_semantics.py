#!/usr/bin/env python3
"""Bounded physical-label feasibility; docs/operation_semantics_feasibility.md."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import mujoco
import numpy as np
from bddl.parsing import scan_tokens


POOLS = {"train": tuple(range(16, 20)), "diagnostic": tuple(range(42, 46))}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_model(demo, assets, robosuite):
    xml = ET.fromstring(demo.attrs["model_file"])
    for element in xml.findall("./asset/*"):
        name = element.get("file")
        if name is None:
            continue
        if "/robosuite/" in name:
            path = robosuite / name.split("/robosuite/")[-1]
        elif "/assets/" in name:
            path = assets / name.split("/assets/")[-1]
        else:
            raise ValueError(f"Unmapped model asset: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        element.set("file", str(path))
    return mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))


def object_schema(model, objects):
    records = []
    for obj in objects:
        matches = lambda name: name == obj or name.startswith(obj + "_")
        bodies = [i for i in range(1, model.nbody) if matches(model.body(i).name)]
        sites = [i for i in range(model.nsite) if matches(model.site(i).name)]
        if not bodies and not sites:
            raise ValueError(f"No object geometry for {obj}")
        owners = set(bodies or [int(model.site_bodyid[i]) for i in sites]) - {0}
        # Descendants retain articulated parts; a region's body is not its spatial extent.
        descendants = set(owners)
        for body in range(1, model.nbody):
            parent = body
            while parent:
                if parent in owners:
                    descendants.add(body)
                    break
                parent = int(model.body_parentid[parent])
        geoms = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in descendants]
        records.append({"object": obj, "site_only": not bodies,
                        "points": [("body", i, model.body(i).name) for i in bodies]
                        + [("site", i, model.site(i).name) for i in sites],
                        "contact_bodies": [model.body(i).name for i in sorted(descendants)],
                        "contact_geoms": geoms,
                        "contact_geom_names": [model.geom(i).name for i in geoms]})
    return records


def extract(demo, objects, assets, robosuite):
    model = load_model(demo, assets, robosuite)
    data = mujoco.MjData(model)
    schema = object_schema(model, objects)
    fingers = []
    for side in (1, 2):
        suffixes = (f"finger{side}_collision", f"finger{side}_pad_collision")
        ids = {i for i in range(model.ngeom) if model.geom(i).name.endswith(suffixes)}
        if len(ids) != 2:
            raise ValueError("Panda finger collision mapping changed")
        fingers.append(ids)
    grip = [i for i in range(model.nsite) if model.site(i).name.endswith("grip_site")]
    if len(grip) != 1:
        raise ValueError("Expected one EEF grip site")
    states = demo["states"][:]
    if states.ndim != 2 or states.shape[1] != 1 + model.nq + model.nv or model.na:
        raise ValueError("Stored state contract changed")
    frames = np.arange(0, len(states) - 1, 5, dtype=np.int32)
    own = demo["obs/ee_states"][:][frames, :3]
    points = [point for record in schema for point in record["points"]]
    geoms = [set(record["contact_geoms"]) for record in schema]
    positions, rotations, contacts, errors = [], [], [], []
    for frame, ee in zip(frames, own, strict=True):
        state = states[frame + 1]
        data.time = state[0] - model.opt.timestep
        data.qpos[:] = state[1:1 + model.nq]
        data.qvel[:] = state[1 + model.nq:]
        mujoco.mj_integratePos(model, data.qpos, data.qvel, -model.opt.timestep)
        mujoco.mj_forward(model, data)
        errors.append(float(np.linalg.norm(data.site_xpos[grip[0]] - ee)))
        positions.append([getattr(data, "xpos" if kind == "body" else "site_xpos")[i].copy()
                          for kind, i, _ in points])
        rotations.append([getattr(data, "xmat" if kind == "body" else "site_xmat")[i].reshape(3, 3).copy()
                          for kind, i, _ in points])
        contact = np.zeros((len(objects), 2), dtype=bool)
        for item in data.contact[:data.ncon]:
            if item.dist > 0:
                continue
            a, b = int(item.geom1), int(item.geom2)
            for obj, members in enumerate(geoms):
                for side, finger in enumerate(fingers):
                    contact[obj, side] |= (a in members and b in finger) or (b in members and a in finger)
        contacts.append(contact)
    p, r, c = np.asarray(positions), np.asarray(rotations), np.asarray(contacts)
    if not len(frames) or max(errors) > 1e-4 or not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError("Invalid physical labels or observation time alignment")
    delta = np.diff(p, axis=0)
    relative = r[:-1].swapaxes(-1, -2) @ r[1:]
    angle = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    arrays = {"frames": frames, "positions": p.astype(np.float32), "rotations": r.astype(np.float32),
              "finger_contact": c, "past_translation": delta.astype(np.float32),
              "past_rotation_radians": angle.astype(np.float32)}
    bilateral = c.all(-1)
    summary = {"frames": len(frames), "object_frames": int(c.shape[0] * c.shape[1]),
               "left_positive": int(c[..., 0].sum()), "right_positive": int(c[..., 1].sum()),
               "bilateral_positive": int(bilateral.sum()),
               "bilateral_transitions": int((bilateral[1:] != bilateral[:-1]).sum()),
               "objects_with_bilateral": int(bilateral.any(0).sum()),
               "max_coordinate_error_m": max(errors), "timestep_seconds": float(model.opt.timestep),
               "point_max_translation_m": np.linalg.norm(delta, axis=-1).max(0).tolist(),
               "point_max_rotation_radians": angle.max(0).tolist(),
               "object_bilateral_positive": bilateral.sum(0).tolist()}
    return arrays, schema, summary


def run(args):
    start = time.time()
    root, out = args.asset_root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root / "configs/pi05_target_data_v1/manifest.json").read_text())
    protocol = json.loads((root / "configs/libero_24_8_8_v1/protocol.json").read_text())
    authority = json.loads((root / "configs/pi05_writer_data_v1.json").read_text())["authorities"]
    assets = (root / authority["libero_assets"]).resolve()
    libero = Path(importlib.util.find_spec("libero").origin).parent / "libero"
    robosuite = Path(importlib.util.find_spec("robosuite").origin).parent
    tasks = [row for row in manifest["tasks"] if row["split_role"] == "train"]
    if len(tasks) != 24:
        raise ValueError("Expected fixed train24")
    save(out / "registration.json", {"contract": "docs/operation_semantics_feasibility.md",
         "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
         "pools": POOLS, "stride": 5, "actions_reward_terminal_read": False,
         "environment_steps": 0, "learned_parameters": 0, "not_a_writer_qualification": True})
    summaries, schemas = [], []
    for task in tasks:
        tid, suite = int(task["global_task_id"]), task["suite"]
        if task["task_id"] not in protocol["split"]["suites"][suite]["train"]:
            raise ValueError("Task crossed fixed split")
        bddl = libero / "bddl_files" / task["problem_folder"] / task["bddl"]["filename"]
        objects = next(g[1:] for g in scan_tokens(filename=str(bddl))
                       if isinstance(g, list) and g[0] == ":obj_of_interest")
        path = root / "data/datasets" / manifest["dataset"]["revision"] / task["hdf5"]["relative_path"]
        with h5py.File(path, "r") as file:
            for pool, demos in POOLS.items():
                for demo in demos:
                    arrays, schema, summary = extract(file[f"data/demo_{demo}"], objects, assets, robosuite)
                    np.savez_compressed(out / f"task{tid}_demo{demo}.npz", **arrays)
                    summaries.append({"task_id": tid, "suite": suite, "pool": pool, "demo": demo, **summary})
                    schemas.append({"task_id": tid, "demo": demo, "objects": schema})
        print(json.dumps({"task_id": tid, "episodes": 8}), flush=True)
    aggregates = []
    for pool in POOLS:
        for suite in sorted({r["suite"] for r in summaries}):
            rows = [r for r in summaries if r["pool"] == pool and r["suite"] == suite]
            aggregates.append({"pool": pool, "suite": suite, "episodes": len(rows),
                **{k: sum(r[k] for r in rows) for k in ("frames", "object_frames", "bilateral_positive", "bilateral_transitions")},
                "episodes_with_bilateral_transitions": sum(r["bilateral_transitions"] > 0 for r in rows)})
    save(out / "episode_summary.json", summaries)
    save(out / "schema.json", schemas)
    save(out / "summary.json", {"aggregates": aggregates, "episode_count": len(summaries),
         "frame_count": sum(r["frames"] for r in summaries),
         "max_coordinate_error_m": max(r["max_coordinate_error_m"] for r in summaries)})
    save(out / "completion.json", {"status": "complete", "exit_code": 0, "elapsed_seconds": time.time() - start})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
