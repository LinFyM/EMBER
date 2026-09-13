"""Build training-only visible-object/rigid-motion labels; never read actions."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
import multiprocessing
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import cv2
import h5py
import mujoco
import numpy as np
from bddl.parsing import scan_tokens


SCHEMA = "visible_object_attention_labels_v1"
CAMERAS = ("agentview", "robot0_eye_in_hand")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def object_bodies(model, objects):
    groups = []
    for name in objects:
        roots = {i for i in range(1, model.nbody)
                 if model.body(i).name == name or model.body(i).name.startswith(name + "_")}
        roots.update(int(model.site_bodyid[i]) for i in range(model.nsite) if model.site(i).name == name)
        roots.discard(0)
        if not roots:
            raise ValueError(f"unresolved OOI {name}")
        members = set(roots)
        for body in range(1, model.nbody):
            parent = body
            while parent and parent not in roots:
                parent = int(model.body_parentid[parent])
            if parent in roots:
                members.add(body)
        groups.append(sorted(members))
    return groups


def restore_observation(model, data, state):
    data.time = state[0] - model.opt.timestep
    data.qpos[:] = state[1:1 + model.nq]
    data.qvel[:] = state[1 + model.nq:]
    mujoco.mj_integratePos(model, data.qpos, data.qvel, -model.opt.timestep)
    mujoco.mj_forward(model, data)


def patch_pool(pixels, *, prior):
    if prior:
        resized = cv2.resize(pixels, (438, 438), interpolation=cv2.INTER_LINEAR)[27:411, 27:411]
        grid, patch = 24, 16
    else:
        resized = cv2.resize(pixels, (224, 224), interpolation=cv2.INTER_LINEAR)
        grid, patch = 16, 14
    return resized.reshape(grid, patch, grid, patch).mean((1, 3)).reshape(-1)


def visible_bodies(renderer, model, data, option, camera):
    renderer.update_scene(data, camera=camera, scene_option=option)
    segmentation = renderer.render()[:, ::-1]
    valid = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    body = np.full(valid.shape, -1, np.int32)
    body[valid] = model.geom_bodyid[segmentation[..., 0][valid]]
    return body


def spatial_targets(body_images, groups, motion, body_ids):
    foreground = np.zeros(body_images.shape, np.float32)
    for members in groups:
        mask = np.isin(body_images, members)
        area = int(mask.sum())
        if area:
            foreground += mask / area
    native = np.concatenate([patch_pool(view, prior=False) for view in foreground])
    if native.sum() > 0:
        native /= native.sum()
    moving = np.zeros(body_images.shape[1:], np.float32)
    for body, distance in zip(body_ids, motion, strict=True):
        mask = body_images[0] == body
        area = int(mask.sum())
        if area:
            moving += mask * (distance / area)
    return native, patch_pool(moving, prior=True)


def episode(job):
    root, output, row, demo_id = job
    root, output = Path(root), Path(output)
    assets = (root / json.loads((root / "configs/pi05_writer_data_v1.json").read_text())
              ["authorities"]["libero_assets"]).resolve()
    robosuite = Path(importlib.util.find_spec("robosuite").origin).parent
    libero = Path(importlib.util.find_spec("libero").origin).parent / "libero"
    tree = scan_tokens(filename=str(libero / "bddl_files" / row["problem_folder"] / row["bddl"]["filename"]))
    objects = next(part[1:] for part in tree if isinstance(part, list) and part[0] == ":obj_of_interest")
    path = root / "data/datasets" / row["revision"] / row["hdf5"]["relative_path"]
    with h5py.File(path, "r") as handle:
        demo = handle[f"data/demo_{demo_id}"]
        # Only geometry and RGB length/shape metadata: no action/state labels enter Writer inputs.
        states = demo["states"][:]
        shape = demo["obs/agentview_rgb"].shape
        if shape != demo["obs/eye_in_hand_rgb"].shape or shape[1:] != (128, 128, 3):
            raise ValueError("stored camera shape differs from the checked rendering contract")
        indices = list(range(0, shape[0], 5))
        if indices[-1] != shape[0] - 1:
            indices.append(shape[0] - 1)
        xml = ET.fromstring(demo.attrs["model_file"])
        for asset in xml.findall("./asset/*"):
            name = asset.get("file")
            if name:
                owner = robosuite / name.split("/robosuite/")[-1] if "/robosuite/" in name else assets / name.split("/assets/")[-1]
                asset.set("file", str(owner))
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    data = mujoco.MjData(model)
    groups = object_bodies(model, objects)
    body_ids = sorted(set().union(*map(set, groups)))
    radii = []
    for body in body_ids:
        geoms = np.flatnonzero((model.geom_bodyid == body) & np.isin(model.geom_group, (1, 2)))
        radii.append(max((np.linalg.norm(model.geom_pos[g]) + model.geom_rbound[g] for g in geoms), default=0.))
    option = mujoco.MjvOption()
    option.geomgroup[0] = 0
    renderer = mujoco.Renderer(model, 128, 128)
    renderer.enable_segmentation_rendering()
    native, prior, distances, visible, supported = [], [], [], [], []
    previous = None
    try:
        for frame in indices:
            valid = frame + 1 < len(states)
            supported.append(valid)
            motion = np.zeros(len(body_ids), np.float64)
            if valid:
                restore_observation(model, data, states[frame + 1])
                position, rotation = data.xpos[body_ids].copy(), data.xmat[body_ids].reshape(-1, 3, 3).copy()
                if previous is not None:
                    relative = previous[1].transpose(0, 2, 1) @ rotation
                    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)
                    motion = np.linalg.norm(position - previous[0], axis=-1) + 2 * np.asarray(radii) * np.sqrt((1 - cosine) / 2)
                previous = position, rotation
                bodies = np.stack([visible_bodies(renderer, model, data, option, camera) for camera in CAMERAS])
                object_target, motion_target = spatial_targets(bodies, groups, motion, body_ids)
                visible.append([int(np.isin(bodies, group).sum()) for group in groups])
            else:
                object_target, motion_target = np.zeros(512, np.float32), np.zeros(576, np.float32)
                visible.append([0] * len(groups))
            native.append(object_target)
            prior.append(motion_target)
            distances.append(motion)
    finally:
        renderer.close()
    arrays = dict(frame_indices=np.asarray(indices, np.int64), native_object=np.asarray(native, np.float32),
                  prior_motion=np.asarray(prior, np.float32), body_motion=np.asarray(distances, np.float32),
                  object_visible_pixels=np.asarray(visible, np.int32), state_supported=np.asarray(supported, bool))
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite physical label")
    task = int(row["global_task_id"])
    np.savez_compressed(output / f"task{task}_demo{demo_id}.npz", **arrays)
    return dict(task=task, demo=demo_id, suite=row["suite"], frames=len(indices), objects=objects,
                body_names=[model.body(body).name for body in body_ids], body_radii=radii,
                object_bodies=[[model.body(body).name for body in group] for group in groups],
                supported_frames=int(sum(supported)), object_frames=int((arrays["native_object"].sum(-1) > 0).sum()),
                motion_frames=int((arrays["prior_motion"].sum(-1) > 0).sum()), motion_mass=float(arrays["prior_motion"].sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("bounded CPU worker count required")
    started = time.monotonic()
    config = json.loads((args.asset_root / "configs/pi05_visible_object_video.json").read_text())
    manifest = json.loads((args.asset_root / "configs/pi05_target_data_v1/manifest.json").read_text())
    tasks = [row | {"revision": manifest["dataset"]["revision"]} for row in manifest["tasks"]
             if row["global_task_id"] in config["data"]["task_ids"]]
    if len(tasks) != 24 or any(row["split_role"] != "train" for row in tasks):
        raise ValueError("fixed train24 authority changed")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "registration.json", dict(schema=SCHEMA, tasks=config["data"]["task_ids"],
               demos=list(range(16)), design="docs/visible_object_grounded_writer_design.md", workers=args.workers,
               gradients=False, environment_steps=0, source_frame_stride=5, include_last_frame=True))
    jobs = [(str(args.asset_root), str(args.output), row, demo) for row in tasks for demo in range(16)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(episode, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            if len(rows) % 16 == 0:
                print(json.dumps(dict(completed=len(rows), total=len(jobs), seconds=time.monotonic() - started)), flush=True)
    rows.sort(key=lambda row: (row["task"], row["demo"]))
    write_json(args.output / "episodes.json", rows)
    summary = dict(schema=SCHEMA, episodes=len(rows), frames=sum(row["frames"] for row in rows),
                   supported_frames=sum(row["supported_frames"] for row in rows),
                   object_frames=sum(row["object_frames"] for row in rows),
                   motion_frames=sum(row["motion_frames"] for row in rows),
                   motion_mass=sum(row["motion_mass"] for row in rows))
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "completion.json", dict(status="complete", seconds=time.monotonic() - started, **summary))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
