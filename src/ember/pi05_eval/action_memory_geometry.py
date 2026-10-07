"""Original absolute retrieval geometry plus its live own-state signature.

Geometry and nearest retain Git 9f90a14d's definitions. Asset paths also accept
the original support XML's repeated package prefix; live reads never rewind.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

def geometry_episode(demo, objects, asset_root: Path, robosuite_root: Path):
    xml = ET.fromstring(demo.attrs["model_file"])
    for element in xml.findall("./asset/*"):
        name = element.get("file")
        if name is None:
            continue
        if "/robosuite/" in name:
            path = robosuite_root / name.split("/robosuite/")[-1].removeprefix("robosuite/")
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
        # Stored EEF sensors use mj_step's last pre-integration kinematics.
        # Reconstruct that substep, rather than changing the high-level action offset.
        mujoco.mj_integratePos(model, data.qpos, data.qvel, -model.opt.timestep)
        data.time -= model.opt.timestep
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
            "signature": signature, "max_coordinate_error_m": max(coordinate_errors),
            "kinematic_backstep_seconds": float(model.opt.timestep)}


def nearest(query, teacher):
    # Direct squared distance avoids cancellation for near-identical coordinates.
    distance = np.square(query[:, None, :] - teacher[None, :, :]).sum(axis=-1)
    indices = distance.argmin(axis=1)
    return indices, distance[np.arange(len(indices)), indices]


def live_query(env, obs, signature):
    """Read the same point signature now; never rewind or gate cached sensors."""
    from ember.pi05_processing import quat2axisangle

    owner = getattr(env, "env", env)
    model, data = owner.sim.model, owner.sim.data
    signature = tuple(tuple(value) for value in signature)
    cached = getattr(env, "_ember_action_memory_registry", None)
    if cached is None or cached[0] != signature:
        objects = tuple(dict.fromkeys(row[0] for row in signature))
        actual, indices = [], []
        for obj in objects:
            group = []
            for kind, count in (("body", model.nbody), ("site", model.nsite)):
                for index in range(count):
                    name = getattr(model, kind + "_id2name")(index)
                    if name == obj or (name is not None and name.startswith(obj + "_")):
                        group.append((kind, name, index))
            for kind, name, index in sorted(group):
                actual.append((obj, kind, name))
                indices.append((kind, index))
        if tuple(actual) != signature:
            raise ValueError("Live/teacher complete object body/site signature differs")
        weights = np.sqrt([1 / (len(objects) * sum(row[0] == obj for row in signature))
                           for obj, _, _ in signature])
        grip = [i for i in range(model.nsite)
                if (model.site_id2name(i) or "").endswith("grip_site")]
        if len(grip) != 1:
            raise ValueError("Live signature requires one original Panda grip site")
        cached = (signature, indices, weights, grip[0])
        env._ember_action_memory_registry = cached
    _, indices, weights, grip = cached
    p = np.asarray([getattr(data, "body_xpos" if kind == "body" else "site_xpos")[i]
                    for kind, i in indices], dtype=np.float64)
    r = np.asarray([getattr(data, "body_xmat" if kind == "body" else "site_xmat")[i]
                    for kind, i in indices], dtype=np.float64)
    ee = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
    rotation = Rotation.from_rotvec(quat2axisangle(obs["robot0_eef_quat"])).as_matrix()
    common = np.concatenate([rotation.reshape(-1) / np.sqrt(2),
                             (r * weights[:, None] / np.sqrt(2)).reshape(-1),
                             gripper / (0.04 * np.sqrt(2))])
    absolute = np.concatenate([common, ee / .10, (p * weights[:, None] / .10).reshape(-1)])
    if not np.isfinite(absolute).all():
        raise ValueError("Nonfinite live retrieval signature")
    difference = np.asarray(data.site_xpos[grip]) - ee
    return absolute, {"eef_cache_minus_sim_m": (-difference).tolist(),
                      "eef_cache_sim_norm_m": float(np.linalg.norm(difference)),
                      "live_rewind_steps": 0, "offline_sync_gate_applied_live": False}
