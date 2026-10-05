"""Temporary task39 CPU recorded-command consumer; retire when this batch closes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from ember.pi05_assets import configure_libero_runtime_assets
from ember.pi05_eval.episode import stage_predicate_snapshot
from ember.pi05_eval.scene import _assert_scene_pair, _restore_scene
from ember.pi05_eval.trajectory_capture import (
    record_passive_step, start_passive_trace,
)

GROUPS = ("mug_door", "robot_door", "mug_robot", "mug_fixed_microwave",
          "mug_table", "robot_fixed_microwave")


def descendants(model, root):
    result = {int(root)}
    for body in range(int(root) + 1, model.nbody):
        if int(model.body_parentid[body]) in result:
            result.add(body)
    return result


def geometry_registry(env):
    owner, model = env.env, env.sim.model
    fixture = int(owner.obj_body_id["microwave_1"])
    children = [i for i in range(model.nbody)
                if int(model.body_parentid[i]) == fixture and model.body_jntnum[i]]
    if len(children) != 1:
        raise ValueError("microwave must have one articulated direct door child")
    door = children[0]
    joint = int(model.body_jntadr[door])
    if model.body_jntnum[door] != 1 or model.jnt_type[joint] != 3:
        raise ValueError("microwave child must have its original hinge")
    body_groups = {
        "door": descendants(model, door),
        "fixed_microwave": descendants(model, fixture) - descendants(model, door),
        "mug": descendants(model, int(owner.obj_body_id["white_yellow_mug_1"])),
        "robot": descendants(model, model.body_name2id(owner.robots[0].robot_model.root_body)),
    }
    geom_groups = {name: {i for i in range(model.ngeom)
                         if int(model.geom_bodyid[i]) in bodies}
                   for name, bodies in body_groups.items()}
    geom_groups["table"] = {i for i in range(model.ngeom)
                            if "table" in (model.geom_id2name(i) or "")
                            and (model.geom_contype[i] or model.geom_conaffinity[i])}
    if not all(geom_groups.values()):
        raise ValueError("a required actual contact entity has no geometry")
    if any(int(model.pair_geom1[i]) in geom_groups["door"] or
           int(model.pair_geom2[i]) in geom_groups["door"] for i in range(model.npair)):
        raise ValueError("explicit door contact pairs cannot be suppressed by bit masks")
    pairs = (("mug", "door"), ("robot", "door"), ("mug", "robot"),
             ("mug", "fixed_microwave"), ("mug", "table"), ("robot", "fixed_microwave"))
    pair_group = {}
    for group, (left, right) in enumerate(pairs):
        for first in geom_groups[left]:
            for second in geom_groups[right]:
                key = tuple(sorted((first, second)))
                if key in pair_group:
                    raise ValueError("contact grouping overlaps")
                pair_group[key] = group
    records = [dict(id=i, name=model.geom_id2name(i),
                    body_id=int(model.geom_bodyid[i]),
                    body_name=model.body_id2name(int(model.geom_bodyid[i])),
                    contype=int(model.geom_contype[i]), conaffinity=int(model.geom_conaffinity[i]))
               for i in sorted(set().union(*geom_groups.values()))]
    return dict(door_body_id=door, door_body_name=model.body_id2name(door),
                door_joint_id=joint, door_joint_name=model.joint_id2name(joint),
                door_qpos_address=int(model.jnt_qposadr[joint]),
                entities={name: sorted(value) for name, value in geom_groups.items()},
                geoms=records), pair_group


def disable_door_contacts(env, registry):
    """Change only contact masks of the identified articulated subtree."""
    model = env.sim.model
    door = registry["entities"]["door"]
    collision = [g for g in door if model.geom_contype[g] or model.geom_conaffinity[g]]
    if not collision:
        raise ValueError("door has no original collision geometry")
    original_masks = np.stack((model.geom_contype.copy(), model.geom_conaffinity.copy()))
    # Narrow invariants directly protect the sole intervention, not tensor identity.
    unchanged = {name: getattr(model, name).copy() for name in (
        "body_mass", "body_inertia", "jnt_range", "jnt_axis", "dof_damping",
        "geom_pos", "geom_quat", "geom_size")}
    model.geom_contype[door] = 0
    model.geom_conaffinity[door] = 0
    others = np.ones(model.ngeom, dtype=bool)
    others[door] = False
    if not np.array_equal(original_masks[:, others],
                          np.stack((model.geom_contype[others], model.geom_conaffinity[others]))):
        raise ValueError("non-door collision mask changed")
    if any(not np.array_equal(value, getattr(model, key)) for key, value in unchanged.items()):
        raise ValueError("door collision intervention changed physical geometry or dynamics")
    return [dict(geom_id=g, before=original_masks[:, g].tolist(), after=[0, 0])
            for g in collision]


def native_box(env):
    owner = env.env
    name = "microwave_1_heating_region"
    site = owner.object_sites_dict[name]
    position = env.sim.data.get_site_xpos(name).copy()
    rotation = env.sim.data.get_site_xmat(name).copy()
    half = np.abs(rotation @ site.size)
    lower, upper = position - half, position + half
    lower[2] -= .01  # exact installed SiteObject.in_box
    return dict(site=name, center_m=position.tolist(), half_size_m=half.tolist(),
                lower_m=lower.tolist(), upper_m=upper.tolist(), lower_z_extension_m=.01)


def movement_summary(arrays, box):
    mug = arrays["body_positions"][:, arrays["body_names"].tolist().index("white_yellow_mug_1")]
    lower, upper = np.asarray(box["lower_m"]), np.asarray(box["upper_m"])
    deficit = np.maximum(np.maximum(lower - mug, mug - upper), 0.)
    distance = np.linalg.norm(deficit, axis=1)
    nearest = int(distance.argmin())
    inside = np.all(mug > lower, axis=1) & np.all(mug < upper, axis=1)
    if not np.array_equal(inside, arrays["predicates"][:, 0]):
        raise ValueError("native In differs from actual native region point test")
    predicates = arrays["predicates"]
    first = lambda values: int(np.flatnonzero(values)[0]) if np.any(values) else None
    return dict(In_ever=bool(predicates[:, 0].any()), Close_ever=bool(predicates[:, 1].any()),
                first_In_step=first(predicates[:, 0]), nearest_step=nearest,
                min_box_distance_cm=float(distance[nearest] * 100),
                final_box_distance_cm=float(distance[-1] * 100),
                nearest_axis_deficit_cm=(deficit[nearest] * 100).tolist(),
                first_within_2cm_step=first(distance <= .02),
                first_within_5cm_step=first(distance <= .05))


def replay_deviation(arrays, original):
    bodies = np.linalg.norm(arrays["body_positions"] - original["body_positions"], axis=-1)
    eef = np.linalg.norm(arrays["eef_pos"] - original["eef_pos"], axis=-1)
    names = arrays["body_names"].tolist()
    mug_max = float(bodies[:, names.index("white_yellow_mug_1")].max() * 100)
    eef_max = float(eef.max() * 100)
    native = bool(np.array_equal(arrays["predicates"], original["predicates"]))
    return dict(body_max_position_error_cm=dict(zip(names, (bodies.max(axis=0) * 100).tolist())),
                body_initial_position_error_cm=dict(zip(names, (bodies[0] * 100).tolist())),
                mug_max_position_error_cm=mug_max, eef_max_position_error_cm=eef_max,
                eef_initial_position_error_cm=float(eef[0] * 100),
                gripper_max_qpos_error_m=float(np.abs(arrays["gripper_qpos"] - original["gripper_qpos"]).max()),
                predicate_mismatch_samples=np.sum(arrays["predicates"] != original["predicates"], axis=0).tolist(),
                predicates_identical=native,
                interpretable_parent=bool(mug_max <= 1 and eef_max <= 1 and native))


def replay(root: Path, row: dict, arm: str, output: Path) -> dict:
    """One full original command stream, without policy, rendering or early success stop."""
    if arm not in ("recorded_parent", "door_noncolliding") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("invalid fixed CPU replay arm or CUDA visibility")
    started, cpu_started = time.monotonic(), time.process_time()
    source = row["source_row"]
    with np.load(source["continuous_control_trace"]["trace"]["path"], allow_pickle=False) as saved:
        original = {key: saved[key] for key in saved.files}
    with np.load(source["scene_reference"]["path"], allow_pickle=False) as saved:
        snapshot = {key: saved[key] for key in saved.files if key != "initial_rgb_canonical180"}
    if original["actions"].shape != (520, 7) or original["predicates"].shape != (521, 2):
        raise ValueError("fixed original action and predicate layout changed")
    contract = json.loads(Path(row["source_run_contract"]).read_text())
    configure_libero_runtime_assets(Path(contract["libero_paths"]["assets"]))
    from libero.libero import benchmark
    from libero.libero.envs.env_wrapper import ControlEnv
    import mujoco

    task = row["task"]
    bddl = Path(contract["libero_paths"]["bddl_files"]) / task["problem_folder"] / task["bddl_file"]
    suite = benchmark.get_benchmark_dict()[task["suite"]]()
    init_states = suite.get_task_init_states(int(task["task_id"]))
    env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False,
                     has_renderer=False, has_offscreen_renderer=False,
                     camera_heights=256, camera_widths=256)
    contact_rows, force_rows, door_angles, body_quaternions = [], [], [], []
    force = np.empty(6, dtype=np.float64)
    try:
        env.seed(int(source["env_seed"]))
        env.reset()
        env.set_init_state(init_states[row["state"]])
        for _ in range(contract["environment"]["dummy_settling_steps"]):
            env.step(np.asarray(contract["environment"]["dummy_action"], dtype=np.float32))
        observation = _restore_scene(env, snapshot)
        names = sorted(env.env.obj_body_id)
        goals = [list(goal) for goal in env.env.parsed_problem["goal_state"]]
        _assert_scene_pair(env, observation, names, goals, snapshot, image=False)
        if env.env.has_renderer or env.env.has_offscreen_renderer or env.sim._render_context_offscreen is not None:
            raise ValueError("CPU consumer unexpectedly created a renderer")
        registry, pair_group = geometry_registry(env)
        modifications = disable_door_contacts(env, registry) if arm == "door_noncolliding" else []
        box = native_box(env)
        states, predicates = stage_predicate_snapshot(env)
        if names != original["body_names"].tolist():
            raise ValueError("own trace body registry differs from actual replay registry")
        capture = {"passive_trace": {"trace_root": str(output.parent)}}
        slot = {"obs": observation, "steps": 0, "stage_predicate_states": states,
                "stage_predicate_last": predicates}
        start_passive_trace(env, slot, capture)
        def sample_contacts(step):
            door_angles.append(float(env.sim.data.qpos[registry["door_qpos_address"]]))
            body_quaternions.append(np.stack([env.sim.data.body_xquat[env.env.obj_body_id[name]] for name in names]))
            if step == 0:
                return
            for index in range(env.sim.data.ncon):
                contact = env.sim.data.contact[index]
                key = tuple(sorted((int(contact.geom1), int(contact.geom2))))
                if key not in pair_group:
                    continue
                mujoco.mj_contactForce(env.sim.model._model, env.sim.data._data, index, force)
                if not np.isfinite(force).all():
                    raise ValueError("non-finite direct contact force")
                contact_rows.append((step, pair_group[key], int(contact.geom1), int(contact.geom2), float(contact.dist)))
                force_rows.append(force.copy())
        sample_contacts(0)
        for step, action in enumerate(original["actions"], 1):
            observation, _, _, _ = env.step(action)
            _, predicates = stage_predicate_snapshot(env, states)
            slot.update(obs=observation, steps=step, stage_predicate_last=predicates)
            record_passive_step(env, slot, action, capture)
            sample_contacts(step)
        trace = slot["passive_trace"]
        arrays = {key: np.stack(trace[key]) for key in (
            "actions", "body_positions", "eef_pos", "eef_quat", "gripper_qpos", "predicates")}
        arrays.update(body_names=np.asarray(names), door_angle=np.asarray(door_angles),
                      body_quaternions=np.asarray(body_quaternions, dtype=np.float32),
                      contact_samples=np.asarray(contact_rows, dtype=np.float64).reshape(-1, 5),
                      contact_forces=np.asarray(force_rows, dtype=np.float32).reshape(-1, 6))
        if not np.array_equal(arrays["actions"], original["actions"]):
            raise ValueError("actual physical commands differ from original")
        movement, deviation = movement_summary(arrays, box), replay_deviation(arrays, original)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.with_suffix(".npz").open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        counts = {}
        contacts = arrays["contact_samples"]
        for index, name in enumerate(GROUPS):
            selected = contacts[contacts[:, 1] == index]
            selected_force = arrays["contact_forces"][contacts[:, 1] == index]
            steps = np.unique(selected[:, 0]).astype(int)
            counts[name] = dict(contact_samples=len(selected), control_samples=len(steps),
                                first_step=int(steps[0]) if len(steps) else None,
                                last_step=int(steps[-1]) if len(steps) else None,
                                geom_pairs=np.unique(selected[:, 2:4].astype(int), axis=0).tolist(),
                                max_normal_force_N=float(selected_force[:, 0].max()) if len(steps) else None)
        torch = sys.modules.get("torch")
        cuda_initialized = bool(torch is not None and torch.cuda.is_initialized())
        if cuda_initialized:
            raise ValueError("CPU recorded-command consumer initialized CUDA")
        result = dict(model=row["model"], state=row["state"], arm=arm, source=row,
                      path=str(output.with_suffix(".npz")), movement=movement, deviation=deviation,
                      original_movement=movement_summary(original, box), native_box=box,
                      contacts=counts, geometry_registry=registry, modifications=modifications,
                      door_angle_initial=float(arrays["door_angle"][0]), door_angle_final=float(arrays["door_angle"][-1]),
                      environment=dict(control_freq=env.env.control_freq, model_timestep=env.env.model_timestep,
                                       control_timestep=env.env.control_timestep, controller=env.env.robots[0].controller.name),
                      sampling="t0 and each original control step end; contacts only step end, not all integrator substeps",
                      contact_columns=["step", "group_id", "geom1_id", "geom2_id", "distance_m"],
                      force_columns="MuJoCo mj_contactForce contact-frame force3 N and torque3 Nm; not grasp or intent",
                      contact_groups=list(GROUPS), wall_seconds=time.monotonic() - started,
                      cpu_seconds=time.process_time() - cpu_started,
                      peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      renderer_created=False, cuda_initialized=cuda_initialized, commands=520, samples=521)
        output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        env.close()
