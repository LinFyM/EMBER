"""Temporary task23 original-command CPU consumer; retire after the fixed batch."""

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
from ember.pi05_eval.trajectory_capture import record_passive_step, start_passive_trace

LAYERS = ("top", "middle", "bottom")
GROUPS = ("robot_fixed_cabinet", "robot_top", "robot_middle", "robot_bottom")
BOWL = "akita_black_bowl_1"
CABINET = "wooden_cabinet_1"
GOAL = (("in", BOWL, f"{CABINET}_top_region"),)


def descendants(model, root):
    result = {int(root)}
    for body in range(int(root) + 1, model.nbody):
        if int(model.body_parentid[body]) in result:
            result.add(body)
    return result


def geometry_registry(env):
    """Resolve each layer from the same site/joint registry used by native Open."""
    owner, model = env.env, env.sim.model
    fixture = int(owner.obj_body_id[CABINET])
    cabinet_bodies = descendants(model, fixture)
    layers, body_groups = [], {}
    for layer in LAYERS:
        site_name = f"{CABINET}_{layer}_region"
        site = owner.object_sites_dict[site_name]
        if site.parent_name != CABINET or len(site.joints) != 1:
            raise ValueError("native cabinet site must resolve one original drawer joint")
        joint_name = site.joints[0]
        joint = int(model.joint_name2id(joint_name))
        site_id = int(model.site_name2id(site_name))
        body = int(model.jnt_bodyid[joint])
        if model.jnt_type[joint] != 2 or body != int(model.site_bodyid[site_id]) or body not in cabinet_bodies:
            raise ValueError("actual region/joint does not map to one cabinet slide body")
        body_groups[layer] = descendants(model, body)
        layers.append(dict(layer=layer, site_name=site_name, site_id=site_id,
                           body_id=body, body_name=model.body_id2name(body),
                           joint_id=joint, joint_name=joint_name,
                           qpos_address=int(model.jnt_qposadr[joint]),
                           qvel_address=int(model.jnt_dofadr[joint]),
                           joint_axis_local=model.jnt_axis[joint].tolist(),
                           joint_range_m=model.jnt_range[joint].tolist(),
                           descendant_body_ids=sorted(body_groups[layer])))
    moving = set().union(*body_groups.values())
    if sum(map(len, body_groups.values())) != len(moving):
        raise ValueError("drawer subtrees overlap")
    body_groups["fixed_cabinet"] = cabinet_bodies - moving
    robot_root = model.body_name2id(owner.robots[0].robot_model.root_body)
    body_groups["robot"] = descendants(model, robot_root)
    geom_groups = {name: {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies}
                   for name, bodies in body_groups.items()}
    if not all(geom_groups.values()):
        raise ValueError("required actual cabinet/robot body has no geometry")
    pair_group = {}
    for group, entity in enumerate(("fixed_cabinet", *LAYERS)):
        for robot in geom_groups["robot"]:
            for cabinet in geom_groups[entity]:
                pair_group[tuple(sorted((robot, cabinet)))] = group
    geoms = [dict(id=g, name=model.geom_id2name(g), body_id=int(model.geom_bodyid[g]),
                  body_name=model.body_id2name(int(model.geom_bodyid[g])),
                  contype=int(model.geom_contype[g]), conaffinity=int(model.geom_conaffinity[g]))
             for g in sorted(set().union(*geom_groups.values()))]
    open_range = owner.get_object(CABINET).object_properties["articulation"]["default_open_ranges"]
    if open_range != [-.16, -.14] or tuple(map(tuple, owner.parsed_problem["goal_state"])) != GOAL:
        raise ValueError("native Open range or official In-only goal changed")
    return dict(layers=layers, entities={k: sorted(v) for k, v in geom_groups.items()},
                body_groups={k: sorted(v) for k, v in body_groups.items()}, geoms=geoms,
                native_Open=dict(range_m=open_range, comparison="qpos < max(range)"),
                official_goal=[list(g) for g in GOAL]), pair_group


def replay_deviation(arrays, original):
    body_error = np.linalg.norm(arrays["body_positions"] - original["body_positions"], axis=-1)
    eef_error = np.linalg.norm(arrays["eef_pos"] - original["eef_pos"], axis=-1)
    names = arrays["body_names"].tolist()
    bowl_max = float(body_error[:, names.index(BOWL)].max() * 100)
    eef_max = float(eef_error.max() * 100)
    native = bool(np.array_equal(arrays["predicates"], original["predicates"]))
    return dict(body_max_position_error_cm=dict(zip(names, (body_error.max(axis=0) * 100).tolist())),
                body_initial_position_error_cm=dict(zip(names, (body_error[0] * 100).tolist())),
                bowl_max_position_error_cm=bowl_max, eef_max_position_error_cm=eef_max,
                eef_initial_position_error_cm=float(eef_error[0] * 100),
                gripper_max_qpos_error_m=float(np.abs(arrays["gripper_qpos"] - original["gripper_qpos"]).max()),
                predicate_mismatch_samples=np.sum(arrays["predicates"] != original["predicates"], axis=0).tolist(),
                predicates_identical=native, interpretable_parent=bool(bowl_max <= 1 and eef_max <= 1 and native))


def first(values):
    indices = np.flatnonzero(values)
    return int(indices[0]) if len(indices) else None


def movement_summary(arrays):
    bowl = arrays["body_positions"][:, arrays["body_names"].tolist().index(BOWL)]
    distance = np.linalg.norm(bowl - bowl[0], axis=1)
    height = bowl[:, 2] - bowl[0, 2]
    eef = arrays["eef_pos"]
    return dict(first_bowl_displacement_1cm_step=first(distance >= .01),
                first_bowl_height_3cm_step=first(height >= .03),
                max_bowl_displacement_cm=float(distance.max() * 100),
                max_bowl_height_cm=float(height.max() * 100),
                final_bowl_displacement_cm=float(distance[-1] * 100),
                first_In_step=first(arrays["predicates"][:, 0]),
                In_ever=bool(arrays["predicates"][:, 0].any()),
                In_final=bool(arrays["predicates"][-1, 0]),
                eef_path_length_m=float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum()),
                eef_min_xyz_m=eef.min(axis=0).tolist(), eef_max_xyz_m=eef.max(axis=0).tolist())


def drawer_summary(arrays, timestep):
    result = {}
    for index, layer in enumerate(LAYERS):
        qpos, opened = arrays["drawer_qpos"][:, index], arrays["drawer_Open"][:, index]
        result[layer] = dict(initial_qpos_m=float(qpos[0]), final_qpos_m=float(qpos[-1]),
                             min_qpos_m=float(qpos.min()), max_qpos_m=float(qpos.max()),
                             max_qpos_change_cm=float(np.max(np.abs(qpos - qpos[0])) * 100),
                             min_qpos_step=int(qpos.argmin()), first_Open_step=first(opened),
                             final_Open=bool(opened[-1]), Open_step_end_samples=int(opened[1:].sum()),
                             Open_step_end_duration_seconds=float(opened[1:].sum() * timestep),
                             initial_Open=bool(opened[0]),
                             max_abs_qvel_m_per_s=float(np.abs(arrays["drawer_qvel"][:, index]).max()))
    return result


def contact_summary(arrays):
    result = {}
    contacts, forces = arrays["contact_samples"], arrays["contact_forces"]
    for index, name in enumerate(GROUPS):
        mask = contacts[:, 1] == index
        selected, selected_force = contacts[mask], forces[mask]
        steps = np.unique(selected[:, 0]).astype(int)
        result[name] = dict(contact_samples=len(selected), control_samples=len(steps),
                            first_step=int(steps[0]) if len(steps) else None,
                            last_step=int(steps[-1]) if len(steps) else None,
                            geom_pairs=np.unique(selected[:, 2:4].astype(int), axis=0).tolist(),
                            max_normal_force_N=float(selected_force[:, 0].max()) if len(steps) else None)
    return result


class DrawerCapture:
    """Passive control-step-end reads; no state writes or physics intervention."""

    def __init__(self, env, registry, pair_group):
        self.env, self.registry, self.pair_group = env, registry, pair_group
        self.qpos, self.qvel, self.opened, self.times = [], [], [], []
        self.contacts, self.forces = [], []

    def sample(self, step):
        import mujoco
        sim, owner = self.env.sim, self.env.env
        rows = self.registry["layers"]
        qpos = np.asarray([sim.data.qpos[r["qpos_address"]] for r in rows])
        self.qpos.append(qpos)
        self.qvel.append(np.asarray([sim.data.qvel[r["qvel_address"]] for r in rows]))
        opened = np.asarray([owner._eval_predicate(["open", r["site_name"]]) for r in rows], dtype=bool)
        if not np.array_equal(opened, qpos < max(self.registry["native_Open"]["range_m"])):
            raise ValueError("recorded Open differs from actual native joint predicate")
        self.opened.append(opened)
        self.times.append(float(sim.data.time))
        if step == 0:
            return
        force = np.empty(6, dtype=np.float64)
        for index in range(sim.data.ncon):
            contact = sim.data.contact[index]
            key = tuple(sorted((int(contact.geom1), int(contact.geom2))))
            if key not in self.pair_group:
                continue
            mujoco.mj_contactForce(sim.model._model, sim.data._data, index, force)
            if not np.isfinite(force).all():
                raise ValueError("non-finite direct contact force")
            self.contacts.append((step, self.pair_group[key], int(contact.geom1), int(contact.geom2), float(contact.dist)))
            self.forces.append(force.copy())

    def arrays(self):
        return dict(drawer_qpos=np.stack(self.qpos), drawer_qvel=np.stack(self.qvel),
                    drawer_Open=np.stack(self.opened), sim_time_seconds=np.asarray(self.times),
                    contact_samples=np.asarray(self.contacts, dtype=np.float64).reshape(-1, 5),
                    contact_forces=np.asarray(self.forces, dtype=np.float32).reshape(-1, 6))


def restore_original_start(env, snapshot, contract):
    observation = _restore_scene(env, snapshot)
    controller = env.env.robots[0].controller
    # Original snapshot explicitly includes these OSC goals; no dummy steps.
    controller.update(force=True)
    controller.goal_pos = snapshot["controller_goal_pos"].copy()
    controller.goal_ori = snapshot["controller_goal_ori"].copy()
    if controller.interpolator_pos is not None or controller.interpolator_ori is not None:
        raise ValueError("original scene did not preserve interpolation runtime")
    # Panda's command accumulator is not in flattened MuJoCo state. Reconstruct
    # its known post-dummy value from the original run's public own-command
    # contract and installed formatter, without executing any physics step.
    robot = env.env.robots[0]
    if type(robot.gripper).__name__ != "PandaGripper":
        raise ValueError("unregistered gripper command accumulator")
    substeps = int(env.env.control_timestep / env.env.model_timestep)
    for _ in range(int(contract["environment"]["dummy_settling_steps"]) * substeps):
        robot.gripper.format_action(np.asarray(contract["environment"]["dummy_action"][-1:]))
    names = sorted(env.env.obj_body_id)
    _assert_scene_pair(env, observation, names, [list(g) for g in GOAL], snapshot, image=False)
    return observation, names


def replay(root: Path, row: dict, output: Path) -> dict:
    """Execute one original command stream once, preserving every recorded sample."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU consumer requires no visible CUDA devices")
    started, cpu_started = time.monotonic(), time.process_time()
    source = row["source_row"]
    with np.load(source["continuous_control_trace"]["trace"]["path"], allow_pickle=False) as saved:
        original = {key: saved[key] for key in saved.files}
    with np.load(source["scene_reference"]["path"], allow_pickle=False) as saved:
        snapshot = {key: saved[key] for key in saved.files if key != "initial_rgb_canonical180"}
    steps = int(source["steps"])
    if original["actions"].shape != (steps, 7) or original["predicates"].shape != (steps + 1, 1):
        raise ValueError("actual original action/predicate length changed")
    contract = json.loads(Path(row["source_run_contract"]).read_text())
    configure_libero_runtime_assets(Path(contract["libero_paths"]["assets"]))
    from libero.libero.envs.env_wrapper import ControlEnv
    task = row["task"]
    bddl = Path(contract["libero_paths"]["bddl_files"]) / task["problem_folder"] / task["bddl_file"]
    env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, camera_names=[],
                     has_renderer=False, has_offscreen_renderer=False)
    try:
        env.seed(int(source["env_seed"]))
        env.reset()
        observation, names = restore_original_start(env, snapshot, contract)
        if env.env.has_renderer or env.env.has_offscreen_renderer or env.sim._render_context_offscreen is not None:
            raise ValueError("CPU consumer unexpectedly created a renderer")
        if names != original["body_names"].tolist():
            raise ValueError("original body registry differs from physical consumer")
        registry, pair_group = geometry_registry(env)
        states, predicates = stage_predicate_snapshot(env)
        capture = {"passive_trace": {"trace_root": str(output.parent)}}
        slot = dict(obs=observation, steps=0, stage_predicate_states=states, stage_predicate_last=predicates)
        start_passive_trace(env, slot, capture)
        added = DrawerCapture(env, registry, pair_group)
        added.sample(0)
        for step, action in enumerate(original["actions"], 1):
            observation, _, _, _ = env.step(action)
            _, predicates = stage_predicate_snapshot(env, states)
            slot.update(obs=observation, steps=step, stage_predicate_last=predicates)
            record_passive_step(env, slot, action, capture)
            added.sample(step)
        trace = slot["passive_trace"]
        arrays = {k: np.stack(trace[k]) for k in ("actions", "body_positions", "eef_pos", "eef_quat", "gripper_qpos", "predicates")}
        arrays.update(body_names=np.asarray(names), drawer_layer_names=np.asarray(LAYERS), **added.arrays())
        if not np.array_equal(arrays["actions"], original["actions"]):
            raise ValueError("actual commands differ from the original recorded stream")
        deviation = replay_deviation(arrays, original)
        arrays.update({f"original_{k}": original[k] for k in ("body_positions", "eef_pos", "eef_quat", "gripper_qpos", "predicates")})
        torch = sys.modules.get("torch")
        cuda_initialized = bool(torch is not None and torch.cuda.is_initialized())
        if cuda_initialized:
            raise ValueError("CPU recorded-command consumer initialized CUDA")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.with_suffix(".npz").open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        result = dict(model=row["model"], state=row["state"], arm="recorded_parent", source=row,
                      path=str(output.with_suffix(".npz")), commands=steps, samples=steps + 1,
                      original_success=bool(source["success"]), replay_success=bool(arrays["predicates"][-1, 0]),
                      movement=movement_summary(arrays), original_movement=movement_summary(original),
                      deviation=deviation, drawers=drawer_summary(arrays, env.env.control_timestep),
                      contacts=contact_summary(arrays), geometry_registry=registry,
                      environment=dict(control_freq=env.env.control_freq, model_timestep=env.env.model_timestep,
                                       control_timestep=env.env.control_timestep, controller=env.env.robots[0].controller.name),
                      sampling="post-dummy t0 and each original control step end; contacts only step ends",
                      contact_columns=["step", "group_id", "geom1_id", "geom2_id", "distance_m"],
                      force_columns="mj_contactForce contact-frame force3 N and torque3 Nm; not grasp/blocking/intent",
                      contact_groups=list(GROUPS), additional_settling_steps=0, physics_modifications=[],
                      restored_runtime=dict(OSC_goals="original scene fields", gripper_accumulator="original dummy formatter calls only, zero physics"),
                      wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu_started,
                      peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      renderer_created=False, cuda_initialized=False, model_loaded=False)
        output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        env.close()
