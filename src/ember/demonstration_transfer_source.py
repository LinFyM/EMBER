"""Saved-source kinematics and uniform On/In movement boundaries.

This is the construction side of the single demonstration-transfer data owner.
It reads authorized train source state/action/XML; it never steps a source env.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCHEMA = "ember_demonstration_transfer_training_support_v1"
CANONICAL_ASSET_REPO = Path("/data1/user/ymdai/projects/EMBER")
DATA = CANONICAL_ASSET_REPO / "data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a"
ASSETS = CANONICAL_ASSET_REPO / "data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
PRODUCER_ROOTS = ("/home/yifengz/workspace/", "/Users/yifengz/workspace/")


class SourceStructureUnsupported(ValueError):
    """The sealed source cannot satisfy the one uniform construction."""


def source_xml(xml: str) -> str:
    import robosuite

    installed = str(Path(robosuite.__file__).resolve().parent / "models/assets")
    value = xml
    for prefix in PRODUCER_ROOTS:
        value = value.replace(prefix + "robosuite-master/robosuite/models/assets", installed)
        value = value.replace(prefix + "libero-dev/chiliocosm/assets", str(ASSETS))
    if value == xml or any(prefix in value for prefix in PRODUCER_ROOTS):
        raise ValueError("source XML asset roots changed")
    return value


def pose(sim: Any, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray(sim.data.body_xpos[body_id], dtype=np.float64).copy(),
            np.asarray(sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3).copy())


def body_id(owner: Any, name: str) -> int:
    if name in owner.obj_body_id:
        return int(owner.obj_body_id[name])
    # A region may specify an exact arena body instead of a registered object.
    try:
        result = int(owner.sim.model.body_name2id(name))
        if result < 0:
            raise ValueError(name)
        return result
    except (ValueError, KeyError) as exc:
        raise SourceStructureUnsupported(f"exact target root absent: {name}") from exc


def destination_root(owner: Any, target: str) -> str:
    if target in owner.obj_body_id:
        return target
    region = owner.parsed_problem["regions"].get(target)
    root = str(region["target"]) if region is not None else target
    body_id(owner, root)
    return root


def roles(owner: Any, reference_roots: tuple[str, ...] = ()) -> tuple[list[str], list[list[str]]]:
    names = sorted(set(owner.obj_body_id) | set(reference_roots))
    for name in names:
        body_id(owner, name)
    goals = [[str(x).lower() for x in row] for row in owner.parsed_problem["goal_state"]]
    return names, goals


def movement_goals(owner: Any, goals: list[list[str]], initial_truth: np.ndarray
                   ) -> list[dict[str, Any]]:
    """Resolve only false initial On/In goals; all other goals are guards."""
    movements: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, (goal, true) in enumerate(zip(goals, initial_truth, strict=True)):
        if true:
            continue
        if len(goal) != 3 or goal[0] not in {"on", "in"}:
            raise SourceStructureUnsupported(f"unsatisfied non-movement goal: {goal}")
        obj, target = goal[1:]
        if obj not in owner.objects_dict or obj in seen:
            raise SourceStructureUnsupported(f"nonunique or nonmovable goal object: {goal}")
        seen.add(obj)
        movements.append({"object": obj, "target": target,
                          "reference_root": destination_root(owner, target),
                          "goal_index": index})
    if not movements:
        raise SourceStructureUnsupported("no initially unsatisfied On/In movement goal")
    return movements


def segment_boundaries(movements: list[dict[str, Any]], names: list[str],
                       positions: np.ndarray, predicates: np.ndarray,
                       actions: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One lift and nonfinal release per object, sorted within this demonstration."""
    n = len(actions)
    ordered = []
    for movement in movements:
        obj = movement["object"]
        z = positions[:, names.index(obj), 2]
        lift = next((i for i in range(1, n) if z[i] - z[0] >= 0.03), None)
        if lift is None:
            raise SourceStructureUnsupported(f"missing first 0.03m lift: {obj}")
        ordered.append({**movement, "lift": lift})
    ordered.sort(key=lambda item: item["lift"])
    if len({item["lift"] for item in ordered}) != len(ordered):
        raise SourceStructureUnsupported("simultaneous first lifts")
    segments: list[dict[str, Any]] = []
    begin = 0
    for index, item in enumerate(ordered):
        lift = item["lift"]
        if not begin < lift < n:
            raise SourceStructureUnsupported(f"interleaved or empty approach: {item['object']}")
        if index == len(ordered) - 1:
            end = n
            item["release"] = None
        else:
            goal_index = item["goal_index"]
            release = next((i for i in range(lift, n)
                            if predicates[i, goal_index] and actions[i, 6] < 0), None)
            if release is None or release >= ordered[index + 1]["lift"]:
                raise SourceStructureUnsupported(f"missing or interleaved release: {item['object']}")
            item["release"] = release
            end = release + 1
        if end <= lift:
            raise SourceStructureUnsupported(f"empty transport: {item['object']}")
        segments.extend(({"start": begin, "stop": lift, "reference_body": item["object"]},
                         {"start": lift, "stop": end, "reference_body": item["reference_root"]}))
        begin = end
    if begin != n:
        raise SourceStructureUnsupported("source segments do not cover all saved actions")
    return ordered, segments


def extract_one(row: dict[str, Any], bddl: Path, demo: int
                ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Restore all saved source states and OSC goals without env.step."""
    from libero.libero.envs.env_wrapper import ControlEnv

    task = int(row["global_task_id"])
    source = DATA / row["hdf5"]["relative_path"]
    if source.stat().st_size != row["hdf5"]["bytes"] or not bddl.is_file():
        raise ValueError(f"source asset identity or BDDL missing: task{task}")
    with h5py.File(source) as handle:
        group = handle[f"data/demo_{demo}"]
        actions = np.asarray(group["actions"], dtype=np.float64)
        states = np.asarray(group["states"], dtype=np.float64)
        xml = source_xml(str(group.attrs["model_file"]))
    if actions.ndim != 2 or actions.shape[1] != 7 or states.shape[0] != len(actions):
        raise ValueError(f"source action/state clock changed: task{task}/demo{demo}")
    env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_offscreen_renderer=False)
    try:
        env.reset()
        try:
            env.reset_from_xml_string(xml)
        except ValueError as exc:
            # A sealed source XML may describe a different object registry
            # from the official BDDL. Keep that source unavailable; do not
            # rename objects or modify the installed simulator to make it fit.
            reason = str(exc).split(" Available ", 1)[0][:180]
            raise SourceStructureUnsupported(f"source XML/BDDL registry mismatch: {reason}") from exc
        owner = env.env
        _, goals = roles(owner)
        controller = owner.robots[0].controller
        if (controller.eef_name != "gripper0_grip_site" or not controller.use_delta
                or controller.impedance_mode != "fixed"
                or not np.allclose(controller.output_max, [.05, .05, .05, .5, .5, .5])):
            raise ValueError("installed OSC_POSE interface changed")
        env.set_state(states[0])
        env.sim.forward()
        initial_truth = np.asarray([owner._eval_predicate(goal) for goal in goals], dtype=np.bool_)
        movements = movement_goals(owner, goals, initial_truth)
        names, _ = roles(owner, tuple(item["reference_root"] for item in movements))
        positions = np.empty((len(actions), len(names), 3), dtype=np.float64)
        rotations = np.empty((len(actions), len(names), 3, 3), dtype=np.float64)
        goal_pos = np.empty((len(actions), 3), dtype=np.float64)
        goal_rot = np.empty((len(actions), 3, 3), dtype=np.float64)
        predicates = np.empty((len(actions), len(goals)), dtype=np.bool_)
        for step, (state, action) in enumerate(zip(states, actions, strict=True)):
            env.set_state(state)
            env.sim.forward()
            controller.update(force=True)
            for j, name in enumerate(names):
                positions[step, j], rotations[step, j] = pose(owner.sim, body_id(owner, name))
            predicates[step] = [bool(owner._eval_predicate(goal)) for goal in goals]
            controller.set_goal(action[:6])
            goal_pos[step] = controller.goal_pos
            goal_rot[step] = controller.goal_ori
        ordered, segments = segment_boundaries(movements, names, positions, predicates, actions)
        metadata = {
            "schema_version": SCHEMA, "task": task, "demo": demo, "hdf5": str(source),
            "bddl": str(bddl), "steps": len(actions), "body_names": names, "goals": goals,
            "movement_goals": ordered, "source_boundaries": ordered,
            "reference_roots": sorted(set(item["reference_root"] for item in movements)),
            "segments": segments,
            "source_initial_object_positions": {item["object"]: positions[0, names.index(item["object"])].tolist()
                                                for item in movements},
            "source_xml": "demo model_file with only installed asset-root remap",
            "source_state_clock": "states[k] pre-action[k]; obs[k] post-action[k]",
            "source_steps_executed": 0,
        }
        arrays = {"actions": actions, "goal_pos": goal_pos, "goal_rot": goal_rot,
                  "body_pos": positions, "body_rot": rotations, "predicates": predicates}
        return metadata, arrays
    finally:
        env.close()
