"""Replay actual saved environment actions before a frozen policy takes over."""

from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.trajectory_capture import record_replan
from ember.pi05_eval_contract import policy_noise_seed
from ember.pi05_processing import libero_policy_input


TRACE_SCHEMA = "ember_frozen_prefix_object_trace_v1"
ROW_SCHEMA = "ember_frozen_prefix_episode_v1"
OBJECT_ROLES = {
    ("libero_object", 4): {
        "target": "ketchup_1", "receptacle": "basket_1",
        "distractors": ("bbq_sauce_1", "salad_dressing_1", "alphabet_soup_1",
                        "cream_cheese_1", "milk_1"),
        "goal": ("in", "ketchup_1", "basket_1_contain_region"),
    },
    ("libero_goal", 1): {
        "target": "akita_black_bowl_1", "receptacle": "flat_stove_1_cook_region",
        "distractors": ("plate_1", "cream_cheese_1", "wine_bottle_1"),
        "goal": ("on", "akita_black_bowl_1", "flat_stove_1_cook_region"),
    },
}


def _roles(env: Any, task: Mapping[str, Any]) -> tuple[tuple[str, ...], dict[str, Any]]:
    owner = env.env
    key = (str(task["suite"]), int(task["task_id"]))
    role = OBJECT_ROLES.get(key)
    if role is None:
        raise Pi05EvaluationError("frozen-prefix object audit has an unregistered task")
    names = tuple(sorted(owner.objects_dict))
    expected = {role["target"], *role["distractors"]}
    if key == ("libero_object", 4):
        expected.add(role["receptacle"])
    goals = tuple(tuple(str(value).lower() for value in goal)
                  for goal in owner.parsed_problem["goal_state"])
    if (set(names) != expected or goals != (role["goal"],)
            or any(name not in owner.obj_body_id for name in names)):
        raise Pi05EvaluationError(f"frozen-prefix BDDL object/goal audit changed: {key}")
    return names, {**role, "distractors": list(role["distractors"]),
                   "object_names": list(names), "bddl_goal": [list(goal) for goal in goals]}


def _sample(env: Any, obs: Mapping[str, Any], slot: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    owner = env.env
    return {
        "object_positions": np.stack([
            np.asarray(owner.sim.data.body_xpos[owner.obj_body_id[name]], dtype=np.float32).copy()
            for name in names
        ]),
        "eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32).copy(),
        "eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float32).copy(),
        "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).copy(),
        "predicates": np.asarray(slot["stage_predicate_last"], dtype=np.bool_),
    }


def _controller_state(env: Any) -> dict[str, Any]:
    owner = env.env
    robot = owner.robots[0]
    controller = robot.controller
    numeric: dict[str, Any] = {}
    for name, value in vars(controller).items():
        if name.startswith("__"):
            continue
        if isinstance(value, (int, float, bool, np.number)):
            numeric[name] = value.item() if isinstance(value, np.generic) else value
        elif isinstance(value, np.ndarray) and value.size <= 256 and value.dtype.kind in "bif":
            numeric[name] = value.tolist()
    for name in ("timestep", "cur_time", "done"):
        value = getattr(owner, name, None)
        if isinstance(value, (int, float, bool, np.number)):
            numeric[f"environment.{name}"] = value.item() if isinstance(value, np.generic) else value
    return {"controller_class": type(controller).__name__, "numeric_state": numeric,
            "sim_state_source": "OffScreenRenderEnv.get_sim_state after actual action replay"}


def _append(trace: dict[str, Any], env: Any, obs: Mapping[str, Any], slot: Mapping[str, Any],
            action: np.ndarray | None) -> None:
    sample = _sample(env, obs, slot, trace["object_names"])
    for name, value in sample.items():
        trace[name].append(value)
    if action is not None:
        trace["actions"].append(np.asarray(action, dtype=np.float32).copy())


def _state_error(actual: Any, saved: Any, *, step: int) -> dict[str, float | int]:
    observed = np.asarray(actual.detach().cpu(), dtype=np.float64).reshape(-1)
    reference = np.asarray(saved.detach().cpu(), dtype=np.float64).reshape(-1)
    if observed.shape != (8,) or reference.shape != (8,) or not np.isfinite(observed).all():
        raise Pi05EvaluationError(f"frozen-prefix saved policy state invalid at step {step}")
    difference = np.abs(observed - reference)
    errors: dict[str, float | int] = {
        "step": step,
        "position_abs_max": float(difference[:3].max()),
        "axis_angle_abs_max": float(difference[3:6].max()),
        "gripper_qpos_abs_max": float(difference[6:].max()),
    }
    if (errors["position_abs_max"] > 1e-4 or errors["axis_angle_abs_max"] > 1e-3
            or errors["gripper_qpos_abs_max"] > 1e-4):
        raise Pi05EvaluationError(f"frozen-prefix action replay state diverged: {errors}")
    return errors


def _load_saved_prefix(
    *, reference: Mapping[str, Any], task: Mapping[str, Any], state_id: int,
    cut: int, root_seed: int,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...], tuple[int, ...], int, bool]:
    import torch

    key = f"{task['suite']}:{int(task['task_id'])}:{state_id}"
    trajectory_path = Path(reference["trajectory"]["path"])
    if (not trajectory_path.is_file()
            or trajectory_path.stat().st_size != int(reference["trajectory"]["bytes"])):
        raise Pi05EvaluationError(f"frozen-prefix trajectory changed: {key}")
    saved = torch.load(trajectory_path, map_location="cpu", weights_only=True)
    chunks = tuple(saved.get("action_chunks", ()))
    actions = tuple(saved.get("executed_action_prefixes", ()))
    states = tuple(saved.get("states", ()))
    steps = tuple(int(value) for value in saved.get("replan_steps", ()))
    seeds = tuple(int(value) for value in saved.get("policy_noise_seeds", ()))
    old_steps = int(reference["steps"])
    replay_steps = min(cut, old_steps)
    count = math.ceil(replay_steps / 5)
    if (saved.get("schema_version") != "ember_pi05_diagnostic_trajectory_v2"
            or saved.get("suite") != task["suite"] or saved.get("task_id") != task["task_id"]
            or saved.get("init_state_id") != state_id
            or bool(saved.get("success")) != bool(reference["success"])
            or int(saved.get("steps", -1)) != old_steps
            or not len(actions) == len(chunks) == len(states) == len(seeds) == len(steps)
            or steps != tuple(range(0, len(chunks) * 5, 5))
            or len(chunks) != math.ceil(old_steps / 5)
            or tuple(reference["policy_noise_seeds"]) != seeds
            or count > len(actions)):
        raise Pi05EvaluationError(f"frozen-prefix saved action trajectory invalid: {key}")
    for index, seed in enumerate(seeds[:count]):
        if seed != policy_noise_seed(root_seed, task["suite"], int(task["task_id"]), state_id, index):
            raise Pi05EvaluationError(f"frozen-prefix saved policy RNG changed: {key}, index {index}")
    terminal = bool(reference["success"] and old_steps <= cut)
    return chunks, actions, states, seeds, replay_steps, terminal


def replay_prefix(
    *, env: Any, slot: dict[str, Any], task: Mapping[str, Any], contract: Mapping[str, Any],
    preprocess: Any,
) -> None:
    """Step the real environment/controller with already denormalized saved actions."""
    if contract.get("approach_channel_intervention") is not None:
        from ember.pi05_eval.approach_channel_replay import replay_channel_prefix

        replay_channel_prefix(env=env, slot=slot, task=task,
                              contract=contract, preprocess=preprocess)
        return
    from ember.pi05_eval.episode import update_stage_predicates

    intervention = contract["frozen_prefix_intervention"]
    state_id = int(slot["init_state_id"])
    key = f"{task['suite']}:{int(task['task_id'])}:{state_id}"
    reference = intervention["references"].get(key)
    if reference is None:
        raise Pi05EvaluationError(f"frozen-prefix reference missing: {key}")
    cut = int(intervention["cut_control_steps"])
    if int(contract["policy"]["replan_steps"]) != 5:
        raise Pi05EvaluationError("frozen-prefix requires canonical five-action replan")
    chunks, actions, states, saved_seeds, replay_steps, prefix_terminal = _load_saved_prefix(
        reference=reference, task=task, state_id=state_id, cut=cut,
        root_seed=int(contract["rng"]["inference_seed"]),
    )
    count = math.ceil(replay_steps / 5)
    names, roles = _roles(env, task)
    trace: dict[str, Any] = {
        "object_names": names, "object_roles": roles,
        "actions": [], "object_positions": [], "eef_pos": [], "eef_quat": [],
        "gripper_qpos": [], "predicates": [],
        "sim_state_initial": np.asarray(env.get_sim_state(), dtype=np.float64).copy(),
    }
    _append(trace, env, slot["obs"], slot, None)
    errors = []
    for index in range(count):
        raw_input = libero_policy_input(slot["obs"], task["language"])
        errors.append(_state_error(raw_input["observation.state"], states[index], step=index * 5))
        planned = np.asarray(actions[index], dtype=np.float32)
        if planned.shape != (5, 7) or not np.isfinite(planned).all():
            raise Pi05EvaluationError(f"frozen-prefix executed environment action invalid: {key}")
        processed = preprocess(raw_input) if slot.get("occupancy_capture_level") == "full" else {}
        record_replan(slot, raw_input, processed, chunks[index], planned)
        slot["policy_noise_seeds"].append(saved_seeds[index])
        slot["replan_index"] += 1
        for action in planned[:min(5, replay_steps - slot["steps"])]:
            observation, _, done, _ = env.step(action)
            slot["obs"] = observation
            slot["steps"] += 1
            update_stage_predicates(env, slot)
            _append(trace, env, observation, slot, action)
            expected_done = prefix_terminal and slot["steps"] == reference["steps"]
            if bool(done) != expected_done:
                raise Pi05EvaluationError(
                    f"frozen-prefix terminal differs from original: {key}, "
                    f"step={slot['steps']}, got={bool(done)}, expected={expected_done}"
                )
    if slot["steps"] != replay_steps:
        raise Pi05EvaluationError(f"frozen-prefix executed step count changed: {key}")
    if not prefix_terminal and replay_steps == cut and cut < reference["steps"]:
        index = cut // 5
        raw_input = libero_policy_input(slot["obs"], task["language"])
        errors.append(_state_error(raw_input["observation.state"], states[index], step=cut))
    trace["sim_state_cut"] = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
    trace["controller_state_cut"] = _controller_state(env)
    slot["prefix_trace"] = trace
    slot["prefix_state_errors"] = errors
    slot["prefix_terminal"] = prefix_terminal
    slot["prefix_replay_steps"] = replay_steps
    slot["prefix_reference"] = reference


def record_tail_step(env: Any, slot: dict[str, Any], action: Any) -> None:
    trace = slot.get("prefix_trace")
    if trace is not None:
        _append(trace, env, slot["obs"], slot, np.asarray(action, dtype=np.float32))
        if "contact_pairs" in trace:
            from ember.pi05_eval.approach_channel_replay import record_tail_contacts

            record_tail_contacts(env, slot)


def _displacements(positions: np.ndarray, names: tuple[str, ...], cut: int) -> dict[str, Any]:
    delta = positions - positions[0:1]
    horizontal = np.linalg.norm(delta[:, :, :2], axis=2)
    upward = delta[:, :, 2]
    exceeded = (horizontal >= 0.03) | (upward >= 0.02)
    result = {}
    for index, name in enumerate(names):
        first = next((step - 2 for step in range(3, len(positions))
                      if exceeded[step - 2:step + 1, index].all()), None)
        result[name] = {
            "first_persistent_displacement_step": first,
            "cut_horizontal_m": float(horizontal[cut, index]),
            "cut_upward_m": float(upward[cut, index]),
            "cut_threshold_exceeded": bool(exceeded[cut, index]),
        }
    return result


def finish_trace(slot: Mapping[str, Any], task: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("approach_channel_intervention") is not None:
        from ember.pi05_eval.approach_channel_replay import finish_channel_trace

        return finish_channel_trace(slot, task, contract)
    trace = slot["prefix_trace"]
    intervention = contract["frozen_prefix_intervention"]
    root = Path(intervention["trace_root"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / (f"{task['suite']}_task_{int(task['task_id']):02d}_"
                   f"state_{int(slot['init_state_id']):03d}_{uuid.uuid4().hex}.npz")
    arrays = {name: np.stack(trace[name]) for name in (
        "actions", "object_positions", "eef_pos", "eef_quat", "gripper_qpos", "predicates")}
    if (arrays["actions"].shape != (int(slot["steps"]), 7)
            or arrays["object_positions"].shape[0] != int(slot["steps"]) + 1):
        raise Pi05EvaluationError("frozen-prefix trace did not record every control step")
    with path.open("xb") as handle:
        np.savez_compressed(
            handle, schema_version=np.asarray(TRACE_SCHEMA),
            object_names=np.asarray(trace["object_names"]),
            sim_state_initial=trace["sim_state_initial"],
            sim_state_cut=trace["sim_state_cut"], **arrays,
        )
    cut_steps = int(slot["prefix_replay_steps"])
    return {
        "schema_version": ROW_SCHEMA,
        "anchor": intervention["anchor"], "follower": intervention["follower"],
        "cut_control_steps": intervention["cut_control_steps"],
        "actual_prefix_steps": cut_steps,
        "prefix_terminal": bool(slot["prefix_terminal"]),
        "follower_start_replan_index": int(slot["prefix_replay_steps"]) // 5,
        "reference": dict(slot["prefix_reference"]),
        "replay_state_errors": list(slot["prefix_state_errors"]),
        "object_roles": trace["object_roles"],
        "controller_state_cut": trace["controller_state_cut"],
        "object_displacement": _displacements(arrays["object_positions"],
                                               trace["object_names"], cut_steps),
        "trace": {"path": str(path), "bytes": path.stat().st_size,
                  "schema_version": TRACE_SCHEMA, "steps": int(slot["steps"])},
    }


def validate_row(row: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    if contract.get("approach_channel_intervention") is not None:
        from ember.pi05_eval.approach_channel_replay import validate_channel_row

        validate_channel_row(row, contract)
        return
    intervention = contract["frozen_prefix_intervention"]
    prefix = row.get("frozen_prefix_intervention")
    if not isinstance(prefix, Mapping) or prefix.get("schema_version") != ROW_SCHEMA:
        raise Pi05EvaluationError("frozen-prefix row evidence missing")
    key = f"{row['suite']}:{int(row['task_id'])}:{int(row['init_state_id'])}"
    reference = intervention["references"].get(key)
    cut = int(intervention["cut_control_steps"])
    terminal = bool(reference["success"] and int(reference["steps"]) <= cut) if reference else None
    trace = prefix.get("trace", {})
    path = Path(str(trace.get("path", "")))
    if (reference is None or prefix.get("reference") != reference
            or prefix.get("anchor") != intervention["anchor"]
            or prefix.get("follower") != intervention["follower"]
            or prefix.get("cut_control_steps") != cut
            or prefix.get("actual_prefix_steps") != min(cut, int(reference["steps"]))
            or prefix.get("prefix_terminal") is not terminal
            or prefix.get("follower_start_replan_index") != min(cut, int(reference["steps"])) // 5
            or (terminal and (row["steps"] != reference["steps"] or row["success"] is not True))
            or (not terminal and int(row["steps"]) <= cut)
            or not path.is_relative_to(Path(intervention["trace_root"]))
            or not path.is_file() or path.stat().st_size != int(trace.get("bytes", -1))
            or int(trace.get("steps", -1)) != int(row["steps"])):
        raise Pi05EvaluationError("frozen-prefix row contract changed")
