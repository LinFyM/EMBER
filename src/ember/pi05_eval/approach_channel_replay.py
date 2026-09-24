"""Apply saved B/C environment-action channels before the canonical policy tail."""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.contact_trace import audit_contact_geometries, sample_robot_object_contacts
from ember.pi05_eval.prefix_replay import (
    _append, _controller_state, _displacements, _load_saved_prefix, _roles, _state_error,
)
from ember.pi05_eval.trajectory_capture import record_replan
from ember.pi05_processing import libero_policy_input


TRACE_SCHEMA = "ember_approach_channel_trace_v1"
CONTACT_SCHEMA = "ember_approach_channel_contacts_v1"
ROW_SCHEMA = "ember_approach_channel_episode_v1"


def _osc_audit(env: Any) -> dict[str, Any]:
    owner = env.env
    robot = owner.robots[0]
    controller = robot.controller
    if (type(controller).__name__ != "OperationalSpaceController"
            or controller.impedance_mode != "fixed"
            or controller.use_delta is not True
            or controller.use_ori is not True
            or int(controller.control_dim) != 6
            or int(robot.action_dim) != 7):
        raise Pi05EvaluationError("approach-channel OSC/action ordering changed")
    return {
        "controller_class": type(controller).__name__,
        "impedance_mode": controller.impedance_mode,
        "control_delta": controller.use_delta,
        "orientation_control": controller.use_ori,
        "arm_control_dim": int(controller.control_dim),
        "robot_action_dim": int(robot.action_dim),
        "action_channels": ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper"],
        "controller_input_min": np.asarray(controller.input_min).tolist(),
        "controller_input_max": np.asarray(controller.input_max).tolist(),
        "controller_output_min": np.asarray(controller.output_min).tolist(),
        "controller_output_max": np.asarray(controller.output_max).tolist(),
        "applied_action_space": "saved denormalized environment inputs; no additional transform",
    }


def _append_with_contacts(trace: dict[str, Any], env: Any, obs: Mapping[str, Any],
                          slot: Mapping[str, Any], action: np.ndarray | None) -> None:
    _append(trace, env, obs, slot, action)
    trace["contact_pairs"].append(sample_robot_object_contacts(env, trace["geom_index"]))


def compose_prefix_commands(donors: Mapping[str, np.ndarray],
                            sources: tuple[str, ...]) -> np.ndarray:
    if (len(sources) != 7 or not set(sources) <= {"B", "C_correct"}
            or any(name not in donors or donors[name].shape != (25, 7)
                   or not np.isfinite(donors[name]).all() for name in ("B", "C_correct"))):
        raise Pi05EvaluationError("approach-channel donor commands or sources invalid")
    return np.stack([donors[sources[channel]][:, channel]
                     for channel in range(7)], axis=1)


def _load_channel_plan(intervention: Mapping[str, Any], task: Mapping[str, Any],
                       state_id: int, root_seed: int) -> tuple[Any, ...]:
    key = f"{task['suite']}:{int(task['task_id'])}:{state_id}"
    refs = {donor: intervention["donor_references"][donor].get(key)
            for donor in ("B", "C_correct")}
    if any(reference is None for reference in refs.values()):
        raise Pi05EvaluationError(f"approach-channel donor reference missing: {key}")
    donor_data = {donor: _load_saved_prefix(
        reference=refs[donor], task=task, state_id=state_id, cut=25,
        root_seed=root_seed)
        for donor in ("B", "C_correct")}
    if (any(data[4] != 25 for data in donor_data.values())
            or donor_data["B"][3][:5] != donor_data["C_correct"][3][:5]):
        raise Pi05EvaluationError(f"approach-channel donor prefix unavailable or RNG differs: {key}")
    donor_actions = {donor: np.concatenate([
        np.asarray(chunk, dtype=np.float32) for chunk in data[1][:5]], axis=0)
        for donor, data in donor_data.items()}
    sources = tuple(intervention["channel_donors"])
    commanded = compose_prefix_commands(donor_actions, sources)
    pure = sources[0] if len(set(sources)) == 1 else None
    if pure is not None and not np.array_equal(commanded, donor_actions[pure]):
        raise Pi05EvaluationError("approach-channel pure donor command changed")
    return key, refs, donor_data, donor_actions, commanded, pure


def replay_channel_prefix(*, env: Any, slot: dict[str, Any], task: Mapping[str, Any],
                          contract: Mapping[str, Any], preprocess: Any) -> None:
    from ember.pi05_eval.episode import update_stage_predicates

    intervention = contract["approach_channel_intervention"]
    state_id = int(slot["init_state_id"])
    if int(contract["policy"]["replan_steps"]) != 5 or intervention["cut_control_steps"] != 25:
        raise Pi05EvaluationError("approach-channel replan/cut clock changed")
    key, refs, donor_data, donor_actions, commanded, pure = _load_channel_plan(
        intervention, task, state_id, int(contract["rng"]["inference_seed"]))
    names, roles = _roles(env, task)
    geom_index, contact_identity = audit_contact_geometries(env, names)
    trace: dict[str, Any] = {
        "object_names": names, "object_roles": roles,
        "actions": [], "object_positions": [], "eef_pos": [], "eef_quat": [],
        "gripper_qpos": [], "predicates": [], "contact_pairs": [],
        "geom_index": geom_index, "contact_identity": contact_identity,
        "sim_state_initial": np.asarray(env.get_sim_state(), dtype=np.float64).copy(),
        "osc_channel_audit": _osc_audit(env),
    }
    _append_with_contacts(trace, env, slot["obs"], slot, None)
    errors = []
    for index in range(5):
        raw_input = libero_policy_input(slot["obs"], task["language"])
        if pure is not None or index == 0:
            expected = donor_data[pure or "B"][2][index]
            errors.append(_state_error(raw_input["observation.state"], expected, step=index * 5))
            if pure is None:
                errors.append(_state_error(raw_input["observation.state"],
                                           donor_data["C_correct"][2][0], step=0))
        planned = commanded[index * 5:(index + 1) * 5]
        processed = preprocess(raw_input) if slot.get("occupancy_capture_level") == "full" else {}
        record_replan(slot, raw_input, processed, None, planned,
                      command_kind="external_saved_action")
        slot["policy_noise_seeds"].append(donor_data["B"][3][index])
        slot["replan_index"] += 1
        for action in planned:
            observation, _, done, _ = env.step(action)
            slot["obs"] = observation
            slot["steps"] += 1
            update_stage_predicates(env, slot)
            _append_with_contacts(trace, env, observation, slot, action)
            if pure is not None:
                reference = refs[pure]
                expected_done = bool(reference["success"] and slot["steps"] == reference["steps"])
                if bool(done) != expected_done:
                    raise Pi05EvaluationError(f"approach-channel pure replay terminal changed: {key}, step {slot['steps']}")
            if done:
                break
        if done:
            import torch

            actual_in_chunk = slot["steps"] - index * 5
            slot["replay_executed_prefixes"][-1] = torch.as_tensor(planned[:actual_in_chunk]).clone()
            break
    if not done and slot["steps"] == 25 and pure is not None:
        state = libero_policy_input(slot["obs"], task["language"])["observation.state"]
        errors.append(_state_error(state, donor_data[pure][2][5], step=25))
    trace["sim_state_cut"] = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
    trace["controller_state_cut"] = _controller_state(env)
    slot["prefix_trace"] = trace
    slot["prefix_state_errors"] = errors
    slot["prefix_terminal"] = bool(done)
    slot["prefix_replay_steps"] = int(slot["steps"])
    slot["prefix_reference"] = refs
    slot["channel_donor_actions"] = donor_actions
    slot["channel_commanded"] = commanded


def record_tail_contacts(env: Any, slot: dict[str, Any]) -> None:
    trace = slot["prefix_trace"]
    trace["contact_pairs"].append(sample_robot_object_contacts(env, trace["geom_index"]))


def _geometry(positions: np.ndarray, eef: np.ndarray, names: tuple[str, ...],
              target: str, contacts: list[list[dict[str, Any]]], cut: int) -> dict[str, Any]:
    target_pos = positions[:, names.index(target)]
    horizontal = np.linalg.norm(eef[:, :2] - target_pos[:, :2], axis=1)
    height = eef[:, 2] - target_pos[:, 2]
    first_contacts = {name: next((step for step, pairs in enumerate(contacts)
                                  if any(pair["object_name"] == name
                                         and pair["minimum_distance_m"] <= 0 for pair in pairs)), None)
                      for name in names}
    return {
        "target_body": target,
        "eef_target_horizontal_error_m_initial": float(horizontal[0]),
        "eef_target_horizontal_error_m_cut": float(horizontal[cut]),
        "eef_target_relative_height_m_initial": float(height[0]),
        "eef_target_relative_height_m_cut": float(height[cut]),
        "first_observed_touch_step_by_object": first_contacts,
        "contact_boundary": "post-control-step MuJoCo geom pair with minimum distance <= 0; no unobserved substep exclusion",
        "position_boundary": "target body center and EEF pose are geometric proxies, not grasp state",
    }


def finish_channel_trace(slot: Mapping[str, Any], task: Mapping[str, Any],
                         contract: Mapping[str, Any]) -> dict[str, Any]:
    trace = slot["prefix_trace"]
    intervention = contract["approach_channel_intervention"]
    root = Path(intervention["trace_root"])
    root.mkdir(parents=True, exist_ok=True)
    stem = (f"{task['suite']}_task_{int(task['task_id']):02d}_"
            f"state_{int(slot['init_state_id']):03d}_{uuid.uuid4().hex}")
    path, contacts_path = root / f"{stem}.npz", root / f"{stem}.contacts.json"
    arrays = {name: np.stack(trace[name]) for name in (
        "actions", "object_positions", "eef_pos", "eef_quat", "gripper_qpos", "predicates")}
    cut = int(slot["prefix_replay_steps"])
    if (arrays["actions"].shape != (int(slot["steps"]), 7)
            or arrays["object_positions"].shape[0] != int(slot["steps"]) + 1
            or len(trace["contact_pairs"]) != int(slot["steps"]) + 1
            or not all(np.isfinite(arrays[name]).all() for name in (
                "actions", "object_positions", "eef_pos", "eef_quat", "gripper_qpos"))):
        raise Pi05EvaluationError("approach-channel continuous trace incomplete or nonfinite")
    positions = arrays["object_positions"]
    eef = arrays["eef_pos"]
    target = trace["object_roles"]["target"]
    target_pos = positions[:, trace["object_names"].index(target)]
    with path.open("xb") as handle:
        np.savez_compressed(
            handle, schema_version=np.asarray(TRACE_SCHEMA),
            object_names=np.asarray(trace["object_names"]),
            sim_state_initial=trace["sim_state_initial"],
            sim_state_cut=trace["sim_state_cut"],
            donor_B_actions=slot["channel_donor_actions"]["B"],
            donor_C_correct_actions=slot["channel_donor_actions"]["C_correct"],
            commanded_prefix_actions=slot["channel_commanded"],
            applied_prefix_actions=arrays["actions"][:cut],
            eef_target_horizontal_error_m=np.linalg.norm(eef[:, :2] - target_pos[:, :2], axis=1),
            eef_target_relative_height_m=eef[:, 2] - target_pos[:, 2],
            **arrays,
        )
    with contacts_path.open("x", encoding="utf-8") as handle:
        json.dump({"schema_version": CONTACT_SCHEMA,
                   "identity": trace["contact_identity"],
                   "samples": [{"step": step, "pairs": pairs}
                               for step, pairs in enumerate(trace["contact_pairs"])]},
                  handle, allow_nan=False)
        handle.write("\n")
    return {
        "schema_version": ROW_SCHEMA,
        "prefix": intervention["prefix"], "follower": intervention["follower"],
        "cut_control_steps": 25, "actual_prefix_steps": cut,
        "prefix_terminal": bool(slot["prefix_terminal"]),
        "follower_start_replan_index": int(slot["replan_index"]) if slot["prefix_terminal"] else 5,
        "channel_donors": list(intervention["channel_donors"]),
        "command_provenance": "saved denormalized environment inputs; prefix action_chunks are null",
        "donor_references": dict(slot["prefix_reference"]),
        "pure_replay_state_errors": list(slot["prefix_state_errors"]),
        "osc_channel_audit": trace["osc_channel_audit"],
        "object_roles": trace["object_roles"],
        "controller_state_cut": trace["controller_state_cut"],
        "object_displacement": _displacements(positions, trace["object_names"], cut),
        "geometry": _geometry(positions, eef, trace["object_names"], target,
                              trace["contact_pairs"], cut),
        "trace": {"path": str(path), "bytes": path.stat().st_size,
                  "schema_version": TRACE_SCHEMA, "steps": int(slot["steps"])},
        "contacts": {"path": str(contacts_path), "bytes": contacts_path.stat().st_size,
                     "schema_version": CONTACT_SCHEMA, "samples": int(slot["steps"]) + 1},
    }


def _channel_assets_valid(prefix: Mapping[str, Any], intervention: Mapping[str, Any],
                          steps: int) -> bool:
    trace, contacts = prefix.get("trace", {}), prefix.get("contacts", {})
    for asset in (trace, contacts):
        path = Path(str(asset.get("path", "")))
        if (not path.is_relative_to(Path(intervention["trace_root"]))
                or not path.is_file() or path.stat().st_size != int(asset.get("bytes", -1))):
            return False
    return int(trace.get("steps", -1)) == steps and int(contacts.get("samples", -1)) == steps + 1


def _channel_timing_valid(prefix: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    actual = int(prefix.get("actual_prefix_steps", -1))
    steps = int(row["steps"])
    terminal = prefix.get("prefix_terminal")
    if type(terminal) is not bool or not 1 <= actual <= 25:
        return False
    if terminal:
        return (row["success"] is True and steps == actual
                and prefix.get("follower_start_replan_index") == math.ceil(actual / 5))
    return actual == 25 and steps > 25 and prefix.get("follower_start_replan_index") == 5


def validate_channel_row(row: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    intervention = contract["approach_channel_intervention"]
    prefix = row.get("approach_channel_intervention")
    if not isinstance(prefix, Mapping) or prefix.get("schema_version") != ROW_SCHEMA:
        raise Pi05EvaluationError("approach-channel row evidence missing")
    key = f"{row['suite']}:{int(row['task_id'])}:{int(row['init_state_id'])}"
    expected_refs = {donor: intervention["donor_references"][donor].get(key)
                     for donor in ("B", "C_correct")}
    if (any(value is None for value in expected_refs.values())
            or prefix.get("donor_references") != expected_refs
            or prefix.get("prefix") != intervention["prefix"]
            or prefix.get("follower") != intervention["follower"]
            or prefix.get("channel_donors") != intervention["channel_donors"]
            or not _channel_timing_valid(prefix, row)
            or not _channel_assets_valid(prefix, intervention, int(row["steps"]))):
        raise Pi05EvaluationError("approach-channel row contract changed")
