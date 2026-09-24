"""Continuous real-environment trace for a frozen whole-episode LoRA cell."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.approach_channel_replay import _append_with_contacts, _osc_audit
from ember.pi05_eval.contact_trace import audit_contact_geometries
from ember.pi05_eval.prefix_replay import _controller_state, _roles


TRACE_SCHEMA = "ember_readout_realization_object_trace_v1"
CONTACT_SCHEMA = "ember_readout_realization_contacts_v1"
ROW_SCHEMA = "ember_readout_realization_episode_v1"


def start_trace(env: Any, slot: dict[str, Any], task: Mapping[str, Any]) -> None:
    names, roles = _roles(env, task)
    geom_index, contact_identity = audit_contact_geometries(env, names)
    trace = {
        "object_names": names, "object_roles": roles,
        "actions": [], "object_positions": [], "eef_pos": [], "eef_quat": [],
        "gripper_qpos": [], "predicates": [], "contact_pairs": [],
        "geom_index": geom_index, "contact_identity": contact_identity,
        "sim_state_initial": np.asarray(env.get_sim_state(), dtype=np.float64).copy(),
        "controller_state_initial": _controller_state(env),
        "osc_channel_audit": _osc_audit(env),
    }
    _append_with_contacts(trace, env, slot["obs"], slot, None)
    slot["readout_trace"] = trace


def record_step(env: Any, slot: dict[str, Any], action: Any) -> None:
    _append_with_contacts(slot["readout_trace"], env, slot["obs"], slot,
                          np.asarray(action, dtype=np.float32))


def finish_trace(slot: Mapping[str, Any], task: Mapping[str, Any],
                 contract: Mapping[str, Any]) -> dict[str, Any]:
    trace = slot["readout_trace"]
    intervention = contract["readout_realization_intervention"]
    root = Path(intervention["trace_root"])
    root.mkdir(parents=True, exist_ok=True)
    stem = (f"{task['suite']}_task_{int(task['task_id']):02d}_"
            f"state_{int(slot['init_state_id']):03d}_{uuid.uuid4().hex}")
    path, contact_path = root / f"{stem}.npz", root / f"{stem}.contacts.json"
    arrays = {name: np.stack(trace[name]) for name in (
        "actions", "object_positions", "eef_pos", "eef_quat", "gripper_qpos", "predicates")}
    steps = int(slot["steps"])
    if (arrays["actions"].shape != (steps, 7)
            or arrays["object_positions"].shape[0] != steps + 1
            or len(trace["contact_pairs"]) != steps + 1
            or not all(np.isfinite(arrays[name]).all() for name in (
                "actions", "object_positions", "eef_pos", "eef_quat", "gripper_qpos"))):
        raise Pi05EvaluationError("readout continuous trace incomplete or nonfinite")
    with path.open("xb") as handle:
        np.savez_compressed(handle, schema_version=np.asarray(TRACE_SCHEMA),
                            object_names=np.asarray(trace["object_names"]),
                            sim_state_initial=trace["sim_state_initial"], **arrays)
    contacts = {
        "schema_version": CONTACT_SCHEMA,
        "sampling": intervention["contact_sampling"],
        "object_names": list(trace["object_names"]),
        "contact_identity": trace["contact_identity"],
        "pairs_after_settling_and_steps": trace["contact_pairs"],
    }
    with contact_path.open("x", encoding="utf-8") as handle:
        json.dump(contacts, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return {
        "schema_version": ROW_SCHEMA,
        "group": intervention["group"],
        "original_condition_id": slot["episode_adapter"].key,
        "object_roles": trace["object_roles"],
        "controller_state_initial": trace["controller_state_initial"],
        "osc_channel_audit": trace["osc_channel_audit"],
        "trace": {"path": str(path), "bytes": path.stat().st_size,
                  "schema_version": TRACE_SCHEMA, "steps": steps},
        "contacts": {"path": str(contact_path), "bytes": contact_path.stat().st_size,
                     "schema_version": CONTACT_SCHEMA, "samples": steps + 1},
    }


def validate_row(row: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    intervention = contract["readout_realization_intervention"]
    info = row.get("readout_realization_intervention")
    if (not isinstance(info, Mapping) or info.get("schema_version") != ROW_SCHEMA
            or info.get("group") != intervention["group"]
            or info.get("original_condition_id") != row["horizon_writer_lora"]["condition_id"]):
        raise Pi05EvaluationError("readout row intervention identity changed")
    for key, count in (("trace", int(row["steps"])), ("contacts", int(row["steps"]) + 1)):
        record = info[key]
        path = Path(record["path"])
        if (not path.is_relative_to(Path(intervention["trace_root"]))
                or not path.is_file() or path.stat().st_size != int(record["bytes"])
                or int(record["steps" if key == "trace" else "samples"]) != count):
            raise Pi05EvaluationError("readout row continuous trace changed")
