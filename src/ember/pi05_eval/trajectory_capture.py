"""Optional trajectory capture used by sealed PI05 diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.pi05_assets import Pi05EvaluationError


def capture_level(
    capture: Mapping[str, Any] | None,
    task: Mapping[str, Any],
    init_state_id: int,
) -> str | None:
    if capture is None:
        return None
    mode = str(capture.get("mode", "full"))
    if mode == "full":
        return "full"
    if mode != "compact":
        raise Pi05EvaluationError("unsupported diagnostic occupancy capture mode")
    key = (str(task["suite"]), int(task["task_id"]), int(init_state_id))
    selected = {
        (str(row["suite"]), int(row["task_id"]), int(row["init_state_id"]))
        for row in capture.get("full_conditions", ())
    }
    return "full" if key in selected else "compact"


def initialize_capture(slot: dict[str, Any], level: str | None) -> None:
    if level is None:
        return
    slot["occupancy_capture_level"] = level
    slot["replay_states"] = []
    slot["replay_action_chunks"] = []
    slot["replay_executed_prefixes"] = []
    slot["replay_replan_steps"] = []
    if level == "full":
        slot["replay_observations"] = []
        slot["replay_replan_predicates"] = []


def record_pre_exploration_means(slots: Any, chunks: Any) -> None:
    """Copy the existing policy output before the registered frozen perturbation."""
    import torch

    if chunks.ndim != 3 or chunks.shape[0] != len(slots) or chunks.shape[1:] != (50, 7):
        raise Pi05EvaluationError("objective-alignment normalized policy chunk shape changed")
    for index, slot in enumerate(slots):
        if "replay_action_chunks" not in slot:
            raise Pi05EvaluationError("objective-alignment requires trajectory capture on every row")
        slot.setdefault("pre_exploration_normalized_means", []).append(
            chunks[index, :5, :7].detach().to(device="cpu", dtype=torch.float32).contiguous()
        )


def record_replan(
    slot: dict[str, Any], raw_input: Mapping[str, Any], processed: Mapping[str, Any],
    chunk: Any | None, executed_prefix: Any | None = None,
    *, command_kind: str = "model_prediction",
) -> None:
    if "replay_action_chunks" not in slot:
        return
    import torch

    slot["replay_states"].append(
        raw_input["observation.state"].detach().to(device="cpu").contiguous()
    )
    if command_kind not in ("model_prediction", "external_saved_action") or (chunk is None) != (command_kind == "external_saved_action"):
        raise Pi05EvaluationError("diagnostic capture command provenance changed")
    slot["replay_action_chunks"].append(
        None if chunk is None else chunk.detach().to(device="cpu").contiguous()
    )
    slot.setdefault("replay_command_kinds", []).append(command_kind)
    slot["replay_replan_steps"].append(int(slot["steps"]))
    if executed_prefix is not None:
        slot["replay_executed_prefixes"].append(torch.as_tensor(executed_prefix).clone())
    if "replay_observations" in slot:
        slot["replay_replan_predicates"].append(
            list(slot["stage_predicate_last"]) if "stage_predicate_last" in slot else None
        )
        slot["replay_observations"].append(
            {
                key: value.detach().to(device="cpu").contiguous()
                for key, value in processed.items()
                if isinstance(value, torch.Tensor)
            }
        )


def save_capture(
    capture: Mapping[str, Any] | None,
    task: Mapping[str, Any],
    slot: Mapping[str, Any],
    *,
    success: bool,
) -> dict[str, Any] | None:
    if capture is None:
        return None
    import torch

    root = Path(str(capture["trajectory_root"]))
    root.mkdir(parents=True, exist_ok=True)
    path = root / (
        f"{task['suite']}_task_{int(task['task_id']):02d}_"
        f"state_{int(slot['init_state_id']):03d}.pt"
    )
    level = str(slot["occupancy_capture_level"])
    common = {
        "suite": task["suite"],
        "task_id": int(task["task_id"]),
        "init_state_id": int(slot["init_state_id"]),
        "success": bool(success),
        "steps": int(slot["steps"]),
        "policy_noise_seeds": tuple(slot["policy_noise_seeds"]),
        "action_chunks": tuple(slot["replay_action_chunks"]),
        "executed_action_prefixes": tuple(slot["replay_executed_prefixes"]),
        "replan_steps": tuple(slot["replay_replan_steps"]),
    }
    if "pre_exploration_normalized_means" in slot:
        if len(slot["pre_exploration_normalized_means"]) != len(slot["replay_action_chunks"]):
            raise Pi05EvaluationError("objective-alignment pre-exploration means are incomplete")
        common["pre_exploration_normalized_means"] = tuple(slot["pre_exploration_normalized_means"])
        common["exploration_noise_seeds"] = tuple(slot.get("exploration_noise_seeds", ()))
    if "mode" not in capture:
        payload = {
            "schema_version": "ember_pi05_occupancy_trajectory_v1",
            **common,
            "observations": tuple(slot["replay_observations"]),
        }
    else:
        payload = {
            "schema_version": ("ember_pi05_approach_channel_trajectory_v1"
                               if capture.get("external_prefix_commands") else
                               "ember_pi05_diagnostic_trajectory_v2"),
            "capture_level": level,
            **common,
            "states": tuple(slot["replay_states"]),
        }
        if level == "full":
            payload["observations"] = tuple(slot["replay_observations"])
            payload["replan_predicates"] = tuple(slot["replay_replan_predicates"])
        if capture.get("external_prefix_commands"):
            payload["command_kinds"] = tuple(slot["replay_command_kinds"])
    torch.save(payload, path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "replans": len(slot["replay_action_chunks"]),
        "capture_level": level,
    }


PASSIVE_TRACE_SCHEMA = "ember_pi05_passive_control_trace_v1"
PASSIVE_ROW_SCHEMA = "ember_pi05_passive_control_episode_v1"


def _passive_body_registry(env: Any, slot: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    owner = getattr(env, "env", env)
    objects = set(owner.objects_dict)
    fixtures = set(owner.fixtures_dict)
    body_ids = owner.obj_body_id
    if not objects or objects & fixtures or set(body_ids) != objects | fixtures:
        raise Pi05EvaluationError("passive trace actual object/fixture body registry changed")
    model = owner.sim.model
    bodies = [
        {"name": name, "kind": "object" if name in objects else "fixture",
         "body_id": int(body_ids[name]),
         "body_name": model.body_id2name(int(body_ids[name]))}
        for name in sorted(body_ids)
    ]
    if any(not isinstance(row["body_name"], str) or not row["body_name"]
           for row in bodies):
        raise Pi05EvaluationError("passive trace body reference has no model name")
    by_name = {row["name"]: row for row in bodies}
    regions = owner.parsed_problem["regions"]
    goals = slot["stage_predicate_states"]
    operands = []
    for goal in goals:
        classified = []
        for name in goal[1:]:
            if name in regions:
                region = regions[name]
                target = str(region["target"])
                if target not in by_name:
                    if (target != getattr(owner, "workspace_name", None)
                            or name not in owner.object_sites_dict):
                        raise Pi05EvaluationError(f"passive trace region target has no actual body: {name}")
                    try:
                        site_id = int(model.site_name2id(name))
                        site_name = model.site_id2name(site_id)
                        parent_id = int(model.site_bodyid[site_id])
                        parent_name = model.body_id2name(parent_id)
                    except (KeyError, ValueError, IndexError) as error:
                        raise Pi05EvaluationError(f"passive trace arena region lacks actual site: {name}") from error
                    if site_name != name or not isinstance(parent_name, str) or not parent_name:
                        raise Pi05EvaluationError(f"passive trace arena region site/body changed: {name}")
                    classified.append({
                        "name": name, "kind": "region", "target": target,
                        "target_kind": "arena_workspace", "site_id": site_id,
                        "site_name": site_name, "site_parent_body_id": parent_id,
                        "site_parent_body_name": parent_name,
                        "ranges": [[float(value) for value in rectangle]
                                   for rectangle in region["ranges"]],
                        "yaw_rotation": [float(value) for value in region["yaw_rotation"]],
                    })
                    continue
                classified.append({"name": name, "kind": "region", "target": target,
                                   "target_body_id": by_name[target]["body_id"],
                                   "target_body_name": by_name[target]["body_name"],
                                   "ranges": [[float(value) for value in rectangle]
                                              for rectangle in region["ranges"]],
                                   "yaw_rotation": [float(value) for value in region["yaw_rotation"]]})
            elif name in by_name:
                classified.append({"name": name, "kind": by_name[name]["kind"],
                                   "body_id": by_name[name]["body_id"],
                                   "body_name": by_name[name]["body_name"]})
            else:
                raise Pi05EvaluationError(f"passive trace BDDL operand has no actual mapping: {name}")
        operands.append(classified)
    return bodies, operands


def start_passive_trace(env: Any, slot: dict[str, Any], capture: Mapping[str, Any] | None) -> None:
    if not capture or not capture.get("passive_trace"):
        return
    if "stage_predicate_states" not in slot:
        raise Pi05EvaluationError("passive trace requires all-row real BDDL predicates")
    bodies, operands = _passive_body_registry(env, slot)
    slot["passive_trace"] = {
        "body_registry": bodies, "goal_operands": operands,
        "actions": [], "body_positions": [], "eef_pos": [], "eef_quat": [],
        "gripper_qpos": [], "predicates": [],
    }
    record_passive_step(env, slot, None, capture)


def record_passive_step(
    env: Any, slot: dict[str, Any], action: Any | None, capture: Mapping[str, Any] | None,
) -> None:
    if not capture or not capture.get("passive_trace"):
        return
    import numpy as np

    trace = slot["passive_trace"]
    if action is not None:
        applied = np.asarray(action, dtype=np.float32)
        if applied.shape != (7,) or not np.isfinite(applied).all():
            raise Pi05EvaluationError("passive trace actual environment action changed")
        trace["actions"].append(applied.copy())
    if len(trace["actions"]) != int(slot["steps"]):
        raise Pi05EvaluationError("passive trace control-step clock changed")
    owner = getattr(env, "env", env)
    positions = np.stack([
        np.asarray(owner.sim.data.body_xpos[row["body_id"]], dtype=np.float32).copy()
        for row in trace["body_registry"]
    ])
    observation = slot["obs"]
    sample = {
        "body_positions": positions,
        "eef_pos": np.asarray(observation["robot0_eef_pos"], dtype=np.float32).copy(),
        "eef_quat": np.asarray(observation["robot0_eef_quat"], dtype=np.float32).copy(),
        "gripper_qpos": np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32).copy(),
        "predicates": np.asarray(slot["stage_predicate_last"], dtype=np.bool_).copy(),
    }
    if (sample["body_positions"].shape != (len(trace["body_registry"]), 3)
            or sample["eef_pos"].shape != (3,) or sample["eef_quat"].shape != (4,)
            or sample["gripper_qpos"].shape != (2,)
            or sample["predicates"].shape != (len(slot["stage_predicate_states"]),)
            or any(not np.isfinite(sample[name]).all() for name in (
                "body_positions", "eef_pos", "eef_quat", "gripper_qpos"))):
        raise Pi05EvaluationError("passive trace simulator/robot/predicate sample invalid")
    for name, value in sample.items():
        trace[name].append(value)


def save_passive_trace(
    capture: Mapping[str, Any] | None, task: Mapping[str, Any], slot: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not capture or not capture.get("passive_trace"):
        return None
    import uuid

    import numpy as np

    trace = slot["passive_trace"]
    steps = int(slot["steps"])
    arrays = {name: np.stack(trace[name]) for name in (
        "actions", "body_positions", "eef_pos", "eef_quat", "gripper_qpos", "predicates")}
    expected = {"actions": (steps, 7), "body_positions": (steps + 1, len(trace["body_registry"]), 3),
                "eef_pos": (steps + 1, 3), "eef_quat": (steps + 1, 4),
                "gripper_qpos": (steps + 1, 2),
                "predicates": (steps + 1, len(slot["stage_predicate_states"]))}
    if (any(arrays[name].shape != shape for name, shape in expected.items())
            or not np.array_equal(arrays["predicates"][-1], np.asarray(slot["stage_predicate_last"]))):
        raise Pi05EvaluationError("passive trace actual steps or final BDDL predicates incomplete")
    root = Path(capture["passive_trace"]["trace_root"])
    root.mkdir(parents=True, exist_ok=True)
    stem = (f"{task['suite']}_task_{int(task['task_id']):02d}_"
            f"state_{int(slot['init_state_id']):03d}_{uuid.uuid4().hex}")
    path = root / f"{stem}.npz"
    with path.open("xb") as handle:
        np.savez_compressed(handle, schema_version=np.asarray(PASSIVE_TRACE_SCHEMA),
                            body_names=np.asarray([row["name"] for row in trace["body_registry"]]),
                            body_ids=np.asarray([row["body_id"] for row in trace["body_registry"]]),
                            **arrays)
    adapter = slot.get("episode_adapter")
    return {
        "schema_version": PASSIVE_ROW_SCHEMA,
        "suite": task["suite"], "task_id": int(task["task_id"]),
        "init_state_id": int(slot["init_state_id"]),
        "condition_id": adapter.key if adapter is not None else None,
        "body_registry": trace["body_registry"], "goal_operands": trace["goal_operands"],
        "goal_predicates": [list(state) for state in slot["stage_predicate_states"]],
        "sampling": "post_settling_t0_and_after_each_executed_control_step_not_integrator_substeps",
        "trace": {"path": str(path), "bytes": path.stat().st_size,
                  "schema_version": PASSIVE_TRACE_SCHEMA,
                  "steps": steps, "samples": steps + 1},
    }


def validate_passive_trace_row(
    row: Mapping[str, Any], contract: Mapping[str, Any], task: Mapping[str, Any],
) -> None:
    capture = contract.get("diagnostic_occupancy_capture") or {}
    if not capture.get("passive_trace"):
        return
    info = row.get("continuous_control_trace")
    stage = row.get("stage_predicates") or {}
    if (not isinstance(info, Mapping) or info.get("schema_version") != PASSIVE_ROW_SCHEMA
            or (info.get("suite"), info.get("task_id"), info.get("init_state_id")) !=
               (task["suite"], int(task["task_id"]), int(row["init_state_id"]))
            or info.get("condition_id") != (row.get("horizon_writer_lora") or {}).get("condition_id")
            or info.get("goal_predicates") != stage.get("predicates")
            or not isinstance(info.get("body_registry"), list)
            or not info["body_registry"] or not isinstance(info.get("goal_operands"), list)):
        raise Pi05EvaluationError("passive trace row identity, bodies or predicates missing")
    record = info.get("trace") or {}
    path = Path(str(record.get("path", "")))
    root = Path(capture["passive_trace"]["trace_root"])
    steps = int(row["steps"])
    if (not path.is_relative_to(root) or not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or record.get("schema_version") != PASSIVE_TRACE_SCHEMA
            or record.get("steps") != steps or record.get("samples") != steps + 1):
        raise Pi05EvaluationError("passive trace file or control-step count missing")
    _validate_passive_trace_file(path, info, stage, steps)


def _validate_passive_trace_file(
    path: Path, info: Mapping[str, Any], stage: Mapping[str, Any], steps: int,
) -> None:
    import numpy as np
    from zipfile import BadZipFile

    bodies = info["body_registry"]
    predicates = stage["predicates"]
    expected = {
        "actions": (steps, 7),
        "body_positions": (steps + 1, len(bodies), 3),
        "eef_pos": (steps + 1, 3),
        "eef_quat": (steps + 1, 4),
        "gripper_qpos": (steps + 1, 2),
        "predicates": (steps + 1, len(predicates)),
    }
    try:
        with np.load(path, allow_pickle=False) as data:
            if (str(data["schema_version"]) != PASSIVE_TRACE_SCHEMA
                    or data["body_names"].tolist() != [body["name"] for body in bodies]
                    or data["body_ids"].tolist() != [body["body_id"] for body in bodies]):
                raise Pi05EvaluationError("passive trace body identity changed")
            for name, shape in expected.items():
                value = data[name]
                if value.shape != shape or (name != "predicates" and not np.isfinite(value).all()):
                    raise Pi05EvaluationError("passive trace sampled field missing or invalid")
            if not np.array_equal(data["predicates"][-1], stage["final_satisfied"]):
                raise Pi05EvaluationError("passive trace final predicates changed")
    except (OSError, KeyError, ValueError, BadZipFile) as exc:
        raise Pi05EvaluationError("passive trace sampled field missing or unreadable") from exc
